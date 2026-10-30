"""층 A 합성 데이터: 악보 → 성부별 합성 음성 → 혼합 → Basic Pitch contour + 성부별 정답 궤적.

    uv run python -m training.synth --data-dir data --out data/synth --variants 4

정렬이 완벽한 학습 쌍을 만든다. test 분할 곡은 만들지 않는다(평가 오염 방지).
렌더링 방식은 M5에서 비교한다: ``harmonic``(하모닉 합성), ``formant``(모음 포먼트 필터).
샘플 기반·노래 합성은 같은 ``render_voice`` 자리에 붙인다.

증강: ±2반음, 템포 ±10%, 성부별 음량 비율, 리버브, 비트박스 잡음, 곡간 성부 교체.

출력 npz(곡·변형마다 하나):
  salience (T, 264) float16  — Basic Pitch contour (앱 입력과 같음)
  target_midi (K, T) int16   — 슬롯별 정답 MIDI, 무음은 -1. 슬롯은 음역 높은 순
  ranges (K, 3) float32      — acappella.pipeline.voices.encode_voice_ranges와 같은 인코딩
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

from acappella.config import VOICE_DEFAULTS, default_voice
from acappella.meta import SongMeta, load_all
from acappella.nwc.reader import read_nwctxt
from acappella.pipeline.pitch import HOP, SR, estimate_pitch
from acappella.pipeline.voices import ONNX_K, encode_voice_ranges, order_voices
from acappella.score_utils import merge_ties
from acappella.types import AudioClip, QNote, Score, VoiceSpec
from training.storage import pack_salience

log = logging.getLogger(__name__)
VP_RE = re.compile(r"(?i)(vp|perc|beat|drum|비트|퍼커|드럼)")

# 모음 포먼트(Hz, 대략값)
VOWELS = {
    "a": (800, 1200, 2500),
    "o": (500, 900, 2400),
    "u": (350, 800, 2300),
    "e": (500, 1800, 2500),
    "i": (300, 2200, 2900),
}


@dataclass
class Aug:
    transpose: int = 0
    tempo_factor: float = 1.0
    gains: tuple[float, ...] = ()
    reverb: float = 0.0  # 잔향 시간(초), 0이면 없음
    beatbox: float = 0.0  # 잡음 상대 크기
    renderer: str = "harmonic"


def sample_aug(rng: random.Random, n_voices: int, renderer: str) -> Aug:
    return Aug(
        transpose=rng.randint(-2, 2),
        tempo_factor=rng.uniform(0.9, 1.1),
        gains=tuple(rng.uniform(0.5, 1.0) for _ in range(n_voices)),
        reverb=rng.choice([0.0, 0.0, rng.uniform(0.3, 1.5)]),
        beatbox=rng.choice([0.0, 0.0, rng.uniform(0.05, 0.3)]),
        renderer=renderer if renderer != "mixed" else rng.choice(["harmonic", "formant"]),
    )


def _resonator(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    a = [1.0, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
    return lfilter([1 - r], a, x)


def render_voice(
    notes: list[tuple[float, float, int]],
    total: float,
    rng: np.random.Generator,
    renderer: str = "harmonic",
    sr: int = SR,
) -> np.ndarray:
    """(onset초, offset초, midi) 목록 → 합성 음성. 비브라토·지터·배음 기울기를 무작위로 준다."""
    y = np.zeros(int(total * sr) + 1)
    tilt = rng.uniform(0.6, 1.4)
    vib_rate, vib_depth = rng.uniform(4.5, 6.5), rng.uniform(0.0, 0.4)
    vowel = VOWELS[rng.choice(list(VOWELS))]
    for on, off, midi in notes:
        a, b = int(on * sr), min(int(off * sr), y.size)
        if b - a < 8:
            continue
        t = np.arange(b - a) / sr
        semis = midi + vib_depth * np.sin(
            2 * np.pi * vib_rate * t + rng.uniform(0, 6.28)
        ) * np.minimum(1, t / 0.3)
        semis = semis + rng.normal(0, 0.1)  # 음마다 조금씩 어긋난 음정
        f0 = 440.0 * 2 ** ((semis - 69) / 12)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        n_h = max(1, min(20, int(0.45 * sr / f0.max())))
        tone = sum(np.sin(h * phase) / h**tilt for h in range(1, n_h + 1))
        att, rel = rng.uniform(0.02, 0.08), rng.uniform(0.03, 0.1)
        env = np.minimum(1, np.minimum(t / att, (t[::-1] + 1e-9) / rel))
        y[a:b] += tone * env
    if renderer == "formant":
        y = sum(_resonator(y, f, 80 + 0.05 * f, sr) for f in vowel)
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def add_reverb(y: np.ndarray, rt: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    n = int(rt * sr)
    ir = rng.normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    wet = fftconvolve(y, ir)[: y.size]
    return 0.7 * y + 0.3 * wet / (np.abs(wet).max() + 1e-9) * np.abs(y).max()


def beatbox_noise(
    n: int, spb: float, level: float, rng: np.random.Generator, sr: int = SR
) -> np.ndarray:
    """박마다 킥(저역 감쇠 사인)과 엇박 스네어(잡음)를 넣는다."""
    out = np.zeros(n)
    step = int(spb * sr / 2)
    for k, start in enumerate(range(0, n, step)):
        m = min(int(0.12 * sr), n - start)
        t = np.arange(m) / sr
        if k % 2 == 0:
            hit = np.sin(2 * np.pi * 60 * t * np.exp(-t * 20)) * np.exp(-t * 25)
        else:
            hit = rng.normal(0, 0.5, m) * np.exp(-t * 35)
        out[start : start + m] += hit
    return level * out


NAME_ALIASES = {
    "s": "S", "sop": "S", "soprano": "S", "s1": "S1", "s2": "S2",
    "a": "A", "alto": "A", "a1": "A1", "a2": "A2", "mezzo": "A", "meso": "A", "m": "A",
    "t": "T", "tenor": "T", "t1": "T1", "t2": "T2",
    "bar": "Bar", "bari": "Bar", "bariton": "Bar", "baritone": "Bar", "br": "Bar",
    "b": "B", "bass": "B",
    "lead": "Lead", "solo": "Lead", "리드": "Lead", "솔로": "Lead",
}  # fmt: skip
_CLASSES = ("S", "A", "T", "Bar", "B")


def staff_voice(name: str, midis: list[int]) -> VoiceSpec:
    """스태프 → 앱 기본 음역의 VoiceSpec. 이름이 성부 이름이면 그대로, 아니면(Staff-1, 사람 이름 등)
    실제 음의 중앙값에 가장 가까운 성부(S/A/T/Bar/B)로 본다. 앱에서 사용자가 고르는 기본 음역과
    같은 인코딩을 학습에 쓰기 위해서다."""
    key = re.sub(r"[\s().\-_0-9]*$", "", name.strip().lower()) if name else ""
    key = NAME_ALIASES.get(name.strip().lower(), NAME_ALIASES.get(key, ""))
    if not key and re.search(r"(?i)(solo|lead|리드|솔로)", name):
        key = "Lead"
    if not key:
        med = float(np.median(midis))
        key = min(
            _CLASSES, key=lambda c: abs((VOICE_DEFAULTS[c][1] + VOICE_DEFAULTS[c][2]) / 2 - med)
        )
    spec = default_voice(key)
    spec.name = name
    return spec


def voices_for(meta: SongMeta, score: Score) -> list[tuple[str, VoiceSpec]]:
    """(스태프 이름, VoiceSpec) 목록. VP 스태프는 뺀다. 최대 K개."""
    staves = [v.name for v in score.voices]
    names = meta.voices if len(meta.voices) == len(staves) else staves
    out = []
    for staff, name in zip(staves, names):
        if VP_RE.search(name) or VP_RE.search(staff):
            continue
        midis = [q.midi for q in score.parts[staff] if q.midi is not None]
        if not midis:
            continue
        out.append((staff, staff_voice(name, midis)))
    return out[:ONNX_K]


def render_example(
    score: Score,
    voices: list[tuple[str, VoiceSpec]],
    aug: Aug,
    seed: int,
    replace_part: tuple[int, list[QNote]] | None = None,
    backend: str = "basic_pitch",
):
    """한 번의 렌더링 → (salience, target_midi, ranges)."""
    rng = np.random.default_rng(seed)
    spb = 60.0 / (score.tempo_bpm * aug.tempo_factor)
    parts = []
    for i, (staff, _v) in enumerate(voices):
        part = merge_ties(score.parts[staff])
        if replace_part is not None and replace_part[0] == i:
            part = replace_part[1]
        parts.append(
            [
                (
                    float(q.start_beat) * spb,
                    float(q.start_beat + q.dur_beats) * spb - 0.01,
                    q.midi + aug.transpose,
                )
                for q in part
            ]
        )
    total = max((n[1] for p in parts for n in p), default=1.0) + 0.5
    mix = np.zeros(int(total * SR) + 1)
    for p, g in zip(parts, aug.gains or [1.0] * len(parts)):
        mix += g * render_voice(p, total, rng, aug.renderer)
    if aug.reverb:
        mix = add_reverb(mix, aug.reverb, rng)
    if aug.beatbox:
        mix += beatbox_noise(mix.size, spb, aug.beatbox * np.abs(mix).max(), rng)
    mix = (0.8 * mix / (np.abs(mix).max() + 1e-9)).astype(np.float32)

    act = estimate_pitch(AudioClip(mix, SR, "synth"), backend)
    T = act.salience.shape[0]
    specs = [v for _, v in voices]
    order = order_voices(specs)
    target = np.full((ONNX_K, T), -1, dtype=np.int16)
    frame_t = np.arange(T) * HOP / SR
    for slot, vi in enumerate(order):
        for on, off, midi in parts[vi]:
            target[slot, (frame_t >= on) & (frame_t < off)] = midi
    shifted = [
        VoiceSpec(v.name, v.clef, v.low_midi + aug.transpose, v.high_midi + aug.transpose)
        for v in specs
    ]
    ranges = encode_voice_ranges([shifted[i] for i in order])
    return act.salience.astype(np.float16), target, ranges


def build(
    data_dir: Path,
    out_dir: Path,
    variants: int,
    renderer: str,
    backend: str,
    cross_mix: float,
    seed: int,
    limit: int | None,
    shard: int = 0,
    num_shards: int = 1,
) -> int:
    metas = [m for m in load_all(data_dir / "meta" / "songs") if m.split in ("train", "valid")]
    if limit:
        metas = metas[:limit]
    mine = {m.song_id for i, m in enumerate(metas) if i % num_shards == shard}
    rng = random.Random(seed)
    scores = {}
    for m in metas:
        path = data_dir / "nwctxt" / f"{m.song_id}.nwctxt"
        if path.exists() and not m.is_medley:
            scores[m.song_id] = read_nwctxt(path)
    count = 0
    for m in metas:
        if m.song_id not in scores or m.song_id not in mine:
            continue
        score = scores[m.song_id]
        voices = voices_for(m, score)
        if not voices:
            continue
        pool = [s for s in scores if s != m.song_id and metas_split(metas, s) == m.split]
        for k in range(variants):
            dst = out_dir / m.split / f"{m.song_id}_{k:02d}.npz"
            if dst.exists():
                count += 1
                continue  # 중단 후 재실행하면 이어서 만든다
            rng = random.Random(f"{seed}:{m.song_id}:{k}")  # 샤드 나눔과 상관없이 재현된다
            aug = sample_aug(rng, len(voices), renderer)
            replace = None
            if pool and rng.random() < cross_mix:
                replace = cross_part(scores[rng.choice(pool)], voices, rng)
            sal, tgt, ranges = render_example(
                score, voices, aug, rng.randrange(2**31), replace, backend
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dst,
                salience=pack_salience(sal),
                target_midi=tgt,
                ranges=ranges,
                song_id=m.song_id,
                variant=k,
            )
            count += 1
            log.info("%s (%d개)", dst.name, count)
    return count


def resplit(data_dir: Path, out_dir: Path) -> dict[str, int]:
    """분할이 바뀐 뒤(예: 유튜브 매칭으로 test 곡이 생긴 뒤) 합성 파일을 현재 분할 폴더로 옮긴다.
    test 곡의 합성본은 평가 오염을 막기 위해 지운다."""
    split_of = {m.song_id: m.split for m in load_all(data_dir / "meta" / "songs")}
    moved = removed = 0
    for f in sorted(Path(out_dir).glob("*/*.npz")):
        sid = f.stem.rsplit("_", 1)[0]
        want = split_of.get(sid, "")
        if want not in ("train", "valid"):
            f.unlink()
            removed += 1
        elif f.parent.name != want:
            (out_dir / want).mkdir(parents=True, exist_ok=True)
            f.replace(out_dir / want / f.name)
            moved += 1
    return {"moved": moved, "removed": removed}


def metas_split(metas: list[SongMeta], song_id: str) -> str:
    return next((m.split for m in metas if m.song_id == song_id), "")


def cross_part(other: Score, voices, rng: random.Random) -> tuple[int, list[QNote]] | None:
    """다른 곡에서 같은 성부(없으면 아무 성부)를 가져와 음역 중심에 맞게 옮긴다."""
    i = rng.randrange(len(voices))
    _, spec = voices[i]
    cands = [n for n in other.parts if n == voices[i][0]] or list(other.parts)
    part = merge_ties(other.parts[rng.choice(cands)])
    midis = [q.midi for q in part]
    if not midis:
        return None
    shift = round((spec.low_midi + spec.high_midi) / 2 - float(np.median(midis)))
    return i, [QNote(q.start_beat, q.dur_beats, q.midi + shift) for q in part]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="층 A 합성 학습 데이터 생성")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--variants", type=int, default=4, help="곡마다 만들 증강 변형 수")
    ap.add_argument("--renderer", default="mixed", choices=["harmonic", "formant", "mixed"])
    ap.add_argument("--backend", default="basic_pitch", choices=["basic_pitch", "cqt"])
    ap.add_argument("--cross-mix", type=float, default=0.2, help="곡간 성부 교체 확률")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0, help="여러 프로세스로 나눌 때 이 프로세스 번호")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resplit", action="store_true", help="만들지 않고, 현재 분할에 맞게 옮기기만")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.resplit:
        print(resplit(args.data_dir, args.out))
        return 0
    n = build(
        args.data_dir,
        args.out,
        args.variants,
        args.renderer,
        args.backend,
        args.cross_mix,
        args.seed,
        args.limit,
        args.shard,
        args.num_shards,
    )
    print(f"합성 예시 {n}개 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
