"""층 B(silver)·층 C(gold) 실제 음원 → 학습용 npz (synth.py와 같은 형식).

    uv run python -m training.real_data --split train --out data/real
    uv run python -m training.real_data --split valid --out data/real

곡마다 음원의 Basic Pitch contour와, 정렬 CSV(data/align/<song_id>.csv)로 악보 박을 초로 옮긴
성부별 정답 궤적을 만든다. 음원은 메타데이터 audio_path → data/audio/<song_id>.* → youtube_url
순서로 찾는다. test 분할은 만들지 않는다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from acappella.config import load_settings
from acappella.meta import SongMeta, load_all
from acappella.nwc.reader import read_nwctxt
from acappella.pipeline.ingest import ingest
from acappella.pipeline.pitch import HOP, SR, estimate_pitch
from acappella.pipeline.preprocess import preprocess
from acappella.pipeline.voices import ONNX_K, encode_voice_ranges, order_voices
from acappella.score_utils import merge_ties
from eval.metrics import load_csv_time_map, transpose
from eval.run_eval import resolve_audio
from training.synth import voices_for

log = logging.getLogger(__name__)


def make_real_example(meta: SongMeta, data_dir: Path, settings, backend: str = "basic_pitch"):
    align = data_dir / "align" / f"{meta.song_id}.csv"
    if not align.exists():
        raise FileNotFoundError(f"{meta.song_id}: 정렬 파일이 없습니다({align}).")
    score = transpose(
        read_nwctxt(data_dir / "nwctxt" / f"{meta.song_id}.nwctxt"), meta.transpose_semitones
    )
    voices = voices_for(meta, score)
    tmap = load_csv_time_map(align)
    clip = ingest(
        resolve_audio(meta, data_dir), data_dir / "cache" / "real" / meta.song_id, settings
    )
    act = estimate_pitch(preprocess(clip), backend)
    T = act.salience.shape[0]
    frame_t = np.arange(T) * HOP / SR
    specs = [v for _, v in voices]
    order = order_voices(specs)
    target = np.full((ONNX_K, T), -1, dtype=np.int16)
    for slot, vi in enumerate(order):
        staff = voices[vi][0]
        for q in merge_ties(score.parts[staff]):
            on, off = tmap(np.array([float(q.start_beat), float(q.start_beat + q.dur_beats)]))
            target[slot, (frame_t >= on) & (frame_t < off)] = q.midi
    ranges = encode_voice_ranges([specs[i] for i in order])
    return act.salience.astype(np.float16), target, ranges


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="실제 음원 학습 데이터(층 B·C)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--split", choices=["train", "valid"], required=True)
    ap.add_argument("--tiers", default="B", help="쓸 tier(예: B 또는 BC)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    metas = [
        m
        for m in load_all(args.data_dir / "meta" / "songs")
        if m.split == args.split and m.tier in args.tiers and not m.is_medley
    ]
    n = 0
    for m in metas:
        dst = args.out / args.split / f"{m.song_id}.npz"
        if dst.exists():
            n += 1
            continue
        try:
            sal, tgt, ranges = make_real_example(m, args.data_dir, settings)
        except Exception as e:  # noqa: BLE001 - 한 곡 실패로 전체를 멈추지 않는다
            log.warning("%s 건너뜀: %s", m.song_id, e)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, salience=sal, target_midi=tgt, ranges=ranges, song_id=m.song_id)
        n += 1
    print(f"{args.split}: {n}곡 → {args.out / args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
