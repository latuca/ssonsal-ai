"""gold 곡으로 파이프라인을 돌려 지표 리포트를 낸다.

    uv run python -m eval.run_eval --meta-dir data/meta/songs --split test
    uv run python -m eval.run_eval --meta-dir data/meta/songs --split valid --voice-method rule

곡마다 필요한 것:
- 메타데이터 JSON (tier C). ``voices``는 악보 스태프 순서와 같은 성부 이름 목록.
- 정답 악보: ``data/nwctxt/<song_id>.nwctxt`` (tools/extract_nwctxt.py) 또는 nwc_path가 .nwctxt
- 음원: 메타데이터 ``audio_path``, 없으면 ``data/audio/<song_id>.*``, 없으면 youtube_url
- (권장) 시간 정렬: ``data/align/<song_id>.csv`` (beat,seconds; tools/align.py). 없으면 악보 템포로
  0초부터 일정하게 흐른다고 가정한다.

결과: ``outputs/eval/<시각>/report.md``, ``report.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from acappella.config import Settings, default_voice, load_settings
from acappella.meta import SongMeta, load_all
from acappella.nwc.reader import read_nwctxt
from acappella.pipeline.run import TranscribeOptions, run_pipeline, setup_logging
from acappella.types import Score, VoiceSpec
from eval.metrics import (
    constant_time_map,
    evaluate,
    load_csv_time_map,
    table_time_map,
    transpose,
)

log = logging.getLogger(__name__)
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm")


def resolve_reference(meta: SongMeta, data_dir: Path) -> Path:
    cand = data_dir / "nwctxt" / f"{meta.song_id}.nwctxt"
    if cand.exists():
        return cand
    if meta.nwc_path.endswith(".nwctxt") and Path(meta.nwc_path).exists():
        return Path(meta.nwc_path)
    raise FileNotFoundError(f"{meta.song_id}: 정답 nwctxt가 없습니다({cand}).")


def resolve_audio(meta: SongMeta, data_dir: Path) -> str:
    if p := meta.extra.get("audio_path"):
        return str(p)
    for ext in AUDIO_EXTS:
        cand = data_dir / "audio" / f"{meta.song_id}{ext}"
        if cand.exists():
            return str(cand)
    if meta.youtube_url:
        return meta.youtube_url
    raise FileNotFoundError(f"{meta.song_id}: 음원이 없습니다.")


def map_reference_voices(ref: Score, names: list[str]) -> tuple[Score, list[VoiceSpec]]:
    """정답 악보의 스태프를 메타데이터 성부 이름으로 바꾼다(이름이 같으면 이름으로, 아니면 순서로)."""
    staves = [v.name for v in ref.voices]
    if not names:
        names = staves
    mapping = {}
    for i, name in enumerate(names):
        if name in staves:
            mapping[name] = name
        elif i < len(staves):
            mapping[name] = staves[i]
    voices = []
    parts = {}
    for name in names:
        if name not in mapping:
            continue
        spec = default_voice(name)
        src = ref.voice(mapping[name])
        spec.clef = spec.clef if src.clef not in ("treble", "treble8vb", "bass") else src.clef
        voices.append(spec)
        parts[name] = ref.parts[mapping[name]]
    return Score(ref.title, ref.tempo_bpm, ref.time_sig, ref.key_fifths, voices, parts), voices


def est_time_map(job_dir: Path, score: Score):
    path = job_dir / "6_timemap.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if len(d["beats"]) >= 2:
            return table_time_map(np.array(d["beats"]), np.array(d["seconds"]))
    return constant_time_map(score.tempo_bpm)


def eval_song(meta: SongMeta, data_dir: Path, settings: Settings, voice_method: str) -> dict:
    ref_raw = read_nwctxt(resolve_reference(meta, data_dir))
    ref, voices = map_reference_voices(transpose(ref_raw, meta.transpose_semitones), meta.voices)
    align = data_dir / "align" / f"{meta.song_id}.csv"
    if align.exists():
        ref_map = load_csv_time_map(align)
    else:
        log.warning("%s: 시간 정렬 파일이 없어 일정 템포를 가정합니다.", meta.song_id)
        ref_map = constant_time_map(ref.tempo_bpm)
    out_dir = settings.outputs_dir / "eval_scores"
    opts = TranscribeOptions(title=meta.song_id, voice_method=voice_method, save_debug=True)
    res = run_pipeline(resolve_audio(meta, data_dir), voices, out_dir, None, opts, settings)
    metrics = evaluate(ref, res.score, ref_map, est_time_map(res.job_dir, res.score))
    return {
        "song_id": meta.song_id,
        "title": meta.title,
        "n_voices": len(voices),
        "genre": meta.genre or "미상",
        "seconds": res.seconds,
        "metrics": metrics,
        "job_dir": str(res.job_dir),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict]) -> dict:
    def agg(rs):
        return {
            "songs": len(rs),
            "note_f1": _mean([r["metrics"]["note"]["f1"] for r in rs]),
            "voice_raw_pitch": _mean(
                [v["raw_pitch_accuracy"] for r in rs for v in r["metrics"]["voices"].values()]
            ),
            "multipitch_acc": _mean([r["metrics"]["multipitch"]["accuracy"] for r in rs]),
            "rhythm_onset": _mean([r["metrics"]["rhythm"]["onset_match"] for r in rs]),
            "seconds": _mean([r["seconds"] for r in rs]),
        }

    by_n, by_genre, by_voice = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_n[
            "4성부" if r["n_voices"] <= 4 else "5~6성부" if r["n_voices"] <= 6 else "7성부 이상"
        ].append(r)
        by_genre[r["genre"]].append(r)
        for name, v in r["metrics"]["voices"].items():
            by_voice[name].append(v["raw_pitch_accuracy"])
    return {
        "all": agg(rows),
        "by_voice_count": {k: agg(v) for k, v in by_n.items()},
        "by_genre": {k: agg(v) for k, v in by_genre.items()},
        "by_voice": {k: _mean(v) for k, v in by_voice.items()},
    }


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def render_markdown(summary: dict, rows: list[dict], info: dict, args) -> str:
    def line(name, a):
        return (
            f"| {name} | {a['songs']} | {a['note_f1']:.3f} | {a['voice_raw_pitch']:.3f} | "
            f"{a['multipitch_acc']:.3f} | {a['rhythm_onset']:.3f} | {a['seconds']:.1f} |"
        )

    head = "| 묶음 | 곡 수 | 노트 F1 | 성부 피치 정확도 | 다중 피치 정확도 | 리듬 일치율 | 곡당 처리(초) |\n|---|---|---|---|---|---|---|"
    out = [
        "# 평가 리포트",
        "",
        f"- 날짜: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 분할: {args.split}, 성부 배정: {args.voice_method}",
        f"- 기기: {info['platform']}, {info['processor']}, CPU {info['cpu_count']}개",
        "",
        "## 요약",
        "",
        head,
        line("전체", summary["all"]),
    ]
    out += [line(k, v) for k, v in summary["by_voice_count"].items()]
    out += ["", "## 장르별", "", head] + [line(k, v) for k, v in summary["by_genre"].items()]
    out += ["", "## 성부별 피치 정확도", "", "| 성부 | Raw Pitch Accuracy |", "|---|---|"]
    out += [f"| {k} | {v:.3f} |" for k, v in summary["by_voice"].items()]
    out += [
        "",
        "## 곡별",
        "",
        "| 곡 | 성부 수 | 노트 F1 | 다중 피치 | 리듬 | 처리(초) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        out.append(
            f"| {r['song_id']} {r['title']} | {r['n_voices']} | {m['note']['f1']:.3f} | "
            f"{m['multipitch']['accuracy']:.3f} | {m['rhythm']['onset_match']:.3f} | {r['seconds']:.1f} |"
        )
    out += ["", "수정 시간·수정 음표 수(실사용 효용)는 동아리원 비교 실험으로 따로 기록한다."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="gold 곡 평가 리포트")
    p.add_argument("--meta-dir", type=Path, default=Path("data/meta/songs"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    p.add_argument("--tier", default="C")
    p.add_argument("--voice-method", default="rule", choices=["rule", "onnx", "auto"])
    p.add_argument("--limit", type=int, help="앞에서부터 몇 곡만")
    p.add_argument("--out", type=Path, help="리포트 폴더(기본: outputs/eval/<시각>)")
    args = p.parse_args(argv)
    setup_logging()
    settings = load_settings()

    songs = [m for m in load_all(args.meta_dir) if m.tier == args.tier]
    if args.split != "all":
        songs = [m for m in songs if m.split == args.split]
    if args.limit:
        songs = songs[: args.limit]
    if not songs:
        print("평가할 곡이 없습니다. 메타데이터의 tier와 split을 확인해 주세요.")
        return 1
    rows = []
    for m in songs:
        try:
            rows.append(eval_song(m, args.data_dir, settings, args.voice_method))
            log.info("%s 완료", m.song_id)
        except Exception:  # noqa: BLE001 - 한 곡 실패로 전체를 멈추지 않는다
            log.exception("%s 평가 실패", m.song_id)
    if not rows:
        return 1
    info = machine_info()
    summary = summarize(rows)
    out = args.out or settings.outputs_dir / "eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(
            {"machine": info, "summary": summary, "songs": rows}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    (out / "report.md").write_text(render_markdown(summary, rows, info, args), encoding="utf-8")
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
