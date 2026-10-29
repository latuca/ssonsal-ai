"""곡당 처리 시간 측정(동아리원 최저 사양 PC에서 실행).

    uv run python -m eval.benchmark                 # 합성 3분 곡(5성부)으로 측정
    uv run python -m eval.benchmark --audio song.mp3 --voices Lead+SATB

단계별 소요 시간과 전체 시간을 출력하고 outputs/benchmark-<시각>.json에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from acappella.config import load_settings, parse_voices
from acappella.pipeline.run import TranscribeOptions, run_pipeline, setup_logging
from eval.run_eval import machine_info


def synth_song(path: Path, seconds: float = 180.0, sr: int = 22050, seed: int = 0) -> Path:
    """임의 화성 진행의 5성부 하모닉 합성음."""
    rng = np.random.default_rng(seed)
    chords = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]]
    n = int(seconds * sr)
    y = np.zeros(n)
    t_chord = 2.4
    for k in range(int(seconds / t_chord)):
        root = chords[k % 4]
        notes = [root[0] - 12, root[0], root[1], root[2], root[2] + 5 + int(rng.integers(0, 3))]
        a, b = int(k * t_chord * sr), int((k + 1) * t_chord * sr)
        t = np.arange(b - a) / sr
        env = np.minimum(1, np.minimum(t / 0.03, (t[::-1] + 1e-9) / 0.05))
        for m in notes:
            f = 440 * 2 ** ((m - 69) / 12)
            y[a:b] += env * sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3)) * 0.1
    y /= np.abs(y).max() + 1e-9
    sf.write(str(path), (0.8 * y).astype(np.float32), sr)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="곡당 처리 시간 측정")
    p.add_argument("--audio", help="측정할 음원(생략하면 합성 3분 곡)")
    p.add_argument("--voices", default="Lead+SATB")
    p.add_argument("--seconds", type=float, default=180.0, help="합성 곡 길이")
    args = p.parse_args(argv)
    setup_logging()
    settings = load_settings()
    stamps: list[tuple[str, float]] = []

    with tempfile.TemporaryDirectory() as tmp:
        src = args.audio or str(synth_song(Path(tmp) / "bench.wav", args.seconds))
        t0 = time.perf_counter()
        res = run_pipeline(
            src,
            parse_voices(args.voices),
            Path(tmp) / "out",
            lambda stage, _f: stamps.append((stage, time.perf_counter() - t0)),
            TranscribeOptions(save_debug=False),
            settings,
        )
    stages = {}
    for (name, t), (_, t_next) in zip(stamps, stamps[1:]):
        stages[name] = round(t_next - t, 2)
    report = {
        "machine": machine_info(),
        "total_seconds": round(res.seconds, 2),
        "stages": stages,
        "audio": args.audio or f"합성 {args.seconds:.0f}초",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = settings.outputs_dir / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
