"""악보와 음원의 크로마 DTW 정렬, 조옮김 감지.

    uv run python -m tools.align --nwctxt data/nwctxt/s0001.nwctxt --audio song.m4a \
        --out data/align/s0001.csv

- 악보 음표로 만든 크로마와 음원 크로마(chroma_cqt)를 부분 수열 DTW로 정렬한다
  (음원 앞뒤의 인트로·아웃트로 허용).
- 조옮김은 반음 이동 12가지 중 정렬 비용이 가장 낮은 것으로 정한다.
- 결과 CSV는 beat,seconds (악보 박마다 음원 시각). eval/run_eval.py가 쓴다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acappella.score_utils import merge_ties
from acappella.types import Score

SR = 22050
HOP = 2048  # 약 93ms


@dataclass
class AlignResult:
    transpose_semitones: int  # 음원이 악보보다 높으면 양수(-6~5)
    cost: float  # 경로 길이로 나눈 평균 비용(코사인 거리, 0~1)
    beats: np.ndarray
    seconds: np.ndarray


def score_chroma(score: Score, frame_sec: float) -> tuple[np.ndarray, float]:
    """(12, T) 크로마와 초당 박 수. 템포는 악보 템포로 일정하다고 본다."""
    spb = 60.0 / score.tempo_bpm
    notes = [q for p in score.parts.values() for q in merge_ties(p)]
    end = max((float(q.start_beat + q.dur_beats) for q in notes), default=1.0) * spb
    T = max(1, int(np.ceil(end / frame_sec)))
    C = np.zeros((12, T))
    for q in notes:
        a = int(float(q.start_beat) * spb / frame_sec)
        b = max(a + 1, int(float(q.start_beat + q.dur_beats) * spb / frame_sec))
        C[q.midi % 12, a:b] += 1.0
    return _normalize(C), spb


def _normalize(C: np.ndarray) -> np.ndarray:
    C = C + 1e-3  # 무음 프레임도 코사인 거리가 정의되도록
    n = np.linalg.norm(C, axis=0, keepdims=True)
    return C / np.maximum(n, 1e-9)


def audio_chroma(samples: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = samples if sr == SR else librosa.resample(samples, orig_sr=sr, target_sr=SR)
    return _normalize(librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP))


def align(score: Score, samples: np.ndarray, sr: int) -> AlignResult:
    import librosa

    frame = HOP / SR
    Y = audio_chroma(samples, sr)
    X0, spb = score_chroma(score, frame)
    best = None
    for k in range(12):
        X = np.roll(X0, k, axis=0)
        # subseq=True면 경로는 마지막 행의 최솟값에서 거슬러 올라간다(wp는 끝→처음 순서)
        D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine", subseq=True, backtrack=True)
        cost = float(D[wp[0, 0], wp[0, 1]]) / len(wp)
        if best is None or cost < best[1]:
            best = (k, cost, wp[::-1])
    k, cost, wp = best
    shift = k if k < 6 else k - 12
    # 악보 프레임 → 음원 프레임 (같은 악보 프레임이 여러 번 나오면 평균)
    sx, sy = wp[:, 0], wp[:, 1]
    xs = np.unique(sx)
    ys = np.array([sy[sx == x].mean() for x in xs])
    n_beats = int(X0.shape[1] * frame / spb)
    beats = np.arange(0, n_beats + 1, dtype=float)
    seconds = np.interp(beats * spb / frame, xs, ys) * frame
    return AlignResult(shift, cost, beats, seconds)


def write_csv(res: AlignResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["beat,seconds"] + [f"{b:g},{s:.3f}" for b, s in zip(res.beats, res.seconds)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    from acappella.config import load_settings
    from acappella.nwc.reader import read_nwctxt
    from acappella.pipeline.ingest import ingest

    ap = argparse.ArgumentParser(description="악보-음원 크로마 DTW 정렬")
    ap.add_argument("--nwctxt", type=Path, required=True)
    ap.add_argument("--audio", required=True, help="음원 파일 또는 URL")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    clip = ingest(args.audio, Path("data/cache/align"), load_settings())
    res = align(read_nwctxt(args.nwctxt), clip.samples, clip.sr)
    write_csv(res, args.out)
    print(f"조옮김 {res.transpose_semitones:+d}반음, 비용 {res.cost:.3f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
