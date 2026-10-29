"""평가 지표: mir_eval 래퍼와 리듬 일치율.

- 노트 F1: mir_eval.transcription (onset 허용 오차, offset 무시), 전 성부를 합쳐서
- 성부별 피치 정확도: 성부 궤적에 mir_eval.melody (같은 이름의 성부끼리)
- 프레임 다중 피치 정확도: mir_eval.multipitch
- 리듬 일치율: 정답 악보와 출력의 (박 위치, 음높이) 일치 비율. 전체를 1/4박 단위로 옮겨 보며
  가장 잘 맞는 이동을 고른다(시작 마디 차이 보정). 일치한 음 중 길이까지 같은 비율도 함께 낸다.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import numpy as np

from acappella.score_utils import merge_ties
from acappella.types import Score

TimeMap = Callable[[np.ndarray], np.ndarray]
HOP = 0.01


def constant_time_map(tempo_bpm: float, offset: float = 0.0) -> TimeMap:
    return lambda beats: offset + np.asarray(beats, dtype=float) * 60.0 / tempo_bpm


def table_time_map(beats: np.ndarray, seconds: np.ndarray) -> TimeMap:
    """(박, 초) 표로 보간하고 범위 밖은 양 끝 기울기로 외삽한다."""
    b = np.asarray(beats, dtype=float)
    s = np.asarray(seconds, dtype=float)
    if b.size < 2:
        raise ValueError("시간 대응표에 점이 둘 이상 있어야 합니다.")
    slope_lo = (s[1] - s[0]) / (b[1] - b[0])
    slope_hi = (s[-1] - s[-2]) / (b[-1] - b[-2])

    def f(x):
        x = np.asarray(x, dtype=float)
        y = np.interp(x, b, s)
        y = np.where(x < b[0], s[0] + (x - b[0]) * slope_lo, y)
        return np.where(x > b[-1], s[-1] + (x - b[-1]) * slope_hi, y)

    return f


def load_csv_time_map(path: Path) -> TimeMap:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return table_time_map(data[:, 0], data[:, 1])


def transpose(score: Score, semitones: int) -> Score:
    if not semitones:
        return score
    from acappella.types import QNote

    parts = {
        k: [
            QNote(
                q.start_beat,
                q.dur_beats,
                None if q.midi is None else q.midi + semitones,
                q.tie_next,
                q.lyric,
            )
            for q in v
        ]
        for k, v in score.parts.items()
    }
    return Score(
        score.title, score.tempo_bpm, score.time_sig, score.key_fifths, score.voices, parts
    )


def _midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def note_arrays(score: Score, tmap: TimeMap, voices: list[str] | None = None):
    """(intervals (n,2) 초, pitches Hz)."""
    iv, hz = [], []
    for name, part in score.parts.items():
        if voices is not None and name not in voices:
            continue
        for q in merge_ties(part):
            a, b = tmap(np.array([float(q.start_beat), float(q.start_beat + q.dur_beats)]))
            if b > a:
                iv.append((a, b))
                hz.append(_midi_hz(q.midi))
    if not iv:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(iv), np.array(hz, dtype=float)


def voice_f0(score: Score, voice: str, tmap: TimeMap, times: np.ndarray) -> np.ndarray:
    f0 = np.zeros_like(times)
    for q in merge_ties(score.parts.get(voice, [])):
        a, b = tmap(np.array([float(q.start_beat), float(q.start_beat + q.dur_beats)]))
        f0[(times >= a) & (times < b)] = _midi_hz(q.midi)
    return f0


def _time_axis(*arrays) -> np.ndarray:
    end = max((float(a[:, 1].max()) for a in arrays if a.size), default=0.0)
    return np.arange(0.0, end + HOP, HOP)


def note_f1(ref: Score, est: Score, ref_map: TimeMap, est_map: TimeMap, onset_tol=0.05) -> dict:
    import mir_eval

    ri, rp = note_arrays(ref, ref_map)
    ei, ep = note_arrays(est, est_map)
    if not len(ri) or not len(ei):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=onset_tol, offset_ratio=None
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def voice_pitch_accuracy(ref: Score, est: Score, ref_map: TimeMap, est_map: TimeMap) -> dict:
    """성부 이름별 mir_eval.melody 지표(Raw Pitch Accuracy, Overall Accuracy)."""
    import mir_eval

    ri, _ = note_arrays(ref, ref_map)
    ei, _ = note_arrays(est, est_map)
    times = _time_axis(ri, ei)
    out = {}
    for v in ref.voices:
        if v.name not in est.parts:
            continue
        rf = voice_f0(ref, v.name, ref_map, times)
        ef = voice_f0(est, v.name, est_map, times)
        if not rf.any():
            continue
        m = mir_eval.melody.evaluate(times, rf, times, ef)
        out[v.name] = {
            "raw_pitch_accuracy": float(m["Raw Pitch Accuracy"]),
            "overall_accuracy": float(m["Overall Accuracy"]),
            "voicing_recall": float(m["Voicing Recall"]),
        }
    return out


def _frames(score: Score, tmap: TimeMap, times: np.ndarray) -> list[np.ndarray]:
    frames: list[list[float]] = [[] for _ in times]
    for v in score.voices:
        f0 = voice_f0(score, v.name, tmap, times)
        for i in np.nonzero(f0)[0]:
            frames[i].append(f0[i])
    return [np.array(sorted(set(f))) for f in frames]


def multipitch_accuracy(ref: Score, est: Score, ref_map: TimeMap, est_map: TimeMap) -> dict:
    import mir_eval

    ri, _ = note_arrays(ref, ref_map)
    ei, _ = note_arrays(est, est_map)
    times = _time_axis(ri, ei)
    m = mir_eval.multipitch.evaluate(
        times, _frames(ref, ref_map, times), times, _frames(est, est_map, times)
    )
    return {
        "accuracy": float(m["Accuracy"]),
        "precision": float(m["Precision"]),
        "recall": float(m["Recall"]),
    }


def rhythm_agreement(ref: Score, est: Score, max_shift_beats: int = 16) -> dict:
    """박 위치 일치율. 전체 이동량은 1/4박 단위로 [-max, max] 안에서 가장 잘 맞는 값."""
    ref_notes = {
        name: [(q.start_beat, q.dur_beats, q.midi) for q in merge_ties(p)]
        for name, p in ref.parts.items()
    }
    est_sets = {
        name: {(q.start_beat, q.midi): q.dur_beats for q in merge_ties(p)}
        for name, p in est.parts.items()
    }
    total = sum(len(v) for v in ref_notes.values())
    if total == 0:
        return {"onset_match": 0.0, "duration_match": 0.0, "shift_beats": 0.0}
    best = (-1, 0, Fraction(0))
    for k in range(-4 * max_shift_beats, 4 * max_shift_beats + 1):
        sh = Fraction(k, 4)
        hit = dur_hit = 0
        for name, notes in ref_notes.items():
            est_map = est_sets.get(name, {})
            for s, d, m in notes:
                got = est_map.get((s + sh, m))
                if got is not None:
                    hit += 1
                    dur_hit += got == d
        if hit > best[0]:
            best = (hit, dur_hit, sh)
    hit, dur_hit, sh = best
    return {
        "onset_match": hit / total,
        "duration_match": dur_hit / hit if hit else 0.0,
        "shift_beats": float(sh),
    }


def evaluate(ref: Score, est: Score, ref_map: TimeMap, est_map: TimeMap) -> dict:
    return {
        "note": note_f1(ref, est, ref_map, est_map),
        "voices": voice_pitch_accuracy(ref, est, ref_map, est_map),
        "multipitch": multipitch_accuracy(ref, est, ref_map, est_map),
        "rhythm": rhythm_agreement(ref, est),
    }
