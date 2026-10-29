"""5단계 노트 분할: 성부별 프레임 궤적 → 음표(NoteEvent).

1) 짧은 흔들림을 메디안 필터로 다듬는다.
2) 같은 음높이가 이어지는 구간을 음표 하나로 본다.
3) 최소 길이보다 짧은 구간은 앞(없으면 뒤) 음표에 붙이거나 버린다.
4) (선택) Basic Pitch 노트 후보의 onset이 같은 음높이 구간 안에 있으면 거기서 나눈다(재발음).
"""

from __future__ import annotations

import numpy as np

from acappella.config import Thresholds
from acappella.types import NoteEvent, PitchActivation, VoiceTrack


def _median_nan(x: np.ndarray, w: int) -> np.ndarray:
    """nan(무음)을 보존하는 메디안 필터. 창 안의 과반이 무음이면 무음."""
    if w <= 1 or x.size == 0:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, w)
    voiced = np.isfinite(win)
    out = np.full(x.shape, np.nan)
    keep = voiced.sum(axis=1) * 2 > w
    if keep.any():
        out[keep] = np.round(np.nanmedian(win[keep], axis=1))
    return out


def _runs(f0: np.ndarray) -> list[tuple[int, int, float]]:
    """(시작 프레임, 끝 프레임(미포함), 음높이) 목록. 무음 구간은 빼고 돌려준다."""
    runs = []
    i, n = 0, f0.size
    while i < n:
        j = i + 1
        while j < n and (f0[j] == f0[i] or (np.isnan(f0[j]) and np.isnan(f0[i]))):
            j += 1
        runs.append((i, j, f0[i]))
        i = j
    return runs


def _frame_dur(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if times.size > 1 else 0.01


def segment_track(
    track: VoiceTrack, th: Thresholds, onsets: list[NoteEvent] | None = None
) -> list[NoteEvent]:
    times = np.asarray(track.times, dtype=np.float64)
    if times.size == 0:
        return []
    dt = _frame_dur(times)
    f0 = _median_nan(np.asarray(track.f0_midi, dtype=np.float64), th.median_window)
    min_frames = max(1, int(round(th.min_note_sec / dt)))

    # 짧은 유성 구간을 앞 음표(바로 붙어 있을 때)에 흡수하거나 지운다
    runs = _runs(f0)
    for k, (a, b, p) in enumerate(runs):
        if np.isnan(p) or b - a >= min_frames:
            continue
        prev = runs[k - 1] if k > 0 else None
        nxt = runs[k + 1] if k + 1 < len(runs) else None
        if prev is not None and not np.isnan(prev[2]):
            f0[a:b] = prev[2]
        elif nxt is not None and not np.isnan(nxt[2]) and nxt[1] - nxt[0] >= min_frames:
            f0[a:b] = nxt[2]
        else:
            f0[a:b] = np.nan
    notes: list[NoteEvent] = []
    for a, b, p in _runs(f0):
        if np.isnan(p):
            continue
        onset, offset = float(times[a]), float(times[b - 1] + dt)
        notes.append(NoteEvent(onset=onset, offset=offset, midi=int(p)))

    if onsets:
        notes = _split_on_onsets(notes, onsets, th.onset_split_min_sec)
    return notes


def _split_on_onsets(
    notes: list[NoteEvent], onsets: list[NoteEvent], min_sec: float
) -> list[NoteEvent]:
    out: list[NoteEvent] = []
    for n in notes:
        cuts = sorted(
            o.onset
            for o in onsets
            if o.midi == n.midi and n.onset + min_sec <= o.onset <= n.offset - min_sec
        )
        start = n.onset
        for c in cuts:
            if c - start >= min_sec:
                out.append(NoteEvent(start, c, n.midi, n.confidence))
                start = c
        out.append(NoteEvent(start, n.offset, n.midi, n.confidence))
    return out


def segment_notes(
    tracks: list[VoiceTrack],
    act: PitchActivation | None = None,
    th: Thresholds | None = None,
) -> dict[str, list[NoteEvent]]:
    th = th or Thresholds()
    onsets = act.notes if act is not None else None
    return {t.voice.name: segment_track(t, th, onsets) for t in tracks}
