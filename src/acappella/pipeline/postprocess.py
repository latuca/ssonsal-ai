"""7단계 후처리: 짧은 가짜 노트 제거, 유니즌 채우기, 조표 추정, 음역 경고."""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np

from acappella.config import Thresholds
from acappella.score_utils import enforce_monophony, notes_only, range_warnings
from acappella.types import QNote, Score

log = logging.getLogger(__name__)

# Krumhansl-Kessler 조성 프로파일
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# 장조 으뜸음 pc → 조표(fifths). F#/Gb는 샵 6개로 쓴다.
_PC_TO_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5}


def _copy(q: QNote) -> QNote:
    return QNote(q.start_beat, q.dur_beats, q.midi, q.tie_next, q.lyric)


def remove_spurious(part: list[QNote], max_beats: Fraction, jump: int) -> list[QNote]:
    """아주 짧은 가짜 음을 정리한다.

    - 같은 음 사이에 끼어 있으면 그 음으로 바꾼다(뒤에서 merge_repeated_fragments가 합친다).
    - 앞뒤 음과 크게 떨어져 있으면 앞 음에 합친다(앞 음이 붙어 있지 않으면 지운다).
    """
    seq = [_copy(q) for q in notes_only(part)]
    out: list[QNote] = []
    for i, q in enumerate(seq):
        prev = out[-1] if out else None
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        if q.dur_beats <= max_beats:
            if (
                prev is not None
                and nxt is not None
                and prev.midi == nxt.midi != q.midi
                and prev.start_beat + prev.dur_beats == q.start_beat
                and q.start_beat + q.dur_beats == nxt.start_beat
            ):
                q.midi = prev.midi  # 같은 음 사이에 낀 짧은 다른 음: 앞뒤 음으로 본다
                out.append(q)
                continue
            far_prev = prev is None or abs(prev.midi - q.midi) >= jump
            far_next = nxt is None or abs(nxt.midi - q.midi) >= jump
            if far_prev and far_next:
                if prev is not None and prev.start_beat + prev.dur_beats == q.start_beat:
                    prev.dur_beats += q.dur_beats
                continue
        out.append(q)
    return out


def merge_repeated_fragments(part: list[QNote], max_beats: Fraction) -> list[QNote]:
    """같은 음이 쉼 없이 아주 짧게 다시 시작하면(분할 잡음) 앞 음에 합친다."""
    out: list[QNote] = []
    for q in (_copy(x) for x in notes_only(part)):
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.midi == q.midi
            and prev.start_beat + prev.dur_beats == q.start_beat
            and min(prev.dur_beats, q.dur_beats) <= max_beats
        ):
            prev.dur_beats += q.dur_beats
            continue
        out.append(q)
    return out


def _gaps(part: list[QNote], end: Fraction) -> list[tuple[Fraction, Fraction]]:
    gaps, t = [], Fraction(0)
    for q in sorted(notes_only(part), key=lambda x: x.start_beat):
        if q.start_beat > t:
            gaps.append((t, q.start_beat))
        t = max(t, q.start_beat + q.dur_beats)
    if end > t:
        gaps.append((t, end))
    return gaps


def fill_unisons(score: Score, max_beats: Fraction) -> Score:
    """앞뒤로 노래하는 성부 안의 짧은 빈 구간을, 그 구간에 음역이 맞는 이웃 성부의 음으로 채운다.

    다중 피치 추정이 유니즌을 하나로만 잡을 때 생기는 구멍을 메운다. 긴 쉼표는 진짜 쉼표로 본다.
    """
    voices = sorted(score.voices, key=lambda v: -(v.low_midi + v.high_midi) / 2)
    parts = {name: [_copy(q) for q in p] for name, p in score.parts.items()}
    for k, v in enumerate(voices):
        own = parts.get(v.name, [])
        if not notes_only(own):
            continue
        first = min(q.start_beat for q in notes_only(own))
        last = max(q.start_beat + q.dur_beats for q in notes_only(own))
        neighbors = [voices[j] for j in (k - 1, k + 1) if 0 <= j < len(voices)]
        added: list[QNote] = []
        for a, b in _gaps(own, last):
            if a < first or b - a > max_beats:
                continue
            for nb in neighbors:
                cand = [
                    q
                    for q in notes_only(score.parts.get(nb.name, []))
                    if q.start_beat < b and q.start_beat + q.dur_beats > a
                ]
                if not cand or not all(v.low_midi - 2 <= q.midi <= v.high_midi + 2 for q in cand):
                    continue
                for q in cand:
                    s = max(q.start_beat, a)
                    e = min(q.start_beat + q.dur_beats, b)
                    if e > s:
                        added.append(QNote(s, e - s, q.midi))
                break
        if added:
            parts[v.name] = enforce_monophony(own + added)
    return Score(
        score.title, score.tempo_bpm, score.time_sig, score.key_fifths, score.voices, parts
    )


def estimate_key(score: Score) -> int:
    """음표 길이로 가중한 pitch class 분포를 Krumhansl 프로파일과 비교해 조표를 고른다.

    단조는 나란한조(장조)의 조표를 쓴다.
    """
    hist = np.zeros(12)
    for part in score.parts.values():
        for q in notes_only(part):
            hist[q.midi % 12] += float(q.dur_beats)
    if hist.sum() == 0:
        return 0
    best, best_r = 0, -np.inf
    for tonic in range(12):
        for profile, rel_major in ((_MAJOR, tonic), (_MINOR, (tonic + 3) % 12)):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if r > best_r + 1e-9:
                best_r, best = r, _PC_TO_FIFTHS[rel_major]
    return best


def postprocess(score: Score, th: Thresholds | None = None, key_fifths: int | None = None) -> Score:
    """key_fifths를 주면 조표 추정을 건너뛰고 그 값을 쓴다."""
    th = th or Thresholds()
    max_spur = Fraction(th.spurious_max_beats).limit_denominator(64)
    parts = {
        name: merge_repeated_fragments(remove_spurious(part, max_spur, th.spurious_jump), max_spur)
        for name, part in score.parts.items()
    }
    out = Score(score.title, score.tempo_bpm, score.time_sig, score.key_fifths, score.voices, parts)
    out = fill_unisons(out, Fraction(th.unison_fill_max_beats).limit_denominator(64))
    out.key_fifths = estimate_key(out) if key_fifths is None else key_fifths
    for msg in range_warnings(out, th.range_warn_margin):
        log.warning(msg)
    return out
