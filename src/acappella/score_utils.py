"""Score/QNote 공통 유틸리티와 불변 조건 검사."""

from __future__ import annotations

import logging
from fractions import Fraction

from acappella.types import QNote, Score

log = logging.getLogger(__name__)


def notes_only(part: list[QNote]) -> list[QNote]:
    return [q for q in part if q.midi is not None]


def part_end(part: list[QNote]) -> Fraction:
    return max((q.start_beat + q.dur_beats for q in part), default=Fraction(0))


def score_end(score: Score) -> Fraction:
    return max((part_end(p) for p in score.parts.values()), default=Fraction(0))


def merge_ties(part: list[QNote]) -> list[QNote]:
    """타이로 이어진 같은 음을 하나로 합치고 쉼표를 뺀 음표 열을 돌려준다(비교·평가용)."""
    out: list[QNote] = []
    pending_tie = False
    for q in sorted(part, key=lambda x: x.start_beat):
        if q.midi is None:
            pending_tie = False
            continue
        prev = out[-1] if out else None
        if (
            pending_tie
            and prev is not None
            and prev.midi == q.midi
            and prev.start_beat + prev.dur_beats == q.start_beat
        ):
            prev.dur_beats += q.dur_beats
            prev.tie_next = q.tie_next
        else:
            out.append(QNote(q.start_beat, q.dur_beats, q.midi, q.tie_next, q.lyric))
        pending_tie = q.tie_next
    for q in out:
        q.tie_next = False
    return out


def monophony_violations(part: list[QNote]) -> list[tuple[QNote, QNote]]:
    """한 성부 안에서 겹치는 (앞 음표, 뒤 음표) 쌍."""
    seq = sorted(notes_only(part), key=lambda x: x.start_beat)
    return [(a, b) for a, b in zip(seq, seq[1:]) if a.start_beat + a.dur_beats > b.start_beat]


def enforce_monophony(part: list[QNote]) -> list[QNote]:
    """겹치는 음표는 앞 음표를 잘라 단선율로 만든다. 길이가 0이 된 음표는 뺀다."""
    seq = sorted(
        (
            QNote(q.start_beat, q.dur_beats, q.midi, q.tie_next, q.lyric)
            for q in part
            if q.midi is not None
        ),
        key=lambda x: (x.start_beat, -x.dur_beats, -(x.midi or 0)),
    )
    out: list[QNote] = []
    for q in seq:
        if out and out[-1].start_beat == q.start_beat:
            continue  # 같은 시작점이면 가장 긴 음(같으면 높은 음)을 남긴다
        if out and out[-1].start_beat + out[-1].dur_beats > q.start_beat:
            out[-1].dur_beats = q.start_beat - out[-1].start_beat
        if q.dur_beats > 0:
            out.append(q)
    return [q for q in out if q.dur_beats > 0]


def range_warnings(score: Score, margin: int = 3) -> list[str]:
    msgs = []
    for v in score.voices:
        for q in notes_only(score.parts.get(v.name, [])):
            if q.midi < v.low_midi - margin or q.midi > v.high_midi + margin:
                msgs.append(
                    f"{v.name}: {q.start_beat}박의 음 {q.midi}이 음역 {v.low_midi}~{v.high_midi}을 벗어났습니다."
                )
    return msgs


def check_invariants(score: Score, range_margin: int = 3) -> list[str]:
    """단선율과 음역 불변 조건을 검사해 위반 목록을 돌려준다. 음역 위반은 로그 경고로도 남긴다.

    마디 길이 불변 조건은 nwctxt 단계에서 검사한다(:func:`acappella.nwc.reader.measure_lengths`).
    """
    problems = []
    for name, part in score.parts.items():
        for a, b in monophony_violations(part):
            problems.append(f"{name}: {a.start_beat}박과 {b.start_beat}박의 음표가 겹칩니다.")
        for q in part:
            if q.dur_beats <= 0:
                problems.append(f"{name}: {q.start_beat}박에 길이가 0 이하인 음표가 있습니다.")
    for msg in range_warnings(score, range_margin):
        log.warning(msg)
    return problems
