"""(v2) 리드 성부 가사: 음절 분할과 음표 배치.

- 한국어: 한글 음절 글자 하나가 한 음절이다.
- 영어: 모음 묶음 기준의 간단한 규칙으로 나눈다(VCV → V-CV, VCCV → VC-CV, 끝의 묵음 e).
  사용자가 ``hel-lo``처럼 하이픈을 넣으면 그 분할을 그대로 쓴다.
- ``_``는 음표 하나를 건너뛴다(그 음표에는 가사를 붙이지 않음).
- 단어 안의 음절은 '-'로 끝난다(NWC Lyric 표기와 같다). 가사 자동 인식은 범위 밖이다.
"""

from __future__ import annotations

import logging
import re

from acappella.types import QNote, Score

log = logging.getLogger(__name__)

_HANGUL = re.compile(r"[가-힣]")
_VOWELS = "aeiouyAEIOUY"
_PUNCT = ".,!?;:\"'()[]{}…“”‘’·~"


def is_korean(text: str) -> bool:
    return bool(_HANGUL.search(text))


def split_english_word(word: str) -> list[str]:
    """영어 단어 하나를 음절로 나눈다(규칙 기반 근사)."""
    w = word
    if len(w) <= 3:
        return [w]
    low = w.lower()
    # 모음 위치(첫 글자 y는 자음, 끝의 묵음 e는 모음 아님)
    is_v = [c in _VOWELS for c in low]
    if low[0] == "y":
        is_v[0] = False
    if low.endswith("e") and not low.endswith(("le", "ee", "ye")) and sum(is_v) > 1:
        is_v[-1] = False
    if low.endswith("es") and len(low) > 4 and low[-3] not in "sxzhcg" and sum(is_v) > 1:
        is_v[-2] = False
    groups = []  # 모음 묶음의 (시작, 끝)
    i = 0
    while i < len(w):
        if is_v[i]:
            j = i
            while j + 1 < len(w) and is_v[j + 1]:
                j += 1
            groups.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(groups) <= 1:
        return [w]
    cuts = []
    for (_, a_end), (b_start, _) in zip(groups, groups[1:]):
        n_cons = b_start - a_end - 1
        if n_cons <= 1:
            cuts.append(a_end + 1)  # V-CV
        else:
            cuts.append(a_end + 2)  # VC-CV
    out, prev = [], 0
    for c in cuts:
        out.append(w[prev:c])
        prev = c
    out.append(w[prev:])
    return [s for s in out if s]


def split_syllables(text: str) -> list[str | None]:
    """가사 문자열 → 음절 토큰 목록(None은 건너뛰는 자리)."""
    tokens: list[str | None] = []
    for raw in text.split():
        if raw == "_":
            tokens.append(None)
            continue
        word = raw.strip(_PUNCT)
        if not word:
            continue
        if is_korean(word):
            sylls = [c for c in word if _HANGUL.match(c) or c.isalnum()]
            tokens.extend(sylls)  # 한국어는 음절마다 띄어 쓴다
            continue
        parts = [p for p in word.split("-") if p] if "-" in word else split_english_word(word)
        for k, p in enumerate(parts):
            tokens.append(p + "-" if k < len(parts) - 1 else p)
    return tokens


def place_lyrics(score: Score, voice: str, text: str) -> Score:
    """voice 성부의 음표에 가사 음절을 앞에서부터 하나씩 붙인 새 Score를 돌려준다."""
    if voice not in score.parts:
        raise KeyError(f"가사를 넣을 성부가 없습니다: {voice}")
    tokens = split_syllables(text)
    part = [QNote(q.start_beat, q.dur_beats, q.midi, q.tie_next, None) for q in score.parts[voice]]
    slots = []
    prev_tie = False
    for q in sorted(part, key=lambda x: x.start_beat):
        if q.midi is None:
            prev_tie = False
            continue
        if not prev_tie:
            slots.append(q)
        prev_tie = q.tie_next
    for q, tok in zip(slots, tokens):
        q.lyric = tok
    extra = len(tokens) - len(slots)
    if extra > 0:
        log.warning("%s 성부의 음표보다 가사 음절이 %d개 많아 뒷부분을 버렸습니다.", voice, extra)
    parts = dict(score.parts)
    parts[voice] = part
    return Score(
        score.title, score.tempo_bpm, score.time_sig, score.key_fifths, score.voices, parts
    )


def default_lyric_voice(score: Score) -> str:
    """가사를 넣을 기본 성부: Lead가 있으면 Lead, 없으면 가장 높은 성부."""
    for v in score.voices:
        if v.name.lower().startswith("lead"):
            return v.name
    return max(score.voices, key=lambda v: v.low_midi + v.high_midi).name
