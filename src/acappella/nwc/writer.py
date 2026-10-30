"""Score → nwctxt.

성부 하나에 스태프 하나, 화음 없이 단선율로 쓴다. 빈 구간은 쉼표로 채워 모든 마디 길이를
박자표에 맞추고, 마디를 넘는 음은 타이로 나눈다. 셋잇단음표는 v1에서 쓰지 않는다.

음가를 어디서 시작할 수 있는지(4/4)는 동아리 악보 412곡의 실제 분포를 따랐다. 예를 들어 4분음표는
박 위(92%)나 반 박 뒤(7%), 점4분음표는 마디 앞·가운데와 반 박 뒤(1/2, 5/2), 2분음표는 0·1·2박에서
시작한다. 8분·16분음표는 4/4에서 반 마디, 그 밖에는 한 박 단위로 빔을 묶는다(16분음표가 섞이면
한 박). 실제 파일의 빔은 모두 ``Opts:Stem=Up|Down,Beam=First|Beam|Beam=End`` 꼴이라 꼬리 방향을 함께 쓴다.

주의: 아래 문법은 docs/design.md의 예시와 공개된 nwctxt 문서를 기준으로 했다. 실제 동아리
NWC에서 내보낸 파일로 왕복 테스트(tests/test_nwc_roundtrip.py, tests/fixtures/local/)를
통과시키기 전까지는 "검증 전"으로 본다.
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from pathlib import Path

from acappella.nwc.pitch_pos import BarAccidentals, fifths_to_signature
from acappella.score_utils import enforce_monophony, merge_ties, monophony_violations, score_end
from acappella.types import QNote, Score

log = logging.getLogger(__name__)

NWC_VERSION = "2.75"

# (박 길이, Dur 문자열, 꾸밈 없는 기본 음가) — 큰 것부터
DURATIONS: list[tuple[Fraction, str, Fraction]] = [
    (Fraction(4), "Whole", Fraction(4)),
    (Fraction(3), "Half,Dotted", Fraction(2)),
    (Fraction(2), "Half", Fraction(2)),
    (Fraction(3, 2), "4th,Dotted", Fraction(1)),
    (Fraction(1), "4th", Fraction(1)),
    (Fraction(3, 4), "8th,Dotted", Fraction(1, 2)),
    (Fraction(1, 2), "8th", Fraction(1, 2)),
    (Fraction(3, 8), "16th,Dotted", Fraction(1, 4)),
    (Fraction(1, 4), "16th", Fraction(1, 4)),
    (Fraction(1, 8), "32nd", Fraction(1, 8)),
    (Fraction(1, 16), "64th", Fraction(1, 16)),
]

CLEF_LINES = {
    "treble": "|Clef|Type:Treble",
    "treble8vb": "|Clef|Type:Treble|OctaveShift:Octave Down",
    "bass": "|Clef|Type:Bass",
    "alto": "|Clef|Type:Alto",
    "tenor": "|Clef|Type:Tenor",
}


def escape_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("|", "\\|").replace("\n", "\\n")


# 4/4에서 음가별로 시작할 수 있는 마디 안 위치(박). 없는 음가는 아래 일반 규칙을 따른다.
_COMMON_TIME_STARTS: dict[Fraction, set[Fraction]] = {
    Fraction(3): {Fraction(0), Fraction(1)},
    Fraction(2): {Fraction(0), Fraction(1), Fraction(2)},
    Fraction(3, 2): {Fraction(0), Fraction(1, 2), Fraction(2), Fraction(5, 2)},
    Fraction(1): {Fraction(k, 2) for k in range(8) if k != 3},  # 3/2는 마디 가운데를 가린다
}


def _allowed(d: Fraction, plain: Fraction, pos: Fraction, bar_len: Fraction) -> bool:
    if d == 4:
        return pos == 0 and bar_len == 4
    if bar_len == 4 and d in _COMMON_TIME_STARTS:
        return pos in _COMMON_TIME_STARTS[d]
    if pos % min(plain if d == plain else 2 * plain, Fraction(1)) != 0:
        return False  # 점음표는 기본 음가 두 배 격자에서만 시작한다
    return not (
        pos % 1 != 0 and pos + d > (pos // 1) + 1
    )  # 박 중간에서 시작하면 다음 박을 넘지 않는다


def split_duration(pos_in_bar: Fraction, length: Fraction, bar_len: Fraction) -> list[Fraction]:
    """마디 안의 한 구간(마디를 넘지 않음)을 표기 가능한 음가 열로 나눈다.

    4/4에서 4분음표 이상은 ``_COMMON_TIME_STARTS`` 위치에서만 시작한다. 그 밖의 음가·박자는
    자기 기본 음가(최대 4분음표) 격자 위에서만 시작하고, 박 중간에서 시작하면 다음 박을 넘지 않는다.
    온음표는 마디 첫머리에서 4/4 마디를 꽉 채울 때만 쓴다.
    """
    out: list[Fraction] = []
    pos, remaining = pos_in_bar, length
    while remaining > 0:
        for d, _, plain in DURATIONS:
            if d <= remaining and _allowed(d, plain, pos, bar_len):
                break
        else:
            # 격자에 맞지 않는 구간: 들어가는 가장 큰 음가로 채운다
            fits = [d for d, _, _ in DURATIONS if d <= remaining]
            if not fits:
                raise ValueError(f"표기할 수 없는 음가입니다(64분음표보다 짧음): {remaining}")
            d = fits[0]
        out.append(d)
        pos += d
        remaining -= d
    return out


def dur_string(d: Fraction) -> str:
    for value, text, _ in DURATIONS:
        if value == d:
            return text
    raise ValueError(f"표기할 수 없는 음가: {d}")


def _fill_rests(part: list[QNote], total: Fraction) -> list[QNote]:
    seq: list[QNote] = []
    t = Fraction(0)
    for q in sorted((q for q in part if q.midi is not None), key=lambda x: x.start_beat):
        if q.start_beat > t:
            seq.append(QNote(t, q.start_beat - t, None))
        seq.append(q)
        t = q.start_beat + q.dur_beats
    if total > t:
        seq.append(QNote(t, total - t, None))
    return seq


def _lyric_text(tokens: list[str]) -> str:
    """음절 토큰을 Lyric 문자열로 잇는다. '-'로 끝난 음절은 다음 음절과 붙이고,
    건너뛰는 자리(_)는 공백으로 둘러싼다."""
    while tokens and tokens[-1] == "_":
        tokens = tokens[:-1]
    text = ""
    for tok in tokens:
        if text and (tok == "_" or not text.endswith("-")):
            text += " "
        text += tok
    return text


_BEAMABLE = Fraction(1, 2)  # 8분음표 이하


def _beam_unit(bar_len: Fraction, has_16th: bool) -> Fraction:
    return Fraction(2) if bar_len == 4 and not has_16th else Fraction(1)


def _beam_opts(events: list[dict], bar_len: Fraction) -> None:
    """같은 묶음 단위 안에서 이어지는 8분·16분음표에 Beam=First/Beam/Beam=End를 붙인다."""
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_key = None
    for ev in events:
        if ev["kind"] != "note" or ev["plain"] > _BEAMABLE:
            if len(cur) > 1:
                groups.append(cur)
            cur, cur_key = [], None
            continue
        key = (ev["bar"], ev["half"])
        if cur and key != cur_key:
            if len(cur) > 1:
                groups.append(cur)
            cur = []
        cur.append(ev)
        cur_key = key
    if len(cur) > 1:
        groups.append(cur)
    for g in groups:
        # 16분음표가 섞인 반 마디는 한 박씩 다시 나눈다
        if any(e["plain"] < _BEAMABLE for e in g) and bar_len == 4:
            sub: dict[Fraction, list[dict]] = {}
            for e in g:
                sub.setdefault(e["pos"] // 1, []).append(e)
            parts = [x for x in sub.values() if len(x) > 1]
        else:
            parts = [g]
        for part in parts:
            mean = sum(e["staff_pos"] for e in part) / len(part)
            stem = "Down" if mean >= 0 else "Up"
            for i, e in enumerate(part):
                beam = "Beam=First" if i == 0 else "Beam=End" if i == len(part) - 1 else "Beam"
                e["opts"] = f"|Opts:Stem={stem},{beam}"


def _staff_events(part: list[QNote], score: Score, clef: str, total: Fraction):
    """한 스태프의 (줄 목록, 가사 토큰 목록)을 만든다."""
    bar_len = score.measure_beats()
    acc = BarAccidentals(score.key_fifths, clef)
    events: list[dict] = []
    lyric_tokens: list[str] = []
    for q in _fill_rests(part, total):
        start, end = q.start_beat, q.start_beat + q.dur_beats
        pieces: list[tuple[Fraction, Fraction]] = []
        t = start
        while t < end:
            bar_start = (t // bar_len) * bar_len
            seg_end = min(end, bar_start + bar_len)
            for d in split_duration(t - bar_start, seg_end - t, bar_len):
                pieces.append((t, d))
                t += d
        for i, (t, d) in enumerate(pieces):
            bar = t // bar_len
            if t > 0 and t % bar_len == 0:
                events.append({"kind": "bar"})
                acc.reset()
            plain = next(p for v, _, p in DURATIONS if v == d)
            ev = {"bar": bar, "pos": t - bar * bar_len, "plain": plain, "d": d, "opts": ""}
            ev["half"] = ev["pos"] // _beam_unit(bar_len, False)
            if q.midi is None:
                events.append({**ev, "kind": "rest"})
                continue
            last = i == len(pieces) - 1
            tie = (not last) or q.tie_next
            pos = acc.encode(q.midi)
            m = re.search(r"-?\d+", pos)
            events.append(
                {
                    **ev,
                    "kind": "note",
                    "pos_text": f"{pos}{'^' if tie else ''}",
                    "staff_pos": int(m.group()) if m else 0,
                }
            )
            if i == 0:
                lyric_tokens.append(q.lyric if q.lyric else "_")
    _beam_opts(events, bar_len)
    lines: list[str] = []
    for ev in events:
        if ev["kind"] == "bar":
            lines.append("|Bar")
        elif ev["kind"] == "rest":
            lines.append(f"|Rest|Dur:{dur_string(ev['d'])}")
        else:
            lines.append(f"|Note|Dur:{dur_string(ev['d'])}|Pos:{ev['pos_text']}{ev['opts']}")
    return lines, lyric_tokens


def score_to_nwctxt(score: Score) -> str:
    problems = []
    parts: dict[str, list[QNote]] = {}
    for v in score.voices:
        part = score.parts.get(v.name, [])
        if monophony_violations(part):
            problems.append(v.name)
            part = enforce_monophony(part)
        parts[v.name] = merge_ties(part)
    if problems:
        log.warning("단선율이 아닌 성부를 잘라서 썼습니다: %s", problems)

    bar_len = score.measure_beats()
    end = score_end(score)
    n_bars = max(1, -(-end // bar_len))  # 올림
    total = n_bars * bar_len

    out = [f"!NoteWorthyComposer({NWC_VERSION})"]
    out.append(f'|SongInfo|Title:"{escape_text(score.title)}"')
    for i, v in enumerate(score.voices):
        if v.clef not in CLEF_LINES:
            raise ValueError(f"{v.name}: 지원하지 않는 음자리표 {v.clef!r}")
        lines, lyric_tokens = _staff_events(parts[v.name], score, v.clef, total)
        out.append(f'|AddStaff|Name:"{escape_text(v.name)}"')
        if any(tok != "_" for tok in lyric_tokens):
            out.append("|Lyrics|Placement:Bottom|Align:Standard Rules|Offset:0")
            out.append(f'|Lyric1|Text:"{escape_text(_lyric_text(lyric_tokens))}"')
        out.append(CLEF_LINES[v.clef])
        out.append(f"|Key|Signature:{fifths_to_signature(score.key_fifths)}")
        num, den = score.time_sig
        out.append(f"|TimeSig|Signature:{num}/{den}")
        if i == 0:
            out.append(f"|Tempo|Tempo:{round(score.tempo_bpm)}")
        out.extend(lines)
    out.append("!NoteWorthyComposer-End")
    return "\n".join(out) + "\n"


def write_nwctxt(score: Score, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NWC는 CRLF와 LF를 모두 읽는다. Windows 메모장 호환을 위해 CRLF로 쓴다.
    path.write_text(score_to_nwctxt(score), encoding="utf-8", newline="\r\n")
    return path
