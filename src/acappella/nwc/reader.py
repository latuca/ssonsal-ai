"""nwctxt → Score (데이터 구축과 평가용).

지원: SongInfo 제목, AddStaff, Clef(OctaveShift 포함), Key, TimeSig, Tempo, Bar, Note, Rest,
Chord/RestChord(가장 높은 음만 남김), 타이, 점음표, 셋잇단음표, 꾸밈음(무시), Lyric1.
반복 기호와 도돌이표는 펼치지 않는다. 곡 중간의 조표·박자표 변경은 음높이·박 위치에는
반영하지만 Score에는 첫 값만 남는다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from acappella.nwc.pitch_pos import BarAccidentals, signature_to_fifths
from acappella.types import QNote, Score, VoiceSpec

log = logging.getLogger(__name__)

_BASE_DUR = {
    "Whole": Fraction(4),
    "Half": Fraction(2),
    "4th": Fraction(1),
    "8th": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
}
_POS_RE = re.compile(r"^([#bnxv]?)(-?\d+)([a-zA-Z]?)(\^?)")


def split_fields(line: str) -> list[str]:
    """``|A|B:"x|y"|C`` → ['A', 'B:"x|y"', 'C']. 따옴표 안의 |와 \\| 이스케이프를 지킨다."""
    out, cur, in_q, i = [], [], False, 0
    body = line[1:] if line.startswith("|") else line
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            cur.append(body[i : i + 2])
            i += 2
            continue
        if ch == '"':
            in_q = not in_q
        if ch == "|" and not in_q:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def unescape_text(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t"}.get(nxt, nxt))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_line(line: str) -> tuple[str, dict[str, str]]:
    parts = split_fields(line)
    kind = parts[0]
    props: dict[str, str] = {}
    for p in parts[1:]:
        if ":" in p:
            k, v = p.split(":", 1)
            props[k] = v
        elif p:
            props[p] = ""
    return kind, props


def parse_dur(text: str) -> Fraction:
    """``4th,Dotted`` → 3/2. 꾸밈음은 0."""
    items = [x.strip() for x in text.split(",")]
    base = None
    mult = Fraction(1)
    for it in items:
        if it in _BASE_DUR:
            base = _BASE_DUR[it]
        elif it == "Dotted":
            mult *= Fraction(3, 2)
        elif it == "DblDotted":
            mult *= Fraction(7, 4)
        elif it.startswith("Triplet"):
            mult *= Fraction(2, 3)
        elif it == "Grace":
            return Fraction(0)
    if base is None:
        raise ValueError(f"음가를 읽을 수 없습니다: {text!r}")
    return base * mult


def parse_clef(props: dict[str, str]) -> str:
    base = props.get("Type", "Treble").lower()
    if base == "percussion":
        base = "treble"
    shift = props.get("OctaveShift", "")
    if shift == "Octave Down":
        return base + "8vb"
    if shift == "Octave Up":
        return base + "8va"
    return base


def parse_timesig(sig: str) -> tuple[int, int]:
    if sig == "Common":
        return (4, 4)
    if sig == "AllaBreve":
        return (2, 2)
    num, den = sig.split("/")
    return (int(num), int(den))


def split_lyric_syllables(text: str) -> list[str | None]:
    """Lyric 문자열 → 음절 목록. ``_``는 건너뛰는 자리(None). 단어 안 음절은 '-'로 끝난다."""
    out: list[str | None] = []
    for tok in text.split():
        if tok == "_":
            out.append(None)
            continue
        for piece in re.findall(r"[^-]+-?|-", tok):
            out.append(None if piece == "_" else piece)
    return out


@dataclass
class _Staff:
    name: str
    clef: str = "treble"
    fifths: int = 0
    acc: BarAccidentals | None = None
    t: Fraction = Fraction(0)
    notes: list[QNote] = field(default_factory=list)
    lyric: str | None = None
    bar_start: Fraction = Fraction(0)
    measures: list[Fraction] = field(default_factory=list)


def _note_midi(staff: _Staff, pos_field: str) -> tuple[int, bool]:
    """``Pos`` 값(화음이면 여러 개) → (가장 높은 MIDI, 타이 여부)."""
    assert staff.acc is not None
    best: tuple[int, bool] | None = None
    for tok in pos_field.split(","):
        m = _POS_RE.match(tok.strip())
        if not m:
            raise ValueError(f"Pos를 읽을 수 없습니다: {tok!r}")
        acc, pos, _head, tie = m.groups()
        midi = staff.acc.decode(int(pos), acc)
        if best is None or midi > best[0]:
            best = (midi, bool(tie))
    assert best is not None
    return best


@dataclass
class ParsedNwc:
    score: Score
    measures: dict[str, list[Fraction]]  # 성부별 마디 길이(박)


def parse_nwctxt(text: str) -> ParsedNwc:
    title = ""
    tempo: float | None = None
    time_sig: tuple[int, int] | None = None
    key: int | None = None
    staves: list[_Staff] = []
    cur: _Staff | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            continue
        if not line.startswith("|"):
            continue
        kind, props = parse_line(line)
        if kind == "SongInfo":
            title = unescape_text(props.get("Title", ""))
        elif kind == "AddStaff":
            name = unescape_text(props.get("Name", "")) or f"Staff{len(staves) + 1}"
            used = {s.name for s in staves}
            base, n = name, 2
            while name in used:
                name, n = f"{base}_{n}", n + 1
            cur = _Staff(name=name)
            cur.acc = BarAccidentals(cur.fifths, cur.clef)
            staves.append(cur)
        elif cur is None:
            continue
        elif kind == "Clef":
            cur.clef = parse_clef(props)
            cur.acc = BarAccidentals(cur.fifths, cur.clef)
        elif kind == "Key":
            cur.fifths = signature_to_fifths(props.get("Signature", "C"))
            cur.acc = BarAccidentals(cur.fifths, cur.clef)
            if key is None:
                key = cur.fifths
        elif kind == "TimeSig":
            ts = parse_timesig(props.get("Signature", "4/4"))
            if time_sig is None:
                time_sig = ts
        elif kind == "Tempo":
            if tempo is None and props.get("Tempo"):
                tempo = float(props["Tempo"])
        elif kind == "Lyric1":
            cur.lyric = unescape_text(props.get("Text", ""))
        elif kind == "Bar":
            assert cur.acc is not None
            cur.acc.reset()
            if cur.t > cur.bar_start:
                cur.measures.append(cur.t - cur.bar_start)
            cur.bar_start = cur.t
        elif kind in ("Note", "Chord", "Rest", "RestChord"):
            d = parse_dur(props.get("Dur", "4th"))
            if d == 0:
                continue  # 꾸밈음
            if kind == "Rest":
                cur.notes.append(QNote(cur.t, d, None))
            else:
                if kind == "RestChord":
                    # 쉼표와 음이 함께 있는 화음: 음 부분만 쓴다
                    d = parse_dur(props.get("Dur2", props.get("Dur", "4th")))
                midi, tie = _note_midi(cur, props.get("Pos", "0"))
                if kind == "Chord" and "," in props.get("Pos", ""):
                    log.debug("%s: %s박의 화음에서 가장 높은 음만 남깁니다.", cur.name, cur.t)
                cur.notes.append(QNote(cur.t, d, midi, tie_next=tie))
            cur.t += d

    voices: list[VoiceSpec] = []
    parts: dict[str, list[QNote]] = {}
    measures: dict[str, list[Fraction]] = {}
    for s in staves:
        if s.t > s.bar_start:
            s.measures.append(s.t - s.bar_start)
        pitched = [q.midi for q in s.notes if q.midi is not None]
        low, high = (min(pitched), max(pitched)) if pitched else (48, 72)
        voices.append(VoiceSpec(name=s.name, clef=s.clef, low_midi=low, high_midi=high))
        if s.lyric:
            _attach_lyrics(s.notes, s.lyric)
        parts[s.name] = s.notes
        measures[s.name] = s.measures

    score = Score(
        title=title,
        tempo_bpm=tempo or 120.0,
        time_sig=time_sig or (4, 4),
        key_fifths=key or 0,
        voices=voices,
        parts=parts,
    )
    return ParsedNwc(score=score, measures=measures)


def _attach_lyrics(part: list[QNote], text: str) -> None:
    syllables = split_lyric_syllables(text)
    slots = []
    prev_tie = False
    for q in part:
        if q.midi is None:
            prev_tie = False
            continue
        if not prev_tie:
            slots.append(q)
        prev_tie = q.tie_next
    for q, syl in zip(slots, syllables):
        q.lyric = syl


def read_nwctxt(path: Path) -> Score:
    return parse_nwctxt(read_text(path)).score


def read_text(path: Path) -> str:
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # 오래된 한국어 Windows에서 저장한 파일
        return data.decode("cp949", errors="replace")


def measure_lengths(path_or_text: Path | str) -> dict[str, list[Fraction]]:
    text = read_text(path_or_text) if isinstance(path_or_text, Path) else path_or_text
    return parse_nwctxt(text).measures
