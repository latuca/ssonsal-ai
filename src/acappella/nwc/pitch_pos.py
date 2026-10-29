"""음높이(MIDI) ↔ NWC 스태프 위치(Pos), 조표, 임시표.

- Pos 0은 오선의 가운데 줄이고, 한 칸(줄↔칸) 올라갈 때마다 1씩 커진다.
- 가운데 줄 음: 높은음자리표 B4, 옥타브 아래 높은음자리표 B3, 낮은음자리표 D3,
  가온음자리표(Alto) C4, 테너음자리표 A3.
- 임시표 문자: ``#`` 샵, ``b`` 플랫, ``n`` 제자리, ``x`` 겹샵, ``v`` 겹플랫.
- 임시표는 같은 마디 안의 같은 Pos에 계속 적용된다(표준 기보 규칙). writer와 reader는
  :class:`BarAccidentals` 하나로 같은 규칙을 공유한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEPS = "CDEFGAB"
_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

ACC_TO_ALTER = {"#": 1, "b": -1, "n": 0, "x": 2, "v": -2}
ALTER_TO_ACC = {1: "#", -1: "b", 0: "n", 2: "x", -2: "v"}

# (clef, octave_shift) → 가운데 줄의 (음이름, 옥타브)
_CLEF_MIDDLE = {
    "treble": ("B", 4),
    "treble8vb": ("B", 3),
    "treble8va": ("B", 5),
    "bass": ("D", 3),
    "bass8vb": ("D", 2),
    "bass8va": ("D", 4),
    "alto": ("C", 4),
    "tenor": ("A", 3),
}

# 다장조에서 조표 밖 음의 관용 표기
_C_MAJOR_CHROMATIC = {1: ("C", 1), 3: ("E", -1), 6: ("F", 1), 8: ("G", 1), 10: ("B", -1)}


def _diatonic_index(step: str, octave: int) -> int:
    return octave * 7 + STEPS.index(step)


def clef_middle_index(clef: str) -> int:
    try:
        step, octave = _CLEF_MIDDLE[clef]
    except KeyError:
        raise ValueError(f"지원하지 않는 음자리표: {clef!r}") from None
    return _diatonic_index(step, octave)


def key_alterations(fifths: int) -> dict[str, int]:
    """조표의 음이름별 변화량. 예: 2 → {'F': 1, 'C': 1, 나머지 0}."""
    if not -7 <= fifths <= 7:
        raise ValueError(f"조표 범위(-7~7)를 벗어났습니다: {fifths}")
    alters = {s: 0 for s in STEPS}
    if fifths > 0:
        for s in SHARP_ORDER[:fifths]:
            alters[s] = 1
    else:
        for s in FLAT_ORDER[:-fifths]:
            alters[s] = -1
    return alters


def fifths_to_signature(fifths: int) -> str:
    """NWC Key 문법: 다장조는 ``C``, 그 외는 ``F#,C#`` / ``Bb,Eb`` 처럼 바뀐 음 목록."""
    if fifths == 0:
        return "C"
    if fifths > 0:
        return ",".join(f"{s}#" for s in SHARP_ORDER[:fifths])
    return ",".join(f"{s}b" for s in FLAT_ORDER[:-fifths])


def signature_to_fifths(sig: str) -> int:
    items = [x.strip() for x in sig.split(",") if x.strip()]
    sharps = sum(1 for x in items if x.endswith("#"))
    flats = sum(1 for x in items if len(x) == 2 and x.endswith("b"))
    if sharps and flats:
        raise ValueError(f"샵과 플랫이 섞인 조표는 지원하지 않습니다: {sig!r}")
    return sharps or -flats


_MAJOR_TONIC = {
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#",
    -1: "F",
    -2: "Bb",
    -3: "Eb",
    -4: "Ab",
    -5: "Db",
    -6: "Gb",
    -7: "Cb",
}


def fifths_to_tonic(fifths: int) -> str:
    return _MAJOR_TONIC[fifths]


def spell(midi: int, fifths: int) -> tuple[str, int, int]:
    """MIDI 음을 조성에 맞는 (음이름, 변화량, 옥타브)로 표기한다."""
    pc = midi % 12
    alters = key_alterations(fifths)
    cands: list[tuple[int, str, int]] = []
    for step in STEPS:
        for alter in (-2, -1, 0, 1, 2):
            if (_NATURAL_PC[step] + alter) % 12 != pc:
                continue
            if alter == alters[step]:
                rank = 0  # 조표 안의 음
            elif fifths == 0 and _C_MAJOR_CHROMATIC.get(pc) == (step, alter):
                rank = 1
            elif alter == 0:
                rank = 2
            elif alter == (1 if fifths >= 0 else -1):
                rank = 3
            elif abs(alter) == 1:
                rank = 4
            else:
                rank = 5
            cands.append((rank, step, alter))
    _, step, alter = min(cands)
    # 옥타브: 표기된 음이름의 자연음 높이에서 역산(B#3 = C4, Cb4 = B3 처리)
    octave = (midi - alter - _NATURAL_PC[step]) // 12 - 1
    return step, alter, octave


def pos_of(step: str, octave: int, clef: str) -> int:
    return _diatonic_index(step, octave) - clef_middle_index(clef)


def step_octave_of(pos: int, clef: str) -> tuple[str, int]:
    idx = pos + clef_middle_index(clef)
    return STEPS[idx % 7], idx // 7


def midi_of(step: str, alter: int, octave: int) -> int:
    return 12 * (octave + 1) + _NATURAL_PC[step] + alter


@dataclass
class BarAccidentals:
    """한 마디 안에서 Pos별로 현재 적용 중인 변화량을 추적한다."""

    fifths: int
    clef: str
    _state: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._state.clear()

    def current(self, pos: int) -> int:
        if pos in self._state:
            return self._state[pos]
        step, _ = step_octave_of(pos, self.clef)
        return key_alterations(self.fifths)[step]

    def encode(self, midi: int) -> str:
        """writer용: MIDI → ``[임시표]Pos`` 문자열. 필요할 때만 임시표를 붙인다."""
        step, alter, octave = spell(midi, self.fifths)
        pos = pos_of(step, octave, self.clef)
        acc = ""
        if self.current(pos) != alter:
            acc = ALTER_TO_ACC[alter]
            self._state[pos] = alter
        return f"{acc}{pos}"

    def decode(self, pos: int, acc: str) -> int:
        """reader용: Pos와 (있으면) 임시표 → MIDI."""
        if acc:
            self._state[pos] = ACC_TO_ALTER[acc]
        step, octave = step_octave_of(pos, self.clef)
        return midi_of(step, self.current(pos), octave)
