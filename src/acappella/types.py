"""데이터 계약.

모든 파이프라인 모듈은 이 파일의 타입만 주고받는다(docs/design.md '모듈 인터페이스와 데이터 계약').
필드를 바꾸는 것은 데이터 계약 변경이므로 먼저 알리고 승인을 받는다.

가정: ``QNote.start_beat``/``dur_beats``의 '박'은 4분음표 하나다(박자표와 무관).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass
class VoiceSpec:
    name: str  # "S1", "S2", "A", "T", "B", "Lead" 등
    clef: str  # "treble" | "treble8vb" | "bass"
    low_midi: int  # 대략 음역 하한(MIDI 번호)
    high_midi: int  # 대략 음역 상한


@dataclass
class AudioClip:
    samples: np.ndarray  # 모노 float32
    sr: int
    source: str  # URL 또는 파일 경로
    title: str = ""


@dataclass
class NoteEvent:
    onset: float  # 초
    offset: float  # 초
    midi: int
    confidence: float = 1.0


@dataclass
class PitchActivation:  # 다중 피치 추정 출력
    times: np.ndarray  # (T,) 프레임 시각, 초
    midi_bins: np.ndarray  # (F,) 각 주파수 빈의 MIDI 음높이
    salience: np.ndarray  # (T, F) 0~1
    notes: list[NoteEvent] = field(default_factory=list)


@dataclass
class VoiceTrack:
    voice: VoiceSpec
    times: np.ndarray  # (T,)
    f0_midi: np.ndarray  # (T,) 프레임별 음높이, 무음은 nan


@dataclass
class QNote:  # 양자화된 음표
    start_beat: Fraction  # 곡 시작부터의 박(4분음표) 위치
    dur_beats: Fraction
    midi: int | None  # None이면 쉼표
    tie_next: bool = False
    lyric: str | None = None


@dataclass
class Score:
    title: str
    tempo_bpm: float
    time_sig: tuple[int, int]  # (4, 4)
    key_fifths: int  # -7(플랫 7개) ~ +7(샵 7개)
    voices: list[VoiceSpec]
    parts: dict[str, list[QNote]]  # 성부 이름 → 음표 열

    def measure_beats(self) -> Fraction:
        """한 마디의 길이(4분음표 박 단위)."""
        num, den = self.time_sig
        return Fraction(num * 4, den)

    def voice(self, name: str) -> VoiceSpec:
        for v in self.voices:
            if v.name == name:
                return v
        raise KeyError(name)
