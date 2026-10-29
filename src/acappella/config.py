"""경로, 외부 실행 파일, 성부 프리셋, 임계값.

설정 우선순위: 환경 변수 > ``<home>/settings.json`` > 기본값.

- ``ACAPPELLA_HOME``: 작업 폴더(outputs/, models/, settings.json). 기본값은
  Windows ``%LOCALAPPDATA%\\Acappella``, 그 외 ``~/.acappella``.
- ``ACAPPELLA_FFMPEG``: ffmpeg 실행 파일 경로. 기본값은 설치본의 ``bin/ffmpeg(.exe)``, 없으면 PATH.
- ``ACAPPELLA_YTDLP``: yt-dlp 실행 파일 경로. 비워 두면 파이썬 모듈(``python -m yt_dlp``)을 쓴다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

from acappella.types import VoiceSpec

log = logging.getLogger(__name__)

APP_NAME = "Acappella"
RELEASE_REPO = "latuca/ssonsal-ai"


# ---------------------------------------------------------------------------
# 음이름 ↔ MIDI
# ---------------------------------------------------------------------------

_NOTE_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)(-?\d)\s*$")
_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PC_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def note_to_midi(name: str | int) -> int:
    """'C4' → 60, 'Bb2' → 46. 정수나 숫자 문자열은 그대로 MIDI 번호로 본다."""
    if isinstance(name, int):
        return name
    s = str(name).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"음이름을 읽을 수 없습니다: {name!r} (예: C4, F#3, Bb2)")
    step, acc, octave = m.groups()
    pc = _STEP_PC[step.upper()] + {"": 0, "#": 1, "b": -1}[acc]
    return 12 * (int(octave) + 1) + pc


def midi_to_note(midi: int) -> str:
    return f"{_PC_NAME[midi % 12]}{midi // 12 - 1}"


# ---------------------------------------------------------------------------
# 성부 프리셋
# ---------------------------------------------------------------------------

CLEFS = ("treble", "treble8vb", "bass")

# 성부 종류별 기본 음자리표와 대략 음역.
VOICE_DEFAULTS: dict[str, tuple[str, int, int]] = {
    "Lead": ("treble", note_to_midi("G3"), note_to_midi("C6")),
    "S": ("treble", note_to_midi("C4"), note_to_midi("A5")),
    "S1": ("treble", note_to_midi("D4"), note_to_midi("B5")),
    "S2": ("treble", note_to_midi("A3"), note_to_midi("F5")),
    "A": ("treble", note_to_midi("F3"), note_to_midi("D5")),
    "A1": ("treble", note_to_midi("G3"), note_to_midi("D5")),
    "A2": ("treble", note_to_midi("E3"), note_to_midi("C5")),
    "T": ("treble8vb", note_to_midi("C3"), note_to_midi("A4")),
    "T1": ("treble8vb", note_to_midi("D3"), note_to_midi("A4")),
    "T2": ("treble8vb", note_to_midi("A2"), note_to_midi("F4")),
    "Bar": ("bass", note_to_midi("A2"), note_to_midi("F4")),
    "B": ("bass", note_to_midi("E2"), note_to_midi("D4")),
}

PRESETS: dict[str, list[str]] = {
    "SATB": ["S", "A", "T", "B"],
    "SSATB": ["S1", "S2", "A", "T", "B"],
    "Lead+SATB": ["Lead", "S", "A", "T", "B"],
    "Lead+SSATTB": ["Lead", "S1", "S2", "A", "T1", "T2", "B"],
}


def default_voice(name: str) -> VoiceSpec:
    """이름으로 기본 VoiceSpec을 만든다. 모르는 이름은 접두어(S/A/T/B/Lead)로 추정한다."""
    key = name
    if key not in VOICE_DEFAULTS:
        for prefix in ("Lead", "Bar", "S", "A", "T", "B"):
            if name.upper().startswith(prefix.upper()):
                key = prefix
                break
        else:
            key = "A"
            log.warning("알 수 없는 성부 이름 %r: 알토 기본값을 씁니다.", name)
    clef, low, high = VOICE_DEFAULTS[key]
    return VoiceSpec(name=name, clef=clef, low_midi=low, high_midi=high)


def parse_voices(spec: str) -> list[VoiceSpec]:
    """'Lead,S,A,T,B' 또는 프리셋 이름 'SATB'를 VoiceSpec 목록으로 바꾼다.

    성부마다 ``이름:음자리표:하한:상한`` 형식으로 기본값을 덮어쓸 수 있다. 예: ``T:bass:C3:G4``.
    """
    spec = spec.strip()
    if spec in PRESETS:
        return [default_voice(n) for n in PRESETS[spec]]
    voices: list[VoiceSpec] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        parts = item.split(":")
        v = default_voice(parts[0])
        if len(parts) >= 2 and parts[1]:
            if parts[1] not in CLEFS:
                raise ValueError(f"음자리표는 {CLEFS} 중 하나여야 합니다: {parts[1]!r}")
            v.clef = parts[1]
        if len(parts) >= 4:
            v.low_midi, v.high_midi = note_to_midi(parts[2]), note_to_midi(parts[3])
        voices.append(v)
    validate_voices(voices)
    return voices


def validate_voices(voices: list[VoiceSpec]) -> None:
    if not voices:
        raise ValueError("성부를 하나 이상 지정해 주세요.")
    if len(voices) > 8:
        raise ValueError("성부는 최대 8개까지 지원합니다.")
    names = [v.name for v in voices]
    if len(set(names)) != len(names):
        raise ValueError(f"성부 이름이 겹칩니다: {names}")
    for v in voices:
        if v.clef not in CLEFS:
            raise ValueError(f"{v.name}: 음자리표는 {CLEFS} 중 하나여야 합니다.")
        if v.low_midi >= v.high_midi:
            raise ValueError(f"{v.name}: 음역 하한이 상한보다 낮아야 합니다.")


# ---------------------------------------------------------------------------
# 임계값
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    # preprocess
    target_rms_db: float = -20.0
    peak_limit: float = 0.99
    # voices (규칙 기반 성부 배정)
    voice_frame_hop: int = 2  # Basic Pitch 프레임(약 11.6ms)을 몇 개씩 묶을지
    peak_abs_threshold: float = 0.25  # 이보다 약한 피치 후보는 버린다
    peak_rel_threshold: float = 0.35  # 프레임 최댓값 대비 비율
    range_margin: int = 3  # 음역 밖 이 반음까지는 벌점 없이 허용
    range_weight: float = 0.6  # 음역 밖 반음당 벌점
    unison_cost: float = 0.6  # 두 성부가 같은 후보를 나눠 가질 때 벌점(유니즌 채우기)
    drop_cost: float = 1.5  # 후보를 아무 성부에도 배정하지 않을 때 (salience 배수) 벌점
    silence_cost: float = 2.0  # 후보가 있는데 성부를 무음으로 둘 때 벌점
    target_weight: float = 1.0  # Viterbi: 프레임 목표 음높이에서 반음당 벌점
    jump_cost: float = 1.5  # Viterbi: 음높이가 바뀔 때 고정 벌점
    step_cost: float = 0.15  # Viterbi: 바뀐 반음당 벌점
    voicing_cost: float = 3.0  # Viterbi: 무음 ↔ 유성 전환 벌점
    # notes
    median_window: int = 5  # 프레임
    min_note_sec: float = 0.08
    onset_split_min_sec: float = 0.12
    # rhythm
    grid_division: int = 4  # 4분음표 하나를 몇 칸으로 나눌지(4 = 16분음표 격자)
    # postprocess
    spurious_max_beats: float = 0.25
    spurious_jump: int = 5
    unison_fill_max_beats: float = 1.0
    range_warn_margin: int = 3


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------


def _default_home() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    return Path.home() / ".acappella"


def app_dir() -> Path:
    """설치본 루트(임베디드 파이썬 옆). 개발 환경에서는 레포 루트."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    home: Path = field(default_factory=_default_home)
    ffmpeg_path: str = ""
    ytdlp_path: str = ""
    keep_audio: bool = False
    auto_update_ytdlp: bool = True
    check_app_update: bool = True
    release_repo: str = RELEASE_REPO
    voice_model: str = "voice_assign.onnx"  # models/ 아래 파일명(v2)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def voice_model_path(self) -> Path:
        return self.models_dir / self.voice_model

    def ffmpeg(self) -> str:
        if self.ffmpeg_path:
            return self.ffmpeg_path
        exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        bundled = app_dir() / "bin" / exe
        if bundled.exists():
            return str(bundled)
        return shutil.which("ffmpeg") or "ffmpeg"

    def ytdlp_cmd(self) -> list[str]:
        if self.ytdlp_path:
            return [self.ytdlp_path]
        return [sys.executable, "-m", "yt_dlp"]


def load_settings(home: Path | None = None) -> Settings:
    s = Settings()
    if env_home := os.environ.get("ACAPPELLA_HOME"):
        s.home = Path(env_home)
    if home is not None:
        s.home = Path(home)
    path = s.home / "settings.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("settings.json을 읽지 못했습니다(%s). 기본값을 씁니다.", e)
            data = {}
        th = data.pop("thresholds", {}) or {}
        for f in fields(Settings):
            if f.name in data and f.name not in ("home", "thresholds"):
                setattr(s, f.name, data[f.name])
        for f in fields(Thresholds):
            if f.name in th:
                setattr(s.thresholds, f.name, type(getattr(s.thresholds, f.name))(th[f.name]))
    s.ffmpeg_path = os.environ.get("ACAPPELLA_FFMPEG", s.ffmpeg_path)
    s.ytdlp_path = os.environ.get("ACAPPELLA_YTDLP", s.ytdlp_path)
    return s
