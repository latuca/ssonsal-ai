"""곡 메타데이터(데이터 구축·학습·평가용). docs/design.md '데이터 파이프라인'의 스키마를 따른다.

필드를 바꾸는 것은 메타데이터 스키마 변경이므로 먼저 알리고 승인을 받는다.
``genre``와 ``audio_path``는 스키마 밖의 선택 필드로, 있으면 평가 리포트와 음원 찾기에 쓴다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

TIERS = ("A", "B", "C")
SPLITS = ("train", "valid", "test", "")


@dataclass
class SongMeta:
    song_id: str
    title: str = ""
    nwc_path: str = ""
    youtube_url: str = ""
    tier: str = "A"  # A: 합성, B: 자동 매칭(silver), C: 사람이 확인(gold)
    voices: list[str] = field(default_factory=list)
    transpose_semitones: int = 0  # 음원이 악보보다 높으면 양수
    align_cost: float | None = None
    is_medley: bool = False
    lyrics_lang: str = ""
    group: str = ""  # 원곡 또는 커버 채널
    split: str = ""  # train | valid | test
    extra: dict = field(default_factory=dict)  # genre, audio_path 등 선택 필드

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.song_id}: tier는 {TIERS} 중 하나여야 합니다: {self.tier!r}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.song_id}: split은 {SPLITS} 중 하나여야 합니다: {self.split!r}")

    @property
    def genre(self) -> str:
        return str(self.extra.get("genre", ""))

    def to_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SongMeta:
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["extra"] = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs)


def load_meta(path: Path) -> SongMeta:
    return SongMeta.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_meta(meta: SongMeta, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_all(meta_dir: Path) -> list[SongMeta]:
    return [load_meta(p) for p in sorted(Path(meta_dir).glob("*.json"))]
