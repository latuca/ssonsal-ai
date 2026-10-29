"""NWC 바이너리(.nwc) 파일에서 버전과 내장 nwctxt를 꺼낸다.

- 압축 파일은 ``[NWZ]`` 머리 뒤에 zlib 데이터가 온다.
- 풀린 데이터는 ``[NoteWorthy ArtWare]``, ``[NoteWorthy Composer]`` 머리글로 시작한다.
- NWC 2.75 이상에서 저장한 파일에는 nwctxt 텍스트가 들어 있다고 알려져 있다(NWC 포럼).
  이 모듈은 그 텍스트(``!NoteWorthyComposer`` ~ ``!NoteWorthyComposer-End``)를 찾기만 하고,
  찾지 못하면 실패로 돌려준다. 바이너리 형식을 추측해서 해석하지 않는다.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

_TEXT_RE = re.compile(
    rb"!NoteWorthyComposer(?:Clip)?\([^)]*\).*?!NoteWorthyComposer(?:Clip)?-End", re.S
)
_VERSION_TXT_RE = re.compile(rb"!NoteWorthyComposer\(([\d.]+)\)")
_HEADER = b"[NoteWorthy Composer]"


@dataclass
class NwcContent:
    version: str  # "2.75", "1.75" 등. 알 수 없으면 ""
    text: str | None  # 내장 nwctxt(없으면 None)
    error: str = ""


def decompress(data: bytes) -> bytes:
    if data.startswith(b"[NWZ]"):
        for off in (6, 5, 7, 8):
            try:
                return zlib.decompress(data[off:])
            except zlib.error:
                continue
        raise ValueError("[NWZ] 압축을 풀 수 없습니다.")
    return data


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_nwc(path: Path) -> NwcContent:
    data = Path(path).read_bytes()
    if data.lstrip(b"\xef\xbb\xbf").startswith(b"!NoteWorthyComposer"):
        m = _VERSION_TXT_RE.search(data)
        return NwcContent(m.group(1).decode() if m else "", _decode(data.lstrip(b"\xef\xbb\xbf")))
    try:
        body = decompress(data)
    except ValueError as e:
        return NwcContent("", None, str(e))
    version = ""
    if (m := _VERSION_TXT_RE.search(body)) is not None:
        version = m.group(1).decode()
    elif (i := body.find(_HEADER)) >= 0:
        # 머리글 뒤 두 바이트(부, 주 버전)로 추정한다. 실제 파일로 확인 전까지는 참고용.
        j = i + len(_HEADER) + 1
        if j + 2 <= len(body) and body[j + 1] in (1, 2) and body[j] < 100:
            version = f"{body[j + 1]}.{body[j]:02d}"
    m = _TEXT_RE.search(body)
    if not m:
        return NwcContent(version, None, "내장 nwctxt가 없습니다(2.75 미만 형식일 수 있음)")
    return NwcContent(version, _decode(m.group(0)))
