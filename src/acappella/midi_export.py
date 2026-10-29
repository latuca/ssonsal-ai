"""검토용 성부별 MIDI(SMF type 1) 쓰기. 외부 의존성 없이 직접 쓴다."""

from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path

from acappella.score_utils import merge_ties
from acappella.types import QNote, Score

TPQ = 480  # 4분음표당 틱


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _track(events: list[tuple[int, bytes]]) -> bytes:
    events.sort(key=lambda e: (e[0], e[1][0] & 0xF0 == 0x90))  # 같은 틱이면 note-off 먼저
    data, last = bytearray(), 0
    for tick, msg in events:
        data += _vlq(tick - last) + msg
        last = tick
    data += _vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


def _ticks(beats: Fraction) -> int:
    return int(round(beats * TPQ))


def _part_events(part: list[QNote], channel: int, name: str) -> list[tuple[int, bytes]]:
    nm = name.encode("utf-8")
    ev: list[tuple[int, bytes]] = [(0, b"\xff\x03" + _vlq(len(nm)) + nm)]
    ev.append((0, bytes([0xC0 | channel, 52])))  # Choir Aahs
    for q in merge_ties(part):
        on, off = _ticks(q.start_beat), _ticks(q.start_beat + q.dur_beats)
        ev.append((on, bytes([0x90 | channel, q.midi, 90])))
        ev.append((off, bytes([0x80 | channel, q.midi, 0])))
    return ev


def score_to_midi(score: Score) -> bytes:
    tempo_us = int(round(60_000_000 / score.tempo_bpm))
    num, den = score.time_sig
    conductor = [
        (0, b"\xff\x51\x03" + tempo_us.to_bytes(3, "big")),
        (0, b"\xff\x58\x04" + bytes([num, den.bit_length() - 1, 24, 8])),
    ]
    tracks = [_track(conductor)]
    for i, v in enumerate(score.voices):
        ch = i if i < 9 else i + 1  # 채널 10(드럼) 건너뜀
        tracks.append(_track(_part_events(score.parts.get(v.name, []), ch, v.name)))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), TPQ)
    return header + b"".join(tracks)


def write_midi(score: Score, path: Path, voice: str | None = None) -> Path:
    """voice를 주면 그 성부만 쓴다."""
    if voice is not None:
        score = Score(
            score.title,
            score.tempo_bpm,
            score.time_sig,
            score.key_fifths,
            [score.voice(voice)],
            {voice: score.parts.get(voice, [])},
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(score_to_midi(score))
    return path
