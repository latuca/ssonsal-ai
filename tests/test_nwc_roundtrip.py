"""왕복 테스트: nwctxt를 reader로 읽고 writer로 다시 쓴 뒤, 다시 읽은 결과가 원본과 같은지 본다.

실제 동아리 파일은 tests/fixtures/local/에 넣는다(gitignore, 외부 공유 금지).
"""

from pathlib import Path

import pytest

from acappella.nwc.reader import parse_nwctxt, read_text
from acappella.nwc.writer import score_to_nwctxt
from acappella.score_utils import merge_ties

FIXTURES = Path(__file__).parent / "fixtures"
LOCAL = sorted((FIXTURES / "local").glob("*.nwctxt")) if (FIXTURES / "local").exists() else []


def _canon(score):
    return {
        name: [(q.start_beat, q.dur_beats, q.midi, q.lyric) for q in merge_ties(part)]
        for name, part in score.parts.items()
    }


def _roundtrip(text: str):
    first = parse_nwctxt(text).score
    second = parse_nwctxt(score_to_nwctxt(first))
    assert _canon(second.score) == _canon(first)
    assert [v.name for v in second.score.voices] == [v.name for v in first.voices]
    assert second.score.time_sig == first.time_sig
    assert second.score.key_fifths == first.key_fifths
    measure = second.score.measure_beats()
    for name, lengths in second.measures.items():
        assert all(m == measure for m in lengths), name


def test_roundtrip_handmade():
    _roundtrip((FIXTURES / "handmade_satb.nwctxt").read_text(encoding="utf-8"))


@pytest.mark.skipif(not LOCAL, reason="tests/fixtures/local/에 실제 NWC 내보내기 파일이 없음")
@pytest.mark.parametrize("path", LOCAL, ids=lambda p: p.name)
def test_roundtrip_real_exports(path):
    _roundtrip(read_text(path))
