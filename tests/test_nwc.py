from fractions import Fraction as F
from pathlib import Path

import pytest

from acappella.nwc.pitch_pos import (
    BarAccidentals,
    fifths_to_signature,
    pos_of,
    signature_to_fifths,
    spell,
    step_octave_of,
)
from acappella.nwc.reader import parse_dur, parse_nwctxt, read_nwctxt, split_lyric_syllables
from acappella.nwc.writer import score_to_nwctxt, split_duration, write_nwctxt
from acappella.score_utils import merge_ties
from acappella.types import QNote, Score, VoiceSpec

FIXTURES = Path(__file__).parent / "fixtures"


# --- pitch_pos ---------------------------------------------------------------


@pytest.mark.parametrize(
    "clef,midi,pos",
    [
        ("treble", 71, 0),
        ("treble", 60, -6),
        ("treble", 76, 3),
        ("treble8vb", 59, 0),
        ("treble8vb", 48, -6),
        ("bass", 50, 0),
        ("bass", 43, -4),
        ("bass", 57, 4),
    ],
)
def test_pos_for_clefs(clef, midi, pos):
    step, _alter, octave = spell(midi, 0)
    assert pos_of(step, octave, clef) == pos
    assert step_octave_of(pos, clef) == (step, octave)


def test_spelling_follows_key():
    assert spell(66, 1)[:2] == ("F", 1)  # G장조 F#
    assert spell(65, 1)[:2] == ("F", 0)  # G장조 F 제자리
    assert spell(70, -1)[:2] == ("B", -1)  # F장조 Bb
    assert spell(70, 0)[:2] == ("B", -1)  # 다장조 Bb(관용)
    assert spell(61, 0)[:2] == ("C", 1)  # 다장조 C#
    assert spell(63, -3)[:2] == ("E", -1)  # Eb장조 Eb
    assert spell(65, 6)[:2] == ("E", 1)  # F#장조 E#
    step, alter, octave = spell(60, 7)  # C#장조에서 B#3
    assert (step, alter, octave) == ("B", 1, 3)


def test_key_signature_strings():
    assert fifths_to_signature(0) == "C"
    assert fifths_to_signature(2) == "F#,C#"
    assert fifths_to_signature(-3) == "Bb,Eb,Ab"
    for k in range(-7, 8):
        assert signature_to_fifths(fifths_to_signature(k)) == k


def test_bar_accidentals_encode_decode_agree():
    for fifths in (-4, 0, 3):
        enc = BarAccidentals(fifths, "treble")
        dec = BarAccidentals(fifths, "treble")
        for midi in [60, 61, 61, 62, 63, 61, 60, 70, 71, 70, 66, 65, 66]:
            token = enc.encode(midi)
            acc = token[0] if token[0] in "#bnxv" else ""
            assert dec.decode(int(token[len(acc) :]), acc) == midi


def test_accidental_persists_within_bar_only():
    acc = BarAccidentals(0, "treble")
    assert acc.encode(61) == "#-6"
    assert acc.encode(61) == "-6"  # 같은 마디: 다시 붙이지 않는다
    assert acc.encode(60) == "n-6"
    acc.reset()
    assert acc.encode(60) == "-6"


# --- reader ------------------------------------------------------------------


def test_parse_dur():
    assert parse_dur("4th") == 1
    assert parse_dur("4th,Dotted") == F(3, 2)
    assert parse_dur("8th,Triplet=First") == F(1, 3)
    assert parse_dur("Half,DblDotted") == F(7, 2)
    assert parse_dur("8th,Grace,Slur") == 0


def test_lyric_syllables():
    assert split_lyric_syllables("도 레 미- _ 솔 sing-ing") == [
        "도",
        "레",
        "미-",
        None,
        "솔",
        "sing-",
        "ing",
    ]


def test_read_handmade_fixture():
    parsed = parse_nwctxt((FIXTURES / "handmade_satb.nwctxt").read_text(encoding="utf-8"))
    score = parsed.score
    assert score.title == "작은 연습곡"
    assert score.tempo_bpm == 92
    assert score.time_sig == (4, 4)
    assert score.key_fifths == 1
    assert [v.name for v in score.voices] == ["S", "A", "T", "B"]
    assert [v.clef for v in score.voices] == ["treble", "treble", "treble8vb", "bass"]

    s = merge_ties(score.parts["S"])
    assert [q.midi for q in s] == [67, 71, 72, 76, 76, 74, 78, 78]
    assert s[2].dur_beats == F(5, 2)  # 8분음표 + 타이 + 2분음표
    assert [q.lyric for q in s] == ["도", "레", "미-", None, "솔", "sing-", "ing", None]

    a = merge_ties(score.parts["A"])
    assert [q.midi for q in a] == [64, 66, 67, 68, 68, 67]  # 마디 안에서 임시표 유지
    t = merge_ties(score.parts["T"])
    assert [(q.midi, q.dur_beats) for q in t] == [(57, 8), (59, 4)]  # 화음은 가장 높은 음
    b = merge_ties(score.parts["B"])
    assert [q.midi for q in b] == [45, 47, 48, 50, 52, 45]

    for name, lengths in parsed.measures.items():
        assert lengths == [4, 4, 4], name


# --- writer ------------------------------------------------------------------


def _score(parts, voices=None, time_sig=(4, 4), key=0):
    voices = voices or [VoiceSpec("S", "treble", 60, 81)]
    return Score("t", 100, time_sig, key, voices, parts)


def test_split_duration_respects_beats():
    assert split_duration(F(0), F(4), F(4)) == [4]
    assert split_duration(F(0), F(3), F(3)) == [3]
    assert split_duration(F(1, 2), F(3, 2), F(4)) == [F(1, 2), 1]
    assert split_duration(F(1, 4), F(1, 2), F(4)) == [F(1, 4), F(1, 4)]
    assert sum(split_duration(F(1), F(5, 2), F(4))) == F(5, 2)


def test_writer_splits_across_bars_with_ties_and_fills_rests():
    part = [QNote(F(3), F(2), 72), QNote(F(6), F(1, 2), 74)]
    text = score_to_nwctxt(_score({"S": part}))
    lines = text.splitlines()
    assert lines[0] == "!NoteWorthyComposer(2.75)"
    assert lines[-1] == "!NoteWorthyComposer-End"
    assert "|Note|Dur:4th|Pos:1^" in lines
    assert lines.count("|Bar") == 1
    parsed = parse_nwctxt(text)
    assert parsed.measures["S"] == [4, 4]
    notes = merge_ties(parsed.score.parts["S"])
    assert [(q.start_beat, q.dur_beats, q.midi) for q in notes] == [(3, 2, 72), (6, F(1, 2), 74)]


@pytest.mark.parametrize("ts", [(4, 4), (3, 4), (6, 8), (2, 2)])
def test_every_measure_matches_time_signature(ts):
    part = [QNote(F(k, 4), F(1 + (k % 5), 4), 60 + (k % 12)) for k in range(0, 80, 3)]
    score = _score({"S": part}, time_sig=ts)
    parsed = parse_nwctxt(score_to_nwctxt(score))
    assert all(m == score.measure_beats() for m in parsed.measures["S"])


def test_writer_uses_clef_and_key_and_lyrics(tmp_path):
    voices = [
        VoiceSpec("Lead", "treble", 55, 79),
        VoiceSpec("T", "treble8vb", 48, 69),
        VoiceSpec("B", "bass", 40, 62),
    ]
    parts = {
        "Lead": [QNote(F(0), F(1), 67, lyric="사"), QNote(F(1), F(1), 69, lyric="랑")],
        "T": [QNote(F(0), F(2), 59)],
        "B": [QNote(F(0), F(2), 43)],
    }
    path = write_nwctxt(_score(parts, voices, key=1), tmp_path / "x.nwctxt")
    text = path.read_text(encoding="utf-8")
    assert "|Clef|Type:Treble|OctaveShift:Octave Down" in text
    assert "|Clef|Type:Bass" in text
    assert text.count("|Key|Signature:F#") == 3
    assert '|Lyric1|Text:"사 랑"' in text
    assert "\r\n" in path.read_bytes().decode("utf-8")
    back = read_nwctxt(path)
    assert [q.lyric for q in merge_ties(back.parts["Lead"])] == ["사", "랑"]


def test_writer_enforces_monophony():
    part = [QNote(F(0), F(2), 60), QNote(F(1), F(1), 64)]
    parsed = parse_nwctxt(score_to_nwctxt(_score({"S": part})))
    notes = merge_ties(parsed.score.parts["S"])
    assert [(q.start_beat, q.dur_beats, q.midi) for q in notes] == [(0, 1, 60), (1, 1, 64)]
