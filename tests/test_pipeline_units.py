"""파이프라인 단계별 단위 테스트(합성 입력)."""

from fractions import Fraction as F

import numpy as np
import pytest

from acappella.config import Thresholds, note_to_midi, parse_voices
from acappella.pipeline.lyrics import place_lyrics, split_english_word, split_syllables
from acappella.pipeline.notes import segment_notes
from acappella.pipeline.pitch import contour_midi_bins, frame_times
from acappella.pipeline.postprocess import estimate_key, fill_unisons, postprocess, remove_spurious
from acappella.pipeline.preprocess import normalize_loudness
from acappella.pipeline.rhythm import (
    BeatGrid,
    estimate_meter,
    fix_tempo_octave,
    quantize,
    with_pickup_shift,
)
from acappella.pipeline.voices import assign_frame, assign_voices, harmonic_weighted
from acappella.score_utils import monophony_violations
from acappella.types import (
    AudioClip,
    NoteEvent,
    PitchActivation,
    QNote,
    Score,
    VoiceSpec,
    VoiceTrack,
)

SATB = parse_voices("SATB")


def make_activation(voice_pitches: list[list[float]], frames_per_step: int = 40, amp=0.8):
    """성부별 음높이 열(무음은 nan)로 Basic Pitch contour 모양 salience를 만든다."""
    bins = contour_midi_bins()
    steps = len(voice_pitches[0])
    T = steps * frames_per_step
    sal = np.zeros((T, bins.size), dtype=np.float32)
    for seq in voice_pitches:
        for k, p in enumerate(seq):
            if np.isnan(p):
                continue
            rows = slice(k * frames_per_step, (k + 1) * frames_per_step)
            sal[rows] = np.maximum(sal[rows], amp * np.exp(-0.5 * ((bins - p) / 0.25) ** 2))
    return PitchActivation(times=frame_times(T), midi_bins=bins, salience=sal)


# --- config ------------------------------------------------------------------


def test_note_names_and_voice_parsing():
    assert note_to_midi("C4") == 60
    assert note_to_midi("Bb2") == 46
    assert note_to_midi("F#5") == 78
    vs = parse_voices("Lead,S,A,T:bass:C3:G4,B")
    assert [v.name for v in vs] == ["Lead", "S", "A", "T", "B"]
    assert vs[3].clef == "bass" and (vs[3].low_midi, vs[3].high_midi) == (48, 67)
    assert [v.name for v in parse_voices("Lead+SSATTB")] == [
        "Lead",
        "S1",
        "S2",
        "A",
        "T1",
        "T2",
        "B",
    ]
    with pytest.raises(ValueError):
        parse_voices("S,S")


# --- preprocess --------------------------------------------------------------


def test_normalize_loudness_hits_target_and_limits_peak():
    x = (0.01 * np.sin(np.linspace(0, 200, 22050)) + 0.3).astype(np.float32)
    y = normalize_loudness(x, -20.0, 0.99)
    assert abs(float(y.mean())) < 1e-4
    assert abs(20 * np.log10(np.sqrt(np.mean(y.astype(float) ** 2))) + 20) < 0.5
    spike = np.zeros(1000, np.float32)
    spike[0] = 1.0
    assert np.max(np.abs(normalize_loudness(spike, 0.0, 0.9))) <= 0.9 + 1e-6
    assert not normalize_loudness(np.zeros(10, np.float32)).any()


# --- voices ------------------------------------------------------------------


def test_assign_frame_rank_order():
    th = Thresholds()
    assert assign_frame([72, 67, 64, 48], [0.8] * 4, SATB, th) == [72, 67, 64, 48]


def test_assign_frame_unison_fills_missing_voice():
    th = Thresholds(unison_cost=0.6)
    # 소프라노·알토가 같은 음(67)이라 후보가 셋뿐
    out = assign_frame([67, 60, 48], [0.9, 0.8, 0.8], SATB, th)
    assert out == [67, 67, 60, 48]


def test_assign_frame_default_prefers_rest_over_duplicate():
    # 기본값은 후보가 모자라면 같은 음을 복사하지 않고 한 성부를 쉬게 한다(중복 음표 방지)
    out = assign_frame([67, 60, 48], [0.9, 0.8, 0.8], SATB, Thresholds())
    assert sorted(p for p in out if p is not None) == [48, 60, 67]
    assert out.count(None) == 1


def test_assign_frame_drops_extra_candidate():
    th = Thresholds()
    out = assign_frame([84, 72, 67, 64, 48], [0.2, 0.8, 0.8, 0.8, 0.8], SATB, th)
    assert out == [72, 67, 64, 48]


def test_harmonic_weighting_penalizes_octave_above_strong_note():
    th = Thresholds()
    w = harmonic_weighted([60, 48], [0.6, 0.9], th)
    assert w[0] < 0.6 and w[1] == 0.9


def test_assign_voices_rule_follows_lines():
    lines = [[72, 74, 76, 74], [67, 67, 67, 65], [64, 59, 60, 59], [48, 55, 48, 55]]
    act = make_activation([list(map(float, x)) for x in lines])
    tracks = assign_voices(act, SATB, "rule")
    assert [t.voice.name for t in tracks] == ["S", "A", "T", "B"]
    for track, line in zip(tracks, lines):
        f0 = track.f0_midi
        n = f0.size // 4
        for k, p in enumerate(line):
            seg = f0[k * n + 3 : (k + 1) * n - 3]
            assert np.nanmedian(seg) == p, (track.voice.name, k)


def test_assign_voices_auto_without_model_uses_rule(tmp_path):
    act = make_activation([[72.0], [60.0]])
    tracks = assign_voices(act, parse_voices("S,B"), "auto", tmp_path / "missing.onnx")
    assert len(tracks) == 2


def test_assign_voices_onnx_with_fake_session(monkeypatch):
    from acappella.pipeline import voices as V

    class FakeSession:
        def run(self, names, feeds):
            x = feeds["salience"]
            assert x.shape == (1, 1, V.ONNX_WINDOW, V.ONNX_IN_BINS)
            assert feeds["voice_ranges"].shape == (1, V.ONNX_K, 3)
            pitch = np.full((1, V.ONNX_K, V.ONNX_WINDOW, V.ONNX_OUT_BINS), -10.0, np.float32)
            pitch[0, 0, :, 72 - 21] = 10.0  # 슬롯 0(가장 높은 성부) = C5
            pitch[0, 1, :, 48 - 21] = 10.0
            active = np.full((1, V.ONNX_K, V.ONNX_WINDOW), -10.0, np.float32)
            active[0, :2] = 10.0
            return [pitch, active]

    act = make_activation([[72.0] * 5, [48.0] * 5])
    voices = parse_voices("B,S")  # 입력 순서와 무관하게 슬롯은 음역 순
    tracks = V.assign_voices_onnx(act, voices, model_path=None, session=FakeSession())
    assert np.all(tracks[0].f0_midi == 48) and np.all(tracks[1].f0_midi == 72)


# --- notes -------------------------------------------------------------------


def _track(f0, dt=0.0232):
    v = VoiceSpec("S", "treble", 60, 81)
    return VoiceTrack(v, np.arange(len(f0)) * dt, np.asarray(f0, float))


def test_segment_notes_merges_blips_and_splits_changes():
    f0 = [60.0] * 30 + [62.0] * 1 + [60.0] * 30 + [np.nan] * 20 + [64.0] * 25
    notes = segment_notes([_track(f0)])["S"]
    assert [n.midi for n in notes] == [60, 64]
    assert notes[0].offset == pytest.approx(61 * 0.0232, abs=0.03)


def test_segment_notes_splits_on_reattack():
    f0 = [67.0] * 80
    act = PitchActivation(
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 0)),
        notes=[NoteEvent(0.0, 0.9, 67), NoteEvent(0.93, 1.8, 67)],
    )
    notes = segment_notes([_track(f0)], act)["S"]
    assert len(notes) == 2 and notes[1].onset == pytest.approx(0.93)


# --- rhythm ------------------------------------------------------------------


def _grid(bpm=120.0, n=64, ts=(4, 4)):
    return BeatGrid(np.arange(n) * 60 / bpm, bpm, ts, 0)


def test_quantize_snaps_to_sixteenth_grid_and_is_monophonic():
    clip = AudioClip(np.zeros(22050, np.float32), 22050, "x", "t")
    ev = {"S": [NoteEvent(0.01, 0.49, 72), NoteEvent(0.52, 0.76, 74), NoteEvent(0.70, 1.0, 76)]}
    score = quantize(ev, clip, [VoiceSpec("S", "treble", 60, 81)], grid=_grid())
    got = [(q.start_beat, q.dur_beats, q.midi) for q in score.parts["S"]]
    assert got == [(0, 1, 72), (1, F(1, 2), 74), (F(3, 2), F(1, 2), 76)]
    assert not monophony_violations(score.parts["S"])
    assert score.tempo_bpm == 120


def test_tempo_octave_correction():
    beats = np.arange(10) * 1.2  # 50 BPM
    b2, bpm = fix_tempo_octave(beats, 50.0, 65, 170)
    assert bpm == 100 and b2.size == 19
    b3, bpm3 = fix_tempo_octave(np.arange(40) * 0.3, 200.0, 65, 170)
    assert bpm3 == 100 and b3.size == 20


def test_meter_estimation_prefers_accent_period():
    acc3 = np.tile([3.0, 1.0, 1.0], 12)
    assert estimate_meter(acc3, None) == ((3, 4), 0)
    acc4 = np.tile([1.0, 3.0, 1.0, 1.5], 12)
    assert estimate_meter(acc4, None) == ((4, 4), 1)
    assert estimate_meter(acc4, (3, 4))[0] == (3, 4)


def test_quantize_shifts_pickup_into_first_measure():
    clip = AudioClip(np.zeros(22050, np.float32), 22050, "x", "t")
    grid = BeatGrid(np.arange(1, 40) * 0.5, 120.0, (4, 4), 0)  # 첫 박이 0.5초
    ev = {"S": [NoteEvent(0.0, 0.5, 67), NoteEvent(0.5, 1.0, 72)]}
    grid = with_pickup_shift(grid, ev, 4)
    assert grid.shift_beats == 4
    score = quantize(ev, clip, [VoiceSpec("S", "treble", 60, 81)], grid=grid)
    assert [q.start_beat for q in score.parts["S"]] == [3, 4]


# --- postprocess ---------------------------------------------------------------


def _score(parts, voices=None):
    return Score("t", 100, (4, 4), 0, voices or SATB, parts)


def test_estimate_key():
    g_major = [67, 69, 71, 72, 74, 76, 78, 79, 74, 71, 67, 62]
    part = [QNote(F(i), F(1), m) for i, m in enumerate(g_major)]
    assert estimate_key(_score({"S": part})) == 1
    e_minor = [64, 66, 67, 69, 71, 72, 75, 76, 71, 67, 64, 59]  # 마단조 → 사장조 조표
    part = [QNote(F(i), F(1), m) for i, m in enumerate(e_minor)]
    assert estimate_key(_score({"S": part})) == 1
    f_major = [65, 67, 69, 70, 72, 74, 76, 77, 72, 70, 65, 60]
    part = [QNote(F(i), F(1), m) for i, m in enumerate(f_major)]
    assert estimate_key(_score({"S": part})) == -1


def test_remove_spurious_short_notes():
    part = [QNote(F(0), F(1), 67), QNote(F(1), F(1, 4), 80), QNote(F(5, 4), F(3, 4), 69)]
    out = remove_spurious(part, F(1, 4), 5)
    assert [(q.start_beat, q.dur_beats, q.midi) for q in out] == [
        (0, F(5, 4), 67),
        (F(5, 4), F(3, 4), 69),
    ]
    part = [QNote(F(0), F(1), 67), QNote(F(1), F(1, 4), 64), QNote(F(5, 4), F(3, 4), 67)]
    assert [q.midi for q in remove_spurious(part, F(1, 4), 5)] == [67, 67, 67]


def test_fill_unisons_short_gap_from_neighbor():
    parts = {
        "S": [QNote(F(0), F(2), 72), QNote(F(3), F(1), 72)],
        "A": [QNote(F(0), F(4), 67)],
        "T": [QNote(F(0), F(4), 60)],
        "B": [QNote(F(0), F(4), 48)],
    }
    out = fill_unisons(_score(parts), F(1))
    s = sorted(out.parts["S"], key=lambda q: q.start_beat)
    assert [(q.start_beat, q.midi) for q in s] == [(0, 72), (2, 67), (3, 72)]


def test_postprocess_keeps_user_key():
    parts = {"S": [QNote(F(0), F(4), 72)], "A": [], "T": [], "B": []}
    assert postprocess(_score(parts), key_fifths=-3).key_fifths == -3


# --- lyrics ------------------------------------------------------------------


def test_split_syllables_korean_english():
    assert split_syllables("사랑해 너를") == ["사", "랑", "해", "너", "를"]
    assert split_syllables("hello world _ sing-ing") == [
        "hel-",
        "lo",
        "world",
        None,
        "sing-",
        "ing",
    ]
    assert split_english_word("love") == ["love"]
    parts = split_english_word("beautiful")
    assert "".join(parts) == "beautiful" and len(parts) == 3


def test_place_lyrics_skips_tied_and_rests():
    parts = {
        "S": [
            QNote(F(0), F(1), 72, tie_next=True),
            QNote(F(1), F(1), 72),
            QNote(F(2), F(1), None),
            QNote(F(3), F(1), 74),
        ]
    }
    out = place_lyrics(Score("t", 100, (4, 4), 0, [SATB[0]], parts), "S", "아 이 우")
    assert [q.lyric for q in out.parts["S"]] == ["아", None, None, "이"]


def test_dedupe_unisons_moves_weaker_voice_to_second_choice():
    from acappella.pipeline.voices import dedupe_unisons

    prob = np.zeros((2, 1, 5))
    prob[0, 0, [3, 2]] = [0.9, 0.05]
    prob[1, 0, [3, 1]] = [0.6, 0.3]  # 1순위가 겹치고 2순위가 충분히 그럴듯하다
    active = np.ones((2, 1), bool)
    assert dedupe_unisons(prob, active, 0.3)[:, 0].tolist() == [3, 1]
    assert dedupe_unisons(prob, active, 0.0)[:, 0].tolist() == [3, 3]
    prob[1, 0, 1] = 0.1  # 2순위가 약하면 진짜 유니즌으로 둔다
    assert dedupe_unisons(prob, active, 0.3)[:, 0].tolist() == [3, 3]
