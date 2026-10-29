"""입력 수집과 전체 파이프라인 테스트(합성 음원, 실제 ffmpeg·Basic Pitch 사용).

불변 조건: 성부별 단선율, 모든 마디 길이 = 박자표, 음역 경고, 같은 입력 → 같은 출력.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acappella.config import Settings, parse_voices
from acappella.nwc.reader import parse_nwctxt, read_text
from acappella.pipeline.ingest import IngestError, ingest, is_url
from acappella.pipeline.run import TranscribeOptions, main, run_pipeline
from acappella.score_utils import merge_ties, monophony_violations

sys.path.insert(0, str(Path(__file__).parent))
from synth_utils import CHORALE, SR, render_satb  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path / "home")


def _write_song(path: Path, parts, bpm=100.0, lead_in=0.5):
    y, _ = render_satb(parts, bpm)
    y = np.concatenate([np.zeros(int(lead_in * SR), np.float32), y])
    sf.write(str(path), y, SR)
    return path


# --- ingest ------------------------------------------------------------------


def test_is_url():
    assert is_url("https://www.youtube.com/watch?v=abc")
    assert not is_url("C:\\music\\song.mp3")


def test_ingest_missing_and_bad_files(tmp_path, settings):
    with pytest.raises(IngestError, match="찾을 수 없습니다"):
        ingest(str(tmp_path / "nope.mp3"), tmp_path / "w", settings)
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    with pytest.raises(IngestError, match="지원하지 않는"):
        ingest(str(bad), tmp_path / "w", settings)


def test_ingest_url_failure_gives_next_action(tmp_path, settings):
    settings.ytdlp_path = str(tmp_path / "no-such-yt-dlp")
    with pytest.raises(IngestError, match="음원 파일로 시도해 주세요"):
        ingest("https://www.youtube.com/watch?v=x", tmp_path / "w", settings)


@needs_ffmpeg
def test_ingest_converts_stereo_44k_to_mono_22k(tmp_path, settings):
    t = np.arange(44100) / 44100
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=1) * 0.3
    src = tmp_path / "곡 이름.wav"
    sf.write(str(src), stereo.astype(np.float32), 44100)
    clip = ingest(str(src), tmp_path / "w", settings)
    assert clip.sr == 22050
    assert clip.samples.ndim == 1 and abs(clip.samples.size - 22050) < 100
    assert clip.title == "곡 이름"


@needs_ffmpeg
def test_ingest_bad_audio_content(tmp_path, settings):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not really an mp3")
    with pytest.raises(IngestError, match="음원 파일을 읽을 수 없습니다"):
        ingest(str(bad), tmp_path / "w", settings)


# --- 전체 파이프라인 ------------------------------------------------------------


def _check_invariants(nwc_path: Path, voices):
    parsed = parse_nwctxt(read_text(nwc_path))
    score = parsed.score
    assert [v.name for v in score.voices] == [v.name for v in voices]
    measure = score.measure_beats()
    for name, lengths in parsed.measures.items():
        assert lengths and all(m == measure for m in lengths), name
    for name, part in score.parts.items():
        assert not monophony_violations(part), name
    return score


def _accuracy(score, ref, voice):
    """정답 음표 중 같은 시작 박·같은 음높이로 나온 비율."""
    got = {(q.start_beat, q.midi) for q in merge_ties(score.parts[voice])}
    return sum((s, m) in got for s, _, m in ref[voice]) / len(ref[voice])


@needs_ffmpeg
def test_transcribe_chorale_end_to_end(tmp_path, settings):
    src = _write_song(tmp_path / "chorale.wav", CHORALE)
    voices = parse_voices("SATB")
    opts = TranscribeOptions(export_midi=True)
    res = run_pipeline(str(src), voices, tmp_path / "out", None, opts, settings)
    assert res.nwctxt.exists() and res.nwctxt.suffix == ".nwctxt"
    assert {p.name for p in res.midi} >= {"chorale.mid", "chorale_S.mid", "chorale_B.mid"}
    for p in res.midi:
        assert p.read_bytes()[:4] == b"MThd"
    # 단계별 중간 결과가 남고, 음원은 지워진다
    names = {p.name for p in res.job_dir.iterdir()}
    assert {
        "3_pitch.npz",
        "4_voices.npz",
        "5_notes.json",
        "7_score.json",
        "8_score.nwctxt",
    } <= names
    assert "input.wav" not in names

    score = _check_invariants(res.nwctxt, voices)
    assert 95 <= res.score.tempo_bpm <= 105
    assert res.score.time_sig == (4, 4)
    assert score.key_fifths == 0
    # 베이스라인 품질 하한(합성 코랄)
    assert _accuracy(res.score, CHORALE, "B") >= 0.85
    assert _accuracy(res.score, CHORALE, "T") >= 0.5
    assert _accuracy(res.score, CHORALE, "S") >= 0.5

    # 같은 입력, 같은 설정 → 같은 출력
    res2 = run_pipeline(str(src), voices, tmp_path / "out2", None, opts, settings)
    assert res2.nwctxt.read_bytes() == res.nwctxt.read_bytes()


@needs_ffmpeg
def test_cli_transcribe(tmp_path, monkeypatch):
    monkeypatch.setenv("ACAPPELLA_HOME", str(tmp_path / "home"))
    src = _write_song(
        tmp_path / "duet.wav", {"S": [(0, 4, 72), (4, 4, 71)], "B": [(0, 4, 48), (4, 4, 55)]}
    )
    code = main(
        [
            "transcribe",
            str(src),
            "--voices",
            "S,B",
            "--out",
            str(tmp_path / "o"),
            "--tempo",
            "100",
            "--time-sig",
            "4/4",
            "--title",
            "듀엣",
            "--lyrics",
            "라 라",
        ]
    )
    assert code == 0
    text = (tmp_path / "o" / "듀엣.nwctxt").read_text(encoding="utf-8")
    assert '|SongInfo|Title:"듀엣"' in text
    assert "|Tempo|Tempo:100" in text
    assert '|Lyric1|Text:"라 라"' in text
    assert main(["transcribe", str(tmp_path / "missing.mp3"), "--out", str(tmp_path / "o")]) == 2


# 3/4 왈츠, 5성부
WALTZ = {
    "S": [(0, 3, 76), (3, 2, 77), (5, 1, 76), (6, 3, 74), (9, 3, 72)],
    "A": [(0, 3, 67), (3, 3, 69), (6, 3, 67), (9, 3, 67)],
    "T": [(0, 3, 64), (3, 3, 65), (6, 3, 62), (9, 3, 64)],
    "B": [(0, 3, 48), (3, 3, 53), (6, 3, 55), (9, 3, 48)],
}
FIVE = {
    "Lead": [(0, 1, 79), (1, 1, 81), (2, 2, 79), (4, 2, 77), (6, 2, 76), (8, 4, 79)],
    "S": [(0, 4, 72), (4, 4, 71), (8, 4, 72)],
    "A": [(0, 4, 67), (4, 4, 67), (8, 4, 64)],
    "T": [(0, 4, 64), (4, 4, 62), (8, 4, 60)],
    "B": [(0, 4, 48), (4, 4, 43), (8, 4, 48)],
}


@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize(
    "name,parts,voices,ts",
    [
        ("chorale", CHORALE, "SATB", None),
        ("waltz", WALTZ, "SATB", (3, 4)),
        ("five", FIVE, "Lead+SATB", None),
    ],
)
def test_three_samples_complete(tmp_path, settings, name, parts, voices, ts):
    """M2 완료 기준: 샘플 3곡이 끝까지 처리되고 불변 조건을 통과한다."""
    src = _write_song(tmp_path / f"{name}.wav", parts, bpm=96)
    vs = parse_voices(voices)
    res = run_pipeline(
        str(src), vs, tmp_path / "out", None, TranscribeOptions(time_sig=ts), settings
    )
    _check_invariants(res.nwctxt, vs)
