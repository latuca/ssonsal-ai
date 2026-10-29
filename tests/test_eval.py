"""평가 지표와 run_eval 리포트 테스트(합성 gold 곡)."""

import json
import shutil
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acappella.config import parse_voices
from acappella.meta import SongMeta, load_meta, save_meta
from acappella.nwc.writer import write_nwctxt
from acappella.types import QNote, Score
from eval.metrics import (
    constant_time_map,
    evaluate,
    rhythm_agreement,
    table_time_map,
    transpose,
)

sys.path.insert(0, str(Path(__file__).parent))
from synth_utils import CHORALE, SR, render_satb  # noqa: E402


def chorale_score(shift_beats=0) -> Score:
    parts = {
        name: [QNote(F(s + shift_beats), F(d), m) for s, d, m in notes]
        for name, notes in CHORALE.items()
    }
    return Score("chorale", 100, (4, 4), 0, parse_voices("SATB"), parts)


def test_perfect_estimate_scores_one():
    ref = chorale_score()
    tm = constant_time_map(100)
    m = evaluate(ref, chorale_score(), tm, tm)
    assert m["note"]["f1"] == pytest.approx(1.0)
    assert m["multipitch"]["accuracy"] == pytest.approx(1.0, abs=0.02)
    assert all(v["raw_pitch_accuracy"] > 0.98 for v in m["voices"].values())
    assert m["rhythm"]["onset_match"] == 1.0


def test_rhythm_agreement_finds_global_shift():
    r = rhythm_agreement(chorale_score(), chorale_score(shift_beats=4))
    assert r["onset_match"] == 1.0 and r["shift_beats"] == 4.0


def test_transpose_and_table_time_map():
    t = transpose(chorale_score(), 2)
    assert t.parts["B"][0].midi == 50
    tm = table_time_map(np.array([0.0, 4.0]), np.array([1.0, 3.0]))
    assert np.allclose(tm(np.array([-2.0, 2.0, 8.0])), [0.0, 2.0, 5.0])


def test_song_meta_roundtrip(tmp_path):
    m = SongMeta(
        "s0001", title="곡", tier="C", voices=["S", "A"], split="test", extra={"genre": "발라드"}
    )
    back = load_meta(save_meta(m, tmp_path / "s0001.json"))
    assert back == m and back.genre == "발라드"
    with pytest.raises(ValueError):
        SongMeta("x", tier="D")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")
def test_run_eval_end_to_end(tmp_path, monkeypatch):
    from eval import run_eval

    data = tmp_path / "data"
    lead_in = 0.5
    y, _ = render_satb(CHORALE, 100)
    (data / "audio").mkdir(parents=True)
    sf.write(
        str(data / "audio" / "g001.wav"),
        np.concatenate([np.zeros(int(lead_in * SR), np.float32), y]),
        SR,
    )
    write_nwctxt(chorale_score(), data / "nwctxt" / "g001.nwctxt")
    (data / "align").mkdir()
    (data / "align" / "g001.csv").write_text(
        "beat,seconds\n" + "\n".join(f"{b},{lead_in + b * 0.6:.3f}" for b in range(0, 17)),
        encoding="utf-8",
    )
    meta = SongMeta(
        "g001",
        title="코랄",
        tier="C",
        voices=["S", "A", "T", "B"],
        split="test",
        extra={"genre": "합창"},
    )
    save_meta(meta, data / "meta" / "songs" / "g001.json")
    monkeypatch.setenv("ACAPPELLA_HOME", str(tmp_path / "home"))
    out = tmp_path / "report"
    code = run_eval.main(
        ["--meta-dir", str(data / "meta" / "songs"), "--data-dir", str(data), "--out", str(out)]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    song = report["songs"][0]
    assert song["metrics"]["note"]["f1"] > 0.5
    assert song["metrics"]["voices"]["B"]["raw_pitch_accuracy"] > 0.8
    assert song["metrics"]["rhythm"]["onset_match"] > 0.5
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "노트 F1" in md and "합창" in md
