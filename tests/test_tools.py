"""데이터 구축 도구 테스트(가짜 아카이브)."""

import csv
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from acappella.meta import SongMeta, load_all, save_meta
from tools import build_dataset, extract_nwctxt, scan_archive
from tools.align import align
from tools.nwcfile import read_nwc

sys.path.insert(0, str(Path(__file__).parent))
from synth_utils import CHORALE, SR, render_satb  # noqa: E402
from test_eval import chorale_score  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "handmade_satb.nwctxt"


def fake_nwc(text: str | None, minor=75, major=2) -> bytes:
    body = b"[NoteWorthy ArtWare]\0\0\0[NoteWorthy Composer]\0" + bytes([minor, major]) + b"\0" * 16
    if text is not None:
        body += text.encode("utf-8")
    return b"[NWZ]\0" + zlib.compress(body)


def test_read_nwc_extracts_embedded_text(tmp_path):
    text = FIXTURE.read_text(encoding="utf-8")
    p = tmp_path / "a.nwc"
    p.write_bytes(fake_nwc(text))
    c = read_nwc(p)
    assert c.text is not None and c.text.startswith("!NoteWorthyComposer(2.75)")
    assert c.version == "2.75"
    p.write_bytes(fake_nwc(None, 75, 1))
    c = read_nwc(p)
    assert c.text is None and c.version == "1.75" and "2.75 미만" in c.error
    p.write_bytes(b"[NWZ]\0garbage")
    assert read_nwc(p).error


def _archive(tmp_path):
    root = tmp_path / "archive"
    (root / "2019").mkdir(parents=True)
    text = FIXTURE.read_text(encoding="utf-8")
    (root / "2019" / "연습곡.nwc").write_bytes(fake_nwc(text))
    (root / "old.nwc").write_bytes(fake_nwc(None, 75, 1))
    (root / "직접.nwctxt").write_text(text, encoding="utf-8")
    (root / "2019" / "연습곡 소프라노 파트.mp3").write_bytes(b"\0" * 100)
    (root / "곡 [abcdefghijk].wav").write_bytes(b"\0" * 10)
    return root


def test_scan_archive(tmp_path):
    rows = {r["path"].replace("\\", "/"): r for r in scan_archive.scan(_archive(tmp_path))}
    assert rows["2019/연습곡.nwc"]["title"] == "작은 연습곡"
    assert rows["2019/연습곡.nwc"]["nwc_version"] == "2.75"
    assert rows["old.nwc"]["has_nwctxt"] == "N"
    assert rows["2019/연습곡 소프라노 파트.mp3"]["part_practice_audio"] == "Y"
    assert "watch?v=abcdefghijk" in rows["곡 [abcdefghijk].wav"]["youtube_links"]


def test_extract_assigns_stable_ids_and_logs_failures(tmp_path):
    root, data = _archive(tmp_path), tmp_path / "data"
    ok, failures = extract_nwctxt.extract(root, data)
    assert ok == 2 and len(failures) == 1 and failures[0]["path"] == "old.nwc"
    ids1 = extract_nwctxt.load_id_map(data / "meta" / "id_map.csv")
    (root / "new.nwctxt").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    extract_nwctxt.extract(root, data)
    ids2 = extract_nwctxt.load_id_map(data / "meta" / "id_map.csv")
    assert all(ids2[k] == v for k, v in ids1.items())
    assert ids2["new.nwctxt"] == "s0004"
    assert len(list((data / "nwctxt").glob("*.nwctxt"))) == 3
    with open(data / "meta" / "extract_failures.csv", encoding="utf-8-sig") as f:
        assert [r["path"] for r in csv.DictReader(f)] == ["old.nwc"]


def test_align_detects_transposition_and_timing():
    score = chorale_score()
    shifted = {k: [(s, d, m + 2) for s, d, m in v] for k, v in CHORALE.items()}
    y, _ = render_satb(shifted, bpm=100)
    lead = 1.0
    y = np.concatenate([np.zeros(int(lead * SR), np.float32), y])
    res = align(score, y, SR)
    assert res.transpose_semitones == 2
    assert res.cost < 0.3
    # 8박 = 4.8초 + 1초 앞여백
    assert res.seconds[8] == pytest.approx(lead + 8 * 0.6, abs=0.35)


def test_build_dataset_splits_by_group(tmp_path):
    data = tmp_path / "data"
    (data / "nwctxt").mkdir(parents=True)
    text = FIXTURE.read_text(encoding="utf-8")
    for i in range(1, 9):
        (data / "nwctxt" / f"s{i:04d}.nwctxt").write_text(
            text.replace("작은 연습곡", f"곡{i} 메들리" if i == 8 else f"곡{i}"), encoding="utf-8"
        )
    # 같은 그룹(채널)의 silver 곡은 gold와 같은 쪽으로 가야 한다
    save_meta(SongMeta("s0005", tier="B", group="채널X"), data / "meta" / "songs" / "s0005.json")
    save_meta(SongMeta("s0006", tier="B", group="채널Y"), data / "meta" / "songs" / "s0006.json")
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "song_id,youtube_url,group,genre\ns0001,https://y/1,채널X,발라드\n"
        "s0002,https://y/2,채널Y,팝\ns0003,https://y/3,채널Z,팝\n",
        encoding="utf-8",
    )
    metas = {m.song_id: m for m in build_dataset.build(data, gold, 0.5)}
    assert {m.split for m in metas.values() if m.tier == "C"} == {"valid", "test"}
    assert metas["s0005"].split == metas["s0001"].split
    assert metas["s0006"].split == metas["s0002"].split
    assert metas["s0004"].split == "train"
    assert metas["s0001"].genre == "발라드"
    assert metas["s0008"].is_medley
    assert metas["s0001"].lyrics_lang == "ko"
    assert metas["s0001"].voices == ["S", "A", "T", "B"]
    assert not build_dataset.check_disjoint(list(metas.values()))
    assert len(load_all(data / "meta" / "songs")) == 8


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")
def test_real_data_example(tmp_path):
    import soundfile as sf

    from acappella.config import Settings
    from acappella.nwc.writer import write_nwctxt
    from training.real_data import make_real_example

    data = tmp_path / "data"
    write_nwctxt(chorale_score(), data / "nwctxt" / "s0001.nwctxt")
    y, _ = render_satb(CHORALE, bpm=100)
    (data / "audio").mkdir()
    sf.write(str(data / "audio" / "s0001.wav"), y, SR)
    (data / "align").mkdir()
    (data / "align" / "s0001.csv").write_text("beat,seconds\n0,0\n16,9.6\n", encoding="utf-8")
    meta = SongMeta("s0001", tier="B", voices=["S", "A", "T", "B"], split="train")
    sal, tgt, ranges = make_real_example(meta, data, Settings(home=tmp_path / "h"), backend="cqt")
    assert sal.shape[1] == 264 and tgt.shape[0] == 8
    t = int(1.0 / (256 / 22050))  # 1초 = 첫 화음
    assert list(tgt[:4, t]) == [72, 67, 64, 48]
