"""메타데이터 생성과 곡·그룹 단위 분할.

    uv run python -m tools.build_dataset --gold data/meta/gold.csv

1) data/nwctxt/*.nwctxt마다 메타데이터(data/meta/songs/<song_id>.json)를 만들거나 갱신한다.
   제목·성부(스태프 이름)·가사 언어·메들리 여부를 악보에서 채우고, 사람이 고친 값은 덮어쓰지 않는다.
2) --gold CSV(song_id[,youtube_url,group,genre])에 있는 곡은 tier C로 둔다.
3) 분할: gold 곡의 그룹을 valid/test로 나누고(해시 기반, 재현 가능), 그 그룹에 속한 곡은 tier와
   상관없이 같은 쪽에 둔다. gold가 없는 실제 음원(silver) 그룹은 train/valid/test를 2:1:1로 나눈다.
   실제 음원이 없는 그룹은 train이고, ``--synth-valid`` 비율만큼은 합성 valid로 뺀다.
   같은 곡의 합성본(A)과 실제본(B)은 같은 song_id이므로 자연히 같은 쪽이다.
4) 그룹이 비어 있으면 아카이브 파일 이름(data/meta/id_map.csv)에서 곡 제목을 뽑아
   ``title:<정규화한 제목>``으로 둔다. 같은 곡의 다른 편곡·조성이 한쪽에만 들어가게 한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from acappella.meta import SongMeta, load_meta, save_meta
from acappella.nwc.reader import parse_nwctxt, read_text

MEDLEY_RE = re.compile(r"(?i)(medley|메들리|mashup|매쉬업)")
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[぀-ヿ]")
_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def lyrics_language(text: str) -> str:
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _CJK.search(text):
        return "zh"
    if _LATIN.search(text):
        return "en"
    return ""


_PREFIX_RE = re.compile(r"^\s*(?:\*?\[([^\]]*)\]|\(([^)]*)\))\s*")
_PARTS_RE = re.compile(r"(?i)^(solo)?[SMATBRPCVL0-9]{3,}$")


def parse_file_name(name: str) -> dict[str, str]:
    """아카이브 파일 이름 규칙(제목_가수_편곡자_..._파트구성)에서 제목·가수·아카펠라 그룹을 뽑는다."""
    stem = re.sub(r"(?i)\.nwc(\.bak)?$", "", name)
    group = ""
    while m := _PREFIX_RE.match(stem):
        inner = (m.group(1) or m.group(2) or "").strip()
        if m.group(1) is not None and not _PARTS_RE.match(inner):
            group = inner
        stem = stem[m.end() :]
    pieces = [p.strip() for p in re.split(r"_| - ", stem) if p.strip()]
    title = pieces[0] if pieces else stem
    if " - " not in stem and "-" in title and not pieces[1:]:
        title, _, rest = title.partition("-")
        pieces = [title, rest]
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip() or title
    artist = pieces[1] if len(pieces) > 1 and not _PARTS_RE.match(pieces[1]) else ""
    return {"file_title": title, "artist": artist, "pro_group": group}


def title_key(title: str) -> str:
    return re.sub(r"[^0-9a-z가-힣ぁ-ヿ一-鿿]", "", title.lower())


def read_id_paths(data_dir: Path) -> dict[str, str]:
    path = data_dir / "meta" / "id_map.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["song_id"]: r["path"] for r in csv.DictReader(f)}


def group_of(m: SongMeta) -> str:
    return m.group or f"song:{m.song_id}"


def _hash01(key: str, salt: str) -> float:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(
    metas: list[SongMeta], valid_ratio: float = 0.5, salt: str = "v1", synth_valid: float = 0.0
) -> None:
    gold_groups = sorted({group_of(m) for m in metas if m.tier == "C"})
    split_of = {g: ("valid" if _hash01(g, salt) < valid_ratio else "test") for g in gold_groups}
    if len(gold_groups) >= 2 and len(set(split_of.values())) == 1:
        # 해시가 한쪽으로만 몰리면 정렬 순서로 반씩 나눈다
        for i, g in enumerate(sorted(gold_groups, key=lambda g: _hash01(g, salt))):
            split_of[g] = "valid" if i < max(1, round(len(gold_groups) * valid_ratio)) else "test"
    # gold가 없는 silver 그룹: 해시 순서로 train 2 : valid 1 : test 1
    silver = sorted(
        {group_of(m) for m in metas if m.tier == "B"} - set(split_of),
        key=lambda g: _hash01(g, salt),
    )
    for i, g in enumerate(silver):
        split_of[g] = ("train", "train", "valid", "test")[i % 4]
    for m in metas:
        g = group_of(m)
        if g in split_of:
            m.split = split_of[g]
        else:
            m.split = "valid" if _hash01(g, salt + ":synth") < synth_valid else "train"


def check_disjoint(metas: list[SongMeta]) -> list[str]:
    seen: dict[str, set[str]] = {}
    for m in metas:
        seen.setdefault(group_of(m), set()).add(m.split)
    return [g for g, s in seen.items() if len(s) > 1]


def update_from_score(m: SongMeta, text: str) -> None:
    score = parse_nwctxt(text).score
    m.title = m.title or score.title
    m.voices = m.voices or [v.name for v in score.voices]
    lyrics = " ".join(q.lyric for p in score.parts.values() for q in p if q.lyric)
    m.lyrics_lang = m.lyrics_lang or lyrics_language(lyrics)
    m.is_medley = m.is_medley or bool(MEDLEY_RE.search(m.title))


def read_gold(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["song_id"]: r for r in csv.DictReader(f)}


def build(
    data_dir: Path, gold_csv: Path | None, valid_ratio: float, synth_valid: float = 0.0
) -> list[SongMeta]:
    meta_dir = data_dir / "meta" / "songs"
    gold = read_gold(gold_csv)
    id_paths = read_id_paths(data_dir)
    metas = []
    for nwctxt in sorted((data_dir / "nwctxt").glob("*.nwctxt")):
        sid = nwctxt.stem
        path = meta_dir / f"{sid}.json"
        m = load_meta(path) if path.exists() else SongMeta(sid, nwc_path=str(nwctxt))
        update_from_score(m, read_text(nwctxt))
        if sid in id_paths:
            info = parse_file_name(Path(id_paths[sid]).name)
            for k, v in info.items():
                m.extra.setdefault(k, v)
            if not m.group and title_key(info["file_title"]):
                m.group = f"title:{title_key(info['file_title'])}"
        if sid in gold:
            g = gold[sid]
            m.tier = "C"
            m.youtube_url = g.get("youtube_url") or m.youtube_url
            m.group = g.get("group") or m.group
            if g.get("genre"):
                m.extra["genre"] = g["genre"]
        metas.append(m)
    assign_splits(metas, valid_ratio, synth_valid=synth_valid)
    bad = check_disjoint(metas)
    if bad:
        raise RuntimeError(f"그룹이 여러 분할에 걸쳐 있습니다: {bad[:5]}")
    for m in metas:
        save_meta(m, meta_dir / f"{m.song_id}.json")
    return metas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="메타데이터 생성과 곡·그룹 단위 분할")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--gold", type=Path, help="gold 곡 CSV (song_id[,youtube_url,group,genre])")
    ap.add_argument("--valid-ratio", type=float, default=0.5, help="gold 그룹 중 valid 비율")
    ap.add_argument(
        "--synth-valid", type=float, default=0.08, help="실제 음원이 없는 그룹 중 합성 valid 비율"
    )
    args = ap.parse_args(argv)
    metas = build(args.data_dir, args.gold, args.valid_ratio, args.synth_valid)
    c = Counter((m.tier, m.split) for m in metas)
    print(f"곡 {len(metas)}개")
    for (tier, split), n in sorted(c.items()):
        print(f"  tier {tier} / {split}: {n}")
    print(
        "메들리:",
        sum(m.is_medley for m in metas),
        " 조옮김 있음:",
        sum(bool(m.transpose_semitones) for m in metas),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
