"""메타데이터 생성과 곡·그룹 단위 분할.

    uv run python -m tools.build_dataset --gold data/meta/gold.csv

1) data/nwctxt/*.nwctxt마다 메타데이터(data/meta/songs/<song_id>.json)를 만들거나 갱신한다.
   제목·성부(스태프 이름)·가사 언어·메들리 여부를 악보에서 채우고, 사람이 고친 값은 덮어쓰지 않는다.
2) --gold CSV(song_id[,youtube_url,group,genre])에 있는 곡은 tier C로 둔다.
3) 분할: gold 곡의 그룹을 valid/test로 나누고(해시 기반, 재현 가능), 그 그룹에 속한 곡은 tier와
   상관없이 같은 쪽에 둔다. 나머지는 train. 같은 곡의 합성본(A)과 실제본(B)은 같은 song_id이므로
   자연히 같은 쪽이다. 그룹이 비어 있으면 곡 하나를 그룹으로 본다.
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


def group_of(m: SongMeta) -> str:
    return m.group or f"song:{m.song_id}"


def _hash01(key: str, salt: str) -> float:
    return int(hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(metas: list[SongMeta], valid_ratio: float = 0.5, salt: str = "v1") -> None:
    gold_groups = sorted({group_of(m) for m in metas if m.tier == "C"})
    split_of = {g: ("valid" if _hash01(g, salt) < valid_ratio else "test") for g in gold_groups}
    if len(gold_groups) >= 2 and len(set(split_of.values())) == 1:
        # 해시가 한쪽으로만 몰리면 정렬 순서로 반씩 나눈다
        for i, g in enumerate(sorted(gold_groups, key=lambda g: _hash01(g, salt))):
            split_of[g] = "valid" if i < max(1, round(len(gold_groups) * valid_ratio)) else "test"
    for m in metas:
        m.split = split_of.get(group_of(m), "train")


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


def build(data_dir: Path, gold_csv: Path | None, valid_ratio: float) -> list[SongMeta]:
    meta_dir = data_dir / "meta" / "songs"
    gold = read_gold(gold_csv)
    metas = []
    for nwctxt in sorted((data_dir / "nwctxt").glob("*.nwctxt")):
        sid = nwctxt.stem
        path = meta_dir / f"{sid}.json"
        m = load_meta(path) if path.exists() else SongMeta(sid, nwc_path=str(nwctxt))
        update_from_score(m, read_text(nwctxt))
        if sid in gold:
            g = gold[sid]
            m.tier = "C"
            m.youtube_url = g.get("youtube_url") or m.youtube_url
            m.group = g.get("group") or m.group
            if g.get("genre"):
                m.extra["genre"] = g["genre"]
        metas.append(m)
    assign_splits(metas, valid_ratio)
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
    args = ap.parse_args(argv)
    metas = build(args.data_dir, args.gold, args.valid_ratio)
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
