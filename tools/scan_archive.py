"""동아리 아카이브 스캔 → data/meta/archive_index.csv, archive_summary.csv

    uv run python -m tools.scan_archive D:/동아리아카이브

파일마다 확장자, 용량, (nwc면) 버전과 제목, 파일명·곡 정보에 남은 유튜브 링크,
파트별 연습 음원 여부를 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from acappella.nwc.reader import parse_line, unescape_text
from tools.nwcfile import read_nwc

YOUTUBE_RE = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]{11}[^\s\"']*)"
)
YTDLP_ID_RE = re.compile(r"\[([\w-]{11})\]")  # yt-dlp 기본 파일 이름 "제목 [영상ID].확장자"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".mid", ".midi"}
PART_RE = re.compile(
    r"(?i)(soprano|alto|tenor|bass|baritone|lead|part|^[SATB]\d?[ _.-]|[ _.-][SATB]\d?[ _.-]|"
    r"소프라노|소프|알토|테너|베이스|바리톤|리드|파트|파트별)"
)


def song_info(text: str) -> dict[str, str]:
    for line in text.splitlines():
        if line.startswith("|SongInfo"):
            _, props = parse_line(line.strip())
            return {k: unescape_text(v) for k, v in props.items()}
    return {}


def scan(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        row = {
            "path": str(p.relative_to(root)),
            "ext": ext or "(없음)",
            "size_bytes": p.stat().st_size,
            "nwc_version": "",
            "has_nwctxt": "",
            "title": "",
            "youtube_links": " ".join(
                YOUTUBE_RE.findall(p.name)
                + [f"https://www.youtube.com/watch?v={v}" for v in YTDLP_ID_RE.findall(p.stem)]
            ),
            "part_practice_audio": "",
        }
        if ext in (".nwc", ".nwctxt"):
            content = read_nwc(p)
            row["nwc_version"] = content.version
            row["has_nwctxt"] = "Y" if content.text else "N"
            if content.text:
                info = song_info(content.text)
                row["title"] = info.get("Title", "")
                found = YOUTUBE_RE.findall(" ".join(info.values()))
                row["youtube_links"] = " ".join(dict.fromkeys(row["youtube_links"].split() + found))
        if ext in AUDIO_EXTS and PART_RE.search(p.stem):
            row["part_practice_audio"] = "Y"
        if not row["title"]:
            row["title"] = p.stem
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="동아리 아카이브 스캔")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/meta"))
    args = ap.parse_args(argv)
    rows = scan(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "archive_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    count, size = Counter(), defaultdict(int)
    for r in rows:
        count[r["ext"]] += 1
        size[r["ext"]] += int(r["size_bytes"])
    with open(args.out_dir / "archive_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ext", "files", "megabytes"])
        for ext, n in count.most_common():
            w.writerow([ext, n, round(size[ext] / 1e6, 1)])
    versions = Counter(r["nwc_version"] or "미상" for r in rows if r["ext"] == ".nwc")
    print(f"파일 {len(rows)}개, {sum(size.values()) / 1e9:.2f}GB")
    for ext, n in count.most_common(10):
        print(f"  {ext}: {n}개, {size[ext] / 1e6:.1f}MB")
    print("NWC 버전:", dict(versions))
    print("파트별 연습 음원:", sum(r["part_practice_audio"] == "Y" for r in rows))
    print("유튜브 링크가 있는 파일:", sum(bool(r["youtube_links"]) for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
