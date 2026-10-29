"""아카이브의 .nwc/.nwctxt → data/nwctxt/<song_id>.nwctxt

    uv run python -m tools.extract_nwctxt D:/동아리아카이브

- 곡 ID는 data/meta/id_map.csv에 고정한다(파일 경로 → s0001...). 다시 돌려도 ID가 바뀌지 않는다.
- 추출한 텍스트는 reader로 읽어 보고, 음표가 있는 스태프가 하나 이상일 때만 성공으로 친다.
- 실패한 파일은 data/meta/extract_failures.csv에 이유와 함께 남긴다. 이 파일들은 NWC에서
  일괄 내보내기(File > Save As > nwctxt) 등 다른 방법을 따로 찾는다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from acappella.nwc.reader import parse_nwctxt
from tools.nwcfile import read_nwc


def load_id_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["path"]: r["song_id"] for r in csv.DictReader(f)}


def save_id_map(path: Path, id_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "song_id"])
        for p, sid in sorted(id_map.items(), key=lambda x: x[1]):
            w.writerow([p, sid])


def assign_ids(paths: list[str], id_map: dict[str, str]) -> dict[str, str]:
    used = {int(s[1:]) for s in id_map.values() if s[1:].isdigit()}
    nxt = max(used, default=0) + 1
    for p in sorted(paths):
        if p not in id_map:
            id_map[p] = f"s{nxt:04d}"
            nxt += 1
    return id_map


def extract(root: Path, data_dir: Path) -> tuple[int, list[dict]]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".nwc", ".nwctxt"))
    rel = [str(p.relative_to(root)) for p in files]
    id_map_path = data_dir / "meta" / "id_map.csv"
    id_map = assign_ids(rel, load_id_map(id_map_path))
    save_id_map(id_map_path, id_map)
    out_dir = data_dir / "nwctxt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, failures = 0, []
    for p, r in zip(files, rel):
        content = read_nwc(p)
        reason = content.error
        if content.text is not None:
            try:
                score = parse_nwctxt(content.text).score
                if not any(any(q.midi is not None for q in part) for part in score.parts.values()):
                    reason = "음표가 있는 스태프가 없습니다"
            except Exception as e:  # noqa: BLE001
                reason = f"nwctxt 해석 실패: {e}"
        if content.text is None or reason:
            failures.append(
                {
                    "path": r,
                    "song_id": id_map[r],
                    "version": content.version,
                    "reason": reason or "알 수 없음",
                }
            )
            continue
        (out_dir / f"{id_map[r]}.nwctxt").write_text(content.text, encoding="utf-8")
        ok += 1
    with open(
        data_dir / "meta" / "extract_failures.csv", "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=["path", "song_id", "version", "reason"])
        w.writeheader()
        w.writerows(failures)
    return ok, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=".nwc에서 nwctxt 추출")
    ap.add_argument("root", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args(argv)
    ok, failures = extract(args.root, args.data_dir)
    print(f"성공 {ok}개, 실패 {len(failures)}개 (data/meta/extract_failures.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
