"""층 B(silver): 곡 제목으로 유튜브 후보를 찾아 악보와 자동 매칭한다.

    uv run python -m tools.match_youtube --auto-threshold 0.25 --review-threshold 0.4

곡마다(메타데이터 youtube_url이 없거나 --all):
1) yt-dlp ``ytsearchN:<제목>``으로 후보 영상 N개를 찾는다(다운로드 없이 목록만).
2) 후보 오디오를 내려받아 tools/align.py로 정렬하고 조옮김을 감지한다.
3) 비용 < auto-threshold → tier B 자동 승인(메타데이터·정렬 CSV 저장)
   auto ≤ 비용 < review-threshold → data/meta/review_queue.csv(사람 검토 목록)
   그 외 → 버림
임계값은 gold 곡(tier C)의 정렬 비용 분포로 정한다(--calibrate).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from acappella.config import load_settings
from acappella.meta import SongMeta, load_all, save_meta
from acappella.nwc.reader import read_nwctxt
from acappella.pipeline.ingest import ingest
from tools.align import align, write_csv

log = logging.getLogger(__name__)


def search_candidates(title: str, n: int, settings) -> list[dict]:
    cmd = settings.ytdlp_cmd() + ["--flat-playlist", "--dump-json", f"ytsearch{n}:{title}"]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    out = []
    for line in proc.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = d.get("id")
        if vid:
            out.append(
                {
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": d.get("title", ""),
                    "channel": d.get("channel") or d.get("uploader") or "",
                    "duration": d.get("duration"),
                }
            )
    return out


def match_song(meta: SongMeta, data_dir: Path, settings, n: int) -> list[dict]:
    score = read_nwctxt(data_dir / "nwctxt" / f"{meta.song_id}.nwctxt")
    results = []
    for cand in search_candidates(meta.title or score.title, n, settings):
        try:
            clip = ingest(cand["url"], data_dir / "cache" / "yt" / meta.song_id, settings)
            res = align(score, clip.samples, clip.sr)
        except Exception as e:  # noqa: BLE001 - 후보 하나 실패는 건너뛴다
            log.warning("%s 후보 실패(%s): %s", meta.song_id, cand["url"], e)
            continue
        results.append(
            {**cand, "cost": res.cost, "transpose": res.transpose_semitones, "align": res}
        )
    return sorted(results, key=lambda r: r["cost"])


def calibrate(metas: list[SongMeta], data_dir: Path, settings) -> None:
    """gold 곡의 정렬 비용 분포를 출력한다. 자동 승인 임계값은 gold 비용의 상위 분위로 잡는다."""
    costs = []
    for m in metas:
        if m.tier != "C" or not m.youtube_url:
            continue
        clip = ingest(m.youtube_url, data_dir / "cache" / "yt" / m.song_id, settings)
        res = align(read_nwctxt(data_dir / "nwctxt" / f"{m.song_id}.nwctxt"), clip.samples, clip.sr)
        costs.append(res.cost)
        print(f"{m.song_id}: 비용 {res.cost:.3f}, 조옮김 {res.transpose_semitones:+d}")
    if costs:
        print(
            f"gold {len(costs)}곡: 중앙값 {np.median(costs):.3f}, 90% {np.percentile(costs, 90):.3f}, "
            f"최대 {max(costs):.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="유튜브 자동 매칭(silver)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--auto-threshold", type=float, default=0.25)
    ap.add_argument("--review-threshold", type=float, default=0.40)
    ap.add_argument("--all", action="store_true", help="이미 링크가 있는 곡도 다시 찾기")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--calibrate", action="store_true", help="gold 곡으로 비용 분포만 출력")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    meta_dir = args.data_dir / "meta" / "songs"
    metas = load_all(meta_dir)
    if args.calibrate:
        calibrate(metas, args.data_dir, settings)
        return 0
    todo = [m for m in metas if m.tier == "A" and (args.all or not m.youtube_url)][: args.limit]
    review_path = args.data_dir / "meta" / "review_queue.csv"
    new_file = not review_path.exists()
    approved = queued = 0
    with open(review_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["song_id", "title", "url", "video_title", "channel", "cost", "transpose"])
        for m in todo:
            results = match_song(m, args.data_dir, settings, args.candidates)
            if not results:
                continue
            best = results[0]
            if best["cost"] < args.auto_threshold:
                m.youtube_url, m.tier = best["url"], "B"
                m.transpose_semitones, m.align_cost = best["transpose"], round(best["cost"], 4)
                m.group = m.group or best["channel"]
                save_meta(m, meta_dir / f"{m.song_id}.json")
                write_csv(best["align"], args.data_dir / "align" / f"{m.song_id}.csv")
                approved += 1
            elif best["cost"] < args.review_threshold:
                w.writerow(
                    [
                        m.song_id,
                        m.title,
                        best["url"],
                        best["title"],
                        best["channel"],
                        f"{best['cost']:.4f}",
                        best["transpose"],
                    ]
                )
                queued += 1
    print(f"자동 승인 {approved}곡, 검토 목록 {queued}곡 ({review_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
