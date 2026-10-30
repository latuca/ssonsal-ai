"""층 B(silver): 곡 제목으로 유튜브 후보를 찾아 악보와 자동 매칭한다.

    uv run python -m tools.match_youtube --auto-threshold 0.25 --review-threshold 0.4

곡마다(메타데이터 youtube_url이 없거나 --all):
1) yt-dlp ``ytsearchN:<그룹> <제목> <가수> acapella``로 후보 영상 N개를 찾는다(다운로드 없이 목록만).
   공연 실황·동아리 영상(``AMATEUR_RE``)과 너무 짧거나 긴 영상은 뺀다. 프로팀 음원만 쓰기 위해서다.
2) 후보 오디오를 임시로 내려받아 tools/align.py로 정렬하고 조옮김을 감지한다.
   음원은 곡 처리가 끝나면 지운다. 남는 것은 URL·정렬 CSV와, 승인된 곡의 Basic Pitch contour
   (data/cache/salience/<song_id>.npz, training/real_data.py가 음원 대신 쓴다)뿐이다.
   후보 전체는 data/meta/match_log.csv에 남긴다.
3) 비용 < auto-threshold이고 margin ≥ auto-margin → tier B 자동 승인(메타데이터·정렬 CSV 저장)
   비용 < review-threshold이고 margin ≥ review-margin → data/meta/review_queue.csv(사람 검토 목록)
   그 외 → 버림. 기본값은 꽃이 피고 지듯이(맞는 짝: 비용 0.134, margin 0.41)와
   다른 곡 짝(비용 0.43, margin 0.10)으로 잡았다.
임계값은 gold 곡(tier C)의 정렬 비용 분포로 정한다(--calibrate).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from acappella.config import load_settings
from acappella.meta import SongMeta, load_all, save_meta
from acappella.nwc.reader import read_nwctxt
from acappella.pipeline.ingest import ingest
from tools.align import align, write_csv

log = logging.getLogger(__name__)

AMATEUR_RE = re.compile(
    r"(?i)(정기\s*공연|동아리|대학교|대학|univ|고등학교|중학교|합창단|교회|성가대|축제|버스킹|busking|"
    r"공연\s*실황|직캠|fancam|ssugar|슈가|tutorial|강좌|piano|피아노|karaoke|노래방|reaction|리액션)"
)


def search_query(meta: SongMeta, title: str) -> str:
    parts = [
        meta.extra.get("pro_group", ""),
        meta.extra.get("file_title") or title,
        meta.extra.get("artist", ""),
    ]
    return " ".join(p for p in parts if p) + " acapella"


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


def score_seconds(score) -> float:
    end = max((q.start_beat + q.dur_beats for p in score.parts.values() for q in p), default=0)
    return float(end) * 60.0 / score.tempo_bpm


def usable(cand: dict, score_sec: float) -> bool:
    if AMATEUR_RE.search(f"{cand['title']} {cand['channel']}"):
        return False
    dur = cand.get("duration")
    return dur is None or 0.5 * score_sec <= dur <= max(2.0 * score_sec, score_sec + 120)


def match_song(meta: SongMeta, data_dir: Path, settings, n: int) -> list[dict]:
    """후보마다 정렬 비용을 매긴다. 가장 좋은 후보는 ``clip``(음원)을 함께 돌려준다."""
    score = read_nwctxt(data_dir / "nwctxt" / f"{meta.song_id}.nwctxt")
    sec = score_seconds(score)
    tmp = data_dir / "cache" / "yt" / meta.song_id
    results, best_clip = [], None
    try:
        for i, cand in enumerate(search_candidates(search_query(meta, score.title), n, settings)):
            if not usable(cand, sec):
                results.append({**cand, "cost": None, "transpose": None, "note": "제외"})
                continue
            try:
                clip = ingest(cand["url"], tmp / str(i), settings)
                res = align(score, clip.samples, clip.sr)
            except Exception as e:  # noqa: BLE001 - 후보 하나 실패는 건너뛴다
                log.warning("%s 후보 실패(%s): %s", meta.song_id, cand["url"], e)
                continue
            finally:
                shutil.rmtree(tmp / str(i), ignore_errors=True)
            if not np.isfinite(res.cost):
                results.append({**cand, "cost": None, "transpose": None, "note": "길이 안 맞음"})
                continue
            results.append(
                {
                    **cand,
                    "cost": res.cost,
                    "margin": res.margin,
                    "transpose": res.transpose_semitones,
                    "align": res,
                }
            )
            if best_clip is None or res.cost < best_clip[0]:
                best_clip = (res.cost, clip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    scored = sorted((r for r in results if r["cost"] is not None), key=lambda r: r["cost"])
    if scored and best_clip is not None:
        scored[0]["clip"] = best_clip[1]
    return scored + [r for r in results if r["cost"] is None]


def save_salience(clip, path: Path) -> None:
    from acappella.pipeline.pitch import estimate_pitch
    from acappella.pipeline.preprocess import preprocess

    act = estimate_pitch(preprocess(clip), "basic_pitch")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, salience=act.salience.astype(np.float16))


def accepted(r: dict, cost: float, margin: float) -> bool:
    return r["cost"] is not None and r["cost"] < cost and r.get("margin", 0.0) >= margin


def _work(job):
    meta, data_dir, n, auto, auto_margin = job
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    try:
        results = match_song(meta, data_dir, settings, n)
    except Exception as e:  # noqa: BLE001
        log.warning("%s 매칭 실패: %s", meta.song_id, e)
        return meta, []
    if results and accepted(results[0], auto, auto_margin):
        save_salience(results[0]["clip"], data_dir / "cache" / "salience" / f"{meta.song_id}.npz")
    for r in results:
        r.pop("clip", None)
    return meta, results


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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--auto-threshold", type=float, default=0.25)
    ap.add_argument("--auto-margin", type=float, default=0.25)
    ap.add_argument("--review-threshold", type=float, default=0.35)
    ap.add_argument("--review-margin", type=float, default=0.15)
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
    log_path = args.data_dir / "meta" / "match_log.csv"
    approved = queued = 0
    new_review, new_log = not review_path.exists(), not log_path.exists()
    jobs = [
        (m, args.data_dir, args.candidates, args.auto_threshold, args.auto_margin) for m in todo
    ]
    with (
        open(review_path, "a", encoding="utf-8-sig", newline="") as f,
        open(log_path, "a", encoding="utf-8-sig", newline="") as lf,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        w, lw = csv.writer(f), csv.writer(lf)
        if new_review:
            w.writerow(
                ["song_id", "title", "url", "video_title", "channel", "cost", "transpose", "margin"]
            )
        if new_log:
            lw.writerow(
                [
                    "song_id",
                    "rank",
                    "url",
                    "video_title",
                    "channel",
                    "duration",
                    "cost",
                    "margin",
                    "transpose",
                    "note",
                ]
            )
        for done, (m, results) in enumerate(pool.map(_work, jobs), 1):
            for k, r in enumerate(results):
                cost = "" if r["cost"] is None else f"{r['cost']:.4f}"
                margin = f"{r['margin']:.4f}" if "margin" in r else ""
                lw.writerow(
                    [
                        m.song_id,
                        k,
                        r["url"],
                        r["title"],
                        r["channel"],
                        r.get("duration"),
                        cost,
                        margin,
                        r["transpose"],
                        r.get("note", ""),
                    ]
                )
            lf.flush()
            best = results[0] if results and results[0]["cost"] is not None else None
            log.info(
                "[%d/%d] %s 최저 비용 %s",
                done,
                len(jobs),
                m.song_id,
                best and f"{best['cost']:.3f}",
            )
            if best is None:
                continue
            if accepted(best, args.auto_threshold, args.auto_margin):
                m.youtube_url, m.tier = best["url"], "B"
                m.transpose_semitones, m.align_cost = best["transpose"], round(best["cost"], 4)
                m.group = m.group or best["channel"]
                m.extra["youtube_channel"] = best["channel"]
                save_meta(m, meta_dir / f"{m.song_id}.json")
                write_csv(best["align"], args.data_dir / "align" / f"{m.song_id}.csv")
                approved += 1
            elif accepted(best, args.review_threshold, args.review_margin):
                w.writerow(
                    [
                        m.song_id,
                        m.title,
                        best["url"],
                        best["title"],
                        best["channel"],
                        f"{best['cost']:.4f}",
                        best["transpose"],
                        f"{best['margin']:.4f}",
                    ]
                )
                queued += 1
            f.flush()
    print(f"자동 승인 {approved}곡, 검토 목록 {queued}곡 ({review_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
