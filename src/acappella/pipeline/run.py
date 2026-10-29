"""transcribe() 오케스트레이션과 CLI.

acappella transcribe <URL 또는 파일> --voices Lead,S,A,T,B
acappella ui
acappella update
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from acappella.config import Settings, load_settings, parse_voices, validate_voices
from acappella.midi_export import write_midi
from acappella.nwc.writer import write_nwctxt
from acappella.pipeline.ingest import IngestError, ingest
from acappella.pipeline.lyrics import default_lyric_voice, place_lyrics
from acappella.pipeline.notes import segment_notes
from acappella.pipeline.pitch import estimate_pitch
from acappella.pipeline.postprocess import postprocess
from acappella.pipeline.preprocess import preprocess
from acappella.pipeline.rhythm import quantize
from acappella.pipeline.voices import assign_voices
from acappella.score_utils import check_invariants, range_warnings
from acappella.types import Score, VoiceSpec

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]

STAGES = [
    ("입력 수집", 0.00),
    ("전처리", 0.10),
    ("다중 피치 추정", 0.15),
    ("성부 배정", 0.55),
    ("노트 분할", 0.75),
    ("박자 추적·양자화", 0.80),
    ("후처리", 0.90),
    ("악보 저장", 0.95),
]


@dataclass
class TranscribeOptions:
    title: str | None = None
    tempo_bpm: float | None = None
    time_sig: tuple[int, int] | None = None
    key_fifths: int | None = None
    lyrics: str | None = None
    lyric_voice: str | None = None
    voice_method: str = "auto"  # rule | onnx | auto
    pitch_backend: str = "auto"  # basic_pitch | cqt | auto
    export_midi: bool = False
    save_debug: bool = True


@dataclass
class TranscribeResult:
    nwctxt: Path
    midi: list[Path] = field(default_factory=list)
    score: Score | None = None
    job_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def safe_filename(name: str, default: str = "score") -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return (name or default)[:80]


def make_job_id(source: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:6]
    return f"{stamp}-{digest}"


def score_to_dict(score: Score) -> dict:
    def q(x):
        return {
            "start_beat": str(x.start_beat),
            "dur_beats": str(x.dur_beats),
            "midi": x.midi,
            "lyric": x.lyric,
        }

    return {
        "title": score.title,
        "tempo_bpm": score.tempo_bpm,
        "time_sig": list(score.time_sig),
        "key_fifths": score.key_fifths,
        "voices": [asdict(v) for v in score.voices],
        "parts": {k: [q(x) for x in v] for k, v in score.parts.items()},
    }


def run_pipeline(
    source: str,
    voices: list[VoiceSpec],
    out_dir: Path,
    progress: Progress | None = None,
    options: TranscribeOptions | None = None,
    settings: Settings | None = None,
) -> TranscribeResult:
    opts = options or TranscribeOptions()
    settings = settings or load_settings()
    th = settings.thresholds
    validate_voices(voices)
    out_dir = Path(out_dir)
    job_dir = settings.outputs_dir / make_job_id(source)
    job_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def step(i: int) -> None:
        name, frac = STAGES[i]
        log.info("[%d/8] %s", i + 1, name)
        if progress:
            progress(name, frac)

    def dump(name: str, obj) -> None:
        if not opts.save_debug:
            return
        path = job_dir / name
        if isinstance(obj, dict) and all(isinstance(v, np.ndarray) for v in obj.values()):
            np.savez_compressed(path, **obj)
        else:
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        step(0)
        clip = ingest(source, job_dir, settings)
        if opts.title:
            clip.title = opts.title
        step(1)
        clip = preprocess(clip, th)
        step(2)
        act = estimate_pitch(clip, opts.pitch_backend)
        dump(
            "3_pitch.npz",
            {"times": act.times, "midi_bins": act.midi_bins, "salience": act.salience},
        )
        step(3)
        tracks = assign_voices(act, voices, opts.voice_method, settings.voice_model_path, th)
        dump(
            "4_voices.npz",
            {"times": tracks[0].times if tracks else np.zeros(0)}
            | {t.voice.name: t.f0_midi for t in tracks},
        )
        step(4)
        notes = segment_notes(tracks, act, th)
        dump("5_notes.json", {k: [asdict(n) for n in v] for k, v in notes.items()})
        step(5)
        score = quantize(notes, clip, voices, opts.tempo_bpm, opts.time_sig, th)
        dump("6_quantized.json", score_to_dict(score))
        step(6)
        score = postprocess(score, th, opts.key_fifths)
        if opts.lyrics and opts.lyrics.strip():
            score = place_lyrics(score, opts.lyric_voice or default_lyric_voice(score), opts.lyrics)
        dump("7_score.json", score_to_dict(score))
        problems = check_invariants(score, th.range_warn_margin)
        if problems:
            raise RuntimeError("불변 조건 위반: " + "; ".join(problems))
        step(7)
        base = safe_filename(score.title or Path(source).stem)
        nwc_path = write_nwctxt(score, out_dir / f"{base}.nwctxt")
        if opts.save_debug:
            shutil.copy2(nwc_path, job_dir / "8_score.nwctxt")
        midi_paths = []
        if opts.export_midi:
            midi_paths.append(write_midi(score, out_dir / f"{base}.mid"))
            for v in score.voices:
                midi_paths.append(
                    write_midi(score, out_dir / f"{base}_{safe_filename(v.name)}.mid", v.name)
                )
    finally:
        if not settings.keep_audio:
            for p in list(job_dir.glob("download.*")) + [job_dir / "input.wav"]:
                p.unlink(missing_ok=True)
    seconds = time.perf_counter() - t0
    if progress:
        progress("완료", 1.0)
    log.info("완료: %s (%.1f초)", nwc_path, seconds)

    return TranscribeResult(
        nwctxt=nwc_path,
        midi=midi_paths,
        score=score,
        job_dir=job_dir,
        warnings=range_warnings(score, th.range_warn_margin),
        seconds=seconds,
    )


def transcribe(
    source: str,
    voices: list[VoiceSpec],
    out_dir: Path,
    progress: Progress | None = None,
    options: TranscribeOptions | None = None,
    settings: Settings | None = None,
) -> Path:
    """URL 또는 음원 파일을 성부별 nwctxt 초안으로 만든다. 반환: nwctxt 경로."""
    return run_pipeline(source, voices, out_dir, progress, options, settings).nwctxt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_time_sig(s: str | None) -> tuple[int, int] | None:
    if not s or s == "auto":
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", s.strip())
    if not m or int(m.group(2)) not in (1, 2, 4, 8, 16):
        raise argparse.ArgumentTypeError(f"박자표 형식이 잘못됐습니다: {s} (예: 4/4)")
    return int(m.group(1)), int(m.group(2))


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acappella", description="아카펠라 자동 채보")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="URL 또는 음원 파일을 nwctxt로 채보")
    t.add_argument("source", help="유튜브 URL 또는 음원 파일(mp3, wav, m4a)")
    t.add_argument(
        "--voices",
        default="SATB",
        help="성부 목록(예: Lead,S,A,T,B) 또는 프리셋(SATB, SSATB, Lead+SATB, Lead+SSATTB). "
        "성부마다 이름:음자리표:하한:상한으로 바꿀 수 있다(예: T:bass:C3:G4)",
    )
    t.add_argument("--out", type=Path, default=Path("."), help="nwctxt를 저장할 폴더")
    t.add_argument("--title")
    t.add_argument("--tempo", type=float, help="템포(BPM). 생략하면 자동 추정")
    t.add_argument("--time-sig", type=_parse_time_sig, help="박자표(예: 4/4). 생략하면 자동 추정")
    t.add_argument("--key", type=int, help="조표(-7~7, 샵 개수는 양수). 생략하면 자동 추정")
    t.add_argument("--lyrics", help="리드 성부 가사(v2)")
    t.add_argument("--lyrics-file", type=Path, help="가사 텍스트 파일(UTF-8)")
    t.add_argument("--lyric-voice", help="가사를 넣을 성부(기본: Lead 또는 가장 높은 성부)")
    t.add_argument("--voice-method", choices=["auto", "rule", "onnx"], default="auto")
    t.add_argument("--pitch-backend", choices=["auto", "basic_pitch", "cqt"], default="auto")
    t.add_argument("--midi", action="store_true", help="검토용 성부별 MIDI도 저장")
    t.add_argument("--keep-audio", action="store_true", help="다운로드·변환한 음원을 지우지 않음")
    t.add_argument("--no-debug", action="store_true", help="단계별 중간 결과를 저장하지 않음")
    t.add_argument("-v", "--verbose", action="store_true")

    u = sub.add_parser("ui", help="로컬 앱(브라우저) 실행")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--no-browser", action="store_true")

    sub.add_parser("update", help="yt-dlp 업데이트와 새 버전 확인")
    m = sub.add_parser("models", help="성부 배정 모델(ONNX) 내려받기")
    m.add_argument("--tag", default="latest", help="GitHub Releases 태그")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    if args.cmd == "ui":
        from acappella.app.ui import launch

        launch(port=args.port, open_browser=not args.no_browser)
        return 0
    if args.cmd == "update":
        from acappella.app.updates import check_app_update, update_ytdlp

        print(update_ytdlp(load_settings()))
        print(check_app_update(load_settings()) or "최신 버전입니다.")
        return 0
    if args.cmd == "models":
        from acappella.app.updates import download_model

        print(download_model(load_settings(), args.tag))
        return 0

    settings = load_settings()
    if args.keep_audio:
        settings.keep_audio = True
    lyrics = args.lyrics
    if args.lyrics_file:
        lyrics = args.lyrics_file.read_text(encoding="utf-8")
    opts = TranscribeOptions(
        title=args.title,
        tempo_bpm=args.tempo,
        time_sig=args.time_sig,
        key_fifths=args.key,
        lyrics=lyrics,
        lyric_voice=args.lyric_voice,
        voice_method=args.voice_method,
        pitch_backend=args.pitch_backend,
        export_midi=args.midi,
        save_debug=not args.no_debug,
    )
    try:
        voices = parse_voices(args.voices)
        res = run_pipeline(args.source, voices, args.out, None, opts, settings)
    except (IngestError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    for w in res.warnings:
        print(f"경고: {w}", file=sys.stderr)
    print(res.nwctxt)
    for p in res.midi:
        print(p)
    return 0


__all__ = ["TranscribeOptions", "TranscribeResult", "main", "run_pipeline", "transcribe"]

if __name__ == "__main__":
    raise SystemExit(main())
