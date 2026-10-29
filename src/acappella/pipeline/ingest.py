"""1단계 입력 수집: 유튜브 URL(yt-dlp) 또는 음원 파일 → 모노 WAV → AudioClip.

파이프라인에서 유일하게 부수 효과(다운로드, 파일 쓰기)가 있는 단계다.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from acappella.config import Settings, load_settings
from acappella.types import AudioClip

log = logging.getLogger(__name__)

SAMPLE_RATE = 22050  # Basic Pitch 입력 샘플레이트
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".mp4"}
_URL_RE = re.compile(r"^https?://", re.I)


class IngestError(RuntimeError):
    """사용자에게 그대로 보여줄 수 있는 입력 단계 오류. 다음 행동을 함께 알려준다."""


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source.strip()))


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    log.debug("실행: %s", " ".join(cmd))
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


def download_audio(url: str, workdir: Path, settings: Settings) -> tuple[Path, str]:
    """yt-dlp로 오디오만 내려받는다. (파일 경로, 영상 제목)을 돌려준다."""
    workdir.mkdir(parents=True, exist_ok=True)
    template = str(workdir / "download.%(ext)s")
    cmd = settings.ytdlp_cmd() + [
        "--no-playlist",
        "--no-progress",
        "-f",
        "bestaudio/best",
        "-o",
        template,
        "--print",
        "after_move:title",
        "--print",
        "after_move:filepath",
        url,
    ]
    try:
        proc = _run(cmd, timeout=900)
    except FileNotFoundError as e:
        raise IngestError("yt-dlp를 찾을 수 없습니다. 음원 파일로 시도해 주세요.") from e
    except subprocess.TimeoutExpired as e:
        raise IngestError("영상 다운로드가 너무 오래 걸립니다. 음원 파일로 시도해 주세요.") from e
    if proc.returncode != 0:
        log.error("yt-dlp 실패: %s", proc.stderr.strip()[-2000:])
        raise IngestError("영상을 받을 수 없습니다. 음원 파일로 시도해 주세요.")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2 or not Path(lines[-1]).exists():
        candidates = sorted(workdir.glob("download.*"))
        if not candidates:
            raise IngestError("영상을 받을 수 없습니다. 음원 파일로 시도해 주세요.")
        return candidates[0], lines[0] if lines else ""
    return Path(lines[-1]), lines[-2]


def to_mono_wav(src: Path, dst: Path, settings: Settings, sr: int = SAMPLE_RATE) -> Path:
    """ffmpeg로 모노 float WAV(sr Hz)로 변환한다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_f32le",
        str(dst),
    ]
    try:
        proc = _run(cmd, timeout=600)
    except FileNotFoundError as e:
        raise IngestError(
            "ffmpeg를 찾을 수 없습니다. 설치본을 다시 설치하거나 설정에서 ffmpeg 경로를 지정해 주세요."
        ) from e
    if proc.returncode != 0 or not dst.exists():
        log.error("ffmpeg 실패: %s", proc.stderr.strip()[-2000:])
        raise IngestError("음원 파일을 읽을 수 없습니다. mp3, wav, m4a 파일인지 확인해 주세요.")
    return dst


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32), int(sr)


def ingest(source: str, workdir: Path, settings: Settings | None = None) -> AudioClip:
    """URL이나 파일 경로를 받아 모노 22.05kHz AudioClip을 만든다.

    다운로드한 원본과 변환한 WAV는 ``workdir`` 아래에 남는다. 처리 후 지우는 것은
    :func:`acappella.pipeline.run.transcribe`가 설정(keep_audio)에 따라 한다.
    """
    settings = settings or load_settings()
    workdir = Path(workdir)
    source = source.strip().strip('"')
    if is_url(source):
        raw, title = download_audio(source, workdir, settings)
    else:
        raw = Path(source)
        if not raw.exists():
            raise IngestError(f"파일을 찾을 수 없습니다: {raw}")
        if raw.suffix.lower() not in AUDIO_EXTS:
            raise IngestError("지원하지 않는 파일 형식입니다. mp3, wav, m4a 파일을 넣어 주세요.")
        title = raw.stem
    wav = to_mono_wav(raw, workdir / "input.wav", settings)
    samples, sr = load_wav(wav)
    if samples.size < sr // 2:
        raise IngestError("음원이 너무 짧습니다(0.5초 미만). 다른 음원으로 시도해 주세요.")
    log.info("입력: %s (%.1f초, 제목 %r)", source, samples.size / sr, title)
    return AudioClip(samples=samples, sr=sr, source=source, title=title)
