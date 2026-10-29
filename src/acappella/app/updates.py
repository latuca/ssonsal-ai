"""업데이트: yt-dlp 자동 업데이트, 새 앱 버전 알림, 모델 파일 내려받기(GitHub Releases).

레포가 비공개면 GitHub API가 인증 없이 404를 돌려준다. 이 경우 ``ACAPPELLA_GITHUB_TOKEN``
(읽기 전용 fine-grained 토큰)을 설정하거나 공개 배포용 레포를 ``release_repo``로 지정한다.
업데이트 확인은 실패해도 앱 동작을 막지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from acappella import __version__
from acappella.config import Settings

log = logging.getLogger(__name__)

API = "https://api.github.com/repos/{repo}/releases/{which}"


def _request(url: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "acappella"})
    if token := os.environ.get("ACAPPELLA_GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    return req


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def update_ytdlp(settings: Settings, timeout: float = 120) -> str:
    """yt-dlp를 최신으로 올린다. 실패해도 예외를 던지지 않는다(파일 입력은 계속 동작)."""
    if settings.ytdlp_path:
        cmd = [settings.ytdlp_path, "-U"]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "-q",
            "--disable-pip-version-check",
            "yt-dlp",
        ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("yt-dlp 업데이트 실패: %s", e)
        return "yt-dlp 업데이트에 실패했습니다. 유튜브 입력이 안 되면 음원 파일을 써 주세요."
    if proc.returncode != 0:
        log.warning("yt-dlp 업데이트 실패: %s", proc.stderr.strip()[-500:])
        return "yt-dlp 업데이트에 실패했습니다. 유튜브 입력이 안 되면 음원 파일을 써 주세요."
    return "yt-dlp가 최신 상태입니다."


def fetch_release(repo: str, tag: str = "latest", timeout: float = 5) -> dict | None:
    which = "latest" if tag == "latest" else f"tags/{tag}"
    try:
        with urllib.request.urlopen(
            _request(API.format(repo=repo, which=which)), timeout=timeout
        ) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        log.info("릴리스 정보를 가져오지 못했습니다(%s): %s", repo, e)
        return None


def check_app_update(settings: Settings) -> str | None:
    """새 버전이 있으면 안내 문구, 없거나 확인할 수 없으면 None."""
    rel = fetch_release(settings.release_repo)
    if not rel or "tag_name" not in rel:
        return None
    latest = rel["tag_name"]
    if _version_tuple(latest) > _version_tuple(__version__):
        url = rel.get("html_url", f"https://github.com/{settings.release_repo}/releases")
        return f"새 버전 {latest}이 있습니다(현재 {__version__}). 내려받기: {url}"
    return None


def download_model(settings: Settings, tag: str = "latest") -> Path:
    """GitHub Releases에서 성부 배정 모델(ONNX)을 models/에 내려받는다."""
    rel = fetch_release(settings.release_repo, tag, timeout=15)
    if not rel:
        raise RuntimeError(
            "릴리스 정보를 가져오지 못했습니다. 인터넷 연결이나 토큰 설정을 확인해 주세요."
        )
    asset = next((a for a in rel.get("assets", []) if a.get("name") == settings.voice_model), None)
    if asset is None:
        raise RuntimeError(
            f"릴리스 {rel.get('tag_name')}에 {settings.voice_model} 파일이 없습니다."
        )
    dst = settings.voice_model_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    req = _request(asset["url"], accept="application/octet-stream")
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    return dst
