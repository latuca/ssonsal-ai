"""로컬 앱(UI 처리 함수)과 업데이트 확인 테스트. 네트워크는 쓰지 않는다."""

import io
import json

import pytest

from acappella.app import updates
from acappella.app.ui import build_app, preset_rows, rows_to_voices, run_job
from acappella.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path / "home")


def test_preset_rows_and_parsing():
    rows = preset_rows("Lead+SATB")
    assert [r[0] for r in rows] == ["Lead", "S", "A", "T", "B"]
    voices = rows_to_voices(rows)
    assert voices[3].clef == "treble8vb" and voices[4].clef == "bass"
    custom = rows_to_voices([["Bar", "bass", "A2", "E4"], ["", "", "", ""]])
    assert len(custom) == 1 and (custom[0].low_midi, custom[0].high_midi) == (45, 64)
    with pytest.raises(ValueError):
        rows_to_voices([["S", "soprano", "C4", "A5"]])


def _job(settings, url="", file=None):
    return run_job(
        url, file, preset_rows("SATB"), "", "자동", 0, "자동", "", "", False, False, settings
    )


def test_run_job_requires_input(settings):
    msg, nwc, _ = _job(settings)
    assert "URL" in msg and nwc is None


def test_run_job_friendly_url_error(settings, tmp_path):
    settings.ytdlp_path = str(tmp_path / "missing-yt-dlp")
    msg, nwc, _ = _job(settings, url="https://youtu.be/x")
    assert "음원 파일로 시도해 주세요" in msg and nwc is None


def test_build_app(settings):
    assert build_app(settings) is not None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_app_update(monkeypatch, settings):
    def respond(payload):
        return lambda req, timeout=0: _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(
        updates.urllib.request, "urlopen", respond({"tag_name": "v9.0.0", "html_url": "u"})
    )
    assert "v9.0.0" in updates.check_app_update(settings)
    monkeypatch.setattr(updates.urllib.request, "urlopen", respond({"tag_name": "v0.0.1"}))
    assert updates.check_app_update(settings) is None

    def offline(req, timeout=0):
        raise updates.urllib.error.URLError("offline")

    monkeypatch.setattr(updates.urllib.request, "urlopen", offline)
    assert updates.check_app_update(settings) is None


def test_download_model(monkeypatch, settings):
    release = {
        "tag_name": "v0.2.0",
        "assets": [{"name": "voice_assign.onnx", "url": "https://api/asset/1"}],
    }

    def fake(req, timeout=0):
        if req.full_url.endswith("/latest"):
            return _Resp(json.dumps(release).encode())
        assert req.get_header("Accept") == "application/octet-stream"
        return _Resp(b"ONNXDATA")

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake)
    path = updates.download_model(settings)
    assert path.read_bytes() == b"ONNXDATA" and path == settings.voice_model_path


def test_update_ytdlp_failure_is_soft(settings, tmp_path):
    settings.ytdlp_path = str(tmp_path / "missing")
    assert "실패" in updates.update_ytdlp(settings)
