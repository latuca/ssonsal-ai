"""Gradio 로컬 앱. 127.0.0.1에서 실행하고 브라우저를 자동으로 연다. 외부 서버·로그인 없음.

실행: ``acappella-app`` 또는 ``acappella ui``
"""

from __future__ import annotations

import logging
import threading
import traceback
from logging.handlers import RotatingFileHandler

from acappella import __version__
from acappella.config import (
    CLEFS,
    PRESETS,
    Settings,
    default_voice,
    load_settings,
    midi_to_note,
    note_to_midi,
    validate_voices,
)
from acappella.pipeline.ingest import IngestError
from acappella.pipeline.run import TranscribeOptions, run_pipeline, setup_logging
from acappella.types import VoiceSpec

log = logging.getLogger(__name__)

HEADERS = ["성부", "음자리표", "음역 하한", "음역 상한"]
TIME_SIGS = ["자동", "4/4", "3/4", "2/4", "6/8", "2/2", "12/8"]
KEYS = {
    "자동": None,
    "C / Am (조표 없음)": 0,
    "G / Em (#1)": 1,
    "D / Bm (#2)": 2,
    "A / F#m (#3)": 3,
    "E / C#m (#4)": 4,
    "B / G#m (#5)": 5,
    "F# / D#m (#6)": 6,
    "F / Dm (b1)": -1,
    "Bb / Gm (b2)": -2,
    "Eb / Cm (b3)": -3,
    "Ab / Fm (b4)": -4,
    "Db / Bbm (b5)": -5,
    "Gb / Ebm (b6)": -6,
}
CUSTOM = "직접 입력"


def preset_rows(preset: str) -> list[list[str]]:
    names = PRESETS.get(preset, PRESETS["SATB"])
    return [voice_row(default_voice(n)) for n in names]


def voice_row(v: VoiceSpec) -> list[str]:
    return [v.name, v.clef, midi_to_note(v.low_midi), midi_to_note(v.high_midi)]


def rows_to_voices(rows) -> list[VoiceSpec]:
    """Dataframe 값(pandas DataFrame 또는 리스트) → VoiceSpec 목록. 빈 칸은 이름 기본값으로 채운다."""
    if hasattr(rows, "values") and hasattr(rows, "columns"):
        rows = rows.values.tolist()
    voices = []
    for row in rows or []:
        cells = [("" if c is None else str(c)).strip() for c in list(row) + [""] * 4]
        name, clef, low, high = cells[:4]
        if not name or name.lower() == "nan":
            continue
        v = default_voice(name)
        if clef and clef.lower() != "nan":
            if clef not in CLEFS:
                raise ValueError(f"{name}: 음자리표는 {', '.join(CLEFS)} 중 하나로 적어 주세요.")
            v.clef = clef
        if low and low.lower() != "nan":
            v.low_midi = note_to_midi(low)
        if high and high.lower() != "nan":
            v.high_midi = note_to_midi(high)
        voices.append(v)
    validate_voices(voices)
    return voices


def _parse_time_sig(s: str) -> tuple[int, int] | None:
    if not s or s == "자동":
        return None
    a, b = s.split("/")
    return int(a), int(b)


def run_job(
    url: str,
    file_path: str | None,
    rows,
    title: str,
    time_sig: str,
    tempo: float | None,
    key: str,
    lyrics: str,
    lyric_voice: str,
    use_model: bool,
    want_midi: bool,
    settings: Settings,
    progress=None,
):
    """UI 버튼 처리. 반환: (상태 문구, nwctxt 경로 또는 None, MIDI 경로 목록 또는 None)."""
    source = (url or "").strip() or (file_path or "")
    if not source:
        return "유튜브 URL을 넣거나 음원 파일을 올려 주세요.", None, None
    try:
        voices = rows_to_voices(rows)
    except ValueError as e:
        return f"성부 구성을 확인해 주세요: {e}", None, None

    def on_progress(stage: str, frac: float) -> None:
        if progress is not None:
            progress(frac, desc=stage)

    opts = TranscribeOptions(
        title=(title or "").strip() or None,
        tempo_bpm=float(tempo) if tempo else None,
        time_sig=_parse_time_sig(time_sig),
        key_fifths=KEYS.get(key),
        lyrics=(lyrics or "").strip() or None,
        lyric_voice=(lyric_voice or "").strip() or None,
        voice_method="auto" if use_model else "rule",
        export_midi=want_midi,
    )
    out_dir = settings.home / "scores"
    try:
        res = run_pipeline(source, voices, out_dir, on_progress, opts, settings)
    except IngestError as e:
        return str(e), None, None
    except Exception as e:  # noqa: BLE001 - 동아리원에게는 다음 행동을 알려준다
        log.error("처리 실패: %s\n%s", e, traceback.format_exc())
        return (
            f"처리 중 오류가 났습니다: {e}\n\n"
            f"같은 문제가 반복되면 로그 파일({settings.home / 'logs' / 'app.log'})을 개발자에게 보내 주세요.",
            None,
            None,
        )
    msg = [
        f"완료했습니다. ({res.seconds:.0f}초)",
        f"템포 {res.score.tempo_bpm:g} BPM, 박자 {res.score.time_sig[0]}/{res.score.time_sig[1]}",
    ]
    if res.warnings:
        msg.append(f"음역을 벗어난 음 {len(res.warnings)}개가 있습니다. NWC에서 확인해 주세요.")
    return "\n\n".join(msg), str(res.nwctxt), [str(p) for p in res.midi] or None


def build_app(settings: Settings | None = None):
    import gradio as gr

    settings = settings or load_settings()
    model_ready = settings.voice_model_path.exists()

    with gr.Blocks(title="아카펠라 자동 채보", analytics_enabled=False) as demo:
        gr.Markdown(
            f"## 아카펠라 자동 채보 <small>v{__version__}</small>\n"
            "유튜브 주소나 음원 파일과 성부 구성을 넣으면 성부별 NWC 악보 초안(.nwctxt)을 만듭니다. "
            "모든 처리는 이 컴퓨터 안에서 끝납니다."
        )
        notice = gr.Markdown(visible=False)
        with gr.Row():
            with gr.Column():
                url = gr.Textbox(
                    label="유튜브 URL", placeholder="https://www.youtube.com/watch?v=..."
                )
                audio = gr.File(
                    label="또는 음원 파일 (mp3, wav, m4a)",
                    file_types=[".mp3", ".wav", ".m4a"],
                    type="filepath",
                )
                title = gr.Textbox(label="곡 제목 (비우면 영상·파일 이름)")
                with gr.Row():
                    time_sig = gr.Dropdown(TIME_SIGS, value="자동", label="박자표")
                    tempo = gr.Number(label="템포 BPM (0 = 자동)", value=0, minimum=0, maximum=300)
                    key = gr.Dropdown(list(KEYS), value="자동", label="조표")
            with gr.Column():
                preset = gr.Dropdown(list(PRESETS) + [CUSTOM], value="SATB", label="성부 프리셋")
                table = gr.Dataframe(
                    headers=HEADERS,
                    value=preset_rows("SATB"),
                    datatype=["str", "str", "str", "str"],
                    col_count=(4, "fixed"),
                    row_count=(4, "dynamic"),
                    interactive=True,
                    label="성부 구성 (행 추가·수정 가능. 음자리표: treble / treble8vb / bass, 음역 예: C4)",
                )
                with gr.Accordion("가사 (v2, 선택)", open=False):
                    lyrics = gr.Textbox(
                        label="리드 성부 가사",
                        lines=4,
                        placeholder="한국어는 글자마다, 영어는 hel-lo처럼 하이픈으로 음절을 나눌 수 있습니다. "
                        "_ 는 음표 하나를 건너뜁니다.",
                    )
                    lyric_voice = gr.Textbox(
                        label="가사를 넣을 성부 (비우면 Lead 또는 가장 높은 성부)"
                    )
                use_model = gr.Checkbox(
                    value=model_ready,
                    interactive=model_ready,
                    label="학습된 성부 배정 모델 사용"
                    + ("" if model_ready else " (모델 파일 없음: 규칙 기반으로 처리)"),
                )
                want_midi = gr.Checkbox(value=False, label="검토용 성부별 MIDI도 만들기")
        run_btn = gr.Button("채보 시작", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            nwc_file = gr.File(label="NWC 악보 (.nwctxt)")
            midi_files = gr.File(label="검토용 MIDI", file_count="multiple")

        preset.change(
            lambda p: preset_rows(p) if p != CUSTOM else [["", "", "", ""]],
            inputs=preset,
            outputs=table,
        )

        def _run(u, f, rows, ti, ts, te, k, ly, lv, um, wm, progress=gr.Progress()):  # noqa: B008 - Gradio 관용
            return run_job(u, f, rows, ti, ts, te, k, ly, lv, um, wm, settings, progress)

        run_btn.click(
            _run,
            inputs=[
                url,
                audio,
                table,
                title,
                time_sig,
                tempo,
                key,
                lyrics,
                lyric_voice,
                use_model,
                want_midi,
            ],
            outputs=[status, nwc_file, midi_files],
            concurrency_limit=1,
        )

        def _check_update():
            if not settings.check_app_update:
                return gr.update(visible=False)
            from acappella.app.updates import check_app_update

            msg = check_app_update(settings)
            return gr.update(value=f"**{msg}**", visible=True) if msg else gr.update(visible=False)

        demo.load(_check_update, outputs=notice)
    return demo


def _file_logging(settings: Settings) -> None:
    log_dir = settings.home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def launch(port: int = 7860, open_browser: bool = True, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    settings.home.mkdir(parents=True, exist_ok=True)
    _file_logging(settings)
    if settings.auto_update_ytdlp:
        from acappella.app.updates import update_ytdlp

        # 업데이트는 뒤에서 한다. 실패해도 파일 입력은 동작한다.
        threading.Thread(target=lambda: log.info(update_ytdlp(settings)), daemon=True).start()
    demo = build_app(settings)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=open_browser,
        share=False,
        show_api=False,
        allowed_paths=[str(settings.home)],
    )


def main() -> None:
    setup_logging()
    launch()


if __name__ == "__main__":
    main()


__all__ = ["build_app", "launch", "main", "rows_to_voices", "run_job"]
