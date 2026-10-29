"""학습 코드 테스트(CPU, 아주 작은 규모). PyTorch가 없으면 건너뛴다(앱 CI에는 PyTorch를 넣지 않는다)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from acappella.config import parse_voices  # noqa: E402
from acappella.meta import SongMeta, save_meta  # noqa: E402
from acappella.pipeline.voices import (  # noqa: E402
    ONNX_IN_BINS,
    ONNX_K,
    ONNX_OUT_BINS,
    ONNX_WINDOW,
    _onnx_session,
    assign_voices_onnx,
)
from training import export_onnx, synth, train  # noqa: E402
from training.dataset import VoiceDataset  # noqa: E402
from training.model import VoiceAssignNet  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "handmade_satb.nwctxt"


@pytest.fixture(scope="module")
def synth_data(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    (data / "nwctxt").mkdir()
    text = FIXTURE.read_text(encoding="utf-8")
    for sid, split in (
        ("s0001", "train"),
        ("s0002", "train"),
        ("s0003", "valid"),
        ("s0004", "test"),
    ):
        (data / "nwctxt" / f"{sid}.nwctxt").write_text(text, encoding="utf-8")
        save_meta(
            SongMeta(sid, voices=["S", "A", "T", "B"], split=split),
            data / "meta" / "songs" / f"{sid}.json",
        )
    out = data / "synth"
    n = synth.build(
        data, out, variants=2, renderer="mixed", backend="cqt", cross_mix=0.5, seed=0, limit=None
    )
    assert n == 6  # test 곡은 만들지 않는다
    return out


def test_synth_outputs(synth_data):
    files = sorted((synth_data / "train").glob("*.npz"))
    assert len(files) == 4 and not (synth_data / "test").exists()
    with np.load(files[0]) as z:
        assert z["salience"].shape[1] == ONNX_IN_BINS
        assert z["target_midi"].shape == (ONNX_K, z["salience"].shape[0])
        assert (z["ranges"][:4, 0] == 1).all() and (z["ranges"][4:, 0] == 0).all()
        voiced = z["target_midi"][:4] >= 0
        assert voiced.mean() > 0.5
        # 슬롯은 음역 높은 순: 같은 프레임에서 위 슬롯 음이 아래 슬롯보다 낮지 않다
        both = voiced[0] & voiced[3]
        assert (z["target_midi"][0][both] >= z["target_midi"][3][both]).all()


def test_dataset_and_model_shapes(synth_data):
    ds = VoiceDataset(synth_data / "train", train=True, max_shift=2, seed=1)
    sal, ranges, cls, act, slot = ds[0]
    assert sal.shape == (1, ONNX_WINDOW, ONNX_IN_BINS) and cls.shape == (ONNX_K, ONNX_WINDOW)
    model = VoiceAssignNet(channels=8)
    pitch, active = model(sal[None], ranges[None])
    assert pitch.shape == (1, ONNX_K, ONNX_WINDOW, ONNX_OUT_BINS) and active.shape == (
        1,
        ONNX_K,
        ONNX_WINDOW,
    )
    assert (active[0, 4:] < -10).all()  # 쓰지 않는 슬롯은 무음


def test_train_resume_and_export(synth_data, tmp_path):
    out = tmp_path / "ckpt"
    common = [
        "--train",
        str(synth_data / "train"),
        "--valid",
        str(synth_data / "valid"),
        "--out",
        str(out),
        "--batch",
        "2",
        "--channels",
        "8",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--ckpt-every",
        "2",
        "--valid-every",
        "2",
        "--backup-dir",
        str(tmp_path / "backup"),
    ]
    r1 = train.train(train.build_args(common + ["--max-steps", "4"]))
    assert r1["step"] == 4 and (out / "last.pt").exists() and (out / "best.pt").exists()
    assert (tmp_path / "backup" / "last.pt").exists()
    # 중단 후 재개: 4스텝부터 이어서 6스텝까지
    r2 = train.train(train.build_args(common + ["--max-steps", "6", "--resume"]))
    assert r2["step"] == 6 and len(r2["history"]) == 3

    # 파인튜닝: 앞단 고정
    r3 = train.train(
        train.build_args(
            [
                "--train",
                str(synth_data / "train"),
                "--valid",
                str(synth_data / "valid"),
                "--out",
                str(tmp_path / "ft"),
                "--batch",
                "2",
                "--channels",
                "8",
                "--workers",
                "0",
                "--device",
                "cpu",
                "--max-steps",
                "2",
                "--valid-every",
                "2",
                "--init",
                str(out / "best.pt"),
                "--freeze-front",
                "--lr",
                "1e-4",
            ]
        )
    )
    assert r3["step"] == 2

    onnx_path = tmp_path / "models" / "voice_assign.onnx"
    info = export_onnx.export(out / "best.pt", onnx_path)
    assert info["max_abs_diff"] < 1e-3 and onnx_path.exists()

    # 앱의 ONNX 경로로 실제 추론
    from acappella.pipeline.pitch import contour_midi_bins, frame_times
    from acappella.types import PitchActivation

    T = 300
    act = PitchActivation(
        frame_times(T), contour_midi_bins(), np.random.rand(T, ONNX_IN_BINS).astype(np.float32)
    )
    tracks = assign_voices_onnx(
        act, parse_voices("SATB"), onnx_path, session=_onnx_session(str(onnx_path))
    )
    assert len(tracks) == 4 and all(t.f0_midi.shape == (T,) for t in tracks)
