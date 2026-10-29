"""학습한 체크포인트 → ONNX (앱의 ONNX Runtime CPU 추론용).

    uv run python -m training.export_onnx --ckpt checkpoints/finetune/best.pt --out models/voice_assign.onnx

내보낸 뒤 ONNX Runtime 출력이 PyTorch와 같은지 확인하고, CPU에서 구간당 추론 시간을 잰다.
배포 기준: test 성능(eval/run_eval.py --voice-method onnx)과 CPU 처리 시간이 규칙 기반보다
나을 때만 GitHub Releases에 올린다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from acappella.pipeline.voices import ONNX_IN_BINS, ONNX_INPUTS, ONNX_K, ONNX_OUTPUTS, ONNX_WINDOW
from training.model import VoiceAssignNet


def export(ckpt: Path, out: Path, opset: int = 17) -> dict:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    channels = state.get("config", {}).get("channels", 32)
    model = VoiceAssignNet(channels)
    model.load_state_dict(state["model"])
    model.eval()
    sal = torch.rand(1, 1, ONNX_WINDOW, ONNX_IN_BINS)
    ranges = torch.zeros(1, ONNX_K, 3)
    ranges[0, :4] = torch.tensor([[1, 0.5, 0.7], [1, 0.4, 0.6], [1, 0.3, 0.5], [1, 0.2, 0.4]])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sal, ranges),
        str(out),
        input_names=list(ONNX_INPUTS),
        output_names=list(ONNX_OUTPUTS),
        dynamic_axes={n: {0: "batch"} for n in (*ONNX_INPUTS, *ONNX_OUTPUTS)},
        opset_version=opset,
        dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feeds = {"salience": sal.numpy(), "voice_ranges": ranges.numpy()}
    o_pitch, o_active = sess.run(list(ONNX_OUTPUTS), feeds)
    with torch.no_grad():
        t_pitch, t_active = model(sal, ranges)
    err = max(
        float(np.abs(o_pitch - t_pitch.numpy()).max()),
        float(np.abs(o_active - t_active.numpy()).max()),
    )
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(list(ONNX_OUTPUTS), feeds)
    ms = (time.perf_counter() - t0) / 5 * 1000
    windows_per_3min = int(np.ceil(180 * 86.13 / (ONNX_WINDOW // 2)))
    info = {
        "ckpt": str(ckpt),
        "step": state.get("step"),
        "valid_best": state.get("best"),
        "max_abs_diff": err,
        "ms_per_window": round(ms, 2),
        "est_seconds_per_3min_song": round(ms * windows_per_3min / 1000, 2),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if err > 1e-3:
        raise RuntimeError(f"ONNX와 PyTorch 출력 차이가 큽니다: {err}")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="체크포인트 → ONNX")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/voice_assign.onnx"))
    args = ap.parse_args(argv)
    print(json.dumps(export(args.ckpt, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
