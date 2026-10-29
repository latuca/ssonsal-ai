"""3단계 다중 피치 추정: Basic Pitch 래퍼.

출력 PitchActivation:
- ``salience``: Basic Pitch contour 출력 (T, 264). 반음당 3빈, 첫 빈은 MIDI 21(A0).
- ``times``: 프레임 시각. 홉 256 샘플 @ 22050Hz(약 11.6ms).
- ``notes``: Basic Pitch 노트 후보.

Basic Pitch를 불러올 수 없는 환경(개발·테스트)에서는 CQT 고조파 합 기반 대체 백엔드를 쓴다.
대체 백엔드는 품질이 낮으므로 실제 채보에는 쓰지 않는다.
"""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from acappella.types import AudioClip, NoteEvent, PitchActivation

log = logging.getLogger(__name__)

SR = 22050
HOP = 256
BASE_MIDI = 21
BINS_PER_SEMITONE = 3
N_BINS = 88 * BINS_PER_SEMITONE


def contour_midi_bins() -> np.ndarray:
    return BASE_MIDI + np.arange(N_BINS) / BINS_PER_SEMITONE


def frame_times(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) * HOP / SR


def basic_pitch_available() -> bool:
    try:
        import basic_pitch.inference  # noqa: F401
    except Exception:  # noqa: BLE001 - tflite/onnx 백엔드 로딩 실패도 포함
        return False
    return True


@lru_cache(maxsize=1)
def _basic_pitch_model():
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    return Model(ICASSP_2022_MODEL_PATH)


def _resample(clip: AudioClip) -> np.ndarray:
    if clip.sr == SR:
        return clip.samples
    import librosa

    return librosa.resample(clip.samples, orig_sr=clip.sr, target_sr=SR)


def estimate_pitch_basic_pitch(clip: AudioClip) -> PitchActivation:
    from basic_pitch.inference import predict

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        sf.write(str(wav), _resample(clip), SR, subtype="FLOAT")
        output, _midi, events = predict(str(wav), _basic_pitch_model())
    contour = np.asarray(output["contour"], dtype=np.float32)
    notes = [
        NoteEvent(onset=float(s), offset=float(e), midi=int(p), confidence=float(a))
        for s, e, p, a, *_ in events
    ]
    notes.sort(key=lambda n: (n.onset, n.midi))
    return PitchActivation(
        times=frame_times(contour.shape[0]),
        midi_bins=contour_midi_bins(),
        salience=np.clip(contour, 0.0, 1.0),
        notes=notes,
    )


def estimate_pitch_cqt(clip: AudioClip) -> PitchActivation:
    """CQT 고조파 합 salience. Basic Pitch와 같은 시간·주파수 격자를 쓴다."""
    import librosa

    y = _resample(clip)
    n_harm_pad = 72  # 4배음까지(2옥타브 = 72빈)
    # 나이퀴스트 주파수 아래까지만 계산하고 나머지는 0으로 채운다
    max_midi = librosa.hz_to_midi(SR / 2 * 0.9)
    n_calc = min(N_BINS + n_harm_pad, int((max_midi - BASE_MIDI) * BINS_PER_SEMITONE))
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=SR,
            hop_length=HOP,
            fmin=librosa.midi_to_hz(BASE_MIDI),
            n_bins=n_calc,
            bins_per_octave=12 * BINS_PER_SEMITONE,
        )
    ).T  # (T, n_calc)
    cqt = np.pad(cqt, ((0, 0), (0, N_BINS + n_harm_pad - n_calc)))
    sal = np.zeros((cqt.shape[0], N_BINS), dtype=np.float64)
    for h, w in ((1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4)):
        shift = int(round(12 * BINS_PER_SEMITONE * np.log2(h)))
        sal += w * cqt[:, shift : shift + N_BINS]
    ref = np.percentile(sal, 99.5) if sal.size else 1.0
    sal = np.clip(sal / (ref + 1e-9), 0.0, 1.0).astype(np.float32)
    return PitchActivation(
        times=frame_times(sal.shape[0]), midi_bins=contour_midi_bins(), salience=sal
    )


def estimate_pitch(clip: AudioClip, backend: str = "auto") -> PitchActivation:
    """backend: "auto" | "basic_pitch" | "cqt"."""
    if backend == "auto":
        backend = "basic_pitch" if basic_pitch_available() else "cqt"
        if backend == "cqt":
            log.warning("Basic Pitch를 불러올 수 없어 품질이 낮은 CQT 대체 백엔드를 씁니다.")
    if backend == "basic_pitch":
        return estimate_pitch_basic_pitch(clip)
    if backend == "cqt":
        return estimate_pitch_cqt(clip)
    raise ValueError(f"알 수 없는 피치 백엔드: {backend!r}")
