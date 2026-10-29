"""2단계 전처리: DC 제거와 음량 정규화. (v2: 보컬 퍼커션 억제 자리)"""

from __future__ import annotations

import numpy as np

from acappella.config import Thresholds
from acappella.types import AudioClip


def normalize_loudness(
    samples: np.ndarray, target_rms_db: float = -20.0, peak_limit: float = 0.99
) -> np.ndarray:
    x = samples.astype(np.float64)
    x = x - x.mean() if x.size else x
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if rms < 1e-8:
        return x.astype(np.float32)
    x *= 10 ** (target_rms_db / 20) / rms
    peak = float(np.max(np.abs(x)))
    if peak > peak_limit:
        x *= peak_limit / peak
    return x.astype(np.float32)


def preprocess(clip: AudioClip, th: Thresholds | None = None) -> AudioClip:
    th = th or Thresholds()
    return AudioClip(
        samples=normalize_loudness(clip.samples, th.target_rms_db, th.peak_limit),
        sr=clip.sr,
        source=clip.source,
        title=clip.title,
    )
