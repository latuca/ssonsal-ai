"""학습 npz의 salience 저장 형식. torch 없이 쓸 수 있어야 한다(합성·매칭은 앱 환경에서도 돈다)."""

from __future__ import annotations

import numpy as np


def pack_salience(sal: np.ndarray) -> np.ndarray:
    """디스크 절약용 8비트 저장(0~1 → 0~255). float16보다 약 3배 작고 오차는 1/510 이하다."""
    return np.round(np.clip(np.asarray(sal, dtype=np.float32), 0, 1) * 255).astype(np.uint8)


def unpack_salience(sal: np.ndarray) -> np.ndarray:
    return sal.astype(np.float32) / 255 if sal.dtype == np.uint8 else sal
