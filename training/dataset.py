"""학습 데이터셋: synth.py npz(층 A)와 실제 음원 npz(층 B)를 같은 형식으로 읽는다.

한 예시 = ONNX_WINDOW 프레임 구간. train은 무작위 구간 + 피치 시프트 증강, valid는 고정 구간.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from acappella.pipeline.voices import (
    ONNX_IN_BINS,
    ONNX_K,
    ONNX_OUT_BASE_MIDI,
    ONNX_OUT_BINS,
    ONNX_WINDOW,
)
from training.storage import unpack_salience

IGNORE = -100


CACHE_FILES = 48  # 워커마다 들고 있는 파일 수. salience는 저장 형식(8비트) 그대로 둔다


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in ("salience", "target_midi", "ranges")}


class VoiceDataset(Dataset):
    def __init__(
        self, root: Path, train: bool, windows_per_file: int = 8, max_shift: int = 2, seed: int = 0
    ):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"npz 파일이 없습니다: {root}")
        self.train = train
        self.windows_per_file = windows_per_file
        self.max_shift = max_shift if train else 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._rng_pid = os.getpid()
        self._cache: dict[int, dict] = {}
        self.items: list[tuple[int, int]] = []
        if not train:  # 고정 구간: 파일마다 겹치지 않게 앞에서부터
            for i, f in enumerate(self.files):
                T = _load(f)["salience"].shape[0]
                for s in range(0, max(T - ONNX_WINDOW, 0) + 1, ONNX_WINDOW):
                    self.items.append((i, s))

    def __len__(self) -> int:
        return len(self.files) * self.windows_per_file if self.train else len(self.items)

    def _get(self, i: int) -> dict:
        if i not in self._cache:
            if len(self._cache) >= CACHE_FILES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = _load(self.files[i])
        return self._cache[i]

    def __getitem__(self, idx: int):
        if os.getpid() != self._rng_pid:
            # DataLoader 워커는 fork로 같은 난수 상태를 물려받는다. 워커마다 다시 시드를 준다
            self._rng_pid = os.getpid()
            self.rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**32])
        if self.train:
            fi = idx % len(self.files)
            d = self._get(fi)
            T = d["salience"].shape[0]
            start = int(self.rng.integers(0, max(T - ONNX_WINDOW, 0) + 1))
            shift = (
                int(self.rng.integers(-self.max_shift, self.max_shift + 1)) if self.max_shift else 0
            )
        else:
            fi, start = self.items[idx]
            d = self._get(fi)
            shift = 0
        return make_example(d, start, shift)


def make_example(d: dict, start: int, shift: int):
    sal = unpack_salience(d["salience"][start : start + ONNX_WINDOW]).astype(np.float32)
    tgt = d["target_midi"][:, start : start + ONNX_WINDOW].astype(np.int64)
    pad = ONNX_WINDOW - sal.shape[0]
    if pad > 0:
        sal = np.pad(sal, ((0, pad), (0, 0)))
        tgt = np.pad(tgt, ((0, 0), (0, pad)), constant_values=-1)
    ranges = d["ranges"].astype(np.float32).copy()
    if shift:
        b = 3 * shift
        sal = np.roll(sal, b, axis=1)
        if b > 0:
            sal[:, :b] = 0
        else:
            sal[:, b:] = 0
        tgt = np.where(tgt >= 0, tgt + shift, tgt)
        ranges[:, 1:] += np.where(ranges[:, :1] > 0, shift / 88, 0)
    active = (tgt >= 0) & (ranges[:, :1] > 0)
    cls = np.where(active, tgt - ONNX_OUT_BASE_MIDI, IGNORE)
    cls = np.where((cls >= 0) & (cls < ONNX_OUT_BINS) | (cls == IGNORE), cls, IGNORE)
    assert sal.shape == (ONNX_WINDOW, ONNX_IN_BINS) and cls.shape == (ONNX_K, ONNX_WINDOW)
    return (
        torch.from_numpy(sal[None]),
        torch.from_numpy(ranges),
        torch.from_numpy(cls),
        torch.from_numpy(active.astype(np.float32)),
        torch.from_numpy((ranges[:, 0] > 0).astype(np.float32)),
    )
