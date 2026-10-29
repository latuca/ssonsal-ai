"""테스트용 합성 음원: 성부별 음표 열을 하모닉 합성음으로 렌더링한다."""

from __future__ import annotations

import numpy as np

SR = 22050


def render_voice(notes, sr=SR, total=None, vibrato=0.0, gain=0.2):
    """notes: [(onset초, offset초, midi)]. 4배음 + 짧은 페이드."""
    end = total or max(n[1] for n in notes) + 0.3
    y = np.zeros(int(end * sr), dtype=np.float64)
    for on, off, midi in notes:
        a, b = int(on * sr), int(off * sr)
        t = np.arange(b - a) / sr
        f0 = 440.0 * 2 ** ((midi - 69) / 12) * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
        phase = 2 * np.pi * np.cumsum(f0) / sr
        tone = sum(np.sin(k * phase) / k for k in (1, 2, 3, 4))
        env = np.minimum(1.0, np.minimum(t / 0.02, (t[::-1] + 1e-9) / 0.03))
        y[a:b] += gain * tone * env
    return y


def render_satb(parts, bpm=100.0, sr=SR):
    """parts: {성부: [(시작 박, 길이 박, midi)]} → (오디오, 성부별 [(onset, offset, midi)])."""
    spb = 60.0 / bpm
    secs = {
        name: [(s * spb, (s + d) * spb - 0.02, m) for s, d, m in notes]
        for name, notes in parts.items()
    }
    total = max(n[1] for v in secs.values() for n in v) + 0.5
    y = sum(render_voice(v, sr, total) for v in secs.values())
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
    return y.astype(np.float32), secs


# 4/4, 100 BPM, 4마디 다장조 코랄(박 단위)
CHORALE = {
    "S": [
        (0, 2, 72),
        (2, 2, 74),
        (4, 2, 76),
        (6, 2, 74),
        (8, 1, 72),
        (9, 1, 71),
        (10, 2, 72),
        (12, 4, 72),
    ],
    "A": [(0, 2, 67), (2, 2, 67), (4, 2, 67), (6, 2, 65), (8, 2, 64), (10, 2, 65), (12, 4, 64)],
    "T": [(0, 2, 64), (2, 2, 59), (4, 2, 60), (6, 2, 59), (8, 2, 60), (10, 2, 62), (12, 4, 55)],
    "B": [(0, 2, 48), (2, 2, 55), (4, 2, 48), (6, 2, 55), (8, 2, 57), (10, 2, 53), (12, 4, 48)],
}
