"""4단계 성부 배정.

v1 규칙 기반(``method="rule"``):
  1) 프레임마다 반음 격자 salience의 봉우리(피치 후보)를 찾는다.
  2) 성부를 음역 중심이 높은 순으로 세우고, 후보를 음높이 순서대로 성부에 배정한다(순위 배정).
     후보가 성부 수보다 적으면 이웃 성부가 같은 후보를 나눠 가진다(유니즌 채우기).
     음역 밖 배정, 버린 후보, 무음 성부에는 벌점을 주고 비용이 가장 작은 배정을 DP로 고른다.
  3) 성부마다 Viterbi로 시간 연속성을 준다(음높이 변화·유무성 전환 벌점).

v2 학습 모델(``method="onnx"``): ``models/voice_assign.onnx``를 ONNX Runtime(CPU)으로 돌린다.
입출력 계약은 아래 ``ONNX_*`` 상수와 training/model.py가 공유한다.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from acappella.config import Thresholds
from acappella.types import PitchActivation, VoiceSpec, VoiceTrack

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------


def order_voices(voices: list[VoiceSpec]) -> list[int]:
    """음역 중심이 높은 성부부터의 인덱스 순서(안정 정렬)."""
    return sorted(range(len(voices)), key=lambda i: -(voices[i].low_midi + voices[i].high_midi) / 2)


def semitone_salience(
    act: PitchActivation, hop: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, 반음 MIDI 배열, salience (T', P)). 반음 주변 빈의 최댓값으로 묶고 hop 프레임씩 최댓값 풀링."""
    sal = np.asarray(act.salience, dtype=np.float32)
    bins = np.asarray(act.midi_bins, dtype=np.float64)
    semis = np.round(bins).astype(int)
    midis = np.arange(semis.min(), semis.max() + 1)
    S = np.zeros((sal.shape[0], midis.size), dtype=np.float32)
    for j, m in enumerate(midis):
        mask = semis == m
        if mask.any():
            S[:, j] = sal[:, mask].max(axis=1)
    times = np.asarray(act.times, dtype=np.float64)
    if hop > 1 and S.shape[0]:
        n = int(np.ceil(S.shape[0] / hop))
        pad = n * hop - S.shape[0]
        Sp = np.pad(S, ((0, pad), (0, 0)))
        S = Sp.reshape(n, hop, -1).max(axis=1)
        times = times[::hop]
    return times, midis, S


def smooth_time(S: np.ndarray, w: int) -> np.ndarray:
    """시간 축 이동 평균(가장자리는 가장자리 값으로 채움)."""
    if w <= 1 or S.shape[0] == 0:
        return S
    pad = w // 2
    Sp = np.pad(S, ((pad, w - 1 - pad), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, S.shape[1]), S.dtype), Sp]), axis=0)
    return ((c[w:] - c[:-w]) / w).astype(S.dtype)


def find_peaks(row: np.ndarray, abs_thr: float, rel_thr: float, max_peaks: int) -> np.ndarray:
    """한 프레임에서 봉우리 인덱스(salience 큰 순으로 최대 max_peaks개)."""
    if row.size == 0:
        return np.zeros(0, dtype=int)
    thr = max(abs_thr, rel_thr * float(row.max()))
    left = np.concatenate(([-np.inf], row[:-1]))
    right = np.concatenate((row[1:], [-np.inf]))
    idx = np.nonzero((row >= left) & (row > right) & (row >= thr))[0]
    if idx.size > max_peaks:
        idx = idx[np.argsort(-row[idx], kind="stable")[:max_peaks]]
    return idx


# ---------------------------------------------------------------------------
# 규칙 기반
# ---------------------------------------------------------------------------

# 배음 간격(반음) → 벌점 세기. 옥타브(2배음)가 가장 흔하게 오검출되고 3배음은 화음 구성음과
# 겹치는 일이 많아(C-G) 약하게 본다.
HARMONIC_STRENGTH = {12: 1.0, 19: 0.4, 24: 0.6}


def harmonic_weighted(pitches: list[int], sal: list[float], th: Thresholds) -> list[float]:
    """아래쪽 더 센(비슷한) 후보의 배음 자리에 있는 후보의 salience를 낮춘다.

    배음 후보는 성부 수보다 후보가 많을 때 먼저 버려지고, 모자랄 때만(옥타브 유니즌 등) 쓰인다.
    """
    by_pitch = dict(zip(pitches, sal))
    out = []
    for p, s in zip(pitches, sal):
        strength = max(
            (
                k
                for h, k in HARMONIC_STRENGTH.items()
                if by_pitch.get(p - h, 0.0) >= th.harmonic_ratio * s
            ),
            default=0.0,
        )
        out.append(s * (1.0 - strength * (1.0 - th.harmonic_penalty)))
    return out


def _range_cost(v: VoiceSpec, pitch: float, th: Thresholds) -> float:
    out = max(0.0, v.low_midi - th.range_margin - pitch, pitch - v.high_midi - th.range_margin)
    center = (v.low_midi + v.high_midi) / 2
    return th.range_weight * out + 0.02 * abs(pitch - center)


def assign_frame(
    pitches: list[int], sal: list[float], voices: list[VoiceSpec], th: Thresholds
) -> list[int | None]:
    """한 프레임의 후보(음높이 내림차순)를 성부(음역 내림차순)에 순서대로 배정한다.

    반환: 성부별 음높이(무음이면 None). 두 성부가 같은 후보를 가질 수 있다(유니즌).
    """
    P, N = len(pitches), len(voices)
    if P == 0:
        return [None] * N
    INF = float("inf")
    prefix = np.concatenate(([0.0], np.cumsum(sal)))

    def dropped(a: int, b: int) -> float:  # 후보 a..b-1을 버리는 비용
        return th.drop_cost * float(prefix[b] - prefix[a]) if b > a else 0.0

    # state: 마지막으로 쓴 후보 인덱스(-1: 아직 없음). cost[state+1]
    cost = [INF] * (P + 1)
    cost[0] = 0.0
    back: list[list[tuple[int, int | None]]] = []
    for v in voices:
        new = [INF] * (P + 1)
        bp: list[tuple[int, int | None]] = [(-1, None)] * (P + 1)
        for s in range(P + 1):
            c = cost[s]
            if c == INF:
                continue
            last = s - 1
            # 무음
            cand = c + th.silence_cost
            if cand < new[s]:
                new[s], bp[s] = cand, (s, None)
            for i in range(max(last, 0), P):
                extra = _range_cost(v, pitches[i], th)
                if i == last:
                    extra += th.unison_cost
                else:
                    extra += dropped(last + 1, i)
                cand = c + extra
                if cand < new[i + 1]:
                    new[i + 1], bp[i + 1] = cand, (s, i)
        cost = new
        back.append(bp)
    final = [cost[s] + dropped(s, P) for s in range(P + 1)]
    s = int(np.argmin(final))
    out: list[int | None] = [None] * N
    for j in range(N - 1, -1, -1):
        prev, i = back[j][s]
        out[j] = pitches[i] if i is not None else None
        s = prev
    return out


def viterbi_track(
    target: np.ndarray, S: np.ndarray, midis: np.ndarray, voice: VoiceSpec, th: Thresholds
) -> np.ndarray:
    """프레임 목표 음높이(무음은 nan)를 시간 연속성 있게 다듬는다. 반환: (T,) MIDI 또는 nan."""
    T = target.shape[0]
    lo = max(int(midis[0]), voice.low_midi - th.range_margin - 4)
    hi = min(int(midis[-1]), voice.high_midi + th.range_margin + 4)
    lo = min(lo, int(np.nanmin(target))) if np.isfinite(target).any() else lo
    hi = max(hi, int(np.nanmax(target))) if np.isfinite(target).any() else hi
    states = np.arange(lo, hi + 1)
    n = states.size
    cols = np.searchsorted(midis, states)
    cols = np.clip(cols, 0, midis.size - 1)

    # 전이 비용 (n+1)x(n+1), 마지막은 무음 상태
    diff = np.abs(states[:, None] - states[None, :]).astype(np.float64)
    trans = np.zeros((n + 1, n + 1))
    trans[:n, :n] = th.step_cost * diff + th.jump_cost * (diff > 0)
    trans[:n, n] = th.voicing_cost
    trans[n, :n] = th.voicing_cost

    # 방출 비용
    emis = np.empty((T, n + 1))
    has = np.isfinite(target)
    dist = np.abs(states[None, :] - np.where(has, target, 0)[:, None])
    emis[:, :n] = th.target_weight * np.minimum(dist, 4.0) + (1.0 - S[:, cols])
    emis[:, n] = th.silence_cost
    emis[~has, :n] = th.silence_cost + 1.0
    emis[~has, n] = 0.0

    D = emis[0].copy()
    D[n] += 0.0
    bp = np.empty((T, n + 1), dtype=np.int32)
    for t in range(1, T):
        tot = D[:, None] + trans
        bp[t] = np.argmin(tot, axis=0)
        D = tot[bp[t], np.arange(n + 1)] + emis[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmin(D))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    f0 = np.where(path < n, states[np.minimum(path, n - 1)], np.nan).astype(np.float64)
    return f0


def assign_voices_rule(
    act: PitchActivation, voices: list[VoiceSpec], th: Thresholds | None = None
) -> list[VoiceTrack]:
    th = th or Thresholds()
    times, midis, S = semitone_salience(act, th.voice_frame_hop)
    Sp = smooth_time(S, th.peak_smooth_frames)  # 프레임마다 후보가 깜빡이지 않게
    order = order_voices(voices)
    ordered = [voices[i] for i in order]
    T, N = S.shape[0], len(voices)
    target = np.full((T, N), np.nan)
    memo: dict[tuple, list[int | None]] = {}
    max_peaks = N + 3
    for t in range(T):
        idx = find_peaks(Sp[t], th.peak_abs_threshold, th.peak_rel_threshold, max_peaks)
        idx = np.sort(idx)[::-1]  # 음높이 내림차순
        pitches = [int(midis[i]) for i in idx]
        sal = [
            round(v * 20) / 20
            for v in harmonic_weighted(pitches, [float(Sp[t, i]) for i in idx], th)
        ]
        key = (tuple(pitches), tuple(sal))
        if key not in memo:
            memo[key] = assign_frame(pitches, sal, ordered, th)
        for j, p in enumerate(memo[key]):
            if p is not None:
                target[t, j] = p
    tracks: dict[int, VoiceTrack] = {}
    for j, vi in enumerate(order):
        f0 = viterbi_track(target[:, j], S, midis, voices[vi], th) if T else np.zeros(0)
        tracks[vi] = VoiceTrack(voice=voices[vi], times=times.copy(), f0_midi=f0)
    return [tracks[i] for i in range(N)]


# ---------------------------------------------------------------------------
# ONNX (v2)
# ---------------------------------------------------------------------------

ONNX_WINDOW = 128  # 프레임
ONNX_K = 8  # 최대 성부 슬롯
ONNX_IN_BINS = 264  # Basic Pitch contour 빈
ONNX_OUT_BASE_MIDI = 21
ONNX_OUT_BINS = 88  # 반음 단위 출력
ONNX_INPUTS = ("salience", "voice_ranges")
ONNX_OUTPUTS = ("pitch_logits", "active_logits")


def encode_voice_ranges(voices_ordered: list[VoiceSpec]) -> np.ndarray:
    """(K, 3) [활성, (하한-21)/88, (상한-21)/88]. 슬롯은 음역 높은 순."""
    r = np.zeros((ONNX_K, 3), dtype=np.float32)
    for k, v in enumerate(voices_ordered[:ONNX_K]):
        r[k] = (
            1.0,
            (v.low_midi - ONNX_OUT_BASE_MIDI) / 88,
            (v.high_midi - ONNX_OUT_BASE_MIDI) / 88,
        )
    return r


@lru_cache(maxsize=2)
def _onnx_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _median_filter_1d(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return x
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, w)
    return np.median(windows, axis=-1)


def dedupe_unisons(prob: np.ndarray, active: np.ndarray, ratio: float) -> np.ndarray:
    """프레임마다 성부 음높이를 고른다(prob (K,T,B), active (K,T) bool → (K,T) 빈 번호, 무음 -1).

    확신이 높은 성부부터 고르고, 이미 다른 성부가 고른 음은 피한다. 남은 음 중 최선의 확률이
    자기 1순위의 ``ratio`` 배 이상일 때만 바꾸고, 아니면 진짜 유니즌으로 보고 같은 음을 둔다.
    모델이 이웃 성부와 같은 음을 고르는 비율(약 30%)이 실제 악보(4~12%)보다 훨씬 높아서 넣었다.
    """
    K, T, _ = prob.shape
    top = prob.argmax(-1)
    out = np.where(active, top, -1)
    if ratio <= 0:
        return out
    conf = prob.max(-1)
    voiced = out >= 0
    same = (out[:, None, :] == out[None, :, :]) & voiced[:, None, :]
    for t in np.nonzero(same.sum((0, 1)) > voiced.sum(0))[0]:  # 대각선 말고도 같은 음이 있는 프레임
        slots = [k for k in np.argsort(-conf[:, t]) if active[k, t]]
        taken: set[int] = set()
        for k in slots:
            choice = int(top[k, t])
            if choice in taken:
                p = prob[k, t].copy()
                p[list(taken)] = -1
                alt = int(p.argmax())
                if p[alt] >= ratio * conf[k, t]:
                    choice = alt
            out[k, t] = choice
            taken.add(choice)
    return out


def assign_voices_onnx(
    act: PitchActivation,
    voices: list[VoiceSpec],
    model_path: Path,
    session=None,
    dedupe_ratio: float = 0.0,
) -> list[VoiceTrack]:
    if len(voices) > ONNX_K:
        raise ValueError(f"ONNX 모델은 성부를 최대 {ONNX_K}개까지 지원합니다.")
    sess = session or _onnx_session(str(model_path))
    sal = np.asarray(act.salience, dtype=np.float32)
    if sal.shape[1] != ONNX_IN_BINS:
        raise ValueError(f"ONNX 모델 입력은 Basic Pitch contour({ONNX_IN_BINS}빈)여야 합니다.")
    T = sal.shape[0]
    order = order_voices(voices)
    ranges = encode_voice_ranges([voices[i] for i in order])[None]
    hop = ONNX_WINDOW // 2
    n_win = max(1, int(np.ceil(max(T - ONNX_WINDOW, 0) / hop)) + 1)
    total = (n_win - 1) * hop + ONNX_WINDOW
    padded = np.pad(sal, ((0, total - T), (0, 0)))
    pitch_acc = np.zeros((ONNX_K, total, ONNX_OUT_BINS), dtype=np.float64)
    act_acc = np.zeros((ONNX_K, total), dtype=np.float64)
    weight = np.zeros(total)
    for w in range(n_win):
        s = w * hop
        x = padded[s : s + ONNX_WINDOW][None, None]
        pitch_logits, active_logits = sess.run(
            list(ONNX_OUTPUTS), {"salience": x, "voice_ranges": ranges}
        )
        pitch_acc[:, s : s + ONNX_WINDOW] += _softmax(pitch_logits[0], axis=-1)
        act_acc[:, s : s + ONNX_WINDOW] += _sigmoid(active_logits[0])
        weight[s : s + ONNX_WINDOW] += 1
    pitch_acc /= weight[None, :, None]
    act_acc /= weight[None, :]
    n = len(voices)
    active_all = np.stack([_median_filter_1d(act_acc[k, :T], 5) > 0.5 for k in range(n)])
    bins = dedupe_unisons(pitch_acc[:n, :T], active_all, dedupe_ratio)
    tracks: dict[int, VoiceTrack] = {}
    for k, vi in enumerate(order):
        # 무음 프레임은 모델 1순위로 메워 두고 메디안을 건다(nan이 번지지 않게)
        filled = np.where(bins[k] >= 0, bins[k], pitch_acc[k, :T].argmax(axis=-1))
        midi = _median_filter_1d(ONNX_OUT_BASE_MIDI + filled.astype(np.float64), 5)
        active = active_all[k]
        f0 = np.where(active, np.round(midi), np.nan)
        tracks[vi] = VoiceTrack(
            voice=voices[vi], times=np.asarray(act.times, float).copy(), f0_midi=f0
        )
    return [tracks[i] for i in range(len(voices))]


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------


def assign_voices(
    act: PitchActivation,
    voices: list[VoiceSpec],
    method: str = "rule",
    model_path: Path | None = None,
    th: Thresholds | None = None,
) -> list[VoiceTrack]:
    """method: "rule" | "onnx" | "auto"(모델 파일이 있으면 onnx, 없으면 rule)."""
    if method == "auto":
        method = "onnx" if model_path is not None and Path(model_path).exists() else "rule"
    if method == "onnx":
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"성부 배정 모델 파일이 없습니다: {model_path}")
        try:
            th = th or Thresholds()
            return assign_voices_onnx(
                act, voices, Path(model_path), dedupe_ratio=th.onnx_dedupe_ratio
            )
        except Exception:
            log.exception("ONNX 성부 배정에 실패해 규칙 기반으로 전환합니다.")
            return assign_voices_rule(act, voices, th)
    if method == "rule":
        return assign_voices_rule(act, voices, th)
    raise ValueError(f"알 수 없는 성부 배정 방식: {method!r}")
