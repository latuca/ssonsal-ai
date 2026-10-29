"""6단계 박자 추적·양자화.

librosa 비트 추적으로 박 시각을 얻고, 초 → 박 위치를 박 사이 선형 보간으로 바꾼 뒤
16분음표 격자(설정 grid_division)에 맞춘다. 박자표는 사용자가 정하지 않으면 박마다의
onset 세기 주기성으로 3/4와 4/4 중에서 고르고, 같은 방법으로 첫 강박 위치를 정한다.

가정: 비트 추적기가 찾은 박을 4분음표로 본다. 6/8처럼 점4분음표가 박인 곡은 박자표와 템포를
직접 넣는 것이 안전하다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from acappella.config import Thresholds
from acappella.score_utils import enforce_monophony
from acappella.types import AudioClip, NoteEvent, QNote, Score, VoiceSpec

log = logging.getLogger(__name__)

HOP = 512
DEFAULT_TEMPO = 120.0


@dataclass
class BeatGrid:
    beat_times: np.ndarray  # 초
    tempo_bpm: float
    time_sig: tuple[int, int]
    downbeat_offset: int  # 첫 강박이 되는 beat_times 인덱스(0..마디박수-1)
    shift_beats: Fraction = Fraction(0)  # 못갖춘마디 때문에 앞에 더한 박 수(마디 단위)

    @property
    def period(self) -> float:
        return 60.0 / self.tempo_bpm

    def to_beats(self, t: np.ndarray | float) -> np.ndarray:
        """초 → 박 위치(beat_times[downbeat_offset]가 0). 범위 밖은 평균 템포로 외삽."""
        bt = self.beat_times
        idx = np.arange(bt.size, dtype=np.float64) - self.downbeat_offset
        t = np.asarray(t, dtype=np.float64)
        out = np.interp(t, bt, idx)
        before, after = t < bt[0], t > bt[-1]
        out[before] = idx[0] - (bt[0] - t[before]) / self.period
        out[after] = idx[-1] + (t[after] - bt[-1]) / self.period
        return out + float(self.shift_beats)

    def time_map(self) -> tuple[np.ndarray, np.ndarray]:
        """(박 위치, 초) 대응표. 평가에서 Score의 박을 초로 바꿀 때 쓴다."""
        idx = np.arange(self.beat_times.size, dtype=np.float64) - self.downbeat_offset
        return idx + float(self.shift_beats), self.beat_times.copy()


def track_beats(
    clip: AudioClip, tempo_bpm: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """(박 시각, onset 세기 엔벨로프, 템포). 박이 둘 미만이면 일정한 격자를 만든다."""
    import librosa

    y, sr = clip.samples, clip.sr
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    kwargs = {"bpm": tempo_bpm} if tempo_bpm else {}
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP, units="time", trim=False, **kwargs
    )
    beats = np.asarray(beats, dtype=np.float64)
    duration = len(y) / sr
    if beats.size < 2:
        bpm = float(tempo_bpm or np.atleast_1d(tempo)[0] or DEFAULT_TEMPO)
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = DEFAULT_TEMPO
        beats = np.arange(0.0, max(duration, 60.0 / bpm * 2), 60.0 / bpm)
        return beats, env, bpm
    bpm = float(tempo_bpm) if tempo_bpm else 60.0 / float(np.median(np.diff(beats)))
    return beats, env, bpm


def fix_tempo_octave(
    beats: np.ndarray, bpm: float, tempo_min: float, tempo_max: float
) -> tuple[np.ndarray, float]:
    """박 추적이 절반·두 배 템포를 잡은 경우를 [tempo_min, tempo_max] 안으로 옮긴다."""
    while bpm < tempo_min and beats.size >= 2:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        bpm *= 2
    while bpm > tempo_max and beats.size >= 4:
        beats = beats[::2]
        bpm /= 2
    return beats, bpm


def beat_accents(beat_times: np.ndarray, env: np.ndarray, sr: int) -> np.ndarray:
    frames = np.clip(np.round(beat_times * sr / HOP).astype(int), 0, max(env.size - 1, 0))
    if env.size == 0:
        return np.zeros(beat_times.size)
    # 박 주변 ±2프레임의 최댓값
    acc = np.array([env[max(f - 2, 0) : f + 3].max() for f in frames])
    return acc


def estimate_meter(
    accents: np.ndarray, time_sig: tuple[int, int] | None, first_beat: int | None = None
) -> tuple[tuple[int, int], int]:
    """(박자표, 첫 강박 인덱스). 박자표가 주어지면 강박 위치만 찾는다.

    first_beat(첫 음이 시작하는 박 인덱스)를 주면, 강세로 뚜렷이 갈리지 않을 때 첫 음을 강박에 둔다.
    """

    def best_phase(m: int) -> tuple[float, int]:
        if accents.size < m * 2:
            return 0.0, (first_beat % m if first_beat is not None else 0)
        mean = accents.mean() + 1e-9
        scores = [accents[p::m].mean() / mean for p in range(m)]
        p = int(np.argmax(scores))
        if first_beat is not None:
            q = first_beat % m
            if scores[p] < scores[q] * 1.2:
                p = q
        return float(scores[p]), p

    if time_sig is not None:
        num, den = time_sig
        beats_per_bar = Fraction(num * 4, den)
        m = int(beats_per_bar) if beats_per_bar.denominator == 1 else 0
        return time_sig, (best_phase(m)[1] if m > 1 else 0)
    s4, p4 = best_phase(4)
    s3, p3 = best_phase(3)
    if s3 > s4 * 1.15:
        return (3, 4), p3
    return (4, 4), p4


def make_grid(
    clip: AudioClip,
    tempo_bpm: float | None = None,
    time_sig: tuple[int, int] | None = None,
    first_onset: float | None = None,
    th: Thresholds | None = None,
) -> BeatGrid:
    th = th or Thresholds()
    beats, env, bpm = track_beats(clip, tempo_bpm)
    if not tempo_bpm:
        beats, bpm = fix_tempo_octave(beats, bpm, th.tempo_min, th.tempo_max)
    first_beat = None
    if first_onset is not None and beats.size:
        first_beat = int(np.argmin(np.abs(beats - first_onset)))
    ts, phase = estimate_meter(beat_accents(beats, env, clip.sr), time_sig, first_beat)
    return BeatGrid(beat_times=beats, tempo_bpm=bpm, time_sig=ts, downbeat_offset=phase)


def with_pickup_shift(grid: BeatGrid, notes: dict[str, list[NoteEvent]], division: int) -> BeatGrid:
    """첫 음이 음수 박에 오면(못갖춘마디) 마디 단위로 밀어 0 이상이 되게 한 격자를 돌려준다."""
    measure = Fraction(grid.time_sig[0] * 4, grid.time_sig[1])
    firsts = [evs[0].onset for evs in notes.values() if evs]
    if not firsts:
        return grid
    base = replace(grid, shift_beats=Fraction(0))
    first = float(base.to_beats(np.array([min(firsts)]))[0])
    earliest = Fraction(int(round(first * division)), division)
    shift = Fraction(0)
    while earliest + shift < 0:
        shift += measure
    return replace(grid, shift_beats=shift)


def quantize_events(events: list[NoteEvent], grid: BeatGrid, division: int) -> list[QNote]:
    if not events:
        return []
    on = grid.to_beats(np.array([e.onset for e in events]))
    off = grid.to_beats(np.array([e.offset for e in events]))
    out = []
    for e, a, b in zip(events, on, off):
        start = Fraction(int(round(a * division)), division)
        end = Fraction(int(round(b * division)), division)
        if end <= start:
            end = start + Fraction(1, division)
        out.append(QNote(start_beat=start, dur_beats=end - start, midi=e.midi))
    return enforce_monophony(out)


def quantize(
    notes: dict[str, list[NoteEvent]],
    clip: AudioClip,
    voices: list[VoiceSpec] | None = None,
    tempo_bpm: float | None = None,
    time_sig: tuple[int, int] | None = None,
    th: Thresholds | None = None,
    grid: BeatGrid | None = None,
) -> Score:
    """grid를 주면 그대로 쓰고(못갖춘마디 이동 포함), 없으면 비트 추적으로 만든다."""
    th = th or Thresholds()
    if grid is None:
        onsets = [evs[0].onset for evs in notes.values() if evs]
        grid = make_grid(clip, tempo_bpm, time_sig, min(onsets) if onsets else None, th)
        grid = with_pickup_shift(grid, notes, th.grid_division)
    parts = {
        name: quantize_events(sorted(evs, key=lambda e: e.onset), grid, th.grid_division)
        for name, evs in notes.items()
    }
    if voices is None:
        voices = [VoiceSpec(name, "treble", 48, 84) for name in notes]
    log.info("템포 %.1f BPM, 박자표 %d/%d", grid.tempo_bpm, *grid.time_sig)
    return Score(
        title=clip.title,
        tempo_bpm=round(grid.tempo_bpm, 1),
        time_sig=grid.time_sig,
        key_fifths=0,
        voices=list(voices),
        parts=parts,
    )
