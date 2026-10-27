"""水平面宽带 DOA：多基线 GCC-PHAT、SRP-PHAT、非相干 MUSIC。

角度沿用 arrays.py 的“从阵列看向声源”约定。线阵只搜索 0–180°，
因为一条直线无法区分镜像半平面；不把不可辨识方向当作算法误差。
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from farfield_asr.arrays import steering_vector
from farfield_asr.beamforming import spatial_covariance
from farfield_asr.transforms import DEFAULT_STFT, stft


@dataclass
class DOAResult:
    azimuth_deg: float
    grid_deg: np.ndarray
    scores: np.ndarray
    frequency_bins: int
    frames: int


def angular_error(estimate: float, reference: float, geometry: str) -> float:
    if geometry == "linear":
        return float(abs(estimate - reference))
    if geometry != "circular":
        raise ValueError("unknown array geometry")
    difference = abs(estimate - reference) % 360
    return float(min(difference, 360 - difference))


def estimate_doa(audio, microphones_m, method: str, geometry: str,
                 sample_rate: int = 16000, sound_speed: float = 343.0,
                 fmin: float = 500, fmax: float = 3500, max_frames: int = 300,
                 n_sources: int = 1, interpolation: int = 16) -> DOAResult:
    """返回扫描谱和最大峰方向；不使用目标真值或干净分量来选峰。

    所有方法均默认寻找最强空间源；多源/强干扰时可能定位到干扰。
    MUSIC 用 n_sources 个信号特征向量，默认单个主导源，不声称已解决目标选择。
    """
    x, positions = np.asarray(audio, dtype=float), np.asarray(microphones_m, dtype=float)
    if (x.ndim != 2 or len(x) < DEFAULT_STFT.n_fft or positions.shape != (x.shape[1], 3)
            or x.shape[1] < 2 or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(positions)) or np.mean(x**2) <= 1e-20):
        raise ValueError("expected nonzero finite multichannel audio and matching 3-D positions")
    if method not in ("gcc_phat", "srp_phat", "music") or geometry not in ("linear", "circular"):
        raise ValueError("unknown DOA method or geometry")
    if (not 0 < fmin < fmax < sample_rate / 2 or sound_speed <= 0
            or type(max_frames) is not int or max_frames < 2
            or type(interpolation) is not int or interpolation < 1
            or type(n_sources) is not int or not 1 <= n_sources < x.shape[1]):
        raise ValueError("invalid frequency range or DOA parameters")
    spectrum = stft(x)
    selected_frames = np.linspace(0, spectrum.shape[1] - 1,
                                  min(max_frames, spectrum.shape[1]), dtype=int)
    spectrum = spectrum[:, selected_frames]
    all_frequencies = np.fft.rfftfreq(DEFAULT_STFT.n_fft, 1 / sample_rate)
    bins = np.flatnonzero((all_frequencies >= fmin) & (all_frequencies <= fmax))
    frequencies = all_frequencies[bins]
    if len(bins) < 2:
        raise ValueError("insufficient frequency bins")
    selected = spectrum[bins]
    grid = np.arange(181 if geometry == "linear" else 360, dtype=float)
    if method == "gcc_phat":
        directions = np.column_stack((np.cos(np.deg2rad(grid)), np.sin(np.deg2rad(grid)),
                                      np.zeros(len(grid))))
        errors = []
        for i, j in combinations(range(x.shape[1]), 2):
            baseline = positions[i] - positions[j]
            distance = np.linalg.norm(baseline)
            if distance < 1e-9:
                continue
            cross = selected[:, :, i] * selected[:, :, j].conj()
            phat = np.mean(cross / np.maximum(np.abs(cross), 1e-20), axis=1)
            full_cross = np.zeros(spectrum.shape[0], dtype=complex)
            full_cross[bins] = phat
            correlation = np.fft.irfft(full_cross, n=DEFAULT_STFT.n_fft * interpolation)
            max_lag = int(np.ceil(distance / sound_speed * sample_rate * interpolation))
            lags = np.arange(-max_lag, max_lag + 1)
            values = correlation[lags % len(correlation)]
            measured = lags[np.argmax(values)] / (sample_rate * interpolation)
            predicted = -directions @ baseline / sound_speed
            errors.append((predicted - measured)**2)
        if not errors:
            raise ValueError("no nonzero microphone baseline")
        raw_scores = -np.mean(errors, axis=0)
    else:
        # (F,G,M)，每个频点分别扫描，再作非相干合并。
        steering = np.stack([steering_vector(positions, frequencies, angle, sound_speed)
                             for angle in grid], axis=1)
        if method == "srp_phat":
            whitened = selected / np.maximum(np.abs(selected), 1e-20)
            covariance = spatial_covariance(whitened)
            raw_scores = np.einsum("fgm,fmn,fgn->g", steering.conj(), covariance,
                                   steering).real / len(frequencies)
        else:
            covariance = spatial_covariance(selected)
            _, vectors = np.linalg.eigh(covariance)
            noise_vectors = vectors[:, :, :x.shape[1] - n_sources]
            projector = np.einsum("fmi,fni->fmn", noise_vectors, noise_vectors.conj())
            denominator = np.einsum("fgm,fmn,fgn->fg", steering.conj(), projector,
                                    steering).real
            per_frequency = 1 / np.maximum(denominator, 1e-10)
            # 频点内归一化，避免一个极尖的窄带峰支配整个宽带结果。
            per_frequency /= np.max(per_frequency, axis=1, keepdims=True)
            raw_scores = per_frequency.mean(axis=0)
    span = np.max(raw_scores) - np.min(raw_scores)
    scores = (raw_scores - np.min(raw_scores)) / max(float(span), 1e-30)
    return DOAResult(float(grid[np.argmax(scores)]), grid, scores,
                     len(frequencies), len(selected_frames))
