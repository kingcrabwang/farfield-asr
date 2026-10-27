"""阵列几何与远场模型。坐标以米计，角度以度计。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def linear_array(n_mics: int = 4, spacing_m: float = 0.04) -> NDArray:
    """返回以原点为中心、沿 x 轴排列的 (M, 3) 麦克风坐标。"""
    if not isinstance(n_mics, int) or isinstance(n_mics, bool) or n_mics < 2:
        raise ValueError("n_mics must be an integer >= 2")
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be finite and positive")
    positions = np.zeros((n_mics, 3))
    positions[:, 0] = (np.arange(n_mics) - (n_mics - 1) / 2) * spacing_m
    return positions


def circular_array(n_mics: int = 6, radius_m: float = 0.04) -> NDArray:
    """返回水平圆阵 (M, 3)，第一个麦克风位于 +x 轴。"""
    if not isinstance(n_mics, int) or isinstance(n_mics, bool) or n_mics < 3:
        raise ValueError("n_mics must be an integer >= 3")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    angles = np.arange(n_mics) * 2 * np.pi / n_mics
    return radius_m * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n_mics)))


def steering_vector(
    positions_m: ArrayLike,
    frequencies_hz: ArrayLike,
    azimuth_deg: float,
    sound_speed: float = 343.0,
    reference_mic: int = 0,
) -> NDArray:
    """返回 (F, M) 导向矢量，方位角表示从阵列看向声源的方向。

    0° 指向 +x，90° 指向 +y。声源方向 u 上的麦克风更早收到信号，
    相对时延 tau_m = -(r_m-r_ref)·u/c；numpy 的负指数 FFT 下，
    X_m(f) = exp(-j*2*pi*f*tau_m) S_ref(f)。该定义贯穿后续 DOA 与 MVDR。
    """
    positions = np.asarray(positions_m, dtype=float)
    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions_m must have shape (M, 3), M >= 2")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("frequencies must be a finite nonnegative vector")
    if not np.isfinite(sound_speed) or sound_speed <= 0 or not np.isfinite(azimuth_deg):
        raise ValueError("invalid sound speed or azimuth")
    if not isinstance(reference_mic, int) or not 0 <= reference_mic < len(positions):
        raise ValueError("reference_mic out of range")
    angle = np.deg2rad(azimuth_deg)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    delays = -(positions - positions[reference_mic]) @ direction / sound_speed
    return np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :])

