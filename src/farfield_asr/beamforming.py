"""频域 DAS 与方向响应；约定输出 Y = w^H X。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def das_weights(steering: ArrayLike) -> NDArray:
    """导向矢量每个元素模长为 1 时，w=a/M 保证 w^H a=1。"""
    a = np.asarray(steering, dtype=complex)
    if a.ndim != 2 or a.shape[1] < 2 or not np.all(np.isfinite(a)):
        raise ValueError("steering must have finite shape (F, M), M >= 2")
    if not np.allclose(np.abs(a), 1):
        raise ValueError("DAS expects unit-magnitude far-field steering vectors")
    return a / a.shape[1]


def apply_weights(spectrum: ArrayLike, weights: ArrayLike) -> NDArray:
    """将 (F, T, M) 复数谱变成 (F, T) 单通道谱；不修改输入。"""
    x = np.asarray(spectrum)
    w = np.asarray(weights)
    if x.ndim != 3 or w.ndim != 2 or (x.shape[0], x.shape[2]) != w.shape:
        raise ValueError("expected spectrum (F, T, M) and weights (F, M)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("spectrum and weights must be finite")
    return np.einsum("fm,ftm->ft", w.conj(), x)


def spatial_covariance(spectrum: ArrayLike, mask: ArrayLike | None = None) -> NDArray:
    """(F,T,M) -> (F,M,M)，按掩码加权 x x^H；全零掩码得到零矩阵。"""
    x = np.asarray(spectrum, dtype=complex)
    if x.ndim != 3 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("expected finite nonempty spectrum (F,T,M)")
    weights = np.ones(x.shape[:2]) if mask is None else np.asarray(mask, dtype=float)
    if (weights.shape != x.shape[:2] or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or np.any(weights > 1)):
        raise ValueError("mask must have shape (F,T) with values in [0,1]")
    covariance = np.einsum("ft,ftm,ftn->fmn", weights, x, x.conj())
    covariance /= np.maximum(weights.sum(axis=1), 1e-12)[:, None, None]
    return (covariance + covariance.conj().transpose(0, 2, 1)) / 2


def mvdr_weights(steering: ArrayLike, noise_covariance: ArrayLike,
                 diagonal_loading: float = 1e-3, absolute_floor: float = 1e-12) -> NDArray:
    """解线性方程得到 R^-1 a/(a^H R^-1 a)，不显式求逆。

    加载量为 loading*trace(R)/M + floor，既随功率缩放，又能处理静音/低秩频点。
    这是给定导向矢量的 MVDR；已知噪声/方向也不保证所有语音指标单调改善。
    """
    a = np.asarray(steering, dtype=complex)
    covariance = np.asarray(noise_covariance, dtype=complex)
    if (a.ndim != 2 or min(a.shape) < 1 or covariance.shape != (a.shape[0], a.shape[1], a.shape[1])
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(covariance))):
        raise ValueError("expected finite steering (F,M) and covariance (F,M,M)")
    if (not np.isfinite(diagonal_loading) or diagonal_loading < 0
            or not np.isfinite(absolute_floor) or absolute_floor <= 0):
        raise ValueError("invalid diagonal loading")
    if not np.allclose(covariance, covariance.conj().transpose(0, 2, 1), atol=1e-10):
        raise ValueError("noise covariance must be Hermitian")
    power = np.trace(covariance, axis1=1, axis2=2).real / a.shape[1]
    if np.any(np.linalg.eigvalsh(covariance) < -np.maximum(1e-12, np.abs(power) * 1e-8)[:, None]):
        raise ValueError("noise covariance must be positive semidefinite")
    loading = diagonal_loading * np.maximum(power, 0) + absolute_floor
    loaded = covariance + loading[:, None, None] * np.eye(a.shape[1])
    # NumPy 2.x 用显式末尾维度，避免批量向量 solve 的维度歧义。
    numerator = np.linalg.solve(loaded, a[:, :, None])[:, :, 0]
    denominator = np.einsum("fm,fm->f", a.conj(), numerator).real
    if np.any(denominator <= 0):
        raise ValueError("degenerate steering vector")
    return numerator / denominator[:, None]

