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

