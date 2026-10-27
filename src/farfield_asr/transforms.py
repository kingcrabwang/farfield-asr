"""可逆的多通道 STFT。显式处理补零与输出长度，避免时延/尾部悄悄变化。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 512
    hop_length: int = 128

    def __post_init__(self):
        if type(self.n_fft) is not int or self.n_fft < 4 or self.n_fft % 2:
            raise ValueError("n_fft must be an even integer >= 4")
        if type(self.hop_length) is not int or not 1 <= self.hop_length <= self.n_fft // 2:
            raise ValueError("hop_length must be an integer in [1, n_fft/2]")

    @property
    def window(self) -> NDArray:
        """周期 Hann 窗；分析与合成都使用此窗。"""
        return np.hanning(self.n_fft + 1)[:-1]


DEFAULT_STFT = STFTConfig()


def stft(audio: ArrayLike, config: STFTConfig = DEFAULT_STFT) -> NDArray:
    """(N,) 或 (N,M) 实数音频 -> (F,T,M) 复数谱，使用未归一化 rFFT。

    两端各补 n_fft/2 个零，再将尾部补到完整帧；第 0 帧中心对应输入第 0 点。
    本函数用于离线实验，不代表因果/流式 STFT。
    """
    x = np.asarray(audio)
    if np.iscomplexobj(x):
        raise ValueError("audio must be real")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("audio must have finite nonempty shape (N,) or (N,M)")
    n_fft, hop = config.n_fft, config.hop_length
    extra = (-len(x)) % hop
    padded = np.pad(x, ((n_fft // 2, n_fft // 2 + extra), (0, 0)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=0)[::hop]
    # sliding_window_view 的布局是 (T,M,n_fft)，统一变成 (T,n_fft,M)。
    frames = frames.transpose(0, 2, 1) * config.window[None, :, None]
    return np.fft.rfft(frames, axis=1).transpose(1, 0, 2)


def istft(spectrum: ArrayLike, n_samples: int, config: STFTConfig = DEFAULT_STFT) -> NDArray:
    """(F,T,M) -> (N,M)；输入 (F,T) 时返回 (N,)。明确裁回原始长度。"""
    z = np.asarray(spectrum)
    mono = z.ndim == 2
    if mono:
        z = z[:, :, None]
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    expected_frames = (n_samples + config.hop_length - 1) // config.hop_length + 1
    if (z.ndim != 3 or z.shape[0] != config.n_fft // 2 + 1
            or z.shape[1] != expected_frames or z.shape[2] < 1
            or not np.all(np.isfinite(z))):
        raise ValueError("spectrum shape/length does not match configuration and n_samples")
    frames = np.fft.irfft(z.transpose(1, 0, 2), n=config.n_fft, axis=1)
    total = (z.shape[1] - 1) * config.hop_length + config.n_fft
    output = np.zeros((total, z.shape[2]))
    normalization = np.zeros(total)
    window = config.window
    for frame_index, frame in enumerate(frames):
        start = frame_index * config.hop_length
        output[start:start + config.n_fft] += frame * window[:, None]
        normalization[start:start + config.n_fft] += window**2
    start = config.n_fft // 2
    selected = slice(start, start + n_samples)
    if np.any(normalization[selected] < 1e-12):
        raise ValueError("window overlap fails reconstruction coverage")
    output = output[selected] / normalization[selected, None]
    return output[:, 0] if mono else output

