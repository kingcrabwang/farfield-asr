"""音频 I/O：固定 (N,M) 布局，不隐式重采样、不自动归一化。"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile


def read_audio(path: str | Path, expected_sample_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"sample rate mismatch: {sample_rate} != {expected_sample_rate}")
    if min(audio.shape) < 1 or not np.all(np.isfinite(audio)):
        raise ValueError("audio must be finite and nonempty")
    return audio


def write_audio(path: str | Path, audio, sample_rate: int = 16000, *, overwrite=False):
    """保存 float32 WAV，允许浮点幅值超过 1，不做 PCM 裁剪或独立通道缩放。"""
    path = Path(path)
    x = np.asarray(audio)
    if (np.iscomplexobj(x) or x.ndim not in (1, 2) or min(x.shape) < 1
            or not np.all(np.isfinite(x))):
        raise ValueError("audio must be real, finite and nonempty")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if path.suffix.lower() != ".wav":
        raise ValueError("write_audio only writes WAV files")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # libsndfile 的浮点 WAV 默认 PEAK chunk 含时间戳，重复写盘会改变文件哈希。
    # SciPy 根据 float32 dtype 写 IEEE FLOAT WAV，不加入该时间戳。
    wavfile.write(path, sample_rate, x.astype(np.float32))

