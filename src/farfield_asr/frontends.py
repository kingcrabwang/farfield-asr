"""统一 process(多通道音频) -> 单通道音频；配置与纯噪声校准信息显式传入。"""

from dataclasses import dataclass

import numpy as np

from farfield_asr.arrays import steering_vector
from farfield_asr.beamforming import apply_weights, das_weights, mvdr_weights, spatial_covariance
from farfield_asr.scenes import Scene
from farfield_asr.transforms import DEFAULT_STFT, STFTConfig, istft, stft


def calibration_covariance(spectrum, sample_rate, prefix_samples, n_mics, config=DEFAULT_STFT):
    """只使用完全落在纯噪声前缀内的帧；返回协方差与实际帧数。"""
    centers = np.arange(spectrum.shape[1]) * config.hop_length
    selected = ((centers - config.n_fft // 2 >= sample_rate // 10)
                & (centers + config.n_fft // 2 < prefix_samples))
    count = int(np.count_nonzero(selected))
    if count < 2 * n_mics:
        raise ValueError("insufficient uncontaminated calibration frames")
    return spatial_covariance(spectrum[:, selected]), count


@dataclass
class Frontend:
    method: str
    scene: Scene
    noise_only_prefix_samples: int = 0
    config: STFTConfig = DEFAULT_STFT
    diagonal_loading: float = 1e-3

    def process(self, audio) -> np.ndarray:
        x = np.asarray(audio, dtype=float)
        if (x.ndim != 2 or x.shape[1] != self.scene.n_mics or len(x) < 1
                or not np.all(np.isfinite(x))):
            raise ValueError("audio layout must be finite (N, scene.n_mics)")
        if self.method == "ref_mic":
            return x[:, self.scene.reference_mic].copy()
        if self.method not in ("das", "oracle_mvdr"):
            raise ValueError(f"unsupported frontend: {self.method}")
        spectrum = stft(x, self.config)
        frequencies = np.fft.rfftfreq(self.config.n_fft, 1 / self.scene.sample_rate)
        a = steering_vector(self.scene.microphones_m, frequencies, self.scene.target_azimuth_deg,
                            self.scene.sound_speed_m_s, self.scene.reference_mic)
        if self.method == "das":
            weights = das_weights(a)
        else:
            # 排除开头传播尚未稳态的 0.1 秒；分析窗右边界必须在目标发声之前。
            covariance, _ = calibration_covariance(
                spectrum, self.scene.sample_rate, self.noise_only_prefix_samples,
                self.scene.n_mics, self.config
            )
            weights = mvdr_weights(a, covariance, self.diagonal_loading)
        return istft(apply_weights(spectrum, weights), len(x), self.config)

