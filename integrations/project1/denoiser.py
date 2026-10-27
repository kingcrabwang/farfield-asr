"""为项目一补建的离线单通道 DL 降噪模块；与多通道实验共享已训练掩码估计器。"""

import numpy as np

from farfield_asr.mask_network import NeuralMaskEstimator
from farfield_asr.transforms import istft, stft


class SingleChannelDenoiser:
    def __init__(self, checkpoint_path, device="cpu"):
        self.estimator = NeuralMaskEstimator(checkpoint_path, device=device)

    def process(self, audio):
        """输入 (N,) -> 输出 (N,)，以估计功率比作为单通道 Wiener 型幅度增益。"""
        x = np.asarray(audio, dtype=float)
        if x.ndim != 1 or len(x) < 1 or not np.all(np.isfinite(x)):
            raise ValueError("single-channel denoiser expects finite nonempty mono audio")
        spectrum = stft(x)
        gain = self.estimator.estimate(spectrum)
        return istft(spectrum[:, :, 0] * gain, len(x))
