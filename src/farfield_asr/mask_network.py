"""轻量参考麦克风掩码网络。两层单向 GRU；整个协方差/MVDR 流程仍为离线算法。"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from farfield_asr.masks import log_power_features


class MaskNetwork(nn.Module):
    def __init__(self, n_bins=257, hidden_size=64):
        super().__init__()
        self.n_bins, self.hidden_size = n_bins, hidden_size
        self.register_buffer("feature_mean", torch.zeros(n_bins))
        self.register_buffer("feature_std", torch.ones(n_bins))
        self.embedding = nn.Sequential(nn.Linear(n_bins, hidden_size), nn.ReLU())
        self.recurrent = nn.GRU(hidden_size, hidden_size, num_layers=2,
                                batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_size, n_bins)

    def forward(self, features):
        """输入 (B,T,F)，返回相同形状的掩码 logits；训练用软目标 BCE。"""
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-4)
        embedded = self.embedding(normalized)
        sequence, _ = self.recurrent(embedded)
        return self.output(sequence)


class NeuralMaskEstimator:
    def __init__(self, checkpoint_path: str | Path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.metadata = checkpoint["metadata"]
        self.model = MaskNetwork(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device).eval()
        self.device = device

    def estimate(self, spectrum, reference_mic=0):
        features = torch.from_numpy(log_power_features(spectrum, reference_mic))[None].to(self.device)
        with torch.inference_mode():
            mask = torch.sigmoid(self.model(features))[0].cpu().numpy().T
        return np.clip(mask, 1e-4, 1 - 1e-4)
