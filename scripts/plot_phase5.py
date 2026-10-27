"""保存训练曲线、真实测试掩码及理想/学习掩码前端对照表。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from farfield_asr.audio_io import read_audio
from farfield_asr.mask_network import NeuralMaskEstimator
from farfield_asr.masks import ideal_power_mask
from farfield_asr.transforms import stft


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/generated/phase5"
    checkpoint = root / "checkpoints/mask_gru"
    history = json.loads((checkpoint / "history.json").read_text())
    training = json.loads((checkpoint / "training_summary.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for field, label in (("train_bce", "train"), ("dev_bce", "dev")):
        axes[0].plot([row["epoch"] for row in history], [row[field] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Soft-mask BCE")
    axes[0].axvline(training["best_epoch"], color="grey", linestyle=":", label="best dev")
    axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["dev_mask_mse"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Dev mask MSE")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "training.png", dpi=160)
    plt.close(figure)
    estimator = NeuralMaskEstimator(checkpoint / "best.pt")
    manifest = [json.loads(line) for line in
                (root / "data/simulated/test_v1/manifest.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(3, 3, figsize=(12, 8), sharey=True)
    diagnostics = []
    for row_index, entry in enumerate(manifest[:3]):
        folder = root / "data/simulated/test_v1" / entry["scene"]["scene_id"]
        mixture = stft(read_audio(folder / "mixture.wav"))
        target = stft(read_audio(folder / "target_reverberant.wav"))[:, :, 0]
        noise = stft(read_audio(folder / "interference.wav") + read_audio(folder / "diffuse.wav"))[:, :, 0]
        ideal = ideal_power_mask(target, noise)
        predicted = estimator.estimate(mixture)
        diagnostics.append({"scene_id": entry["scene"]["scene_id"],
                            "mask_mse": float(np.mean((ideal - predicted) ** 2))})
        for column, (values, title) in enumerate(((ideal, "Ideal power mask"),
                                                 (predicted, "Predicted mask"),
                                                 (np.abs(ideal - predicted), "Absolute error"))):
            axis = axes[row_index, column]
            displayed = axis.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=1,
                                    extent=(0, values.shape[1] * 128 / 16000, 0, 8))
            axis.set(title=f"{entry['scene']['scene_id']}: {title}", xlabel="Time (s)")
            if column == 0:
                axis.set_ylabel("Frequency (kHz)")
            figure.colorbar(displayed, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output / "test_masks.png", dpi=160)
    plt.close(figure)
    (output / "mask_visualization.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    base = json.loads((root / "results/generated/phase3/summary.json").read_text())
    learned = json.loads((output / "summary.json").read_text())
    lines = ["# 掩码训练与前端对照", "", "相同的 200 条原始 AISHELL 测试语音；所有指标排除 0.6 s 校准前缀。",
             "SI-SDR/PESQ/STOI 使用同一参考麦克风直达目标信号。CER 在第六阶段汇总。", "",
             "| 前端 | N | SI-SDR (dB) | PESQ-WB | STOI |", "| --- | ---: | ---: | ---: | ---: |"]
    for method, result in {**base["methods"], **learned["methods"]}.items():
        metrics = result["metrics"]
        lines.append(f"| {method} | {result['n']} | {metrics['si_sdr_db']:.4f} | "
                     f"{metrics['pesq_wb']:.4f} | {metrics['stoi']:.4f} |")
    lines += ["", "## 实际结论", "",
              ("1. IRM-MVDR 的 SI-SDR 为 −1.9778 dB；神经 MVDR 为 −6.1756 dB，差距约 4.20 dB，"
              "表明理想分量监督提供了明显但尚未达到的参考水平。IRM 含测试真值，不能部署；也不保证逐指标绝对上界。"),
              ("2. 神经 MVDR 在当前训练规模下没有超过 DAS（−6.0904 dB，STOI 0.5242）。"
              "验证损失下降并不等于空间滤波增益提升，不能把训练成功写成性能胜出。"),
              ("3. 相同网络的单通道补充模块 SI-SDR 为 −4.8706 dB、PESQ 为 1.1641，"
              "但 STOI 仅 0.4997。掩码误差、协方差偏差、混响参考差异是待验证假设；CER 需独立实测。"), "",
              "## 证据", "",
              "网络 83,137 参数，两层单向 GRU，参考通道 log-power 输入，软 IRM BCE 训练。",
              "320 条原始 train 语音、80 条原始 dev 语音，各两次仿真；20 个 epoch，最佳 dev epoch=19。",
              "单 batch 576 步 MSE 从 0.157479 降至 0.009991；该预检模型未用于正式训练。",
              "train/dev/test 说话人与源噪声分离；归一化仅拟合 train。完整检查点与每轮日志保存在本机 checkpoints/mask_gru/。",
              ("项目一模块是本轮在 /home/kingcrab/speech-front-end/denoising 补建的对照，"
              "复用同一模型权重；不是独立预训练系统，不能声称项目一完整 AEC/VAD 已完成。"), "",
              "![训练曲线](../docs/assets/phase5_training.png)", "![测试掩码](../docs/assets/phase5_test_masks.png)", ""]
    (root / "results/masks.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
