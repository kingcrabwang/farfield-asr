# 项目一补充的单通道 DL 降噪模块

检查原 `/home/kingcrab/speech-front-end` 时只找到基础演示，未找到现成神经降噪权重。
本模块是本轮为 D6 补建的对照，不声称原项目一的 AEC/VAD/实时链路已完成。

模型使用项目二自训练的参考麦克风 GRU 功率掩码，对参考单通道频谱做 Wiener 型增益，
然后逆 STFT。与神经 MVDR 共享同一掩码估计器和训练权重，比较的是相同掩码下
“单通道抑噪”和“多通道协方差空间滤波”的差别，不是两个独立模型的架构竞赛。

导出后位置：`/home/kingcrab/speech-front-end/denoising/`。不覆盖既有项目一文件，
也不改动原来的 `audio-front-end` 环境；运行时使用本轮 `farfield-asr` 环境及已验证权重。

```bash
conda activate farfield-asr
python /home/kingcrab/speech-front-end/denoising/demo_denoise.py input.wav output.wav \
    --checkpoint /home/kingcrab/farfield-asr/checkpoints/mask_gru/best.pt
```

输入必须是 16 kHz 单声道，输出保持原始长度，不自动裁剪、重采样或独立归一化。
