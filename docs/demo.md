# 音频演示

## 公开诊断音频

以下是公式合成的谐波信号经过房间、点源干扰、扩散噪声和前端后的演示。
它不是可懂语音，不参与 CER 或正式指标，不替代真实测试结果。
三个文件共用播放增益，去掉相同校准前缀，保留传播时延与混响尾部。

- [未处理参考通道](assets/demo/ref_mic.wav)
- [DAS](assets/demo/das.wav)
- [已知方向/纯噪声校准 MVDR](assets/demo/oracle_mvdr.wav)

完全自制，无第三方录音；作为 CC0-1.0 提供。公式、随机种子、场景和文件哈希见
[生成脚本](../scripts/create_public_demo.py) 与 [来源清单](assets/demo/provenance.json)。

## 五组真实语音的本地对比

正式测试使用真实 AISHELL 语音与 MUSAN 噪声，数据来源条件见 [数据来源](data_sources.md)。
运行 `bash scripts/reproduce.sh` 后，打开本地 [五组试听页](demo_phase3.md)，
对应文件在 `results/generated/phase3/demo/`。真实录音和派生音频留在本机，未作为本仓库的 CC0 素材发布。
同场景的 IRM、神经 MVDR、单通道结果另见 `results/generated/phase5/audio/`。

机器已检查完整性和数值，不把这项检查称为人工试听。建议按相同播放音量，记录干扰残留、混响、语音失真与错字位置。
