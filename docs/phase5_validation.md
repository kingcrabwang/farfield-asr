# 第五阶段验收：自训练掩码与 MVDR

验收时间：2026-10-28。主仓库 `/home/kingcrab/farfield-asr`。

## 已完成的实测

- 原始 AISHELL train 320 条（8 位说话人）与 dev 80 条（4 位说话人），各两次场景仿真，640/160 个特征样本。未把 test 重标成 train。
- train/dev/test 说话人、源噪声 ID 分离。源 CSV SHA256、ZIP 成员 CRC、音频 SHA256 和特征哈希保留在本机数据清单；没有验证原始完整 15 GB 归档 MD5。
- 参考通道 log-power → 64 维投影 → 两层单向 GRU → 257 维 sigmoid 掩码，83,137 参数。
- 归一化只使用 train；100-step 检查后继续到 576 步，单 batch MSE 从 0.157479 降到 0.009991。随后重新初始化正式网络。
- CPU 正式训练 20 epoch，最佳 dev BCE 为 0.1986275，epoch 19。每轮记录 train BCE、dev BCE/MSE，保存优化器和 RNG 的断点。
- 最佳权重 SHA256：`a2ff24a03585caa38972744b03e46a9e0f5e2dcb3836aa843817dbce77f196d8`。
- 200 条测试上的 IRM-MVDR、神经 MVDR、单通道补充模块，合计 600 份指标均有效；测试集与检查点隔离检查通过。
- 当前完整测试 75 项通过，Ruff 通过。

## 算法约定

功率 IRM 为 `|S|²/(|S|²+|N|²)`，不是开平方的幅度 IRM。
`S` 为参考麦克风混响目标，`N` 为点源干扰与扩散噪声之和。
训练 BCE 使用软标签；语音/噪声协方差用预测掩码及其补集，对所有帧加权平均。
Souden MVDR 使用 `Rn_loaded^-1 Rs / trace(Rn_loaded^-1 Rs)` 的参考通道列。
加载量 `1e-3 * trace(Rn)/M + 1e-12`，退化语音协方差回退参考通道。
即使网络是单向 GRU，当前全句协方差仍是离线处理，不能宣称整体流式。

## 结果与限制

详细对照表和图见 [results/masks.md](../results/masks.md)。
IRM 与神经 MVDR 的直达参考 SI-SDR 相差约 4.20 dB。
当前神经 MVDR 未超过 DAS，不能用验证损失下降替代增强性能验收。
三条固定测试样本的掩码图显示预测较平滑、语音高占比时频点估计不足；这是观察，不能单凭三图断定因果。

项目一原目录仅有基础示例，本轮补建 `/home/kingcrab/speech-front-end/denoising`。
该单通道基线与神经 MVDR 共用网络权重，只改变后处理，便于控制比较；不代表完整项目一已完成。
导出脚本拒绝覆盖内容不同的已有用户文件，导出源与权重哈希见 phase5/project1_integration.json。
Colab 未经验证，本次训练实际使用 Ubuntu CPU。

## 复现

```bash
bash scripts/install_torch_cpu.sh
python scripts/prepare_training_speech.py
python scripts/prepare_mask_training.py
python scripts/train_mask.py
python scripts/export_project1_denoiser.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_mask_frontends.py
python scripts/plot_phase5.py
```

先完成主环境安装、测试数据与噪声准备；ASR 的 NumPy 1.26.4 环境独立安装，见 environment-asr.yml。
所有缓存校验代码/输入/产物哈希，修改协议时应使用新的输出目录并保留旧实验，不能混用旧分数。
