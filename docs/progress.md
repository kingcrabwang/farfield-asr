# 进度与验收记录

## 第一部分：基础环境与阵列/DAS（已完成，2026-10-27）

已准备：独立环境定义、安装脚本、项目结构、统一坐标与张量约定、
线阵/圆阵、导向矢量、DAS 原语、方向图脚本与物理约束测试。

Ubuntu 已启动，用户名 kingcrab，Miniconda 在 /home/kingcrab/miniconda3，
共享目录 /mnt/hgfs/sharefile。最终项目已创建在 /home/kingcrab/farfield-asr。
SSH 专用密钥连接已验证，主机指纹与用户提供的指纹一致。
专用 Conda 环境在 /home/kingcrab/miniconda3/envs/farfield-asr。

实际验收：Ubuntu 下 8 项测试通过（0.10 秒）；Ruff 与 pip check 通过。
pyroomacoustics 原生仿真成功生成四路非零、有限 RIR，长度为 506、504、502、500。
方向图已生成并检查，主瓣及 4 kHz 栅瓣标注可读。
环境实际版本及构建信息已导出，见 results/generated/。
本地 Git 仓库已初始化为 main 分支，尚未发布到 GitHub。

完整证据和日常命令见 docs/phase1_validation.md。

## 第二部分：音频/STFT 与仿真基础设施（已完成，正式语料待接入）

Ubuntu 本地已实现 audio_io、transforms、scenes、simulation、corpus 模块，
以及生成、验收和语料清单检查脚本。46 项测试与 Ruff 通过。
100 个场景、4 页俯视图、6 组 RIR、30 份合成诊断 WAV 已生成；
两次独立运行的 50 个产物逐字节一致，写盘后 SNR/SIR 与分量加和检查通过。
完整记录见 docs/phase2_validation.md，数据协议见 docs/data_protocol.md。
当时用户没有现成语音/噪声数据；后续第三部分已导入真实测试语料并生成正式测试集。

## 第三部分：真实数据、DAS 与 oracle MVDR（核心完成，人工试听待做）

导入 AISHELL-1 原始测试划分的 200 条语音、20 位说话人，每人 10 条真实文本语句；
读取 30 份真实噪声，测试使用其中 15 份，训练/开发按源文件分离。
200 条多通道测试混合音约 2.2 GB，覆盖线阵 97 条、圆阵 103 条；三档 RT60 均覆盖。
全部文件哈希、纯噪声前缀、SNR/SIR、分量加和及独立重生成检查通过。
59 项测试、Ruff 和依赖检查通过；整条第三阶段复现脚本已跑通。

200 条上，未处理/DAS/oracle MVDR 的平均直达参考 SI-SDR 为
−8.3272/−6.0904/−2.5008 dB，STOI 为 0.4860/0.5242/0.6163。
600 份输出的 SI-SDR、PESQ、STOI 均有效，CER 尚未运行。
五组可供人工试听的音频、方向图、时延图和 48 个跨指标不一致组合已保存。
详见 docs/phase3_validation.md、docs/data_sources.md、results/ref_mic.md。
下一部分为 GCC-PHAT、SRP-PHAT、MUSIC 的 DOA 估计与混响/SNR 分组评测。

## 后续分块明细

| 部分 | 内容 | 对应手册交付 |
| --- | --- | --- |
| 2 | 基础设施完成；真实数据已在第三部分接入 | D2（部分） |
| 3 | 正式测试集、三种前端与增强指标完成；人工试听、CER 仍未完成 | D2、D3（部分） |
| 4 | GCC-PHAT、SRP-PHAT、宽带 MUSIC 和失败分析 | D4 |
| 5 | IRM 协方差 MVDR、轻量掩码网络、自训练及对照 | D5 |
| 6 | 固定 ASR 与 CER、项目一单通道基线、案例分析和消融 | D6、D7 |
| 7 | README、音频 demo、技术总结、简历与问答、仓库发布 | D1、D8 |

项目一的单通道降噪模型尚未定位；接入阶段确认已有权重与接口。
仓库发布前需确定 GitHub 仓库、数据许可与可发布内容；不会以占位表格算完成。
所有 D1–D8 满足前，整体项目保持未完成状态。


## 2026-10-28 更新：第四至第七部分的实际验收

本节覆盖上文的历史待办状态；项目尚未整体完成。

- 第四部分完成：200 条混合音、三种 DOA，共 600 次估计，RT60/SNR 曲线与三个失败谱图已生成并检查。实际 MAE 约 62–64°，强干扰主导源混淆如实保留，不用真值选峰美化结果。
- 第五部分完成：原始 AISHELL train/dev 说话人独立，640/160 个仿真特征；83,137 参数 GRU 在 Ubuntu CPU 完成 20 epoch，最佳 epoch=19，单 batch 576 步预检通过。600 份掩码前端增强指标均有效。见 docs/phase5_validation.md 和 results/masks.md。
- 第六部分核心完成：SenseVoice int8 ONNX 本地解码，200 干净语音加六种前端，共 1400 份成功解码。CER 表见 results/asr.md，十个不同场景的指标不一致分析见 analysis.md。
- 第七部分实验完成：20 位测试说话人的配对阵列/RT60/DOA 误差消融，共 60 次联合房间仿真、1320 行有限分数。见 results/ablations.md、docs/phase6_7_validation.md。
- 75 项测试、全仓 Ruff、两套环境 pip check 通过。训练/掩码/CER/消融日志均在 results/generated/，检查点在 checkpoints/，不加入 Git。

关键负结果：神经 MVDR CER=33.37%，尚不及 DAS 的 28.91%；单通道补充模型 CER=51.85%，虽然 SI-SDR/PESQ 改善但识别退化。
Oracle MVDR 为 17.27%，未处理为 37.89%。不能将训练跑通写成神经方法胜出。

项目一已在 /home/kingcrab/speech-front-end/denoising 补建单通道对照，复用相同掩码权重；没有声称完整项目一 AEC/VAD 已完成。
Colab 尚未验证，此次训练实际使用 Ubuntu CPU。

下一步仅复用已有实验，不重跑训练或整批评测：

1. D1：整合 README、全流程单命令、整理可发布的图/音频及许可证，发布公开 GitHub 仓库。
2. D8：两页技术总结、一页水声到麦克风阵列笔记、三条简历 bullet、三分钟讲述稿、20 题问答。
3. 核对官方 ASR 基准的协议差异，最终 D1–D8 审计；人工听音和练习不冒充完成。

以上待办在下述最终交付中已补齐。

## 最终交付：D1-D8 工程产物齐备（2026-10-28）

公开仓库：https://github.com/kingcrabwang/farfield-asr，public / main。
完整入口 `bash scripts/reproduce.sh` 与 `--verify-only` 验收通过；从公开仓库重新克隆后 75 项测试再次通过。
README 含架构、六前端结果、单命令、公开自制诊断音频及真实语音本地试听入口。
两页技术总结与一页迁移笔记已逐页渲染检查，三条简历、三分钟稿与二十题问答已完成。
完整逐项证据与边界见 docs/delivery_audit.md。

实验数据、模型、连接凭据留在本机；没有购买或使用 Colab 算力。
神经方法未胜 DAS、单通道识别退化和 DOA 大误差均如实报告。
人工试听、口头自测和讲述练习未冒充完成；这些是使用者后续学习活动。
自动项目续做任务到此结束，不再周期性启动训练或评测。
