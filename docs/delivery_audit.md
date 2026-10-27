# D1-D8 交付审计

日期：2026-10-28。Ubuntu 主项目 `/home/kingcrab/farfield-asr`。
工程产物与实验可以交付；人工试听、口头自测和求职练习需由使用者实际完成。

| 编号 | 产物与证据 | 核验 |
| --- | --- | --- |
| D1 | README 架构、六前端表、scripts/reproduce.sh、docs/demo.md、docs/assets；公开仓库发布状态见下文 | 完整入口在当前机器成功；另有只读验收入口 |
| D2 | 200 条 AISHELL 原始测试语音、多房间/角度/距离、4 麦线阵/6 麦圆阵、点源+扩散场；所有分量在 data/simulated/test_v1 | 哈希、接收比值、纯噪声前缀、独立重生成；四种指标均有真实结果 |
| D3 | DAS、对角加载 MVDR、oracle、方向响应图 | 600 份传统前端指标有效，oracle 的额外信息限制已说明 |
| D4 | GCC-PHAT、SRP-PHAT、MUSIC、两张分组曲线、三个失败谱图 | 600 份估计；低性能与干扰源混淆如实报告 |
| D5 | 功率 IRM、协方差 Souden MVDR、83,137 参数 GRU、训练日志与检查点、掩码图 | 100-step/单 batch 检查通过，20 epoch，train/dev/test 隔离，600 份新前端指标有效 |
| D6 | results/asr.md、results/ref_mic.md、analysis.md、项目一 denoising 补充模块 | 200 干净加六前端，共 1400 份成功解码；10 个不同场景的跨指标案例 |
| D7 | results/ablations.md、共享联合场/中心参考协议、两个性能曲线 | 20 位说话人、60 次场景仿真、1320 次有效评分，三类消融均有结论 |
| D8 | docs/assets/technical_summary.pdf、underwater_to_microphone.pdf、docs/interview.md | PDF 严格 2 页/1 页，三页逐页渲染检查通过；3 条简历、3 分钟稿、20 题问答 |

完整运行：`bash scripts/reproduce.sh`，退出码 0，日志在本机 `results/generated/full-reproduction.log`。
完整入口复用已核验的训练与评测缓存，没有重新训练；不是另一台新机器的全新下载验收。
75 项测试与 Ruff 通过，两环境 pip check 通过；原始完整归档 MD5 未核验。

## 结果边界

- 真实测试录音、派生语音和预训练模型留在本机，未作为代码许可素材上传。公开音频完全由公式合成，明确标为诊断信号，不用于正式指标。
- 神经 MVDR CER 33.37%，未超过 DAS 28.91%；单通道补充模型 CER 51.85%，比未处理 37.89% 更差。
- Oracle MVDR CER 17.27%，不能解释为可无条件部署系统的性能。
- 项目一仅补建单通道对照，没有声称其完整 AEC/VAD 系统完成。
- 当前全句协方差是离线算法；未证明真实录音泛化、在线延迟或所有总体场景的统计结论。
- 干净 50/200 条子集与官方全量条件不同，对应比较见 docs/asr_baseline_check.md。
- Colab 未验证、未使用；本轮训练使用 Ubuntu CPU。

## 公开仓库

代码与可公开图表准备完成，仓库地址与远端修订在发布成功后补入本节。
