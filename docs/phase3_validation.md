# 第三阶段：真实测试数据、DAS、oracle MVDR 与统一评测

项目：`/home/kingcrab/farfield-asr`；环境：`farfield-asr`。本阶段完成 200 条真实语音仿真
测试数据、三种前端和 SI-SDR/PESQ/STOI 的批量评测。ASR 尚未运行，CER 明确标为未运行。

## 实际结果

以下为同一组 200 条语句的逐句算术平均；三项指标每种前端均有 200 个有效值，无失败项。

| 前端 | SI-SDR（dB） | PESQ-WB | STOI | CER |
| --- | ---: | ---: | ---: | --- |
| 参考麦克风未处理 | −8.3272 | 1.0638 | 0.4860 | 未运行 |
| DAS（真值方向） | −6.0904 | 1.0790 | 0.5242 | 未运行 |
| oracle MVDR（真值方向、纯噪声段） | −2.5008 | 1.1024 | 0.6163 | 未运行 |

对同一条语句配对比较：DAS 平均 SI-SDR 提升 2.2368 dB，oracle MVDR 提升 5.8264 dB。
两者各在 199/200 条上提升 SI-SDR；STOI 改善分别为 198/200、200/200 条。
不能将均值改善理解为每个场景、每种指标都改善。已保存 48 个“SI-SDR 上升而 PESQ
下降”的（场景，前端）组合，供后续案例分析；它们不是 48 个独立语句，也不是 CER 案例。

![分组结果](../results/generated/phase3/metrics_by_rt60.png)

图中的误差棒是语句间标准误，横轴为**设定** RT60。各组有不同语句、SNR、SIR 和几何，
这只是当前测试集描述，不能当作固定其他条件的 RT60 消融；正式配对消融留到 D7。
实际 RT60 的 T30 拟合值逐条保存在清单中，未用设定值替换测量值。

## 数据从哪里来

- 语音：AISHELL-1 原始测试划分，由 ModelScope 官方分发的数据与文本清单按需读取；
  固定元数据 SHA256，ZIP 成员 CRC 校验，并保存每个落盘原始 WAV 的 SHA256。
- 抽样：原始 20 位测试说话人，每人 10 条完整语句，共 200 条；仅选 2–8 秒的单声道
  16 kHz 音频，使用种子 20261027 的稳定哈希排序，不截断语句或猜测文本。
- 点源噪声：OpenSLR28 官方 ZIP 的 MUSAN 点源真实噪声，按需读取 30 份；其中 15 份
  测试、10 份训练、5 份开发，按源文件分离。本阶段混合只使用测试噪声。
- 扩散噪声：本项目三维球面各向同性白扩散场模型；独立于真实点源噪声保存。

这是 AISHELL-1 **测试子集**上的仿真评测，不是完整 AISHELL-1 官方识别成绩。
没有把原始训练说话人改标成测试说话人；也未声称任何未运行的 ASR 模型在这些数据上
“从未见过”。来源、许可声明和分发限制见 [数据来源记录](data_sources.md)。

## 200 条测试场景

位置：`data/simulated/test_v1/`，约 2.2 GB。
线阵 97 条、圆阵 103 条；设定 RT60 0.2/0.4/0.6 秒分别为 72/64/64 条。
覆盖三种房间、不同距离和角度、四档 SNR、三档 SIR，沿用第二阶段的场景配置。
这不是麦克风数或几何的受控对照实验。

每条场景保存：

```text
scene_0000/
  mixture.wav              多通道混合音
  target_reverberant.wav    多通道目标混响声
  target_direct.wav         多通道目标直达声
  interference.wav          经房间传播的真实点源噪声
  diffuse.wav               合成的空间相关扩散噪声
  rir.npz                   目标/干扰/直达路径 RIR
  metadata.json             文本、源 ID/哈希、场景、RT60、SNR/SIR、缩放与校准信息
manifest.jsonl              200 条完整记录
```

每条混合音前 0.6 秒目标不发声，但噪声活动。校准帧完全落在该前缀内，并排除最初
0.1 秒的传播起始段；512 点窗、128 点步长下使用 58 帧。所有波形保留传播时延和
pyroomacoustics 分数时延 FIR 的公共群延迟。

## 前端与指标协议

`Frontend(method, scene, noise_only_prefix_samples).process(audio)` 接受 `(N,M)` 音频，
返回相同长度的 `(N,)` 单通道波形，三种前端共用同一个 STFT 和合成接口。

MVDR 的噪声协方差只从已知纯噪声校准帧估计，不用混合音中含目标的片段。
求解 `R_loaded z = a`，再归一化为 `w = z/(aᴴz)`；不显式求逆。
对角加载是 `1e-3 * trace(R)/M + 1e-12`，处理低秩和静音频点。
oracle 名称表示已知目标方向和已知纯噪声时段，是有特权的参考；有限校准数据、
噪声非平稳、有限距离和混响造成的导向失配都可能影响结果，不是所有指标的严格上界。

主参考为第 0 个麦克风处的目标直达声。每种前端都只去掉同样的 0.6 秒校准段，
保留传播时延、完整语句与混响尾部；不按输出搜索最佳对齐，不独立归一化。
PESQ 使用 16 kHz wideband，STOI 使用标准 STOI 而非 ESTOI。
评测读取**落盘后**的前端 WAV，后续 ASR 将读取相同文件。

同时保存以目标混响声为参考的 SI-SDR。其均值分别为 −2.3825、0.0892、−0.0548 dB，
DAS 和 MVDR 的排序与直达声参考下不同。这说明“保留混响目标”与“逼近直达声”是
不同评价目标；不能把两个参考的结果混进同一列。

当前整体 PESQ 接近下限，不能称为高保真增强。强混响/低信噪比、对白直达参考的残余
混响以及 MVDR 失配都值得进一步研究；本阶段没有用猜测代替控制实验来断言具体原因。

![方向图](../results/generated/phase3/beampatterns.png)

![时延检查](../results/generated/phase3/delay_check.png)

时延图将整数采样峰值、球面传播模型和远场模型分开；16 kHz 的一采样间隔为
0.0625 ms，不能把整数峰值的量化差异直接当作分数时延实现错误。

## 验收

- 59 项测试通过；Ruff 与 pip 依赖检查通过。
- 200 条原始语音来自 20 位原始测试说话人，每条有真实文本和源文件哈希。
- 所有生成文件的哈希、通道、长度和有限值检查通过。
- 混合分量加和最大误差 `6.333e-8`；SNR/SIR 最大误差 `2.255e-8 dB`。
- 纯噪声前缀中目标数值泄漏最大 `1.960e-16`。
- 独立重生成第一条真实场景：5 份 WAV 和 RIR 共 6 个文件逐字节一致。
- 600 个前端输出均有三项有效指标，无指标失败；输出没有幅值大于 1 的样本。
- `scripts/run_phase3.sh` 已完整跑通，缓存文件复用前重新校验哈希。

五组音频对比见 [本地试听页面](demo_phase3.md)。文件已做数值检查；仍需人工试听，
本阶段不声称已经完成了人工听音验收。

## 使用方式

```bash
conda activate farfield-asr
cd /home/kingcrab/farfield-asr
bash scripts/run_phase3.sh
```

按需单独运行：

```bash
python scripts/verify_testset.py --reproduce-one
python scripts/evaluate_frontends.py
python scripts/plot_phase3.py
python scripts/summarize_phase3.py
```

新增依赖已写入 `environment.yml`，也可以使用本地包的 `[evaluation,data]` extras。
首次下载需要 curl 和可访问数据分发站点；当前完整数据已保存在 Ubuntu 本地。
生成和评测支持校验后的断点继续；代码/配置指纹不同会拒绝复用旧结果，不静默混用。

主要证据：`results/generated/phase3/` 中的 `summary.json`、`scores.jsonl`、
`dataset_verification.json`、`evaluation_verification.json` 以及逐条 `scores/`。
版本化结果表是 `results/ref_mic.md`。依赖实际版本继续保存在 `results/generated/`。

## 完整手册仍缺什么

本阶段未接 ASR、未产生 CER，也未进行神经掩码训练。CER 的手写编辑距离、固定文字
归一化和字符加权聚合已有单测；真实 CER 必须等 ASR 推理后计算。
D2/D3 的部分内容已完成；人工试听、DOA、IRM/神经 MVDR、单通道基线、ASR、
受控消融与最终发布等仍需按后续阶段推进，不能据此标记整个项目完成。

