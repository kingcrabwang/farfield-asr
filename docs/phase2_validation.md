# 第二阶段：音频接口、可逆 STFT 和仿真框架

验收日期：2026-10-27（Asia/Shanghai）。Ubuntu 项目位置：
`/home/kingcrab/farfield-asr`，使用独立环境 `farfield-asr`。

## 这一阶段完成了什么

1. 音频读写：读取后统一为 `(样本数, 通道数)`；采样率不符立即报错，
   不隐式重采样；输出 float32 WAV，不独立缩放通道，不加入时间戳。
2. STFT / ISTFT：默认 512 点 FFT、128 点步长、周期 Hann 窗；补零边界，
   合成时显式指定原始长度。支持单通道、4/6 通道以及 DAS 单通道谱输出。
3. 场景配置：100 条固定种子 JSONL，覆盖三种房间、两种阵列、三档设定 RT60、
   两种距离、三种目标方位、四档 SNR 与三档 SIR；俯视图共 4 页，每页 25 个场景。
4. 房间仿真：两种阵列 × 三档 RT60，共 6 组目标声源、点源干扰和直达路径 RIR。
   使用 `inverse_sabine` 推荐反射阶数，保留完整尾部。
5. 扩散噪声：三维球面各向同性噪声，频率相关空间相干性满足
   `Γ_ij(f) = sinc(2 f d_ij / c)`；使用 NumPy 的归一化 sinc。
   这与各麦克风独立生成白噪声不同，已用理论宽带相关系数验证。
6. 分量保存：每组保存混合声、目标混响声、目标直达声、点源干扰、扩散噪声五份 WAV，
   以及 `rir.npz` 和元数据。当前示例为扫频与随机噪声，**不是语音数据**。
7. 真实语音清单校验接口：检查文本、来源/许可声明、说话人集合隔离、音频采样率、
   单声道、非零能量和跨集合重复文件；尚未导入正式语料。

## 统一的信号和指标约定

- 时域：`(N,M)`；频域：`(F,T,M)`；频域 DAS 输出 `(F,T)`。
- SNR：参考麦克风处「目标混响声 / 扩散噪声」的全长度平均功率比。
- SIR：参考麦克风处「目标混响声 / 点源干扰」的全长度平均功率比。
- SINR 与上述两个量不同：其分母应包含点源干扰和扩散噪声的混合，不混用名称。
- 主评测参考计划使用第 0 个麦克风处的目标直达声，保留传播时延及分数时延 FIR
  的公共群延迟；不是直接拿无传播时延的原始干净音频当参考。
- 所有波形使用同一个最终增益，混合峰值不超过 0.95；各分量仍严格可加。
- STFT 两端补零并在 ISTFT 裁回原长，此流程属于离线处理，不代表流式零延迟。

## RT60 实际测量

RT60 表示声能衰减 60 dB 所需时间。这里对 RIR 做 Schroeder 反向能量积分，
在 −5 到 −35 dB 区间拟合，再外推 RT60（T30 方法）；保存拟合区间和 R²。
Sabine 反演的配置值是近似值，不能直接当作实测真值。

| 阵列 | 设定 RT60（秒） | T30 外推 RT60（秒） |
| --- | --- | --- |
| 4 麦线阵 | 0.2 | 0.180 |
| 4 麦线阵 | 0.4 | 0.472 |
| 4 麦线阵 | 0.6 | 0.784 |
| 6 麦圆阵 | 0.2 | 0.184 |
| 6 麦圆阵 | 0.4 | 0.481 |
| 6 麦圆阵 | 0.6 | 0.817 |

每种阵列的三次 RT60 对比固定了房间与声源/阵列位置，只改变吸声相关参数。
两种阵列的示例取自不同场景，**这张表不能用来得出阵列几何造成 RT60 差异的结论**。
后续消融必须固定房间、位置、语料与噪声，只改变所研究因素。
对较高 RT60 的偏差原因仍需研究：Sabine 扩散场近似、有限房间几何、源/接收位置和
拟合区间都会影响结果；本阶段未证实具体贡献。正式数据按设定值和实测值同时记录/分组，
若要求精确 RT60，应先标定吸声系数，而不是改写测量结果。

DRR 是直达声能量与混响声能量之比，关注直达/反射分配；RT60 关注衰减速度。
两者不是同一指标，同 RT60 下距离改变仍可显著改变 DRR。

![RIR 衰减曲线](../results/generated/phase2/rir_decay.png)

## 实际验收证据

- `46 passed in 1.60s`：包含第一阶段测试、边界音频/多通道重建、DAS 波形链路、
  几何约束、扩散场相关性、RIR 传播时延/复现、分量混合和语音清单泄漏检查。
- Ruff：`All checks passed!`。
- 两次独立运行：50 个实验文件逐字节一致，包括 JSON、RIR、WAV 和 PNG。
  该结论适用于当前安装环境；不宣称不同 BLAS/平台下仍逐字节一致。
- 从保存的 WAV 重新计算：SNR 均为 5 dB、SIR 均为 0 dB，误差小于 `1e-5 dB`。
- WAV 分量可加性最大绝对误差约 `6.34e-8`，符合 float32 量化量级。
- 每组实际写盘的多通道 WAV 均完成 STFT/ISTFT 重建检查，最大误差小于 `1e-10`。

最终产物：`results/generated/phase2/`，约 31 MB；早期诊断和重复运行保存在
`results/generated/archive/`，正式脚本不会覆盖非空输出目录。

## 复现命令

```bash
conda activate farfield-asr
cd /home/kingcrab/farfield-asr
python -m pytest -q
ruff check src tests scripts

# 使用新目录，避免覆盖已验收结果。
python scripts/run_phase2.py --output results/generated/phase2-new
python scripts/verify_phase2.py results/generated/phase2-new

# 可选：再次独立运行并校验全部产物哈希。
python scripts/run_phase2.py --output results/generated/phase2-new-repeat
python scripts/verify_phase2.py results/generated/phase2-new \
    --repeat results/generated/phase2-new-repeat
```

`summary.json` 记录全部示例参数、实测 SNR/SIR、RT60 拟合、STFT 重建误差和文件哈希；
`verification.json` 记录从 WAV 重新验证的结果；`reproducibility.json` 记录重复运行比较。

## 尚未完成的完整项目目标

用户目前没有语音/噪声数据；接下来需获取并核对公开语料来源和许可，导入真实文本与
说话人信息，再生成 200 条正式测试多通道 WAV。本阶段不计算语音增强分数或 CER，
不将合成扫频作为语音或音频 demo。D2 整体尚未完成。

参考：[pyroomacoustics 官方 RT60 示例](https://github.com/LCAV/pyroomacoustics/blob/master/examples/room_from_rt60.py)、
[Sabine 反演接口](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.acoustics.html)。

