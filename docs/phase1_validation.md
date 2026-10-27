# 第一阶段验收：Ubuntu 环境、阵列与 DAS

验收日期：2026-10-27（Asia/Shanghai）。

## 实际安装位置

- 项目：`/home/kingcrab/farfield-asr`
- Conda：`/home/kingcrab/miniconda3`
- 专用环境：`/home/kingcrab/miniconda3/envs/farfield-asr`
- 可编辑安装源码：`/home/kingcrab/farfield-asr/src/farfield_asr`

## 实测结果

| 验收 | 结果 | 意义 |
| --- | --- | --- |
| Python | 3.11.18 | 使用专用 Conda 解释器 |
| NumPy / SciPy | 2.4.6 / 1.17.1 | 基础数值依赖可用 |
| Matplotlib / soundfile | 3.11.2 / 0.14.0 | 绘图与音频文件依赖已安装 |
| pyroomacoustics | 0.10.1 | Linux 原生房间仿真运行成功 |
| pytest | 8 passed in 0.10s | 几何、相位符号、DAS 保真/白噪声增益和非法输入 |
| Ruff | All checks passed | 当前源码、脚本、测试的静态检查 |
| pip check | No broken requirements found | 已安装包的声明依赖一致 |
| 四麦 RIR | 长度 506 / 504 / 502 / 500 | 四路非零、有限响应，非正式评测集 |
| DAS 方向图 | 已生成并目视检查 | 主瓣与栅瓣标注清楚 |

## 本地复现

```bash
cd /home/kingcrab/farfield-asr
conda activate farfield-asr
python scripts/check_environment.py
python -m pytest -q
ruff check src tests scripts
python -m pip check
python scripts/plot_beampattern.py
```

也可执行 `bash scripts/bootstrap.sh` 完成以上流程并重新导出依赖版本。
已存在环境时该脚本不会自动升级依赖，避免无意改变经过验证的环境。

## 验收文件

- `results/generated/environment_check.json`：实际解释器、源码路径、版本和 RIR 检查。
- `results/generated/das_beampattern.png`：4 麦线阵 1/4 kHz 方向图与栅瓣对照。
- `results/generated/conda-linux-64.explicit.txt`：Conda 包精确构建清单。
- `results/generated/pip-freeze.txt`：包含 pip 安装包在内的 Python 依赖版本。
- `results/generated/bootstrap.log`：首次安装日志。

本阶段尚未进行真实语音 SI-SDR/PESQ/STOI/CER 评测，未训练网络。
方向图与理论白噪声增益不能代替手册中的 D2–D8 实验交付。

