# 固定 ASR 与受控消融验收

日期：2026-10-28。主目录 `/home/kingcrab/farfield-asr`。

## ASR

使用独立环境 `farfield-asr-decode`，Python 3.11、NumPy 1.26.4、
funasr-onnx 0.4.1、onnxruntime 1.22.1、torch/torchaudio 2.6.0+cpu。
与主前端环境的 NumPy 2.x 隔离；两环境 pip check 均通过。

官方 `iic/SenseVoiceSmall-onnx` 的 int8 权重、配置、CMVN 和 token 文件逐一核验 SHA256。
旧版运行时还需官方 `iic/SenseVoiceSmall` 的 SentencePiece 文件，固定 revision
`70514a3da51f1160f51d18449dab6128bbd4928b`，不以自造 tokenizer 替代。
所有模型和语料都只保存在本机，不加入 Git。

200 条干净语音和六种前端，共 1400 条解码均成功。
主评测每种前端使用相同 200 条测试语音、相同文本和 0.6 秒前缀裁剪。
固定中文、woitn、batch=1、CPU 两线程；移除 SenseVoice 控制 token，保留原始输出。
CER 为总编辑数除以总参考字符数；空预测按删除错误计分，异常中止并保留错误记录。
缓存绑定模型文件、解码配置、代码和音频哈希。

结果见 [ASR 表](../results/asr.md)，十个不同场景的跨指标不一致案例见 [分析](../analysis.md)。
参考/DAS/oracle-MVDR/IRM-MVDR/神经-MVDR/单通道补充模型的 CER 分别为
37.89%/28.91%/17.27%/17.41%/33.37%/51.85%。
神经模型未胜过 DAS；单通道模型虽然改善 SI-SDR/PESQ，却明显损害 CER，报告保留这一负结果。
原因部分是可检验假设，未冒充人工听音结论。

干净 200 条 CER 为 2.66%，固定 50 条说话人平衡子集为 1.75%。
小子集与官方全量基准不等价；官方精度的对应条件仍需在最终文档中核对。

```bash
conda env create -f environment-asr.yml
bash scripts/install_torch_cpu.sh
conda run -n farfield-asr-decode pip install -e .
conda run -n farfield-asr-decode python scripts/prepare_asr_model.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run -n farfield-asr-decode python scripts/evaluate_asr.py
python scripts/summarize_asr.py
```

## 消融

20 位原始测试说话人，每人一条最长 4 秒片段，未参与训练或选权重。
固定最大孔径 0.12 m，比较线阵 2/4/6 麦与圆阵 4/6 麦。
各条件的麦克风联合仿真，使用一个共享扩散声场；校准与指标参考均为共同虚拟中心。
RT60=0.2/0.4/0.6 秒，接收 SNR=5 dB、SIR=0 dB 在中心重新校准。
4 麦线阵额外测试导向误差 −20/−10/−5/0/5/10/20 度。

完成 60 次联合场景仿真和 1320 次前端评分，全部有限且无指标错误。
每个 trial 的代码/源文件哈希、每个 RT60 的混合 float32 字节哈希、中心接收比值和 T30 均保存。
验收检查 20 位说话人、每个 trial 66 行、中心 SNR/SIR 误差小于 1e-8 dB。
曲线误差条为跨说话人均值标准误，已检查可读性。

结果与逐项结论见 [消融报告](../results/ablations.md)。
该中心参考实验的数值不能与主测试物理参考麦克风的数值直接拼接。
两麦圆阵退化为一个麦克风对，已用 linear2 表示；没有重复命名制造几何差异。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_ablations.py
python scripts/summarize_ablations.py
```

## 交付补充

D1 的公开 GitHub 发布、一条命令全流程与 README，以及 D8 的两页技术总结、
一页迁移笔记、简历 bullet、讲述稿和 20 题问答已在最终整理中补齐，见 docs/delivery_audit.md。
五组音频已准备，但不把用户人工试听/口头自测写成机器已经完成的事项。
