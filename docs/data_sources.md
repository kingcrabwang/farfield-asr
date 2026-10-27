# 数据来源与依赖记录

## AISHELL-1

原始发布信息：[OpenSLR33](https://www.openslr.org/33/)。原始页面列出 Apache License 2.0，
并含学术用途描述。当前取用的分发为 [ModelScope 官方测试数据](https://www.modelscope.cn/datasets/modelscope/speech_asr_aishell1_testsets)，
其卡片同样列出 Apache 2.0，并保留学术用途/未经许可不得商用的文字。
这些文字原样作为来源条件保留，没有将它们改写为本项目授予的权利。

元数据文件：`aishell1_test.csv`，检查到的文件修订为
`847f9aded68606a8f7308fb3716cd0715be021ba`，SHA256 为
`a028ad8b8318fbe35748c6d565893ca225bee353095e6618255d3abbbf5320c8`。
脚本固定该哈希，来源内容变化时拒绝静默继续。

实际音频从公开分发 ZIP 按成员读取，验证 ZIP CRC，并记录本地 SHA256；
**没有声称做过完整原始 15 GB 包的 MD5 验证**。公共数据临时访问凭证仅在内存中使用，
未写入文件、Git、日志或报告，不使用用户账号的访问令牌。

引用：Bu, Hui; Du, Jiayu; Na, Xingyu; Wu, Bengu; Zheng, Hao.
“AISHELL-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline”,
Oriental COCOSDA, 2017。

## 点源噪声

分发：[OpenSLR28](https://www.openslr.org/28/)，页面将整包标为 Apache 2.0，并说明点源
噪声源于 MUSAN。原始来源：[MUSAN / OpenSLR17](https://www.openslr.org/17/)，
标为 CC BY 4.0。保留源文件名、ZIP 成员路径/CRC、原始来源链接及文件哈希，不将
这些录音当作本项目自行录制的内容或将其来源许可覆盖为代码许可。

引用：Snyder, David; Chen, Guoguo; Povey, Daniel.
“MUSAN: A Music, Speech, and Noise Corpus”, arXiv:1510.08484, 2015。

本阶段选 30 份 `noise-free-sound-*.wav`，测试只用其中 15 份，训练/开发保留不同文件。
所有原始数据和仿真音频留在本地，`data/` 与生成结果不自动进入 Git。
本地 demo 也未上传；对外发布音频前仍需按这些实际来源条件确定可发布的形式。

## 算法依赖

- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)：0.10.1，房间 RIR。
- [pesq Python wrapper](https://pypi.org/project/pesq/)：0.0.4，16 kHz wideband PESQ。
  Python 包装器与其底层 PESQ 实现的许可条件应按源项目分别保留，不据包装器名称推断权利。
- [pystoi](https://pypi.org/project/pystoi/)：0.4.1，标准 STOI。
- [oss2](https://pypi.org/project/oss2/)：2.19.1，仅用于公开语料传输。

原始来源页面保存在 Ubuntu 的 `data/provenance/`，语音完整清单与核验记录也在此目录。
精确运行依赖见本地 `results/generated/pip-freeze.txt` 和 Conda 显式包清单。

