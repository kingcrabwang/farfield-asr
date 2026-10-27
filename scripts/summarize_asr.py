"""将实际完整 ASR 缓存与增强指标合并，导出十个跨指标不一致案例。"""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/generated/phase6"
    summary = json.loads((output / "summary.json").read_text())
    methods = ("ref_mic", "das", "oracle_mvdr", "irm_mvdr", "neural_mvdr", "project1_singlechannel")
    for method in ("clean", *methods):
        record = summary["methods"][method]
        if record["n"] != 200 or record["valid"] != 200 or record["failed"]:
            raise ValueError(f"incomplete ASR evaluation: {method}")
    acoustic = {}
    for phase in ("phase3", "phase5"):
        for line in (root / f"results/generated/{phase}/scores.jsonl").read_text().splitlines():
            row = json.loads(line)
            acoustic[row["scene_id"], row["method"]] = row
    scores = {}
    for path in (output / "scores").glob("*/*.json"):
        row = json.loads(path.read_text())
        scores[row["scene_id"], row["method"]] = row
    lines = ["# 固定 ASR 与全前端评测", "",
             "相同 200 条测试语音、相同文本归一化、SenseVoiceSmall int8 ONNX CPU。",
             "CER 为总编辑数 / 总参考字符数，不是逐句 CER 的算术平均。",
             "固定中文、关闭 ITN、batch=1，去控制 token，不调用添加 emoji 的后处理。", "",
             "| 前端 | N | SI-SDR (dB) | PESQ-WB | STOI | CER |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    combined = {}
    for phase in ("phase3", "phase5"):
        combined.update(json.loads((root / f"results/generated/{phase}/summary.json").read_text())["methods"])
    for method in methods:
        metrics = combined[method]["metrics"]
        cer = summary["methods"][method]["cer"]
        lines.append(f"| {method} | 200 | {metrics['si_sdr_db']:.4f} | {metrics['pesq_wb']:.4f} | "
                     f"{metrics['stoi']:.4f} | {cer:.2%} |")
    lines += ["", f"干净源语音 200 条 CER：{summary['methods']['clean']['cer']:.2%}。",
              f"固定说话人平衡的 50 条干净语音 CER：{summary['clean_50_gate']['cer']:.2%}。",
              "此小子集不是官方全量 AISHELL 测试集，不据此声称复现官方精度。", "",
              "IRM 使用真实目标/噪声分量，oracle MVDR 使用真实方向及已知纯噪声校准前缀；两者不是可部署系统。",
              "project1_singlechannel 为本轮补建并与神经 MVDR 共用权重的单通道对照。", "",
              "原始解码、输入和模型哈希、耗时、每句编辑数保存在 results/generated/phase6/。", ""]
    (root / "results/asr.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "results/ref_mic.md").write_text("\n".join(lines), encoding="utf-8")
    cases = []
    for (scene_id, method), row in acoustic.items():
        if method == "ref_mic":
            continue
        baseline = acoustic[scene_id, "ref_mic"]
        delta_si = row["si_sdr_db"] - baseline["si_sdr_db"]
        delta_pesq = row["pesq_wb"] - baseline["pesq_wb"]
        delta_cer = scores[scene_id, method]["cer"] - scores[scene_id, "ref_mic"]["cer"]
        if (delta_si * delta_cer > 0) or (delta_pesq * delta_cer > 0):
            cases.append({"scene_id": scene_id, "method": method, "delta_si_sdr_db": delta_si,
                          "delta_pesq": delta_pesq, "delta_cer": delta_cer,
                          "reference": scores[scene_id, method]["reference"],
                          "baseline_hypothesis": scores[scene_id, "ref_mic"]["hypothesis"],
                          "enhanced_hypothesis": scores[scene_id, method]["hypothesis"]})
    cases.sort(key=lambda row: abs(row["delta_cer"]), reverse=True)
    chosen, seen = [], set()
    for row in cases:
        if row["scene_id"] not in seen:
            chosen.append(row)
            seen.add(row["scene_id"])
        if len(chosen) == 10:
            break
    if len(chosen) < 10:
        raise ValueError("fewer than ten distinct mismatch cases; report actual count and inspect")
    (output / "mismatch_cases.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n")
    analysis = ["# 十个增强指标与 ASR 不一致的案例", "",
                "按相对未处理参考通道的 CER 变化幅度排序，每个场景最多一例，固定选前十个。",
                "正 ΔSI-SDR/ΔPESQ 表示增强指标改善；负 ΔCER 表示识别改善。以下文字是待验证假设，不是听音或因果证明。", ""]
    for index, row in enumerate(chosen, 1):
        analysis += [f"## {index}. {row['scene_id']} / {row['method']}", "",
                     (f"ΔSI-SDR={row['delta_si_sdr_db']:+.3f} dB，ΔPESQ={row['delta_pesq']:+.3f}，"
                      f"ΔCER={row['delta_cer']:+.2%}。"), "",
                     f"参考：{row['reference']}", "", f"未处理：{row['baseline_hypothesis']}", "",
                     f"处理后：{row['enhanced_hypothesis']}", ""]
        if row["delta_cer"] > 0:
            explanation = ("波形误差或感知质量改善，但关键音素仍可能被掩码/空间滤波抑制；"
                           "ASR 声学分布偏移、短句插入错误也可能放大 CER。应检查错字对应时频区域，"
                           "再用保守增益或掩码下限作配对干预，才能验证原因。")
        else:
            explanation = ("ASR 可能更受残余干扰结构和音素可辨性影响，而非整体波形逼近；"
                           "直达参考会把保留混响算作失真。可增加混响目标参考 SI-SDR、"
                           "核查替换/删除位置，并通过固定模型的滤波强度扫描验证这一假设。")
        analysis += ["原因假设与验证办法：" + explanation, ""]
    (root / "analysis.md").write_text("\n".join(analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
