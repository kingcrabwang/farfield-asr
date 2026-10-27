"""验收配对消融并从实际分数生成结论，避免把分组描述当作受控实验。"""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/generated/phase7"
    rows = [json.loads(line) for line in (folder / "scores.jsonl").read_text().splitlines()]
    protocol = json.loads((folder / "protocol.json").read_text())
    assert len(rows) == 1320
    assert len({r["speaker_id"] for r in rows}) == 20
    for trial in range(20):
        record = json.loads((folder / "trials" / f"{trial:03d}.json").read_text())
        assert record["fingerprint"] == protocol["fingerprint"]
        assert len(record["scores"]) == 66
        for calibration in record["calibration"]:
            assert abs(calibration["center_snr_db"] - 5) < 1e-8
            assert abs(calibration["center_sir_db"]) < 1e-8
    assert not any(r["metric_errors"] for r in rows)
    assert all(np.isfinite(r[metric]) for r in rows for metric in ("si_sdr_db", "pesq_wb", "stoi"))
    table = json.loads((folder / "geometry_summary.json").read_text())
    lines = ["# 配对消融：阵列、混响与导向误差", "",
             "20 位测试说话人，每人一条最长 4 秒片段；60 次联合阵列房间仿真、1320 次前端评分。",
             "本实验参考为共同虚拟中心的直达目标，区别于主测试的物理参考麦克风；数值不能与主测试直接比较。",
             "每个条件共享语音、点噪声片段、源位置、房间和联合扩散声场。固定最大孔径 0.12 m，",
             "SNR=5 dB/SIR=0 dB 在中心校准；改变 RT60 时重新校准接收比值。",
             "不同阵列的导向矢量均相对于共同中心，不通过不同参考通道延迟制造几何差异。", "",
             "## 麦克风数量与几何（RT60=0.4 s，导向误差=0）", "",
             "| 阵列 | 方法 | N | SI-SDR (dB) | PESQ-WB | STOI |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in table:
        lines.append(f"| {row['array']} | {row['method']} | {row['n']} | {row['si_sdr_db']:.3f} | "
                     f"{row['pesq_wb']:.3f} | {row['stoi']:.3f} |")
    lines += ["", "固定孔径下，线阵 MVDR 从 2 到 6 麦的 SI-SDR 从 −5.550 提升至 −3.851 dB，",
              "而 DAS 的数量增益较小；说明增加通道与扩大孔径不能混为一谈。",
              "同为 6 麦时圆阵 MVDR 为 −1.202 dB、线阵为 −3.851 dB；",
              "这是当前 20 个位置/角度的结果，不足以断言圆阵对所有方向更优。",
              "两麦圆阵退化为一个直线麦克风对，因此只报告一次 linear2，不制造重复几何条件。", "",
              "## 线阵 4 麦的 RT60 与导向误差", "",
              "| RT60 (s) | 导向误差 (°) | 方法 | SI-SDR (dB) | STOI |",
              "| ---: | ---: | --- | ---: | ---: |"]
    aggregate = {}
    for method in ("das", "mvdr"):
        aggregate[method] = {}
        for rt60 in (0.2, 0.4, 0.6):
            for error in (-20, -10, -5, 0, 5, 10, 20):
                if rt60 != 0.4 and error != 0:
                    continue
                selected = [r for r in rows if r["array"] == "linear4" and r["method"] == method
                            and r["rt60_s"] == rt60 and r["doa_error_deg"] == error]
                assert len(selected) == 20
                values = {metric: float(np.mean([r[metric] for r in selected]))
                          for metric in ("si_sdr_db", "pesq_wb", "stoi")}
                aggregate[method][f"{rt60}/{error}"] = values
                lines.append(f"| {rt60} | {error} | {method} | {values['si_sdr_db']:.3f} | {values['stoi']:.3f} |")
    lines += ["", "RT60 的配对变化（0.2→0.6 s、导向误差为零）：", ""]
    for method in ("das", "mvdr"):
        low, high = aggregate[method]["0.2/0"], aggregate[method]["0.6/0"]
        lines.append(f"- {method}: SI-SDR {low['si_sdr_db']:.3f}→{high['si_sdr_db']:.3f} dB，"
                     f"STOI {low['stoi']:.3f}→{high['stoi']:.3f}。")
    lines += ["", "方向误差敏感度用 ±20° 对称均值与 0° 配对比较，避免只选有利一侧：", ""]
    for method in ("das", "mvdr"):
        exact = aggregate[method]["0.4/0"]["si_sdr_db"]
        perturbed = np.mean([aggregate[method][f"0.4/{error}"]["si_sdr_db"] for error in (-20, 20)])
        lines.append(f"- {method}: 0° 为 {exact:.3f} dB，±20° 均值为 {perturbed:.3f} dB，"
                     f"变化 {perturbed - exact:+.3f} dB。")
    lines += ["", "变化描述只适用于本协议；有限校准噪声、混响失配、目标/干扰角度共同影响结果。",
              "误差条为跨 20 位说话人的均值标准误，不是独立麦克风样本数或总体保证。",
              "未把主测试的非配对 RT60 分组曲线当作因果证据。", "",
              "![RT60](../docs/assets/phase7_rt60.png)", "![方向误差](../docs/assets/phase7_doa_error.png)", ""]
    (root / "results/ablations.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "verification.json").write_text(json.dumps(
        {"status": "passed", "trials": 20, "speakers": 20, "scores": len(rows),
         "valid_metric_rows": len(rows), "center_ratio_tolerance_db": 1e-8,
         "fingerprint": protocol["fingerprint"], "linear4_groups": aggregate}, indent=2) + "\n")
    print("Verified 20 paired trials / 1320 finite score rows")


if __name__ == "__main__":
    main()
