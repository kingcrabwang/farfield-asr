"""在 200 条真实测试混合音上评估三种 DOA，输出分组曲线和失败案例。"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from farfield_asr.audio_io import read_audio
from farfield_asr.doa import angular_error, estimate_doa
from farfield_asr.scenes import Scene

METHODS = ("gcc_phat", "srp_phat", "music")


def main():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data/simulated/test_v1"
    output = root / "results/generated/phase4"
    output.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    rows = []
    scan_cache = {}
    for index, entry in enumerate(manifest, 1):
        scene = Scene(**entry["scene"])
        audio = read_audio(dataset / scene.scene_id / "mixture.wav")
        audio = audio[entry["noise_only_prefix_samples"]:]
        interference = np.subtract(scene.interference_position_m, scene.array_center_m)
        interference_angle = float(np.rad2deg(np.arctan2(interference[1], interference[0])) % 360)
        if scene.array_geometry == "linear" and interference_angle > 180:
            interference_angle = 360 - interference_angle
        for method in METHODS:
            result = estimate_doa(audio, scene.microphones_m, method, scene.array_geometry)
            error = angular_error(result.azimuth_deg, scene.target_azimuth_deg, scene.array_geometry)
            row = {"scene_id": scene.scene_id, "method": method,
                   "array_geometry": scene.array_geometry, "true_azimuth_deg": scene.target_azimuth_deg,
                   "estimated_azimuth_deg": result.azimuth_deg, "error_deg": error,
                   "interference_azimuth_deg": interference_angle,
                   "interference_error_deg": angular_error(result.azimuth_deg, interference_angle,
                                                            scene.array_geometry),
                   "requested_rt60_s": scene.rt60_s, "measured_rt60_s": entry["rt60_t30_s"],
                   "snr_db": scene.snr_db, "sir_db": scene.sir_db,
                   "frequency_bins": result.frequency_bins, "frames": result.frames}
            rows.append(row)
            scan_cache[(scene.scene_id, method)] = result
        print(f"DOA {index}/{len(manifest)}: {scene.scene_id}", flush=True)
    (output / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {"utterances": len(manifest), "frequency_range_hz": [500, 3500],
               "grid_step_deg": 1, "music_signal_subspace_dimension": 1, "methods": {}}
    lines = ["# 三种宽带 DOA 的真实测试结果", "",
             "| 方法 | 样本 | MAE（度） | 中位误差（度） | 误差≤10°比例 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        errors = np.array([r["error_deg"] for r in rows if r["method"] == method])
        result = {"n": len(errors), "mae_deg": float(errors.mean()),
                  "median_error_deg": float(np.median(errors)), "within_10deg": float(np.mean(errors <= 10))}
        summary["methods"][method] = result
        lines.append(f"| {method} | {len(errors)} | {result['mae_deg']:.3f} | "
                     f"{result['median_error_deg']:.3f} | {result['within_10deg']:.3f} |")
    for field, values, label, filename in (
        ("requested_rt60_s", [0.2, 0.4, 0.6], "Requested RT60 (s)", "mae_by_rt60.png"),
        ("snr_db", [-5, 0, 5, 10], "Diffuse-field SNR (dB)", "mae_by_snr.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, geometry in zip(axes, ("linear", "circular")):
            for method in METHODS:
                means = [np.mean([r["error_deg"] for r in rows if r["method"] == method
                                  and r["array_geometry"] == geometry and r[field] == value])
                         for value in values]
                ax.plot(values, means, marker="o", label=method)
            ax.set(title=geometry, xlabel=label, ylabel="Mean absolute azimuth error (deg)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    # 选择三个不同场景的最差定位失败，不以真值选择算法输出的谱峰。
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
    failed_ids = sorted(by_scene, key=lambda key: max(r["error_deg"] for r in by_scene[key]), reverse=True)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lines += ["", "## 三个失败案例", ""]
    for ax, scene_id in zip(axes, failed_ids):
        case = by_scene[scene_id]
        for row in case:
            result = scan_cache[(scene_id, row["method"])]
            ax.plot(result.grid_deg, result.scores, label=row["method"])
        first = case[0]
        ax.axvline(first["true_azimuth_deg"], color="black", linestyle="--", label="target")
        ax.axvline(first["interference_azimuth_deg"], color="red", linestyle=":", label="interference")
        ax.set(title=f"{scene_id}: RT60={first['requested_rt60_s']}, "
                     f"SNR={first['snr_db']}, SIR={first['sir_db']}", xlabel="Azimuth (deg)")
        ax.legend(fontsize=7)
        errors = ", ".join(f"{r['method']}={r['error_deg']:.1f}°" for r in case)
        confused = [r["method"] for r in case if r["interference_error_deg"] < r["error_deg"]]
        lines += [f"### {scene_id}", "", f"误差：{errors}。", "",
                  (f"配置 RT60={first['requested_rt60_s']} s，SNR={first['snr_db']} dB，"
                   f"SIR={first['sir_db']} dB。"), "",
                  (f"估计更接近干扰方向的方法：{', '.join(confused) or '无'}。"
                   "该现象提示可能存在主导源混淆；混响、非平稳噪声和单源模型失配的贡献需另做受控实验。"), ""]
    fig.tight_layout()
    fig.savefig(output / "failure_scans.png", dpi=160)
    plt.close(fig)
    lines += ["## 协议与限制", "",
              "输入只使用混合音，排除已知校准前缀；不读取干净分量或真值方向来选峰。",
              "线阵搜索 0–180°；圆阵搜索 0–359°，误差使用圆周距离。",
              "GCC-PHAT 先求各基线相关峰，再用时差拟合方向；SRP-PHAT 累积相位一致性；",
              "MUSIC 逐频点分解协方差，按归一化伪谱非相干合并，默认单主导源。",
              "这些算法没有说话人身份信息，强干扰可能被选为主导源；没有把此失败隐藏成真值辅助定位。",
              "RT60/SNR 图为当前测试集分组描述，其他场景因素未配对固定，不等价于因果消融。", ""]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (root / "results/doa.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
