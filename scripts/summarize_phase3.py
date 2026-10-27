"""汇总实际指标、跨指标不一致案例，并导出五组可供人工试听的音频。"""

import json
import shutil
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio, write_audio


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/generated/phase3"
    manifest = [json.loads(line) for line in (root / "data/simulated/test_v1/manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if len(scores) != 600 or any(
        count != 200 for value in summary["methods"].values() for count in value["valid_counts"].values()
    ):
        raise ValueError("expected 600 scored outputs and 200 valid values for every metric")
    if any(score["metric_errors"] for score in scores):
        raise ValueError("per-utterance metric failures require review")
    baseline = {score["scene_id"]: score for score in scores if score["method"] == "ref_mic"}
    improvements = {}
    disagreements = []
    for method in ("das", "oracle_mvdr"):
        selected = [score for score in scores if score["method"] == method]
        improvements[method] = {}
        for metric in ("si_sdr_db", "pesq_wb", "stoi"):
            deltas = [score[metric] - baseline[score["scene_id"]][metric] for score in selected]
            improvements[method][metric] = {"mean_delta": float(np.mean(deltas)),
                                            "improved_cases": int(np.count_nonzero(np.array(deltas) > 0)),
                                            "total_cases": len(deltas)}
        for score in selected:
            original = baseline[score["scene_id"]]
            if score["si_sdr_db"] > original["si_sdr_db"] and score["pesq_wb"] < original["pesq_wb"]:
                disagreements.append({"scene_id": score["scene_id"], "method": method,
                                      "si_sdr_delta_db": score["si_sdr_db"] - original["si_sdr_db"],
                                      "pesq_delta": score["pesq_wb"] - original["pesq_wb"]})
    report = {"status": "passed", "scored_outputs": len(scores), "metric_failures": 0,
              "max_output_peak": max(score["output_peak"] for score in scores),
              "samples_abs_gt1": sum(score["samples_abs_gt1"] for score in scores),
              "paired_improvements": improvements, "si_sdr_up_pesq_down_cases": disagreements,
              "cer_status": "not run; reserved for ASR stage"}
    (output / "evaluation_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected, combinations = [], set()
    for entry in manifest:
        key = (entry["scene"]["array_geometry"], entry["scene"]["rt60_s"])
        if key not in combinations:
            selected.append(entry)
            combinations.add(key)
        if len(selected) == 5:
            break
    demo_root = output / "demo"
    demo_lines = ["# 五组真实语音前端对比", "",
                  "以下文件供人工试听。程序已检查文件完整性；不将自动数值检查称作人工听音。", "",
                  "统一去掉 0.6 秒噪声校准段，保留目标传播时延与完整混响尾部；不独立归一化。", ""]
    for entry in selected:
        scene_id = entry["scene"]["scene_id"]
        folder = demo_root / scene_id
        folder.mkdir(parents=True, exist_ok=True)
        demo_lines += [f"## {scene_id}", "", f"文本：{entry['transcript']}", "",
                       f"阵列：{entry['scene']['array_geometry']}，设定 RT60：{entry['scene']['rt60_s']} 秒。", ""]
        for method in ("ref_mic", "das", "oracle_mvdr"):
            audio = read_audio(output / "audio" / method / f"{scene_id}.wav")[:, 0]
            write_audio(folder / f"{method}.wav", audio[entry["noise_only_prefix_samples"]:],
                        overwrite=True)
            demo_lines.append(f"- [{method}](../results/generated/phase3/demo/{scene_id}/{method}.wav)")
        demo_lines.append("")
    (root / "docs/demo_phase3.md").write_text("\n".join(demo_lines), encoding="utf-8")
    (root / "results").mkdir(exist_ok=True)
    shutil.copyfile(output / "results.md", root / "results/ref_mic.md")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "si_sdr_up_pesq_down_cases"}, ensure_ascii=False, indent=2))
    print("SI-SDR-up/PESQ-down cases:", len(disagreements))


if __name__ == "__main__":
    main()
