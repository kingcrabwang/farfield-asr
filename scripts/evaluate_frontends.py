"""固定数据与裁剪协议，对未处理、DAS、纯噪声协方差 oracle MVDR 逐条评测。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio, write_audio
from farfield_asr.frontends import Frontend
from farfield_asr.metrics import audio_metrics, si_sdr
from farfield_asr.scenes import Scene

METHODS = ("ref_mic", "das", "oracle_mvdr")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data/simulated/test_v1")
    parser.add_argument("--output", type=Path, default=root / "results/generated/phase3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("empty simulation manifest")
    all_scores = []
    code_paths = [Path(__file__)] + [root / "src/farfield_asr" / f"{name}.py" for name in
                                  ("frontends", "beamforming", "transforms", "metrics", "audio_io")]
    evaluation_hash = hashlib.sha256(b"".join(path.read_bytes() for path in code_paths)).hexdigest()
    for index, row in enumerate(rows, 1):
        scene = Scene(**row["scene"])
        folder = args.dataset / scene.scene_id
        mixture = read_audio(folder / "mixture.wav", scene.sample_rate)
        direct = read_audio(folder / "target_direct.wav", scene.sample_rate)[:, scene.reference_mic]
        reverberant = read_audio(folder / "target_reverberant.wav", scene.sample_rate)[:, scene.reference_mic]
        # 所有前端使用相同区间，不按输出做时延搜索或独立重采样。
        start = row["noise_only_prefix_samples"]
        for method in METHODS:
            target_path = args.output / "audio" / method / f"{scene.scene_id}.wav"
            score_path = args.output / "scores" / method / f"{scene.scene_id}.json"
            if score_path.exists():
                score = json.loads(score_path.read_text(encoding="utf-8"))
                if (score["input_fingerprint"] != row["input_fingerprint"]
                        or score["evaluation_code_sha256"] != evaluation_hash):
                    raise ValueError("refusing to reuse scores from different input or code")
                if hashlib.sha256(target_path.read_bytes()).hexdigest() != score["output_sha256"]:
                    raise ValueError("cached frontend audio hash mismatch")
            else:
                processed = Frontend(method, scene, start).process(mixture)
                write_audio(target_path, processed, scene.sample_rate, overwrite=True)
                # 指标使用最终写盘波形，以保证后续 ASR 读取同一份前端输出。
                processed = read_audio(target_path, scene.sample_rate)[:, 0]
                score = {"scene_id": scene.scene_id, "method": method,
                         "evaluation_code_sha256": evaluation_hash,
                         "output_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                         "input_fingerprint": row["input_fingerprint"],
                         "utterance_id": row["utterance_id"], "transcript": row["transcript"],
                         "array_geometry": scene.array_geometry, "n_mics": scene.n_mics,
                         "requested_rt60_s": scene.rt60_s, "measured_rt60_s": row["rt60_t30_s"],
                         "audio_path": str(target_path.resolve()), "metric_start_sample": start,
                         "output_peak": float(np.max(np.abs(processed))),
                         "samples_abs_gt1": int(np.count_nonzero(np.abs(processed) > 1)),
                         "metric_n_samples": len(processed) - start,
                         **audio_metrics(direct[start:], processed[start:], scene.sample_rate),
                         "si_sdr_reverberant_db": si_sdr(reverberant[start:], processed[start:]),
                         "cer": None, "cer_status": "deferred: no ASR backend at this stage"}
                score_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = score_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
                temporary.replace(score_path)
            all_scores.append(score)
        print(f"Evaluated {index}/{len(rows)}: {scene.scene_id}", flush=True)
    (args.output / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_scores), encoding="utf-8"
    )
    summary = {"dataset_count": len(rows), "reference": "direct-path target at reference mic",
               "crop": "exclude 0.6-second target-free calibration; retain propagation and full tail",
               "asr_status": "not run", "methods": {}}
    lines = ["# Traditional frontend baseline", "",
             "Real AISHELL-1 test subset; direct-path reference. CER not run in this stage.", "",
             "| Frontend | N | SI-SDR (dB) | PESQ-WB | STOI | CER |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for method in METHODS:
        selected = [row for row in all_scores if row["method"] == method]
        aggregate = {"n": len(selected), "metrics": {}, "valid_counts": {}}
        for metric in ("si_sdr_db", "pesq_wb", "stoi", "si_sdr_reverberant_db"):
            values = [row[metric] for row in selected if row[metric] is not None]
            aggregate["metrics"][metric] = float(np.mean(values)) if values else None
            aggregate["valid_counts"][metric] = len(values)
        summary["methods"][method] = aggregate
        values = [aggregate["metrics"][name] for name in ("si_sdr_db", "pesq_wb", "stoi")]
        cells = [f"{value:.4f}" if value is not None else "unavailable" for value in values]
        lines.append(f"| {method} | {len(selected)} | {' | '.join(cells)} | not run |")
    lines += ["", "Means are utterance averages. Check summary.json valid_counts and per-case errors.",
              "Oracle MVDR uses true DOA and a known target-free prefix; it is a privileged",
              "reference, not a guarantee of higher perceptual scores under steering mismatch.", ""]
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (args.output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
