"""评测理想功率掩码 MVDR、神经掩码 MVDR 与项目一补充的单通道模型。"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio, write_audio
from farfield_asr.beamforming import apply_weights
from farfield_asr.masks import ideal_power_mask, mask_mvdr_weights
from farfield_asr.metrics import audio_metrics, si_sdr
from farfield_asr.scenes import Scene
from farfield_asr.transforms import istft, stft

METHODS = ("irm_mvdr", "neural_mvdr", "project1_singlechannel")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints/mask_gru/best.pt")
    args = parser.parse_args()
    dataset, output = root / "data/simulated/test_v1", root / "results/generated/phase5"
    output.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    estimator, single_channel, checkpoint_hash = None, None, None
    if any(method != "irm_mvdr" for method in args.methods):
        from farfield_asr.mask_network import NeuralMaskEstimator
        training = json.loads((args.checkpoint.parent / "training_summary.json").read_text())
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if training["status"] != "completed" or checkpoint_hash != training["best_checkpoint_sha256"]:
            raise ValueError("learned evaluation requires completed training and verified weights")
        test_speakers = {entry["speaker_id"] for entry in manifest}
        if test_speakers & (set(training["train_speaker_ids"]) | set(training["dev_speaker_ids"])):
            raise ValueError("checkpoint training/development speakers overlap test speakers")
        test_noises = {entry["noise_source"]["noise_id"] for entry in manifest}
        if test_noises & (set(training["train_noise_ids"]) | set(training["dev_noise_ids"])):
            raise ValueError("checkpoint training/development noise overlaps test noise")
        estimator = NeuralMaskEstimator(args.checkpoint)
        if "project1_singlechannel" in args.methods:
            path = Path.home() / "speech-front-end/denoising/denoiser.py"
            integration = json.loads((output / "project1_integration.json").read_text(encoding="utf-8"))
            if hashlib.sha256(path.read_bytes()).hexdigest() != integration["files"]["denoiser.py"]["sha256"]:
                raise ValueError("project-one module differs from exported source")
            specification = importlib.util.spec_from_file_location("project1_supplementary_denoiser", path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            single_channel = module.SingleChannelDenoiser(args.checkpoint)
    common_files = [Path(__file__)] + [root / "src/farfield_asr" / f"{name}.py" for name in
                                    ("masks", "beamforming", "transforms", "audio_io", "metrics")]
    common_hash = hashlib.sha256(b"".join(path.read_bytes() for path in common_files)).hexdigest()
    fingerprints = {}
    for method in args.methods:
        extra = "ideal-mask-oracle"
        if method != "irm_mvdr":
            extra = checkpoint_hash + hashlib.sha256((root / "src/farfield_asr/mask_network.py").read_bytes()).hexdigest()
        if method == "project1_singlechannel":
            extra += hashlib.sha256((Path.home() / "speech-front-end/denoising/denoiser.py").read_bytes()).hexdigest()
        fingerprints[method] = hashlib.sha256((common_hash + extra).encode()).hexdigest()
    scores = []
    for index, entry in enumerate(manifest, 1):
        scene = Scene(**entry["scene"])
        folder = dataset / scene.scene_id
        mixture = read_audio(folder / "mixture.wav")
        direct = read_audio(folder / "target_direct.wav")[:, scene.reference_mic]
        reverberant = read_audio(folder / "target_reverberant.wav")[:, scene.reference_mic]
        spectrum = stft(mixture)
        estimated_mask = None
        for method in args.methods:
            audio_path = output / "audio" / method / f"{scene.scene_id}.wav"
            score_path = output / "scores" / method / f"{scene.scene_id}.json"
            if score_path.exists():
                row = json.loads(score_path.read_text(encoding="utf-8"))
                if (row["input_fingerprint"] != entry["input_fingerprint"]
                        or row["evaluation_fingerprint"] != fingerprints[method]
                        or hashlib.sha256(audio_path.read_bytes()).hexdigest() != row["output_sha256"]):
                    raise ValueError("cached mask-frontend result changed; use a distinct evaluation directory")
            else:
                if method == "irm_mvdr":
                    noise = read_audio(folder / "interference.wav") + read_audio(folder / "diffuse.wav")
                    mask = ideal_power_mask(stft(reverberant)[:, :, 0],
                                             stft(noise[:, scene.reference_mic])[:, :, 0])
                elif method == "neural_mvdr":
                    if estimated_mask is None:
                        estimated_mask = estimator.estimate(spectrum, scene.reference_mic)
                    mask = estimated_mask
                if method == "project1_singlechannel":
                    processed = single_channel.process(mixture[:, scene.reference_mic])
                else:
                    weights = mask_mvdr_weights(spectrum, mask, scene.reference_mic)
                    processed = istft(apply_weights(spectrum, weights), len(mixture))
                write_audio(audio_path, processed, scene.sample_rate, overwrite=True)
                processed = read_audio(audio_path)[:, 0]
                start = entry["noise_only_prefix_samples"]
                row = {"scene_id": scene.scene_id, "method": method,
                       "input_fingerprint": entry["input_fingerprint"],
                       "evaluation_fingerprint": fingerprints[method], "checkpoint_sha256": checkpoint_hash,
                       "output_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                       "audio_path": str(audio_path), "utterance_id": entry["utterance_id"],
                       "transcript": entry["transcript"], "array_geometry": scene.array_geometry,
                       "n_mics": scene.n_mics, "requested_rt60_s": scene.rt60_s,
                       "measured_rt60_s": entry["rt60_t30_s"], "metric_start_sample": start,
                       "metric_n_samples": len(processed) - start,
                       "output_peak": float(np.max(np.abs(processed))),
                       "samples_abs_gt1": int(np.count_nonzero(np.abs(processed) > 1)),
                       **audio_metrics(direct[start:], processed[start:], scene.sample_rate),
                       "si_sdr_reverberant_db": si_sdr(reverberant[start:], processed[start:]),
                       "cer": None, "cer_status": "not run yet"}
                score_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = score_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary.replace(score_path)
            scores.append(row)
        print(f"Mask frontends {index}/{len(manifest)}: {scene.scene_id}", flush=True)
    (output / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scores), encoding="utf-8"
    )
    summary = {"utterances": len(manifest), "methods": {}, "cer_status": "not run yet"}
    for method in args.methods:
        selected = [row for row in scores if row["method"] == method]
        result = {"n": len(selected), "metrics": {}, "valid_counts": {}}
        for metric in ("si_sdr_db", "pesq_wb", "stoi", "si_sdr_reverberant_db"):
            values = [row[metric] for row in selected if row[metric] is not None]
            result["metrics"][metric] = float(np.mean(values)) if values else None
            result["valid_counts"][metric] = len(values)
        summary["methods"][method] = result
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
