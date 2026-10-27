"""生成固定种子掩码训练样本；保存参考麦克风特征与功率 IRM，不使用测试语音。"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from farfield_asr.audio_io import read_audio
from farfield_asr.corpus import validate_speech_manifest
from farfield_asr.masks import ideal_power_mask, log_power_features
from farfield_asr.scenes import Scene, sample_scenes
from farfield_asr.simulation import render_scene, room_responses
from farfield_asr.transforms import stft


def make_example(job):
    folder, scene_dict, speech, noise, noise_path, code_hash = job
    scene = Scene(**scene_dict)
    directory = Path(folder) / speech["split"]
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{scene.scene_id}.npz"
    metadata_path = data_path.with_suffix(".json")
    fingerprint = hashlib.sha256(json.dumps(
        [scene_dict, speech["sha256"], noise["sha256"], code_hash], sort_keys=True
    ).encode()).hexdigest()
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (existing["input_fingerprint"] != fingerprint
                or hashlib.sha256(data_path.read_bytes()).hexdigest() != existing["feature_sha256"]):
            raise ValueError("cached training example changed")
        return existing
    rng = np.random.default_rng(scene.seed + 300000)
    clean = read_audio(speech["audio_path"], scene.sample_rate)[:, 0]
    count = min(len(clean), 4 * scene.sample_rate)
    crop_start = int(rng.integers(0, len(clean) - count + 1))
    target = np.pad(clean[crop_start:crop_start + count], (round(0.6 * scene.sample_rate), 0))
    source_noise = read_audio(noise_path, scene.sample_rate)[:, 0]
    for _ in range(20):
        offset = int(rng.integers(0, len(source_noise)))
        interference = source_noise[(offset + np.arange(len(target))) % len(source_noise)]
        if np.mean(interference**2) > 1e-16:
            break
    else:
        raise ValueError("unable to choose nonzero training noise")
    signals = render_scene(scene, target, interference, room_responses(scene))
    ref = scene.reference_mic
    mixture = stft(signals["mixture"][:, ref])
    speech_spectrum = stft(signals["target_reverberant"][:, ref])[:, :, 0]
    noise_spectrum = stft((signals["interference"] + signals["diffuse"])[:, ref])[:, :, 0]
    features = log_power_features(mixture)
    mask = ideal_power_mask(speech_spectrum, noise_spectrum).T.astype(np.float32)
    temporary = data_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, features=features, mask=mask)
    temporary.replace(data_path)
    metadata = {"example_id": scene.scene_id, "split": speech["split"],
                "utterance_id": speech["utterance_id"], "speaker_id": speech["speaker_id"],
                "speech_sha256": speech["sha256"], "noise_id": noise["noise_id"],
                "noise_sha256": noise["sha256"], "crop_start_sample": crop_start,
                "source_crop_samples": count, "noise_offset_samples": offset,
                "scene": scene_dict, "feature_path": str(data_path.resolve()),
                "shape": list(features.shape), "input_fingerprint": fingerprint,
                "feature_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "mask_definition": "power IRM: |S_reverberant_ref|^2/(|S|^2+|N_total_ref|^2)"}
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.views <= 3:
        raise ValueError("unsupported workers or views")
    root = Path(__file__).resolve().parents[1]
    base_config = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    combined = validate_speech_manifest(root / "data/manifests/speech_all.jsonl")
    all_speech = combined["records"]
    # validate_speech_manifest 已检验全体说话人跨集合隔离。
    noises = [json.loads(line) for line in (root / "data/manifests/noise.jsonl").read_text(
        encoding="utf-8").splitlines()]
    source_files = [Path(__file__)] + [root / "src/farfield_asr" / f"{name}.py" for name in
                                    ("simulation", "scenes", "transforms", "audio_io", "masks")]
    code_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    folder = root / "data/masks_v1"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, seed_offset in (("train", 10000), ("dev", 20000)):
        speech = [record for record in all_speech if record["split"] == split]
        split_noises = [record for record in noises if record["split"] == split]
        if not speech or not split_noises:
            raise ValueError(f"missing speech/noise for {split}")
        config = {**base_config, "seed": base_config["seed"] + seed_offset}
        scenes = sample_scenes(config, len(speech) * args.views)
        for index, scene in enumerate(scenes):
            scene = replace(scene, scene_id=f"{split}_{index:04d}")
            record = speech[index % len(speech)]
            noise = split_noises[index % len(split_noises)]
            noise_path = (root / "data/manifests" / noise["audio_path"]).resolve()
            if hashlib.sha256(noise_path.read_bytes()).hexdigest() != noise["sha256"]:
                raise ValueError("training source noise changed")
            jobs.append((str(folder), scene.to_dict(), record, noise, str(noise_path), code_hash))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = []
        for index, result in enumerate(executor.map(make_example, jobs), 1):
            records.append(result)
            print(f"Mask data {index}/{len(jobs)}: {result['example_id']}", flush=True)
    manifest = folder / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                        encoding="utf-8")
    print(f"Prepared {len(records)} mask examples without test speech/noise", flush=True)


if __name__ == "__main__":
    main()
