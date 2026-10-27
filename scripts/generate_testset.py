"""用带文本真实测试语音、真实点源噪声和扩散场生成 200 条可追溯多通道测试音频。"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from farfield_asr.audio_io import read_audio, write_audio
from farfield_asr.corpus import validate_speech_manifest
from farfield_asr.scenes import Scene, sample_scenes
from farfield_asr.simulation import estimate_t30, render_scene, room_responses


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def render_one(job):
    output, scene_dict, speech, noise, noise_path, prefix_samples, pipeline_hash = job
    scene = Scene(**scene_dict)
    folder = Path(output) / scene.scene_id
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(
        [scene_dict, speech["sha256"], noise["sha256"], prefix_samples, pipeline_hash], sort_keys=True
    ).encode()).hexdigest()
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["input_fingerprint"] != fingerprint:
            raise ValueError(f"refusing to replace a different scene: {folder}")
        if all(sha256(folder / name) == digest for name, digest in previous["files_sha256"].items()):
            return previous
        raise ValueError(f"existing completed scene failed integrity check: {folder}")
    target = read_audio(speech["audio_path"], scene.sample_rate)[:, 0]
    target = np.pad(target, (prefix_samples, 0))
    source_noise = read_audio(noise_path, scene.sample_rate)[:, 0]
    rng = np.random.default_rng(scene.seed + 100000)
    for _ in range(20):
        offset = int(rng.integers(0, len(source_noise)))
        indices = (np.arange(len(target)) + offset) % len(source_noise)
        interference = source_noise[indices]
        if np.mean(interference**2) > 1e-16:
            break
    else:
        raise ValueError(f"unable to select non-silent source noise: {noise_path}")
    responses = room_responses(scene)
    signals = render_scene(scene, target, interference, responses)
    for name, audio in signals.items():
        if name != "metadata":
            write_audio(folder / f"{name}.wav", audio, scene.sample_rate, overwrite=True)
    np.savez_compressed(folder / "rir.npz", target=responses.target,
                        interference=responses.interference, direct=responses.direct)
    row = {"scene": scene_dict, "utterance_id": speech["utterance_id"],
           "transcript": speech["transcript"], "speaker_id": speech["speaker_id"],
           "split": "test", "speech_source": speech, "noise_source": noise,
           "noise_offset_samples": offset,
           "noise_source_wraps": int((offset + len(target) - 1) // len(source_noise)),
           "noise_only_prefix_samples": prefix_samples, "input_fingerprint": fingerprint,
           "generation_code_sha256": pipeline_hash,
           "source_speech_n_samples": speech["n_samples"], **signals["metadata"],
           **estimate_t30(responses.target[:, scene.reference_mic], scene.sample_rate),
           "absorption": responses.absorption, "rir_max_order": responses.max_order,
           "files_sha256": {name: sha256(folder / name) for name in
                             ("mixture.wav", "target_reverberant.wav", "target_direct.wav",
                              "interference.wav", "diffuse.wav", "rir.npz")}}
    temporary = folder / "metadata.json.tmp"
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=root / "data/simulated/test_v1")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    speech_manifest = root / "data/manifests/speech_test.jsonl"
    speech = validate_speech_manifest(speech_manifest)["records"]
    if args.count > len(speech) or args.count < 1:
        raise ValueError("each scene needs a distinct real test utterance")
    noise_manifest = root / "data/manifests/noise.jsonl"
    noises = [json.loads(line) for line in noise_manifest.read_text(encoding="utf-8").splitlines()]
    noises = [record for record in noises if record["split"] == "test"]
    if not noises:
        raise ValueError("test noise manifest is empty")
    config = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    scenes = sample_scenes(config, args.count)
    rng = np.random.default_rng(config["seed"])
    code_paths = [Path(__file__), root / "src/farfield_asr/simulation.py",
                  root / "src/farfield_asr/audio_io.py", root / "src/farfield_asr/scenes.py"]
    pipeline_hash = hashlib.sha256(b"".join(path.read_bytes() for path in code_paths)).hexdigest()
    speech = [speech[i] for i in rng.permutation(len(speech))[:args.count]]
    jobs = []
    for index, (scene, utterance) in enumerate(zip(scenes, speech)):
        noise = noises[index % len(noises)]
        noise_path = (noise_manifest.parent / noise["audio_path"]).resolve()
        if sha256(noise_path) != noise["sha256"]:
            raise ValueError(f"source noise hash mismatch: {noise_path}")
        jobs.append((str(args.output.resolve()), scene.to_dict(), utterance, noise,
                     str(noise_path), round(0.6 * scene.sample_rate), pipeline_hash))
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = []
        for index, row in enumerate(executor.map(render_one, jobs), 1):
            rows.append(row)
            print(f"Rendered {index}/{len(jobs)}: {row['scene']['scene_id']}", flush=True)
    temporary = args.output / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    temporary.replace(args.output / "manifest.jsonl")
    print(f"Completed {len(rows)} real speech scenes: {args.output}", flush=True)


if __name__ == "__main__":
    main()

