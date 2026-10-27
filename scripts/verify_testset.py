"""完整核验正式测试集：文件哈希、真实语音唯一性、纯噪声前缀、比例与分量可加性。"""

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio
from farfield_asr.scenes import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce-one", action="store_true", help="独立重生成第一条并比较六个文件哈希")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data/simulated/test_v1"
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(rows) != 200 or len({r["utterance_id"] for r in rows}) != 200:
        raise ValueError("expected 200 distinct real speech utterances")
    if len({r["speaker_id"] for r in rows}) != 20 or any(r["split"] != "test" for r in rows):
        raise ValueError("expected original test subset from 20 speakers")
    largest_sum_error, largest_ratio_error, largest_prefix_leak = 0.0, 0.0, 0.0
    for row in rows:
        scene = Scene(**row["scene"])
        folder = dataset / scene.scene_id
        for name, digest in row["files_sha256"].items():
            with (folder / name).open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                    raise ValueError(f"hash mismatch: {folder / name}")
        components = {name: read_audio(folder / f"{name}.wav", scene.sample_rate) for name in
                      ("mixture", "target_reverberant", "target_direct", "interference", "diffuse")}
        if any(audio.shape != (row["n_samples"], scene.n_mics) for audio in components.values()):
            raise ValueError("inconsistent component shape")
        residual = components["mixture"] - (components["target_reverberant"]
                                            + components["interference"] + components["diffuse"])
        largest_sum_error = max(largest_sum_error, float(np.max(np.abs(residual))))
        prefix = row["noise_only_prefix_samples"]
        leak = np.max(np.abs(components["target_reverberant"][:prefix]))
        largest_prefix_leak = max(largest_prefix_leak, float(leak))
        ref = scene.reference_mic
        power = np.mean(components["target_reverberant"][:, ref]**2)
        for name, expected in (("diffuse", scene.snr_db), ("interference", scene.sir_db)):
            actual = 10 * np.log10(power / np.mean(components[name][:, ref]**2))
            largest_ratio_error = max(largest_ratio_error, abs(float(actual) - expected))
        if np.max(np.abs(components["mixture"])) > 0.950001:
            raise ValueError("mixture peak exceeded common-gain headroom")
    if largest_sum_error > 2e-7 or largest_ratio_error > 1e-5 or largest_prefix_leak > 1e-8:
        raise ValueError("component, ratio or calibration-prefix verification failed")
    report = {"status": "passed", "utterances": len(rows), "speakers": 20,
              "array_counts": dict(Counter(r["scene"]["array_geometry"] for r in rows)),
              "rt60_requested_counts": dict(Counter(str(r["scene"]["rt60_s"]) for r in rows)),
              "largest_component_sum_error": largest_sum_error,
              "largest_snr_sir_error_db": largest_ratio_error,
              "largest_target_leak_in_noise_prefix": largest_prefix_leak,
              "reference": "target_direct.wav, reference microphone; no delay realignment"}
    output = root / "results/generated/phase3"
    output.mkdir(parents=True, exist_ok=True)
    if args.reproduce_one:
        from generate_testset import render_one
        code_paths = [root / "scripts/generate_testset.py", root / "src/farfield_asr/simulation.py",
                      root / "src/farfield_asr/audio_io.py", root / "src/farfield_asr/scenes.py"]
        code_hash = hashlib.sha256(b"".join(path.read_bytes() for path in code_paths)).hexdigest()
        first = rows[0]
        if code_hash != first["generation_code_sha256"]:
            raise ValueError("generation code changed since this dataset was created")
        with tempfile.TemporaryDirectory(prefix="reproduce-", dir=output) as directory:
            reproduced = render_one((directory, first["scene"], first["speech_source"],
                                     first["noise_source"], str((root / "data/manifests"
                                     / first["noise_source"]["audio_path"]).resolve()),
                                     first["noise_only_prefix_samples"], code_hash))
        if reproduced["files_sha256"] != first["files_sha256"]:
            raise ValueError("independently reproduced scene differs from verified dataset")
        report["independent_reproduction"] = {"scene_id": first["scene"]["scene_id"],
                                               "byte_identical_files": 6}
    (output / "dataset_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
