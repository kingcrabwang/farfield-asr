"""从落盘产物独立检查通道、长度、分量可加性、SNR/SIR 和文件哈希。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio
from farfield_asr.scenes import Scene


def verify(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (directory / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]
    scenes = [Scene(**row) for row in rows]
    if len(scenes) != 100 or len({s.scene_id for s in scenes}) != 100:
        raise ValueError("expected 100 distinct scenes")
    if len({(s.array_geometry, s.rt60_s) for s in scenes}) != 6:
        raise ValueError("missing array/RT60 combinations")
    if len(summary["experiments"]) != 6:
        raise ValueError("expected six rendered diagnostic scenes")
    checks = []
    for entry in summary["experiments"]:
        scene = Scene(**entry["scene"])
        folder = directory / scene.scene_id
        for name, expected in entry["files_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"hash mismatch: {folder / name}")
        audio = {name: read_audio(folder / f"{name}.wav", scene.sample_rate)
                 for name in ("mixture", "target_reverberant", "target_direct",
                              "interference", "diffuse")}
        if any(x.shape != (entry["n_samples"], scene.n_mics) for x in audio.values()):
            raise ValueError("saved audio shape mismatch")
        residual = audio["mixture"] - (audio["target_reverberant"]
                                       + audio["interference"] + audio["diffuse"])
        max_error = float(np.max(np.abs(residual)))
        if max_error > 2e-7:
            raise ValueError(f"components do not add up: {max_error}")
        ref = scene.reference_mic
        target_power = np.mean(audio["target_reverberant"][:, ref]**2)
        snr = float(10 * np.log10(target_power / np.mean(audio["diffuse"][:, ref]**2)))
        sir = float(10 * np.log10(target_power / np.mean(audio["interference"][:, ref]**2)))
        if abs(snr - scene.snr_db) > 1e-5 or abs(sir - scene.sir_db) > 1e-5:
            raise ValueError("saved audio SNR/SIR mismatch")
        if np.max(np.abs(audio["mixture"])) > 0.950001:
            raise ValueError("unexpected mixture peak")
        checks.append({"scene_id": scene.scene_id, "snr_db_from_wav": snr,
                       "sir_db_from_wav": sir, "component_sum_max_error": max_error,
                       "rt60_t30_s": entry["rt60_t30_s"]})
    return {"status": "passed", "scene_count": len(scenes), "rendered_checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeat", type=Path, help="比较另一独立运行目录的全部生成文件")
    args = parser.parse_args()
    report = verify(args.directory)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (args.directory / "verification.json").write_text(text, encoding="utf-8")
    print(text)
    if args.repeat is not None:
        # 验收报告在实验之后生成，不计入实验产物本身。
        excluded = {"verification.json", "reproducibility.json"}
        first = {p.relative_to(args.directory): p for p in args.directory.rglob("*")
                 if p.is_file() and p.name not in excluded}
        second = {p.relative_to(args.repeat): p for p in args.repeat.rglob("*")
                  if p.is_file() and p.name not in excluded}
        if first.keys() != second.keys():
            raise ValueError("two runs produced different file sets")
        mismatches = [str(name) for name in first
                      if hashlib.sha256(first[name].read_bytes()).digest()
                      != hashlib.sha256(second[name].read_bytes()).digest()]
        if mismatches:
            raise ValueError(f"reproducibility mismatch: {mismatches}")
        result = {"status": "passed", "byte_identical_files": len(first),
                  "scope": "two independent runs in the same installed environment"}
        (args.directory / "reproducibility.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()

