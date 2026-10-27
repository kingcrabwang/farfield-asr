"""一条命令生成 100 个场景俯视图、6 组 RIR 与合成信号链路验证产物。"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import chirp

from farfield_asr.audio_io import read_audio, write_audio
from farfield_asr.scenes import sample_scenes
from farfield_asr.simulation import energy_decay, estimate_t30, render_scene, room_responses
from farfield_asr.transforms import istft, stft


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_scenes(scenes, output):
    for page in range(4):
        fig, axes = plt.subplots(5, 5, figsize=(15, 14))
        for ax, scene in zip(axes.flat, scenes[page * 25:(page + 1) * 25]):
            width, depth, _ = scene.room_dimensions_m
            ax.plot([0, width, width, 0, 0], [0, 0, depth, depth, 0], color="black")
            ax.scatter(*scene.microphones_m[:, :2].T, s=7, color="blue")
            ax.scatter(*scene.target_position_m[:2], marker="*", color="green", s=60)
            ax.scatter(*scene.interference_position_m[:2], marker="x", color="red", s=30)
            ax.set(title=f"{scene.scene_id}: {scene.array_geometry}{scene.n_mics}\n"
                         f"RT60={scene.rt60_s:g}s, SNR={scene.snr_db:g}dB",
                   xlim=(-0.2, width + 0.2), ylim=(-0.2, depth + 0.2))
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
        fig.suptitle("Blue: microphones | Green: target | Red: point interference (metres)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output / f"scenes_{page + 1:02d}.png", dpi=120)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/base.yaml")
    parser.add_argument("--output", type=Path, default=root / "results/generated/phase2")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空目录，请指定新的 --output：{output}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    write_json(output / "config_snapshot.json", config)
    scenes = sample_scenes(config, 100)
    with (output / "scenes.jsonl").open("w", encoding="utf-8") as file:
        for scene in scenes:
            file.write(json.dumps(scene.to_dict(), ensure_ascii=False) + "\n")
    plot_scenes(scenes, output)
    summary = {"signal_kind": "synthetic non-speech diagnostic; no ASR evaluation",
               "scene_count": 100, "experiments": []}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, geometry in zip(axes, ("linear", "circular")):
        base = next(s for s in scenes if s.array_geometry == geometry)
        for rt60 in config["scenes"]["rt60_s"]:
            name = f"{geometry}{base.n_mics}_rt{round(rt60 * 1000):04d}"
            scene = replace(base, scene_id=name, rt60_s=rt60, snr_db=5.0, sir_db=0.0)
            directory = output / name
            directory.mkdir()
            print(f"Simulating {name} ...", flush=True)
            responses = room_responses(scene)
            np.savez_compressed(directory / "rir.npz", target=responses.target,
                                interference=responses.interference, direct=responses.direct)
            fs = scene.sample_rate
            time = np.arange(fs) / fs
            target = 0.2 * chirp(time, f0=180, f1=3200, t1=1.0) * np.sin(np.pi * time)**2
            interference = np.random.default_rng(scene.seed + 10000).normal(0, 0.1, len(time))
            result = render_scene(scene, target, interference, responses)
            for component, signal in result.items():
                if component != "metadata":
                    write_audio(directory / f"{component}.wav", signal, fs)
            # 在实际写盘后的多通道 WAV 上验证 STFT，不只验证内存中的信号。
            disk_audio = read_audio(directory / "mixture.wav", fs)
            reconstructed = istft(stft(disk_audio), len(disk_audio))
            error = float(np.max(np.abs(reconstructed - disk_audio)))
            if error > 1e-10:
                raise RuntimeError(f"STFT reconstruction failed: {error}")
            fit = estimate_t30(responses.target[:, scene.reference_mic], fs)
            entry = {"scene": scene.to_dict(), **result["metadata"], **fit,
                     "absorption": responses.absorption, "max_order": responses.max_order,
                     "stft_max_abs_error": error, "files_sha256": {
                         path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(directory.iterdir())
                     }}
            write_json(directory / "metadata.json", entry)
            summary["experiments"].append(entry)
            decay = energy_decay(responses.target[:, scene.reference_mic])
            ax.plot(np.arange(len(decay)) / fs, decay,
                    label=f"target {rt60:.1f}s / T30 extrap. {fit['rt60_t30_s']:.3f}s")
        ax.set(title=f"{geometry} array", xlabel="Time after emission (s)",
               ylim=(-65, 1), xlim=(0, 0.9))
        ax.axhline(-5, color="gray", linestyle=":")
        ax.axhline(-35, color="gray", linestyle=":")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Schroeder energy decay (dB)")
    fig.tight_layout()
    fig.savefig(output / "rir_decay.png", dpi=160)
    plt.close(fig)
    write_json(output / "summary.json", summary)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()

