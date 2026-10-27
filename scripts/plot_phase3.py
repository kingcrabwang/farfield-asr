"""绘制实际测试集结果、噪声校准 MVDR 方向图和传播时延检查。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from farfield_asr.arrays import steering_vector
from farfield_asr.audio_io import read_audio
from farfield_asr.beamforming import das_weights, mvdr_weights
from farfield_asr.frontends import calibration_covariance
from farfield_asr.scenes import Scene
from farfield_asr.transforms import DEFAULT_STFT, stft


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/generated/phase3"
    dataset = root / "data/simulated/test_v1"
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    manifest = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    methods = ("ref_mic", "das", "oracle_mvdr")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ("si_sdr_db", "pesq_wb", "stoi"),
                               ("SI-SDR (dB)", "PESQ-WB", "STOI")):
        for method in methods:
            means, errors = [], []
            for rt60 in (0.2, 0.4, 0.6):
                values = [r[metric] for r in scores if r["method"] == method
                          and r["requested_rt60_s"] == rt60 and r[metric] is not None]
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1) / np.sqrt(len(values)))
            ax.errorbar([0.2, 0.4, 0.6], means, yerr=errors, marker="o", capsize=3, label=method)
        ax.set(xlabel="Requested RT60 (s)", ylabel=title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Actual test-subset scores; bars: standard error across utterances")
    fig.tight_layout()
    fig.savefig(output / "metrics_by_rt60.png", dpi=160)
    plt.close(fig)

    config = DEFAULT_STFT
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / 16000)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    delay_fig, delay_axes = plt.subplots(1, 2, figsize=(12, 4))
    for row_index, geometry in enumerate(("linear", "circular")):
        entry = next(row for row in manifest if row["scene"]["array_geometry"] == geometry)
        scene = Scene(**entry["scene"])
        x = read_audio(dataset / scene.scene_id / "mixture.wav")
        spectrum = stft(x)
        covariance, count = calibration_covariance(spectrum, scene.sample_rate,
                                                   entry["noise_only_prefix_samples"], scene.n_mics)
        a = steering_vector(scene.microphones_m, frequencies, scene.target_azimuth_deg,
                            scene.sound_speed_m_s, scene.reference_mic)
        weights = {"DAS": das_weights(a), "oracle MVDR": mvdr_weights(a, covariance)}
        angles = np.linspace(0, 180 if geometry == "linear" else 360, 721)
        for column, frequency in enumerate((1000, 4000)):
            ax = axes[row_index, column]
            bin_index = int(np.argmin(np.abs(frequencies - frequency)))
            vectors = np.stack([steering_vector(scene.microphones_m, [frequency], angle,
                                               scene.sound_speed_m_s, scene.reference_mic)[0]
                                for angle in angles])
            for name, weight in weights.items():
                response = np.abs(vectors @ weight[bin_index].conj())
                ax.plot(angles, 20 * np.log10(np.maximum(response, 1e-4)), label=name)
            ax.axvline(scene.target_azimuth_deg, color="black", linestyle="--", label="true target")
            ax.set(title=f"{geometry}{scene.n_mics}, {frequency} Hz, calibration frames={count}",
                   xlabel="Azimuth (deg)", ylabel="Response (dB)", ylim=(-50, 10))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        with np.load(dataset / scene.scene_id / "rir.npz") as rir:
            peak = np.argmax(np.abs(rir["direct"]), axis=0)
        ref = scene.reference_mic
        measured = (peak - peak[ref]) / scene.sample_rate * 1000
        direction = np.array([np.cos(np.deg2rad(scene.target_azimuth_deg)),
                              np.sin(np.deg2rad(scene.target_azimuth_deg)), 0])
        farfield = (-(scene.microphones_m - scene.microphones_m[ref]) @ direction
                    / scene.sound_speed_m_s * 1000)
        distances = np.linalg.norm(scene.microphones_m - scene.target_position_m, axis=1)
        spherical = (distances - distances[ref]) / scene.sound_speed_m_s * 1000
        ax = delay_axes[row_index]
        for values, name in ((measured, "RIR peak (sample quantized)"),
                             (spherical, "spherical propagation"), (farfield, "far-field steering")):
            ax.plot(range(scene.n_mics), values, marker="o", label=name)
        ax.set(title=f"{geometry}{scene.n_mics}: reference-mic relative delays",
               xlabel="Microphone index", ylabel="Delay (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "beampatterns.png", dpi=160)
    plt.close(fig)
    delay_fig.tight_layout()
    delay_fig.savefig(output / "delay_check.png", dpi=160)
    plt.close(delay_fig)
    print(output)


if __name__ == "__main__":
    main()
