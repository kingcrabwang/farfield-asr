"""同一声场中的配对消融：固定孔径的阵列、RT60、导向误差。"""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from farfield_asr.arrays import circular_array, linear_array, steering_vector
from farfield_asr.audio_io import read_audio
from farfield_asr.beamforming import apply_weights, das_weights, mvdr_weights, spatial_covariance
from farfield_asr.metrics import audio_metrics
from farfield_asr.simulation import diffuse_noise, estimate_t30
from farfield_asr.transforms import istft, stft

ROOT = Path(__file__).resolve().parents[1]
RT60S = (0.2, 0.4, 0.6)
ERRORS = (-20, -10, -5, 0, 5, 10, 20)


def geometry():
    """最大两点距离均为 0.12 m；2 麦圆阵退化成线阵，因此不重复命名。"""
    arrays = {f"linear{count}": linear_array(count, 0.12 / (count - 1))
              for count in (2, 4, 6)}
    arrays.update({f"circular{count}": circular_array(count, 0.06) for count in (4, 6)})
    unique, indices = [], {}
    for name, positions in arrays.items():
        indices[name] = []
        for position in positions:
            found = next((i for i, point in enumerate(unique)
                          if np.allclose(point, position, atol=1e-12)), None)
            if found is None:
                found = len(unique)
                unique.append(position)
            indices[name].append(found)
    unique.append(np.zeros(3))  # virtual center: common calibration/metric reference
    return arrays, np.asarray(unique), indices


def trial(task):
    trial_id, entry, fingerprint = task
    output = ROOT / "results/generated/phase7/trials" / f"{trial_id:03d}.json"
    if output.exists():
        result = json.loads(output.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("ablation source/protocol changed; preserve results and use a new directory")
        return result
    fs, prefix = 16000, 9600
    arrays, relative, indices = geometry()
    scene = entry["scene"]
    center = np.asarray(scene["array_center_m"])
    positions = relative + center
    speech_path = Path(entry["speech_source"]["audio_path"])
    noise_path = (ROOT / "data/manifests" / entry["noise_source"]["audio_path"]).resolve()
    for path, expected in ((speech_path, entry["speech_source"]["sha256"]),
                           (noise_path, entry["noise_source"]["sha256"])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source hash changed: {path}")
    speech = read_audio(speech_path)[:4 * fs, 0]
    speech = np.pad(speech, (prefix, 0))
    noise = read_audio(noise_path)[:, 0]
    offset = entry["noise_offset_samples"]
    noise = noise[(np.arange(len(speech)) + offset) % len(noise)]
    responses = {}
    for rt60 in RT60S:
        absorption, order = pra.inverse_sabine(rt60, scene["room_dimensions_m"])
        room = pra.ShoeBox(scene["room_dimensions_m"], fs=fs,
                           materials=pra.Material(absorption), max_order=int(order))
        room.add_microphone_array(positions.T)
        room.add_source(scene["target_position_m"])
        room.add_source(scene["interference_position_m"])
        room.compute_rir()
        responses[rt60] = room.rir
    direct_room = pra.ShoeBox(scene["room_dimensions_m"], fs=fs, max_order=0)
    direct_room.add_microphone_array(positions.T)
    direct_room.add_source(scene["target_position_m"])
    direct_room.compute_rir()
    length = len(speech) + max(len(h) for group in responses.values()
                               for microphone in group for h in microphone) - 1

    def convolve(source, rir):
        values = [fftconvolve(source, h) for h in rir]
        return np.column_stack([np.pad(value, (0, length - len(value))) for value in values])

    direct = convolve(speech, [microphone[0] for microphone in direct_room.rir])[:, -1]
    field = diffuse_noise(length, positions, fs, 20261028 + trial_id)
    direction = np.asarray(scene["target_position_m"]) - center
    azimuth = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
    frequencies = np.fft.rfftfreq(512, 1 / fs)
    rows, hashes, calibration = [], {}, []
    for rt60, rir in responses.items():
        target = convolve(speech, [microphone[0] for microphone in rir])
        point = convolve(noise, [microphone[1] for microphone in rir])
        target_power = np.mean(target[:, -1] ** 2)
        point *= np.sqrt(target_power / np.mean(point[:, -1] ** 2))  # SIR = 0 dB
        diffuse = field * np.sqrt(target_power / (np.mean(field[:, -1] ** 2) * 10 ** 0.5))
        mixture = target + point + diffuse
        hashes[str(rt60)] = hashlib.sha256(mixture.astype("<f4").tobytes()).hexdigest()
        calibration.append({"rt60": rt60, "measured_center_t30": estimate_t30(rir[-1][0], fs),
                            "center_snr_db": float(10 * np.log10(
                                target_power / np.mean(diffuse[:, -1] ** 2))),
                            "center_sir_db": float(10 * np.log10(
                                target_power / np.mean(point[:, -1] ** 2)))})
        for name, mic_indices in indices.items():
            x = stft(mixture[:, mic_indices])
            # All selected windows lie inside target-free prefix (same as stage 3).
            frame_centers = np.arange(x.shape[1]) * 128
            selected = (frame_centers - 256 >= 1600) & (frame_centers + 256 < prefix)
            covariance = spatial_covariance(x[:, selected])
            errors = ERRORS if name == "linear4" else (0,)
            for error in errors:
                # Append virtual center to get absolute center-referenced phase, then remove it.
                a = steering_vector(np.vstack((arrays[name], np.zeros(3))), frequencies,
                                    azimuth + error, reference_mic=len(mic_indices))[:, :-1]
                for method, weights in (("das", das_weights(a)),
                                        ("mvdr", mvdr_weights(a, covariance))):
                    processed = istft(apply_weights(x, weights), length)
                    rows.append({"trial": trial_id, "utterance_id": entry["utterance_id"],
                                 "speaker_id": entry["speaker_id"], "array": name,
                                 "n_mics": len(mic_indices), "rt60_s": rt60,
                                 "doa_error_deg": error, "method": method,
                                 **audio_metrics(direct[prefix:], processed[prefix:], fs)})
    result = {"fingerprint": fingerprint, "trial": trial_id, "source_scene": scene["scene_id"],
              "speech_sha256": entry["speech_source"]["sha256"],
              "noise_sha256": entry["noise_source"]["sha256"],
              "mixture_float32_sha256_by_rt60": hashes, "calibration": calibration, "scores": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return result


def main():
    manifest = [json.loads(line) for line in
                (ROOT / "data/simulated/test_v1/manifest.jsonl").read_text().splitlines()]
    speakers = sorted({entry["speaker_id"] for entry in manifest})
    entries = [next(entry for entry in manifest if entry["speaker_id"] == speaker)
               for speaker in speakers]
    files = [Path(__file__)] + [ROOT / "src/farfield_asr" / f"{name}.py" for name in
                              ("arrays", "audio_io", "beamforming", "metrics", "simulation", "transforms")]
    fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in files)
                                 + json.dumps(entries, sort_keys=True).encode()).hexdigest()
    output = ROOT / "results/generated/phase7"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"fingerprint": fingerprint, "n_trials": len(entries), "seed": 20261028,
                "speech": "first test utterance per speaker, first <=4 seconds; no training speech",
                "arrays": "linear 2/4/6 and circular 4/6; common maximum aperture 0.12m",
                "pairing": "all microphones simulated jointly; one common diffuse field across geometries and RT60",
                "reference": "direct target at common virtual center; center-referenced steering for every array",
                "received_ratios": "SNR5/SIR0 dB at center, recalibrated per RT60; same point-source waveform",
                "rt60_s": RT60S, "doa_error_deg": ERRORS,
                "scope": "DOA errors on linear4; zero-error geometry and RT60 comparisons on all arrays",
                "limitation": "20 short excerpts; two-mic circle degenerates to a line; not a population guarantee"}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trial, [(i, entry, fingerprint) for i, entry in enumerate(entries)]):
            results.extend(result["scores"])
            print(f"Ablation trial {result['trial'] + 1}/{len(entries)} done", flush=True)
    (output / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
    table = []
    for name in geometry()[0]:
        for method in ("das", "mvdr"):
            rows = [r for r in results if r["array"] == name and r["method"] == method
                    and r["rt60_s"] == 0.4 and r["doa_error_deg"] == 0]
            table.append({"array": name, "method": method, "n": len(rows),
                          **{metric: float(np.mean([r[metric] for r in rows])) for metric in
                             ("si_sdr_db", "pesq_wb", "stoi")}})
    (output / "geometry_summary.json").write_text(json.dumps(table, indent=2) + "\n")
    for field, values, filename in (("rt60_s", RT60S, "rt60.png"),
                                    ("doa_error_deg", ERRORS, "doa_error.png")):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, metric in zip(axes, ("si_sdr_db", "pesq_wb", "stoi"), strict=True):
            for method in ("das", "mvdr"):
                means, sem = [], []
                for value in values:
                    rows = [r[metric] for r in results if r["array"] == "linear4"
                            and r["method"] == method and r[field] == value
                            and (r["doa_error_deg"] == 0 if field == "rt60_s" else r["rt60_s"] == 0.4)]
                    means.append(np.mean(rows))
                    sem.append(np.std(rows, ddof=1) / np.sqrt(len(rows)))
                axis.errorbar(values, means, yerr=sem, marker="o", label=method, capsize=3)
            axis.set(xlabel=field, ylabel=metric)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle("Paired linear4 trials; mean +/- standard error (20 speakers)")
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
