"""生成完全由公式合成的公开诊断音频；不使用第三方录音或 ASR 训练素材。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from farfield_asr.frontends import Frontend
from farfield_asr.scenes import sample_scenes
from farfield_asr.simulation import render_scene


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs/assets/demo"
    destination.mkdir(parents=True, exist_ok=True)
    fs, prefix = 16000, 9600
    time = np.arange(4 * fs) / fs
    # 谐波组随时间改变，模拟有声信号的频谱活动；这不是可懂语音。
    phase = 2 * np.pi * np.cumsum(150 + 25 * np.sin(2 * np.pi * 0.7 * time)) / fs
    voiced = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 24))
    envelope = (0.5 - 0.5 * np.cos(2 * np.pi * np.minimum(time % 0.8 / 0.5, 1)))
    voiced *= envelope * 0.1
    target = np.pad(voiced, (prefix, 0))
    rng = np.random.default_rng(20261028)
    point = rng.standard_normal(len(target)) * 0.1
    config = yaml.safe_load((root / "configs/base.yaml").read_text())
    scene = sample_scenes(config, 1)[0]
    signals = render_scene(scene, target, point)
    outputs = {method: Frontend(method, scene, prefix).process(signals["mixture"])[prefix:]
               for method in ("ref_mic", "das", "oracle_mvdr")}
    # 一个共同增益避免播放削波；不能分别归一化输出掩盖幅度差异。
    gain = min(1.0, 0.95 / max(float(np.max(np.abs(value))) for value in outputs.values()))
    files = {}
    for method, values in outputs.items():
        path = destination / f"{method}.wav"
        wavfile.write(path, fs, np.round(values * gain * 32767).astype(np.int16))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {"type": "procedural harmonic diagnostic, not intelligible speech",
                  "third_party_recordings": False, "not_part_of_benchmark": True,
                  "seed": 20261028, "common_playback_gain": gain, "scene": scene.to_dict(),
                  "sha256": files, "license": "CC0-1.0", "creator": "kingcrabwang project code"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
