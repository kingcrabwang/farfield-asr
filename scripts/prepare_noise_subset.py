"""从 OpenSLR28 官方 ZIP 按需读取真实 MUSAN 点源噪声，保存来源和 CRC/哈希。"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from farfield_asr.data_access import HTTPRangeBucket, RangeReader, SharedRangeCache

URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/manifests/noise.jsonl"
    if manifest.exists():
        cached = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        if len(cached) != 30 or len({r["noise_id"] for r in cached}) != 30:
            raise ValueError("unexpected cached noise inventory")
        for record in cached:
            path = (manifest.parent / record["audio_path"]).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"cached noise hash mismatch: {path}")
        print("Verified cached 30 real noises; no network audio reads needed", flush=True)
        return
    cache = SharedRangeCache(HTTPRangeBucket(URL), "rirs_noises.zip")
    records = []
    with zipfile.ZipFile(RangeReader(cache)) as archive:
        names = [name for name in archive.namelist()
                 if "/pointsource_noises/" in name and Path(name).name.startswith("noise-")
                 and name.endswith(".wav")]
        names.sort(key=lambda name: hashlib.sha256(("20261027:" + name).encode()).hexdigest())
        print(f"Candidate real point-source noises: {len(names)}", flush=True)
        if len(names) < 30:
            raise ValueError("insufficient matching original noise recordings")
        for name in names:
            destination = root / "data/noise/openslr28" / Path(name).name
            raw = destination.read_bytes() if destination.exists() else archive.read(name)
            audio, fs = sf.read(io.BytesIO(raw), always_2d=True)
            if fs != 16000 or audio.shape[1] != 1 or len(audio) < 2 * fs:
                continue
            if not np.all(np.isfinite(audio)) or np.mean(audio**2) <= 0:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            index = len(records)
            records.append({"noise_id": destination.stem,
                            "audio_path": "../" + str(destination.relative_to(root / "data")),
                            "split": "test" if index < 15 else "train" if index < 25 else "dev",
                            "sample_rate": fs, "n_samples": len(audio),
                            "license": "OpenSLR28 Apache-2.0 distribution; MUSAN CC-BY-4.0 origin",
                            "source_url": "https://www.openslr.org/28/",
                            "original_source_url": "https://www.openslr.org/17/",
                            "archive_member": name, "zip_crc32": archive.getinfo(name).CRC,
                            "sha256": hashlib.sha256(raw).hexdigest()})
            print(f"Prepared noise {len(records)}/30: {destination.name}", flush=True)
            if len(records) == 30:
                break
    if len(records) != 30:
        raise ValueError("not enough usable noise recordings")
    manifests = root / "data/manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "noise.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    print("Prepared 30 real noise files with disjoint source-file splits", flush=True)


if __name__ == "__main__":
    main()
