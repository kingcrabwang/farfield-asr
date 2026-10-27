"""固定 SenseVoice ONNX 解码协议；逐条原子缓存，异常不计为零错误率。"""

import argparse
import hashlib
import importlib.metadata
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio
from farfield_asr.metrics import corpus_cer, normalize_text

METHODS = ("clean", "ref_mic", "das", "oracle_mvdr", "irm_mvdr",
           "neural_mvdr", "project1_singlechannel")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int, default=0, help="0 means all 200 test utterances")
    args = parser.parse_args()
    output = root / "results/generated/phase6"
    output.mkdir(parents=True, exist_ok=True)
    model_dir = root / "checkpoints/asr_sensevoice"
    assets = json.loads((model_dir / "verified_assets.json").read_text())
    for name, expected in assets["sha256"].items():
        if digest(model_dir / name) != expected:
            raise ValueError(f"ASR model asset changed: {name}")
    protocol = {"model": "iic/SenseVoiceSmall-onnx", "assets": assets,
                "language": "zh", "textnorm": "woitn", "batch_size": 1,
                "device": "CPU", "quantize": True, "threads": 2,
                "prefix": "remove exactly known 9600-sample noise-only prefix from enhanced audio",
                "normalization": "NFKC lowercase; remove whitespace and Unicode punctuation",
                "sensevoice_tags": "remove <|...|> control tokens; no emoji postprocessing",
                "versions": {name: importlib.metadata.version(name) for name in
                             ("funasr-onnx", "onnxruntime", "numpy", "torch")},
                "script_sha256": digest(Path(__file__)),
                "metrics_sha256": digest(root / "src/farfield_asr/metrics.py")}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    from funasr_onnx import SenseVoiceSmall

    model = SenseVoiceSmall(str(model_dir), batch_size=1, device_id="-1", quantize=True,
                           intra_op_num_threads=2)
    manifest = [json.loads(line) for line in
                (root / "data/simulated/test_v1/manifest.jsonl").read_text().splitlines()]
    selected = manifest[:args.limit] if args.limit else manifest
    rows = []
    for index, entry in enumerate(selected, 1):
        scene_id = entry["scene"]["scene_id"]
        for method in args.methods:
            if method == "clean":
                path = Path(entry["speech_source"]["audio_path"])
                start = 0
            else:
                phase = "phase3" if method in ("ref_mic", "das", "oracle_mvdr") else "phase5"
                path = root / f"results/generated/{phase}/audio/{method}/{scene_id}.wav"
                start = entry["noise_only_prefix_samples"]
            input_hash = digest(path)
            cache = output / "scores" / method / f"{scene_id}.json"
            if cache.exists():
                row = json.loads(cache.read_text())
                if row["input_sha256"] != input_hash or row["protocol_fingerprint"] != fingerprint:
                    raise ValueError(f"stale ASR cache; preserve it and select a new output directory: {cache}")
            else:
                samples = read_audio(path)[start:, 0].astype(np.float32)
                row = {"scene_id": scene_id, "utterance_id": entry["utterance_id"],
                       "speaker_id": entry["speaker_id"], "method": method,
                       "reference": entry["transcript"], "input_sha256": input_hash,
                       "protocol_fingerprint": fingerprint, "start_sample": start,
                       "n_samples": len(samples), "sample_rate": 16000}
                beginning = time.monotonic()
                try:
                    prediction = model(samples, language="zh", textnorm="woitn")
                    if not isinstance(prediction, list) or len(prediction) != 1:
                        raise ValueError(f"unexpected ASR output: {prediction!r}")
                    raw = prediction[0]
                    hypothesis = re.sub(r"<\|[^|]*\|>", "", raw)
                    row.update(status="ok", raw_hypothesis=raw, hypothesis=hypothesis,
                               normalized_reference=normalize_text(entry["transcript"]),
                               normalized_hypothesis=normalize_text(hypothesis),
                               **corpus_cer([(entry["transcript"], hypothesis)]))
                except Exception as error:  # noqa: BLE001 -- preserve third-party backend failure, then abort
                    row.update(status="failed", error=f"{type(error).__name__}: {error}", cer=None)
                row["decode_seconds"] = time.monotonic() - beginning
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                temporary.replace(cache)
            rows.append(row)
            print(f"ASR {index}/{len(selected)} {method} {scene_id}: {row['status']} CER={row['cer']}",
                  flush=True)
        if any(row["status"] != "ok" for row in rows[-len(args.methods):]):
            raise RuntimeError("ASR failure recorded; inspect score JSON before resuming")

    # Aggregate every compatible cache, including methods finished by earlier invocations.
    groups = defaultdict(list)
    for path in sorted((output / "scores").glob("*/*.json")):
        row = json.loads(path.read_text())
        if row["protocol_fingerprint"] != fingerprint:
            raise ValueError(f"mixed ASR protocols: {path}")
        groups[row["method"]].append(row)
    summary = {"protocol_fingerprint": fingerprint, "methods": {}}
    for method, group in groups.items():
        valid = [row for row in group if row["status"] == "ok"]
        summary["methods"][method] = {
            "n": len(group), "valid": len(valid), "failed": len(group) - len(valid),
            **corpus_cer([(row["reference"], row["hypothesis"]) for row in valid]),
            "decode_seconds": sum(row["decode_seconds"] for row in valid)}
    # Fixed speaker-balanced gate: two per speaker, plus one extra from first ten speakers.
    speakers = sorted({entry["speaker_id"] for entry in manifest})
    clean_by_speaker = defaultdict(list)
    for row in groups.get("clean", []):
        clean_by_speaker[row["speaker_id"]].append(row)
    gate = []
    for index, speaker in enumerate(speakers):
        gate.extend(sorted(clean_by_speaker[speaker], key=lambda row: row["scene_id"])
                    [:3 if index < 10 else 2])
    if len(gate) == 50 and all(row["status"] == "ok" for row in gate):
        summary["clean_50_gate"] = {"n": 50, "scene_ids": [row["scene_id"] for row in gate],
                                     **corpus_cer([(row["reference"], row["hypothesis"])
                                                   for row in gate])}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
