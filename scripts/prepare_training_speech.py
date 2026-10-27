"""按原始 AISHELL-1 train/dev 划分准备掩码训练语音，严格隔离现有测试说话人。"""

import csv
import hashlib
import io
import json
import subprocess
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from farfield_asr.corpus import validate_speech_manifest
from farfield_asr.data_access import RangeReader, public_dataset_zip

DATASET = "speech_asr_aishell1_trainsets"
BASE_URL = f"https://www.modelscope.cn/api/v1/datasets/speech_asr/{DATASET}/repo?Revision=master&FilePath="
EXPECTED = {
    "speech_asr_aishell1_trainsets.json": "72a962f5987be5ffe78331ad88042dac33b1967c2efa2da019300de088c4330a",
    "speech_asr_aishell_trainsets.csv": "acb2cde1189b87686bec471ecdac29b11f6da787f5476b4eccd8c989a7626b59",
    "speech_asr_aishell_devsets.csv": "8ffada3891f783e2228031a14aaf2c6a14c09f766bb024ec52ddcad80dcf1130",
}


def fetch_metadata(folder, name):
    destination = folder / name
    valid = destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == EXPECTED[name]
    if not valid:
        # 元数据端点不支持 Range；给完整 CSV 足够的时间，不用短超时反复丢弃已收字节。
        subprocess.run(["curl", "-4", "-fsSL", "--retry", "3", "--retry-all-errors", "--max-time", "1800",
                        BASE_URL + name, "-o", str(destination)], check=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED[name]:
        raise ValueError(f"source metadata SHA256 mismatch: {name}")
    return destination


def main():
    root = Path(__file__).resolve().parents[1]
    provenance = root / "data/provenance"
    source_config = json.loads(fetch_metadata(provenance, "speech_asr_aishell1_trainsets.json").read_text())
    source_config = source_config["default"]
    test_manifest = root / "data/manifests/speech_test.jsonl"
    test_records = [json.loads(line) for line in test_manifest.read_text(encoding="utf-8").splitlines()]
    test_speakers = {record["speaker_id"] for record in test_records}
    all_records = []
    for split, original_key, speaker_count, per_speaker in (
        ("train", "train", 8, 40), ("dev", "validation", 4, 20),
    ):
        existing = root / "data/manifests" / f"speech_{split}.jsonl"
        inventory_path = provenance / f"speech_{split}_inventory.json"
        if existing.exists() and inventory_path.exists():
            checked = validate_speech_manifest(existing)
            recorded = json.loads(inventory_path.read_text(encoding="utf-8"))
            hashes = {record["utterance_id"]: record["sha256"] for record in recorded["records"]}
            if checked["utterances"] != speaker_count * per_speaker or any(
                r["sha256"] != hashes.get(r["utterance_id"]) or r["speaker_id"] in test_speakers
                for r in checked["records"]
            ):
                raise ValueError("cached training inventory changed or overlaps test speakers")
            all_records.extend(json.loads(line) for line in existing.read_text(encoding="utf-8").splitlines())
            print(f"Verified cached {split}: {checked['utterances']}", flush=True)
            continue
        specification = source_config[original_key]
        csv_path = fetch_metadata(provenance, specification["meta"])
        groups = defaultdict(list)
        for row in csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))):
            member = Path(row["Audio:FILE"])
            if split not in member.parts:
                raise ValueError(f"unexpected original split in {member}")
            groups[member.parent.name].append(row)
        speakers = sorted(groups)[:speaker_count]
        if set(speakers) & test_speakers:
            raise ValueError("original test speakers present in training data")
        cache = public_dataset_zip("speech_asr", DATASET, specification["file"])

        def choose(speaker, groups=groups, cache=cache, split=split, per_speaker=per_speaker):
            candidates = sorted(groups[speaker], key=lambda candidate_row: hashlib.sha256(
                ("20261027-mask:" + candidate_row["Audio:FILE"]).encode()).hexdigest())
            selected = []
            with zipfile.ZipFile(RangeReader(cache)) as archive:
                for candidate in candidates:
                    member = candidate["Audio:FILE"]
                    destination = root / "data/clean/aishell1" / split / speaker / Path(member).name
                    raw = archive.read(member)
                    info = sf.info(io.BytesIO(raw))
                    if info.samplerate != 16000 or info.channels != 1 or not 2 <= info.duration <= 8:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(raw)
                    selected.append({"utterance_id": destination.stem,
                                     "audio_path": "../" + str(destination.relative_to(root / "data")),
                                     "speaker_id": speaker, "split": split,
                                     "transcript": candidate["Text:LABEL"],
                                     "license": "Apache-2.0; academic-use source wording retained",
                                     "source_url": f"https://www.modelscope.cn/datasets/speech_asr/{DATASET}"})
                    if len(selected) == per_speaker:
                        break
            if len(selected) != per_speaker:
                raise ValueError(f"insufficient eligible utterances: {speaker}")
            print(f"{split} {speaker}: {len(selected)}", flush=True)
            return selected

        with ThreadPoolExecutor(max_workers=4) as executor:
            records = [record for group in executor.map(choose, speakers) for record in group]
        existing.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                            encoding="utf-8")
        inventory = validate_speech_manifest(existing)
        inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_records.extend(records)
    all_records.extend(test_records)
    combined = root / "data/manifests/speech_all.jsonl"
    combined.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records),
                        encoding="utf-8")
    report = validate_speech_manifest(combined)
    (provenance / "speech_split_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared original train/dev subsets; all train/dev/test speakers disjoint", flush=True)


if __name__ == "__main__":
    main()
