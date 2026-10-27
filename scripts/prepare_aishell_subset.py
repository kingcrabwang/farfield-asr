"""按官方测试说话人均衡选 200 条 AISHELL-1，不把官方训练集改名为测试集。"""

import csv
import hashlib
import io
import json
import subprocess
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from farfield_asr.corpus import validate_speech_manifest
from farfield_asr.data_access import RangeReader, public_aishell_zip

CSV_SHA256 = "a028ad8b8318fbe35748c6d565893ca225bee353095e6618255d3abbbf5320c8"
SOURCE = "https://www.modelscope.cn/datasets/modelscope/speech_asr_aishell1_testsets"
CSV_URL = ("https://modelscope.cn/api/v1/datasets/modelscope/speech_asr_aishell1_testsets/"
           "repo?Revision=master&FilePath=aishell1_test.csv")


def main():
    root = Path(__file__).resolve().parents[1]
    provenance = root / "data/provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    csv_path = provenance / "aishell1_test.csv"
    if not csv_path.exists():
        response = subprocess.run(["curl", "-fsSL", "--retry", "3", "--retry-all-errors",
                                   "--max-time", "40", CSV_URL], check=True, capture_output=True)
        csv_path.write_bytes(response.stdout)
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError("AISHELL test CSV differs from inspected source revision")
    groups = defaultdict(list)
    for row in csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))):
        path = Path(row["Audio:FILE"])
        if "test" not in path.parts:
            raise ValueError("CSV unexpectedly contains non-test audio")
        groups[path.parent.name].append(row)
    if len(groups) != 20:
        raise ValueError(f"expected 20 official test speakers, got {len(groups)}")
    existing = root / "data/manifests/speech_test.jsonl"
    inventory_path = provenance / "speech_test_inventory.json"
    if existing.exists() and inventory_path.exists():
        inventory = validate_speech_manifest(existing)
        recorded = json.loads(inventory_path.read_text(encoding="utf-8"))
        expected_hashes = {r["utterance_id"]: r["sha256"] for r in recorded["records"]}
        expected_texts = {Path(r["Audio:FILE"]).stem: r["Text:LABEL"]
                          for group in groups.values() for r in group}
        if (inventory["utterances"] != 200 or inventory["speakers"] != 20
                or any(r["split"] != "test" or r["sha256"] != expected_hashes.get(r["utterance_id"])
                       or r["transcript"] != expected_texts.get(r["utterance_id"])
                       for r in inventory["records"])):
            raise ValueError("existing speech subset differs from its recorded source inventory")
        print("Verified cached 200 real test utterances; no network audio reads needed", flush=True)
        return
    cache = public_aishell_zip()
    print(f"Official test speakers: {len(groups)}; reading selected ZIP members", flush=True)

    def select_speaker(item):
        speaker, candidates = item
        candidates.sort(key=lambda row: hashlib.sha256(
            ("20261027:" + row["Audio:FILE"]).encode()).hexdigest())
        selected = []
        with zipfile.ZipFile(RangeReader(cache)) as archive:
            names = set(archive.namelist())
            for candidate in candidates:
                name = candidate["Audio:FILE"]
                if name not in names:
                    raise ValueError(f"CSV audio not present in dataset ZIP: {name}")
                destination = root / "data/clean/aishell1/test" / speaker / Path(name).name
                if destination.exists():
                    raw = destination.read_bytes()
                else:
                    for attempt in range(3):
                        try:
                            raw = archive.read(name)  # zipfile 同时验证 CRC。
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                            time.sleep(1)
                info = sf.info(io.BytesIO(raw))
                duration = info.frames / info.samplerate
                if info.samplerate != 16000 or info.channels != 1 or not 2 <= duration <= 8:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                selected.append({"utterance_id": destination.stem,
                                 "audio_path": str(destination.relative_to(root / "data")),
                                 "transcript": candidate["Text:LABEL"], "speaker_id": speaker,
                                 "split": "test", "license": "Apache-2.0; academic-use wording retained",
                                 "source_url": SOURCE})
                if len(selected) == 10:
                    break
        if len(selected) != 10:
            raise ValueError(f"insufficient valid utterances for {speaker}")
        print(f"{speaker}: selected 10", flush=True)
        return selected

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = [record for group in executor.map(select_speaker, sorted(groups.items()))
                   for record in group]
    manifests = root / "data/manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / "speech_test.jsonl"
    for record in records:
        record["audio_path"] = "../" + record["audio_path"]
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                        encoding="utf-8")
    inventory = validate_speech_manifest(manifest)
    (provenance / "speech_test_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (provenance / "aishell_subset_source.json").write_text(json.dumps({
        "dataset": "AISHELL-1 official test split subset", "distribution": SOURCE,
        "original_source": "https://www.openslr.org/33/", "test_csv_sha256": CSV_SHA256,
        "test_csv_revision": "847f9aded68606a8f7308fb3716cd0715be021ba",
        "selection": "10 utterances per each of 20 original test speakers; 2-8 seconds",
        "selection_seed": 20261027, "archive_crc_verified": True,
        "credentials": "temporary public-dataset STS used in memory only",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} real test utterances", flush=True)


if __name__ == "__main__":
    main()

