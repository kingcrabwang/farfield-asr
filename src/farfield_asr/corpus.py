"""带文本语音 JSONL 清单检查：防止说话人泄漏、静默重采样和缺少来源记录。"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from farfield_asr.audio_io import read_audio


@dataclass(frozen=True)
class SpeechRecord:
    utterance_id: str
    audio_path: str
    transcript: str
    speaker_id: str
    split: str
    license: str
    source_url: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("all speech record fields must be nonempty strings")
        if self.split not in ("train", "dev", "test"):
            raise ValueError("split must be train, dev or test")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must identify the dataset source")


def validate_speech_manifest(path: str | Path, sample_rate: int = 16000) -> dict:
    """检查真实文件与清单，返回时长和哈希；只记录许可声明，不自动判定授权有效性。"""
    path = Path(path).resolve()
    records = []
    utterances, paths = set(), set()
    speaker_splits, hash_splits = {}, {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = SpeechRecord(**json.loads(line))
        except (ValueError, TypeError) as error:
            raise ValueError(f"manifest line {line_number}: {error}") from error
        audio_path = (path.parent / record.audio_path).resolve()
        if record.utterance_id in utterances or audio_path in paths:
            raise ValueError(f"duplicate utterance or audio path: {record.utterance_id}")
        utterances.add(record.utterance_id)
        paths.add(audio_path)
        previous_split = speaker_splits.setdefault(record.speaker_id, record.split)
        if previous_split != record.split:
            raise ValueError(f"speaker leakage across splits: {record.speaker_id}")
        audio = read_audio(audio_path, sample_rate)
        if audio.shape[1] != 1 or np.mean(audio**2) <= 0:
            raise ValueError(f"clean speech must be nonzero mono audio: {audio_path}")
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        # 同一文件即使复制改名或改说话人字段，也不能出现在不同集合。
        if hash_splits.setdefault(digest, record.split) != record.split:
            raise ValueError(f"duplicate audio content across splits: {record.utterance_id}")
        records.append({**asdict(record), "audio_path": str(audio_path),
                        "sample_rate": sample_rate, "n_samples": len(audio),
                        "duration_s": len(audio) / sample_rate, "sha256": digest})
    if not records:
        raise ValueError("speech manifest is empty")
    return {"status": "passed", "manifest": str(path), "utterances": len(records),
            "speakers": len(speaker_splits), "duration_s": sum(r["duration_s"] for r in records),
            "license_validation": "declarations recorded; permissions require source review",
            "records": records}

