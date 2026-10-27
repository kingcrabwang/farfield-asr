"""临时 WAV 仅用于数据接口单测，文字不是这些合成信号的真实语音标签。"""

import json

import numpy as np
import pytest

from farfield_asr.audio_io import write_audio
from farfield_asr.corpus import validate_speech_manifest


def record(utterance="a", speaker="s1", split="train"):
    return {"utterance_id": utterance, "audio_path": f"{utterance}.wav",
            "transcript": "接口测试文本", "speaker_id": speaker, "split": split,
            "license": "synthetic unit-test fixture only",
            "source_url": "https://example.invalid/unit-test"}


def manifest(tmp_path, rows):
    path = tmp_path / "speech.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_valid_corpus_inventory(tmp_path):
    write_audio(tmp_path / "a.wav", np.full(1600, 0.1))
    report = validate_speech_manifest(manifest(tmp_path, [record()]))
    assert report["utterances"] == 1
    assert report["duration_s"] == 0.1
    assert len(report["records"][0]["sha256"]) == 64


def test_speaker_leakage_rejected(tmp_path):
    write_audio(tmp_path / "a.wav", np.full(1600, 0.1))
    with pytest.raises(ValueError, match="speaker leakage"):
        validate_speech_manifest(manifest(tmp_path, [record(), record("b", split="test")]))


def test_renamed_duplicate_file_across_splits_rejected(tmp_path):
    for name in ("a", "b"):
        write_audio(tmp_path / f"{name}.wav", np.full(1600, 0.1))
    with pytest.raises(ValueError, match="duplicate audio content"):
        validate_speech_manifest(manifest(tmp_path, [record(), record("b", "s2", "test")]))


@pytest.mark.parametrize("field,value", [("transcript", ""), ("license", ""), ("split", "val")])
def test_invalid_metadata_rejected(tmp_path, field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        validate_speech_manifest(manifest(tmp_path, [row]))


def test_non_mono_clean_speech_rejected(tmp_path):
    write_audio(tmp_path / "a.wav", np.full((1600, 2), 0.1))
    with pytest.raises(ValueError, match="mono"):
        validate_speech_manifest(manifest(tmp_path, [record()]))

