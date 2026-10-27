"""下载公开官方 SenseVoice CPU ONNX 权重；每个文件均校验固定 SHA256。"""

import hashlib
import json
import subprocess
from pathlib import Path

ASSETS = {
    "am.mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
    "config.yaml": "f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19",
    "configuration.json": "c57f6a580d63f7465c6a22ba95847aee05a1ae1181f5abddffb943d9febda061",
    "model_quant.onnx": "21dc965f689a78d1604717bf561e40d5a236087c85a95584567835750549e822",
    "tokens.json": "a2594fc1474e78973149cba8cd1f603ebed8c39c7decb470631f66e70ce58e97",
    "chn_jpn_yue_eng_ko_spectok.bpe.model":
        "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
}


def main():
    folder = Path(__file__).resolve().parents[1] / "checkpoints/asr_sensevoice"
    folder.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, expected in ASSETS.items():
        repo = "iic/SenseVoiceSmall-onnx"
        revision = "master"
        if name.endswith("bpe.model"):
            repo = "iic/SenseVoiceSmall"
            revision = "70514a3da51f1160f51d18449dab6128bbd4928b"
        url = f"https://www.modelscope.cn/models/{repo}/resolve/{revision}/{name}"
        sources[name] = url
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            subprocess.run(["curl", "-4", "-fL", "--retry", "5", "--retry-all-errors",
                            "--connect-timeout", "30", "-o", str(temporary), url], check=True)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"download hash mismatch: {name}; retained temporary for inspection")
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"local model asset hash mismatch: {name}")
        print(f"Verified {name}", flush=True)
    provenance = {"model": "iic/SenseVoiceSmall-onnx", "sha256": ASSETS, "sources": sources,
                  "tokenizer": "official SenseVoiceSmall SentencePiece; fixed revision",
                  "license": "see official SenseVoice model repository; raw weights excluded from Git"}
    (folder / "verified_assets.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
