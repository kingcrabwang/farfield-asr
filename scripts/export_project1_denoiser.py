"""将本轮补建的单通道模块导出到现有项目一的独立子目录，不覆盖用户文件。"""

import hashlib
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "checkpoints/mask_gru/best.pt"
    if not (root / "checkpoints/mask_gru/training_summary.json").exists() or not checkpoint.exists():
        raise ValueError("export requires completed, verified training")
    source = root / "integrations/project1"
    destination = Path.home() / "speech-front-end/denoising"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"origin": "project-one supplementary module built during the farfield project",
              "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "files": {}}
    for name in ("denoiser.py", "demo_denoise.py", "README.md"):
        original, target = source / name, destination / name
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"refusing to replace user-edited project-one file: {target}")
        shutil.copyfile(original, target)
        target.chmod(0o644)
        report["files"][name] = {"path": str(target), "sha256": digest}
    output = root / "results/generated/phase5"
    output.mkdir(parents=True, exist_ok=True)
    (output / "project1_integration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
