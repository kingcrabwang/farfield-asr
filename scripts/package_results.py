"""仅打包公开图表、汇总与自制诊断音频；原始录音/模型保持本地。"""

import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "docs/assets"
    target.mkdir(parents=True, exist_ok=True)
    for phase, names in {
        "phase3": ("beampatterns.png", "delay_check.png"),
        "phase4": ("mae_by_rt60.png", "mae_by_snr.png", "failure_scans.png"),
        "phase5": ("training.png", "test_masks.png"),
        "phase7": ("rt60.png", "doa_error.png"),
    }.items():
        for name in names:
            shutil.copyfile(root / f"results/generated/{phase}/{name}", target / f"{phase}_{name}")
    for name in ("technical_summary.pdf", "underwater_to_microphone.pdf"):
        source = root / "output/pdf" / name
        if source.exists():
            shutil.copyfile(source, target / name)
        elif not (target / name).exists():
            raise FileNotFoundError(f"render final handoff PDF first: {name}")
    summaries = {phase: json.loads((root / f"results/generated/{phase}/summary.json").read_text())
                 for phase in ("phase3", "phase4", "phase5", "phase6")}
    (target / "benchmark_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for relative in ("results/masks.md", "results/ablations.md"):
        path = root / relative
        content = path.read_text()
        for phase in ("phase5", "phase7"):
            content = content.replace(f"(generated/{phase}/", f"(../docs/assets/{phase}_")
        path.write_text(content)
    print("Packaged public figures, summaries and PDFs")


if __name__ == "__main__":
    main()
