"""校验真实语音清单；此命令不下载数据，也不创建或猜测文本标签。"""

import argparse
import json
from pathlib import Path

from farfield_asr.corpus import validate_speech_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = validate_speech_manifest(args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Validated {report['utterances']} utterances, {report['speakers']} speakers, "
          f"{report['duration_s']:.2f} seconds")


if __name__ == "__main__":
    main()

