"""在 Ubuntu 验证环境、可编辑安装和 pyroomacoustics 原生仿真模块。"""

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

import farfield_asr
from farfield_asr.arrays import linear_array


def main():
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Linux":
        raise RuntimeError("此验收要求在 Ubuntu/Linux 的 Conda 环境中执行")
    if Path(sys.prefix).name != "farfield-asr":
        raise RuntimeError(f"请使用 farfield-asr 环境，当前 prefix={sys.prefix}")
    package_path = Path(farfield_asr.__file__).resolve()
    if not package_path.is_relative_to(root / "src"):
        raise RuntimeError(f"导入的包不是当前项目源码：{package_path}")

    # 小型确定性仿真只检查原生模块和阵列接口，不作为正式 RT60/语音评测。
    positions = linear_array().T + np.array([[2.0], [1.5], [1.2]])
    room = pra.ShoeBox(
        [4.0, 3.0, 2.5], fs=16000, materials=pra.Material(0.4), max_order=2
    )
    room.add_microphone_array(positions)
    room.add_source([3.0, 2.0, 1.2])
    room.compute_rir()
    if len(room.rir) != 4:
        raise RuntimeError("RIR 麦克风数量错误")
    lengths = []
    for mic_rir in room.rir:
        rir = np.asarray(mic_rir[0])
        if rir.size == 0 or not np.all(np.isfinite(rir)) or np.sum(rir**2) <= 0:
            raise RuntimeError("生成了空、非有限或零能量 RIR")
        lengths.append(int(rir.size))

    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "conda_prefix": sys.prefix,
        "project_root": str(root),
        "package_source": str(package_path),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "matplotlib", "soundfile", "PyYAML",
                         "pyroomacoustics", "pytest")
        },
        "native_room_simulation": {"microphones": 4, "rir_lengths": lengths},
        "status": "passed",
    }
    output = root / "results" / "generated" / "environment_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

