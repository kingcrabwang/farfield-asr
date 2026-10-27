"""在 farfield-asr 环境中运行补建的项目一单通道模型。"""

import argparse
from pathlib import Path

from denoiser import SingleChannelDenoiser

from farfield_asr.audio_io import read_audio, write_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    audio = read_audio(args.input)
    if audio.shape[1] != 1:
        raise ValueError("input must be mono; explicitly select the reference microphone first")
    denoiser = SingleChannelDenoiser(args.checkpoint)
    write_audio(args.output, denoiser.process(audio[:, 0]))


if __name__ == "__main__":
    main()
