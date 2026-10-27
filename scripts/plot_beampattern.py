"""生成 4 麦线阵的 DAS 方向图，以及用于解释栅瓣的对照图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from farfield_asr.arrays import linear_array, steering_vector
from farfield_asr.beamforming import das_weights


def main():
    output = Path(__file__).resolve().parents[1] / "results" / "generated"
    output.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(0, 180, 1801)
    frequencies = [1000, 4000]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # 4 cm 在 4 kHz 满足 d <= c/(2f)；8 cm 对照展示偏转波束的栅瓣。
    for ax, spacing, look in zip(axes, [0.04, 0.08], [60, 30]):
        positions = linear_array(4, spacing)
        weights = das_weights(steering_vector(positions, frequencies, look))
        response = np.stack([
            np.abs(np.sum(weights.conj() * steering_vector(positions, frequencies, a), axis=1))
            for a in angles
        ])
        for index, frequency in enumerate(frequencies):
            db = 20 * np.log10(np.maximum(response[:, index], 1e-4))
            ax.plot(angles, db, label=f"{frequency / 1000:g} kHz")
        ax.axvline(look, color="black", linestyle="--", label="Look direction / main lobe")
        if spacing == 0.08:
            # cos(theta_alias) = cos(theta_look) - lambda/d。
            alias = np.rad2deg(np.arccos(np.cos(np.deg2rad(look)) - 343 / 4000 / spacing))
            ax.axvline(alias, color="red", linestyle=":", label="4 kHz grating lobe")
        ax.set(xlabel="Source azimuth (degrees)", ylim=(-60, 2), xlim=(0, 180),
               title=f"4 microphones, d={spacing * 100:g} cm, look={look} deg")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Response (dB)")
    fig.tight_layout()
    fig.savefig(output / "das_beampattern.png", dpi=160)
    plt.close(fig)
    print(output / "das_beampattern.png")


if __name__ == "__main__":
    main()

