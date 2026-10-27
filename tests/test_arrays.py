"""用独立的物理预期约束几何、时延符号与 DAS，而非只做运行检查。"""

import numpy as np
import pytest

from farfield_asr.arrays import circular_array, linear_array, steering_vector
from farfield_asr.beamforming import apply_weights, das_weights


def test_geometry():
    np.testing.assert_allclose(linear_array().mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(np.diff(linear_array()[:, 0]), 0.04)
    np.testing.assert_allclose(np.linalg.norm(circular_array(), axis=1), 0.04)


def test_positive_x_source_arrives_earlier_at_positive_x_mic():
    # 343 Hz、0.1 m 间距 -> 相位超前 0.2*pi，锁定 FFT 与方向符号。
    positions = np.array([[0, 0, 0], [0.1, 0, 0]])
    a = steering_vector(positions, [343], 0)
    np.testing.assert_allclose(a[0, 1], np.exp(0.2j * np.pi))


def test_broadside_no_interchannel_delay():
    a = steering_vector(linear_array(), [1000, 4000], 90)
    np.testing.assert_allclose(a, 1, atol=1e-14)


def test_das_preserves_target_and_suppresses_spatially_white_noise():
    a = steering_vector(linear_array(), [1000, 4000], 35)
    w = das_weights(a)
    rng = np.random.default_rng(20261027)
    signal = rng.normal(size=(2, 20)) + 1j * rng.normal(size=(2, 20))
    mixture = signal[:, :, None] * a[:, None, :]
    np.testing.assert_allclose(apply_weights(mixture, w), signal, atol=1e-14)
    # 单位白噪声协方差 I 下，输出噪声功率是 w^H I w=1/M。
    np.testing.assert_allclose(np.sum(np.abs(w) ** 2, axis=1), 0.25)


@pytest.mark.parametrize("spacing", [0, -0.1, float("nan")])
def test_reject_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        linear_array(spacing_m=spacing)


def test_reject_channel_mismatch():
    with pytest.raises(ValueError):
        apply_weights(np.zeros((3, 8, 4)), np.zeros((3, 6)))

