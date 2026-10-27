"""MVDR 的物理约束、协方差共轭方向和编辑距离的独立已知答案。"""

import numpy as np
import pytest

from farfield_asr.arrays import linear_array, steering_vector
from farfield_asr.beamforming import das_weights, mvdr_weights, spatial_covariance
from farfield_asr.frontends import Frontend
from farfield_asr.metrics import corpus_cer, edit_distance, normalize_text, si_sdr
from farfield_asr.scenes import Scene


def test_covariance_complex_orientation_and_zero_mask():
    x = np.array([[[1, 1j], [1, 1j]]])
    expected = np.array([[[1, -1j], [1j, 1]]])
    np.testing.assert_allclose(spatial_covariance(x), expected)
    np.testing.assert_allclose(spatial_covariance(x, [[0, 0]]), 0)


def test_mvdr_distortionless_and_lower_noise_than_das():
    positions = linear_array()
    target = steering_vector(positions, [1000, 4000], 60)
    interference = steering_vector(positions, [1000, 4000], 150)
    covariance = 50 * np.einsum("fm,fn->fmn", interference, interference.conj()) + np.eye(4)
    mvdr = mvdr_weights(target, covariance)
    das = das_weights(target)
    np.testing.assert_allclose(np.einsum("fm,fm->f", mvdr.conj(), target), 1, atol=1e-12)
    mvdr_noise = np.einsum("fm,fmn,fn->f", mvdr.conj(), covariance, mvdr).real
    das_noise = np.einsum("fm,fmn,fn->f", das.conj(), covariance, das).real
    assert np.all(mvdr_noise < das_noise)


def test_singular_and_silent_covariance_is_stable():
    target = np.ones((3, 4), dtype=complex)
    weights = mvdr_weights(target, np.zeros((3, 4, 4)))
    np.testing.assert_allclose(weights, 0.25)
    with pytest.raises(ValueError, match="semidefinite"):
        mvdr_weights(target, np.tile(-np.eye(4), (3, 1, 1)))


def test_frontend_interface_and_calibration_guard():
    scene = Scene("test", 1, (6, 5, 3), "linear", 4, 0.04, (3, 2.5, 1),
                  (3, 3.5, 1), (2, 2.5, 1), 0.2, 5, 0)
    target = np.random.default_rng(8).normal(size=16000)
    x = np.tile(target[:, None], (1, 4))
    np.testing.assert_allclose(Frontend("das", scene).process(x), target, atol=1e-12)
    np.testing.assert_array_equal(Frontend("ref_mic", scene).process(x), target)
    with pytest.raises(ValueError, match="calibration"):
        Frontend("oracle_mvdr", scene).process(x)


def test_si_sdr_known_ratio_and_scale_invariance():
    ref = np.array([1, -1, 1, -1], dtype=float)
    orthogonal = np.array([1, 1, -1, -1], dtype=float)
    assert si_sdr(ref, ref + orthogonal) == pytest.approx(0)
    assert si_sdr(ref, 3 * (ref + orthogonal)) == pytest.approx(0)
    assert si_sdr(ref, 2 * ref) > 110
    with pytest.raises(ValueError):
        si_sdr(ref, np.zeros_like(ref))


@pytest.mark.parametrize("reference,hypothesis,expected", [
    ("", "", 0), ("", "你好", 2), ("你好", "", 2), ("你好", "您好", 1),
    ("kitten", "sitting", 3), ("天气很好", "今天天气好", 3),
])
def test_edit_distance(reference, hypothesis, expected):
    assert edit_distance(reference, hypothesis) == expected


def test_cer_is_character_weighted_not_utterance_mean():
    result = corpus_cer([("你", ""), ("你好世界", "你好世界")])
    assert result == {"cer": 0.2, "edits": 1, "reference_characters": 5}
    assert normalize_text("你好， ＡＳＲ！123") == "你好asr123"
