"""验证掩码定义与 Souden 在秩一语音模型下退化为经典 MVDR。"""

import numpy as np

from farfield_asr.arrays import linear_array, steering_vector
from farfield_asr.beamforming import mvdr_weights
from farfield_asr.masks import ideal_power_mask, log_power_features, souden_weights


def test_power_mask_definition_and_silence():
    target, noise = np.array([[2, 0]], dtype=complex), np.array([[1, 0]], dtype=complex)
    np.testing.assert_allclose(ideal_power_mask(target, noise), [[0.8, 0]])


def test_souden_matches_classical_rank_one_mvdr():
    target = steering_vector(linear_array(), [1000, 4000], 60)
    interferer = steering_vector(linear_array(), [1000, 4000], 145)
    speech = np.einsum("fm,fn->fmn", target, target.conj())
    noise = 8 * np.einsum("fm,fn->fmn", interferer, interferer.conj()) + np.eye(4)
    np.testing.assert_allclose(souden_weights(speech, noise), mvdr_weights(target, noise), atol=1e-12)
    np.testing.assert_allclose(np.einsum("fm,fm->f", souden_weights(speech, noise).conj(), target), 1)


def test_zero_speech_fallback_and_feature_layout():
    weights = souden_weights(np.zeros((2, 4, 4)), np.zeros((2, 4, 4)))
    np.testing.assert_array_equal(weights, [[1, 0, 0, 0], [1, 0, 0, 0]])
    features = log_power_features(np.ones((257, 30, 4), dtype=complex))
    assert features.shape == (30, 257) and features.dtype == np.float32
    np.testing.assert_allclose(features, 0)
