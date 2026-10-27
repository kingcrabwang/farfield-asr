"""边界样本、通道顺序和精确长度是后续所有前端评测的共同前提。"""

import numpy as np
import pytest

from farfield_asr.arrays import linear_array, steering_vector
from farfield_asr.audio_io import read_audio, write_audio
from farfield_asr.beamforming import apply_weights, das_weights
from farfield_asr.transforms import STFTConfig, istft, stft


@pytest.mark.parametrize("length", [1, 127, 512, 1001, 16000])
@pytest.mark.parametrize("channels", [1, 4, 6])
def test_roundtrip_preserves_boundaries_and_channel_order(length, channels):
    x = np.random.default_rng(42).normal(size=(length, channels))
    x[0] += 2
    x[-1] -= 3
    z = stft(x)
    assert z.shape[0] == 257
    np.testing.assert_allclose(istft(z, length), x, atol=1e-12, rtol=1e-12)


def test_single_channel_frontend_output_is_supported():
    x = np.random.default_rng(8).normal(size=1234)
    z = stft(x)
    np.testing.assert_allclose(istft(z[:, :, 0], len(x)), x, atol=1e-12)


@pytest.mark.parametrize("hop", [100, 128, 256])
def test_das_stft_chain_preserves_broadside_waveform(hop):
    config = STFTConfig(512, hop)
    target = np.random.default_rng(12).normal(size=1003)
    channels = np.tile(target[:, None], (1, 4))
    a = steering_vector(linear_array(), np.fft.rfftfreq(512, 1 / 16000), 90)
    output = istft(apply_weights(stft(channels, config), das_weights(a)), len(target), config)
    np.testing.assert_allclose(output, target, atol=1e-12)


def test_gain_and_frequency_bin_not_changed():
    config = STFTConfig(512, 128)
    n = np.arange(4096)
    x = np.cos(2 * np.pi * 32 * n / config.n_fft)
    z = stft(x, config)
    assert np.argmax(np.abs(z[:, 4, 0])) == 32
    np.testing.assert_allclose(np.abs(z[32, 4, 0]), config.n_fft / 4, atol=1e-10)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        istft(stft(np.ones(1000)), 500)


@pytest.mark.parametrize("audio", [[], [float("nan")], [1j]])
def test_invalid_audio_rejected(audio):
    with pytest.raises(ValueError):
        stft(audio)


def test_wav_float_roundtrip_and_sample_rate_guard(tmp_path):
    x = np.random.default_rng(7).normal(size=(517, 4))
    path = tmp_path / "multichannel.wav"
    write_audio(path, x)
    second = tmp_path / "repeat.wav"
    write_audio(second, x)
    assert path.read_bytes() == second.read_bytes()
    np.testing.assert_allclose(read_audio(path), x, atol=2e-7)
    with pytest.raises(ValueError, match="sample rate"):
        read_audio(path, expected_sample_rate=8000)
    with pytest.raises(FileExistsError):
        write_audio(path, x)

