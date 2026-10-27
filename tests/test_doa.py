"""用独立生成的远场宽带信号验证符号、镜像处理、圆阵方位及异常输入。"""

import numpy as np
import pytest

from farfield_asr.arrays import circular_array, linear_array, steering_vector
from farfield_asr.doa import angular_error, estimate_doa


@pytest.mark.parametrize("geometry,angle", [("linear", 30), ("linear", 90),
                                           ("circular", 210), ("circular", 315)])
@pytest.mark.parametrize("method", ["gcc_phat", "srp_phat", "music"])
def test_farfield_broadband_direction(geometry, angle, method):
    positions = linear_array() if geometry == "linear" else circular_array()
    rng = np.random.default_rng(42)
    target = rng.normal(size=16000)
    frequencies = np.fft.rfftfreq(len(target), 1 / 16000)
    a = steering_vector(positions, frequencies, angle)
    audio = np.fft.irfft(np.fft.rfft(target)[:, None] * a, n=len(target), axis=0)
    audio += rng.normal(0, 0.02, size=audio.shape)
    result = estimate_doa(audio, positions, method, geometry)
    assert angular_error(result.azimuth_deg, angle, geometry) <= 3
    assert np.all(np.isfinite(result.scores))


def test_circular_distance_wrap_and_reject_silence():
    assert angular_error(359, 1, "circular") == 2
    with pytest.raises(ValueError):
        estimate_doa(np.zeros((1024, 4)), linear_array(), "music", "linear")
