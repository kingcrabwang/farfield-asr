"""检验物理空间相干性、SNR 定义与分量可加性，而非只测试函数能运行。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from farfield_asr.arrays import linear_array
from farfield_asr.scenes import Scene, sample_scenes
from farfield_asr.simulation import (
    RoomResponses,
    diffuse_noise,
    energy_decay,
    estimate_t30,
    render_scene,
    room_responses,
)


@pytest.fixture
def scene():
    return Scene("test", 42, (6, 5, 3), "linear", 4, 0.04, (3, 2.5, 1.2),
                 (4, 3.5, 1.2), (2, 1.5, 1.2), 0.2, 5, -3)


def test_scene_manifest_reproducibility_and_coverage():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/base.yaml").read_text())
    first = sample_scenes(config)
    second = sample_scenes(config)
    assert [s.to_dict() for s in first] == [s.to_dict() for s in second]
    assert len({s.seed for s in first}) == 100
    assert len({(s.array_geometry, s.rt60_s) for s in first}) == 6
    assert len({s.room_dimensions_m for s in first}) == 3
    assert len({s.snr_db for s in first}) == 4
    for item in first:
        # 与 JSON 清单反序列化后的对象保持相同物理坐标。
        np.testing.assert_allclose(Scene(**item.to_dict()).microphones_m, item.microphones_m)


def test_invalid_geometry_rejected(scene):
    with pytest.raises(ValueError, match="inside"):
        replace(scene, target_position_m=(6, 3, 1))
    with pytest.raises(ValueError, match="reference"):
        replace(scene, reference_mic=4)


def test_diffuse_field_matches_integrated_isotropic_coherence():
    positions = linear_array()
    n = 65536
    noise = diffuse_noise(n, positions, 16000, 42)
    np.testing.assert_array_equal(noise, diffuse_noise(n, positions, 16000, 42))
    frequencies = np.fft.rfftfreq(n, 1 / 16000)
    expected = [np.mean(np.sinc(2 * frequencies * distance / 343))
                for distance in (0.04, 0.08, 0.12)]
    correlation = np.corrcoef(noise.T)[0, 1:]
    np.testing.assert_allclose(correlation, expected, atol=0.025)


def test_schroeder_fit_on_known_exponential_decay():
    fs = 16000
    target_rt60 = 0.4
    t = np.arange(fs) / fs
    # 能量以 60 dB / RT60 的斜率衰减。
    rir = 10 ** (-3 * t / target_rt60)
    result = estimate_t30(rir, fs)
    assert result["rt60_t30_s"] == pytest.approx(target_rt60, abs=0.001)
    decay = energy_decay(rir)
    assert decay[0] == 0 and np.all(np.diff(decay) <= 0)


def test_render_preserves_components_ratios_and_determinism(scene):
    # 人工 RIR 锁定混合定义，避免测试结果依赖 Sabine 近似。
    impulse = np.zeros((10, 4))
    impulse[3] = [1, 0.8, 0.7, 0.6]
    rir = RoomResponses(impulse, impulse * 0.3, impulse, 0.5, 0)
    rng = np.random.default_rng(9)
    target, interference = rng.normal(size=(2, 4096))
    result = render_scene(scene, target, interference, rir)
    repeat = render_scene(scene, target, interference, rir)
    np.testing.assert_allclose(result["mixture"], result["target_reverberant"]
                               + result["interference"] + result["diffuse"], atol=1e-15)
    np.testing.assert_array_equal(result["mixture"], repeat["mixture"])
    assert np.max(np.abs(result["mixture"])) <= 0.950000001
    assert result["metadata"]["snr_db_measured"] == pytest.approx(scene.snr_db, abs=1e-10)
    assert result["metadata"]["sir_db_measured"] == pytest.approx(scene.sir_db, abs=1e-10)


def test_real_room_rir_reproducibility_and_direct_path_delay(scene):
    first = room_responses(scene)
    second = room_responses(scene)
    np.testing.assert_array_equal(first.target, second.target)
    assert first.target.shape[1] == 4
    assert first.max_order > 0
    distance = np.linalg.norm(scene.microphones_m[0] - scene.target_position_m)
    # pyroomacoustics 的分数时延 FIR 引入 (L-1)/2 点公共群延迟。
    import pyroomacoustics as pra
    group_delay = (pra.constants.get("frac_delay_length") - 1) / 2
    expected = distance / scene.sound_speed_m_s * scene.sample_rate + group_delay
    assert abs(np.argmax(np.abs(first.direct[:, 0])) - expected) <= 1
    assert np.sum(first.target**2) > np.sum(first.direct**2)

