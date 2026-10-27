"""房间 RIR、三维球面各向同性扩散噪声及可分解的多通道混合。"""

from dataclasses import dataclass

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from farfield_asr.scenes import Scene


@dataclass
class RoomResponses:
    target: np.ndarray
    interference: np.ndarray
    direct: np.ndarray
    absorption: float
    max_order: int


def room_responses(scene: Scene) -> RoomResponses:
    """生成 (L,M) RIR；max_order 使用 Sabine 反演值，不人为截短混响尾部。"""
    absorption, order = pra.inverse_sabine(
        scene.rt60_s, scene.room_dimensions_m, c=scene.sound_speed_m_s
    )
    room = pra.ShoeBox(scene.room_dimensions_m, fs=scene.sample_rate,
                       materials=pra.Material(absorption), max_order=int(order))
    room.set_sound_speed(scene.sound_speed_m_s)
    room.add_microphone_array(scene.microphones_m.T)
    room.add_source(scene.target_position_m)
    room.add_source(scene.interference_position_m)
    room.compute_rir()
    direct_room = pra.ShoeBox(scene.room_dimensions_m, fs=scene.sample_rate, max_order=0)
    direct_room.set_sound_speed(scene.sound_speed_m_s)
    direct_room.add_microphone_array(scene.microphones_m.T)
    direct_room.add_source(scene.target_position_m)
    direct_room.compute_rir()
    arrays = [[room.rir[m][s] for m in range(scene.n_mics)] for s in (0, 1)]
    arrays.append([direct_room.rir[m][0] for m in range(scene.n_mics)])
    length = max(len(rir) for group in arrays for rir in group)
    padded = [np.column_stack([np.pad(rir, (0, length - len(rir))) for rir in group])
              for group in arrays]
    return RoomResponses(*padded, absorption=float(absorption), max_order=int(order))


def energy_decay(rir) -> np.ndarray:
    """Schroeder 反向积分曲线（dB）。零能量 RIR 显式报错。"""
    h = np.asarray(rir, dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.all(np.isfinite(h)) or np.sum(h**2) <= 0:
        raise ValueError("rir must be a finite nonzero 1-D vector")
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    return 10 * np.log10(np.maximum(energy / energy[0], 1e-12))


def estimate_t30(rir, sample_rate: int) -> dict:
    """拟合 -5 到 -35 dB 衰减并外推 RT60，同时返回拟合质量，绝不冒充精确真值。"""
    decay = energy_decay(rir)
    selected = (decay <= -5) & (decay >= -35)
    if np.count_nonzero(selected) < 10:
        raise ValueError("insufficient -5 to -35 dB decay for T30 fit")
    time = np.arange(len(decay)) / sample_rate
    slope, intercept = np.polyfit(time[selected], decay[selected], 1)
    predicted = slope * time[selected] + intercept
    residual = np.sum((decay[selected] - predicted)**2)
    total = np.sum((decay[selected] - decay[selected].mean())**2)
    if slope >= 0 or total == 0:
        raise ValueError("invalid decay fit")
    return {"rt60_t30_s": float(-60 / slope), "r_squared": float(1 - residual / total),
            "fit_start_s": float(time[selected][0]), "fit_end_s": float(time[selected][-1])}


def diffuse_noise(n_samples: int, microphones_m, sample_rate: int, seed: int,
                  sound_speed: float = 343.0) -> np.ndarray:
    """用 Γ_ij(f)=sinc(2fd_ij/c) 的谱矩阵平方根合成三维球面各向同性白扩散场。

    每个频点独立高斯激励；特征值仅截去数值误差导致的负值。
    采用整段 FFT，属于离线有限长度噪声；不把独立通道白噪声冒充扩散场。
    """
    positions = np.asarray(microphones_m, dtype=float)
    if (type(n_samples) is not int or n_samples < 2 or positions.ndim != 2
            or positions.shape[1] != 3 or positions.shape[0] < 2
            or not np.all(np.isfinite(positions))):
        raise ValueError("invalid noise length or microphone positions")
    if sample_rate <= 0 or not np.isfinite(sound_speed) or sound_speed <= 0:
        raise ValueError("invalid sample rate or sound speed")
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    frequencies = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    coherence = np.sinc(2 * frequencies[:, None, None] * distances / sound_speed)
    eigenvalues, eigenvectors = np.linalg.eigh(coherence)
    rng = np.random.default_rng(seed)
    excitation = (rng.standard_normal((len(frequencies), len(positions)))
                  + 1j * rng.standard_normal((len(frequencies), len(positions)))) / np.sqrt(2)
    excitation[0] = rng.standard_normal(len(positions))
    if n_samples % 2 == 0:
        excitation[-1] = rng.standard_normal(len(positions))
    spectrum = np.einsum("fij,fj->fi", eigenvectors,
                         np.sqrt(np.maximum(eigenvalues, 0)) * excitation)
    noise = np.fft.irfft(spectrum, n=n_samples, axis=0)
    return noise / np.sqrt(np.mean(noise**2))


def render_scene(scene: Scene, target_audio, interference_audio, responses=None) -> dict:
    """返回同长分量。SNR 指目标混响声/扩散噪声，SIR 指目标混响声/点源干扰。

    两者均在参考麦克风、含传播前静音与完整混响尾部的全长度上测量。
    最后对所有分量使用同一个增益，限制混合峰值到 0.95，保持比值与可加性。
    """
    sources = [np.asarray(x, dtype=float) for x in (target_audio, interference_audio)]
    if any(x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x))
           or np.mean(x**2) <= 0 for x in sources):
        raise ValueError("sources must be finite nonzero mono signals")
    if responses is None:
        responses = room_responses(scene)
    n_samples = max(len(x) for x in sources) + len(responses.target) - 1

    def convolve(source, rir):
        signal = np.column_stack([fftconvolve(source, rir[:, m]) for m in range(scene.n_mics)])
        return np.pad(signal, ((0, n_samples - len(signal)), (0, 0)))

    target = convolve(sources[0], responses.target)
    direct = convolve(sources[0], responses.direct)
    interference = convolve(sources[1], responses.interference)
    diffuse = diffuse_noise(n_samples, scene.microphones_m, scene.sample_rate,
                            scene.seed, scene.sound_speed_m_s)
    ref = scene.reference_mic
    target_power = np.mean(target[:, ref]**2)
    for component, ratio_db in ((interference, scene.sir_db), (diffuse, scene.snr_db)):
        power = np.mean(component[:, ref]**2)
        if target_power <= 0 or power <= 0:
            raise ValueError("reference-channel component power must be positive")
        component *= np.sqrt(target_power / (power * 10**(ratio_db / 10)))
    mixture = target + interference + diffuse
    gain = min(1.0, 0.95 / np.max(np.abs(mixture)))
    components = {"mixture": mixture, "target_reverberant": target,
                  "target_direct": direct, "interference": interference, "diffuse": diffuse}
    components = {name: signal * gain for name, signal in components.items()}
    components["metadata"] = {
        "snr_db_measured": float(10 * np.log10(target_power / np.mean(diffuse[:, ref]**2))),
        "sir_db_measured": float(10 * np.log10(target_power / np.mean(interference[:, ref]**2))),
        "common_gain": float(gain), "n_samples": n_samples,
        "reference_mic": ref, "reference_definition": "direct-path target at reference mic",
        "alignment": "propagation delay and simulator fractional-delay filter retained",
    }
    return components

