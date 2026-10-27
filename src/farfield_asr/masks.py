"""理想功率掩码、加权协方差及 Souden 掩码 MVDR。"""

import numpy as np

from farfield_asr.beamforming import spatial_covariance


def ideal_power_mask(target_spectrum, noise_spectrum):
    """功率 IRM = |S|²/(|S|²+|N|²)，不是直接乘幅度谱时常见的平方根 IRM。"""
    speech, noise = np.asarray(target_spectrum), np.asarray(noise_spectrum)
    if (speech.ndim != 2 or speech.shape != noise.shape or min(speech.shape) < 1
            or not np.all(np.isfinite(speech)) or not np.all(np.isfinite(noise))):
        raise ValueError("expected matching finite spectra (F,T)")
    speech_power, noise_power = np.abs(speech)**2, np.abs(noise)**2
    return speech_power / np.maximum(speech_power + noise_power, 1e-20)


def souden_weights(speech_covariance, noise_covariance, reference_mic=0,
                    diagonal_loading=1e-3, absolute_floor=1e-12):
    """w = (Rn_loaded^-1 Rs / trace(Rn_loaded^-1 Rs)) e_ref。

    通过语音协方差选参考通道响应，不读取目标 DOA。退化的零语音频点回退到参考麦克风。
    """
    speech, noise = np.asarray(speech_covariance), np.asarray(noise_covariance)
    if (speech.ndim != 3 or speech.shape != noise.shape or speech.shape[1] != speech.shape[2]
            or min(speech.shape) < 1 or not np.all(np.isfinite(speech))
            or not np.all(np.isfinite(noise))):
        raise ValueError("expected matching finite covariance matrices (F,M,M)")
    n_mics = speech.shape[1]
    if type(reference_mic) is not int or not 0 <= reference_mic < n_mics:
        raise ValueError("reference microphone out of range")
    if (not np.isfinite(diagonal_loading) or diagonal_loading < 0
            or not np.isfinite(absolute_floor) or absolute_floor <= 0):
        raise ValueError("invalid covariance loading")
    for covariance in (speech, noise):
        if not np.allclose(covariance, covariance.conj().transpose(0, 2, 1), atol=1e-10):
            raise ValueError("covariance must be Hermitian")
        scale = np.maximum(np.abs(np.trace(covariance, axis1=1, axis2=2).real), 1e-12)
        if np.any(np.linalg.eigvalsh(covariance) < -1e-8 * scale[:, None]):
            raise ValueError("covariance must be positive semidefinite")
    power = np.maximum(np.trace(noise, axis1=1, axis2=2).real / n_mics, 0)
    loaded = noise + (diagonal_loading * power + absolute_floor)[:, None, None] * np.eye(n_mics)
    product = np.linalg.solve(loaded, speech)
    denominator = np.trace(product, axis1=1, axis2=2).real
    weights = np.zeros((speech.shape[0], n_mics), dtype=complex)
    valid = denominator > 1e-20
    weights[valid] = product[valid, :, reference_mic] / denominator[valid, None]
    weights[~valid, reference_mic] = 1
    return weights


def mask_mvdr_weights(spectrum, speech_mask, reference_mic=0, diagonal_loading=1e-3):
    """掩码 -> 两个加权协方差 -> Souden MVDR；噪声掩码使用 1−语音掩码。"""
    mask = np.asarray(speech_mask, dtype=float)
    speech_cov = spatial_covariance(spectrum, mask)
    noise_cov = spatial_covariance(spectrum, 1 - mask)
    return souden_weights(speech_cov, noise_cov, reference_mic, diagonal_loading)


def log_power_features(spectrum, reference_mic=0):
    """(F,T,M) 参考麦克风对数功率 -> float32 (T,F)，不读取干净标签。"""
    x = np.asarray(spectrum)
    if (x.ndim != 3 or not 0 <= reference_mic < x.shape[2] or not np.all(np.isfinite(x))):
        raise ValueError("invalid multichannel spectrum or reference index")
    return np.log(np.maximum(np.abs(x[:, :, reference_mic])**2, 1e-10)).T.astype(np.float32)
