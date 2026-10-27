"""统一对齐和裁剪后的指标。缺失或失败的指标返回 null + 原因，不伪造零分。"""

import unicodedata

import numpy as np


def si_sdr(reference, estimate) -> float:
    reference, estimate = np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float)
    if (reference.ndim != 1 or reference.shape != estimate.shape or len(reference) < 2
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate))):
        raise ValueError("SI-SDR expects same-length finite mono signals")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    reference_power = np.dot(reference, reference)
    if reference_power <= 1e-20 or np.dot(estimate, estimate) <= 1e-20:
        raise ValueError("SI-SDR undefined for zero-energy reference or prediction")
    projection = np.dot(estimate, reference) / reference_power * reference
    error = estimate - projection
    floor = reference_power * 1e-12
    return float(10 * np.log10(max(np.dot(projection, projection), floor)
                               / max(np.dot(error, error), floor)))


def audio_metrics(reference, estimate, sample_rate: int = 16000) -> dict:
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    scores, errors = {}, {}
    functions = {"si_sdr_db": lambda: si_sdr(reference, estimate)}
    try:
        from pesq import pesq
        functions["pesq_wb"] = lambda: float(pesq(sample_rate, reference, estimate, "wb"))
    except ImportError:
        errors["pesq_wb"] = "pesq dependency unavailable"
    try:
        from pystoi import stoi
        functions["stoi"] = lambda: float(stoi(reference, estimate, sample_rate, extended=False))
    except ImportError:
        errors["stoi"] = "pystoi dependency unavailable"
    for name, function in functions.items():
        try:
            value = function()
            if not np.isfinite(value):
                raise ValueError("nonfinite metric result")
            scores[name] = value
        except (ValueError, RuntimeError) as error:
            errors[name] = str(error)
    for name in ("si_sdr_db", "pesq_wb", "stoi"):
        scores.setdefault(name, None)
    return {**scores, "metric_errors": errors}


def normalize_text(text: str) -> str:
    """固定 CER 协议：NFKC、小写，去 Unicode 标点与空白，保留数字和其他符号。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(char for char in text if not char.isspace()
                   and not unicodedata.category(char).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    """手写 Levenshtein 编辑距离，O(NM) 时间、O(M) 额外空间。"""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def corpus_cer(pairs) -> dict:
    errors, characters = 0, 0
    for reference, hypothesis in pairs:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
        errors += edit_distance(reference, hypothesis)
        characters += len(reference)
    if characters == 0:
        raise ValueError("CER requires at least one reference character")
    return {"cer": errors / characters, "edits": errors, "reference_characters": characters}
