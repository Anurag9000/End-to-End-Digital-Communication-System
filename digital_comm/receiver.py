from __future__ import annotations

import numpy as np

from .config import ReceiverConfig
from .backend import ArrayBackend, resolve_backend, to_scalar
from .results import ReceiverResult


def apply_matched_filter(noisy_waveform, pulse, normalize: bool = True, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    mf = pulse[::-1]
    if len(noisy_waveform) == 0:
        return xp.asarray([], dtype=float)
    y = xp.convolve(noisy_waveform, mf, mode="full")
    if normalize:
        energy = float(xp.sum(pulse**2))
        if energy > 0:
            y = y / energy
    return y


def detect_bits(
    matched_waveform,
    original_bits,
    encoding: str,
    samples_per_bit: int,
    timing_offset: int,
    cfg: ReceiverConfig,
    backend: ArrayBackend | None = None,
) -> ReceiverResult:
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    n = len(original_bits)
    sample_indices = timing_offset + xp.arange(n, dtype=int) * samples_per_bit
    valid_mask = (sample_indices >= 0) & (sample_indices < len(matched_waveform))
    sample_indices = sample_indices[valid_mask]
    sampled_voltages = matched_waveform[sample_indices]

    if cfg.threshold_mode == "manual":
        threshold = float(cfg.manual_threshold)
    else:
        threshold = 0.35 if encoding in {"unipolar", "bipolar"} else 0.0

    detected = xp.zeros(len(sampled_voltages), dtype=int)
    if encoding == "unipolar":
        detected[sampled_voltages > threshold] = 1
    elif encoding == "bipolar":
        detected[xp.abs(sampled_voltages) > threshold] = 1
    else:
        detected[sampled_voltages > threshold] = 1

    compare_len = min(len(original_bits), len(detected))
    errors = int(to_scalar(xp.sum(xp.asarray(original_bits[:compare_len]) != detected[:compare_len])))
    # compare_len == 0 means every sample index was out of bounds (e.g. manual
    # timing offset beyond waveform length). Return BER=1.0 (worst case) rather
    # than NaN so the UI always gets a valid, displayable number.
    ber = float(errors / compare_len) if compare_len else 1.0

    return ReceiverResult(
        matched_waveform=matched_waveform,
        detected_bits=detected[:compare_len],
        sampled_voltages=sampled_voltages[:compare_len],
        sample_indices=sample_indices[:compare_len],
        ber=ber,
        errors=errors,
        threshold=threshold,
        timing_offset=timing_offset,
    )
