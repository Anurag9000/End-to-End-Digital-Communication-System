from __future__ import annotations

import numpy as np

from .config import PCMConfig
from .backend import ArrayBackend, resolve_backend, to_numpy
from .results import PCMResult
from .utils import bits_to_binary_strings, binary_strings_to_bits


def _interp_backend(xp, x, xp_samples, y):
    interp = getattr(xp, "interp", None)
    if interp is not None:
        return interp(xp_samples, x, y)
    sampled = np.interp(to_numpy(xp_samples), to_numpy(x), to_numpy(y))
    return xp.asarray(sampled)


def resolve_quantizer_spec(cfg: PCMConfig) -> tuple[int, int]:
    if cfg.quantizer_mode == "levels":
        levels = max(2, int(cfg.quantizer_levels))
        bits_per_sample = max(1, int(np.ceil(np.log2(levels))))
        return bits_per_sample, levels

    bits_per_sample = max(1, int(cfg.bits_per_sample))
    levels = 2**bits_per_sample
    return bits_per_sample, levels


def sample_and_quantize(t, signal, cfg: PCMConfig, backend: ArrayBackend | None = None) -> PCMResult:
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    if len(t) == 0 or len(signal) == 0:
        empty = xp.asarray([], dtype=float)
        empty_bits = np.array([], dtype=int)
        bits_per_sample, levels = resolve_quantizer_spec(cfg)
        return PCMResult(
            empty,
            empty,
            empty,
            np.array([], dtype=int),
            empty_bits,
            bits_per_sample,
            levels,
            cfg.quantizer_mode,
            float(cfg.sample_fs),
            0.0,
            0.0,
            0.0,
            float("nan"),
            [],
        )
    if cfg.sample_fs <= 0:
        raise ValueError("sample_fs must be positive")
    if len(t) != len(signal):
        raise ValueError(
            f"t and signal must have the same length, got {len(t)} and {len(signal)}"
        )

    # Use a tiny epsilon (1e-9 sample periods) instead of 0.5 sample periods to
    # avoid generating a spurious extra sample when t[-1] is exactly divisible
    # by 1/fs (which happens at standard sim parameters like duration=0.05, fs=5000).
    t_sampled = xp.arange(float(t[0]), float(t[-1]) + 1e-9 / cfg.sample_fs, 1.0 / cfg.sample_fs)
    sampled_signal = _interp_backend(xp, t, t_sampled, signal)

    bits_per_sample, levels = resolve_quantizer_spec(cfg)

    if cfg.range_mode == "manual":
        v_min = float(cfg.manual_min)
        v_max = float(cfg.manual_max)
    else:
        v_min = float(xp.min(signal))
        v_max = float(xp.max(signal))
    if v_max <= v_min:
        if cfg.range_mode == "auto":
            v_max = v_min + 1.0
        else:
            raise ValueError("Quantizer top voltage must be greater than bottom voltage")

    delta = (v_max - v_min) / levels if v_max > v_min else 1.0

    indices = xp.floor((sampled_signal - v_min) / delta).astype(int) if delta > 0 else xp.zeros_like(sampled_signal, dtype=int)
    indices = xp.clip(indices, 0, levels - 1)
    quantized_signal = v_min + (indices + 0.5) * delta

    codewords = bits_to_binary_strings(to_numpy(indices).astype(int), int(bits_per_sample))
    pcm_bits = binary_strings_to_bits(codewords)

    reconstructed_signal = _interp_backend(xp, t_sampled, t, quantized_signal)
    quantization_noise = signal - reconstructed_signal
    signal_power = float(xp.mean(signal**2))
    noise_power = float(xp.mean(quantization_noise**2))
    if signal_power == 0.0:
        sqnr_db = float("-inf")
    elif noise_power == 0.0:
        sqnr_db = float("inf")
    else:
        sqnr_db = float(10 * np.log10(signal_power / noise_power))

    return PCMResult(
        t_sampled=t_sampled,
        sampled_signal=sampled_signal,
        quantized_signal=quantized_signal,
        quantization_indices=indices,
        pcm_bits=pcm_bits,
        bits_per_sample=int(bits_per_sample),
        levels=int(levels),
        quantizer_mode=str(cfg.quantizer_mode),
        sample_fs=float(cfg.sample_fs),
        delta=float(delta),
        v_min=v_min,
        v_max=v_max,
        sqnr_db=sqnr_db,
        codewords=codewords,
    )
