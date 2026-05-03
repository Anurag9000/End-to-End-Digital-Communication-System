from __future__ import annotations

from dataclasses import replace

import numpy as np

from .backend import ArrayBackend, resolve_backend
from .config import PCMConfig, SourceConfig
from .pcm import sample_and_quantize
from .results import QuantizerSweepResult, SamplingSweepResult
from .source import generate_source_signal, nyquist_rate


def _as_int_array(values) -> np.ndarray:
    return np.asarray(list(values), dtype=int)


def _as_float_array(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def run_quantizer_sweep(
    source_cfg: SourceConfig,
    pcm_cfg: PCMConfig,
    *,
    backend: ArrayBackend | None = None,
    sweep_axis: str = "bits",
    sweep_values: list[int] | np.ndarray | None = None,
) -> QuantizerSweepResult:
    backend = backend or resolve_backend("auto")
    source = generate_source_signal(source_cfg, backend)

    if sweep_values is None:
        sweep_values = list(range(2, 9))

    values = _as_int_array(sweep_values)
    bits = np.zeros_like(values)
    levels = np.zeros_like(values)
    sqnr = np.zeros_like(values, dtype=float)
    delta = np.zeros_like(values, dtype=float)

    for idx, value in enumerate(values):
        if sweep_axis == "levels":
            pcm = sample_and_quantize(source.t, source.signal, replace(pcm_cfg, quantizer_mode="levels", quantizer_levels=int(value)), backend)
        else:
            pcm = sample_and_quantize(source.t, source.signal, replace(pcm_cfg, quantizer_mode="bits", bits_per_sample=int(value)), backend)
        bits[idx] = pcm.bits_per_sample
        levels[idx] = pcm.levels
        sqnr[idx] = pcm.sqnr_db
        delta[idx] = pcm.delta

    return QuantizerSweepResult(
        values=values,
        bits_per_sample=bits,
        levels=levels,
        sqnr_db=sqnr,
        delta=delta,
        quantizer_mode=sweep_axis,
    )


def run_sampling_sweep(
    source_cfg: SourceConfig,
    pcm_cfg: PCMConfig,
    *,
    backend: ArrayBackend | None = None,
    sample_rates: list[float] | np.ndarray | None = None,
) -> SamplingSweepResult:
    backend = backend or resolve_backend("auto")
    source = generate_source_signal(source_cfg, backend)

    if sample_rates is None:
        sample_rates = np.linspace(max(100.0, source_cfg.sim_fs / 8.0), source_cfg.sim_fs * 2.0, 7)

    sample_rates_arr = _as_float_array(sample_rates)
    sqnr = np.zeros_like(sample_rates_arr, dtype=float)
    counts = np.zeros_like(sample_rates_arr, dtype=int)
    below = np.zeros_like(sample_rates_arr, dtype=bool)
    nyq = nyquist_rate(source_cfg)

    for idx, fs in enumerate(sample_rates_arr):
        pcm = sample_and_quantize(source.t, source.signal, replace(pcm_cfg, sample_fs=float(fs)), backend)
        sqnr[idx] = pcm.sqnr_db
        counts[idx] = len(pcm.t_sampled)
        below[idx] = bool(fs < nyq) if nyq > 0 else False

    return SamplingSweepResult(
        sample_fs=sample_rates_arr,
        sqnr_db=sqnr,
        sample_counts=counts,
        nyquist_rate=float(nyq),
        is_below_nyquist=below,
    )
