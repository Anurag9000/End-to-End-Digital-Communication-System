from __future__ import annotations

import numpy as np

from .config import LineCodeConfig
from .backend import ArrayBackend, resolve_backend
from .pulses import rectangular_pulse, sinc_pulse, raised_cosine_pulse
from .results import LineCodeResult


def bits_to_symbols(bits, encoding: str, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    bits = xp.asarray(bits, dtype=int)
    if encoding == "unipolar":
        return bits.astype(float)
    if encoding == "bipolar":
        mask = bits == 1
        parity = xp.cumsum(mask.astype(int)) % 2
        alternating = xp.where(parity == 1, 1.0, -1.0)
        return xp.where(mask, alternating, 0.0).astype(float)
    return (2 * bits - 1).astype(float)


def build_pulse(samples_per_bit: int, cfg: LineCodeConfig, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    if cfg.pulse_shape == "rect_rz":
        pulse, energy = rectangular_pulse(samples_per_bit, duty_cycle=cfg.rz_duty_cycle, normalize=cfg.normalize_pulse, backend=backend)
        return pulse, "rect_rz", energy
    if cfg.pulse_shape == "sinc":
        pulse, energy = sinc_pulse(samples_per_bit, cfg.pulse_span_symbols, normalize=cfg.normalize_pulse, backend=backend)
        return pulse, "sinc", energy
    if cfg.pulse_shape == "raised_cosine":
        pulse, energy = raised_cosine_pulse(samples_per_bit, cfg.pulse_span_symbols, cfg.rc_alpha, normalize=cfg.normalize_pulse, backend=backend)
        return pulse, "raised_cosine", energy
    pulse, energy = rectangular_pulse(samples_per_bit, duty_cycle=1.0, normalize=cfg.normalize_pulse, backend=backend)
    return pulse, "rect_nrz", energy


def build_line_code(bits, bit_rate: float, sim_fs: float, cfg: LineCodeConfig, backend: ArrayBackend | None = None) -> LineCodeResult:
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    bits = xp.asarray(bits, dtype=int)
    if bit_rate <= 0:
        raise ValueError("bit_rate must be positive")

    raw_samples_per_bit = sim_fs / bit_rate
    samples_per_bit = max(2, int(round(raw_samples_per_bit)))
    effective_bit_rate = sim_fs / samples_per_bit

    symbols = bits_to_symbols(bits, cfg.encoding, backend)
    impulse_train = xp.zeros(len(symbols) * samples_per_bit, dtype=float)
    impulse_train[::samples_per_bit] = symbols

    pulse, pulse_name, pulse_energy = build_pulse(samples_per_bit, cfg, backend)
    if len(impulse_train) == 0:
        waveform = xp.asarray([], dtype=float)
    else:
        waveform = xp.convolve(impulse_train, pulse, mode="full")
    t = xp.arange(len(waveform), dtype=float) / sim_fs
    # timing_offset is the correct sampling index in the MATCHED FILTER output
    # for the first bit. After two full convolutions (line code + MF), each of
    # length L, the total delay is 2*(L-1)//2 = L-1 samples.
    # For eye diagrams of the pre-MF line code waveform, use timing_offset//2
    # (the peak of the first bit in the line code waveform itself).
    timing_offset = len(pulse) - 1

    return LineCodeResult(
        bits=bits,
        symbols=symbols,
        impulse_train=impulse_train,
        pulse=pulse,
        waveform=waveform,
        t=t,
        bit_rate=effective_bit_rate,
        samples_per_bit=samples_per_bit,
        timing_offset=timing_offset,
        pulse_name=pulse_name,
        pulse_energy=pulse_energy,
    )
