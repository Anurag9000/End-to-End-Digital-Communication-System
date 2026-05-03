from __future__ import annotations

import numpy as np

from .backend import ArrayBackend, resolve_backend


def normalize_energy(pulse, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    energy = float(xp.sum(pulse**2))
    if energy > 0:
        pulse = pulse / xp.sqrt(energy)
    return pulse, energy


def rectangular_pulse(samples_per_bit: int, duty_cycle: float = 1.0, normalize: bool = False, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    samples_per_bit = max(1, int(samples_per_bit))
    active = max(1, min(samples_per_bit, int(round(samples_per_bit * duty_cycle))))
    pulse = xp.zeros(samples_per_bit, dtype=float)
    pulse[:active] = 1.0
    if normalize:
        pulse, energy = normalize_energy(pulse, backend)
    else:
        energy = float(xp.sum(pulse**2))
    return pulse, energy


def sinc_pulse(samples_per_bit: int, span_symbols: int, normalize: bool = False, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    samples_per_bit = max(1, int(samples_per_bit))
    span_symbols = max(1, int(span_symbols))
    n = 2 * span_symbols * samples_per_bit + 1
    t = xp.linspace(-span_symbols, span_symbols, n)
    pulse = xp.sinc(t)
    if normalize:
        pulse, energy = normalize_energy(pulse, backend)
    else:
        energy = float(xp.sum(pulse**2))
    return pulse.astype(float), energy


def raised_cosine_pulse(samples_per_bit: int, span_symbols: int, alpha: float, normalize: bool = False, backend: ArrayBackend | None = None):
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    samples_per_bit = max(1, int(samples_per_bit))
    span_symbols = max(1, int(span_symbols))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n = 2 * span_symbols * samples_per_bit + 1
    t = xp.linspace(-span_symbols, span_symbols, n)
    sinc_part = xp.sinc(t)
    denom = 1 - (2 * alpha * t) ** 2
    # The limit of cos(pi*alpha*t) / (1 - (2*alpha*t)^2) as 2*alpha*t -> +-1 is pi/4
    safe_denom = xp.where(xp.abs(denom) < 1e-12, 1.0, denom)
    cosine_part = xp.cos(xp.pi * alpha * t) / safe_denom
    cosine_part = xp.where(xp.abs(denom) < 1e-12, xp.pi / 4, cosine_part)
    pulse = sinc_part * cosine_part
    if normalize:
        pulse, energy = normalize_energy(pulse, backend)
    else:
        energy = float(xp.sum(pulse**2))
    return pulse.astype(float), energy
