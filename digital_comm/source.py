from __future__ import annotations

import numpy as np

from .config import SourceConfig
from .backend import ArrayBackend, resolve_backend
from .results import SourceResult


def generate_time_vector(duration: float, fs: float, backend: ArrayBackend):
    xp = backend.xp
    return xp.arange(0.0, max(duration, 0.0), 1.0 / fs)


def generate_tone(t, wave_type: str, amplitude: float, frequency: float, phase: float, backend: ArrayBackend):
    xp = backend.xp
    if wave_type == "cos":
        return amplitude * xp.cos(2 * xp.pi * frequency * t + phase)
    return amplitude * xp.sin(2 * xp.pi * frequency * t + phase)


def max_tone_frequency(cfg: SourceConfig) -> float:
    tones = cfg.tones or []
    if not tones:
        return 0.0
    if cfg.mode == "single":
        tones = tones[:1]
    return max(abs(float(tone.frequency)) for tone in tones)


def nyquist_rate(cfg: SourceConfig) -> float:
    return 2.0 * max_tone_frequency(cfg)


def generate_source_signal(cfg: SourceConfig, backend: ArrayBackend | None = None) -> SourceResult:
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    t = generate_time_vector(cfg.duration, cfg.sim_fs, backend)
    signal = xp.zeros_like(t)
    components: list[tuple[str, np.ndarray]] = []

    tones = cfg.tones or []
    if cfg.mode == "single" and tones:
        tones = tones[:1]

    for idx, tone in enumerate(tones):
        comp = generate_tone(t, tone.wave_type, tone.amplitude, tone.frequency, tone.phase, backend)
        signal = signal + comp
        label = f"tone_{idx + 1}_{tone.wave_type}_{tone.frequency:g}Hz"
        components.append((label, comp))

    return SourceResult(t=t, signal=signal, components=components)
