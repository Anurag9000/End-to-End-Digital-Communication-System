from __future__ import annotations

import math
import numpy as np

from .config import ChannelConfig
from .backend import ArrayBackend, resolve_backend
from .results import ChannelResult


def add_awgn_noise(waveform, cfg: ChannelConfig, backend: ArrayBackend | None = None) -> ChannelResult:
    backend = backend or resolve_backend("auto")
    xp = backend.xp
    tx = xp.asarray(waveform, dtype=float)
    if len(tx) == 0:
        empty = xp.asarray([], dtype=float)
        return ChannelResult(empty, empty, empty, 0.0, 0.0, cfg.snr_db, float("nan"))

    signal_power = float(xp.mean(tx**2))
    if not cfg.enable_awgn:
        noisy = tx.copy()
        noise = xp.zeros_like(tx)
        noise_power = 0.0
        achieved = float("inf")
        return ChannelResult(tx, noisy, noise, signal_power, noise_power, cfg.snr_db, achieved)

    # Clamp snr_db to a finite range so NaN/±inf inputs still produce a deterministic channel.
    snr_db_safe = float(np.clip(cfg.snr_db, -200.0, 200.0)) if math.isfinite(cfg.snr_db) else -200.0
    snr_linear = 10 ** (snr_db_safe / 10.0)
    noise_power = signal_power / snr_linear if snr_linear > 0 else signal_power
    rng = xp.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, float(np.sqrt(noise_power)), size=tx.shape)
    noisy = tx + noise
    actual_noise_power = float(xp.mean(noise**2))
    if signal_power == 0.0:
        achieved = float("-inf")
    elif actual_noise_power <= 1e-300:
        achieved = float("inf")
    else:
        achieved = float(10 * np.log10(signal_power / actual_noise_power))
    return ChannelResult(tx, noisy, noise, signal_power, noise_power, cfg.snr_db, achieved)
