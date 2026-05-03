from __future__ import annotations

from dataclasses import replace as dc_replace
import numpy as np

from .backend import resolve_backend
from .config import AppConfig
from .channel import add_awgn_noise
from .line_coding import build_line_code
from .pcm import resolve_quantizer_spec, sample_and_quantize
from .receiver import apply_matched_filter, detect_bits
from .results import EndToEndResult, SweepResult
from .source import generate_source_signal
from .utils import parse_bit_string


def _select_payload_bits(cfg: AppConfig, pcm_bits, num_bits: int | None = None, backend=None):
    mode = cfg.line.payload_mode
    if mode == "custom_bits":
        bits = parse_bit_string(cfg.line.custom_bits)
        if len(bits) == 0:
            bits = pcm_bits
        return bits
    if mode == "random_bits":
        n = int(num_bits if num_bits is not None else cfg.ber.num_bits)
        xp = backend.xp if backend is not None else np
        rng = xp.random.default_rng(cfg.ber.seed)
        return rng.integers(0, 2, size=n, dtype=int)
    return pcm_bits


def _line_bit_rate(cfg: AppConfig, quantizer_bits: int) -> float:
    if cfg.line.bit_rate_mode == "manual":
        return float(cfg.line.manual_bit_rate)
    if cfg.pcm.sample_fs <= 0:
        return float(cfg.line.manual_bit_rate)
    return float(cfg.pcm.sample_fs * max(1, int(quantizer_bits)))


def run_end_to_end(cfg: AppConfig) -> EndToEndResult:
    backend = resolve_backend(cfg.compute_backend)
    source = generate_source_signal(cfg.source, backend)
    pcm = sample_and_quantize(source.t, source.signal, cfg.pcm, backend)
    tx_bits = _select_payload_bits(cfg, pcm.pcm_bits, backend=backend)
    bit_rate = _line_bit_rate(cfg, pcm.bits_per_sample)
    line = build_line_code(tx_bits, bit_rate, cfg.source.sim_fs, cfg.line, backend)
    channel = add_awgn_noise(line.waveform, cfg.channel, backend)
    matched = apply_matched_filter(channel.noisy_waveform, line.pulse, normalize=cfg.receiver.normalize_matched_filter, backend=backend)
    receiver = detect_bits(
        matched_waveform=matched,
        original_bits=tx_bits,
        encoding=cfg.line.encoding,
        samples_per_bit=line.samples_per_bit,
        timing_offset=line.timing_offset if cfg.receiver.timing_mode == "auto" or cfg.receiver.manual_timing_offset is None else int(cfg.receiver.manual_timing_offset),
        cfg=cfg.receiver,
        backend=backend,
    )
    return EndToEndResult(source=source, pcm=pcm, line=line, channel=channel, receiver=receiver)


def run_ber_sweep(cfg: AppConfig) -> SweepResult:
    backend = resolve_backend(cfg.compute_backend)
    start = cfg.ber.snr_start_db
    stop = cfg.ber.snr_stop_db
    step = abs(cfg.ber.snr_step_db)
    if step < 1e-6:
        step = 1.0
    if start <= stop:
        snrs = np.arange(start, stop + 0.5 * step, step)
    else:
        snrs = np.arange(start, stop - 0.5 * step, -step)

    ber = np.zeros_like(snrs, dtype=float)
    errors = np.zeros_like(snrs, dtype=int)

    rng = backend.xp.random.default_rng(cfg.ber.seed)
    tx_bits = rng.integers(0, 2, size=int(cfg.ber.num_bits), dtype=int)
    quantizer_bits, _ = resolve_quantizer_spec(cfg.pcm)
    bit_rate = _line_bit_rate(cfg, quantizer_bits)
    line = build_line_code(tx_bits, bit_rate, cfg.source.sim_fs, cfg.line, backend)

    for i, snr_db in enumerate(snrs):
        trial_errors = 0
        trial_bits = 0
        for trial in range(max(1, int(cfg.ber.trials))):
            trial_channel_cfg = dc_replace(cfg.channel, snr_db=float(snr_db), seed=cfg.ber.seed + trial)
            channel = add_awgn_noise(line.waveform, trial_channel_cfg, backend)
            matched = apply_matched_filter(channel.noisy_waveform, line.pulse, normalize=cfg.receiver.normalize_matched_filter, backend=backend)
            receiver = detect_bits(
                matched_waveform=matched,
                original_bits=tx_bits,
                encoding=cfg.line.encoding,
                samples_per_bit=line.samples_per_bit,
                timing_offset=line.timing_offset,
                cfg=cfg.receiver,
                backend=backend,
            )
            trial_errors += receiver.errors
            trial_bits += len(receiver.detected_bits)
        ber[i] = trial_errors / max(trial_bits, 1)
        errors[i] = trial_errors

    return SweepResult(snr_db=snrs, ber=ber, errors=errors, total_bits=len(tx_bits))
