#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import math
import random
from typing import Iterable
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digital_comm.backend import cupy_available, resolve_backend, to_numpy
from digital_comm.channel import add_awgn_noise
from digital_comm.config import AppConfig, ChannelConfig, LineCodeConfig, PCMConfig, ReceiverConfig, SourceConfig, ToneSpec
from digital_comm.line_coding import build_line_code
from digital_comm.pcm import resolve_quantizer_spec, sample_and_quantize
from digital_comm.receiver import apply_matched_filter, detect_bits
from digital_comm.simulation import run_ber_sweep, run_end_to_end
from digital_comm.sweeps import run_quantizer_sweep, run_sampling_sweep
from digital_comm.source import generate_source_signal


def _make_source(mode: str, fs: float, rng: random.Random) -> SourceConfig:
    if mode == "single":
        tone = ToneSpec(
            wave_type=rng.choice(["sin", "cos"]),
            amplitude=1.0,
            frequency=100.0,
            phase=0.0,
        )
        return SourceConfig(mode="single", duration=0.02, sim_fs=fs, tones=[tone])

    tones = [
        ToneSpec(wave_type="sin", amplitude=1.0, frequency=80.0, phase=0.0),
        ToneSpec(wave_type="cos", amplitude=0.5, frequency=140.0, phase=0.4),
    ]
    return SourceConfig(mode="multi", duration=0.02, sim_fs=fs, tones=tones)


def _make_line(
    pulse_shape: str,
    encoding: str,
    payload_mode: str = "random_bits",
    custom_bits: str = "",
    manual_bit_rate: float = 20_000.0,
    rc_alpha: float = 0.35,
    pulse_span_symbols: int = 4,
    rz_duty_cycle: float = 0.5,
    normalize_pulse: bool = False,
) -> LineCodeConfig:
    return LineCodeConfig(
        payload_mode=payload_mode,
        encoding=encoding,
        bit_rate_mode="manual",
        manual_bit_rate=manual_bit_rate,
        pulse_shape=pulse_shape,
        rc_alpha=rc_alpha,
        pulse_span_symbols=pulse_span_symbols,
        rz_duty_cycle=rz_duty_cycle,
        normalize_pulse=normalize_pulse,
        custom_bits=custom_bits,
    )


def _manual_threshold(encoding: str) -> float:
    return 0.0 if encoding == "polar" else 0.5


def _expected_pulse_len(cfg: AppConfig) -> int:
    spb = max(2, int(round(cfg.source.sim_fs / cfg.line.manual_bit_rate)))
    if cfg.line.pulse_shape in {"sinc", "raised_cosine"}:
        return 2 * cfg.line.pulse_span_symbols * spb + 1
    return spb


def _expected_timing_offset(cfg: AppConfig) -> int:
    return _expected_pulse_len(cfg) - 1


def _base_case(
    backend_mode: str,
    source_mode: str,
    encoding: str,
    pulse_shape: str,
    *,
    noise: bool,
    seed: int,
    source_fs: float = 100_000.0,
    sample_fs: float = 5_000.0,
    bits_per_sample: int = 4,
    quantizer_mode: str = "bits",
    quantizer_levels: int = 16,
    range_mode: str = "auto",
    custom_bits: str = "",
    rc_alpha: float = 0.35,
    pulse_span_symbols: int = 4,
    rz_duty_cycle: float = 0.5,
    manual_receiver: bool = False,
    snr_db: float = 10.0,
) -> AppConfig:
    cfg = AppConfig(
        compute_backend=backend_mode,
        source=_make_source(source_mode, source_fs, random.Random(seed)),
        pcm=PCMConfig(
            sample_fs=sample_fs,
            bits_per_sample=bits_per_sample,
            quantizer_mode=quantizer_mode,
            quantizer_levels=quantizer_levels,
            range_mode=range_mode,
            manual_min=-2.0,
            manual_max=2.0,
        ),
        line=_make_line(
            pulse_shape=pulse_shape,
            encoding=encoding,
            payload_mode="custom_bits" if custom_bits else "random_bits",
            custom_bits=custom_bits,
            rc_alpha=rc_alpha,
            pulse_span_symbols=pulse_span_symbols,
            rz_duty_cycle=rz_duty_cycle,
        ),
        channel=ChannelConfig(snr_db=snr_db, enable_awgn=noise, seed=seed),
        receiver=ReceiverConfig(
            threshold_mode="manual" if manual_receiver else "auto",
            manual_threshold=_manual_threshold(encoding),
            timing_mode="manual" if manual_receiver else "auto",
            manual_timing_offset=_expected_timing_offset(
                AppConfig(
                    source=_make_source(source_mode, source_fs, random.Random(seed)),
                    line=_make_line(
                        pulse_shape=pulse_shape,
                        encoding=encoding,
                        payload_mode="random_bits",
                        rc_alpha=rc_alpha,
                        pulse_span_symbols=pulse_span_symbols,
                        rz_duty_cycle=rz_duty_cycle,
                    ),
                )
            )
            if manual_receiver
            else None,
            normalize_matched_filter=True,
        ),
        ber=replace(AppConfig().ber, seed=seed, num_bits=1024, trials=1),
        plot=AppConfig().plot,
    )

    return cfg


def _assert_close(actual: float, expected: float, *, rel: float = 1e-9, abs_tol: float = 1e-12, label: str = ""):
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(f"{label} expected {expected}, got {actual}")


def validate_case(case_name: str, cfg: AppConfig):
    backend = resolve_backend(cfg.compute_backend)
    source = generate_source_signal(cfg.source, backend)
    pcm = sample_and_quantize(source.t, source.signal, cfg.pcm, backend)
    quant_bits, quant_levels = resolve_quantizer_spec(cfg.pcm)
    spb = max(2, int(round(cfg.source.sim_fs / cfg.line.manual_bit_rate)))
    guard_len = max(2 * cfg.line.pulse_span_symbols * spb, spb)
    guard_bits = np.zeros(guard_len, dtype=int)
    if cfg.line.payload_mode == "pcm_bits":
        payload_bits = to_numpy(pcm.pcm_bits).astype(int)
    elif cfg.line.payload_mode == "custom_bits":
        payload_bits = np.array([int(ch) for ch in cfg.line.custom_bits if ch in "01"], dtype=int)
    else:
        payload_bits = to_numpy(
            resolve_backend("numpy").xp.random.default_rng(cfg.ber.seed).integers(0, 2, size=int(cfg.ber.num_bits), dtype=int)
        ).astype(int)
    line_bits = np.concatenate([guard_bits, payload_bits, guard_bits])
    line = build_line_code(line_bits, cfg.line.manual_bit_rate if cfg.line.bit_rate_mode == "manual" else cfg.pcm.sample_fs * max(1, quant_bits), cfg.source.sim_fs, cfg.line, backend)
    channel = add_awgn_noise(line.waveform, cfg.channel, backend)
    matched = apply_matched_filter(channel.noisy_waveform, line.pulse, normalize=cfg.receiver.normalize_matched_filter, backend=backend)
    receiver = detect_bits(
        matched_waveform=matched,
        original_bits=line.bits,
        encoding=cfg.line.encoding,
        samples_per_bit=line.samples_per_bit,
        timing_offset=cfg.receiver.manual_timing_offset if cfg.receiver.timing_mode == "manual" and cfg.receiver.manual_timing_offset is not None else line.timing_offset,
        cfg=cfg.receiver,
        backend=backend,
    )

    t = to_numpy(source.t)
    signal = to_numpy(source.signal)
    sample_t = to_numpy(pcm.t_sampled)
    sampled = to_numpy(pcm.sampled_signal)
    quantized = to_numpy(pcm.quantized_signal)
    indices = to_numpy(pcm.quantization_indices)
    waveform = to_numpy(line.waveform)
    pulse = to_numpy(line.pulse)
    noisy = to_numpy(channel.noisy_waveform)
    matched_np = to_numpy(matched)
    detected = to_numpy(receiver.detected_bits)
    sample_idx = to_numpy(receiver.sample_indices)

    if len(t) == 0 or len(signal) == 0:
        raise AssertionError(f"{case_name}: empty source output")
    if t.shape != signal.shape:
        raise AssertionError(f"{case_name}: source t/signal shape mismatch")
    if len(sample_t) != len(sampled) or len(sample_t) != len(quantized):
        raise AssertionError(f"{case_name}: PCM sample arrays are not aligned")
    if not np.all(indices >= 0):
        raise AssertionError(f"{case_name}: PCM indices below zero")
    if not np.all(indices < quant_levels):
        raise AssertionError(f"{case_name}: PCM indices exceed quantizer levels")
    if len(pcm.pcm_bits) != len(sample_t) * quant_bits:
        raise AssertionError(f"{case_name}: PCM bit length mismatch")
    if pcm.bits_per_sample != quant_bits:
        raise AssertionError(f"{case_name}: effective quantizer bit width mismatch")
    if pcm.levels != quant_levels:
        raise AssertionError(f"{case_name}: effective quantizer levels mismatch")
    if len(pcm.codewords) != len(sample_t):
        raise AssertionError(f"{case_name}: PCM codeword count mismatch")
    if len(waveform) != len(line.impulse_train) + len(pulse) - 1:
        raise AssertionError(f"{case_name}: line waveform length mismatch")
    if len(line.t) != len(waveform):
        raise AssertionError(f"{case_name}: time vector and waveform length mismatch")
    if len(noisy) != len(waveform):
        raise AssertionError(f"{case_name}: channel waveform length mismatch")
    if len(matched_np) != len(noisy) + len(pulse) - 1:
        raise AssertionError(f"{case_name}: matched-filter length mismatch")
    if len(sample_idx) != len(detected):
        raise AssertionError(f"{case_name}: receiver sample/detected length mismatch")
    if not (0.0 <= receiver.ber <= 1.0):
        raise AssertionError(f"{case_name}: BER outside [0, 1]")

    if cfg.channel.enable_awgn:
        expected_noise = channel.signal_power / (10 ** (cfg.channel.snr_db / 10.0))
        _assert_close(channel.noise_power, expected_noise, label=f"{case_name} noise power")
        if not np.isfinite(channel.achieved_snr_db):
            raise AssertionError(f"{case_name}: achieved SNR is not finite")
    else:
        if channel.noise_power != 0.0:
            raise AssertionError(f"{case_name}: noise power should be zero when AWGN is disabled")
        if not np.allclose(noisy, waveform):
            raise AssertionError(f"{case_name}: noisy waveform must equal tx waveform when AWGN is disabled")
        if cfg.line.pulse_shape == "sinc" and cfg.line.encoding == "bipolar":
            if receiver.ber > 0.05:
                raise AssertionError(f"{case_name}: noise-free truncated sinc/bipolar case should stay below 5% BER")
        elif receiver.errors != 0 or receiver.ber != 0.0:
            raise AssertionError(f"{case_name}: noise-free case should recover all bits exactly")

    if cfg.receiver.timing_mode == "manual" and cfg.receiver.manual_timing_offset is not None:
        expected_offset = cfg.receiver.manual_timing_offset
        if expected_offset != _expected_timing_offset(cfg):
            raise AssertionError(f"{case_name}: manual timing offset does not match expected pulse delay")

    if cfg.line.payload_mode == "custom_bits":
        custom_bits = np.array([int(ch) for ch in cfg.line.custom_bits if ch in "01"], dtype=int)
        start = guard_len
        stop = guard_len + len(custom_bits)
        if len(custom_bits) and len(detected) >= stop:
            if not np.array_equal(detected[start:stop], custom_bits):
                raise AssertionError(f"{case_name}: custom bit payload was not recovered exactly")

    print(
        f"[PASS] {case_name} | backend={backend.name} | pulse={cfg.line.pulse_shape} | "
        f"encoding={cfg.line.encoding} | noise={'on' if cfg.channel.enable_awgn else 'off'} | BER={receiver.ber:.6f}"
    )


def build_cases(seed: int) -> list[tuple[str, AppConfig]]:
    rng = random.Random(seed)
    backends = ["numpy", "auto"]
    if cupy_available():
        backends.append("cupy")

    cases: list[tuple[str, AppConfig]] = []

    for backend_mode, source_mode, encoding, pulse_shape, noise in product(
        backends,
        ["single", "multi"],
        ["unipolar", "polar", "bipolar"],
        ["rect_nrz", "rect_rz", "sinc", "raised_cosine"],
        [False, True],
    ):
        manual_receiver = noise is False
        cfg = _base_case(
            backend_mode,
            source_mode,
            encoding,
            pulse_shape,
            noise=noise,
            seed=rng.randint(1, 10_000_000),
            manual_receiver=manual_receiver,
            snr_db=-5.0 if noise else 20.0,
        )
        cases.append((f"grid::{backend_mode}::{source_mode}::{encoding}::{pulse_shape}::noise={int(noise)}", cfg))

    edge_cases = [
        _base_case("numpy", "single", "polar", "rect_nrz", noise=False, seed=101, custom_bits="00000000", manual_receiver=True),
        _base_case("numpy", "single", "unipolar", "rect_rz", noise=False, seed=102, custom_bits="11111111", manual_receiver=True),
        _base_case("numpy", "multi", "bipolar", "sinc", noise=False, seed=103, custom_bits="01010101", pulse_span_symbols=1, manual_receiver=True),
        _base_case("numpy", "single", "polar", "raised_cosine", noise=False, seed=104, rc_alpha=0.0, pulse_span_symbols=1, manual_receiver=True),
        _base_case("numpy", "single", "polar", "raised_cosine", noise=True, seed=105, rc_alpha=1.0, pulse_span_symbols=8, snr_db=0.0),
        _base_case("numpy", "multi", "polar", "rect_nrz", noise=True, seed=106, range_mode="manual", source_fs=50_000.0, sample_fs=2_500.0, bits_per_sample=2, snr_db=30.0),
        _base_case("numpy", "single", "polar", "rect_nrz", noise=False, seed=107, quantizer_mode="levels", quantizer_levels=32, sample_fs=4_000.0, manual_receiver=True),
    ]
    for idx, cfg in enumerate(edge_cases, start=1):
        cases.append((f"edge::{idx}", cfg))

    random_cases = []
    for idx in range(6):
        backend_mode = rng.choice(backends)
        source_mode = rng.choice(["single", "multi"])
        encoding = rng.choice(["unipolar", "polar", "bipolar"])
        pulse_shape = rng.choice(["rect_nrz", "rect_rz", "sinc", "raised_cosine"])
        noise = bool(rng.randint(0, 1))
        cfg = _base_case(
            backend_mode,
            source_mode,
            encoding,
            pulse_shape,
            noise=noise,
            seed=rng.randint(1, 10_000_000),
            rc_alpha=rng.choice([0.0, 0.35, 1.0]),
            pulse_span_symbols=rng.choice([1, 4, 8]),
            rz_duty_cycle=rng.choice([0.25, 0.5, 1.0]),
            range_mode=rng.choice(["auto", "manual"]),
            snr_db=rng.choice([-5.0, 0.0, 10.0, 20.0]),
            manual_receiver=not noise,
        )
        random_cases.append((f"random::{idx}", cfg))

    cases.extend(random_cases)
    return cases


def verify_sweep(seed: int):
    cfg = AppConfig(
        compute_backend="auto",
        source=SourceConfig(mode="single", duration=0.02, sim_fs=100_000.0, tones=[ToneSpec(wave_type="sin", amplitude=1.0, frequency=100.0, phase=0.0)]),
        pcm=PCMConfig(sample_fs=5_000.0, bits_per_sample=4, range_mode="auto", manual_min=-1.0, manual_max=1.0),
        line=LineCodeConfig(
            payload_mode="random_bits",
            encoding="polar",
            bit_rate_mode="manual",
            manual_bit_rate=20_000.0,
            pulse_shape="rect_nrz",
        ),
        channel=ChannelConfig(snr_db=10.0, enable_awgn=True, seed=seed),
        receiver=ReceiverConfig(threshold_mode="auto", timing_mode="auto", normalize_matched_filter=True),
        ber=replace(AppConfig().ber, snr_start_db=0.0, snr_stop_db=12.0, snr_step_db=2.0, num_bits=20_000, trials=1, seed=seed),
        plot=AppConfig().plot,
    )
    sweep = run_ber_sweep(cfg)
    snr = to_numpy(sweep.snr_db)
    ber = to_numpy(sweep.ber)
    quant_sweep = run_quantizer_sweep(cfg.source, cfg.pcm, backend=resolve_backend("numpy"), sweep_axis="bits", sweep_values=[2, 4, 6])
    samp_sweep = run_sampling_sweep(cfg.source, cfg.pcm, backend=resolve_backend("numpy"), sample_rates=[500.0, 1_000.0, 2_000.0])
    if len(snr) != 7:
        raise AssertionError("BER sweep should produce 7 SNR points for 0..12 dB in 2 dB steps")
    if not np.all(ber >= 0):
        raise AssertionError("BER sweep must not produce negative BER")
    if ber[0] < ber[-1]:
        raise AssertionError("BER sweep did not decrease from low SNR to high SNR")
    if len(quant_sweep.values) != 3 or len(samp_sweep.sample_fs) != 3:
        raise AssertionError("Runtime sweeps should return the requested number of sweep points")
    print(f"[PASS] sweep | points={len(snr)} | ber_start={ber[0]:.6f} | ber_end={ber[-1]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Verify a parameter x stage grid across the digital communication pipeline.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random case generation.")
    args = parser.parse_args()

    cases = build_cases(args.seed)
    failures: list[str] = []
    print(f"Running {len(cases)} stage-grid cases...")
    for case_name, cfg in cases:
        try:
            validate_case(case_name, cfg)
        except Exception as exc:  # pragma: no cover - surfaced in CLI
            failures.append(f"{case_name}: {exc}")
            print(f"[FAIL] {case_name}: {exc}")

    try:
        verify_sweep(args.seed)
    except Exception as exc:  # pragma: no cover - surfaced in CLI
        failures.append(f"sweep: {exc}")
        print(f"[FAIL] sweep: {exc}")

    if failures:
        print("\nVerification failed.")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nAll parameter-grid checks passed.")


if __name__ == "__main__":
    main()
