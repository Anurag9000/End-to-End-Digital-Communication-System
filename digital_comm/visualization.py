from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .backend import to_numpy


def _setup_axes(ax, title, xlabel="Time (s)", ylabel="Amplitude"):
    ax.set_title(title, fontweight="600", fontsize=12, pad=15)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="500")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="500")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)


def _plot_eye_segments(ax, waveform: np.ndarray, samples_per_bit: int, fs: float, offset: int, traces: int, title: str, span_bits: int):
    waveform = to_numpy(waveform)
    span_bits = max(2, int(span_bits))
    chunk = span_bits * samples_per_bit
    if samples_per_bit <= 0 or len(waveform) < chunk:
        ax.set_title(title, fontweight="600")
        ax.text(0.5, 0.5, "Not enough samples for eye diagram", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    ideal_start = offset - (span_bits // 2) * samples_per_bit
    if ideal_start < 0:
        skip = (-ideal_start + samples_per_bit - 1) // samples_per_bit
        start = ideal_start + skip * samples_per_bit
    else:
        start = ideal_start

    # X-axis: normalized symbol periods so eye is always centered at 0
    # regardless of bit rate. Range = [-span_bits/2, +span_bits/2] T.
    t_eye = np.linspace(-span_bits / 2.0, span_bits / 2.0, chunk)

    count = 0
    alpha_val = max(0.03, min(0.20, 15.0 / max(traces, 1)))
    for i in range(start, len(waveform) - chunk, samples_per_bit):
        ax.plot(t_eye, waveform[i : i + chunk], color="#f43f5e", alpha=alpha_val, linewidth=1.2)
        count += 1
        if count >= traces:
            break

    # Mark optimal sampling instant
    ax.axvline(0, color="#1e293b", linewidth=1.0, linestyle="--", alpha=0.5, label="Sampling instant")
    _setup_axes(ax, title, "Time (symbol periods T)", "Amplitude")


def plot_eye_diagram(waveform: np.ndarray, samples_per_bit: int, fs: float, offset: int, title: str, traces: int = 150, span_bits: int = 2):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _plot_eye_segments(ax, waveform, samples_per_bit, fs, offset, traces, title, span_bits)
    fig.tight_layout()
    return fig


def plot_time_series(t: np.ndarray, y: np.ndarray, title: str, xlabel: str = "Time (s)", ylabel: str = "Amplitude", max_points: int = 5000):
    t = to_numpy(t)
    y = to_numpy(y)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    n = min(len(t), max_points)
    ax.plot(t[:n], y[:n], color="#0ea5e9", linewidth=2.0)
    ax.fill_between(t[:n], y[:n], 0, color="#0ea5e9", alpha=0.1)
    
    _setup_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def plot_two_series(t: np.ndarray, y1: np.ndarray, y2: np.ndarray, label1: str, label2: str, title: str, max_points: int = 5000):
    t = to_numpy(t)
    y1 = to_numpy(y1)
    y2 = to_numpy(y2)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    n = min(len(t), len(y1), len(y2), max_points)
    ax.plot(t[:n], y1[:n], label=label1, color="#8b5cf6", linewidth=2.0, alpha=0.9)
    ax.plot(t[:n], y2[:n], label=label2, color="#10b981", linewidth=2.0, alpha=0.8)
    
    _setup_axes(ax, title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_frequency_response(x: np.ndarray, fs: float, title: str):
    x = to_numpy(x)
    n = len(x)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    if n == 0:
        ax.set_title(title, fontweight="600")
        ax.text(0.5, 0.5, "Empty signal", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs))
    mag = np.abs(np.fft.fftshift(np.fft.fft(x)))
    mag = mag / max(np.max(mag), 1e-12)

    ax.plot(freqs, mag, color="#d946ef", linewidth=2.0)
    ax.fill_between(freqs, mag, 0, color="#d946ef", alpha=0.15)
    
    _setup_axes(ax, title, "Frequency (Hz)", "Normalized magnitude")
    fig.tight_layout()
    return fig


def plot_ber_curve(snr_db: np.ndarray, ber: np.ndarray):
    snr_db = to_numpy(snr_db)
    ber = to_numpy(ber)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    ax.semilogy(snr_db, np.maximum(ber, 1e-12), marker="o", markersize=8, color="#10b981", linewidth=2.5)
    _setup_axes(ax, "BER vs SNR", "SNR (dB)", "Bit Error Rate")
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_sweep(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, *, xscale: str = "linear", yscale: str = "linear", marker: str = "o"):
    x = to_numpy(x)
    y = to_numpy(y)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    ax.plot(x, y, marker=marker, markersize=8, color="#0ea5e9", linewidth=2.5)
    _setup_axes(ax, title, xlabel, ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sampling_sweep(sample_fs: np.ndarray, sqnr_db: np.ndarray, nyquist_rate: float):
    sample_fs = to_numpy(sample_fs)
    sqnr_db = to_numpy(sqnr_db)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    
    ax.plot(sample_fs, sqnr_db, marker="o", markersize=8, color="#f59e0b", linewidth=2.5)
    if nyquist_rate > 0:
        ax.axvline(nyquist_rate, linestyle="--", color="#ef4444", linewidth=2.0, alpha=0.8, label=f"Nyquist = {nyquist_rate:.2f} Hz")
        ax.legend(frameon=False)
        
    _setup_axes(ax, "SQNR vs Sampling Rate", "Sampling rate Fs (Hz)", "SQNR (dB)")
    fig.tight_layout()
    return fig
