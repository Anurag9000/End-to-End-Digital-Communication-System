from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SourceResult:
    t: np.ndarray
    signal: np.ndarray
    components: list[tuple[str, np.ndarray]]


@dataclass
class PCMResult:
    t_sampled: np.ndarray
    sampled_signal: np.ndarray
    quantized_signal: np.ndarray
    quantization_indices: np.ndarray
    pcm_bits: np.ndarray
    bits_per_sample: int
    levels: int
    quantizer_mode: str
    sample_fs: float
    delta: float
    v_min: float
    v_max: float
    sqnr_db: float
    codewords: list[str]


@dataclass
class LineCodeResult:
    bits: np.ndarray
    symbols: np.ndarray
    impulse_train: np.ndarray
    pulse: np.ndarray
    waveform: np.ndarray
    t: np.ndarray
    bit_rate: float
    samples_per_bit: int
    timing_offset: int
    pulse_name: str
    pulse_energy: float


@dataclass
class ChannelResult:
    tx_waveform: np.ndarray
    noisy_waveform: np.ndarray
    noise: np.ndarray
    signal_power: float
    noise_power: float
    snr_db: float
    achieved_snr_db: float


@dataclass
class ReceiverResult:
    matched_waveform: np.ndarray
    detected_bits: np.ndarray
    sampled_voltages: np.ndarray
    sample_indices: np.ndarray
    ber: float
    errors: int
    threshold: float
    timing_offset: int


@dataclass
class SweepResult:
    snr_db: np.ndarray
    ber: np.ndarray
    errors: np.ndarray
    total_bits: int


@dataclass
class QuantizerSweepResult:
    values: np.ndarray
    bits_per_sample: np.ndarray
    levels: np.ndarray
    sqnr_db: np.ndarray
    delta: np.ndarray
    quantizer_mode: str


@dataclass
class SamplingSweepResult:
    sample_fs: np.ndarray
    sqnr_db: np.ndarray
    sample_counts: np.ndarray
    nyquist_rate: float
    is_below_nyquist: np.ndarray


@dataclass
class EndToEndResult:
    source: SourceResult
    pcm: PCMResult
    line: LineCodeResult
    channel: ChannelResult
    receiver: ReceiverResult
