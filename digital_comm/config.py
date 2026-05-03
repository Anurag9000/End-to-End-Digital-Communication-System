from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

SignalMode = Literal["single", "multi"]
WaveType = Literal["sin", "cos"]
EncodingType = Literal["unipolar", "polar", "bipolar"]
PulseShape = Literal["rect_nrz", "rect_rz", "sinc", "raised_cosine"]
PayloadMode = Literal["pcm_bits", "random_bits", "custom_bits"]
BitRateMode = Literal["auto_from_pcm", "manual"]
ThresholdMode = Literal["auto", "manual"]
TimingMode = Literal["auto", "manual"]
QuantRangeMode = Literal["auto", "manual"]
QuantizerMode = Literal["bits", "levels"]
BackendMode = Literal["auto", "numpy", "cupy"]


@dataclass
class ToneSpec:
    wave_type: WaveType = "sin"
    amplitude: float = 1.0
    frequency: float = 100.0
    phase: float = 0.0


@dataclass
class SourceConfig:
    mode: SignalMode = "single"
    duration: float = 0.05
    sim_fs: float = 100_000.0
    tones: list[ToneSpec] = field(default_factory=lambda: [ToneSpec()])


@dataclass
class PCMConfig:
    sample_fs: float = 5_000.0
    bits_per_sample: int = 4
    quantizer_mode: QuantizerMode = "bits"
    quantizer_levels: int = 16
    range_mode: QuantRangeMode = "auto"
    manual_min: float = -1.0
    manual_max: float = 1.0


@dataclass
class LineCodeConfig:
    payload_mode: PayloadMode = "pcm_bits"
    encoding: EncodingType = "polar"
    bit_rate_mode: BitRateMode = "auto_from_pcm"
    manual_bit_rate: float = 20_000.0
    pulse_shape: PulseShape = "rect_nrz"
    rc_alpha: float = 0.35
    pulse_span_symbols: int = 4
    rz_duty_cycle: float = 0.5
    normalize_pulse: bool = False
    custom_bits: str = ""


@dataclass
class ChannelConfig:
    snr_db: float = 10.0
    enable_awgn: bool = True
    seed: int = 7


@dataclass
class ReceiverConfig:
    threshold_mode: ThresholdMode = "auto"
    manual_threshold: float = 0.0
    timing_mode: TimingMode = "auto"
    manual_timing_offset: int | None = None
    normalize_matched_filter: bool = True


@dataclass
class BERConfig:
    snr_start_db: float = 0.0
    snr_stop_db: float = 12.0
    snr_step_db: float = 2.0
    num_bits: int = 100_000
    trials: int = 1
    seed: int = 7


@dataclass
class PlotConfig:
    preview_bits: int = 40
    eye_traces: int = 150
    eye_span_bits: int = 2
    show_frequency_domain: bool = True
    compare_sinc_rc: bool = True


@dataclass
class AppConfig:
    compute_backend: BackendMode = "auto"
    source: SourceConfig = field(default_factory=SourceConfig)
    pcm: PCMConfig = field(default_factory=PCMConfig)
    line: LineCodeConfig = field(default_factory=LineCodeConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    ber: BERConfig = field(default_factory=BERConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    def to_dict(self) -> dict:
        return asdict(self)
