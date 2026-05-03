"""Digital communication system package."""

from .backend import resolve_backend, to_numpy
from .config import (
    AppConfig,
    BERConfig,
    ChannelConfig,
    LineCodeConfig,
    PCMConfig,
    PlotConfig,
    ReceiverConfig,
    SourceConfig,
    ToneSpec,
)
from .simulation import run_ber_sweep, run_end_to_end
from .sweeps import run_quantizer_sweep, run_sampling_sweep
