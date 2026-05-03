from __future__ import annotations

from dataclasses import asdict

import streamlit as st
import numpy as np

from digital_comm.backend import resolve_backend, to_numpy
from digital_comm.config import AppConfig, BERConfig, ChannelConfig, LineCodeConfig, PCMConfig, PlotConfig, ReceiverConfig, SourceConfig, ToneSpec
from digital_comm.line_coding import build_line_code
from digital_comm.simulation import run_ber_sweep, run_end_to_end
from digital_comm.source import nyquist_rate
from digital_comm.sweeps import run_quantizer_sweep, run_sampling_sweep
from digital_comm.visualization import plot_ber_curve, plot_eye_diagram, plot_frequency_response, plot_metric_sweep, plot_sampling_sweep, plot_time_series, plot_two_series


st.set_page_config(page_title="Digital Communication System", page_icon="📡", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Modern Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            box-shadow: 0 4px 14px 0 rgba(14, 165, 233, 0.25);
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(14, 165, 233, 0.35);
        }
        
        /* Metric Value Text Gradient */
        [data-testid="stMetricValue"] {
            background: -webkit-linear-gradient(45deg, #0ea5e9, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2rem;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 600;
            color: #71717a;
        }
        .stTabs [aria-selected="true"] {
            color: #0ea5e9 !important;
        }
        
        /* App Title */
        h1 {
            font-weight: 800;
            font-size: 2.75rem;
            background: -webkit-linear-gradient(45deg, #0ea5e9, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: -1rem;
        }
        </style>
    """, unsafe_allow_html=True)


def _tone_widget(prefix: str, idx: int, default: ToneSpec) -> ToneSpec:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wave_type = st.selectbox(f"{prefix} Wave", ["sin", "cos"], index=0 if default.wave_type == "sin" else 1, key=f"{prefix}_{idx}_wave")
    with c2:
        amplitude = st.number_input(f"{prefix} Amp", value=float(default.amplitude), step=0.1, key=f"{prefix}_{idx}_amp")
    with c3:
        frequency = st.number_input(f"{prefix} Freq (Hz)", value=float(default.frequency), step=1.0, key=f"{prefix}_{idx}_freq")
    with c4:
        phase = st.number_input(f"{prefix} Phase (rad)", value=float(default.phase), step=0.1, key=f"{prefix}_{idx}_phase")
    return ToneSpec(wave_type=wave_type, amplitude=float(amplitude), frequency=float(frequency), phase=float(phase))


def build_config_from_sidebar() -> AppConfig:
    st.sidebar.header("Simulation Controls")
    st.sidebar.caption("Click Generate to run the entire chain.")

    with st.sidebar.expander("0) Compute Backend", expanded=True):
        compute_backend = st.selectbox("Array backend", ["auto", "numpy", "cupy"], index=0)

    with st.sidebar.expander("1) Source Signal", expanded=True):
        mode = st.selectbox("Signal mode", ["single", "multi"], index=0)
        duration = st.number_input("Duration (s)", min_value=0.001, value=0.05, step=0.01)
        sim_fs = st.number_input("Simulation sample rate (Hz)", min_value=1_000.0, value=100_000.0, step=1_000.0)
        tone_count = 1 if mode == "single" else st.slider("Number of tones", min_value=1, max_value=5, value=2)
        tones = []
        for i in range(tone_count):
            tones.append(_tone_widget("Tone", i, ToneSpec(amplitude=1.0, frequency=100.0 * (i + 1))))
        source_nyq = 2.0 * max((tone.frequency for tone in tones), default=0.0)
        if source_nyq > 0:
            st.caption(f"Nyquist suggestion: Fs > {source_nyq:.2f} Hz")

    with st.sidebar.expander("2) PCM Sampling & Quantization", expanded=True):
        sample_fs = st.number_input("Sampling frequency Fs (Hz)", min_value=100.0, value=5_000.0, step=100.0)
        quantizer_mode = st.radio("Quantizer control", ["bits", "levels"], index=0, horizontal=True)
        if quantizer_mode == "bits":
            bits_per_sample = st.slider("Bits per sample (n)", min_value=1, max_value=10, value=4)
            quantizer_levels = 2**int(bits_per_sample)
        else:
            quantizer_levels = st.selectbox("Number of levels (L)", [2**i for i in range(1, 11)], index=3)
            bits_per_sample = int(np.log2(int(quantizer_levels)))
        range_mode = st.selectbox("Quantizer range", ["auto", "manual"], index=0)
        manual_min = st.number_input("Bottom voltage", value=-1.0, step=0.1)
        manual_max = st.number_input("Top voltage", value=1.0, step=0.1)
        if source_nyq > 0:
            if float(sample_fs) < source_nyq:
                st.warning("Fs is below Nyquist. Aliasing will appear in the sampled signal.")
            else:
                st.success("Fs is at or above Nyquist for the current source tones.")

    with st.sidebar.expander("3) Payload / Line Coding", expanded=True):
        payload_mode = st.selectbox("Payload source", ["pcm_bits", "random_bits", "custom_bits"], index=0)
        encoding = st.selectbox("Encoding", ["polar", "unipolar", "bipolar"], index=0)
        bit_rate_mode = st.selectbox("Bit rate mode", ["auto_from_pcm", "manual"], index=0)
        manual_bit_rate = st.number_input("Manual bit rate (bps)", min_value=1.0, value=20_000.0, step=100.0)
        pulse_shape = st.selectbox("Pulse shape", ["rect_nrz", "rect_rz", "sinc", "raised_cosine"], index=0)
        rc_alpha = st.slider("Raised cosine alpha", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        pulse_span_symbols = st.slider("Pulse span (symbols)", min_value=1, max_value=8, value=4)
        rz_duty_cycle = st.slider("RZ duty cycle", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        normalize_pulse = st.checkbox("Normalize pulse energy", value=False)
        custom_bits = st.text_area("Custom bitstring", value="10110011", height=70)
        if bit_rate_mode == "manual":
            max_achievable = float(sim_fs) / 2.0
            if float(manual_bit_rate) > max_achievable:
                st.warning(
                    f"Manual bit rate ({manual_bit_rate:,.0f} bps) exceeds sim Fs/2 "
                    f"({max_achievable:,.0f} bps). Engine will clamp samples/bit to 2 "
                    f"→ effective rate = {max_achievable:,.0f} bps."
                )

    with st.sidebar.expander("4) Channel", expanded=True):
        snr_db = st.slider("SNR (dB)", min_value=-10.0, max_value=30.0, value=10.0, step=1.0)
        enable_awgn = st.checkbox("Enable AWGN", value=True)
        seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    with st.sidebar.expander("5) Receiver", expanded=True):
        threshold_mode = st.selectbox("Threshold mode", ["auto", "manual"], index=0)
        manual_threshold = st.number_input("Manual threshold", value=0.0, step=0.1)
        timing_mode = st.selectbox("Timing mode", ["auto", "manual"], index=0)
        manual_timing_offset = st.number_input("Manual timing offset (samples)", value=0, step=1)
        normalize_matched_filter = st.checkbox("Normalize matched-filter output", value=True)

    with st.sidebar.expander("6) BER Sweep", expanded=False):
        snr_start_db = st.number_input("Sweep start (dB)", value=0.0, step=1.0)
        snr_stop_db = st.number_input("Sweep stop (dB)", value=12.0, step=1.0)
        snr_step_db = st.number_input("Sweep step (dB)", min_value=0.1, value=2.0, step=1.0)
        ber_bits = st.number_input("Bits per SNR point", min_value=1000, value=100_000, step=1000)
        trials = st.number_input("Trials per SNR point", min_value=1, value=1, step=1)

    with st.sidebar.expander("7) Plotting", expanded=False):
        preview_bits = st.slider("Preview bits", min_value=10, max_value=200, value=40)
        eye_traces = st.slider("Eye traces", min_value=20, max_value=500, value=150)
        eye_span_bits = st.slider("Eye span (bits)", min_value=2, max_value=4, value=2)
        show_frequency_domain = st.checkbox("Show frequency-domain plots", value=True)
        compare_sinc_rc = st.checkbox("Compare RC vs sinc when RC is selected", value=True)

    source = SourceConfig(mode=mode, duration=float(duration), sim_fs=float(sim_fs), tones=tones)
    pcm = PCMConfig(
        sample_fs=float(sample_fs),
        bits_per_sample=int(bits_per_sample),
        quantizer_mode=quantizer_mode,
        quantizer_levels=int(quantizer_levels),
        range_mode=range_mode,
        manual_min=float(manual_min),
        manual_max=float(manual_max),
    )
    line = LineCodeConfig(
        payload_mode=payload_mode,
        encoding=encoding,
        bit_rate_mode=bit_rate_mode,
        manual_bit_rate=float(manual_bit_rate),
        pulse_shape=pulse_shape,
        rc_alpha=float(rc_alpha),
        pulse_span_symbols=int(pulse_span_symbols),
        rz_duty_cycle=float(rz_duty_cycle),
        normalize_pulse=bool(normalize_pulse),
        custom_bits=str(custom_bits),
    )
    channel = ChannelConfig(snr_db=float(snr_db), enable_awgn=bool(enable_awgn), seed=int(seed))
    receiver = ReceiverConfig(
        threshold_mode=threshold_mode,
        manual_threshold=float(manual_threshold),
        timing_mode=timing_mode,
        manual_timing_offset=int(manual_timing_offset) if timing_mode == "manual" else None,
        normalize_matched_filter=bool(normalize_matched_filter),
    )
    ber = BERConfig(snr_start_db=float(snr_start_db), snr_stop_db=float(snr_stop_db), snr_step_db=float(snr_step_db), num_bits=int(ber_bits), trials=int(trials), seed=int(seed))
    plot = PlotConfig(preview_bits=int(preview_bits), eye_traces=int(eye_traces), eye_span_bits=int(eye_span_bits), show_frequency_domain=bool(show_frequency_domain), compare_sinc_rc=bool(compare_sinc_rc))
    return AppConfig(compute_backend=compute_backend, source=source, pcm=pcm, line=line, channel=channel, receiver=receiver, ber=ber, plot=plot)


def _fmt_db(v: float) -> str:
    """Format a dB value that may be ±inf or NaN for display."""
    import math
    if math.isnan(v):
        return "N/A"
    if v == float("inf"):
        return "+∞ dB"
    if v == float("-inf"):
        return "-∞ dB"
    return f"{v:.2f} dB"


def _fmt_ber(v: float) -> str:
    """Format a BER value that may be NaN."""
    import math
    if math.isnan(v):
        return "N/A"
    return f"{v:.6f}"


def show_config_summary(cfg: AppConfig):
    with st.expander("Current configuration", expanded=False):
        st.json(cfg.to_dict())


def show_source_tab(result):
    st.subheader("1) Continuous-Time Source")
    st.write(f"Samples: {len(result.source.t)}")
    st.write(f"Signal length: {len(result.source.signal)}")
    if result.source.components:
        st.write("Source components:")
        st.dataframe(
            {
                "component": [name for name, _ in result.source.components],
                "peak_amplitude": [float(np.max(np.abs(to_numpy(comp)))) if len(comp) > 0 else 0.0 for _, comp in result.source.components],
            },
            width="stretch",
        )
    st.pyplot(plot_time_series(result.source.t, result.source.signal, "Source Signal"))


def show_pcm_tab(result):
    st.subheader("2) Sampling and Quantization")
    pcm = result.pcm
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SQNR", _fmt_db(pcm.sqnr_db))
    c2.metric("Delta", f"{pcm.delta:.4f}")
    c3.metric("Samples", len(pcm.t_sampled))
    c4.metric("PCM bits", len(pcm.pcm_bits))
    c5, c6, c7 = st.columns(3)
    c5.metric("Quantizer mode", pcm.quantizer_mode)
    c6.metric("Bits per sample", pcm.bits_per_sample)
    c7.metric("Levels", pcm.levels)
    st.pyplot(plot_two_series(pcm.t_sampled, pcm.sampled_signal, pcm.quantized_signal, "Sampled", "Quantized", "Sampling and Quantization"))
    st.write("First codewords")
    preview = min(16, len(pcm.codewords))
    st.code(" ".join(pcm.codewords[:preview]) if preview else "")
    st.dataframe(
        {
            "t_sampled": to_numpy(pcm.t_sampled[:preview]),
            "sampled": to_numpy(pcm.sampled_signal[:preview]),
            "quantized": to_numpy(pcm.quantized_signal[:preview]),
            "index": to_numpy(pcm.quantization_indices[:preview]),
            "codeword": pcm.codewords[:preview],
        },
        width="stretch",
    )


def show_quantizer_sweep_tab(cfg: AppConfig, source_result):
    st.subheader("Quantizer SQNR Sweep")
    source_nyq = nyquist_rate(cfg.source)
    sweep_axis = st.radio("Sweep axis", ["bits", "levels"], horizontal=True, key="quantizer_sweep_axis")
    c1, c2, c3 = st.columns(3)
    if sweep_axis == "bits":
        start = int(c1.number_input("Start bits", min_value=1, max_value=12, value=2, step=1, key="quant_bits_start"))
        stop = int(c2.number_input("Stop bits", min_value=start, max_value=12, value=8, step=1, key="quant_bits_stop"))
        step = int(c3.number_input("Step", min_value=1, max_value=4, value=1, step=1, key="quant_bits_step"))
        values = list(range(start, stop + 1, step))
    else:
        start_pow = int(c1.number_input("Start exponent p", min_value=1, max_value=12, value=2, step=1, key="quant_lvl_start"))
        stop_pow = int(c2.number_input("Stop exponent p", min_value=start_pow, max_value=12, value=8, step=1, key="quant_lvl_stop"))
        step = int(c3.number_input("Step", min_value=1, max_value=4, value=1, step=1, key="quant_lvl_step"))
        values = [2**p for p in range(start_pow, stop_pow + 1, step)]
    if st.button("Run quantizer sweep", key="run_quantizer_sweep"):
        backend = resolve_backend(cfg.compute_backend)
        st.session_state.quantizer_sweep = run_quantizer_sweep(cfg.source, cfg.pcm, backend=backend, sweep_axis=sweep_axis, sweep_values=values)
    sweep = st.session_state.get("quantizer_sweep")
    if sweep is None:
        st.info("Choose a sweep range and click Run quantizer sweep.")
        return
    if sweep.quantizer_mode == "levels":
        st.pyplot(plot_metric_sweep(sweep.levels, sweep.sqnr_db, "SQNR vs Levels", "Levels L", "SQNR (dB)"))
    else:
        fig = plot_metric_sweep(sweep.bits_per_sample, sweep.sqnr_db, "SQNR vs Bits", "Bits per sample n", "SQNR (dB)")
        st.pyplot(fig)
    st.dataframe(
        {
            "input_value": sweep.values,
            "bits": sweep.bits_per_sample,
            "levels": sweep.levels,
            "delta": sweep.delta,
            "sqnr_db": sweep.sqnr_db,
        },
        width="stretch",
    )
    if source_nyq > 0:
        st.caption(f"Source Nyquist reference is {source_nyq:.2f} Hz for the current tones.")


def show_sampling_sweep_tab(cfg: AppConfig, source_result):
    st.subheader("Sampling Rate vs Nyquist Sweep")
    nyq = nyquist_rate(cfg.source)
    if nyq <= 0:
        st.info("Add at least one tone to compute a Nyquist reference.")
        return
    c1, c2, c3 = st.columns(3)
    start = float(c1.number_input("Start Fs (Hz)", min_value=1.0, value=max(0.5 * nyq, 100.0), step=100.0, key="samp_start"))
    stop = float(c2.number_input("Stop Fs (Hz)", min_value=start, value=max(2.0 * nyq, start + 100.0), step=100.0, key="samp_stop"))
    step = float(c3.number_input("Step Fs (Hz)", min_value=1.0, value=max(0.25 * nyq, 100.0), step=100.0, key="samp_step"))
    values = np.arange(start, stop + 0.5 * step, step)
    if st.button("Run sampling sweep", key="run_sampling_sweep"):
        backend = resolve_backend(cfg.compute_backend)
        st.session_state.sampling_sweep = run_sampling_sweep(cfg.source, cfg.pcm, backend=backend, sample_rates=values)
    sweep = st.session_state.get("sampling_sweep")
    if sweep is None:
        st.info("Choose a sampling-rate range and click Run sampling sweep.")
        return
    st.pyplot(plot_sampling_sweep(sweep.sample_fs, sweep.sqnr_db, sweep.nyquist_rate))
    st.dataframe(
        {
            "Fs (Hz)": sweep.sample_fs,
            "below_nyquist": sweep.is_below_nyquist,
            "sample_count": sweep.sample_counts,
            "SQNR (dB)": sweep.sqnr_db,
        },
        width="stretch",
    )
    st.caption(
        "The points left of the dashed line are below Nyquist and should show aliasing impact in the sampled waveform."
    )


def show_line_tab(result, cfg: AppConfig):
    st.subheader("3) Line Coding and Pulse Shaping")
    backend = resolve_backend(cfg.compute_backend)
    line = result.line
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Encoding", cfg.line.encoding)
    c2.metric("Pulse", line.pulse_name)
    c3.metric("Samples/bit", line.samples_per_bit)
    c4.metric("Bit rate", f"{line.bit_rate:.2f} bps")
    st.pyplot(plot_time_series(line.t, line.waveform, "Line-Coded Waveform"))
    st.pyplot(plot_eye_diagram(line.waveform, line.samples_per_bit, cfg.source.sim_fs, line.timing_offset // 2, "Eye Diagram: Line-Coded Waveform", traces=cfg.plot.eye_traces, span_bits=cfg.plot.eye_span_bits))
    if cfg.plot.show_frequency_domain:
        st.pyplot(plot_frequency_response(line.pulse, cfg.source.sim_fs, "Pulse Spectrum"))
    alt_shape = None
    if cfg.plot.compare_sinc_rc:
        if cfg.line.pulse_shape == "raised_cosine":
            alt_shape = "sinc"
        elif cfg.line.pulse_shape == "sinc":
            alt_shape = "raised_cosine"
    if alt_shape:
        alt_cfg = AppConfig(
            compute_backend=cfg.compute_backend,
            source=cfg.source,
            pcm=cfg.pcm,
            line=LineCodeConfig(**{**asdict(cfg.line), "pulse_shape": alt_shape}),
            channel=cfg.channel,
            receiver=cfg.receiver,
            ber=cfg.ber,
            plot=cfg.plot,
        )
        alt_line = build_line_code(line.bits, line.bit_rate, cfg.source.sim_fs, alt_cfg.line, backend)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_eye_diagram(line.waveform, line.samples_per_bit, cfg.source.sim_fs, line.timing_offset // 2, f"Eye: {line.pulse_name}", traces=cfg.plot.eye_traces, span_bits=cfg.plot.eye_span_bits))
        with c2:
            st.pyplot(plot_eye_diagram(alt_line.waveform, alt_line.samples_per_bit, cfg.source.sim_fs, alt_line.timing_offset // 2, f"Eye: {alt_line.pulse_name}", traces=cfg.plot.eye_traces, span_bits=cfg.plot.eye_span_bits))
        st.caption("Eye-diagram comparison between the selected Nyquist pulse and its alternate.")


def show_channel_tab(result, cfg: AppConfig):
    st.subheader("4) AWGN Channel")
    ch = result.channel
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal power", f"{ch.signal_power:.4f}")
    c2.metric("Noise power", f"{ch.noise_power:.4f}")
    c3.metric("Achieved SNR", _fmt_db(ch.achieved_snr_db))
    st.pyplot(plot_two_series(result.line.t, ch.tx_waveform, ch.noisy_waveform, "Tx", "Rx noisy", "Channel Output"))
    st.pyplot(plot_eye_diagram(ch.noisy_waveform, result.line.samples_per_bit, cfg.source.sim_fs, result.line.timing_offset // 2, "Eye Diagram: Noisy Signal", traces=cfg.plot.eye_traces, span_bits=cfg.plot.eye_span_bits))


def show_receiver_tab(result, cfg: AppConfig):
    st.subheader("5) Matched Filter and Detection")
    rx = result.receiver
    c1, c2, c3 = st.columns(3)
    c1.metric("BER", _fmt_ber(rx.ber))
    c2.metric("Errors", rx.errors)
    c3.metric("Threshold", f"{rx.threshold:.2f}")
    mf_t = np.arange(len(rx.matched_waveform), dtype=float) / cfg.source.sim_fs
    st.pyplot(plot_time_series(mf_t, rx.matched_waveform, "Matched-Filter Output"))
    st.pyplot(plot_eye_diagram(rx.matched_waveform, result.line.samples_per_bit, cfg.source.sim_fs, rx.timing_offset, "Eye Diagram: Matched Filter Output", traces=cfg.plot.eye_traces, span_bits=cfg.plot.eye_span_bits))
    sample_indices = rx.sample_indices
    sampled = rx.sampled_voltages
    if len(sample_indices):
        fig = plot_time_series(mf_t, rx.matched_waveform, "Matched-Filter Output with Sampling Instants")
        ax = fig.axes[0]
        plot_n = min(len(mf_t), 5000)
        sample_indices_np = to_numpy(sample_indices).astype(int)
        sampled_np = to_numpy(sampled)
        valid = sample_indices_np < plot_n
        ax.plot(mf_t[sample_indices_np[valid]], sampled_np[valid], "ko", markersize=4, label="Samples")
        ax.legend()
        st.pyplot(fig)
    st.write("Detected bits")
    st.code("".join(map(str, rx.detected_bits[:cfg.plot.preview_bits].tolist())))
    st.dataframe(
        {
            "sample_index": to_numpy(rx.sample_indices[:cfg.plot.preview_bits]),
            "sampled_voltage": to_numpy(rx.sampled_voltages[:cfg.plot.preview_bits]),
            "detected_bit": to_numpy(rx.detected_bits[:cfg.plot.preview_bits]),
        },
        width="stretch",
    )


def show_sweep_tab(result, cfg: AppConfig):
    st.subheader("Runtime Sweeps")

    with st.expander("SQNR vs Quantizer Size", expanded=True):
        show_quantizer_sweep_tab(cfg, result)

    with st.expander("SQNR vs Sampling Rate", expanded=True):
        show_sampling_sweep_tab(cfg, result)

    st.divider()
    st.subheader("BER vs SNR Sweep")
    if st.button("Run BER sweep", key="run_sweep_button"):
        try:
            st.session_state.sweep_result = run_ber_sweep(cfg)
        except Exception as exc:
            st.error(str(exc))
            return
    sweep = st.session_state.get("sweep_result")
    if sweep is None:
        st.info("Click Run BER sweep to generate the waterfall curve.")
        return
    st.pyplot(plot_ber_curve(sweep.snr_db, sweep.ber))
    st.dataframe({"SNR dB": sweep.snr_db, "BER": sweep.ber, "Errors": sweep.errors}, width="stretch")


def main():
    inject_custom_css()
    st.title("End-to-End Digital Communication")
    st.caption("Source -> PCM -> Line Coding -> Pulse Shaping -> AWGN -> Matched Filter -> Detection -> BER")

    draft_cfg = build_config_from_sidebar()
    if "applied_cfg" not in st.session_state:
        st.session_state.applied_cfg = draft_cfg

    if st.sidebar.button("Apply / Refresh Settings", use_container_width=True):
        st.session_state.applied_cfg = draft_cfg
        st.session_state.pop("result", None)
        st.session_state.pop("sweep_result", None)
        st.session_state.pop("quantizer_sweep", None)
        st.session_state.pop("sampling_sweep", None)
        st.rerun()

    applied_cfg = st.session_state.applied_cfg
    show_config_summary(draft_cfg)

    if draft_cfg.to_dict() != applied_cfg.to_dict():
        st.warning("Sidebar values differ from the applied settings. Click Apply / Refresh Settings before running the pipeline.")
        with st.expander("Applied configuration", expanded=False):
            st.json(applied_cfg.to_dict())

    generate = st.sidebar.button("Generate End-to-End Flow", type="primary", use_container_width=True)
    if generate or "result" in st.session_state:
        if generate:
            try:
                run_cfg = st.session_state.applied_cfg
                st.session_state.result = run_end_to_end(run_cfg)
                st.session_state.cfg = run_cfg
            except Exception as exc:
                st.error(str(exc))
                return
        result = st.session_state.result
        cfg = st.session_state.get("cfg", applied_cfg)
        tabs = st.tabs(["📡 Source", "⚙️ PCM", "〰️ Line Coding", "🌪️ Channel", "🎯 Receiver", "📈 Sweeps"])
        with tabs[0]:
            show_source_tab(result)
        with tabs[1]:
            show_pcm_tab(result)
        with tabs[2]:
            show_line_tab(result, cfg)
        with tabs[3]:
            show_channel_tab(result, cfg)
        with tabs[4]:
            show_receiver_tab(result, cfg)
        with tabs[5]:
            show_sweep_tab(result, cfg)
    else:
        st.info("Set the parameters in the sidebar, then click Generate End-to-End Flow.")


if __name__ == "__main__":
    main()
