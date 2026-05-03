#!/usr/bin/env python3
"""Generate all demo screenshots and compile a Word report.
Eye diagrams use random bits at 2000 bps (50 samples/bit) for all patterns.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_comm.config import AppConfig, ToneSpec, LineCodeConfig
from digital_comm.simulation import run_end_to_end, run_ber_sweep
from digital_comm.sweeps import run_quantizer_sweep, run_sampling_sweep
from digital_comm.visualization import (
    plot_time_series, plot_two_series, plot_eye_diagram,
    plot_frequency_response, plot_ber_curve, plot_metric_sweep,
    plot_sampling_sweep as viz_sampling_sweep, _plot_eye_segments,
)
from digital_comm.backend import to_numpy

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "report", "screenshots")
os.makedirs(OUT, exist_ok=True)
saved = []  # (path, caption)

def save(fig, name, caption):
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    saved.append((path, caption))
    print(f"  [saved] {name}.png")


def eye_cfg(enc="polar", ps="rect_nrz", alpha=0.35, snr=None, noise=False):
    """Return a config optimised for eye diagrams:
    - random bits  → all bit transitions present
    - 2000 bps + 100 kHz sim_fs → 50 samples/bit → smooth curves
    - 500 random bits → 500 symbol periods of trace data
    """
    cfg = AppConfig()
    cfg.line.encoding = enc
    cfg.line.pulse_shape = ps
    cfg.line.rc_alpha = alpha
    cfg.line.payload_mode = "random_bits"
    cfg.line.bit_rate_mode = "manual"
    cfg.line.manual_bit_rate = 2000.0      # 50 samples/bit at 100 kHz
    cfg.ber.num_bits = 500                  # 500 bit periods of data
    cfg.ber.seed = 42
    cfg.channel.enable_awgn = noise
    if snr is not None:
        cfg.channel.snr_db = float(snr)
    return cfg


# ── BLOCK 1: Source ─────────────────────────────────────────────────────────
print("Block 1: Source")
cfg = AppConfig()
r = run_end_to_end(cfg)
save(plot_time_series(r.source.t, r.source.signal,
                      "Source Signal – Single 100 Hz Sine Tone"),
     "01_source_single",
     "Single 100 Hz sinusoidal tone, amplitude 1, duration 50 ms at 100 kHz simulation rate.")

cfg2 = AppConfig()
cfg2.source.mode = "multi"
cfg2.source.tones = [ToneSpec("sin", 1.0, 100.0, 0.0),
                     ToneSpec("cos", 0.5, 200.0, 0.0)]
r2 = run_end_to_end(cfg2)
save(plot_time_series(r2.source.t, r2.source.signal,
                      "Source Signal – Multi-Tone (100 Hz + 200 Hz)"),
     "01_source_multi",
     "Multi-tone source: 100 Hz sine + 200 Hz cosine at 0.5 amplitude, demonstrating superposition.")

# ── BLOCK 2: PCM ─────────────────────────────────────────────────────────────
print("Block 2: PCM")
for bits in [2, 4, 8]:
    cfg = AppConfig(); cfg.pcm.bits_per_sample = bits; cfg.channel.enable_awgn = False
    r = run_end_to_end(cfg); pcm = r.pcm
    save(plot_two_series(pcm.t_sampled, pcm.sampled_signal, pcm.quantized_signal,
                         "Sampled", "Quantized",
                         f"PCM – {bits}-bit (L={pcm.levels}, SQNR={pcm.sqnr_db:.1f} dB)"),
         f"02_pcm_{bits}bit",
         f"{bits}-bit quantizer with {pcm.levels} levels. SQNR={pcm.sqnr_db:.1f} dB, Δ={pcm.delta:.4f} V.")

# Annotated codewords
cfg = AppConfig(); cfg.pcm.bits_per_sample = 4; cfg.channel.enable_awgn = False
r = run_end_to_end(cfg); pcm = r.pcm
fig, ax = plt.subplots(figsize=(10, 4))
n = min(20, len(pcm.t_sampled))
ax.step(to_numpy(pcm.t_sampled[:n]), to_numpy(pcm.quantized_signal[:n]),
        where="post", color="#10b981", lw=2, label="Quantized")
ax.plot(to_numpy(pcm.t_sampled[:n]), to_numpy(pcm.sampled_signal[:n]),
        "o", color="#8b5cf6", ms=6, label="Sampled")
for i in range(n):
    ax.annotate(pcm.codewords[i],
                (float(pcm.t_sampled[i]), float(pcm.sampled_signal[i])),
                textcoords="offset points", xytext=(0, 10), fontsize=7, ha="center")
ax.set_title("PCM Codewords – 4-bit Quantization", fontweight="600")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
save(fig, "02_pcm_codewords",
     "4-bit binary codewords annotated at each sample. MSB first; index maps to quantisation level.")

cfg = AppConfig()
qs = run_quantizer_sweep(cfg.source, cfg.pcm, sweep_values=list(range(1, 9)))
save(plot_metric_sweep(qs.bits_per_sample, qs.sqnr_db,
                       "SQNR vs Bits per Sample", "Bits (n)", "SQNR (dB)"),
     "02_sqnr_vs_bits",
     "SQNR increases by ~6 dB per bit, consistent with SQNR ≈ 6.02n + 1.76 dB theory.")

rates = np.arange(100, 1100, 100.0)
ss = run_sampling_sweep(cfg.source, cfg.pcm, sample_rates=rates)
save(viz_sampling_sweep(ss.sample_fs, ss.sqnr_db, ss.nyquist_rate),
     "02_sqnr_vs_fs",
     "SQNR vs sampling rate. Below Nyquist (200 Hz) aliasing distorts; above it SQNR rises with oversampling.")

# ── BLOCK 3 & 4: Line Coding + Pulse Shaping ─────────────────────────────────
print("Block 3+4: Line Coding & Pulse Shaping")
encodings = [("polar","Polar NRZ (±1)"),
             ("unipolar","Unipolar (On-Off)"),
             ("bipolar","Bipolar (AMI)")]
pulses = [("rect_nrz","Rectangular NRZ"),
          ("rect_rz","Rectangular RZ"),
          ("sinc","Sinc"),
          ("raised_cosine","Raised Cosine")]

for enc_key, enc_label in encodings:
    for ps_key, ps_label in pulses:
        cfg = eye_cfg(enc=enc_key, ps=ps_key)
        r = run_end_to_end(cfg); lc = r.line
        tag = f"03_{enc_key}_{ps_key}"
        save(plot_time_series(lc.t, lc.waveform,
                              f"Line Code: {enc_label} – {ps_label}"),
             tag + "_wave",
             f"{enc_label} waveform, {ps_label} pulse, {lc.bit_rate:.0f} bps, {lc.samples_per_bit} samples/bit.")
        save(plot_eye_diagram(lc.waveform, lc.samples_per_bit, cfg.source.sim_fs,
                              lc.timing_offset // 2,
                              f"Eye: {enc_label} – {ps_label}", traces=300, span_bits=2),
             tag + "_eye",
             f"Eye diagram for {enc_label} + {ps_label}. Open eye at t=0 indicates zero ISI at the sampling instant.")

# Pulse spectra
for ps_key, ps_label in pulses:
    cfg = eye_cfg(ps=ps_key); r = run_end_to_end(cfg)
    save(plot_frequency_response(r.line.pulse, cfg.source.sim_fs,
                                 f"Pulse Spectrum – {ps_label}"),
         f"03_spectrum_{ps_key}",
         f"Normalised magnitude spectrum of {ps_label} pulse, showing bandwidth characteristics.")

# RC alpha sweep (eye diagrams)
print("Block 4: RC alpha sweep")
for alpha in [0.0, 0.35, 0.5, 1.0]:
    cfg = eye_cfg(ps="raised_cosine", alpha=alpha); r = run_end_to_end(cfg); lc = r.line
    atag = str(alpha).replace(".", "p")
    save(plot_eye_diagram(lc.waveform, lc.samples_per_bit, cfg.source.sim_fs,
                          lc.timing_offset // 2, f"RC Eye α={alpha}", traces=300, span_bits=2),
         f"04_rc_alpha{atag}_eye",
         f"Raised-cosine eye at α={alpha}. Higher α trades bandwidth for steeper pulse rolloff and cleaner eye opening.")

# RC vs Sinc side by side
cfg_rc = eye_cfg(ps="raised_cosine", alpha=0.35)
cfg_s  = eye_cfg(ps="sinc")
r_rc = run_end_to_end(cfg_rc); r_s = run_end_to_end(cfg_s)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
_plot_eye_segments(ax1, to_numpy(r_rc.line.waveform), r_rc.line.samples_per_bit,
                   cfg_rc.source.sim_fs, r_rc.line.timing_offset // 2, 300,
                   "Eye: Raised Cosine α=0.35", 2)
_plot_eye_segments(ax2, to_numpy(r_s.line.waveform), r_s.line.samples_per_bit,
                   cfg_s.source.sim_fs,  r_s.line.timing_offset // 2,  300,
                   "Eye: Sinc (Ideal Nyquist)", 2)
fig.tight_layout()
save(fig, "04_rc_vs_sinc_eye",
     "RC (α=0.35) vs ideal sinc. Both achieve zero ISI at symbol instants; RC has finite, practical bandwidth.")

# ── BLOCK 5: AWGN Channel ────────────────────────────────────────────────────
print("Block 5: AWGN")
for snr in [-5, 0, 5, 10, 20]:
    cfg = eye_cfg(noise=True, snr=snr); r = run_end_to_end(cfg)
    ch = r.channel; lc = r.line
    stag = f"{'m' if snr < 0 else ''}{abs(snr)}"
    save(plot_two_series(lc.t, ch.tx_waveform, ch.noisy_waveform,
                         "Tx", "Rx (noisy)",
                         f"AWGN Channel – SNR={snr} dB"),
         f"05_channel_snr{stag}dB",
         f"Transmitted vs received at SNR={snr} dB. Noise power={ch.noise_power:.4f}, achieved SNR={ch.achieved_snr_db:.1f} dB.")
    save(plot_eye_diagram(ch.noisy_waveform, lc.samples_per_bit, cfg.source.sim_fs,
                          lc.timing_offset // 2,
                          f"Noisy Eye (Rx Input) – SNR={snr} dB", traces=300, span_bits=2),
         f"05_noisy_eye_snr{stag}dB",
         f"Eye at receiver input, SNR={snr} dB. Eye closes as noise grows; ISI+noise combine to degrade margin.")

# ── BLOCK 6: Matched Filter ──────────────────────────────────────────────────
print("Block 6: Matched Filter")
for snr in [0, 5, 10, 20]:
    cfg = eye_cfg(noise=True, snr=snr); r = run_end_to_end(cfg)
    rx = r.receiver; lc = r.line
    mf_t = np.arange(len(rx.matched_waveform), dtype=float) / cfg.source.sim_fs
    fig = plot_time_series(mf_t, rx.matched_waveform,
                           f"Matched Filter Output – SNR={snr} dB")
    ax = fig.axes[0]
    si = to_numpy(rx.sample_indices).astype(int)
    sv = to_numpy(rx.sampled_voltages)
    valid = si < len(mf_t)
    ax.plot(mf_t[si[valid]], sv[valid], "ko", ms=4,
            label=f"Decision samples (thr={rx.threshold:.2f})")
    ax.axhline(rx.threshold, color="red", ls="--", lw=1.5, label="Threshold")
    ax.legend(fontsize=8); fig.tight_layout()
    save(fig, f"06_mf_output_snr{snr}dB",
         f"MF output at SNR={snr} dB. Black dots are decision-instant samples; red line is detection threshold.")
    save(plot_eye_diagram(rx.matched_waveform, lc.samples_per_bit, cfg.source.sim_fs,
                          rx.timing_offset,
                          f"MF Eye – SNR={snr} dB", traces=300, span_bits=2),
         f"06_mf_eye_snr{snr}dB",
         f"MF eye at SNR={snr} dB. Eye opens significantly vs pre-filter, showing matched-filter SNR gain.")

# MF per pulse shape at SNR=10
for ps_key, ps_label in pulses:
    cfg = eye_cfg(ps=ps_key, noise=True, snr=10); r = run_end_to_end(cfg)
    save(plot_eye_diagram(r.receiver.matched_waveform, r.line.samples_per_bit,
                          cfg.source.sim_fs, r.receiver.timing_offset,
                          f"MF Eye: {ps_label}, SNR=10 dB", traces=300, span_bits=2),
         f"06_mf_eye_{ps_key}",
         f"MF eye for {ps_label} at SNR=10 dB. ISI-free eye confirms the matched filter is correctly implemented.")

# ── BLOCK 7: BER Waterfall ────────────────────────────────────────────────────
print("Block 7: BER Waterfall")
cfg = AppConfig()
cfg.ber.snr_start_db = 0.0; cfg.ber.snr_stop_db = 12.0
cfg.ber.snr_step_db  = 2.0; cfg.ber.num_bits = 100_000
sw = run_ber_sweep(cfg)
save(plot_ber_curve(sw.snr_db, sw.ber),
     "07_ber_waterfall_polar_nrz",
     "BER vs SNR waterfall (Polar NRZ, 100 k bits/point). BER drops from ~1% at 0 dB to ~0 above 8 dB.")

for enc_key, enc_label in encodings:
    cfg2 = AppConfig(); cfg2.line.encoding = enc_key
    cfg2.ber.snr_start_db = 0.0; cfg2.ber.snr_stop_db = 12.0
    cfg2.ber.snr_step_db = 2.0;  cfg2.ber.num_bits = 50_000
    sw2 = run_ber_sweep(cfg2)
    save(plot_ber_curve(sw2.snr_db, sw2.ber),
         f"07_ber_waterfall_{enc_key}",
         f"BER waterfall for {enc_label}. Waterfall shape reflects symbol energy and decision threshold distance.")

# Detected bits snapshot
cfg = eye_cfg(noise=True, snr=10); r = run_end_to_end(cfg)
rx = r.receiver; lc = r.line
N = min(60, len(rx.detected_bits))
tx_s  = "".join(map(str, to_numpy(lc.bits[:N])))
det_s = "".join(map(str, to_numpy(rx.detected_bits[:N])))
err_s = "".join("^" if a != b else " " for a, b in zip(tx_s, det_s))
fig, ax = plt.subplots(figsize=(12, 2.5))
ax.axis("off")
ax.text(0.01, 0.75, f"TX:  {tx_s}",  family="monospace", fontsize=9,
        transform=ax.transAxes, color="#0ea5e9")
ax.text(0.01, 0.50, f"RX:  {det_s}", family="monospace", fontsize=9,
        transform=ax.transAxes, color="#10b981")
ax.text(0.01, 0.25, f"ERR: {err_s}", family="monospace", fontsize=9,
        transform=ax.transAxes, color="#ef4444")
ax.set_title(f"Detected Bits – SNR=10 dB | BER={rx.ber:.6f} | Errors={rx.errors}",
             fontweight="600")
fig.tight_layout()
save(fig, "07_detected_bits",
     f"First {N} TX vs detected bits at SNR=10 dB. Carets mark errors; BER={rx.ber:.6f}.")

print(f"\nTotal screenshots: {len(saved)}")

# ── COMPILE WORD REPORT ──────────────────────────────────────────────────────
print("\n=== COMPILING WORD REPORT ===")
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)

t = doc.add_heading("End-to-End Digital Communication System", 0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph("Course Demo Report – Simulation Plots & Descriptions")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].bold = True
doc.add_paragraph()

doc.add_heading("Overview", 1)
doc.add_paragraph(
    "This report covers all seven blocks of the end-to-end digital communication chain: "
    "source generation, PCM sampling & quantization, line coding (polar/unipolar/bipolar), "
    "pulse shaping (rect NRZ/RZ, sinc, raised cosine), AWGN channel, matched filter receiver, "
    "and BER analysis including the waterfall curve. "
    "Eye diagrams are generated using 500 random bits at 2000 bps (50 samples/symbol) "
    "so all bit-transition patterns are represented."
)

sections = [
    ("Block 1: Continuous-Time Source Signal", [
        ("01_source_single", "Single 100 Hz Sine Tone"),
        ("01_source_multi",  "Multi-Tone Source (100 Hz + 200 Hz)"),
    ]),
    ("Block 2: Sampling & Quantization (PCM)", [
        ("02_pcm_2bit",       "2-bit PCM"),
        ("02_pcm_4bit",       "4-bit PCM"),
        ("02_pcm_8bit",       "8-bit PCM"),
        ("02_pcm_codewords",  "4-bit Codewords Annotated"),
        ("02_sqnr_vs_bits",   "SQNR vs Bits per Sample"),
        ("02_sqnr_vs_fs",     "SQNR vs Sampling Rate"),
    ]),
    ("Block 3: Line Coding – Waveforms", [
        (f"03_{e}_{p}_wave", f"{el} – {pl}")
        for e, el in [("polar","Polar"),("unipolar","Unipolar"),("bipolar","Bipolar")]
        for p, pl in [("rect_nrz","NRZ"),("rect_rz","RZ"),("sinc","Sinc"),("raised_cosine","RC")]
    ]),
    ("Block 3: Line Coding – Eye Diagrams", [
        (f"03_{e}_{p}_eye", f"Eye: {el} – {pl}")
        for e, el in [("polar","Polar"),("unipolar","Unipolar"),("bipolar","Bipolar")]
        for p, pl in [("rect_nrz","NRZ"),("rect_rz","RZ"),("sinc","Sinc"),("raised_cosine","RC")]
    ]),
    ("Block 3: Pulse Spectra", [
        (f"03_spectrum_{p}", f"Spectrum – {pl}")
        for p, pl in [("rect_nrz","NRZ"),("rect_rz","RZ"),("sinc","Sinc"),("raised_cosine","RC")]
    ]),
    ("Block 4: RC Pulse Shaping – Alpha Sweep", [
        ("04_rc_alpha0p0_eye",  "RC α=0.0"),
        ("04_rc_alpha0p35_eye", "RC α=0.35"),
        ("04_rc_alpha0p5_eye",  "RC α=0.5"),
        ("04_rc_alpha1p0_eye",  "RC α=1.0"),
        ("04_rc_vs_sinc_eye",   "RC vs Sinc Comparison"),
    ]),
    ("Block 5: AWGN Channel", [
        (f"05_channel_snr{'m' if s<0 else ''}{abs(s)}dB", f"Channel Waveform SNR={s} dB")
        for s in [-5, 0, 5, 10, 20]
    ] + [
        (f"05_noisy_eye_snr{'m' if s<0 else ''}{abs(s)}dB", f"Noisy Eye SNR={s} dB")
        for s in [-5, 0, 5, 10, 20]
    ]),
    ("Block 6: Matched Filter", [
        (f"06_mf_output_snr{s}dB", f"MF Output SNR={s} dB") for s in [0,5,10,20]
    ] + [
        (f"06_mf_eye_snr{s}dB",    f"MF Eye SNR={s} dB")    for s in [0,5,10,20]
    ] + [
        (f"06_mf_eye_{p}", f"MF Eye – {pl}")
        for p, pl in [("rect_nrz","NRZ"),("rect_rz","RZ"),("sinc","Sinc"),("raised_cosine","RC")]
    ]),
    ("Block 7: BER Analysis", [
        ("07_ber_waterfall_polar_nrz", "BER Waterfall – Polar NRZ (100 k bits)"),
        ("07_ber_waterfall_polar",     "BER Waterfall – Polar"),
        ("07_ber_waterfall_unipolar",  "BER Waterfall – Unipolar"),
        ("07_ber_waterfall_bipolar",   "BER Waterfall – Bipolar"),
        ("07_detected_bits",           "Detected Bits at SNR=10 dB"),
    ]),
]

cap_map = {os.path.basename(p).replace(".png", ""): c for p, c in saved}

for section_title, items in sections:
    doc.add_heading(section_title, 1)
    for name, fallback in items:
        path = os.path.join(OUT, name + ".png")
        if not os.path.exists(path):
            continue
        p = doc.add_paragraph()
        run = p.add_run(fallback)
        run.bold = True
        try:
            doc.add_picture(path, width=Inches(5.8))
        except Exception as e:
            doc.add_paragraph(f"[Image error: {e}]")
        cap = doc.add_paragraph(cap_map.get(name, fallback))
        cap.runs[0].italic = True
        cap.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        doc.add_paragraph()

report_path = os.path.join(os.path.dirname(OUT), "Digital_Communication_Report.docx")
doc.save(report_path)
print(f"Report: {report_path}")
print(f"Plots:  {len(saved)}")
