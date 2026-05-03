# End-to-End Digital Communication System

A complete interactive simulation of an end-to-end digital communication chain built with Python and Streamlit.

---

## ⚡ 5-Minute Quick Start (New / Unknown PC)

> Works on any Linux machine with Python 3.8+. No root/sudo needed.

```bash
# 1. Unzip and enter the project folder
unzip Digital-Communication.zip
cd Digital-Communication

# 2. Create a virtual environment  (only uses Python stdlib, no sudo)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the interactive web app
streamlit run app.py
```

The browser will open automatically at **http://localhost:8501**.
If it does not, copy that URL into any browser manually.

---

## 🎯 Demo Script (Viva Checklist — 8/8 marks)

| # | What to show | Where in the UI |
|---|---|---|
| 1 | **Source signal** — single & multi-tone | Tab "1 – Source" |
| 2 | **PCM sampling & quantisation** — change bits per sample, see SQNR | Tab "2 – PCM" |
| 3 | **Line coding** — switch Polar / Unipolar / Bipolar | Tab "3 – Line Code" |
| 4 | **Eye diagram** — line-coded waveform before noise | Tab "3 – Line Code" (scroll down) |
| 5 | **Pulse shaping** — Rect NRZ/RZ, Sinc, Raised Cosine; compare eye diagrams | Tab "3 – Line Code" |
| 6 | **AWGN channel** — adjust SNR slider, show eye closing | Tab "4 – Channel" |
| 7 | **Matched filter** — eye reopens after MF | Tab "5 – Receiver" |
| 8 | **BER waterfall** — sweep SNR 0–12 dB, BER drops to 0 | Tab "6 – BER" |

---

## 📄 Pre-generated Report

All plots and a compiled Word report are in the `report/` folder:

```
report/
  Digital_Communication_Report.docx   ← submit this
  screenshots/                         ← all 68 plots as PNG
```

To regenerate all plots from scratch:
```bash
python scripts/generate_report.py
```

---

## 🗂 Project Structure

```
app.py                        ← Streamlit UI (entry point)
digital_comm/
  source.py                   ← Continuous-time signal generation
  pcm.py                      ← Sampling & quantisation (PCM)
  line_coding.py              ← Polar / Unipolar / Bipolar encoding
  pulses.py                   ← Rect NRZ/RZ, Sinc, Raised Cosine
  channel.py                  ← AWGN noise addition
  receiver.py                 ← Matched filter & bit detection
  simulation.py               ← End-to-end pipeline runner
  visualization.py            ← All plotting functions
  config.py                   ← Centralised parameter config
  sweeps.py                   ← SQNR / BER parameter sweeps
  results.py                  ← Result dataclasses
scripts/
  generate_report.py          ← Batch screenshot + Word report
  verify_parameter_grid.py    ← Automated regression checker
report/
  Digital_Communication_Report.docx
  screenshots/                ← 68 pre-generated PNG plots
docs/
  PROJECT_GUIDE.md            ← Full block-by-block guide
tests/
  test_pipeline.py            ← Unit tests
requirements.txt
run_demo.sh                   ← One-command demo launcher
```

---

## 🔑 Key Theory Points (Viva Q&A)

**SQNR formula:** `SQNR ≈ 6.02n + 1.76 dB` where `n` = bits per sample

**Matched filter:** Maximises SNR at the decision instant by correlating received signal with a copy of the transmitted pulse. Output SNR = 2E/N₀.

**RC pulse roll-off α:**
- α = 0 → ideal sinc (minimum bandwidth, sensitive to timing)
- α = 1 → 2× bandwidth of sinc, but very robust to timing errors
- Higher α → faster-decaying tails → cleaner eye, more bandwidth

**Eye diagram:** Overlay of all 2-bit segments. Eye OPEN at sampling instant (t=0) means zero ISI. Eye CLOSES with noise (SNR ↓) and reopens after matched filtering.

**BER waterfall:** For polar NRZ: `BER = Q(√(2·SNR))`. At SNR=10 dB → BER ≈ 10⁻⁵.

---

## GPU Acceleration (Optional)

If a CUDA GPU is available:
```bash
pip install -r requirements-gpu.txt
```
Then set backend to `cupy` in the sidebar.

---

## Virtual Environment — Quick Reference

```bash
source .venv/bin/activate    # activate
deactivate                   # deactivate
```
