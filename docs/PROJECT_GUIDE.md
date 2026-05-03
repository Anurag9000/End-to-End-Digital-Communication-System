# Repository Guidelines

## Overview
This project implements an end-to-end digital communication chain for the course demo: continuous-time source generation, PCM sampling and quantization, line coding, pulse shaping, AWGN channel simulation, matched filtering, bit detection, and BER analysis. The Streamlit UI runs the full pipeline locally and exposes the main parameters as runtime controls.

## Step-by-Step Flow
1. **Source generation**: create one or more sinusoidal tones over a configurable observation window.
2. **Sampling and quantization**: sample the source at `Fs`, quantize it with either `n` bits or `L` levels, and convert samples to PCM bits.
3. **Line coding**: map bits to `unipolar`, `polar`, or `bipolar` symbols.
4. **Pulse shaping**: generate `rect_nrz`, `rect_rz`, `sinc`, or `raised_cosine` pulses and build the transmit waveform.
5. **Channel**: add AWGN at the selected SNR.
6. **Receiver**: apply the matched filter, sample at the decision instants, and detect bits.
7. **Metrics**: compute BER for the current run and BER vs SNR for the sweep.

## Runtime Controls
Use the sidebar, then click **Apply / Refresh Settings**, then **Generate End-to-End Flow**.

- **Source**: duration, simulation sample rate, tone count, sine/cosine choice, amplitude, frequency, phase.
- **PCM**: choose quantizer mode as `bits` or `levels`, set `n` or `L`, and set bottom/top voltages for the quantizer range.
- **Sampling/Nyquist**: set `Fs` above or below the Nyquist rate to observe aliasing and SQNR changes.
- **Line coding**: select payload source, encoding (`unipolar`, `polar`, `bipolar`), bit rate, and pulse type.
- **Pulse shape**: tune `rc_alpha`, pulse span, RZ duty cycle, and pulse normalization.
- **Channel**: enable/disable AWGN, set SNR, and set the random seed.
- **Receiver**: choose automatic or manual thresholding and timing offset.
- **Sweeps**: run SQNR vs bits/levels, SQNR vs sampling rate, and BER vs SNR.

## Demo Outputs
The UI exposes each stage separately so you can inspect:
- sampled and quantized PCM output,
- line-coded waveform,
- eye diagram before and after pulse shaping,
- noisy waveform and noisy eye diagram,
- matched-filter output and eye diagram,
- detected bits and BER,
- quantizer SQNR sweep,
- sampling-rate sweep around Nyquist,
- BER waterfall curve.

## Key Files
- `app.py`: Streamlit entry point and UI orchestration.
- `digital_comm/source.py`: source generation and Nyquist helpers.
- `digital_comm/pcm.py`: sampling, quantization, and PCM encoding.
- `digital_comm/line_coding.py`: bit-to-symbol mapping and pulse construction.
- `digital_comm/pulses.py`: rectangular, sinc, and raised-cosine pulse generators.
- `digital_comm/channel.py`: AWGN channel model.
- `digital_comm/receiver.py`: matched filter and detection.
- `digital_comm/sweeps.py`: SQNR and sampling-rate sweeps.
- `digital_comm/simulation.py`: end-to-end flow and BER sweep.

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional GPU acceleration:
```bash
pip install -r requirements-gpu.txt
```

For a one-command demo launch with verification:
```bash
bash run_demo.sh
```
