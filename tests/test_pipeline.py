from __future__ import annotations

import unittest

import numpy as np

from digital_comm.backend import cupy_available, resolve_backend, to_numpy
from digital_comm.config import AppConfig, PCMConfig
from digital_comm.simulation import run_ber_sweep, run_end_to_end
from digital_comm.sweeps import run_quantizer_sweep, run_sampling_sweep


class DigitalCommPipelineTests(unittest.TestCase):
    def test_numpy_end_to_end_smoke(self):
        cfg = AppConfig(compute_backend="numpy")
        result = run_end_to_end(cfg)

        self.assertGreater(len(result.source.t), 0)
        self.assertGreater(len(result.line.waveform), 0)
        self.assertEqual(len(result.receiver.detected_bits), len(result.line.bits))
        self.assertTrue(np.isfinite(result.channel.signal_power))
        self.assertTrue(np.isfinite(result.receiver.ber))

    def test_pcm_levels_mode_and_sweeps(self):
        cfg = AppConfig(compute_backend="numpy")
        cfg.pcm = PCMConfig(sample_fs=4_000.0, quantizer_mode="levels", quantizer_levels=32, range_mode="auto")
        result = run_end_to_end(cfg)
        self.assertEqual(result.pcm.levels, 32)
        self.assertEqual(result.pcm.bits_per_sample, 5)
        self.assertEqual(len(result.pcm.pcm_bits), len(result.pcm.t_sampled) * result.pcm.bits_per_sample)

        quant_sweep = run_quantizer_sweep(cfg.source, cfg.pcm, backend=resolve_backend("numpy"), sweep_axis="levels", sweep_values=[4, 8, 16])
        self.assertEqual(len(quant_sweep.values), 3)
        self.assertTrue(np.all(np.isfinite(quant_sweep.sqnr_db)))

        samp_sweep = run_sampling_sweep(cfg.source, cfg.pcm, backend=resolve_backend("numpy"), sample_rates=[500.0, 1_000.0, 2_000.0])
        self.assertEqual(len(samp_sweep.sample_fs), 3)
        self.assertEqual(len(samp_sweep.sqnr_db), 3)
        self.assertTrue(np.all(samp_sweep.sample_counts > 0))

    def test_backend_auto_selection_respects_cupy_availability(self):
        backend = resolve_backend("auto")
        if cupy_available():
            self.assertEqual(backend.name, "cupy")
        else:
            self.assertEqual(backend.name, "numpy")

    @unittest.skipUnless(cupy_available(), "CuPy CUDA backend not available")
    def test_cupy_end_to_end_and_sweep(self):
        import cupy as cp

        cfg = AppConfig(compute_backend="cupy")
        result = run_end_to_end(cfg)
        self.assertIsInstance(result.source.t, cp.ndarray)
        self.assertIsInstance(result.line.waveform, cp.ndarray)
        self.assertIsInstance(result.channel.noisy_waveform, cp.ndarray)
        self.assertIsInstance(result.receiver.matched_waveform, cp.ndarray)
        self.assertEqual(to_numpy(result.receiver.detected_bits).shape[0], len(result.line.bits))

        cfg.ber.snr_start_db = 0.0
        cfg.ber.snr_stop_db = 10.0
        cfg.ber.snr_step_db = 5.0
        cfg.ber.num_bits = 20_000
        sweep = run_ber_sweep(cfg)
        self.assertEqual(len(sweep.snr_db), 3)
        self.assertGreaterEqual(sweep.ber[0], sweep.ber[-1])
        self.assertTrue(np.all(to_numpy(sweep.ber) >= 0))


if __name__ == "__main__":
    unittest.main()
