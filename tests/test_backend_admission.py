"""CPU-safe native NumPy/CuPy simulation admission regressions.

These mock CUDA operations test policy behavior without claiming physical GPU
simulation or machine-learning training.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import digital_comm.backend as backend


def fake_cupy(*, devices=1, error=None):
    probe = SimpleNamespace(fill=mock.Mock())
    cp = SimpleNamespace(
        uint8=object(),
        cuda=SimpleNamespace(runtime=SimpleNamespace(
            getDeviceCount=mock.Mock(return_value=devices),
            deviceSynchronize=mock.Mock(),
        )),
        empty=mock.Mock(return_value=probe, side_effect=error),
        asnumpy=mock.Mock(return_value=np.asarray([7])),
    )
    return cp, probe


@pytest.fixture(autouse=True)
def clean_policy(monkeypatch):
    for key in ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
                "OPF_ADP_DISABLE_GPU_ACCELERATORS", "TRAINING_CONTROL_BACKEND",
                "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("policy", (
    {"CPU_ONLY": "true", "CUDA_VISIBLE_DEVICES": "0"},
    {"TRAINING_CONTROL_CPU_ONLY": "YES", "CUDA_VISIBLE_DEVICES": "0"},
    {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "on", "CUDA_VISIBLE_DEVICES": "0"},
    {"TRAINING_CONTROL_BACKEND": "cpu", "CUDA_VISIBLE_DEVICES": "0"},
    {"CUDA_VISIBLE_DEVICES": ""},
    {"CUDA_VISIBLE_DEVICES": " -1 "},
))
def test_cpu_admission_avoids_cupy_import_and_probe(monkeypatch, policy):
    for name, value in policy.items():
        monkeypatch.setenv(name, value)
    with mock.patch.object(backend, "cp", None), \
         mock.patch("builtins.__import__", side_effect=AssertionError("optional import attempted")):
        assert backend.cupy_available() is False
        assert backend.resolve_backend("auto").name == "numpy"
        assert backend.resolve_backend("numpy").name == "numpy"
        with pytest.raises(RuntimeError, match="central CPU admission"):
            backend.resolve_backend("cupy")
    assert np.array_equal(backend.to_numpy(np.array([1, 2])), [1, 2])


def test_cupy_probe_requires_allocation_operation_and_synchronization(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cp, probe = fake_cupy()
    with mock.patch.object(backend, "cp", cp):
        assert backend.cupy_available()
        assert backend.resolve_backend("auto").is_gpu
        assert backend.resolve_backend("cupy").xp is cp
    assert cp.empty.call_count == 3
    assert probe.fill.call_count == 3
    assert cp.cuda.runtime.deviceSynchronize.call_count == 3


def test_available_device_without_usable_allocation_is_not_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cp, _ = fake_cupy(error=RuntimeError("driver mismatch"))
    with mock.patch.object(backend, "cp", cp):
        assert not backend.cupy_available()
        assert backend.resolve_backend("auto").name == "numpy"
        with pytest.raises(RuntimeError, match="unavailable"):
            backend.resolve_backend("cupy")
    cp.cuda.runtime.deviceSynchronize.assert_not_called()


def test_gpu_admission_cannot_silently_select_numpy_auto(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cp, _ = fake_cupy(devices=0)
    with mock.patch.object(backend, "cp", cp):
        with pytest.raises(RuntimeError, match="GPU-admitted simulation"):
            backend.resolve_backend("auto")
        with pytest.raises(RuntimeError, match="GPU-admitted simulation"):
            backend.resolve_backend("cupy")
        # Explicit NumPy cases are required baseline/reference cohorts
        # inside the GPU-capable full parameter-grid comparison.
        assert backend.resolve_backend("numpy").name == "numpy"
    cp.empty.assert_not_called()


def test_conflicting_admission_fails_without_cuda_import(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "gpu")
    monkeypatch.setenv("CPU_ONLY", "1")
    cp, _ = fake_cupy()
    with mock.patch.object(backend, "cp", cp):
        with pytest.raises(RuntimeError, match="conflicting CPU and GPU"):
            backend.resolve_backend("auto")
    cp.cuda.runtime.getDeviceCount.assert_not_called()


def test_mid_probe_backend_change_never_authorizes_numpy_fallback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cp, probe = fake_cupy()
    probe.fill.side_effect = lambda value: monkeypatch.setenv(
        "TRAINING_CONTROL_BACKEND", "cpu"
    )
    with mock.patch.object(backend, "cp", cp):
        with pytest.raises(RuntimeError, match="create a fresh worker"):
            backend.resolve_backend("auto")


def test_mid_probe_failure_still_checks_original_admission(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def fail_and_mask(*args, **kwargs):
        monkeypatch.setenv("CPU_ONLY", "yes")
        raise RuntimeError("driver mismatch")
    cp, _ = fake_cupy(error=fail_and_mask)
    with mock.patch.object(backend, "cp", cp):
        with pytest.raises(RuntimeError, match="create a fresh worker"):
            backend.cupy_available()


def test_cpu_admission_rejects_existing_gpu_arrays_before_device_read(monkeypatch):
    CupyArray = type("CupyArray", (), {
        "device": property(lambda self: (_ for _ in ()).throw(
            AssertionError("GPU property accessed")
        ))
    })
    CupyArray.__module__ = "cupy._core.core"
    monkeypatch.setenv("CPU_ONLY", "1")
    with mock.patch.object(backend, "cp", None):
        with pytest.raises(RuntimeError, match="CPU-admitted simulation"):
            backend.to_numpy(CupyArray())
        with pytest.raises(RuntimeError, match="CPU-admitted simulation"):
            backend.to_scalar(CupyArray())
        with pytest.raises(RuntimeError, match="CPU-admitted simulation"):
            backend.to_numpy(SimpleNamespace(device=SimpleNamespace(type="cuda")))


def test_gpu_array_conversion_requires_unchanged_admission(monkeypatch):
    CupyArray = type("CupyArray", (), {})
    CupyArray.__module__ = "cupy._core.core"
    data = CupyArray()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cp, _ = fake_cupy()
    cp.asnumpy.side_effect = lambda value: (
        monkeypatch.setenv("CPU_ONLY", "1") or np.asarray([7])
    )
    with mock.patch.object(backend, "cp", cp):
        with pytest.raises(RuntimeError, match="create a fresh worker"):
            backend.to_numpy(data)
