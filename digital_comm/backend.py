"""NumPy/CuPy simulation backend, separate from any ML-training claim.

A CPU-admitted worker never imports CuPy. CUDA discovery requires an actual
allocated array, operation and synchronization; GPU-admitted work fails closed
when the native CuPy execution path cannot be used.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

# Import optional accelerators lazily so a CPU-only scheduler child never
# loads their native modules merely by importing the simulation package.
cp: Any | None = None

_TRUE = {"1", "true", "yes", "on", "y"}
_CPU_FLAGS = ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY",
              "OPF_ADP_DISABLE_GPU_ACCELERATORS")


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    is_gpu: bool = False


def admission_signature() -> tuple[bool, bool, str | None]:
    """CPU/GPU admission and device visibility for one backend operation."""
    parent = os.environ.get("TRAINING_CONTROL_BACKEND", "").strip().lower()
    if parent and parent not in {"cpu", "gpu"}:
        raise RuntimeError(f"invalid admitted simulation backend: {parent!r}")
    cpu = (
        parent == "cpu"
        or any(os.environ.get(name, "").strip().lower() in _TRUE
               for name in _CPU_FLAGS)
        or ("CUDA_VISIBLE_DEVICES" in os.environ
            and os.environ["CUDA_VISIBLE_DEVICES"].strip() in {"", "-1"})
    )
    gpu = parent == "gpu"
    if cpu and gpu:
        raise RuntimeError("conflicting CPU and GPU simulation admission")
    return cpu, gpu, os.environ.get("CUDA_VISIBLE_DEVICES")


def _assert_admission(original: tuple[bool, bool, str | None]) -> None:
    if admission_signature() != original:
        raise RuntimeError(
            "simulation accelerator admission changed during backend operation; "
            "create a fresh worker"
        )


def _cupy_module() -> Any | None:
    global cp
    if admission_signature()[0]:
        return None
    if cp is None:
        try:
            import cupy as module  # type: ignore
        except Exception:  # optional dependency may not load without drivers
            return None
        cp = module
    return cp


def cupy_available() -> bool:
    original = admission_signature()
    if original[0]:
        return False
    module = _cupy_module()
    _assert_admission(original)
    if module is None:
        return False
    try:
        if module.cuda.runtime.getDeviceCount() < 1:
            _assert_admission(original)
            return False
        probe = module.empty((1,), dtype=module.uint8)
        probe.fill(1)
        module.cuda.runtime.deviceSynchronize()
        del probe
    except Exception:
        # Driver visibility or an allocation alone is not CUDA execution.
        _assert_admission(original)
        return False
    _assert_admission(original)
    return True


def resolve_backend(mode: str = "auto") -> ArrayBackend:
    mode = (mode or "auto").lower()
    if mode not in {"auto", "numpy", "cupy"}:
        raise ValueError(f"unsupported compute backend: {mode!r}")
    original = admission_signature()
    cpu_only, gpu_admitted, _ = original
    if mode == "numpy":
        # Explicit NumPy is a deliberate CPU reference cohort inside the
        # parameter-grid parity runner, not an automatic GPU-job fallback.
        return ArrayBackend("numpy", np, False)
    if cpu_only:
        if mode == "cupy":
            raise RuntimeError("central CPU admission forbids a CuPy backend")
        return ArrayBackend("numpy", np, False)
    available = cupy_available()
    _assert_admission(original)
    if available:
        return ArrayBackend("cupy", cp, True)
    if gpu_admitted:
        raise RuntimeError(
            "GPU-admitted simulation worker has no usable CuPy CUDA runtime"
        )
    if mode == "cupy":
        raise RuntimeError("CuPy is not installed or CUDA execution is unavailable")
    return ArrayBackend("numpy", np, False)


def to_numpy(x):
    original = admission_signature()
    is_cupy = getattr(type(x), "__module__", "").startswith("cupy")
    if original[0] and is_cupy:
        raise RuntimeError("CPU-admitted simulation cannot convert an accelerator array")
    device = getattr(x, "device", None)
    if original[0] and getattr(device, "type", None) in {"cuda", "hip", "mps", "xpu"}:
        raise RuntimeError("CPU-admitted simulation cannot convert an accelerator array")
    if not is_cupy:
        return np.asarray(x)
    module = _cupy_module()
    _assert_admission(original)
    if module is None:
        raise RuntimeError("CuPy array requires a usable CuPy module")
    result = module.asnumpy(x)
    _assert_admission(original)
    return result


def to_scalar(x) -> float:
    return float(np.asarray(to_numpy(x)).item())
