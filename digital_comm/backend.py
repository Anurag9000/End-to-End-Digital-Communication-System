from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    is_gpu: bool = False


def cupy_available() -> bool:
    if (os.environ.get("OPF_ADP_DISABLE_GPU_ACCELERATORS", "").strip() == "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}):
        return False
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        # Driver/device visibility does not guarantee usable CUDA allocations.
        probe = cp.empty((1,), dtype=cp.uint8)
        del probe
        return True
    except Exception:
        return False


def resolve_backend(mode: str = "auto") -> ArrayBackend:
    mode = (mode or "auto").lower()
    if mode not in {"auto", "numpy", "cupy"}:
        raise ValueError(f"unsupported compute backend: {mode!r}")
    if mode == "numpy":
        return ArrayBackend("numpy", np, False)
    if mode == "cupy":
        if cp is None:
            raise RuntimeError("CuPy is not installed.")
        if not cupy_available():
            raise RuntimeError("CuPy is installed but no CUDA device is available.")
        return ArrayBackend("cupy", cp, True)
    if cupy_available():
        return ArrayBackend("cupy", cp, True)
    return ArrayBackend("numpy", np, False)


def to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):  # pragma: no branch - fast path
        return cp.asnumpy(x)
    return np.asarray(x)


def to_scalar(x) -> float:
    return float(np.asarray(to_numpy(x)).item())

