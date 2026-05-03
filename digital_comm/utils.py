from __future__ import annotations

import numpy as np


def parse_bit_string(bit_string: str) -> np.ndarray:
    cleaned = "".join(ch for ch in bit_string if ch in "01")
    if not cleaned:
        return np.array([], dtype=int)
    return np.array([int(ch) for ch in cleaned], dtype=int)


def bits_to_binary_strings(bits: np.ndarray, width: int) -> list[str]:
    bits = np.asarray(bits, dtype=int)
    if width <= 0:
        return [str(int(b)) for b in bits]
    return [format(int(b), f"0{width}b") for b in bits]


def binary_strings_to_bits(strings: list[str]) -> np.ndarray:
    return np.array([int(bit) for s in strings for bit in s], dtype=int)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")

