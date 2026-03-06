from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return clamp01(value / scale)
