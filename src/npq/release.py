"""Ca-dependent fusion-rate functions."""
from __future__ import annotations

from typing import Callable

ReleaseRateFn = Callable[[float], float]


def hill_rate(k_max: float, K: float = 2.0, n: float = 4.0) -> ReleaseRateFn:
    """Return a fusion-rate function k_fuse(ca) = k_max * ca^n / (K^n + ca^n).

    Parameters are chosen so that at peak post-spike Ca the per-step
    fusion probability (1 - exp(-k_fuse * dt)) matches a target release
    probability. n~=4 matches synaptotagmin cooperativity estimates.
    """
    if k_max < 0 or K <= 0 or n <= 0:
        raise ValueError("k_max must be >=0; K and n must be >0")
    Kn = K**n

    def k_fuse(ca: float) -> float:
        if ca <= 0.0:
            return 0.0
        ca_n = ca**n
        return k_max * ca_n / (Kn + ca_n)

    return k_fuse
