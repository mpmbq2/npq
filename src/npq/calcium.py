"""Presynaptic calcium transient: exponential decay with per-spike kicks."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalciumTransient:
    baseline: float = 0.05
    tau: float = 0.02
    delta: float = 2.0
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.value == 0.0:
            self.value = self.baseline

    def kick(self) -> None:
        self.value += self.delta

    def decay(self, dt: float) -> None:
        self.value = self.baseline + (self.value - self.baseline) * math.exp(-dt / self.tau)
