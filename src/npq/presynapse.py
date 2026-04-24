"""Presynapse: active zone with docking sites, vesicles, and Ca-gated fusion."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from npq.calcium import CalciumTransient
from npq.docking import DockingSite
from npq.vesicle import Vesicle, VesicleState

if TYPE_CHECKING:
    from npq.neuron import Neuron


@dataclass
class Presynapse:
    sites: list[DockingSite]
    vesicles: list[Vesicle]
    calcium: CalciumTransient = field(default_factory=CalciumTransient)
    targets: list[tuple["Neuron", float]] = field(default_factory=list)

    def add_target(self, neuron: "Neuron", weight: float = 1.0) -> None:
        self.targets.append((neuron, weight))

    @property
    def docked_count(self) -> int:
        return sum(1 for v in self.vesicles if v.is_docked)

    @property
    def free_count(self) -> int:
        return sum(1 for v in self.vesicles if v.is_free)

    @property
    def fused_count(self) -> int:
        return sum(1 for v in self.vesicles if v.is_fused)

    def on_spike(self, t: float) -> None:
        """Presynaptic AP: inject a Ca transient. Fusion is not triggered here."""
        self.calcium.kick()

    def step(self, dt: float, rng: np.random.Generator) -> float:
        """Advance one tick. Return total quantal size released this step."""
        self.calcium.decay(dt)
        ca = self.calcium.value

        released = 0.0

        # 1) docking: FREE -> DOCKED into an empty site
        for v in self.vesicles:
            if v.state is not VesicleState.FREE:
                continue
            empty = next((s for s in self.sites if s.is_empty), None)
            if empty is None:
                break  # no room to dock any more
            p_dock = 1.0 - math.exp(-v.docking_rate * dt)
            if rng.random() < p_dock:
                empty.bind(v)

        # 2) fusion: every DOCKED vesicle rolls against Ca-dependent rate every tick
        for v in self.vesicles:
            if v.state is not VesicleState.DOCKED:
                continue
            k = v.fusion_rate(ca)
            if k <= 0.0:
                continue
            p_fuse = 1.0 - math.exp(-k * dt)
            if rng.random() < p_fuse:
                if v.site is not None:
                    v.site.release()
                v.state = VesicleState.FUSED
                released += v.quantal_size

        # 3) recovery: FUSED -> FREE
        for v in self.vesicles:
            if v.state is not VesicleState.FUSED:
                continue
            p_rec = 1.0 - math.exp(-v.recovery_rate * dt)
            if rng.random() < p_rec:
                v.state = VesicleState.FREE

        return released
