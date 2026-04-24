"""Neuron: owns outgoing presynapses and receives quanta as input current."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from npq.presynapse import Presynapse


@dataclass
class Neuron:
    name: str = ""
    presynapses: list["Presynapse"] = field(default_factory=list)
    spike_times: list[float] = field(default_factory=list)
    input_current: float = 0.0
    _spike_idx: int = field(default=0, repr=False)

    def add_presynapse(self, presynapse: "Presynapse") -> None:
        self.presynapses.append(presynapse)

    def receive(self, quantum: float, t: float) -> None:
        self.input_current += quantum

    def fire(self, t: float) -> None:
        for p in self.presynapses:
            p.on_spike(t)

    def pending_spikes(self, t_start: float, t_end: float) -> list[float]:
        """Return scheduled spike times in [t_start, t_end), advancing the cursor."""
        fired: list[float] = []
        while self._spike_idx < len(self.spike_times):
            t = self.spike_times[self._spike_idx]
            if t >= t_end:
                break
            if t >= t_start:
                fired.append(t)
            self._spike_idx += 1
        return fired
