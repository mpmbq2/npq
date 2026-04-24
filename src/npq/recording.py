"""Trace recorders for simulation observables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from npq.neuron import Neuron
    from npq.presynapse import Presynapse


@dataclass
class PresynapseRecorder:
    presynapse: "Presynapse"
    calcium: list[float] = field(default_factory=list)
    docked: list[int] = field(default_factory=list)
    released: list[float] = field(default_factory=list)

    def sample(self, released_this_step: float) -> None:
        self.calcium.append(self.presynapse.calcium.value)
        self.docked.append(self.presynapse.docked_count)
        self.released.append(released_this_step)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "calcium": np.asarray(self.calcium),
            "docked": np.asarray(self.docked),
            "released": np.asarray(self.released),
        }


@dataclass
class NeuronRecorder:
    neuron: "Neuron"
    input_current: list[float] = field(default_factory=list)

    def sample(self) -> None:
        self.input_current.append(self.neuron.input_current)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"input_current": np.asarray(self.input_current)}
