"""Simulation engine: orchestrates neurons, presynapses, and quanta dispatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from npq.neuron import Neuron
from npq.recording import NeuronRecorder, PresynapseRecorder


@dataclass
class SimulationEngine:
    neurons: list[Neuron]
    dt: float = 1e-4
    t: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    presynapse_recorders: list[PresynapseRecorder] = field(default_factory=list)
    neuron_recorders: list[NeuronRecorder] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def record_presynapse(self, presynapse) -> PresynapseRecorder:
        rec = PresynapseRecorder(presynapse=presynapse)
        self.presynapse_recorders.append(rec)
        return rec

    def record_neuron(self, neuron: Neuron) -> NeuronRecorder:
        rec = NeuronRecorder(neuron=neuron)
        self.neuron_recorders.append(rec)
        return rec

    def step(self) -> None:
        t0, t1 = self.t, self.t + self.dt

        # 1) fire any scheduled spikes in [t0, t1) so Ca is kicked before the step advances
        for n in self.neurons:
            for t_spike in n.pending_spikes(t0, t1):
                n.fire(t_spike)

        # 2) clear observer input currents so the recorder captures a per-step trace
        for n in self.neurons:
            n.input_current = 0.0

        # 3) advance every presynapse and dispatch quanta
        per_presynapse_release: dict[int, float] = {}
        for n in self.neurons:
            for p in n.presynapses:
                released = p.step(self.dt, self.rng)
                per_presynapse_release[id(p)] = released
                if released > 0.0 and p.targets:
                    for target, weight in p.targets:
                        target.receive(released * weight, t1)

        # 4) record
        self.times.append(t1)
        for rec in self.presynapse_recorders:
            rec.sample(per_presynapse_release.get(id(rec.presynapse), 0.0))
        for rec in self.neuron_recorders:
            rec.sample()

        self.t = t1

    def run(self, t_end: float) -> None:
        while self.t + self.dt <= t_end + 1e-12:
            self.step()

    def time_array(self) -> np.ndarray:
        return np.asarray(self.times)
