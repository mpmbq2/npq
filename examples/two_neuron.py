"""Two-neuron example: a presynaptic neuron firing a short spike train
drives an observer neuron through a single biophysical presynapse.

Run:
    uv run python examples/two_neuron.py
(or set PLOT=1 to display the figure)
"""
from __future__ import annotations

import os

import numpy as np

from npq import (
    CalciumTransient,
    DockingSite,
    Neuron,
    Presynapse,
    SimulationEngine,
    Vesicle,
    hill_rate,
)


def build() -> tuple[SimulationEngine, Presynapse, Neuron]:
    pre = Neuron(name="pre", spike_times=[0.010, 0.030, 0.050, 0.070])
    post = Neuron(name="post")

    sites = [DockingSite() for _ in range(5)]
    vesicles = [
        Vesicle(
            docking_rate=200.0,
            fusion_rate=hill_rate(k_max=800.0, K=2.0, n=4.0),
            quantal_size=1.0,
            recovery_rate=5.0,
        )
        for _ in range(20)
    ]
    syn = Presynapse(
        sites=sites,
        vesicles=vesicles,
        calcium=CalciumTransient(baseline=0.05, tau=0.01, delta=6.0),
    )
    syn.add_target(post, weight=1.0)
    pre.add_presynapse(syn)

    engine = SimulationEngine(
        neurons=[pre, post],
        dt=1e-4,
        rng=np.random.default_rng(0),
    )
    return engine, syn, post


def main() -> None:
    engine, syn, post = build()
    syn_rec = engine.record_presynapse(syn)
    post_rec = engine.record_neuron(post)
    engine.run(t_end=0.15)

    t = engine.time_array()
    s = syn_rec.as_arrays()
    p = post_rec.as_arrays()

    print(f"total quanta released:       {s['released'].sum():.1f}")
    print(f"peak calcium:                {s['calcium'].max():.3f}")
    print(f"mean docked count:           {s['docked'].mean():.2f}")
    print(f"total observer input charge: {p['input_current'].sum() * engine.dt:.4f}")

    if os.environ.get("PLOT"):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
        axes[0].plot(t, s["calcium"]); axes[0].set_ylabel("[Ca] (µM)")
        axes[1].plot(t, s["docked"]); axes[1].set_ylabel("docked")
        axes[2].plot(t, s["released"]); axes[2].set_ylabel("released / step")
        axes[3].plot(t, p["input_current"]); axes[3].set_ylabel("post I")
        axes[3].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
