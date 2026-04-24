from __future__ import annotations

import numpy as np

from npq.calcium import CalciumTransient
from npq.docking import DockingSite
from npq.engine import SimulationEngine
from npq.neuron import Neuron
from npq.presynapse import Presynapse
from npq.release import hill_rate
from npq.vesicle import Vesicle


def _build_two_neuron_pair(seed: int = 0):
    pre = Neuron(name="pre", spike_times=[0.005, 0.025, 0.045])
    post = Neuron(name="post")

    sites = [DockingSite() for _ in range(5)]
    vesicles = [
        Vesicle(
            docking_rate=500.0,
            fusion_rate=hill_rate(k_max=800.0, K=2.0, n=4.0),
            quantal_size=1.0,
            recovery_rate=5.0,
        )
        for _ in range(15)
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
        rng=np.random.default_rng(seed),
    )
    return engine, pre, post, syn


def test_presynaptic_spike_delivers_current_to_observer():
    engine, pre, post, syn = _build_two_neuron_pair(seed=0)
    syn_rec = engine.record_presynapse(syn)
    post_rec = engine.record_neuron(post)
    engine.run(t_end=0.1)

    arrs = post_rec.as_arrays()
    # Observer should see a non-trivial amount of input current over the run.
    assert float(arrs["input_current"].sum()) > 0.0

    syn_arrs = syn_rec.as_arrays()
    # Ca trace should have large peaks near each spike time
    assert syn_arrs["calcium"].max() > 1.0
    # Some release events should have been recorded
    assert float(syn_arrs["released"].sum()) > 0.0


def test_no_spikes_means_no_release_and_no_input_current():
    pre = Neuron(name="pre", spike_times=[])
    post = Neuron(name="post")
    sites = [DockingSite() for _ in range(3)]
    vesicles = [
        Vesicle(
            docking_rate=500.0,
            fusion_rate=hill_rate(k_max=1e4, K=2.0, n=4.0),
            quantal_size=1.0,
            recovery_rate=5.0,
        )
        for _ in range(6)
    ]
    syn = Presynapse(
        sites=sites,
        vesicles=vesicles,
        calcium=CalciumTransient(baseline=0.05, tau=0.01, delta=6.0),
    )
    syn.add_target(post, weight=1.0)
    pre.add_presynapse(syn)
    engine = SimulationEngine(neurons=[pre, post], dt=1e-4, rng=np.random.default_rng(0))
    post_rec = engine.record_neuron(post)
    engine.run(t_end=0.05)
    assert float(post_rec.as_arrays()["input_current"].sum()) == 0.0


def test_release_scales_with_weight():
    e1, _, post1, _ = _build_two_neuron_pair(seed=7)
    e2, _, post2, syn2 = _build_two_neuron_pair(seed=7)
    syn2.targets = [(post2, 2.0)]  # double the weight

    rec1 = e1.record_neuron(post1)
    rec2 = e2.record_neuron(post2)
    e1.run(t_end=0.1)
    e2.run(t_end=0.1)

    total1 = float(rec1.as_arrays()["input_current"].sum())
    total2 = float(rec2.as_arrays()["input_current"].sum())
    # Same RNG seed + same release sequence -> exactly 2x current.
    assert total2 == 2 * total1
    assert total1 > 0.0


def test_multiple_targets_each_receive_quanta():
    pre = Neuron(name="pre", spike_times=[0.005])
    a = Neuron(name="a")
    b = Neuron(name="b")
    sites = [DockingSite() for _ in range(4)]
    vesicles = [
        Vesicle(
            docking_rate=500.0,
            fusion_rate=hill_rate(k_max=800.0, K=2.0, n=4.0),
            quantal_size=1.0,
            recovery_rate=5.0,
        )
        for _ in range(10)
    ]
    syn = Presynapse(
        sites=sites,
        vesicles=vesicles,
        calcium=CalciumTransient(baseline=0.05, tau=0.01, delta=6.0),
    )
    syn.add_target(a, weight=1.0)
    syn.add_target(b, weight=0.5)
    pre.add_presynapse(syn)

    engine = SimulationEngine(neurons=[pre, a, b], dt=1e-4, rng=np.random.default_rng(3))
    rec_a = engine.record_neuron(a)
    rec_b = engine.record_neuron(b)
    engine.run(t_end=0.05)

    ia = float(rec_a.as_arrays()["input_current"].sum())
    ib = float(rec_b.as_arrays()["input_current"].sum())
    assert ia > 0.0
    assert ib == 0.5 * ia
