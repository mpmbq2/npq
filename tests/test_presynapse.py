from __future__ import annotations

import numpy as np
import pytest

from npq.calcium import CalciumTransient
from npq.docking import DockingSite
from npq.presynapse import Presynapse
from npq.release import hill_rate
from npq.vesicle import Vesicle, VesicleState


def _build(n_sites: int, n_vesicles: int, k_max: float = 0.0) -> Presynapse:
    sites = [DockingSite() for _ in range(n_sites)]
    vesicles = [
        Vesicle(
            docking_rate=1000.0,  # fast docking so RRP fills quickly
            fusion_rate=hill_rate(k_max=k_max, K=2.0, n=4.0),
            quantal_size=1.0,
            recovery_rate=10.0,
        )
        for _ in range(n_vesicles)
    ]
    return Presynapse(
        sites=sites,
        vesicles=vesicles,
        calcium=CalciumTransient(baseline=0.05, tau=0.02, delta=5.0),
    )


def test_docking_fills_sites_over_time():
    rng = np.random.default_rng(0)
    p = _build(n_sites=5, n_vesicles=10, k_max=0.0)
    for _ in range(200):
        p.step(dt=1e-3, rng=rng)
    assert p.docked_count == 5  # RRP saturates at N_sites


def test_no_fusion_at_baseline_calcium():
    rng = np.random.default_rng(0)
    p = _build(n_sites=5, n_vesicles=10, k_max=1e5)
    # Let RRP fill
    for _ in range(200):
        p.step(dt=1e-3, rng=rng)
    assert p.docked_count == 5
    # Now simulate for a long time with no spikes. Baseline Ca is tiny,
    # so Hill rate is ~0 and no fusion should happen.
    total_released = 0.0
    for _ in range(1000):
        total_released += p.step(dt=1e-3, rng=rng)
    assert total_released == 0.0
    assert p.docked_count == 5


def test_spike_drives_release_via_calcium():
    rng = np.random.default_rng(42)
    p = _build(n_sites=5, n_vesicles=20, k_max=500.0)
    # fill RRP
    for _ in range(500):
        p.step(dt=1e-4, rng=rng)
    assert p.docked_count == 5

    # one spike, then watch the Ca transient drive release over ~5 tau
    p.on_spike(t=0.0)
    released = 0.0
    for _ in range(1500):
        released += p.step(dt=1e-4, rng=rng)
    # With N=5 docked and a strong transient we expect several quanta released.
    assert released > 0.0


def test_spike_only_kicks_calcium_not_fusion():
    rng = np.random.default_rng(0)
    p = _build(n_sites=5, n_vesicles=10, k_max=1.0)
    ca_before = p.calcium.value
    p.on_spike(t=0.0)
    # on_spike must not release; it just raises Ca.
    assert p.calcium.value > ca_before
    # No vesicle transitioned to FUSED just from the spike.
    assert p.fused_count == 0


def test_recovery_returns_fused_to_free():
    rng = np.random.default_rng(1)
    p = _build(n_sites=2, n_vesicles=4, k_max=0.0)
    # Manually put a vesicle into FUSED state
    v = p.vesicles[0]
    v.state = VesicleState.FUSED
    # With recovery_rate=10/s and dt=0.1s over many steps, it should recover.
    for _ in range(2000):
        p.step(dt=1e-3, rng=rng)
    assert p.fused_count == 0
