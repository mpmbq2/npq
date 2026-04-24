import math

import pytest

from npq.calcium import CalciumTransient


def test_initial_value_defaults_to_baseline():
    ca = CalciumTransient(baseline=0.1, tau=0.01, delta=2.0)
    assert ca.value == pytest.approx(0.1)


def test_kick_adds_delta():
    ca = CalciumTransient(baseline=0.1, tau=0.01, delta=2.0)
    ca.kick()
    assert ca.value == pytest.approx(2.1)


def test_decay_returns_to_baseline_exponentially():
    ca = CalciumTransient(baseline=0.1, tau=0.01, delta=2.0)
    ca.kick()
    start = ca.value
    ca.decay(dt=0.01)  # one tau
    expected = 0.1 + (start - 0.1) * math.exp(-1.0)
    assert ca.value == pytest.approx(expected, rel=1e-9)


def test_long_decay_reaches_baseline():
    ca = CalciumTransient(baseline=0.1, tau=0.01, delta=5.0)
    ca.kick()
    ca.decay(dt=1.0)  # 100 tau
    assert ca.value == pytest.approx(0.1, abs=1e-9)
