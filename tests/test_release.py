import math

import pytest

from npq.release import hill_rate


def test_hill_rate_zero_at_zero_ca():
    k = hill_rate(k_max=1000.0, K=2.0, n=4.0)
    assert k(0.0) == 0.0
    assert k(-1.0) == 0.0


def test_hill_rate_saturates_to_k_max():
    k_max = 1000.0
    k = hill_rate(k_max=k_max, K=2.0, n=4.0)
    assert k(1e6) == pytest.approx(k_max, rel=1e-6)


def test_hill_rate_monotonic_in_ca():
    k = hill_rate(k_max=1000.0, K=2.0, n=4.0)
    xs = [0.1, 0.5, 1.0, 2.0, 5.0, 20.0]
    ys = [k(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))


def test_hill_rate_half_max_at_K():
    K = 2.0
    k = hill_rate(k_max=1000.0, K=K, n=4.0)
    assert k(K) == pytest.approx(500.0, rel=1e-9)


def test_hill_rate_invalid_params():
    with pytest.raises(ValueError):
        hill_rate(k_max=-1.0)
    with pytest.raises(ValueError):
        hill_rate(k_max=1.0, K=0.0)
    with pytest.raises(ValueError):
        hill_rate(k_max=1.0, n=0.0)


def test_hill_rate_cooperativity_makes_slope_sharper():
    k4 = hill_rate(k_max=1.0, K=2.0, n=4.0)
    k1 = hill_rate(k_max=1.0, K=2.0, n=1.0)
    # At ca just below K, higher n means a smaller fraction activated.
    assert k4(1.0) < k1(1.0)
    # And at ca just above K, higher n means a larger fraction activated.
    assert k4(4.0) > k1(4.0)
