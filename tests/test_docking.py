import pytest

from npq.docking import DockingSite
from npq.vesicle import Vesicle, VesicleState


def _make_vesicle(state: VesicleState = VesicleState.FREE) -> Vesicle:
    v = Vesicle(
        docking_rate=10.0,
        fusion_rate=lambda ca: 0.0,
        quantal_size=1.0,
        recovery_rate=5.0,
    )
    v.state = state
    return v


def test_bind_sets_state_and_links_both_directions():
    site = DockingSite()
    v = _make_vesicle()
    site.bind(v)
    assert site.occupant is v
    assert v.site is site
    assert v.state is VesicleState.DOCKED
    assert not site.is_empty


def test_release_clears_both_sides_and_returns_vesicle():
    site = DockingSite()
    v = _make_vesicle()
    site.bind(v)
    # mimic fusion wiping the state flag before release
    v.state = VesicleState.FUSED
    got = site.release()
    assert got is v
    assert site.occupant is None
    assert v.site is None


def test_cannot_double_occupy_a_site():
    site = DockingSite()
    site.bind(_make_vesicle())
    with pytest.raises(RuntimeError):
        site.bind(_make_vesicle())


def test_only_free_vesicles_may_dock():
    site = DockingSite()
    with pytest.raises(RuntimeError):
        site.bind(_make_vesicle(VesicleState.FUSED))


def test_release_empty_site_raises():
    site = DockingSite()
    with pytest.raises(RuntimeError):
        site.release()
