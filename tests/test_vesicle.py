from npq.vesicle import Vesicle, VesicleState


def _make_vesicle() -> Vesicle:
    return Vesicle(
        docking_rate=10.0,
        fusion_rate=lambda ca: 0.0,
        quantal_size=1.0,
        recovery_rate=5.0,
    )


def test_defaults_start_free():
    v = _make_vesicle()
    assert v.state is VesicleState.FREE
    assert v.is_free
    assert not v.is_docked
    assert not v.is_fused
    assert v.site is None


def test_state_flags_cover_all_states():
    v = _make_vesicle()
    v.state = VesicleState.DOCKED
    assert v.is_docked and not v.is_free and not v.is_fused
    v.state = VesicleState.FUSED
    assert v.is_fused and not v.is_free and not v.is_docked
