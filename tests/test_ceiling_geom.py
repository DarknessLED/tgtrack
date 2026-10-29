"""E091: the geometry-conditional lost-regime ceiling widening.

The invariant that matters: at the default knob the tracker is bit-identical to the
shipped path, and when the knob is armed the wider limit applies ONLY to candidates that
sit inside the motion gate (``norm_centre <= tau``).  The tests drive ``_lost_geom_limit``
directly with a fake breakdown so they do not need a detector.
"""
from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import Config
from tgtrack.target_tracker import TargetTracker

BAND = 0.40


class _Break:
    """Minimal stand-in for a CostBreakdown: only ``norm_centre`` matters here."""

    def __init__(self, norm_centre):
        self.norm_centre = np.asarray(norm_centre, np.float64)


def _tracker(**overrides) -> TargetTracker:
    cfg = Config().clone(**{"cost.anchor_max_dist": BAND, **overrides})
    return TargetTracker(cfg)


def test_default_geom_scale_is_inert():
    tt = _tracker()
    assert float(tt.cfg.cost.anchor_band_lost_geom_scale) == 1.0
    tt.frames_since_update = 500
    out = tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([0.1, 0.9]))
    assert out == BAND  # scalar, exactly the shipped limit
    assert "anchor_band_lost_geom_scale" not in tt._last_diag


def test_inert_before_the_lost_threshold():
    tt = _tracker(**{"cost.anchor_band_lost_geom_scale": 2.5})
    tt.frames_since_update = 0
    out = tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([0.1, 0.1]))
    assert out == BAND


def test_widens_only_the_candidates_inside_the_gate():
    tt = _tracker(**{"cost.anchor_band_lost_geom_scale": 2.5})
    tt.frames_since_update = 500
    out = tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([0.2, 3.0, 1.5]))
    arr = np.asarray(out, np.float64)
    assert arr.shape == (3,)
    assert arr[0] == pytest.approx(BAND * 2.5)   # inside the gate -> widened
    assert arr[1] == pytest.approx(BAND)         # far outside -> shipped limit
    assert arr[2] == pytest.approx(BAND)         # 1.5 > tau = 1.0 -> shipped limit
    assert tt._last_diag["anchor_band_lost_geom_n"] == 1


def test_tau_is_honoured():
    tt = _tracker(**{"cost.anchor_band_lost_geom_scale": 2.0,
                     "cost.anchor_band_lost_geom_tau": 2.0})
    tt.frames_since_update = 500
    arr = np.asarray(tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([1.5, 2.5])), np.float64)
    assert arr[0] == pytest.approx(BAND * 2.0)
    assert arr[1] == pytest.approx(BAND)


def test_no_near_candidate_returns_the_scalar():
    tt = _tracker(**{"cost.anchor_band_lost_geom_scale": 2.5})
    tt.frames_since_update = 500
    out = tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([5.0, 9.0]))
    assert out == BAND
    assert not isinstance(out, np.ndarray)


def test_none_limit_stays_none_and_missing_geometry_is_safe():
    tt = _tracker(**{"cost.anchor_band_lost_geom_scale": 2.5})
    tt.frames_since_update = 500
    assert tt._lost_geom_limit(tt.cfg.cost, None, _Break([0.1])) is None
    assert tt._lost_geom_limit(tt.cfg.cost, BAND, _Break([])) == BAND
    assert tt._lost_geom_limit(tt.cfg.cost, BAND, object()) == BAND


def test_arm_contract():
    from scripts.run_target import build_variants

    variants = {v.name: v for v in build_variants("ceiling_geom")}
    assert {"geom_widen_1.50", "geom_widen_2.50", "geom_widen_1.00"} <= set(variants)
    control = variants["geom_widen_1.00"]
    assert control.overrides.get("cost.anchor_band_lost_geom_scale") == 1.0
    for name in ("geom_widen_1.50", "geom_widen_2.50"):
        assert variants[name].overrides.get("cost.anchor_vote") == "off"
