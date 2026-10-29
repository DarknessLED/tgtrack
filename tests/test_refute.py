"""E095 -- bounded-lag refutation: the pass revises excursions and nothing else."""
from __future__ import annotations

import numpy as np

from tgtrack.config import Config
from tgtrack.refute import consensus_index, refute_statuses
from tgtrack.target_tracker import TargetTracker
from tgtrack.types import TargetStatus


def _status(frame: int, box) -> TargetStatus:
    return TargetStatus(
        frame_idx=frame,
        state="tracking",
        xyxy=None if box is None else np.asarray(box, float),
    )


def _cands(*boxes):
    return [list(map(float, b)) for b in boxes]


def test_default_is_off_and_records_nothing():
    tt = TargetTracker(Config())
    assert tt.refute_candidates() == {}
    assert int(Config().state.decision_lag) == 0


def test_knob_order_is_read_from_state():
    tt = TargetTracker(Config().clone(**{"state.decision_lag": 8, "state.refute_window": 6}))
    assert tt._refute_on is True
    assert int(tt.cfg.state.refute_window) == 6


def test_isolated_excursion_is_reattributed():
    # frames 0..4 track box A; frame 2 is an excursion onto B, then back to A.
    a = [0.0, 0.0, 10.0, 10.0]
    b = [40.0, 40.0, 50.0, 50.0]
    statuses = {
        0: _status(0, a),
        1: _status(1, a),
        2: _status(2, b),
        3: _status(3, a),
        4: _status(4, a),
    }
    cands = {i: _cands(a, b) for i in range(5)}
    out = refute_statuses(statuses, cands, window=4)
    assert np.allclose(out[2].xyxy, a)
    assert out[2].diag.get("refuted_from") == b
    # the agreeing frames are untouched (same objects)
    for i in (0, 1, 3, 4):
        assert out[i] is statuses[i]


def test_report_count_is_preserved():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [40.0, 40.0, 50.0, 50.0]
    statuses = {i: _status(i, b if i == 2 else a) for i in range(5)}
    statuses[7] = _status(7, None)
    cands = {i: _cands(a, b) for i in range(5)}
    out = refute_statuses(statuses, cands, window=4)
    assert out[7].xyxy is None
    assert sum(1 for st in out.values() if st.xyxy is not None) == 5


def test_no_consensus_leaves_statuses_identical():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [40.0, 40.0, 50.0, 50.0]
    statuses = {i: _status(i, a) for i in range(5)}
    cands = {i: _cands(a, b) for i in range(5)}
    out = refute_statuses(statuses, cands, window=4)
    assert out is statuses  # nothing revised -> the same mapping is returned


def test_missing_candidates_disable_the_pass():
    statuses = {i: _status(i, [0.0, 0.0, 10.0, 10.0]) for i in range(4)}
    assert refute_statuses(statuses, {}, window=4) is statuses
    assert refute_statuses({}, {0: _cands([0, 0, 1, 1])}, window=4) == {}


def test_min_share_rejects_thin_agreement():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [40.0, 40.0, 50.0, 50.0]
    c = [80.0, 80.0, 90.0, 90.0]
    reports = {0: np.asarray(a), 1: np.asarray(b), 2: np.asarray(c)}
    cands = {i: _cands(a, b, c) for i in range(3)}
    # position 0 skipped: the remaining window votes once for b and once for c
    assert consensus_index([0, 1, 2], cands, reports, 0, window=8, min_share=0.75) is None


def test_arm_contract_in_run_target():
    from scripts.run_target import build_variants

    names = {v.name: v for v in build_variants("refute")}
    assert "lag_refute_L8" in names and "lag_refute_L0" in names
    assert float(names["lag_refute_L8"].overrides["state.decision_lag"]) == 8.0
    assert float(names["lag_refute_L0"].overrides["state.decision_lag"]) == 0.0
    for name, v in names.items():
        if name.startswith("lag_refute"):
            assert v.overrides.get("cost.anchor_vote") == "off"
