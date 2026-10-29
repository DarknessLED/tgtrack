"""E092: hold a lost-regime re-acquisition until the same candidate persists.

The invariant that matters: at the default (1) the tracker never defers anything, so the
shipped path is bit-identical; when armed, a candidate that passes every existing gate is
still refused for identity purposes until it has been offered on N consecutive frames with
IoU >= ``reacquire_confirm_iou`` against the previous provisional box.
"""
from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import Config
from tgtrack.target_tracker import TargetState, TargetTracker
from tgtrack.types import Detection


def _tracker(**overrides) -> TargetTracker:
    return TargetTracker(Config().clone(**{"cost.anchor_vote": "off", **overrides}))


def _det(x0: float, y0: float = 10.0, size: float = 40.0) -> Detection:
    return Detection(
        xyxy=np.array([x0, y0, x0 + size, y0 + size], np.float32),
        score=0.9,
    )


def _lost(tt: TargetTracker) -> TargetState:
    """A state the tracker treats as blind (re-acquisition territory)."""
    return TargetState.LOST_SHORT


def test_default_is_one_and_never_defers():
    tt = _tracker()
    assert int(tt.cfg.state.reacquire_confirm_frames) == 1
    lost = _lost(tt)
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is True
    assert tt._deferred_frame is False
    assert tt._pending_hits == 0


def test_armed_defers_the_first_candidate_then_commits():
    tt = _tracker(**{"state.reacquire_confirm_frames": 2})
    lost = _lost(tt)
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False
    assert tt._deferred_frame is True
    assert tt._pending_hits == 1
    # The same candidate persists -> commit.
    tt._deferred_frame = False
    assert tt._provisional_reacquire_ok(_det(11.0), lost) is True
    assert tt._pending_box is None
    assert tt._pending_hits == 0


def test_a_different_candidate_restarts_the_count():
    tt = _tracker(**{"state.reacquire_confirm_frames": 3})
    lost = _lost(tt)
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False
    assert tt._pending_hits == 1
    assert tt._provisional_reacquire_ok(_det(300.0), lost) is False   # IoU 0 -> new candidate
    assert tt._pending_hits == 1
    assert float(tt._pending_box[0]) == pytest.approx(300.0)
    assert tt._provisional_reacquire_ok(_det(301.0), lost) is False
    assert tt._pending_hits == 2
    tt._deferred_frame = False
    assert tt._provisional_reacquire_ok(_det(302.0), lost) is True    # third consecutive


def test_ordinary_tracking_is_never_deferred():
    tt = _tracker(**{"state.reacquire_confirm_frames": 3})
    # A non-blind previous state means the target was being tracked: commit immediately.
    assert tt._provisional_reacquire_ok(_det(10.0), TargetState.TRACKING) is True
    assert tt._pending_hits == 0


def test_iou_threshold_is_honoured():
    tt = _tracker(**{"state.reacquire_confirm_frames": 2,
                     "state.reacquire_confirm_iou": 0.95})
    lost = _lost(tt)
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False
    # Shifted 10 px of 40 -> IoU ~0.64 < 0.95, so this is a *new* candidate.
    assert tt._provisional_reacquire_ok(_det(20.0), lost) is False
    assert tt._pending_hits == 1
    tt._deferred_frame = False
    assert tt._provisional_reacquire_ok(_det(21.0), lost) is True


def test_arm_contract():
    from scripts.run_target import build_variants

    variants = {v.name: v for v in build_variants("reacquire")}
    assert {"reacq_confirm_2", "reacq_confirm_3", "reacq_confirm_1"} <= set(variants)
    assert variants["reacq_confirm_1"].overrides.get("state.reacquire_confirm_frames") == 1
    for name in ("reacq_confirm_2", "reacq_confirm_3"):
        assert variants[name].overrides.get("cost.anchor_vote") == "off"

# --------------------------------------------------------------------------- #
# E092b -- gap tolerance
# --------------------------------------------------------------------------- #
def test_window_zero_is_the_strict_streak():
    tt = _tracker(**{"state.reacquire_confirm_frames": 2, "state.reacquire_confirm_window": 0})
    lost = _lost(tt)
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False
    tt._deferred_frame = False
    assert tt._provisional_reacquire_ok(_det(300.0), lost) is False   # a distant candidate
    tt._deferred_frame = False
    # ... resets the streak, so two frames of the new candidate commit, not one.
    assert tt._provisional_reacquire_ok(_det(301.0), lost) is True


def test_window_tolerates_one_gap():
    tt = _tracker(**{"state.reacquire_confirm_frames": 2, "state.reacquire_confirm_window": 4})
    lost = _lost(tt)
    tt.frame_idx = 100
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False       # 1 agreeing
    tt._deferred_frame = False
    tt.frame_idx = 101
    assert tt._provisional_reacquire_ok(_det(400.0), lost) is False      # distractor, no agree
    tt._deferred_frame = False
    tt.frame_idx = 102
    assert tt._provisional_reacquire_ok(_det(11.0), lost) is True        # 2 of 3 agree -> commit


def test_window_requires_n_agreeing_candidates():
    tt = _tracker(**{"state.reacquire_confirm_frames": 3, "state.reacquire_confirm_window": 6})
    lost = _lost(tt)
    for i, box_x in enumerate((10.0, 11.0)):
        tt.frame_idx = 200 + i
        assert tt._provisional_reacquire_ok(_det(box_x), lost) is False
        tt._deferred_frame = False
    # Two agreeing candidates are not enough for N = 3.
    assert tt._pending_hits == 2
    tt.frame_idx = 202
    assert tt._provisional_reacquire_ok(_det(12.0), lost) is True


def test_window_falls_out_of_the_window():
    tt = _tracker(**{"state.reacquire_confirm_frames": 2, "state.reacquire_confirm_window": 3})
    lost = _lost(tt)
    tt.frame_idx = 300
    assert tt._provisional_reacquire_ok(_det(10.0), lost) is False
    tt._deferred_frame = False
    # Two frames later the first candidate is still inside a 3-frame window (300, 302).
    tt.frame_idx = 302
    assert tt._provisional_reacquire_ok(_det(11.0), lost) is True
