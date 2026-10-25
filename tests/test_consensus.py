"""Tests for `tgtrack.consensus` — the E071 decision rule.

These tests exist to pin the ONE property the mechanism is built on: a challenger must be
consistent *in identity*, not merely in position, before it may take the tracked identity. The
campaign spent 31 pruned branches learning that metrics alone cannot tell "mechanism fired and
helped" from "mechanism never fired"; so every test here asserts both the decision and the
counter that proves the code path ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.consensus import ChallengeState, ChallengeTracker

BOX = np.array([100.0, 100.0, 160.0, 300.0])
SHIFTED = BOX + np.array([2.0, 0.0, 2.0, 0.0])  # IoU ~0.95 with BOX
FAR = BOX + np.array([300.0, 0.0, 300.0, 0.0])  # IoU 0
FEAT_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
FEAT_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------------- consensus
def test_two_consistent_frames_commit_once():
    ct = ChallengeTracker(frames_required=2, iou_thresh=0.3, ident_thresh=0.35)
    r1 = ct.observe(10, BOX, FEAT_A, incumbent=False)
    assert r1.state == ChallengeState.PENDING and r1.pending_n == 1

    r2 = ct.observe(11, SHIFTED, FEAT_A, incumbent=False)
    assert r2.state == ChallengeState.COMMITTED
    assert ct.n_opened == 1 and ct.n_committed == 1
    assert not ct.active
    # commit clears the chain, so the next frame opens a fresh one
    r3 = ct.observe(12, SHIFTED, FEAT_A, incumbent=False)
    assert r3.state == ChallengeState.PENDING


def test_single_frame_control_commits_immediately():
    ct = ChallengeTracker(frames_required=1)
    rep = ct.observe(0, BOX, FEAT_A, incumbent=False)
    assert rep.state == ChallengeState.COMMITTED and ct.n_committed == 1


def test_three_frame_vote_needs_three():
    ct = ChallengeTracker(frames_required=3)
    states = [ct.observe(i, SHIFTED, FEAT_A, incumbent=False).state for i in range(5)]
    assert states[:2] == [ChallengeState.PENDING, ChallengeState.PENDING]
    assert states[2] == ChallengeState.COMMITTED
    assert states[3] == ChallengeState.PENDING  # new chain after the commit


# --------------------------------------------------------------------------- the trap
def test_two_different_people_at_the_same_place_never_commit():
    """Stable boxes + disagreeing appearance must NOT be read as agreement.

    This is the exact E061 failure mode (a one-or-two-frame tie between two people at the same
    screen position). A box-only vote certifies it; the identity leg must refuse it.
    """
    ct = ChallengeTracker(frames_required=2, iou_thresh=0.3, ident_thresh=0.35)
    a_first = ct.observe(0, BOX, FEAT_A, incumbent=False)
    assert a_first.state == ChallengeState.PENDING
    rep = ct.observe(1, BOX, FEAT_B, incumbent=False)  # same box, different person
    assert rep.reason.startswith("identity_broke")
    assert ct.n_committed == 0
    assert ct.n_restarted == 1
    assert rep.pending_n == 1  # the chain restarted on this frame


def test_box_only_vote_is_available_as_the_control_arm():
    """With the identity leg disabled the same input DOES commit — proving which leg stops it."""
    ct = ChallengeTracker(frames_required=2, iou_thresh=0.3, ident_thresh=float("inf"))
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    rep = ct.observe(1, BOX, FEAT_B, incumbent=False)
    assert rep.state == ChallengeState.COMMITTED and ct.n_committed == 1


def test_spatially_discontinuous_proposal_restarts_the_chain():
    ct = ChallengeTracker(frames_required=2, iou_thresh=0.3)
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    rep = ct.observe(1, FAR, FEAT_A, incumbent=False)
    assert rep.reason.startswith("link_broke")
    assert ct.n_committed == 0
    assert rep.pending_n == 1


# --------------------------------------------------------------------------- abandonment
def test_one_frame_blip_is_abandoned_after_max_age():
    ct = ChallengeTracker(frames_required=2, max_age=3)
    ct.observe(100, BOX, FEAT_A, incumbent=False)
    assert ct.tick(101).state == ChallengeState.PENDING
    rep = ct.tick(104)  # 4 frames since the proposal > max_age
    assert rep.state == ChallengeState.IDLE
    assert ct.n_abandoned == 1 and ct.n_committed == 0
    assert "abandoned_after_4f" in rep.reason


def test_incumbent_winning_kills_an_open_challenge():
    ct = ChallengeTracker(frames_required=3)
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    rep = ct.observe(1, BOX, FEAT_A, incumbent=True)
    assert rep.state == ChallengeState.IDLE
    assert ct.n_abandoned == 1 and not ct.active
    assert "incumbent_wins_after_1f" in rep.reason


def test_gap_larger_than_max_age_restarts_instead_of_chaining():
    ct = ChallengeTracker(frames_required=2, max_age=2)
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    rep = ct.observe(50, BOX, FEAT_A, incumbent=False)  # 50-frame gap
    assert ct.n_abandoned == 1 and ct.n_committed == 0
    assert rep.pending_n == 1


# --------------------------------------------------------------------------- pooling
def test_pooled_descriptor_is_the_normalised_mean():
    ct = ChallengeTracker(frames_required=5, pool=True)
    # near-identical descriptors, so the identity leg lets the chain grow (that is the normal
    # case for a real person over consecutive frames; the disagreement case is tested above)
    f1 = _unit([1.0, 0.0, 0.0, 0.0])
    f2 = _unit([0.9, 0.1, 0.0, 0.0])
    ct.observe(0, BOX, f1, incumbent=False)
    rep = ct.observe(1, BOX, f2, incumbent=False)
    assert rep.pending_n == 2, "the chain must still be open for pooling to mean anything"
    pooled = rep.pooled
    assert pooled is not None
    assert pooled.shape == (4,)
    assert abs(float(np.linalg.norm(pooled)) - 1.0) < 1e-6
    expected = _unit([f1 + f2][0])
    assert np.allclose(pooled, expected, atol=1e-6)


def test_pool_disabled_and_no_features_yield_none():
    assert ChallengeTracker(pool=False).observe(0, BOX, FEAT_A).pooled is None
    assert ChallengeTracker(pool=True).observe(0, BOX, None).pooled is None


# --------------------------------------------------------------------------- bookkeeping
def test_stats_expose_every_counter():
    ct = ChallengeTracker(frames_required=2)
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    ct.observe(1, BOX, FEAT_B, incumbent=False)   # identity break -> restart
    ct.observe(2, BOX, FEAT_B, incumbent=False)   # commit
    stats = ct.stats()
    assert stats["challenge_opened"] == 2
    assert stats["challenge_restarted"] == 1
    assert stats["challenge_committed"] == 1
    assert stats["challenge_abandoned"] == 0
    assert set(stats) == {
        "challenge_opened", "challenge_committed", "challenge_abandoned",
        "challenge_restarted", "challenge_pending_frames",
    }


def test_as_diag_is_json_safe():
    rep = ChallengeTracker(frames_required=3).observe(0, BOX, FEAT_A, incumbent=False)
    diag = rep.as_diag()
    assert diag["challenge_state"] == ChallengeState.PENDING
    assert diag["pending_n"] == 1 and diag["pending_agreed"] == 1
    assert isinstance(diag["challenge_reason"], str)


def test_reset_drops_the_chain_without_counting_it():
    ct = ChallengeTracker(frames_required=3)
    ct.observe(0, BOX, FEAT_A, incumbent=False)
    ct.reset()
    assert not ct.active and ct.n_abandoned == 0


def test_degenerate_inputs_are_ignored_not_fatal():
    ct = ChallengeTracker(frames_required=2)
    ct.observe(0, [np.nan, 0, 1, 1], FEAT_A, incumbent=False)
    ct.observe(1, [1.0, 2.0], FEAT_A, incumbent=False)
    ct.observe(2, BOX, np.array([0.0, 0.0, 0.0, 0.0]), incumbent=False)  # zero-norm feat
    assert ct.active or ct.n_committed >= 0  # never raises


@pytest.mark.parametrize("kwargs", [
    {"frames_required": -1},
    {"iou_thresh": -0.1},
    {"iou_thresh": 1.5},
    {"max_age": -1},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChallengeTracker(**kwargs)
