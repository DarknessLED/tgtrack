"""Tests for the target-association core (Lead-owned).

These pin down the behaviour the whole strategy rests on:
part affinity, gate adaptivity, distractor refusal, hard negatives, and the
partial-view matching preference.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.association import (
    PART_AFFINITY,
    appearance_cost_of_query,
    build_part_affinity,
    compute_costs,
    direction_cost,
    gate_radius,
    motion_cost,
    part_affinity,
    select_candidate,
    size_cost,
)
from tgtrack.config import CostCfg
from tgtrack.types import Detection, cosine_distance_matrix


# --------------------------------------------------------------------------- #
# Part affinity
# --------------------------------------------------------------------------- #


def test_part_affinity_identity_and_symmetry_properties():
    keys = ("full", "upper", "lower", "head", "torso")
    m = build_part_affinity(keys)
    assert m.shape == (5, 5)
    for i, k in enumerate(keys):
        assert m[i, i] == pytest.approx(1.0), k
    # upper/lower barely overlap -> low affinity; upper/full overlap a lot.
    assert m[keys.index("upper"), keys.index("full")] > 0.5
    assert m[keys.index("upper"), keys.index("lower")] < 0.2
    # 'torso' sits inside 'upper' and 'full'
    assert part_affinity("torso", "upper") > 0.5
    assert part_affinity("torso", "full") > 0.3
    assert part_affinity("head", "lower") < 0.2
    assert part_affinity("nonsense", "full") == 0.0
    assert PART_AFFINITY.shape == (len(PART_AFFINITY), len(PART_AFFINITY))


# --------------------------------------------------------------------------- #
# Motion / size / direction
# --------------------------------------------------------------------------- #


def test_motion_cost_prefers_overlap_and_is_bounded():
    pred = np.array([100, 100, 200, 300], dtype=np.float32)
    dets = np.array([[100, 100, 200, 300], [400, 400, 500, 600]], dtype=np.float32)
    cost, iou = motion_cost(pred, dets, gate_radius=500.0)
    assert cost.shape == (2,)
    assert 0.0 <= cost.min() and cost.max() <= 1.0
    assert cost[0] < cost[1]
    assert iou[0] == pytest.approx(1.0, abs=1e-5)
    assert iou[1] == pytest.approx(0.0, abs=1e-5)


def test_motion_cost_gives_gradient_under_no_overlap():
    """Fast motion: zero IoU must not mean zero information."""
    pred = np.array([0, 0, 10, 20], dtype=np.float32)
    near = np.array([[30, 0, 40, 20]], dtype=np.float32)   # no overlap, close
    far = np.array([[600, 0, 610, 20]], dtype=np.float32)  # no overlap, far
    cost, iou = motion_cost(pred, np.vstack([near, far]), gate_radius=1000.0)
    assert iou.max() == pytest.approx(0.0, abs=1e-6)
    assert cost[0] < cost[1], "center distance must still discriminate"


def test_gate_radius_grows_with_speed():
    still = gate_radius(np.zeros(4), base=0.1, speed_gain=2.5, frame_diag=1000.0)
    slow = gate_radius(np.array([1.0, 0.0, 0, 0]), base=0.1, speed_gain=2.5, frame_diag=1000.0)
    fast = gate_radius(np.array([80.0, 0.0, 0, 0]), base=0.1, speed_gain=2.5, frame_diag=1000.0)
    assert still == pytest.approx(100.0)
    assert slow == pytest.approx(100.0 + 2.5 * 1.0)
    assert fast == pytest.approx(100.0 + 2.5 * 80.0)
    assert fast > slow > still
    assert gate_radius(None, 0.1, 2.5, 1000.0) == pytest.approx(100.0)
    assert gate_radius(None, 0.0, 0.0, None) >= 1.0  # never degenerate


def test_size_cost_penalizes_scale_mismatch():
    pred = np.array([0, 0, 50, 200], dtype=np.float32)
    same = np.array([[0, 0, 50, 200]], dtype=np.float32)
    half = np.array([[0, 0, 50, 100]], dtype=np.float32)
    c = size_cost(pred, np.vstack([same, half]))
    assert c[0] == pytest.approx(0.0, abs=1e-6)
    assert c[1] > c[0]


def test_direction_cost_rewards_continuation_and_punishes_reversal():
    vel = np.array([10.0, 0.0, 0.0, 0.0])
    last = np.array([0, 0, 20, 40], dtype=np.float32)  # center (10, 20)
    forward = np.array([[30, 0, 50, 40]], dtype=np.float32)   # center (40, 20)
    backward = np.array([[-30, 0, -10, 40]], dtype=np.float32)  # center (-20, 20)
    c = direction_cost(vel, last, np.vstack([forward, backward]), weight=0.2)
    assert c[0] == pytest.approx(0.0, abs=1e-6)
    assert c[1] > 0.15
    # Weight scales the output monotonically.
    c2 = direction_cost(vel, last, np.vstack([forward, backward]), weight=0.4)
    assert c2[1] > c[1]
    # No history -> no opinion, and no crash.
    assert np.allclose(direction_cost(None, None, np.vstack([forward, backward])), 0.0)
    assert np.allclose(direction_cost(np.zeros(4), last, forward), 0.0)


# --------------------------------------------------------------------------- #
# Appearance: partial-view behaviour
# --------------------------------------------------------------------------- #


class _Entry:
    def __init__(self, part: str, feat: np.ndarray):
        self.part = part
        self.feat = feat
        self.quality = 1.0
        self.frame_idx = 0
        self.recency = 1.0


class _FakeGallery:
    """Minimal stand-in for LongTermGallery with controllable part banks."""

    def __init__(self, banks: dict[str, np.ndarray], hn: float = 0.0):
        self.banks = banks
        self.hn = hn

    def best(self, feat, part=None, frame_idx=None):
        if part not in self.banks:
            return float("inf"), None
        d = float(cosine_distance_matrix(np.asarray(feat, np.float32), self.banks[part])[0])
        return d, _Entry(part, self.banks[part][0])

    def hard_negative_penalty(self, feat):
        return self.hn


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_part_aware_query_prefers_the_matching_bank():
    """An upper-body-only crop must be scored against the `upper` bank.

    In the real system the `full` bank holds a whole-body template whose
    embedding genuinely differs from an upper-body crop (different framing,
    different background share). Declaring the query part correctly is what
    lets it reach the bank that actually matches.
    """
    e_upper = _unit([1.0, 0.0, 0.0, 0.0])
    e_full = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal => cosine distance 1.0
    gal = _FakeGallery({"upper": e_upper[None], "full": e_full[None]})
    cfg = CostCfg()

    # Correct declaration: this crop is the upper body.
    cost_ok, pair_ok, _ = appearance_cost_of_query({"upper": e_upper}, None, gal, cfg, 0)
    # Wrong declaration: treat the same crop as a whole body.
    cost_bad, pair_bad, _ = appearance_cost_of_query({"full": e_upper}, None, gal, cfg, 0)

    assert pair_ok == ("upper", "upper")
    assert pair_bad == ("full", "full")
    assert cost_ok == pytest.approx(0.0, abs=1e-5)
    assert cost_bad > cost_ok, "declaring the part must beat a generic full-body query"


def test_appearance_cost_neutral_without_features():
    cfg = CostCfg()
    cost, pair, hn = appearance_cost_of_query({}, None, None, cfg, 0)
    assert cost == pytest.approx(0.5)
    assert pair is None
    assert hn == 0.0


def test_hard_negative_penalty_is_surfaced():
    gal = _FakeGallery({}, hn=0.8)
    q = {"full": _unit([1, 0, 0, 0])}
    cost, _, hn = appearance_cost_of_query(q, _unit([1, 0, 0, 0]), gal, CostCfg(), 0)
    assert hn == pytest.approx(0.8)
    assert cost == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Cost assembly + selection
# --------------------------------------------------------------------------- #


def _det(box, score=0.9, feat=None, parts=None):
    d = Detection(xyxy=np.asarray(box, np.float32), score=score)
    d.feat = feat
    d.part_feats = parts or {}
    return d


def test_compute_costs_gates_out_far_candidates():
    cfg = CostCfg()
    pred = np.array([100, 100, 200, 300], np.float32)
    dets = [_det([100, 100, 200, 300]), _det([900, 900, 1000, 1100])]
    b = compute_costs(dets, pred, cfg, velocity=np.zeros(4, np.float32), frame_shape=(720, 1280))
    assert len(b) == 2
    assert b.gate[0] and not b.gate[1]
    assert np.isfinite(b.total[0]) and np.isinf(b.total[1])
    assert b.excluded[1].startswith("outside_radius")


def test_select_candidate_refuses_when_all_gated():
    cfg = CostCfg()
    pred = np.array([0, 0, 10, 20], np.float32)
    dets = [_det([900, 900, 1000, 1100])]
    b = compute_costs(dets, pred, cfg, frame_shape=(720, 1280))
    r = select_candidate(b, cfg)
    assert not r.accepted and r.index is None
    assert "gated" in r.reason


def test_select_candidate_accepts_the_obvious_match():
    cfg = CostCfg()
    pred = np.array([100, 100, 200, 300], np.float32)
    dets = [_det([100, 100, 200, 300], feat=_unit([1, 0, 0, 0])), _det([600, 100, 700, 300])]
    b = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=_unit([1, 0, 0, 0]), frame_shape=(720, 1280),
    )
    r = select_candidate(b, cfg)
    assert r.accepted, r.reason
    assert r.index == 0
    assert r.cost < 0.3


def test_distractor_margin_refuses_a_toss_up():
    """Two near-identical candidates must NOT be resolved by a coin flip."""
    cfg = CostCfg(distractor_margin=0.5)
    pred = np.array([100, 100, 200, 300], np.float32)
    dets = [
        _det([100, 100, 200, 300], feat=_unit([1, 0, 0, 0])),
        _det([100, 100, 200, 300], feat=_unit([1, 0, 0, 0])),
    ]
    b = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=_unit([1, 0, 0, 0]), frame_shape=(720, 1280),
    )
    r = select_candidate(b, cfg)
    assert not r.accepted
    assert "ambiguous" in r.reason
    # With the margin disabled the same situation is accepted.
    r2 = select_candidate(b, cfg, require_margin=False)
    assert r2.accepted


def test_lost_margin_override_is_stricter_than_tracking():
    """The looser cost bound while lost must NOT loosen the ambiguity test.

    While the motion gate is off, appearance is the only evidence left, so a
    rebind has to clear a HIGHER margin even though `accept_cost_lost` lets the
    absolute cost go up.
    """
    cfg = CostCfg(distractor_margin=0.05, distractor_margin_lost=0.5)
    pred = np.array([100, 100, 200, 300], np.float32)
    stm = _unit([1, 0, 0, 0])
    dets = [
        _det([100, 100, 200, 300], feat=_unit([1, 0, 0, 0])),
        _det([100, 100, 200, 300], feat=_unit([0.95, 0.3122, 0, 0])),
    ]
    b = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=stm, frame_shape=(720, 1280),
    )
    tracked = select_candidate(b, cfg, lost=False)
    assert tracked.accepted, tracked.reason
    assert 0.05 < tracked.margin < 0.5

    lost = select_candidate(b, cfg, lost=True)
    assert not lost.accepted
    assert "ambiguous" in lost.reason
    # Only the threshold moved -- the measured margin is the same evidence.
    assert lost.margin == tracked.margin
    assert "0.500" in lost.reason

    # 0 disables the override, so `lost` falls back to `distractor_margin`.
    cfg2 = CostCfg(distractor_margin=0.05, distractor_margin_lost=0.0)
    b2 = compute_costs(
        dets, pred, cfg2, velocity=np.zeros(4, np.float32),
        stm_feat=stm, frame_shape=(720, 1280),
    )
    assert select_candidate(b2, cfg2, lost=True).accepted


def test_lost_state_uses_the_looser_threshold():
    """`lost=True` raises the absolute bound, but a lone candidate is still capped.

    Two detections are used for the threshold claim on purpose: with a single
    finite candidate `lone_candidate_accept_cost` also applies, so a one-detection
    fixture would be measuring the lone-candidate rule instead of the threshold.
    The final block asserts that interaction explicitly.
    """
    cfg = CostCfg(accept_cost=0.4, accept_cost_lost=1.2)
    pred = np.array([100, 100, 200, 300], np.float32)
    dets = [
        _det([140, 100, 240, 300], feat=_unit([0.8, 0.6, 0, 0])),
        # The runner-up must stay INSIDE the gate radius, otherwise it is filtered
        # out and the surviving detection is a lone candidate, which the bound --
        # not the absolute threshold -- would then be measuring.
        _det([110, 100, 210, 300], feat=_unit([0.0, 0.0, 1.0, 0.0])),
    ]
    b = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=_unit([1, 0, 0, 0]), frame_shape=(720, 1280),
    )
    assert not select_candidate(b, cfg, lost=False).accepted
    assert select_candidate(b, cfg, lost=True).accepted

    # Remove the runner-up and the same detection becomes a lone candidate that
    # the bound refuses even while lost -- the loose absolute threshold must not
    # silently override the evidence test.
    #
    # The bound is set explicitly here because it is NOT the shipped default: on
    # real MOT17 it costs 3.6 pp of target success on the upper-body-only mask with
    # no measurable false-acquisition benefit, so `Config` leaves it at None. The
    # RULE still has to be tested, so this fixture opts in.
    lone_cfg = CostCfg(
        accept_cost=0.4, accept_cost_lost=1.2, lone_candidate_accept_cost=0.5,
    )
    lone = compute_costs(
        dets[:1], pred, lone_cfg, velocity=np.zeros(4, np.float32),
        stm_feat=_unit([1, 0, 0, 0]), frame_shape=(720, 1280),
    )
    lone_res = select_candidate(lone, lone_cfg, lost=True)
    assert not lone_res.accepted, lone_res.reason
    assert lone_res.reason.startswith("lone_candidate"), lone_res.reason


def test_lone_candidate_bound_refuses_a_lone_stranger_but_not_the_target():
    """The margin test cannot protect a lone candidate, so a bound must.

    With exactly one finite candidate the runner-up margin is ``inf`` and
    ``margin < distractor_margin`` is false for every threshold, so the
    anti-latch test is vacuous. This is not hypothetical: motion-dev's trace of
    `unidentifiable_missing_s060` hands the identity over on the frame the
    target stops being detected, when the bystander is the only detection
    (n_dets 2 -> 1, accepted at cost 0.90, no runner-up), while legitimate lone
    accepts in the no-bystander control cost 0.02.
    """
    pred = np.array([100, 100, 200, 300], np.float32)
    # accept_cost is raised out of the way so ONLY the lone rule can refuse.
    kwargs = dict(accept_cost=5.0, accept_cost_lost=5.0, distractor_margin=0.0)

    stranger = _det([100, 100, 200, 300], feat=_unit([0.0, 1.0, 0, 0]))
    target = _det([100, 100, 200, 300], feat=_unit([1.0, 0, 0, 0]))
    stm = _unit([1.0, 0, 0, 0])

    def costs(dets, cfg):
        return compute_costs(
            dets, pred, cfg, velocity=np.zeros(4, np.float32),
            stm_feat=stm, frame_shape=(720, 1280),
        )

    # Disabled by default: the lone stranger is still accepted.
    off = CostCfg(lone_candidate_accept_cost=None, **kwargs)
    assert select_candidate(costs([stranger], off), off).accepted

    # Enabled: a lone candidate that looks nothing like the template is refused.
    on = CostCfg(lone_candidate_accept_cost=0.4, **kwargs)
    refused = select_candidate(costs([stranger], on), on)
    assert not refused.accepted
    assert refused.reason.startswith("lone_candidate"), refused.reason
    assert refused.index == 0 and refused.cost > 0.4

    # The genuine target is a lone candidate too and must sail through.
    ok = select_candidate(costs([target], on), on)
    assert ok.accepted, ok.reason
    assert ok.cost < 0.4

    # A lone candidate is bounded in BOTH directions of the state machine. The
    # lost path used to be exempt "because re-acquisition costs more" — and that
    # exemption is exactly what let the `unidentifiable_target` family complete
    # its handover two frames later, once the state had gone blind and the bound
    # stopped applying (see the `--lone` sweep in `scripts/ablate_motion.py`,
    # where removing the exemption takes lone-candidate handovers from 3/4 runs
    # to 0/4). A genuine re-acquisition is unharmed: it arrives with a LOW cost,
    # because there the appearance channel really does recognise the target.
    lost_refused = select_candidate(costs([stranger], on), on, lost=True)
    assert not lost_refused.accepted, "a lone stranger must not be accepted while lost either"
    assert lost_refused.reason.startswith("lone_candidate"), lost_refused.reason
    assert select_candidate(costs([target], on), on, lost=True).accepted

    # With a runner-up present the margin test governs, not this bound: the
    # winner here costs more than the bound yet must still be accepted.
    two = select_candidate(costs([stranger, _det([100, 100, 200, 300], feat=_unit([0.5, 0.866, 0, 0]))], on), on)
    assert two.accepted, two.reason
    assert two.cost > 0.4


def test_no_gate_mode_allows_global_redetection():
    cfg = CostCfg()
    far = _det([900, 900, 1000, 1100], feat=_unit([1, 0, 0, 0]))
    pred = np.array([100, 100, 200, 300], np.float32)
    gated = compute_costs([far], pred, cfg, frame_shape=(720, 1280))
    global_ = compute_costs([far], pred, cfg, apply_gate=False, frame_shape=(720, 1280))
    assert np.isinf(gated.total[0])
    assert np.isfinite(global_.total[0])


def test_hard_negative_raises_the_total_cost():
    cfg = CostCfg(hard_negative_penalty=0.5)
    pred = np.array([100, 100, 200, 300], np.float32)
    feat = _unit([1, 0, 0, 0])
    dets = [_det([100, 100, 200, 300], feat=feat)]
    clean = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=feat, ltm=_FakeGallery({}, hn=0.0), frame_shape=(720, 1280),
    )
    poisoned = compute_costs(
        dets, pred, cfg, velocity=np.zeros(4, np.float32),
        stm_feat=feat, ltm=_FakeGallery({}, hn=1.0), frame_shape=(720, 1280),
    )
    assert poisoned.total[0] > clean.total[0]


def test_compute_costs_handles_empty_input():
    b = compute_costs([], None, CostCfg())
    assert len(b) == 0
    r = select_candidate(b, CostCfg())
    assert not r.accepted and r.reason == "no_detections"


def test_compute_costs_is_finite_and_crash_free_without_features():
    cfg = CostCfg()
    pred = np.array([100, 100, 200, 300], np.float32)
    dets = [_det([100, 100, 200, 300]), _det([110, 105, 205, 305])]
    b = compute_costs(dets, pred, cfg, velocity=np.zeros(4, np.float32), frame_shape=(720, 1280))
    assert np.all(np.isfinite(b.total[b.gate]))
    assert np.all(b.appearance == pytest.approx(0.5))
