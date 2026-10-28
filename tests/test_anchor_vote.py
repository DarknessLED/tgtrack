"""E079 -- the reference vote over the E027/E029 anchor veto.

The mechanism (docs/E079_PLAN.md): the appearance veto is a one-reference test
against the frozen click crop, and E077 localised the remaining blindness onto it
(``all_gated:anchor_ceiling``).  E079 adds two more references that answer the
SAME question with the SAME threshold -- the E029 band -- and admits a vetoed
candidate when at least k of 3 pass it:

* R1 the frozen click crop (never updated, unpoisonable);
* R2 a refreshed anchor: an EMA of accepted descriptors, moved ONLY on frames on
  which R1 and R3 both already accept the same box;
* R3 the long-term gallery (nearest-entry cosine distance).

What must remain true:

* with ``anchor_vote="off"`` (the shipped default) the veto arithmetic is exactly
  the pre-E079 one -- a candidate over the band is set to ``inf``;
* R2 cannot move without the R1-and-R3 agreement, which is the anti-poisoning rule
  and the reason a drift cannot train the reference that later admits candidates;
* R1 alone still refuses: when R2 and R3 both disagree the veto stands, and with
  ``unanimous`` (k = 3) nothing is ever admitted;
* ``unanimous`` is bit-identical to ``off`` on a whole run, because R1 has already
  rejected every candidate that reaches the vote.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.association import CostBreakdown
from tgtrack.config import Config
from tgtrack.target_tracker import TargetTracker
from tgtrack.types import Detection

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
BOX = [500.0, 200.0, 600.0, 500.0]
BAND = 0.40


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


ANCHOR = _unit([1.0, 0.0, 0.0, 0.0])


def _vec_at(dist: float) -> np.ndarray:
    """A unit 4-vector whose *cosine distance* from ``ANCHOR`` is ``dist``."""
    d = float(np.clip(dist, 0.0, 2.0))
    theta = float(np.arccos(np.clip(1.0 - d, -1.0, 1.0)))
    return _unit([np.cos(theta), np.sin(theta), 0.0, 0.0])


class RampEmbedder:
    """Deterministic encoder whose distance to the anchor is keyed by box x-centre."""

    def __init__(self, table: dict[float, float]) -> None:
        self.table = dict(table)

    def dist_for(self, box) -> float:
        cx = (float(box[0]) + float(box[2])) / 2.0
        key = min(self.table, key=lambda k: abs(float(k) - cx))
        return float(self.table[key])

    def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        out = []
        for b in boxes:
            f = _vec_at(self.dist_for(b))
            out.append({p: f.copy() for p in parts})
        return out


def _tracker(embedder: RampEmbedder | None = None, **overrides) -> TargetTracker:
    """Tracker clicked on ``BOX`` with a FIXED ``BAND`` so the vote is testable."""
    cfg = Config().clone(
        **{"cost.anchor_adaptive": False, "cost.anchor_max_dist": BAND, **overrides}
    )
    emb = embedder or RampEmbedder({550.0: 0.0})
    tt = TargetTracker(cfg, emb)
    tt.initialize(FRAME, BOX, frame_idx=0)
    return tt


def _det(cx: float, feat: np.ndarray) -> Detection:
    d = Detection(xyxy=np.array([cx - 50.0, 200.0, cx + 50.0, 500.0], np.float32), score=0.9, cls=0)
    d.feat = feat
    return d


def _gallery_vouches(tt: TargetTracker, dist: float) -> None:
    """Make R3 answer ``dist`` for any candidate: "the gallery knows this look".

    Both paths are stubbed: ``_vote_admit`` reads R3 through the batched
    ``_vote_gallery_distances`` accessor, while the R2 agreement check in
    ``_vote_update_r2`` still asks the gallery directly.
    """
    tt.ltm.best = lambda feat, part=None, frame_idx=None, touch=False: (float(dist), None)  # type: ignore[method-assign]

    def _batched(feats, _dist: float = float(dist)) -> np.ndarray:
        arr = np.asarray(feats, dtype=np.float32)
        return np.full(int(arr.shape[0]) if arr.ndim == 2 else 1, _dist, dtype=np.float32)

    tt._vote_gallery_distances = _batched  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# shipped default
# --------------------------------------------------------------------------- #
def test_vote_defaults_are_off_and_alpha_0_1() -> None:
    cost = Config().cost
    assert cost.anchor_vote == "off"
    assert cost.anchor_vote_alpha == pytest.approx(0.1)


def test_unknown_vote_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        TargetTracker(Config().clone(**{"cost.anchor_vote": "supermajority"}), None)


def test_off_admits_nothing() -> None:
    """With the vote off the mask is all-False whatever the distances are."""
    tt = _tracker()
    assert tt._vote_k == 0
    feats = np.asarray([_vec_at(0.10), _vec_at(0.90)], np.float32)
    dist = np.asarray([0.10, 0.90], np.float64)
    over = dist > BAND
    admitted = tt._vote_admit(feats, dist, over, BAND)
    assert not admitted.any()
    assert tt._vote_n == 0


# --------------------------------------------------------------------------- #
# vote arithmetic
# --------------------------------------------------------------------------- #
def test_majority_admits_on_two_of_three() -> None:
    """R1 fails by construction; R2 and R3 agreeing is exactly k = 2."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.70})
    tt = _tracker(emb, **{"cost.anchor_vote": "majority"})
    assert tt._vote_k == 2
    # R2 has drifted onto this look (the only way it ever moves), and the gallery
    # has stored it: two refreshed references pass while the click crop does not.
    tt._vote_r2 = _vec_at(0.70)
    _gallery_vouches(tt, 0.15)
    feats = np.asarray([_vec_at(0.70)], np.float32)
    admitted = tt._vote_admit(feats, np.asarray([0.70]), np.asarray([True]), BAND)
    assert admitted.tolist() == [True]
    assert tt._vote_rescued == 1
    assert tt._vote_r1_pass == 0, "a candidate that reaches the vote has failed R1"
    assert tt._vote_r2_pass == 1
    assert tt._vote_r3_pass == 1


def test_majority_still_refuses_when_only_one_other_reference_passes() -> None:
    """R1 alone still refuses: R2 passing while R3 disagrees is 1 of 3, not k = 2."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.70})
    tt = _tracker(emb, **{"cost.anchor_vote": "majority"})
    tt._vote_r2 = _vec_at(0.70)  # R2 passes
    _gallery_vouches(tt, float("inf"))  # R3 cannot answer
    feats = np.asarray([_vec_at(0.70)], np.float32)
    admitted = tt._vote_admit(feats, np.asarray([0.70]), np.asarray([True]), BAND)
    assert admitted.tolist() == [False]
    assert tt._vote_rescued == 0
    assert tt._vote_r2_pass == 1 and tt._vote_r3_pass == 0


def test_unanimous_can_never_admit_a_vetoed_candidate() -> None:
    """k = 3 is unreachable here: reaching the vote means R1 already said no."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.10})
    tt = _tracker(emb, **{"cost.anchor_vote": "unanimous"})
    assert tt._vote_k == 3
    tt._vote_r2 = _vec_at(0.70)
    _gallery_vouches(tt, 0.05)
    feats = np.asarray([_vec_at(0.70)], np.float32)
    admitted = tt._vote_admit(feats, np.asarray([0.70]), np.asarray([True]), BAND)
    assert admitted.tolist() == [False]
    assert tt._vote_rescued == 0
    assert tt._vote_r2_pass == 1 and tt._vote_r3_pass == 1, "R2 and R3 both pass, and it is moot"


def test_candidates_outside_the_veto_are_never_touched() -> None:
    tt = _tracker(RampEmbedder({550.0: 0.0}), **{"cost.anchor_vote": "majority"})
    tt._vote_r2 = _vec_at(0.90)
    _gallery_vouches(tt, 0.10)
    feats = np.asarray([_vec_at(0.10), _vec_at(0.90)], np.float32)
    admitted = tt._vote_admit(
        feats, np.asarray([0.10, 0.90]), np.asarray([False, True]), BAND
    )
    assert admitted.tolist() == [False, True]
    assert tt._vote_n == 1, "only the vetoed candidate enters the vote"


# --------------------------------------------------------------------------- #
# R2 movement (the anti-poisoning rule)
# --------------------------------------------------------------------------- #
def test_r2_does_not_move_without_r1_and_r3_agreement() -> None:
    """A crop the gallery refuses must not teach the refreshed anchor."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.90})
    tt = _tracker(emb, **{"cost.anchor_vote": "majority"})
    before = np.array(tt._vote_r2, np.float32, copy=True)
    tt._vote_update_r2(tt.cfg.cost, _vec_at(0.90))  # outside the 0.40 band for R1 and R3
    assert np.allclose(tt._vote_r2, before), "R2 must not move on a refused crop"
    assert tt._vote_r2_updates == 0


def test_r2_moves_when_r1_and_r3_agree() -> None:
    """A crop both the click crop and the gallery accept IS evidence, so R2 moves."""
    emb = RampEmbedder({550.0: 0.0})
    tt = _tracker(emb, **{"cost.anchor_vote": "majority"})
    before = np.array(tt._vote_r2, np.float32, copy=True)
    tt._vote_update_r2(tt.cfg.cost, _vec_at(0.20))
    assert not np.allclose(tt._vote_r2, before)
    assert tt._vote_r2_updates == 1
    # the EMA moves a fraction of the way, not all of it: alpha 0.1
    moved = float(np.arccos(np.clip(float(np.dot(tt._vote_r2, before)), -1.0, 1.0)))
    assert 0.0 < moved < 0.20


def test_r2_is_inert_when_the_vote_is_off() -> None:
    tt = _tracker(RampEmbedder({550.0: 0.0}))
    assert tt._vote_r2 is None
    tt._vote_update_r2(tt.cfg.cost, _vec_at(0.20))
    assert tt._vote_r2 is None and tt._vote_r2_updates == 0


# --------------------------------------------------------------------------- #
# the veto block, end to end
# --------------------------------------------------------------------------- #
def _associate_once(tt: TargetTracker, dets: list[Detection]):
    tt.frame_idx += 1
    return tt._associate(FRAME, dets, 720, 1280)


def test_off_veto_sets_inf_on_an_out_of_band_candidate() -> None:
    tt = _tracker(RampEmbedder({550.0: 0.0, 900.0: 0.90}))
    # The box sits inside the motion gate; only its DESCRIPTOR is out of band, so
    # this measures the E027 veto and not the gate that runs before it.
    res, full, _chosen = _associate_once(tt, [_det(560.0, _vec_at(0.90))])
    assert np.isinf(full.total).all()
    assert not res.accepted
    assert res.reason.startswith("all_gated")
    assert tt._last_diag["anchor_rejected"] == 1


def test_majority_rescues_the_same_candidate_the_off_arm_refuses() -> None:
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.90})
    off = _tracker(emb)
    _res, off_full, _ = _associate_once(off, [_det(560.0, _vec_at(0.90))])
    assert np.isinf(off_full.total).all()

    maj = _tracker(emb, **{"cost.anchor_vote": "majority"})
    maj._vote_r2 = _vec_at(0.90)
    _gallery_vouches(maj, 0.15)
    _res2, maj_full, _ = _associate_once(maj, [_det(560.0, _vec_at(0.90))])
    assert np.isfinite(maj_full.total).all(), "the vote must leave the candidate finite"
    assert maj.stats()["vote_n"] == 1
    assert maj.stats()["vote_rescued"] == 1


def test_unanimous_leaves_the_veto_identical_to_off() -> None:
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.90})
    off = _tracker(emb)
    una = _tracker(emb, **{"cost.anchor_vote": "unanimous"})
    _r1, f1, _ = _associate_once(off, [_det(560.0, _vec_at(0.90))])
    _r2, f2, _ = _associate_once(una, [_det(560.0, _vec_at(0.90))])
    assert np.array_equal(f1.total, f2.total)
    assert f1.excluded == f2.excluded
    assert una.stats()["vote_n"] == 1, "the control still COUNTS the veto"
    assert una.stats()["vote_rescued"] == 0


# --------------------------------------------------------------------------- #
# engagement counters
# --------------------------------------------------------------------------- #
def test_stats_expose_the_engagement_counters_in_both_arms() -> None:
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.90})
    off = _tracker(emb)
    _associate_once(off, [_det(560.0, _vec_at(0.90))])
    s = off.stats()
    assert s["anchor_vote"] == "off"
    assert s["vote_n"] == 0 and s["vote_rescued"] == 0 and s["vote_r2_updates"] == 0
    assert s["anchor_band_frames"] == 1

    maj = _tracker(emb, **{"cost.anchor_vote": "majority"})
    _associate_once(maj, [_det(560.0, _vec_at(0.90))])
    s = maj.stats()
    assert s["anchor_vote"] == "majority"
    assert s["vote_frames"] == 1 and s["vote_n"] == 1
    assert s["vote_rescued"] == 0, "R2 has not drifted yet, so nothing is rescued"
    assert s["anchor_band_saturation"] == pytest.approx(0.0)  # fixed band, not adaptive


def test_band_saturation_is_reported_on_the_adaptive_band() -> None:
    """The E078 75 % saturation number is re-measured by the same run."""
    emb = RampEmbedder({550.0: 0.0})
    tt = _tracker(
        emb,
        **{
            "cost.anchor_adaptive": True,
            "cost.anchor_max_dist": None,
            "cost.anchor_adaptive_ceil": 0.75,
        },
    )
    # A wildly noisy history pins the median + 4*1.4826*MAD far above the ceiling.
    for d in (0.05, 1.20, 0.06, 1.30, 0.05, 1.10, 0.07, 1.25):
        tt._anchor_dist_hist.append(d)
    _associate_once(tt, [_det(550.0, _vec_at(0.0))])
    assert tt._last_diag["anchor_band"] == pytest.approx(0.75)
    assert tt._last_diag["anchor_band_sat"] is True
    assert tt.stats()["anchor_band_saturated"] == 1


# --------------------------------------------------------------------------- #
# whole-run equivalence (the off arm must not move the shipped path)
# --------------------------------------------------------------------------- #
def _replay(mode: str) -> list[tuple]:
    """A short deterministic run: the click crop, a look-alike, then a far person."""
    emb = RampEmbedder({550.0: 0.0, 560.0: 0.10, 900.0: 0.95})
    tt = _tracker(emb, **{"cost.anchor_vote": mode})
    trace: list[tuple] = []
    for k, (cx, dist) in enumerate([(560.0, 0.10), (900.0, 0.95), (560.0, 0.10)]):
        dets = [_det(cx, _vec_at(dist)), _det(940.0, _vec_at(1.20))]
        status = tt.step(FRAME, dets, frame_idx=k + 1)
        trace.append((
            status.state,
            status.diag.get("reason"),
            bool(status.diag.get("accepted")),
            None if status.xyxy is None else tuple(np.round(status.xyxy, 6)),
        ))
    return trace


def test_unanimous_reproduces_the_off_trace_over_a_whole_run() -> None:
    assert _replay("unanimous") == _replay("off")


# --------------------------------------------------------------------------- #
# E081 -- the distress gate over the vote
from tgtrack.target_tracker import TargetState  # noqa: E402
# --------------------------------------------------------------------------- #
def _adaptive(embedder=None, tau=0.35, **overrides) -> TargetTracker:
    return _tracker(
        embedder,
        **{
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_distress_tau": float(tau),
            "cost.anchor_vote_distress_window": 10,
            **overrides,
        },
    )


def _vetoed_candidate(tt: TargetTracker, cx: float = 900.0):
    """A candidate R1 refuses, which R2 and R3 both pass (feat = the click crop)."""
    feat = np.asarray(tt._vote_r2, np.float32).copy()
    d = _det(cx, feat)
    return feat, d


def test_adaptive_tau_negative_is_bit_identical_to_majority() -> None:
    """tau < 0 must disable the gate, so the mask equals the majority arm's."""
    ref = _tracker(**{"cost.anchor_vote": "majority"})
    _gallery_vouches(ref, 0.0)
    feat_ref, det_ref = _vetoed_candidate(ref)
    feats = np.asarray([feat_ref], np.float32)
    dist = np.asarray([BAND + 0.2], np.float64)
    over = dist > BAND
    mask_ref = ref._vote_admit(feats, dist, over, BAND)
    assert bool(mask_ref[0]) is True  # R2 and R3 both vouch -> k = 2 admits

    ad = _adaptive(tau=-1.0)
    _gallery_vouches(ad, 0.0)
    feat_ad, _ = _vetoed_candidate(ad)
    feats_ad = np.asarray([feat_ad], np.float32)
    mask_ad = ad._vote_admit(feats_ad, dist, over, BAND)
    assert np.array_equal(mask_ref, mask_ad)
    assert ad._vote_gated == 0


def test_adaptive_tau_one_never_votes_and_counts_the_gate() -> None:
    """tau = 1.0 is the null control: distress can never exceed 1.0."""
    ad = _adaptive(tau=1.0)
    _gallery_vouches(ad, 0.0)
    feat, _ = _vetoed_candidate(ad)
    feats = np.asarray([feat], np.float32)
    dist = np.asarray([BAND + 0.2], np.float64)
    over = dist > BAND
    mask = ad._vote_admit(feats, dist, over, BAND)
    assert not mask.any()
    assert ad._vote_n == 0            # the vote never even ran
    assert ad._vote_gated == 1        # but the gate is visibly doing the work


def test_distress_is_the_share_of_blind_frames_in_the_trailing_window() -> None:
    tt = _adaptive()
    assert tt._vote_distress() == 0.0
    for _ in range(6):
        tt.state = TargetState.LOST_SHORT
        tt._vote_sample()
    assert tt._vote_distress() == pytest.approx(1.0)
    for _ in range(4):
        tt.state = TargetState.TRACKING
        tt._vote_sample()
    assert len(tt._vote_hist) == 10
    assert tt._vote_distress() == pytest.approx(0.6)


def test_distress_window_keeps_only_the_trailing_samples() -> None:
    tt = _adaptive()  # window = 10
    tt.state = TargetState.LOST_LONG
    for _ in range(10):
        tt._vote_sample()
    assert len(tt._vote_hist) == 10
    tt.state = TargetState.TRACKING
    for _ in range(10):
        tt._vote_sample()
    assert len(tt._vote_hist) == 10
    assert tt._vote_distress() == pytest.approx(0.0)


def test_soft_occlusion_counts_as_distress_but_tracking_does_not() -> None:
    tt = _adaptive()
    tt.state = TargetState.SOFT_OCCLUSION
    tt._vote_sample()
    assert tt._vote_distress() == pytest.approx(1.0)
    tt._vote_hist.clear()
    tt.state = TargetState.TRACKING
    tt._vote_sample()
    assert tt._vote_distress() == pytest.approx(0.0)


def test_stats_expose_the_distress_and_gate_counters() -> None:
    st = _adaptive().stats()
    assert "vote_distress" in st and "vote_gated" in st
    assert st["vote_distress"] == pytest.approx(0.0)
    assert st["vote_gated"] == 0


def test_adaptive_is_inert_while_the_off_arm_is_shipped() -> None:
    """The gate must cost nothing in the shipped configuration."""
    tt = _tracker()
    feats = np.asarray([_vec_at(0.10)], np.float32)
    dist = np.asarray([0.10], np.float64)
    over = dist > BAND
    assert not tt._vote_admit(feats, dist, over, BAND).any()
    assert tt._vote_gated == 0
    assert tt._vote_hist == []
