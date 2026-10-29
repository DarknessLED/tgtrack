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


# --------------------------------------------------------------------------- #
# E082 -- the CONTRAST gate over the vote
#
#     contrast(c) = d_R1(c) - min(d_R2(c), d_R3(c))
#     admit iff passes >= 2 AND contrast >= delta
#
# A candidate reaches the vote only because R1 refused it, so contrast > 0 for any
# candidate the blanket vote would rescue; delta = 0 is therefore pinned to
# `vote_majority` and is the experiment's plumbing falsifier.  The deltas of
# interest are the ones that make the gate BIND on that population.
# --------------------------------------------------------------------------- #
def _contrast(delta: float, embedder=None, **overrides) -> TargetTracker:
    return _tracker(
        embedder,
        **{
            "cost.anchor_vote": "contrast",
            "cost.anchor_vote_contrast": float(delta),
            **overrides,
        },
    )


def _contrast_admit(tt: TargetTracker, d1: float, d2: float, d3: float,
                    over: bool | None = None) -> np.ndarray:
    """One candidate whose three reference distances are EXACTLY (d1, d2, d3).

    The three references are independent inputs of ``_vote_admit``: ``dist_r1`` is
    passed in, R2 is pinned to ``ANCHOR`` so a feature at cosine distance ``d2`` from
    ``ANCHOR`` is at ``d2`` from R2, and R3 is stubbed to ``d3``.  That makes the
    contrast a controlled number instead of a property of the encoder.
    """
    tt._vote_r2 = ANCHOR.copy()
    _gallery_vouches(tt, float(d3))
    feats = np.asarray([_vec_at(d2)], np.float32)
    dist_r1 = np.asarray([float(d1)], np.float64)
    mask = over if over is not None else bool(float(d1) > BAND)
    return tt._vote_admit(feats, dist_r1, np.asarray([mask]), BAND)


def test_contrast_defaults_are_disabled_and_shipped_off() -> None:
    """The knob ships negative (gate off) and `anchor_vote` still ships 'off'."""
    cost = Config().cost
    assert cost.anchor_vote == "off"
    assert cost.anchor_vote_contrast == pytest.approx(-1.0)
    tt = _tracker()
    assert tt._vote_k == 0
    assert tt._vote_contrast_n == 0 and tt._vote_contrast_blocked == 0


def test_contrast_delta_zero_is_bit_identical_to_majority() -> None:
    """The plumbing falsifier: delta = 0 must be the blanket majority vote."""
    maj = _tracker(**{"cost.anchor_vote": "majority"})
    con = _contrast(0.0)
    # Two candidates: one that R2 and R3 both rescue, one that only R2 passes.
    for tt in (maj, con):
        tt._vote_r2 = ANCHOR.copy()
        _gallery_vouches(tt, 0.15)  # R3 passes both
    feats = np.asarray([_vec_at(0.30), _vec_at(0.30)], np.float32)
    dist = np.asarray([0.62, 0.55], np.float64)  # R1 refuses both
    over = dist > BAND
    mask_maj = maj._vote_admit(feats, dist, over, BAND)
    mask_con = con._vote_admit(feats, dist, over, BAND)
    assert mask_maj.tolist() == [True, True]
    assert np.array_equal(mask_maj, mask_con)
    for key in ("vote_n", "vote_r1_pass", "vote_r2_pass", "vote_r3_pass", "vote_rescued"):
        assert getattr(maj, "_" + key) == getattr(con, "_" + key), key
    assert con._vote_contrast_blocked == 0, "a delta = 0 gate can never bind"
    assert con._vote_contrast_n == 2, "both rescuable candidates are on the curve"


def test_contrast_delta_zero_reproduces_the_majority_veto_end_to_end() -> None:
    """Same frame through `_associate`: identical cost arrays and counters."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.90})
    maj = _tracker(emb, **{"cost.anchor_vote": "majority"})
    con = _contrast(0.0, emb)
    for tt in (maj, con):
        tt._vote_r2 = _vec_at(0.90)
        _gallery_vouches(tt, 0.15)
    _r1, f_maj, _ = _associate_once(maj, [_det(560.0, _vec_at(0.90))])
    _r2, f_con, _ = _associate_once(con, [_det(560.0, _vec_at(0.90))])
    assert np.isfinite(f_maj.total).all()
    assert np.array_equal(f_maj.total, f_con.total)
    assert f_maj.excluded == f_con.excluded
    assert maj.stats()["vote_rescued"] == con.stats()["vote_rescued"] == 1
    assert con.stats()["anchor_vote_contrast"] == pytest.approx(0.0)
    assert con.stats()["vote_contrast_blocked"] == 0


def test_contrast_admits_a_candidate_closer_to_the_refreshed_references() -> None:
    """d_R1 0.60 vs min(d_R2, d_R3) 0.30: the crop is the stale reference."""
    tt = _contrast(0.05)
    admitted = _contrast_admit(tt, d1=0.60, d2=0.30, d3=0.34)
    assert admitted.tolist() == [True]
    assert tt._vote_rescued == 1 and tt._vote_contrast_blocked == 0
    assert tt._vote_contrast_n == 1
    assert tt._last_diag["vote_contrast_min"] == pytest.approx(0.30, abs=1e-5)
    st = tt.stats()
    assert st["vote_contrast_ge_0.20"] == 1
    assert st["vote_contrast_ge_0.30"] == 0  # 0.6 - 0.3 lands one ulp below 0.30
    assert st["vote_contrast_mean"] == pytest.approx(0.30, abs=1e-5)


def test_contrast_blocks_a_marginal_rescue() -> None:
    """A rescue on weak evidence: contrast 0.015 does not clear delta 0.05.

    This is the population E082 is about -- R2/R3 only have to be *just* inside the
    band for the blanket vote, even when they are no closer than R1.
    """
    d1, d2, d3 = 0.405, 0.395, 0.390
    blocked = _contrast(0.05)
    mask = _contrast_admit(blocked, d1=d1, d2=d2, d3=d3)
    assert mask.tolist() == [False]
    assert blocked._vote_rescued == 0
    assert blocked._vote_contrast_blocked == 1
    assert blocked.stats()["vote_contrast_mean"] == pytest.approx(d1 - d3, abs=1e-4)

    # the SAME frame at delta = 0 is the blanket majority vote: admitted
    blanket = _contrast(0.0)
    assert _contrast_admit(blanket, d1=d1, d2=d2, d3=d3).tolist() == [True]
    assert blanket._vote_contrast_blocked == 0


def test_contrast_exactly_at_delta_is_admitted() -> None:
    """`contrast >= delta` is inclusive; one ulp above the margin blocks."""
    probe = _contrast(0.0)
    _contrast_admit(probe, d1=0.70, d2=0.30, d3=0.34)
    margin = float(probe._last_diag["vote_contrast_min"])
    assert margin > 0.0

    at = _contrast(margin)
    assert _contrast_admit(at, d1=0.70, d2=0.30, d3=0.34).tolist() == [True]
    assert at._vote_contrast_blocked == 0

    just_above = _contrast(margin + 1e-6)
    assert _contrast_admit(just_above, d1=0.70, d2=0.30, d3=0.34).tolist() == [False]
    assert just_above._vote_contrast_blocked == 1


def test_contrast_handles_non_finite_distances_by_the_letter_of_the_rule() -> None:
    """NaN compares False (blocked); +inf clears every delta; -inf never does."""
    nan = _contrast(0.0)
    assert _contrast_admit(
        nan, d1=float("nan"), d2=0.20, d3=0.20, over=True).tolist() == [False]
    assert nan._vote_contrast_blocked == 1
    assert nan._vote_rescued == 0

    inf = _contrast(1.00)
    assert _contrast_admit(inf, d1=float("inf"), d2=0.20, d3=0.20).tolist() == [True]
    assert inf._vote_contrast_blocked == 0

    finite_but_far = _contrast(0.05)
    assert _contrast_admit(
        finite_but_far, d1=0.50, d2=0.20, d3=0.20, over=True).tolist() == [True]
    # ... and the contrast arithmetic itself is a bare difference
    c = TargetTracker._vote_contrast(
        np.asarray([0.50]), np.asarray([0.20]), np.asarray([0.60]))
    assert float(c[0]) == pytest.approx(0.30)
    c_inf = TargetTracker._vote_contrast(
        np.asarray([np.inf]), np.asarray([np.inf]), np.asarray([np.inf]))
    assert np.isnan(float(c_inf[0])), "inf - inf must not be read as a huge margin"


def test_contrast_mode_is_inert_when_the_vote_is_off() -> None:
    """The shipped default: a contrast delta must change nothing at all."""
    tt = _tracker(**{"cost.anchor_vote_contrast": 0.05})
    assert tt._vote_k == 0 and tt._vote_mode == "off"
    feats = np.asarray([_vec_at(0.30)], np.float32)
    dist = np.asarray([0.90], np.float64)
    assert not tt._vote_admit(feats, dist, dist > BAND, BAND).any()
    assert tt._vote_contrast_n == 0 and tt._vote_contrast_blocked == 0
    st = tt.stats()
    assert st["anchor_vote"] == "off"
    assert st["vote_contrast_n"] == 0


def test_contrast_negative_delta_is_bit_identical_to_majority() -> None:
    """delta < 0 disables the gate, so the arm equals the blanket vote again."""
    maj = _tracker(**{"cost.anchor_vote": "majority"})
    off_gate = _contrast(-1.0)
    for tt in (maj, off_gate):
        tt._vote_r2 = ANCHOR.copy()
        _gallery_vouches(tt, 0.15)
    feats = np.asarray([_vec_at(0.30)], np.float32)
    dist = np.asarray([0.62], np.float64)
    over = dist > BAND
    assert np.array_equal(maj._vote_admit(feats, dist, over, BAND),
                          off_gate._vote_admit(feats, dist, over, BAND))
    assert off_gate._vote_contrast_blocked == 0


def test_contrast_stats_expose_the_retention_curve() -> None:
    """The engagement record carries the curve the delta grid is chosen from."""
    tt = _contrast(0.20)
    _contrast_admit(tt, d1=0.55, d2=0.30, d3=0.30)   # contrast 0.25, admitted
    _contrast_admit(tt, d1=0.45, d2=0.30, d3=0.30)   # contrast 0.15, blocked
    _contrast_admit(tt, d1=0.42, d2=0.30, d3=0.30)   # contrast 0.12, blocked
    st = tt.stats()
    assert st["vote_contrast_n"] == 3
    assert st["vote_contrast_blocked"] == 2
    assert st["vote_contrast_ge_0.00"] == 3
    assert st["vote_contrast_ge_0.10"] == 3
    assert st["vote_contrast_ge_0.20"] == 1
    assert st["vote_contrast_min"] == pytest.approx(0.12, abs=1e-4)
    assert st["vote_contrast_max"] == pytest.approx(0.25, abs=1e-4)


# --------------------------------------------------------------------------- #
# E084b -- k = 1, the corner of the k-axis that was never measured
# --------------------------------------------------------------------------- #
def test_any_mode_is_k1() -> None:
    """``any`` is k = 1; the campaign had only ever measured k = 2 and k = 3."""
    tt = _tracker(**{"cost.anchor_vote": "any"})
    assert tt._vote_k == 1


def test_any_admits_when_only_the_gallery_passes() -> None:
    """k = 1 admits on R3 alone -- the case the E084 step-0 audit found 18 of.

    R1 can never pass here (the vote only sees candidates R1 already refused), so
    "one of three" means R2 *or* R3.  With R2 still un-moved and the gallery
    vouching, exactly one reference passes and k = 2 would refuse.
    """
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.70})
    tt = _tracker(emb, **{"cost.anchor_vote": "any"})
    _gallery_vouches(tt, 0.15)
    feats = np.asarray([_vec_at(0.70)], np.float32)
    admitted = tt._vote_admit(feats, np.asarray([0.70]), np.asarray([True]), BAND)
    assert admitted.tolist() == [True]
    assert tt._vote_rescued == 1
    assert (tt._vote_r1_pass, tt._vote_r2_pass, tt._vote_r3_pass) == (0, 0, 1)


def test_any_is_a_superset_of_majority_on_the_same_frame() -> None:
    """Same frame, same two references: k = 1 admits what k = 2 refuses."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 0.70})
    tt_any = _tracker(emb, **{"cost.anchor_vote": "any"})
    _gallery_vouches(tt_any, 0.15)
    tt_maj = _tracker(emb, **{"cost.anchor_vote": "majority"})
    _gallery_vouches(tt_maj, 0.15)
    feats = np.asarray([_vec_at(0.70)], np.float32)
    dist = np.asarray([0.70])
    over = np.asarray([True])
    assert tt_any._vote_admit(feats, dist, over, BAND).tolist() == [True]
    assert tt_maj._vote_admit(feats, dist, over, BAND).tolist() == [False]
