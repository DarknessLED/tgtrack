"""E083 -- two structurally different hypotheses, decided on future frames.

``docs/E083_PLAN.md`` pre-registered K clones of the tracker; the amendment in
``docs/EXPERIMENTS.md`` supersedes that with K = 2 hypotheses that differ in
exactly one thing -- which reference set they trust:

* ``H_crop``    the shipped tracker, one frozen reference: the click crop (R1);
* ``H_gallery`` the E079 k = 2 arm: the refreshed anchor R2 plus the long-term
  gallery R3 (``anchor_vote="majority"``).

What must remain true, and is pinned here:

* ``mh_mode="off"`` (the shipped default) delegates to ONE ``TargetTracker`` and
  is bit-identical to constructing that tracker directly -- the falsifier for the
  plumbing;
* while the two reported boxes agree (IoU >= 0.5) the reported box is H_crop's,
  so before any disagreement the arm IS the shipped tracker;
* a disagreement opens a validation window that closes after
  ``mh_validate_frames`` comparable frames, freezes both reference sets at the
  open, and scores each hypothesis only on the crops THAT hypothesis accepts
  afterwards -- ``r_h = mean(d_own) / (mean(d_other) + eps)``, ``logw = -lambda r``;
* the reported box moves only on a posterior margin confirmed
  ``mh_commit_frames`` times (``ChallengeTracker``'s consecutive-frame logic), and
  a tie or a degenerate window keeps H_crop;
* ``mh_mode="dual_frozen"`` never moves the reported hypothesis and never opens a
  window: it is the control for the compute alone.

The likelihood tests use a cued stand-in for ``TargetTracker`` so that the
*decided-by-future-frames* arithmetic is exercised exactly (the two reference
beds, the drift ratios, the commit chain); the cases where the shipped trackers
themselves must stay unchanged use real ``TargetTracker`` instances.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import Config
from tgtrack.hypotheses import (
    AGREE_IOU,
    CROP,
    GALLERY,
    MH_MODES,
    MultiHypothesisTracker,
    ReferenceSet,
    mode_from_cfg,
)
from tgtrack.target_tracker import TargetTracker
from tgtrack.types import Detection, TargetStatus, cosine_distance_matrix

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
BOX = [500.0, 200.0, 600.0, 500.0]
#: Same y-range as ``BOX``, no overlap in x: the two hypotheses really disagree.
GAL_BOX = [850.0, 200.0, 950.0, 500.0]
BAND = 0.40


# --------------------------------------------------------------------------- #
# geometry: two reference beds 60 degrees apart
# --------------------------------------------------------------------------- #
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


def _at(deg: float) -> np.ndarray:
    """A unit 4-vector ``deg`` degrees from ``e0`` in the (e0, e1) plane."""
    t = float(np.deg2rad(float(deg)))
    return _unit([np.cos(t), np.sin(t), 0.0, 0.0])


#: H_crop's one reference (R1, the click crop).
CROP_REF = _at(0.0)
#: H_gallery's refreshed anchor R2 (and its gallery entry R3).
GAL_REF = _at(60.0)


def _dist(a, b) -> float:
    q = np.asarray(a, dtype=np.float32)
    bank = np.asarray(b, dtype=np.float32).reshape(1, -1)
    return float(cosine_distance_matrix(q, bank)[0])


# --------------------------------------------------------------------------- #
# cued stand-in for TargetTracker (the decision arithmetic, without association)
# --------------------------------------------------------------------------- #
class _FakeGallery:
    def __init__(self, rows=()) -> None:
        self.rows = list(rows)

    def entries(self) -> list:
        return list(self.rows)


class _Entry:
    """Just enough of ``GalleryEntry`` for ``_gallery_bank`` to read."""

    def __init__(self, feat, part: str = "full") -> None:
        self.feat = feat
        self.part = part


class CuedTracker:
    """Reports exactly what the frame plan says, and nothing else.

    ``script`` is indexed by ``frame_idx``; each entry is ``{"crop": (box, feat),
    "gallery": (box, feat)}``.  The reported box is copied verbatim (as the real
    tracker does on a match), so ``MultiHypothesisTracker`` recovers the accepted
    detection by box and the likelihood consumes the feature on that detection.
    """

    def __init__(
        self,
        kind: str,
        script: list,
        *,
        crop_ref=None,
        gallery_ref=None,
        gallery_rows=(),
    ) -> None:
        self.kind = kind
        self.script = list(script)
        self.events: list = []
        self._anchor_feat = None if crop_ref is None else np.asarray(crop_ref, np.float32)
        self._vote_r2 = None if gallery_ref is None else np.asarray(gallery_ref, np.float32)
        self.ltm = _FakeGallery(gallery_rows)
        self.frame_idx = 0
        self._status: TargetStatus | None = None

    def _entry(self, frame_idx: int):
        if not self.script:
            return None, None
        idx = int(np.clip(int(frame_idx), 0, len(self.script) - 1))
        box, feat = self.script[idx][self.kind]
        return box, feat

    def _make(self, frame_idx: int) -> TargetStatus:
        box, _feat = self._entry(frame_idx)
        xyxy = None if box is None else np.asarray(box, np.float32)
        return TargetStatus(
            frame_idx=int(frame_idx),
            state="tracking" if xyxy is not None else "occluded_soft",
            xyxy=xyxy,
            score=0.9,
            frames_since_update=0 if xyxy is not None else 1,
            age=int(frame_idx),
            reacquired=False,
            diag={"accepted": xyxy is not None, "reason": "scripted"},
        )

    def initialize(self, frame_bgr, box_xyxy, frame_idx: int = 0) -> TargetStatus:
        self.frame_idx = int(frame_idx)
        self._status = self._make(self.frame_idx)
        return self._status

    def step(self, frame_bgr, dets, frame_idx=None, frame_shape=None) -> TargetStatus:
        self.frame_idx = int(frame_idx) if frame_idx is not None else self.frame_idx + 1
        self._status = self._make(self.frame_idx)
        return self._status

    def status(self, frame_idx=None) -> TargetStatus:
        if self._status is None:
            return self._make(self.frame_idx)
        return self._status

    def stats(self) -> dict:
        return {"anchor_vote": "off" if self.kind == "crop" else "majority", "state": "tracking"}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _config(**cost) -> Config:
    overrides = {"cost.anchor_adaptive": False, "cost.anchor_max_dist": BAND}
    for key, value in cost.items():
        if key.startswith(("mh_", "anchor_")):
            overrides[f"cost.{key}"] = value
        else:  # the E083 knobs take the short spelling in the tests
            overrides[f"cost.mh_{key}"] = value
    return Config().clone(**overrides)


def _mh(mode: str = "dual", **cost) -> MultiHypothesisTracker:
    return MultiHypothesisTracker(_config(mh_mode=mode, **cost), None, mode=mode)


def _cue(mh: MultiHypothesisTracker, script: list, *, crop_ref=CROP_REF, gallery_ref=GAL_REF,
         gallery_rows=None) -> MultiHypothesisTracker:
    """Replace the pair's real trackers with cued ones sharing one frame plan."""
    rows = () if gallery_rows is None else gallery_rows
    mh.hypotheses[0].tracker = CuedTracker("crop", script, crop_ref=crop_ref)
    mh.hypotheses[1].tracker = CuedTracker(
        "gallery", script, crop_ref=crop_ref, gallery_ref=gallery_ref, gallery_rows=rows
    )
    return mh


def _dets(frame: dict) -> list[Detection]:
    out: list[Detection] = []
    for key in ("crop", "gallery"):
        box, feat = frame[key]
        if box is None:
            continue
        det = Detection(xyxy=np.asarray(box, np.float32), score=0.9, cls=0)
        det.feat = None if feat is None else np.asarray(feat, np.float32)
        out.append(det)
    return out


def _drive(mh: MultiHypothesisTracker, plan: list, *, init_box=BOX) -> list[TargetStatus]:
    statuses = [mh.initialize(FRAME, init_box, frame_idx=0)]
    for i in range(1, len(plan)):
        statuses.append(mh.step(FRAME, _dets(plan[i]), frame_idx=i))
    return statuses


def _agree(crop_feat, gallery_feat, n: int = 1) -> list[dict]:
    return [{"crop": (BOX, crop_feat), "gallery": (BOX, gallery_feat)} for _ in range(n)]


def _disagree(crop_feat, gallery_feat, n: int = 1) -> list[dict]:
    return [{"crop": (BOX, crop_feat), "gallery": (GAL_BOX, gallery_feat)} for _ in range(n)]


#: The frame on which a window opened at frame 1 decides (1 + validate_frames).
DECIDE = 13


def _window_plan(crop_feat, gallery_feat, *, tail: int = 0) -> list[dict]:
    """One agreement frame, then ``DECIDE`` disagreement frames, then a tail."""
    return _agree(crop_feat, gallery_feat) + _disagree(
        crop_feat, gallery_feat, DECIDE + tail
    )


def _real_tracker(mode: str = "dual", embedder=None, **cost) -> MultiHypothesisTracker:
    cfg = _config(mh_mode=mode, anchor_vote="off" if mode == "off" else "majority", **cost)
    return MultiHypothesisTracker(cfg, embedder, mode=mode)


# --------------------------------------------------------------------------- #
# shipped default
# --------------------------------------------------------------------------- #
def test_mh_defaults_are_off_and_inert() -> None:
    cost = Config().cost
    assert cost.mh_mode == "off"
    assert cost.mh_commit_margin == pytest.approx(0.15)
    assert cost.mh_commit_frames == 3
    assert cost.mh_validate_frames == 12
    assert cost.mh_drift_lambda == pytest.approx(2.0)
    assert mode_from_cfg(Config()) == "off"
    assert MH_MODES == ("off", "dual", "dual_frozen")


def test_unknown_mh_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiHypothesisTracker(_config(mh_mode="triple"), None)
    with pytest.raises(ValueError):
        MultiHypothesisTracker(Config(), None, mode="clones")
    # An unusable knob is refused at construction rather than silently ignored.
    with pytest.raises(ValueError):
        MultiHypothesisTracker(Config(), None, mode="dual", commit_frames=0)
    with pytest.raises(ValueError):
        MultiHypothesisTracker(Config(), None, mode="dual", validate_frames=0)


def test_off_mode_is_bit_identical_to_a_bare_tracker() -> None:
    """The falsifier: ``mh_mode="off"`` must not move the shipped path."""

    class RampEmbedder:
        def __init__(self, table):
            self.table = dict(table)

        def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
            out = []
            for box in np.asarray(boxes_xyxy, np.float32).reshape(-1, 4):
                cx = (float(box[0]) + float(box[2])) / 2.0
                key = min(self.table, key=lambda k: abs(float(k) - cx))
                feat = _at(np.rad2deg(np.arccos(np.clip(1.0 - self.table[key], -1.0, 1.0))))
                out.append({p: feat.copy() for p in parts})
            return out

    def det(cx: float, dist: float) -> Detection:
        d = Detection(xyxy=np.array([cx - 50.0, 200.0, cx + 50.0, 500.0], np.float32), score=0.9)
        d.feat = _at(np.rad2deg(np.arccos(np.clip(1.0 - dist, -1.0, 1.0))))
        return d

    script = [(560.0, 0.10), (900.0, 0.95), (560.0, 0.10), (900.0, 0.95), (560.0, 0.10)]

    def trace(tracker_like, embedder) -> list[tuple]:
        tracker_like.initialize(FRAME, BOX, frame_idx=0)
        out = []
        for k, (cx, dist) in enumerate(script):
            status = tracker_like.step(FRAME, [det(cx, dist), det(940.0, 1.20)], frame_idx=k + 1)
            out.append(
                (
                    status.state,
                    status.diag.get("reason"),
                    bool(status.diag.get("accepted")),
                    None if status.xyxy is None else tuple(np.round(status.xyxy, 6)),
                    int(status.frames_since_update),
                )
            )
        return out

    emb_a = RampEmbedder({550.0: 0.0, 560.0: 0.10, 900.0: 0.95})
    emb_b = RampEmbedder({550.0: 0.0, 560.0: 0.10, 900.0: 0.95})
    bare = trace(TargetTracker(_config(), emb_a), emb_a)
    mh = _real_tracker("off", emb_b)
    assert mh.k == 1 and mh.hypotheses == []
    delegated = trace(mh, emb_b)
    assert delegated == bare
    assert mh.reported() == 0
    assert mh.stats()["mh_k"] == 1


def test_dual_builds_the_two_structural_hypotheses() -> None:
    mh = _real_tracker("dual")
    assert mh.k == 2
    assert mh.n_spawned == 2
    assert mh.hypotheses[0].kind == CROP and mh.hypotheses[1].kind == GALLERY
    assert mh.hypotheses[0].tracker._vote_mode == "off"
    assert mh.hypotheses[1].tracker._vote_mode == "majority"
    assert np.allclose(mh.weights(), [0.5, 0.5])
    stats = mh.stats()
    assert stats["mh_mode"] == "dual" and stats["mh_k"] == 2
    assert [h["stats"]["anchor_vote"] for h in stats["hypotheses"]] == ["off", "majority"]


def test_agreement_phase_reports_crop_with_real_trackers() -> None:
    """No disagreement -> no window -> the run IS the shipped tracker's."""

    class FlatEmbedder:
        def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
            feat = _at(0.0)
            return [{p: feat.copy() for p in parts} for _ in range(len(boxes_xyxy))]

    def flat_det() -> Detection:
        d = Detection(xyxy=np.asarray(BOX, np.float32), score=0.9)
        d.feat = _at(0.0)
        return d

    mh = _real_tracker("dual", FlatEmbedder())
    mh.initialize(FRAME, BOX, frame_idx=0)
    for i in range(1, 21):
        status = mh.step(FRAME, [flat_det()], frame_idx=i)
        assert mh.reported() == 0
        assert status.diag["mh_reported"] == CROP
        assert np.allclose(status.xyxy, BOX)
    stats = mh.stats()
    assert stats["n_windows_opened"] == 0
    assert stats["n_commits"] == 0
    assert stats["mh_iou"] >= AGREE_IOU


# --------------------------------------------------------------------------- #
# the window
# --------------------------------------------------------------------------- #
def test_disagreement_opens_a_window_that_closes_after_validate_frames() -> None:
    plan = _window_plan(_at(30.0), GAL_REF)
    mh = _cue(_mh(), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)

    # Frame 1 is the first disagreement: the window opens, H_crop keeps the box.
    assert mh.stats()["mh_window_start"] == 1
    assert statuses[1].diag["mh_window"] == 1
    assert statuses[1].diag["mh_reported"] == CROP
    assert statuses[1].diag["mh_window_age"] == 0  # opened this frame, not yet aged
    assert mh.n_windows_opened == 1

    # It ages one frame per comparable frame and closes exactly at frame 13.
    assert statuses[2].diag["mh_window_age"] == 1
    assert statuses[DECIDE - 1].diag["mh_window"] == 1
    assert statuses[DECIDE].diag["mh_window"] == 0
    assert mh.n_windows_closed == 1
    assert mh.n_evidence_samples == 2 * mh.validate_frames

    # Both reference beds were frozen as they stood at the open, and released at close.
    short = _cue(_mh(), _agree(_at(30.0), GAL_REF) + _disagree(_at(30.0), GAL_REF, 3),
                 gallery_rows=[_Entry(GAL_REF)])
    _drive(short, short.hypotheses[0].tracker.script)
    assert short.hypotheses[0].refs is not None and short.hypotheses[1].refs is not None
    assert short.hypotheses[0].refs.size == 1  # H_crop: the frozen click crop
    assert short.hypotheses[1].refs.size == 2  # H_gallery: R2 + one gallery entry
    assert short.hypotheses[0].refs.size <= short.hypotheses[1].refs.size
    assert mh.hypotheses[0].refs is None and mh.hypotheses[1].refs is None


def test_hypothesis_whose_own_reference_predicts_better_wins() -> None:
    """The gallery bed predicts both beds' crops; the crop bed only its own."""
    plan = _window_plan(_at(30.0), GAL_REF)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)

    r_crop = mh.hypotheses[0].drift_ratio
    r_gallery = mh.hypotheses[1].drift_ratio
    assert r_gallery < r_crop
    assert r_gallery == pytest.approx(0.0, abs=1e-3)  # the gallery crops ARE R2/R3
    assert r_crop == pytest.approx(_dist(_at(30.0), GAL_REF) / (_dist(_at(30.0), CROP_REF) + 1e-6), rel=1e-3)

    assert statuses[DECIDE - 1].diag["mh_reported"] == CROP  # decision not taken yet
    assert statuses[DECIDE].diag["mh_reported"] == GALLERY
    assert np.allclose(statuses[DECIDE].xyxy, GAL_BOX)
    assert mh.reported() == 1
    assert mh.n_commits == 1 and mh.n_swaps_reported == 1
    weights = mh.weights()
    assert weights[1] > 0.5
    assert weights[1] == pytest.approx(1.0 / (1.0 + np.exp(-mh.drift_lambda * (r_crop - r_gallery))), rel=1e-3)


def test_a_tie_keeps_the_crop_hypothesis() -> None:
    """Symmetric evidence (r = 0 on both sides) must not flip anything."""
    plan = _window_plan(CROP_REF, GAL_REF)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)
    assert mh.hypotheses[0].drift_ratio == pytest.approx(0.0, abs=1e-3)
    assert mh.hypotheses[1].drift_ratio == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(mh.weights(), [0.5, 0.5])
    assert mh.reported() == 0
    assert all(s.diag["mh_reported"] == CROP for s in statuses)
    assert mh.n_crop_holds == 1 and mh.n_commits == 0


def test_a_margin_below_the_bar_keeps_the_crop_hypothesis() -> None:
    """The gallery is ahead, but by less than ``mh_commit_margin``."""
    plan = _window_plan(_at(12.0), GAL_REF)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)
    assert mh.hypotheses[1].drift_ratio < mh.hypotheses[0].drift_ratio
    assert mh._last_win == 1
    assert 0.0 < mh._last_margin < mh.commit_margin
    assert mh.reported() == 0
    assert all(s.diag["mh_reported"] == CROP for s in statuses)
    assert mh.n_margin_missed == 1 and mh.n_commits == 0


def test_commit_frames_blocks_a_one_window_flip_and_commits_on_the_third() -> None:
    win = _window_plan(_at(30.0), GAL_REF)

    # One window, then the boxes agree again: the chain never matures.
    one = _cue(_mh(commit_frames=3), win + _agree(_at(30.0), GAL_REF, 3), gallery_rows=[_Entry(GAL_REF)])
    _drive(one, one.hypotheses[0].tracker.script)
    assert one.n_commit_blocked == 1
    assert one.n_commits == 0
    assert one.reported() == 0
    assert one.n_agree_resets == 1

    # Three windows in a row (decisions at 13, 26, 39): the third commits.
    three = _cue(_mh(commit_frames=3), _disagree(_at(30.0), GAL_REF, 45), gallery_rows=[_Entry(GAL_REF)])
    _drive(three, three.hypotheses[0].tracker.script)
    assert three.n_commit_blocked == 2
    assert three.n_commits == 1
    assert three.n_swaps_reported == 1
    assert three.reported() == 1


def test_dual_frozen_never_switches_and_never_opens_a_window() -> None:
    plan = _window_plan(_at(30.0), GAL_REF) + _disagree(_at(30.0), GAL_REF, 20)
    mh = _cue(_mh("dual_frozen", commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)
    assert all(s.diag["mh_reported"] == CROP for s in statuses)
    assert mh.reported() == 0
    assert mh.n_commits == 0 and mh.n_swaps_reported == 0
    assert mh.n_windows_opened == 0
    assert mh.n_disagree_frames > 0
    assert mh.stats()["mh_mode"] == "dual_frozen"


def test_reagreement_resets_to_the_crop_hypothesis() -> None:
    plan = _agree(_at(30.0), GAL_REF) + _disagree(_at(30.0), GAL_REF, 6) + _agree(_at(30.0), GAL_REF, 4)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    statuses = _drive(mh, plan)
    assert statuses[5].diag["mh_window"] == 1
    assert statuses[7].diag["mh_window"] == 0
    assert statuses[7].diag["mh_reported"] == CROP
    assert mh.n_agree_resets == 1
    assert np.allclose(mh.weights(), [0.5, 0.5])
    assert mh.n_commits == 0


def test_engagement_counters_increment() -> None:
    plan = _window_plan(_at(30.0), GAL_REF)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    _drive(mh, plan)
    stats = mh.stats()
    assert stats["n_spawned"] == 2
    assert stats["n_windows_opened"] == 1
    assert stats["n_windows_closed"] == 1
    assert stats["n_commits"] == 1
    assert stats["n_swaps_reported"] == 1
    assert stats["n_commit_blocked"] == 0
    assert stats["n_disagree_frames"] == len(plan) - 1
    assert stats["n_evidence_samples"] == 2 * mh.validate_frames
    assert stats["n_evidence_dropped"] == 0
    emitted = sum(h["reported_frames"] for h in stats["hypotheses"])
    assert emitted == len(plan) - 1  # every step reported exactly one box
    assert stats["mh_last_drift_ratios"] is not None
    assert stats["mh_commit_chain_agreed"] == 0  # the chain reset on commit


# --------------------------------------------------------------------------- #
# frames with no comparison, and degenerate evidence
# --------------------------------------------------------------------------- #
def test_frames_without_detections_do_not_open_a_window() -> None:
    plan = [{"crop": (None, None), "gallery": (None, None)} for _ in range(20)]
    plan[1] = {"crop": (BOX, _at(30.0)), "gallery": (None, None)}  # only one hypothesis sees anything
    mh = _cue(_mh(), plan, gallery_rows=[_Entry(GAL_REF)])
    _drive(mh, plan)
    assert mh.n_windows_opened == 0
    assert mh.n_disagree_frames == 0
    assert mh.reported() == 0


def test_missing_reference_sets_fall_back_to_the_crop_hypothesis() -> None:
    """An unusable reference bed is not evidence for anybody: no crash, no switch."""
    plan = _window_plan(_at(30.0), GAL_REF)
    mh = _cue(_mh(commit_frames=1), plan, gallery_ref=np.full(4, np.nan), gallery_rows=())
    _drive(mh, plan)
    assert mh.reported() == 0
    assert mh.n_windows_closed == 1
    assert mh.n_degenerate_windows == 1
    assert mh.n_commits == 0


def test_nan_features_are_dropped_and_never_award_a_win() -> None:
    nan = np.full(4, np.nan, np.float32)
    plan = _window_plan(nan, nan)
    mh = _cue(_mh(commit_frames=1), plan, gallery_rows=[_Entry(GAL_REF)])
    _drive(mh, plan)
    assert mh.n_evidence_samples == 0
    assert mh.n_evidence_dropped >= 2 * mh.validate_frames
    assert mh.n_degenerate_windows == 1
    assert mh.reported() == 0


# --------------------------------------------------------------------------- #
# reference-set plumbing
# --------------------------------------------------------------------------- #
def test_reference_sets_answer_the_same_question() -> None:
    crop = ReferenceSet(bank=CROP_REF.reshape(1, -1), kind=CROP)
    gallery = ReferenceSet(bank=np.stack([GAL_REF, GAL_REF]), kind=GALLERY)
    probe = _at(30.0)
    assert crop.size == 1 and gallery.size == 2
    assert crop.distance(probe) == pytest.approx(_dist(probe, CROP_REF), rel=1e-6)
    assert gallery.distance(probe) == pytest.approx(_dist(probe, GAL_REF), rel=1e-6)
    # Unusable inputs answer `inf` (dropped), never a small "good" distance.
    assert np.isinf(crop.distance(None))
    assert np.isinf(crop.distance(np.full(4, np.nan)))
    assert np.isinf(crop.distance(_at(0.0)[:3].reshape(3)))
    assert np.isinf(ReferenceSet(bank=np.zeros((0, 4), np.float32)).distance(probe))
