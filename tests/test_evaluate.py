"""Tests for :mod:`tgtrack.evaluate` — MOT metrics and target-centric metrics.

Correctness strategy
--------------------
1. **Hand-computed synthetic cases** with exact expected values (perfect tracks,
   one ID switch, one missed detection, a GT gap, a lost-and-recovered lock).
2. **Published reference fixtures** — the exact numerical fixtures and expected
   values from TrackEval's own test suite (``tests/test_metrics.py``), realised at
   the box level with an exact ``{0, 1}`` IoU structure, so CLEAR-MOT, Identity
   (IDF1) and HOTA are checked against numbers this project did not produce.
3. **Optional cross-check** against a real ``trackeval`` installation when it is
   importable (skipped otherwise — it is not a dependency).
4. **Target-centric semantics** — every DJI-critical failure mode gets its own
   test: latching onto a look-alike person, hallucinating a box, losing the
   target, never re-acquiring, and silently swapping identity.

All numbers here are exact values derived by hand from the algorithms.
"""

from __future__ import annotations

import csv
import math
from typing import Iterator

import numpy as np
import pytest

from tgtrack.config import EvalCfg
from tgtrack.evaluate import (
    HOTA_ALPHAS,
    GtBox,
    PredBox,
    combine_mot,
    combine_target,
    evaluate_mot,
    evaluate_target,
    mot_frame_data,
    target_frame_table,
    to_percent,
    write_frame_csv,
)
from tgtrack.types import TargetEvent, TargetStatus

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
TGT = [0.0, 0.0, 10.0, 20.0]      # the clicked target's box (GT id 1)
DIST = [50.0, 0.0, 60.0, 20.0]    # a look-alike distractor (GT id 2)
PHANTOM = [900.0, 900.0, 910.0, 910.0]  # far from every GT box


def _one_track(
    frames: Iterator[int],
    tid: int = 1,
    pred_tid: int = 1,
    box: list[float] | None = None,
    miss: tuple[int, ...] = (),
    shift: int = 0,
) -> tuple[dict[int, list[GtBox]], dict[int, list[PredBox]]]:
    """GT/pred dicts for a single perfectly-tracked object (optionally missed)."""
    box = TGT if box is None else box
    gt: dict[int, list[GtBox]] = {}
    pred: dict[int, list[PredBox]] = {}
    for f in frames:
        gt[f + shift] = [GtBox(f + shift, tid, box)]
        if f not in miss:
            pred[f + shift] = [PredBox(f + shift, pred_tid, box)]
    return gt, pred


def _st(
    frame: int,
    box: list[float] | None,
    tid: int | None = None,
    state: str = "tracking",
    reacquired: bool = False,
) -> TargetStatus:
    """One :class:`TargetStatus`; ``tid`` lands in ``diag["track_id"]``."""
    return TargetStatus(
        frame_idx=frame,
        state=state,
        xyxy=None if box is None else np.asarray(box, dtype=np.float64),
        diag={} if tid is None else {"track_id": tid},
        reacquired=reacquired,
    )


def _all_gt(
    n_frames: int,
    target: bool = True,
    distractor: bool = True,
    target_frames: set[int] | None = None,
) -> dict[int, list[GtBox]]:
    """All-person GT with the target (id 1) and/or a distractor (id 2) present.

    ``target_frames`` restricts the target identity to the frames where it is
    actually visible, so the all-person GT stays consistent with ``gt_target``.
    """
    out: dict[int, list[GtBox]] = {}
    for f in range(n_frames):
        boxes = []
        if target and (target_frames is None or f in target_frames):
            boxes.append(GtBox(f, 1, TGT))
        if distractor:
            boxes.append(GtBox(f, 2, DIST))
        out[f] = boxes
    return out


def _gt_target(n_frames: int, visible: set[int] | None = None) -> dict[int, np.ndarray | None]:
    """Target GT: ``TGT`` where visible (default: every frame), else ``None``."""
    return {
        f: (np.asarray(TGT, dtype=np.float64) if (visible is None or f in visible) else None)
        for f in range(n_frames)
    }


# --------------------------------------------------------------------------- #
# 1. Hand-computed CLEAR-MOT cases
# --------------------------------------------------------------------------- #
def test_perfect_tracking_is_all_ones() -> None:
    gt, pred = _one_track(range(10))
    gt[0].append(GtBox(0, 2, [100.0, 0.0, 110.0, 20.0]))
    pred[0].append(PredBox(0, 2, [100.0, 0.0, 110.0, 20.0]))
    for f in range(1, 10):
        gt[f].append(GtBox(f, 2, [100.0, 0.0, 110.0, 20.0]))
        pred[f].append(PredBox(f, 2, [100.0, 0.0, 110.0, 20.0]))

    r = evaluate_mot(gt, pred)
    assert r["MOTA"] == 1.0
    assert r["IDF1"] == 1.0
    assert r["HOTA"] == 1.0
    assert r["DetA"] == 1.0 and r["AssA"] == 1.0 and r["LocA"] == 1.0
    assert r["MOTP"] == 1.0
    assert r["IDSW"] == 0 and r["Frag"] == 0
    assert (r["TP"], r["FP"], r["FN"]) == (20, 0, 0)
    assert (r["MT"], r["PT"], r["ML"]) == (2, 0, 0)
    assert r["MTR"] == 1.0 and r["MLR"] == 0.0
    assert r["num_gt_ids"] == 2 and r["num_pred_ids"] == 2
    assert r["num_frames"] == 10 and r["num_gt_dets"] == 20
    assert r["iou_thresh"] == 0.5


def test_single_id_switch() -> None:
    """10 frames, one GT identity, the tracker splits it into ids 1 then 2."""
    gt: dict[int, list[GtBox]] = {}
    pred: dict[int, list[PredBox]] = {}
    for f in range(10):
        gt[f] = [GtBox(f, 7, TGT)]
        pred[f] = [PredBox(f, 1 if f < 5 else 2, TGT)]

    r = evaluate_mot(gt, pred)
    assert r["IDSW"] == 1
    assert r["MOTA"] == pytest.approx(0.9)          # (10 - 0 - 1) / 10
    assert r["IDF1"] == pytest.approx(0.5)          # IDTP 5, IDFN 5, IDFP 5
    assert (r["IDTP"], r["IDFN"], r["IDFP"]) == (5, 5, 5)
    assert r["IDP"] == pytest.approx(0.5) and r["IDR"] == pytest.approx(0.5)
    assert r["Frag"] == 0
    assert r["MT"] == 1
    # HOTA: DetA = 1 (all detections matched at every alpha), AssA = 1/2.
    assert r["DetA"] == 1.0
    assert r["AssA"] == pytest.approx(0.5)
    assert r["HOTA"] == pytest.approx(math.sqrt(0.5))


def test_missed_detection_in_two_object_frame() -> None:
    gt = {0: [GtBox(0, 1, TGT), GtBox(0, 2, DIST)]}
    pred = {0: [PredBox(0, 1, TGT)]}
    r = evaluate_mot(gt, pred)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 0, 1)
    assert r["MOTA"] == pytest.approx(0.5)          # (1 - 0 - 0) / 2
    assert r["MODA"] == pytest.approx(0.5)
    assert r["DetA"] == pytest.approx(0.5)
    assert r["AssA"] == 1.0
    assert r["HOTA"] == pytest.approx(math.sqrt(0.5))
    assert r["IDF1"] == pytest.approx(2.0 / 3.0)    # 1 / (1 + 0 + 0.5)
    assert r["IDP"] == 1.0 and r["IDR"] == pytest.approx(0.5)


def test_fragmentation_counts_interior_gap() -> None:
    """One object over 10 frames, missed at 3 and 4.

    ``Frag`` is pinned to TrackEval's accounting *exactly*.  Note the quirk the
    authoritative implementation has (and which we deliberately reproduce so
    that our numbers are comparable with published MOTChallenge results): a gap
    in which the tracker emits no detection at all does not reset the per-id
    previous-match state, so that gap alone is not counted as a fragmentation.
    """
    gt, pred = _one_track(range(10), miss=(3, 4))
    r = evaluate_mot(gt, pred)
    assert (r["TP"], r["FP"], r["FN"]) == (8, 0, 2)
    assert r["Frag"] == 0
    assert r["MOTA"] == pytest.approx(0.8)
    # tracked_ratio = 0.8 is *not* > 0.8, so this identity is partially tracked.
    assert (r["MT"], r["PT"], r["ML"]) == (0, 1, 0)
    assert r["PTR"] == 1.0 and r["MTR"] == 0.0

    # Same gap, but the tracker emits an unrelated detection during it: the
    # per-id state is reset, so the recovered run IS counted as a fragment.
    pred2 = dict(pred)
    for f in (3, 4):
        pred2[f] = [PredBox(f, 999, [500.0, 500.0, 520.0, 540.0])]
    r2 = evaluate_mot(gt, pred2)
    assert (r2["TP"], r2["FP"], r2["FN"]) == (8, 2, 2)
    assert r2["Frag"] == 1
    assert r2["MOTA"] == pytest.approx(0.6)


def test_motp_is_mean_iou_of_matched_pairs() -> None:
    gt_box = [0.0, 0.0, 10.0, 10.0]
    pred_box = [1.1111111111111112, 0.0, 11.11111111111111, 10.0]  # IoU = 0.8
    gt = {f: [GtBox(f, 1, gt_box)] for f in range(5)}
    pred = {f: [PredBox(f, 1, pred_box)] for f in range(5)}
    r = evaluate_mot(gt, pred)
    assert r["MOTA"] == 1.0
    assert r["MOTP"] == pytest.approx(0.8)
    assert r["MOTP_sum"] == pytest.approx(4.0)


def test_iou_threshold_boundary_is_inclusive() -> None:
    """IoU exactly 0.5 matches at threshold 0.5 and not at 0.51."""
    gt = {0: [GtBox(0, 1, [0, 0, 10, 10])]}
    pred = {0: [PredBox(0, 1, [0, 0, 10, 20])]}   # IoU = 100/200 = 0.5 exactly
    assert evaluate_mot(gt, pred)["TP"] == 1
    strict = evaluate_mot(gt, pred, EvalCfg(iou_thresh=0.51))
    assert (strict["TP"], strict["FN"], strict["FP"]) == (0, 1, 1)
    assert strict["MOTA"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# 2. Empty / degenerate inputs
# --------------------------------------------------------------------------- #
def test_no_predictions() -> None:
    gt = {f: [GtBox(f, 1, TGT)] for f in range(4)}
    r = evaluate_mot(gt, {})
    assert r["FN"] == 4 and r["TP"] == 0 and r["FP"] == 0
    assert r["MOTA"] == 0.0 and r["IDF1"] == 0.0 and r["HOTA"] == 0.0
    assert r["DetA"] == 0.0 and r["LocA"] == 1.0  # TrackEval's convention
    assert r["ML"] == 1 and r["MLR"] == 1.0


def test_no_gt() -> None:
    pred = {0: [PredBox(0, 1, TGT), PredBox(0, 2, DIST), PredBox(0, 3, PHANTOM)]}
    r = evaluate_mot({}, pred)
    assert r["FP"] == 3 and r["TP"] == 0 and r["FN"] == 0
    assert r["MOTA"] == pytest.approx(-3.0)  # (0 - 3 - 0) / max(1, 0)
    assert r["IDF1"] == 0.0 and r["HOTA"] == 0.0
    assert r["LocA"] == 1.0  # TrackEval's 1e-10/1e-10 convention, as above


def test_both_empty() -> None:
    r = evaluate_mot({}, {})
    assert r["num_frames"] == 0 and r["num_gt_dets"] == 0
    assert r["MOTA"] == 0.0 and r["IDF1"] == 0.0 and r["HOTA"] == 0.0
    # No matched pair exists, so LocA keeps TrackEval's vacuous value of 1.0.
    assert r["LocA"] == 1.0


# --------------------------------------------------------------------------- #
# 3. HOTA alpha sweep
# --------------------------------------------------------------------------- #
def test_hota_alpha_sweep_matches_partial_match_formula() -> None:
    """IoU = 9/11 = 0.8181...: matched at 16 of the 19 alphas, not at 0.85..0.95."""
    gt_box = [0.0, 0.0, 10.0, 100.0]
    pred_box = [1.0, 0.0, 11.0, 100.0]           # 900 / 1100 = 9/11
    gt = {f: [GtBox(f, 1, gt_box)] for f in range(5)}
    pred = {f: [PredBox(f, 1, pred_box)] for f in range(5)}

    r = evaluate_mot(gt, pred, return_curves=True)
    assert r["MOTP"] == pytest.approx(9.0 / 11.0)
    assert r["MOTA"] == 1.0 and r["IDF1"] == 1.0
    assert r["DetA"] == pytest.approx(16.0 / 19.0)
    assert r["AssA"] == pytest.approx(16.0 / 19.0)
    assert r["HOTA"] == pytest.approx(16.0 / 19.0)
    # LocA: unmatched alphas contribute TrackEval's eps/eps = 1.0.
    assert r["LocA"] == pytest.approx((16 * (9.0 / 11.0) + 3.0) / 19.0)

    curves = r["curves"]
    assert [round(a, 2) for a in curves["alphas"]] == [round(a, 2) for a in HOTA_ALPHAS]
    assert len(curves["alphas"]) == 19
    assert curves["alphas"][0] == pytest.approx(0.05)
    assert curves["alphas"][-1] == pytest.approx(0.95)
    assert curves["DetA_curve"] == pytest.approx([1.0] * 16 + [0.0] * 3)
    assert curves["HOTA_TP_curve"] == pytest.approx([5.0] * 16 + [0.0] * 3)
    assert np.mean(curves["HOTA_curve"]) == pytest.approx(r["HOTA"])
    assert np.mean(curves["DetA_curve"]) == pytest.approx(r["DetA"])
    assert np.allclose(
        np.sqrt(np.asarray(curves["DetA_curve"]) * np.asarray(curves["AssA_curve"])),
        curves["HOTA_curve"],
    )


def test_mot_frame_data_packs_trackeval_arrays() -> None:
    gt = {0: [GtBox(0, 9, [100, 0, 110, 10]), GtBox(0, 5, [0, 0, 10, 10])]}
    pred = {1: [PredBox(1, 3, [0, 0, 10, 10])]}
    data = mot_frame_data(gt, pred)
    assert data["frames"] == [0, 1]
    assert data["num_timesteps"] == 2
    assert data["num_gt_ids"] == 2 and data["num_tracker_ids"] == 1
    assert data["num_gt_dets"] == 2 and data["num_tracker_dets"] == 1
    # GT boxes are sorted by (track_id, x0) -> id 5 first, then id 9.
    assert data["gt_id_list"] == [5, 9]
    assert data["pred_id_list"] == [3]
    np.testing.assert_allclose(data["gt_boxes"][0], [[0, 0, 10, 10], [100, 0, 110, 10]])
    assert data["similarity_scores"][0].shape == (2, 0)
    assert data["similarity_scores"][1].shape == (0, 1)


def test_to_percent_scales_rates_only() -> None:
    gt, pred = _one_track(range(10))
    r = evaluate_mot(gt, pred)
    pct = to_percent(r)
    assert pct["MOTA"] == 100.0 and pct["HOTA"] == 100.0 and pct["IDF1"] == 100.0
    assert pct["TP"] == 10            # counts stay counts
    assert pct["IDSW"] == 0
    assert r["MOTA"] == 1.0           # the input dict is not mutated
    assert to_percent(r, keys=("MOTA",))["HOTA"] == 1.0


# --------------------------------------------------------------------------- #
# 4. Published TrackEval reference fixtures
# --------------------------------------------------------------------------- #
def _build_no_confusion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = np.zeros([5, 2, 2])
    sim[:, 0, 1] = [0, 0, 0, 1, 1]
    sim[:, 1, 0] = [1, 1, 0, 0, 0]
    gt_present = np.zeros([5, 2])
    gt_present[:, 0] = [1, 1, 1, 1, 1]
    gt_present[:, 1] = [1, 1, 1, 0, 0]
    tr_present = np.zeros([5, 2])
    tr_present[:, 0] = [1, 1, 1, 1, 0]
    tr_present[:, 1] = [1, 1, 1, 1, 1]
    return sim, gt_present, tr_present


def _build_with_confusion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim, gt_present, tr_present = _build_no_confusion()
    sim[:, 0, 0] = [0, 0, 1, 0, 0]
    sim[:, 1, 1] = [0, 1, 0, 0, 0]
    return sim, gt_present, tr_present


def _build_split_tracks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = np.zeros([5, 2, 5])
    sim[:, 0, 0] = [1, 1, 0, 0, 0]
    sim[:, 0, 3] = [0, 0, 0, 1, 1]
    sim[:, 1, 1] = [0, 0, 1, 1, 0]
    sim[:, 1, 2] = [0, 0, 0, 0, 1]
    sim[:, 1, 4] = [1, 1, 0, 0, 0]
    gt_present = np.zeros([5, 2])
    gt_present[:, 0] = [1, 1, 0, 1, 1]
    gt_present[:, 1] = [1, 1, 1, 1, 1]
    tr_present = np.zeros([5, 5])
    tr_present[:, 0] = [1, 1, 0, 0, 0]
    tr_present[:, 1] = [0, 0, 1, 1, 1]
    tr_present[:, 2] = [0, 0, 0, 0, 1]
    tr_present[:, 3] = [0, 0, 1, 1, 1]
    tr_present[:, 4] = [1, 1, 0, 0, 0]
    return sim, gt_present, tr_present


#: TrackEval published fixtures + the expected values from its own test suite,
#: plus the HOTA numbers derived by hand from TrackEval's ``metrics/hota.py``.
FIXTURES: dict[str, dict] = {
    "no_confusion": {
        "build": _build_no_confusion,
        "clear": {"CLR_TP": 4, "CLR_FN": 4, "CLR_FP": 5, "IDSW": 0, "MOTA": 1 - 9 / 8},
        "identity": {"IDTP": 4, "IDFN": 4, "IDFP": 5, "IDR": 4 / 8, "IDP": 4 / 9, "IDF1": 2 * 4 / 17},
        "hota": {"HOTA_TP": 4, "DetA": 4 / 13, "AssA": 13 / 40, "LocA": 1.0,
                 "HOTA": math.sqrt(4 / 13 * 13 / 40)},
    },
    "with_confusion": {
        "build": _build_with_confusion,
        "clear": {"CLR_TP": 5, "CLR_FN": 3, "CLR_FP": 4, "IDSW": 1, "MOTA": 1 - 8 / 8},
        "identity": {"IDTP": 4, "IDFN": 4, "IDFP": 5, "IDR": 4 / 8, "IDP": 4 / 9, "IDF1": 2 * 4 / 17},
        "hota": {"HOTA_TP": 5, "DetA": 5 / 12, "AssA": 0.285, "LocA": 1.0,
                 "HOTA": math.sqrt(5 / 12 * 0.285)},
    },
    "split_tracks": {
        "build": _build_split_tracks,
        "clear": {"CLR_TP": 9, "CLR_FN": 0, "CLR_FP": 2, "IDSW": 3, "MOTA": 1 - 5 / 9},
        "identity": {"IDTP": 4, "IDFN": 5, "IDFP": 7, "IDR": 4 / 9, "IDP": 4 / 11, "IDF1": 2 * 4 / 20},
        "hota": {"HOTA_TP": 9, "DetA": 9 / 11, "AssA": 52 / 135, "LocA": 1.0,
                 "HOTA": math.sqrt(9 / 11 * 52 / 135)},
    },
}

_FIXTURE_BOXES: dict[str, tuple[dict[int, list[GtBox]], dict[int, list[PredBox]]]] = {}


def _boxes_from_dense(
    similarity: np.ndarray, gt_present: np.ndarray, tr_present: np.ndarray
) -> tuple[dict[int, list[GtBox]], dict[int, list[PredBox]]]:
    """Realise an exact ``{0, 1}`` similarity structure with disjoint 10x10 boxes.

    Every pair whose fixture similarity is ``1`` shares a box (IoU exactly 1.0);
    every other pair gets a private box 1000px away (IoU exactly 0.0) — the same
    trick TrackEval's ``_from_dense`` needs, but at the box level.
    """
    n_t = similarity.shape[0]
    gt: dict[int, list[GtBox]] = {}
    pred: dict[int, list[PredBox]] = {}
    for t in range(n_t):
        parent: dict[tuple[str, int], tuple[str, int]] = {}

        def find(key: tuple[str, int]) -> tuple[str, int]:
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a: tuple[str, int], b: tuple[str, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        g_ids = [int(g) for g in np.flatnonzero(gt_present[t])]
        tr_ids = [int(x) for x in np.flatnonzero(tr_present[t])]
        for g in g_ids:
            for tr in tr_ids:
                if similarity[t, g, tr] == 1.0:
                    union(("g", g), ("t", tr))

        slots: dict[tuple[str, int], int] = {}

        def box_for(key: tuple[str, int]) -> list[float]:
            root = find(key)
            if root not in slots:
                slots[root] = len(slots)
            s = slots[root]
            return [s * 1000.0, 0.0, s * 1000.0 + 10.0, 10.0]

        if g_ids:
            gt[t] = [GtBox(t, g, box_for(("g", g))) for g in g_ids]
        if tr_ids:
            pred[t] = [PredBox(t, tr, box_for(("t", tr))) for tr in tr_ids]
    return gt, pred


def _fixture(name: str) -> tuple[dict[int, list[GtBox]], dict[int, list[PredBox]]]:
    if name not in _FIXTURE_BOXES:
        sim, gt_present, tr_present = FIXTURES[name]["build"]()
        _FIXTURE_BOXES[name] = _boxes_from_dense(sim, gt_present, tr_present)
    return _FIXTURE_BOXES[name]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_published_clear_mot_fixture(name: str) -> None:
    gt, pred = _fixture(name)
    r = evaluate_mot(gt, pred)
    for field, expected in FIXTURES[name]["clear"].items():
        assert r[field] == pytest.approx(expected), f"{name}: CLEAR {field}"


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_published_identity_fixture(name: str) -> None:
    gt, pred = _fixture(name)
    r = evaluate_mot(gt, pred)
    for field, expected in FIXTURES[name]["identity"].items():
        assert r[field] == pytest.approx(expected), f"{name}: Identity {field}"


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_published_hota_fixture(name: str) -> None:
    gt, pred = _fixture(name)
    r = evaluate_mot(gt, pred)
    for field, expected in FIXTURES[name]["hota"].items():
        if field == "HOTA_TP":
            continue
        assert r[field] == pytest.approx(expected), f"{name}: HOTA {field}"
    # Every nonzero similarity is 1.0, so the whole alpha sweep is flat.
    curves = evaluate_mot(gt, pred, return_curves=True)["curves"]
    assert curves["HOTA_TP_curve"] == pytest.approx(
        [FIXTURES[name]["hota"]["HOTA_TP"]] * 19
    )
    assert np.allclose(curves["HOTA_curve"], r["HOTA"])
    assert r["HOTA"] == pytest.approx(math.sqrt(r["DetA"] * r["AssA"]))


# --------------------------------------------------------------------------- #
# 5. Cross-sequence aggregation
# --------------------------------------------------------------------------- #
def test_combine_mot_reproduces_the_concatenation() -> None:
    """Two sequences with *disjoint* identities: aggregation must be exact."""
    gt_a, pred_a = _one_track(range(5), tid=1, pred_tid=1)
    gt_b, pred_b = _one_track(range(5), tid=2, pred_tid=2, miss=(2,))

    extra: dict[int, list[GtBox]] = {}
    extra_pred: dict[int, list[PredBox]] = {}
    for f, boxes in gt_b.items():
        extra[f + 5] = [GtBox(f + 5, b.track_id, b.xyxy) for b in boxes]
    for f, boxes in pred_b.items():
        extra_pred[f + 5] = [PredBox(f + 5, b.track_id, b.xyxy) for b in boxes]

    union = {**gt_a, **extra}
    union_pred = {**pred_a, **extra_pred}

    single = evaluate_mot(union, union_pred)
    combined = combine_mot([
        evaluate_mot(gt_a, pred_a, return_state=True),
        evaluate_mot(gt_b, pred_b, return_state=True),
    ])

    for field in ("TP", "FP", "FN", "IDSW", "Frag", "MT", "PT", "ML", "num_frames",
                  "num_gt_dets", "num_pred_dets", "num_gt_ids", "num_pred_ids"):
        assert combined[field] == single[field], field
    for field in ("MOTA", "MOTP", "IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "LocA"):
        assert combined[field] == pytest.approx(single[field], abs=1e-12), field

    assert combined["num_sequences"] == 2
    assert combined["MOTA"] == pytest.approx(0.9)
    assert combined["IDF1"] == pytest.approx(9.0 / 9.5)
    assert combined["AssA"] == pytest.approx(8.2 / 9.0)
    assert combined["HOTA"] == pytest.approx(math.sqrt(0.9 * (8.2 / 9.0)))
    assert combined["HOTA"] == pytest.approx(math.sqrt(combined["DetA"] * combined["AssA"]))


def test_combine_mot_single_sequence_and_errors() -> None:
    gt, pred = _one_track(range(6))
    solo = evaluate_mot(gt, pred)
    combined = combine_mot([evaluate_mot(gt, pred, return_state=True)])
    assert combined["MOTA"] == pytest.approx(solo["MOTA"])
    assert combined["HOTA"] == pytest.approx(solo["HOTA"])
    assert combined["IDF1"] == pytest.approx(solo["IDF1"])
    assert combined["num_sequences"] == 1
    with pytest.raises(ValueError):
        combine_mot([solo])  # no state -> cannot aggregate


# --------------------------------------------------------------------------- #
# 6. Target-centric metrics
# --------------------------------------------------------------------------- #
def test_target_perfect_tracking() -> None:
    n = 20
    statuses = {f: _st(f, TGT, tid=42) for f in range(n)}
    events = [TargetEvent(0, "init", {"track_id": 42})]
    r = evaluate_target(_gt_target(n), statuses, events, _all_gt(n))

    assert r["n_frames"] == 20 and r["n_visible_frames"] == 20 and r["n_absent_frames"] == 0
    assert r["n_report_frames"] == 20 and r["n_correct_frames"] == 20
    assert r["target_success_rate"] == 1.0
    assert r["loss_rate"] == 0.0 and r["no_report_rate"] == 0.0 and r["wrong_box_rate"] == 0.0
    assert r["target_precision"] == 1.0 and r["target_precision_visible"] == 1.0
    assert r["reacquire_events"] == 0 and r["reacquire_ok"] == 0
    assert math.isnan(r["reacquire_rate"]) and math.isnan(r["reacquire_latency"])
    assert r["false_acquisition_rate"] == 0.0
    assert r["false_acquisition_frames"] == 0 and r["false_acquisition_events"] == 0
    assert r["hallucination_frames"] == 0
    assert r["initial_target_id"] == 42
    assert r["id_consistency"] == 1.0 and r["target_switch_count"] == 0
    assert r["distinct_followed_ids"] == 1
    assert r["fragmentation"] == 0
    assert r["longest_correct_streak"] == 20 and r["longest_lost_streak"] == 0
    assert r["mean_correct_streak"] == 20.0 and r["n_correct_streaks"] == 1
    assert r["n_events_init"] == 1


def test_target_follows_the_wrong_person_every_frame() -> None:
    """Latch onto the look-alike distractor for the whole sequence -> FAR == 1."""
    n = 10
    statuses = {f: _st(f, DIST, tid=7) for f in range(n)}
    events = [TargetEvent(0, "init", {"track_id": 1})]  # clicked GT identity 1
    r = evaluate_target(_gt_target(n), statuses, events, _all_gt(n))

    assert r["target_success_rate"] == 0.0
    assert r["loss_rate"] == 1.0 and r["wrong_box_rate"] == 1.0 and r["no_report_rate"] == 0.0
    assert r["target_precision"] == 0.0
    assert r["false_acquisition_rate"] == 1.0
    assert r["false_acquisition_frames"] == 10
    assert r["false_acquisition_events"] == 1          # one continuous latch-on
    assert r["hallucination_frames"] == 0              # it merged with a real person
    assert r["id_consistency"] == 0.0                  # left the clicked identity
    assert r["target_switch_count"] == 0               # ...but never switched again
    assert r["distinct_followed_ids"] == 1
    assert r["longest_correct_streak"] == 0 and r["longest_lost_streak"] == 10
    assert r["n_correct_streaks"] == 0 and math.isnan(r["mean_correct_streak"])


def test_target_success_rate_follows_the_right_person() -> None:
    """Mirror of the previous test: the same setup on the *correct* identity."""
    n = 10
    statuses = {f: _st(f, TGT, tid=3) for f in range(n)}
    events = [TargetEvent(0, "init", {"track_id": 3})]
    r = evaluate_target(_gt_target(n), statuses, events, _all_gt(n))
    assert r["target_success_rate"] == 1.0
    assert r["false_acquisition_rate"] == 0.0
    assert r["id_consistency"] == 1.0


def test_target_mixed_success_and_identity_swap() -> None:
    n = 10
    statuses = {}
    for f in range(n):
        statuses[f] = _st(f, TGT, tid=42) if f < 6 else _st(f, DIST, tid=43)
    events = [TargetEvent(0, "init", {"track_id": 42})]
    r = evaluate_target(_gt_target(n), statuses, events, _all_gt(n))

    assert r["target_success_rate"] == pytest.approx(0.6)
    assert r["n_correct_frames"] == 6
    assert r["loss_rate"] == pytest.approx(0.4)
    assert r["wrong_box_rate"] == pytest.approx(0.4)
    assert r["target_precision"] == pytest.approx(0.6)
    assert r["false_acquisition_rate"] == pytest.approx(0.4)
    assert r["false_acquisition_frames"] == 4
    assert r["false_acquisition_events"] == 1
    assert r["id_consistency"] == pytest.approx(0.6)
    assert r["target_switch_count"] == 1
    assert r["distinct_followed_ids"] == 2
    assert r["longest_correct_streak"] == 6
    assert r["longest_lost_streak"] == 4
    assert r["mean_correct_streak"] == 6.0
    assert r["fragmentation"] == 0  # the broken run touches the end of the sequence


def test_target_gt_gap_and_reacquire_latency() -> None:
    """Target leaves at 5-9 and returns at 10; the tracker locks again at 12."""
    n = 20
    visible = set(range(0, 5)) | set(range(10, 20))
    statuses = {}
    for f in range(0, 5):
        statuses[f] = _st(f, TGT, tid=42)
    for f in range(5, 10):
        statuses[f] = _st(f, None, tid=None, state="lost_long")
    statuses[10] = _st(10, None, tid=None, state="lost_long")
    statuses[11] = _st(11, DIST, tid=99)
    for f in range(12, 20):
        statuses[f] = _st(f, TGT, tid=42)
    events = [
        TargetEvent(0, "init", {"track_id": 42}),
        TargetEvent(5, "lost", {"reason": "out_of_frame"}),
        TargetEvent(12, "reacquire", {"track_id": 42}),
    ]
    r = evaluate_target(_gt_target(n, visible), statuses, events, _all_gt(n, target_frames=visible))

    assert r["n_visible_frames"] == 15 and r["n_absent_frames"] == 5
    assert r["n_report_frames"] == 14
    assert r["n_correct_frames"] == 13
    assert r["target_success_rate"] == pytest.approx(13.0 / 15.0)
    assert r["no_report_frames"] == 1 and r["wrong_box_frames"] == 1
    assert r["loss_rate"] == pytest.approx(2.0 / 15.0)

    # Both gap types are detected; the latency clock starts at re-appearance (10).
    assert r["reacquire_events"] == 2 and r["reacquire_ok"] == 2
    assert r["reacquire_rate"] == 1.0
    assert r["reacquire_latency"] == pytest.approx(2.0)
    assert r["reacquire_latency_p90"] == pytest.approx(2.0)
    assert r["reacquire_latency_max"] == pytest.approx(2.0)
    assert r["reacquire_events_gt_gap"] == 1 and r["reacquire_rate_gt_gap"] == 1.0
    assert r["reacquire_latency_gt_gap"] == pytest.approx(2.0)
    assert r["reacquire_events_tracker_lost"] == 1
    assert r["reacquire_latency_tracker_lost"] == pytest.approx(2.0)
    assert r["reacquire_censored"] == 0

    assert r["false_acquisition_frames"] == 1
    assert r["false_acquisition_rate"] == pytest.approx(1.0 / 14.0)
    assert r["fragmentation"] == 1
    assert r["longest_correct_streak"] == 8
    # Longest run of *visible* frames that were not correctly tracked: the two
    # frames right after re-appearance (10 and 11), not the GT absence itself.
    assert r["longest_lost_streak"] == 2
    assert r["n_events_lost"] == 1 and r["n_events_reacquire"] == 1


def test_target_tracker_lost_while_visible() -> None:
    """The target never leaves the frame, but the tracker stops reporting 5-9."""
    n = 20
    statuses = {}
    for f in range(n):
        statuses[f] = _st(f, TGT, tid=42) if not (5 <= f <= 9) else _st(f, None, tid=None, state="lost_short")
    r = evaluate_target(_gt_target(n), statuses, [], _all_gt(n))

    assert r["n_visible_frames"] == 20 and r["n_report_frames"] == 15
    assert r["target_success_rate"] == pytest.approx(0.75)
    assert r["no_report_frames"] == 5
    assert r["loss_rate"] == pytest.approx(0.25)
    assert r["reacquire_events"] == 1
    assert r["reacquire_events_tracker_lost"] == 1
    assert r["reacquire_events_gt_gap"] == 0
    assert r["reacquire_rate_tracker_lost"] == 1.0
    assert r["reacquire_latency_tracker_lost"] == pytest.approx(5.0)
    assert r["reacquire_latency"] == pytest.approx(5.0)
    assert math.isnan(r["reacquire_latency_gt_gap"])
    assert r["fragmentation"] == 1
    assert r["longest_lost_streak"] == 5
    assert r["longest_correct_streak"] == 10
    assert r["false_acquisition_rate"] == 0.0


def test_target_never_reacquires_is_fail_not_nan() -> None:
    n = 20
    statuses = {f: (_st(f, TGT, tid=42) if f < 5 else _st(f, None, tid=None, state="lost_long"))
                for f in range(n)}
    r = evaluate_target(_gt_target(n), statuses, [], _all_gt(n))

    assert r["target_success_rate"] == pytest.approx(0.25)
    assert r["no_report_frames"] == 15
    assert r["loss_rate"] == pytest.approx(0.75)
    assert r["reacquire_events"] == 1 and r["reacquire_ok"] == 0
    assert r["reacquire_rate"] == 0.0
    assert math.isnan(r["reacquire_latency"])
    assert r["reacquire_fail"] == 1
    assert r["reacquire_censored"] == 1     # the window ran past the end of the sequence
    assert r["longest_lost_streak"] == 15
    assert r["fragmentation"] == 0          # the broken run touches the end


def test_target_reacquire_window_is_respected() -> None:
    statuses = {f: (_st(f, TGT, tid=42) if not (5 <= f <= 9) else _st(f, None, tid=None))
                for f in range(20)}
    gt_target = _gt_target(20)

    generous = evaluate_target(gt_target, statuses, [], _all_gt(20))
    assert generous["reacquire_rate"] == 1.0 and generous["reacquire_latency"] == pytest.approx(5.0)

    tight = evaluate_target(gt_target, statuses, [], _all_gt(20), EvalCfg(reacquire_window=3))
    assert tight["reacquire_rate"] == 0.0
    assert tight["reacquire_fail"] == 1
    assert tight["reacquire_censored"] == 0  # failed, but not because the video ended
    assert tight["reacquire_window"] == 3


def test_target_hallucination_is_not_false_acquisition() -> None:
    """Reporting a box that matches nobody is a hallucination, not a latch-on."""
    n = 12
    visible = set(range(0, 5)) | set(range(10, 12))
    statuses = {}
    for f in range(0, 5):
        statuses[f] = _st(f, TGT, tid=42)
    for f in range(5, 10):
        statuses[f] = _st(f, PHANTOM, tid=404)
    for f in range(10, 12):
        statuses[f] = _st(f, TGT, tid=42)
    r = evaluate_target(_gt_target(n, visible), statuses, [TargetEvent(0, "init", {"track_id": 42})],
                        _all_gt(n, target_frames=visible))

    assert r["n_visible_frames"] == 7 and r["n_report_frames"] == 12
    assert r["target_success_rate"] == 1.0      # it never failed while the target was visible
    assert r["loss_rate"] == 0.0
    assert r["false_acquisition_rate"] == 0.0   # nothing real to latch onto
    assert r["false_acquisition_frames"] == 0
    assert r["hallucination_frames"] == 5
    assert r["hallucination_rate"] == pytest.approx(5.0 / 12.0)
    assert r["target_precision"] == pytest.approx(7.0 / 12.0)
    assert r["target_precision_visible"] == 1.0
    assert r["fragmentation"] == 1
    assert r["mean_correct_streak"] == pytest.approx(3.5)
    assert r["n_correct_streaks"] == 2


def test_target_latches_onto_distractor_while_target_is_absent() -> None:
    """The DJI-critical case: perfect while visible, wrong person once it leaves."""
    n = 8
    visible = set(range(0, 4))
    statuses = {}
    for f in range(0, 4):
        statuses[f] = _st(f, TGT, tid=1)
    for f in range(4, 8):
        statuses[f] = _st(f, DIST, tid=2)
    r = evaluate_target(_gt_target(n, visible), statuses, [TargetEvent(0, "init", {"track_id": 1})],
                        _all_gt(n, target_frames=visible))

    assert r["target_success_rate"] == 1.0        # every visible frame was correct
    assert r["loss_rate"] == 0.0                  # ...and it never reported nothing
    assert r["n_report_frames"] == 8
    assert r["false_acquisition_rate"] == pytest.approx(0.5)   # 4/8 frames
    assert r["false_acquisition_frames"] == 4
    assert r["false_acquisition_events"] == 1
    assert r["hallucination_frames"] == 0         # it merged with a real, different person
    assert r["target_precision"] == pytest.approx(0.5)
    assert r["target_precision_visible"] == 1.0
    assert r["id_consistency"] == pytest.approx(0.5)
    assert r["target_switch_count"] == 1
    assert r["distinct_followed_ids"] == 2


def test_target_geometry_correct_but_identity_swaps() -> None:
    """Box is always right, so TSR is 1.0 — id_consistency is what drops."""
    n = 10
    statuses = {f: _st(f, TGT, tid=5 if f < 5 else 6) for f in range(n)}
    r = evaluate_target(_gt_target(n), statuses, [TargetEvent(0, "init", {"track_id": 5})], _all_gt(n))

    assert r["target_success_rate"] == 1.0
    assert r["false_acquisition_rate"] == 0.0
    assert r["id_consistency"] == pytest.approx(0.5)
    assert r["target_switch_count"] == 1
    assert r["distinct_followed_ids"] == 2
    assert r["longest_correct_streak"] == 10


def test_target_id_falls_back_to_geometry_without_diag() -> None:
    """No tracker-reported id: identity is inferred from the all-person GT."""
    n = 6
    statuses = {f: _st(f, TGT) for f in range(n)}   # no diag at all
    r = evaluate_target(_gt_target(n), statuses, [], _all_gt(n))
    assert r["target_success_rate"] == 1.0
    assert r["id_consistency"] == 1.0
    assert r["distinct_followed_ids"] == 1
    assert r["n_undetermined_id_frames"] == 0
    # Without all_gt there is no identity to check: undefined, not wrong.
    r2 = evaluate_target(_gt_target(n), statuses, [], {})
    assert r2["target_success_rate"] == 1.0
    assert math.isnan(r2["id_consistency"])
    assert r2["n_undetermined_id_frames"] == n
    assert r2["false_acquisition_rate"] == 0.0


def test_target_empty_and_absent_target_inputs() -> None:
    empty = evaluate_target({}, {}, [], {})
    assert empty["n_frames"] == 0
    for field in ("target_success_rate", "loss_rate", "false_acquisition_rate",
                  "id_consistency", "target_precision", "reacquire_rate"):
        assert math.isnan(empty[field]), field
    assert empty["longest_correct_streak"] == 0 and empty["fragmentation"] == 0

    # Target visible but the tracker says nothing at all.
    quiet = evaluate_target(_gt_target(3), {}, [], _all_gt(3))
    assert quiet["n_visible_frames"] == 3 and quiet["n_report_frames"] == 0
    assert quiet["target_success_rate"] == 0.0
    assert quiet["loss_rate"] == 1.0
    assert math.isnan(quiet["false_acquisition_rate"])
    assert math.isnan(quiet["target_precision"])
    assert quiet["reacquire_events"] == 1        # lost-while-visible run covering 0-2


# --------------------------------------------------------------------------- #
# 7. Per-frame debug dump
# --------------------------------------------------------------------------- #
def test_target_frame_table_and_csv(tmp_path) -> None:
    n = 6
    visible = set(range(0, 4))
    statuses = {0: _st(0, TGT, tid=1), 1: _st(1, DIST), 2: _st(2, None, tid=None),
                3: _st(3, PHANTOM), 4: _st(4, DIST, tid=2)}
    rows = target_frame_table(_gt_target(n, visible), statuses, _all_gt(n, target_frames=visible))
    assert len(rows) == n
    assert [r["frame"] for r in rows] == list(range(n))

    assert rows[0]["gt_present"] is True
    assert rows[0]["iou_target"] == 1.0
    assert rows[0]["correct"] is True and rows[0]["false_acquisition"] is False
    assert rows[0]["target_person_id"] == 1        # geometry identifies the target
    assert rows[0]["followed_id"] == 1 and rows[0]["followed_id_from_diag"] is True

    assert rows[1]["correct"] is False and rows[1]["false_acquisition"] is True
    assert rows[1]["best_gt_id"] == 2 and rows[1]["best_gt_iou"] == 1.0
    assert rows[1]["followed_id"] == 2 and rows[1]["followed_id_from_diag"] is False
    assert rows[1]["state"] == "tracking"

    assert rows[2]["reported"] is False and rows[2]["pred_xyxy"] is None
    assert math.isnan(rows[2]["iou_target"])
    assert rows[2]["state"] == "tracking"

    assert rows[3]["correct"] is False and rows[3]["false_acquisition"] is False
    assert rows[3]["best_gt_iou"] == 0.0
    # Frame 2 reports nothing although the target is visible: the row stays, the
    # IoU is undefined, and the target identity is still identifiable.
    assert rows[2]["target_person_id"] == 1 and rows[2]["target_person_id_iou"] == 1.0
    # Frame 4: the target is gone from the frame and the report latches onto the
    # distractor -> a false acquisition with no target box to score against.
    assert rows[4]["gt_present"] is False
    assert math.isnan(rows[4]["iou_target"])
    assert rows[4]["false_acquisition"] is True and rows[4]["correct"] is False
    assert rows[5]["reported"] is False

    path = write_frame_csv(tmp_path / "seq" / "target.csv", rows)
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == n
    assert back[0]["frame"] == "0" and back[0]["correct"] == "True"
    assert back[1]["false_acquisition"] == "True"
    assert back[2]["pred_xyxy"] == ""       # csv writes None as an empty field


# --------------------------------------------------------------------------- #
# 7b. Confident (locked) vs blind-coast false acquisition
# --------------------------------------------------------------------------- #
#: The tracker's own blind states, read from the shared config rather than
#: retyped here, so this test fails loudly if the state machine changes.
from tgtrack.config import StateCfg  # noqa: E402  (kept next to its use)

BLIND = tuple(StateCfg().frozen_states)


def test_locked_split_blind_coast_is_not_a_confident_false_acquisition() -> None:
    """A blind state means "not locked on anyone": the coast is not confident.

    This is motion-dev's measured failure — the tracker *refuses* two genuinely
    ambiguous candidates and then coasts onto the bystander.  It must still count
    in the headline ``false_acquisition_rate`` (it is a real latch) but it must
    NOT count as a confident one, because the tracker's own state says it is not
    holding anybody.
    """
    assert BLIND == ("occluded_soft", "lost_short", "lost_long", "out_of_frame")
    n = 6
    blind_states = {f: _st(f, DIST, tid=7, state="lost_short") for f in range(n)}
    r = evaluate_target(_gt_target(n), blind_states, [], _all_gt(n))

    # Real latch: counted in the headline rate.
    assert r["false_acquisition_rate"] == 1.0
    assert r["false_acquisition_frames"] == 6
    assert r["false_acquisition_events"] == 1            # one continuous latch
    # ...but never a confident one.
    assert r["false_acquisition_frames_locked"] == 0
    assert r["false_acquisition_events_locked"] == 0
    assert r["n_reported_locked"] == 0
    assert r["n_locked_frames"] == 0
    assert math.isnan(r["false_acquisition_rate_locked"])  # no locked reports at all
    assert r["loss_rate_locked"] == 0.0                    # nothing was claimed

    # Every frame is a loss (the tracker is on the wrong person) — total only.
    assert r["loss_rate"] == 1.0

    # Mixed run: 3 blinded frames then 3 confident ones -> the split is exact.
    mixed = {f: _st(f, DIST, tid=7, state=("lost_short" if f < 3 else "tracking"))
             for f in range(n)}
    m = evaluate_target(_gt_target(n), mixed, [], _all_gt(n))
    assert m["false_acquisition_frames"] == 6
    assert m["false_acquisition_frames_locked"] == 3
    assert m["false_acquisition_events_locked"] == 1
    assert m["n_reported_locked"] == 3 and m["n_locked_frames"] == 3
    assert m["false_acquisition_rate_locked"] == 1.0       # 3 locked reports, all wrong
    assert m["loss_rate_locked"] == pytest.approx(0.5)     # 3 of 6 visible frames


def test_locked_split_confident_wrong_lock_counts_in_both_rates() -> None:
    """The dangerous failure: state ``tracking``, accepted observation, wrong person.

    This one must never be excused by the split — it is a confident wrong lock,
    so it counts in the headline rate AND in the locked rate.
    """
    n = 6
    statuses = {f: _st(f, DIST, tid=7, state="tracking") for f in range(n)}
    r = evaluate_target(_gt_target(n), statuses, [], _all_gt(n))

    assert r["false_acquisition_rate"] == 1.0
    assert r["false_acquisition_rate_locked"] == 1.0
    assert r["false_acquisition_frames"] == 6
    assert r["false_acquisition_frames_locked"] == 6
    assert r["false_acquisition_events"] == 1
    assert r["false_acquisition_events_locked"] == 1
    assert r["n_reported_locked"] == 6 and r["n_locked_frames"] == 6
    assert r["loss_rate"] == 1.0 and r["loss_rate_locked"] == 1.0

    # A confident lock that is *correct* is not a failure anywhere.
    good = {f: _st(f, TGT, tid=7, state="tracking") for f in range(n)}
    g = evaluate_target(_gt_target(n), good, [], _all_gt(n))
    assert g["false_acquisition_rate"] == 0.0
    assert g["false_acquisition_rate_locked"] == 0.0
    assert g["loss_rate_locked"] == 0.0
    assert g["n_reported_locked"] == 6


def test_combine_target_pools_locked_split_by_counts_not_rates() -> None:
    """Pooling must sum counts and recompute — never average the rates."""
    # Seq A: 2 blinded latch frames + 2 confident ones (4 reports).
    a = evaluate_target(
        _gt_target(4),
        {f: _st(f, DIST, tid=7, state=("lost_short" if f < 2 else "tracking")) for f in range(4)},
        [], _all_gt(4),
    )
    # Seq B: 6 correct confident frames + 2 confident latch frames (8 reports).
    b_statuses = {f: _st(f, (DIST if f >= 6 else TGT), tid=7, state="tracking") for f in range(8)}
    b = evaluate_target(_gt_target(8), b_statuses, [], _all_gt(8))

    assert a["false_acquisition_rate_locked"] == 1.0       # 2 / 2
    assert b["false_acquisition_rate_locked"] == pytest.approx(0.25)   # 2 / 8

    out = combine_target([a, b])
    # Counts: (2 + 2) / (2 + 8) = 0.4.  Averaging the rates gives 0.625 instead.
    assert out["n_reported_locked"] == 10
    assert out["false_acquisition_frames_locked"] == 4
    assert out["false_acquisition_rate_locked"] == pytest.approx(0.4)
    assert out["false_acquisition_rate_locked"] != pytest.approx(0.625)
    assert out["false_acquisition_events_locked"] == 2
    # The total rate pools the same way and still counts every latch.
    assert out["false_acquisition_frames"] == 6
    assert out["false_acquisition_rate"] == pytest.approx(6.0 / 12.0)
    # loss_rate_locked: (2 + 2) locked losses / (4 + 8) visible = 1/3,
    # not the mean of 0.5 and 0.25.
    assert out["loss_frames_locked"] == 4
    assert out["loss_rate_locked"] == pytest.approx(1.0 / 3.0)
    assert out["n_locked_frames"] == 10

    # Single sequence and the empty case stay well defined.
    solo = combine_target([a])
    assert solo["false_acquisition_rate_locked"] == 1.0
    assert solo["n_locked_frames"] == 2
    nothing = combine_target([])
    assert nothing["n_sequences"] == 0
    assert nothing["false_acquisition_frames_locked"] == 0
    assert math.isnan(nothing["false_acquisition_rate_locked"])
    assert nothing["loss_frames_locked"] == 0
    assert math.isnan(nothing["loss_rate_locked"])


def test_frame_table_has_a_locked_column(tmp_path) -> None:
    """``locked`` must be inspectable frame by frame and land in the CSV."""
    n = 4
    statuses = {
        0: _st(0, TGT, tid=1, state="tracking"),
        1: _st(1, TGT, tid=1, state="lost_short"),
        2: _st(2, TGT, tid=1, state="reacquiring"),
        3: _st(3, None, tid=None, state="out_of_frame"),
    }
    rows = target_frame_table(_gt_target(n), statuses, _all_gt(n))
    assert [r["locked"] for r in rows] == [True, False, True, False]
    assert [r["state"] for r in rows] == ["tracking", "lost_short", "reacquiring", "out_of_frame"]
    # No status at all is no claim, so it is not a confident lock.
    bare = target_frame_table(_gt_target(1), {}, _all_gt(1))
    assert bare[0]["state"] is None and bare[0]["locked"] is False

    path = write_frame_csv(tmp_path / "frames.csv", rows)
    with path.open(encoding="utf-8", newline="") as fh:
        back = list(csv.DictReader(fh))
    assert [r["locked"] for r in back] == ["True", "False", "True", "False"]


# --------------------------------------------------------------------------- #
# 8. Cross-sequence aggregation (target)
# --------------------------------------------------------------------------- #
def test_combine_target_pools_counts() -> None:
    n = 10
    perfect = evaluate_target(
        _gt_target(n), {f: _st(f, TGT, tid=42) for f in range(n)},
        [TargetEvent(0, "init", {"track_id": 42})], _all_gt(n),
    )
    wrong = evaluate_target(
        _gt_target(n), {f: _st(f, DIST, tid=7) for f in range(n)},
        [TargetEvent(0, "init", {"track_id": 1})], _all_gt(n),
    )
    out = combine_target([perfect, wrong])

    assert out["n_sequences"] == 2
    assert out["n_frames"] == 20 and out["n_visible_frames"] == 20 and out["n_report_frames"] == 20
    assert out["target_success_rate"] == pytest.approx(0.5)
    assert out["loss_rate"] == pytest.approx(0.5)
    assert out["target_precision"] == pytest.approx(0.5)
    assert out["false_acquisition_rate"] == pytest.approx(0.5)
    assert out["false_acquisition_frames"] == 10
    assert out["false_acquisition_events"] == 1
    assert out["id_consistency"] == pytest.approx(0.5)
    assert out["reacquire_events"] == 0
    assert math.isnan(out["reacquire_rate"])

    solo = combine_target([perfect])
    assert solo["target_success_rate"] == 1.0
    assert solo["false_acquisition_rate"] == 0.0
    assert solo["target_precision"] == 1.0
    assert solo["id_consistency"] == 1.0
    assert combine_target([])["n_sequences"] == 0


# --------------------------------------------------------------------------- #
# 9. Optional cross-check against a real TrackEval install
# --------------------------------------------------------------------------- #
def test_trackeval_cross_check_when_available() -> None:
    """If ``trackeval`` is importable, its numbers must match ours exactly."""
    trackeval = pytest.importorskip("trackeval")

    gt, pred = _fixture("with_confusion")
    data = mot_frame_data(gt, pred)
    mine = evaluate_mot(gt, pred)
    try:
        clear = trackeval.metrics.CLEAR()
        res_clear = clear.eval_sequence(data)
        identity = trackeval.metrics.Identity()
        res_id = identity.eval_sequence(data)
        hota = trackeval.metrics.HOTA()
        out = hota.eval_sequence(data)
        res_hota = out[0] if isinstance(out, tuple) else out
        if not isinstance(out, tuple) and hasattr(hota, "compute_final_fields"):
            hota.compute_final_fields(res_hota)
    except Exception as exc:  # pragma: no cover - depends on the installed version
        pytest.skip(f"trackeval API mismatch: {exc!r}")

    assert float(res_clear["MOTA"]) == pytest.approx(mine["MOTA"])
    assert float(res_clear["IDSW"]) == pytest.approx(mine["IDSW"])
    assert float(res_id["IDF1"]) == pytest.approx(mine["IDF1"])
    assert float(np.mean(res_hota["HOTA"])) == pytest.approx(mine["HOTA"])
