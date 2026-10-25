"""Tests for MOTChallenge ignore-region handling (E008/E009).

MOT17's public GT marks distractor pedestrians, static persons and reflections with
`conf=0`. MOTChallenge/TrackEval never score a detection that lands on such a region and
matches no valid box; `tgtrack.data.load_mot_sequence(min_conf=1.0)` drops those rows, so
without this the harness charges a measured 33.5 % of our MOT17 false positives against us.
The two invariants that make the fix safe are pinned here:

1. a box that is already a valid-GT match is **never** excused (so recall cannot be inflated);
2. with no ignore rows present, predictions pass through byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from eval_mot import excuse_ignore_regions, load_ignore_boxes  # noqa: E402

from tgtrack.evaluate import GtBox, PredBox  # noqa: E402


def _pred(frame: int, xyxy, tid: int = 1, score: float = 0.9) -> PredBox:
    return PredBox(frame=frame, track_id=tid, xyxy=np.asarray(xyxy, np.float64), score=score)


def _gt(frame: int, xyxy, tid: int = 1) -> GtBox:
    return GtBox(frame=frame, track_id=tid, xyxy=np.asarray(xyxy, np.float64))


# --------------------------------------------------------------------------- #
# load_ignore_boxes
# --------------------------------------------------------------------------- #
def test_load_ignore_boxes_uses_1_based_file_frames_and_renumbers(tmp_path: Path) -> None:
    """gt.txt is 1-based; the run is 0-based and renumbered by `subsample`.

    Kept original frames are o = 0, 3, 6, ... -> file frames 1, 4, 7, ... -> indices 0, 1, 2.
    """
    gt = tmp_path / "gt.txt"
    gt.write_text(
        "1,1,0,0,10,20,0,1,1\n"      # o=0 -> index 0, ignore
        "4,2,50,0,10,20,1,1,1\n"     # o=3 -> valid row, never an ignore box
        "7,3,100,0,10,20,0,1,1\n"    # o=6 -> index 2, ignore
        "8,4,200,0,10,20,0,1,1\n",   # o=7 -> not sampled, dropped
        encoding="utf-8",
    )
    boxes = load_ignore_boxes(gt, subsample=3)
    assert set(boxes) == {0, 2}, boxes
    assert boxes[0][0].tolist() == [0.0, 0.0, 10.0, 20.0]
    assert boxes[2][0].tolist() == [100.0, 0.0, 110.0, 20.0]


def test_load_ignore_boxes_respects_max_frames(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,0,0,10,20,0,1,1\n7,3,100,0,10,20,0,1,1\n", encoding="utf-8")
    assert set(load_ignore_boxes(gt, subsample=3, max_frames=2)) == {0}


def test_load_ignore_boxes_subsample_one_is_identity_remap(tmp_path: Path) -> None:
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,0,0,10,20,0,1,1\n2,2,50,0,10,20,0,1,1\n", encoding="utf-8")
    assert set(load_ignore_boxes(gt, subsample=1)) == {0, 1}


def test_load_ignore_boxes_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_ignore_boxes(tmp_path / "nope.txt") == {}


# --------------------------------------------------------------------------- #
# excuse_ignore_regions
# --------------------------------------------------------------------------- #
def test_excuses_a_detection_that_lands_on_an_ignore_region() -> None:
    gt = {0: [_gt(0, [0, 0, 50, 100])]}
    pred = {0: [_pred(0, [100, 0, 150, 100], tid=7)]}
    ignore = {0: np.asarray([[100.0, 0.0, 150.0, 100.0]])}

    out, excused = excuse_ignore_regions(pred, gt, ignore)

    assert excused == 1
    assert out == {}


def test_never_excuses_a_box_that_matches_valid_gt() -> None:
    """The critical safety property: an overlap with an ignore region must not remove a TP."""
    box = [100, 0, 150, 100]
    gt = {0: [_gt(0, box)]}
    pred = {0: [_pred(0, box, tid=3)]}
    ignore = {0: np.asarray([[100.0, 0.0, 150.0, 100.0]])}

    out, excused = excuse_ignore_regions(pred, gt, ignore)

    assert excused == 0
    assert out[0][0].track_id == 3


def test_keeps_a_detection_that_overlaps_nothing() -> None:
    gt = {0: [_gt(0, [0, 0, 10, 20])]}
    pred = {0: [_pred(0, [500, 500, 540, 600], tid=9)]}
    ignore = {0: np.asarray([[0.0, 0.0, 10.0, 20.0]])}

    out, excused = excuse_ignore_regions(pred, gt, ignore)

    assert excused == 0
    assert len(out[0]) == 1


def test_no_ignore_regions_is_a_pass_through() -> None:
    gt = {0: [_gt(0, [0, 0, 10, 20])]}
    pred = {0: [_pred(0, [0, 0, 10, 20])], 1: [_pred(1, [300, 300, 340, 420], tid=2)]}

    out, excused = excuse_ignore_regions(pred, gt, {})

    assert excused == 0
    assert out is pred


def test_iou_threshold_is_respected() -> None:
    """A box that only grazes an ignore region is not excused at the stricter IoU."""
    gt: dict = {}
    ignore = {0: np.asarray([[100.0, 0.0, 200.0, 100.0]])}
    # shifted by half the width -> IoU ~0.33
    pred = {0: [_pred(0, [150, 0, 250, 100], tid=5)]}

    loose, excused_loose = excuse_ignore_regions(pred, gt, ignore, iou_thresh=0.3)
    strict, excused_strict = excuse_ignore_regions(pred, gt, ignore, iou_thresh=0.5)

    assert excused_loose == 1 and loose == {}
    assert excused_strict == 0 and len(strict[0]) == 1


def test_frame_with_only_ignore_rows_does_not_crash() -> None:
    gt = {0: []}
    pred = {0: [_pred(0, [0, 0, 10, 20])]}
    ignore = {0: np.asarray([[0.0, 0.0, 10.0, 20.0]])}

    out, excused = excuse_ignore_regions(pred, gt, ignore)

    assert excused == 1 and out == {}
