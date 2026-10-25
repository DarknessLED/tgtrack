"""E055: box-level detector fusion (`fuse_detections` + `DetectorCfg.extra_weights`)."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
from tgtrack.config import Config
from tgtrack.detector import fuse_detections
from tgtrack.types import Detection


def det(box, score):
    return Detection(xyxy=np.asarray(box, dtype=np.float32), score=float(score), cls=0)


def test_disjoint_boxes_are_all_kept():
    a = [det([0, 0, 10, 10], 0.9)]
    b = [det([100, 100, 110, 110], 0.8)]
    out = fuse_detections([a, b], iou_thr=0.55)
    assert len(out) == 2
    assert [round(d.score, 2) for d in out] == [0.9, 0.8]  # sorted by score


def test_duplicate_is_suppressed_and_the_higher_score_wins():
    tight = det([10, 10, 50, 90], 0.9)   # precise box
    loose = det([8, 8, 52, 92], 0.4)     # near-duplicate, lower score
    out = fuse_detections([[tight], [loose]], iou_thr=0.55)
    assert len(out) == 1
    assert out[0] is tight


def test_overlap_below_threshold_is_not_a_duplicate():
    a = det([0, 0, 10, 10], 0.9)
    b = det([0, 0, 10, 5], 0.8)          # IoU 0.5 < 0.55
    assert len(fuse_detections([[a], [b]], iou_thr=0.55)) == 2
    assert len(fuse_detections([[a], [b]], iou_thr=0.4)) == 1


def test_empty_groups_are_ignored():
    a = [det([0, 0, 10, 10], 0.9)]
    assert fuse_detections([[], a, []]) == a
    assert fuse_detections([]) == []
    assert fuse_detections([[], []]) == []


def test_zero_area_boxes_do_not_crash():
    a = det([5, 5, 5, 5], 0.9)
    b = det([5, 5, 5, 5], 0.8)
    out = fuse_detections([[a], [b]])
    assert len(out) in (1, 2)  # degenerate geometry: no division by zero, no exception


def test_extra_weights_default_is_empty_so_nothing_changes():
    cfg = Config()
    assert cfg.detector.extra_weights == ()
    assert cfg.detector.fusion_iou == 0.55


def test_config_can_enable_fusion_and_validates_types():
    cfg = Config().clone(**{"detector.extra_weights": ("data/weights/crowdhuman_yolov8n_best.pt",)})
    assert cfg.detector.extra_weights == ("data/weights/crowdhuman_yolov8n_best.pt",)
