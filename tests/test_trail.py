"""Tests for the E059 motion-signature ("trail") evidence channel."""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.trail import (
    TrailLinker,
    box_trail,
    trail_descriptor,
    trail_distance,
    velocity_disagreement,
)
from tgtrack.types import Detection

STEPS = 8


def _walk(n: int = STEPS + 1, vx: float = 10.0, h: float = 100.0) -> np.ndarray:
    """A straight walk as (N, 3) = (cx, cy, height)."""
    t = np.arange(n, dtype=np.float64)
    out = np.zeros((n, 3))
    out[:, 0] = 100.0 + vx * t
    out[:, 1] = 200.0 + 0.0 * t
    out[:, 2] = h
    return out


def test_descriptor_rejects_short_and_degenerate_trails():
    assert trail_descriptor(_walk(), steps=STEPS) is not None
    assert trail_descriptor(_walk(n=STEPS), steps=STEPS) is None  # needs steps+1 points
    zero_h = _walk()
    zero_h[:, 2] = 0.0
    assert trail_descriptor(zero_h, steps=STEPS) is None
    nan = _walk()
    nan[3, 0] = np.nan
    assert trail_descriptor(nan, steps=STEPS) is None


def test_descriptor_is_translation_invariant():
    a = trail_descriptor(_walk(), steps=STEPS)
    b = trail_descriptor(_walk() + np.array([5000.0, -3000.0, 0.0]), steps=STEPS)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_descriptor_is_scale_and_rotation_invariant():
    base = _walk()
    a = trail_descriptor(base, steps=STEPS)
    # scale positions AND heights together: a 2x person walking twice as far
    scaled = base.copy()
    scaled[:, :2] *= 2.0
    scaled[:, 2] *= 2.0
    np.testing.assert_allclose(a, trail_descriptor(scaled, steps=STEPS), atol=1e-6)
    # rotate every displacement by a constant angle: turning angles are unchanged
    ang = 0.7
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    rotated = base.copy()
    rotated[:, :2] = (base[:, :2] - base[0, :2]) @ rot.T + base[0, :2]
    np.testing.assert_allclose(a, trail_descriptor(rotated, steps=STEPS), atol=1e-5)


def test_descriptor_separates_a_turning_person_from_a_straight_one():
    straight = trail_descriptor(_walk(), steps=STEPS)
    # same speed, but curving: distinct turn pattern
    t = np.arange(STEPS + 1, dtype=np.float64)
    ang = 0.25 * t
    curve = np.zeros((STEPS + 1, 3))
    curve[:, 0] = 100.0 + 10.0 * np.cos(ang)
    curve[:, 1] = 200.0 + 10.0 * np.sin(ang)
    curve[:, 2] = 100.0
    turning = trail_descriptor(curve, steps=STEPS)
    assert trail_distance(straight, straight) == pytest.approx(0.0)
    assert trail_distance(straight, turning) > 0.05


def test_distance_handles_missing_evidence_as_infinity():
    d = trail_descriptor(_walk(), steps=STEPS)
    assert trail_distance(None, d) == float("inf")
    assert trail_distance(d, None) == float("inf")
    assert trail_distance(None, None) == float("inf")
    assert trail_distance(d, np.ones(3)) == float("inf")  # shape mismatch, not a crash


def test_box_trail_uses_centre_and_height():
    boxes = [np.array([0.0, 0.0, 10.0, 20.0]), np.array([10.0, 0.0, 20.0, 20.0])]
    pts = box_trail(boxes)
    np.testing.assert_allclose(pts[:, 0], [5.0, 15.0])
    np.testing.assert_allclose(pts[:, 1], [10.0, 10.0])
    np.testing.assert_allclose(pts[:, 2], [20.0, 20.0])
    assert box_trail([]) is None


def _det(x0: float, w: float = 40.0, h: float = 100.0, y0: float = 100.0) -> Detection:
    return Detection(xyxy=np.array([x0, y0, x0 + w, y0 + h], dtype=np.float64), score=0.9)


def test_linker_builds_a_descriptor_only_after_enough_frames():
    linker = TrailLinker(max_len=16, min_len=STEPS, iou_link=0.2)
    seen = []
    for k in range(STEPS + 2):
        desc = linker.update([_det(10.0 * k)], frame_idx=k)
        seen.append(desc[0])
    assert all(d is None for d in seen[:STEPS])  # not enough history yet
    assert seen[STEPS + 1] is not None


def test_linker_keeps_identity_across_a_short_gap_and_resets_on_a_long_one():
    linker = TrailLinker(max_len=16, min_len=3, iou_link=0.2)
    for k in range(5):
        linker.update([_det(10.0 * k)], frame_idx=k)
    assert len(linker) == 1
    # one skipped frame: still the same tube (contiguous enough to link)
    linker.update([_det(10.0 * 5)], frame_idx=6)
    assert len(linker) == 1
    # a long gap must throw the tube away: turning angles across it are meaningless
    linker.update([_det(10.0 * 9)], frame_idx=12)
    assert len(linker) == 1  # replaced, and the stale history was dropped
    assert linker.update([_det(10.0 * 10)], frame_idx=13)[0] is None  # history restarted


def test_linker_separates_two_detections_into_two_tracks():
    linker = TrailLinker(max_len=16, min_len=4, iou_link=0.2)
    a0, b0 = _det(0.0), _det(200.0)
    for k in range(6):
        linker.update([_det(10.0 * k), _det(200.0 + 10.0 * k)], frame_idx=k)
    assert len(linker) == 2
    # they should have different descriptors because they started far apart in time
    da, db = linker.update([_det(10.0 * 6), _det(200.0 + 10.0 * 6)], frame_idx=6)
    assert da is not None and db is not None


def test_linker_exposes_a_per_frame_tube_velocity():
    linker = TrailLinker(max_len=16, min_len=3, iou_link=0.2)
    for k in range(4):
        linker.update([_det(10.0 * k), _det(500.0 - 10.0 * k)], frame_idx=k)
    vels = linker.vels
    assert len(vels) == 2
    # heights are 100, so 10 px/frame is 0.1 body heights per frame
    np.testing.assert_allclose(vels[0], [0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(vels[1], [-0.1, 0.0], atol=1e-6)


def test_velocities_are_none_before_a_tube_has_two_points():
    linker = TrailLinker(max_len=16, min_len=3, iou_link=0.2)
    vels = linker.update([_det(0.0), _det(300.0)], frame_idx=0)
    assert vels == [None, None]
    assert linker.vels == [None, None]


def test_velocity_is_aligned_to_the_frame_that_was_just_passed():
    linker = TrailLinker(max_len=16, min_len=3, iou_link=0.2)
    linker.update([_det(100.0)], frame_idx=0)
    linker.update([_det(110.0)], frame_idx=1)
    linker.update([_det(120.0), _det(600.0)], frame_idx=2)
    # the new detection on the right is a fresh tube: no velocity yet, and the
    # list must stay aligned to the detections of THIS frame
    assert len(linker.vels) == 2
    assert linker.vels[0] is not None and linker.vels[1] is None


def test_velocity_disagreement_is_zero_when_the_headings_agree():
    assert velocity_disagreement(np.array([0.05, 0.0]), np.array([0.02, 0.0])) == pytest.approx(0.0)
    assert velocity_disagreement(np.array([0.03, 0.0]), np.array([0.01, 0.0])) == pytest.approx(0.0)


def test_velocity_disagreement_is_one_when_the_headings_oppose():
    assert velocity_disagreement(np.array([-0.05, 0.0]), np.array([0.02, 0.0])) == pytest.approx(1.0)
    assert velocity_disagreement(np.array([0.0, -0.04]), np.array([0.0, 0.01])) == pytest.approx(1.0)


def test_velocity_disagreement_is_half_for_a_right_angle():
    d = velocity_disagreement(np.array([0.0, 0.05]), np.array([0.02, 0.0]))
    assert d == pytest.approx(0.5)


def test_velocity_disagreement_treats_missing_or_negligible_evidence_as_agreement():
    # no tube, no target velocity, or a stationary tube: never punish a candidate
    assert velocity_disagreement(None, np.array([0.02, 0.0])) == 0.0
    assert velocity_disagreement(np.array([0.05, 0.0]), None) == 0.0
    assert velocity_disagreement(np.array([0.0, 0.0]), np.array([0.02, 0.0])) == 0.0
    assert velocity_disagreement(np.array([0.05, 0.0]), np.array([0.001, 0.0])) == 0.0
    assert velocity_disagreement(np.array([np.nan, 0.0]), np.array([0.02, 0.0])) == 0.0
    # a slower-than-target but aligned heading is still agreement
    assert velocity_disagreement(np.array([0.005, 0.0]), np.array([0.05, 0.0])) == pytest.approx(0.0)
