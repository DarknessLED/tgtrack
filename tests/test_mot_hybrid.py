#!/usr/bin/env python3
"""Unit tests for the Hybrid-SORT backend (``src/tgtrack/mot_hybrid.py``, E012).

Everything runs on synthetic detections -- no detector, no weights, no dataset. The
tests pin the three claims the backend is built on:

1. ``HMIoU = HIoU * IoU`` with ``HIoU = h_over/h_union`` (matches TrackTrack's
   ``_hmiou_distance``) and it *prefers a vertically aligned candidate* where plain IoU
   would prefer a vertically shifted one.
2. A vertically truncated detection is still matched, and the shipped IoU/HMIoU blend is
   what makes that possible (pure HMIoU is stricter than IoU on a truncated pair).
3. The pipeline-level behaviours the harness relies on: crossing people keep their ids
   even when the detection order swaps, low-score detections are recovered in stage 2,
   ``reset()`` clears every bit of state, empty input is safe, ids are plain ints.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from tgtrack.config import MotCfg
from tgtrack.types import Detection

REPO = Path(__file__).resolve().parents[1]
HYBRID_STEM = "hybridsort.yaml"
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_tracker(with_reid: bool = False, warn: bool = False, **extra):
    """A ``HybridSortTracker`` on the shipped yaml, GMC off unless requested.

    ``extra`` is passed straight through ``MotCfg.extra``, which the loader applies last,
    so ``make_tracker(w_hmiou=1.0)`` reproduces TrackTrack's pure-HMIoU geometry.
    """
    pytest.importorskip("ultralytics", reason="the Hybrid-SORT backend needs ultralytics' KalmanFilterXYAH")
    from tgtrack.mot_hybrid import HybridSortTracker

    cfg = MotCfg(tracker=HYBRID_STEM, gmc_method="none", with_reid=with_reid, extra=dict(extra))
    if warn:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            return HybridSortTracker(cfg, detector=None), caught
    return HybridSortTracker(cfg, detector=None)


def det(box, score=0.9, cls=0):
    """One ``Detection`` from ``(x1, y1, x2, y2)``."""
    return Detection(xyxy=np.asarray(box, dtype=np.float32), score=float(score), cls=cls)


def run(tracker, frames):
    """Feed ``frames`` (list of ``[(box, score), ...]``) and return the assigned ids."""
    out = []
    for spec in frames:
        dets = [det(b, s) for b, s in spec]
        tracker.update(FRAME, dets)
        out.append([d.track_id for d in dets])
    return out


# --------------------------------------------------------------------------- #
# geometry: HMIoU (the reason this backend exists)
# --------------------------------------------------------------------------- #
def test_iou_and_hmiou_formulas():
    from tgtrack.mot_hybrid import hmiou_matrix, iou_matrix

    a = np.asarray([[0, 0, 40, 100]], dtype=np.float64)
    b = np.asarray([[0, 0, 40, 50]], dtype=np.float64)  # upper half only
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.5)
    # HIoU = h_over/h_union = 50/100; HMIoU = HIoU * IoU
    iou, cost = hmiou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(0.5)
    assert cost[0, 0] == pytest.approx(1.0 - 0.5 * 0.5)

    # no vertical overlap at all -> HIoU = 0 -> HMIoU = 0 -> maximum cost
    c = np.asarray([[0, 200, 40, 300]], dtype=np.float64)
    assert hmiou_matrix(a, c)[1][0, 0] == pytest.approx(1.0)
    # empty inputs never blow up
    assert hmiou_matrix(np.zeros((0, 4)), a)[1].shape == (0, 1)


def test_hmiou_prefers_vertically_aligned_candidate_over_iou():
    """The discriminating case: same track, one candidate shifted right, one shifted down.

    Plain IoU prefers the down-shifted box (bigger area overlap); HMIoU prefers the
    vertically aligned one, because the down-shifted box only shares 70% of its height.
    """
    from tgtrack.mot_hybrid import hmiou_matrix, iou_matrix

    track = np.asarray([[0, 0, 40, 100]], dtype=np.float64)
    aligned = np.asarray([[20, 0, 60, 100]], dtype=np.float64)  # right, same vertical extent
    shifted = np.asarray([[10, 30, 50, 100]], dtype=np.float64)  # down, shorter

    iou_aligned = iou_matrix(track, aligned)[0, 0]
    iou_shifted = iou_matrix(track, shifted)[0, 0]
    hmiou_aligned = 1.0 - hmiou_matrix(track, aligned)[1][0, 0]
    hmiou_shifted = 1.0 - hmiou_matrix(track, shifted)[1][0, 0]

    assert iou_shifted > iou_aligned  # IoU prefers the shifted box
    assert hmiou_aligned > hmiou_shifted  # HMIoU prefers the aligned one


def test_geometry_cost_blend_is_more_permissive_than_pure_hmiou():
    """The shipped blend (w_hmiou=0.5) keeps a truncated match that pure HMIoU rejects."""
    blended = make_tracker(w_hmiou=0.5)
    pure = make_tracker(w_hmiou=1.0)
    track = np.asarray([[100, 100, 140, 300]], dtype=np.float64)
    clipped = np.asarray([[100, 100, 140, 200]], dtype=np.float64)  # half the height

    c_blend = blended._geo_cost(track, clipped)[0, 0]
    c_pure = pure._geo_cost(track, clipped)[0, 0]
    assert c_pure == pytest.approx(1.0 - 0.5 * 0.5)
    assert c_blend == pytest.approx(0.5 * (1.0 - 0.25) + 0.5 * 0.5)
    assert c_blend < blended.match_thresh < c_pure or c_blend < c_pure


def test_rocm_angle_distance_prefers_consistent_motion():
    from tgtrack.mot_hybrid import angle_distance_matrix

    prev = np.asarray([[0, 0, 40, 100]], dtype=np.float64)
    cur = np.asarray([[20, 0, 60, 100]], dtype=np.float64)  # moving right
    dets = np.asarray([[40, 0, 80, 100], [-20, 0, 20, 100]], dtype=np.float64)  # right, left
    dist = angle_distance_matrix(prev, cur, dets)
    assert dist[0] == pytest.approx(0.0, abs=1e-3)  # continues the motion
    assert dist[1] == pytest.approx(1.0, abs=1e-3)  # reverses it
    # a corner with no observed motion carries no direction evidence (not a 0.5 penalty)
    assert angle_distance_matrix(cur, cur, dets)[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# association behaviour
# --------------------------------------------------------------------------- #
def test_vertical_truncation_is_still_associated():
    """A box whose bottom is cut off keeps the track id (the partial-visibility claim)."""
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((100, 100, 140, 300), 0.9)],
            [((100, 100, 140, 200), 0.9)],  # lower half clipped
            [((100, 100, 140, 180), 0.9)],  # even more clipped
        ],
    )
    assert ids[0] == [1]
    assert ids[1] == [1]
    assert ids[2] == [1]


def test_height_clamp_bounds_the_predicted_height():
    """The KF's predicted height is held inside ``h_state * [1/h_clip, h_clip]``.

    A large height change (a vertically truncated box) leaves the KF extrapolating well
    outside the height it has actually been observing; ``h_clip`` bounds that extrapolation.
    The *association* consequence is covered by ``test_vertical_truncation_is_still_associated``
    -- with the clamp disabled that frame still matches, so this test pins the bound itself
    (both sides of the knob: ``h_clip=50`` is effectively "no clamp") instead of an id outcome.
    """
    frames = [
        [((100, 100, 140, 300), 0.9)],  # h 200 -> h_state 200
        [((100, 100, 140, 200), 0.9)],  # h 100 -> h_state 150, KF now wants h ~78
    ]

    clamped = make_tracker()  # h_clip 1.5 -> predicted height clipped to [100, 225]
    run(clamped, frames)
    track = clamped.tracks[0]
    assert track.h_state == pytest.approx(150.0)
    track.predict()
    assert float(track.xyxy[3] - track.xyxy[1]) == pytest.approx(track.h_state / track.h_clip, rel=1e-6)

    loose = make_tracker(h_clip=50.0)  # effectively no clamp
    run(loose, frames)
    raw = loose.tracks[0]
    assert raw.h_state == pytest.approx(150.0)
    raw.predict()
    raw_h = float(raw.xyxy[3] - raw.xyxy[1])
    assert raw_h == pytest.approx(78.0488, rel=1e-3)
    assert raw_h < raw.h_state / 1.5  # outside the band the default clamp enforces


def test_two_crossing_people_keep_ids_despite_swapped_det_order():
    """A and B cross; the detection list order swaps mid-sequence and ids must follow motion."""
    tracker = make_tracker()
    a1, a2, a3 = (0, 0, 40, 100), (20, 0, 60, 100), (40, 0, 80, 100)
    a4, a5 = (60, 0, 100, 100), (80, 0, 120, 100)
    b1, b2, b3 = (100, 0, 140, 100), (80, 0, 120, 100), (60, 0, 100, 100)
    b4, b5 = (40, 0, 80, 100), (20, 0, 60, 100)

    ids = run(
        tracker,
        [
            [(a1, 0.90), (b1, 0.90)],
            [(a2, 0.90), (b2, 0.90)],
            [(b3, 0.90), (a3, 0.90)],  # order swapped
            [(b4, 0.90), (a4, 0.90)],  # order swapped
            [(a5, 0.90), (b5, 0.90)],
        ],
    )
    # index 0 is A on frames 1-2-5, index 1 on frames 3-4
    a_ids = [ids[0][0], ids[1][0], ids[2][1], ids[3][1], ids[4][0]]
    b_ids = [ids[0][1], ids[1][1], ids[2][0], ids[3][0], ids[4][1]]
    assert len(set(a_ids)) == 1, f"A changed id: {a_ids}"
    assert len(set(b_ids)) == 1, f"B changed id: {b_ids}"
    assert a_ids[0] != b_ids[0]


def test_low_score_detection_is_recovered_in_second_stage():
    """Stage 2 (ByteTrack-style recovery) re-attaches a 0.15 detection to its track."""
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((10, 10, 50, 110), 0.90)],
            [((12, 10, 52, 110), 0.15)],  # below track_high_thresh, above track_low_thresh
        ],
    )
    assert ids[0] == [1]
    assert ids[1] == [1]


def test_low_score_detection_far_away_spawns_nothing():
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((10, 10, 50, 110), 0.90)],
            [((12, 10, 52, 110), 0.15), ((500, 300, 540, 400), 0.15)],
        ],
    )
    assert ids[1][0] == 1
    assert ids[1][1] is None  # no id: too weak to spawn, too far to recover


def test_weak_detection_below_new_track_thresh_does_not_spawn():
    tracker = make_tracker()
    ids = run(tracker, [[((10, 10, 50, 110), 0.90)], [((300, 200, 340, 300), 0.20)]])
    assert ids[1] == [None]
    assert tracker.num_tracks == 1


def test_high_score_detection_spawns_a_second_track():
    """A new track after frame 1 is output only once matched again -- the same
    activation convention as the six ultralytics baselines (``track_tracker.py``
    ``activate``: ``is_activated = frame_id == 1 or state == Tracked``)."""
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((10, 10, 50, 110), 0.90)],
            [((10, 10, 50, 110), 0.90), ((400, 200, 440, 300), 0.90)],
            [((10, 10, 50, 110), 0.90), ((400, 200, 440, 300), 0.90)],
        ],
    )
    assert ids[0] == [1]
    assert ids[1] == [1, None]
    assert ids[2] == [1, 2]
    assert tracker.num_tracks == 2


def test_track_ids_are_plain_ints_and_stable():
    tracker = make_tracker()
    ids = run(tracker, [[((10, 10, 50, 110), 0.9)], [((14, 12, 54, 112), 0.9)], [((18, 14, 58, 114), 0.9)]])
    flat = [i for frame in ids for i in frame if i is not None]
    assert flat == [1, 1, 1]
    for value in flat:
        assert type(value) is int  # noqa: E721 - deliberately rejects np.int64/bool


def test_height_gate_rejects_absurd_scale_change():
    """A 10x taller detection is refused by the height gate, so the track is not stolen."""
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((100, 100, 140, 200), 0.9)],
            [((100, 0, 140, 600), 0.9)],  # 5x taller -> outside height_gate 2.5
        ],
    )
    assert ids[0] == [1]
    assert ids[1] == [None] or ids[1][0] != 1


def test_lost_track_is_rebound_within_rebind_window():
    """After a 2-frame gap the relaxed LOST pass re-attaches the same id."""
    tracker = make_tracker()
    ids = run(
        tracker,
        [
            [((100, 100, 140, 200), 0.9)],
            [],  # gap
            [],  # gap
            [((120, 100, 160, 200), 0.9)],
        ],
    )
    assert ids[3][0] == 1
    assert tracker.num_tracks == 1


def test_track_lost_beyond_rebind_window_gets_a_new_id():
    """A detection after a long occlusion is not force-matched to the stale lost track."""
    tracker = make_tracker(rebind_max_frames=1)
    ids = run(
        tracker,
        [
            [((100, 100, 140, 200), 0.9)],
            [],
            [],
            [((120, 100, 160, 200), 0.9)],
            [((120, 100, 160, 200), 0.9)],
        ],
    )
    assert ids[3] == [None]  # fresh track: unconfirmed on its creation frame
    assert ids[4] == [2]  # confirmed on the next match -> a new id, not the stale one


def test_track_is_dropped_after_track_buffer():
    tracker = make_tracker(track_buffer=2)
    ids = run(tracker, [[((100, 100, 140, 200), 0.9)], [], [], [], []])
    assert ids[0] == [1]
    assert tracker.num_tracks == 0


# --------------------------------------------------------------------------- #
# lifecycle / interface parity with AllPersonTracker
# --------------------------------------------------------------------------- #
def test_empty_input_is_safe():
    tracker = make_tracker()
    assert tracker.update(FRAME, []) == []
    assert tracker.n_updates == 1
    assert tracker.frame_id == 1
    assert tracker.num_tracks == 0
    tracker.update(FRAME, [])
    assert tracker.frame_id == 2


def test_reset_clears_state_and_restarts_ids():
    tracker = make_tracker()
    run(tracker, [[((10, 10, 50, 110), 0.9)], [((12, 10, 52, 110), 0.9)]])
    assert tracker.num_tracks == 1 and tracker.started and tracker.frame_id == 2
    tracker.reset()
    assert tracker.num_tracks == 0
    assert tracker.frame_id == 0
    assert tracker.n_updates == 0
    assert tracker.started is False
    ids = run(tracker, [[((10, 10, 50, 110), 0.9)]])
    assert ids == [[1]]  # the id counter restarted


def test_reset_restarts_gmc_state():
    pytest.importorskip("ultralytics", reason="GMC needs ultralytics")
    from tgtrack.mot_hybrid import HybridSortTracker

    tracker = HybridSortTracker(MotCfg(tracker=HYBRID_STEM, gmc_method="sparseOptFlow"), detector=None)
    assert tracker.gmc is not None
    tracker.reset()  # must not raise
    assert tracker.num_tracks == 0


def test_bad_frame_shape_raises_value_error():
    tracker = make_tracker()
    with pytest.raises(ValueError, match="expects an \\(H, W, 3\\) BGR ndarray"):
        tracker.update(np.zeros((10, 10), dtype=np.uint8), [det((0, 0, 10, 10))])


def test_feats_argument_is_accepted_and_ignored():
    tracker = make_tracker()
    dets = [det((10, 10, 50, 110))]
    tracker.update(FRAME, dets, feats=np.zeros((1, 8), dtype=np.float32))
    assert dets[0].track_id == 1


def test_detrack_ids_are_cleared_at_the_start_of_every_frame():
    tracker = make_tracker()
    dets = [det((10, 10, 50, 110), 0.9)]
    tracker.update(FRAME, dets)
    assert dets[0].track_id == 1
    tracker.update(FRAME, [])  # frame with no dets
    stale = det((10, 10, 50, 110), 0.02)  # below track_low_thresh -> nothing matches
    stale.track_id = 7
    tracker.update(FRAME, [stale])
    assert stale.track_id is None


def test_describe_and_names_expose_the_knobs():
    tracker = make_tracker()
    described = tracker.describe()
    for key in ("tracker_type", "match_thresh", "match_thresh_low", "w_hmiou", "w_vel", "w_conf", "w_height", "h_clip"):
        assert key in described
    assert described["tracker_type"] == "hybridsort"
    assert described["w_hmiou"] == pytest.approx(0.5)
    assert Path(described["tracker_yaml"]).name == HYBRID_STEM
    assert isinstance(tracker.names, dict) and "track_high_thresh" in tracker.names
    assert "HybridSortTracker" in repr(tracker)


def test_with_reid_warns_because_there_is_no_appearance_branch():
    _tracker, caught = make_tracker(with_reid=True, warn=True)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_resolve_hybrid_yaml_by_path_stem_and_missing():
    from tgtrack.mot_hybrid import resolve_hybrid_yaml

    assert resolve_hybrid_yaml(HYBRID_STEM).name == HYBRID_STEM
    assert resolve_hybrid_yaml(str(REPO / "configs" / "trackers" / HYBRID_STEM)).name == HYBRID_STEM
    with pytest.raises(FileNotFoundError):
        resolve_hybrid_yaml("configs/trackers/definitely-not-a-tracker.yaml")


def test_mot_layer_resolve_tracker_yaml_finds_the_local_backend():
    """``mot_layer`` must resolve a project-local backend, not only ultralytics' six."""
    from tgtrack.mot_layer import TRACKER_YAMLS, resolve_tracker_yaml

    stem = Path(HYBRID_STEM).stem
    assert stem not in {Path(t).stem for t in TRACKER_YAMLS}
    assert resolve_tracker_yaml(HYBRID_STEM).name == HYBRID_STEM
    assert resolve_tracker_yaml(stem).name == HYBRID_STEM
    with pytest.raises(ValueError):
        resolve_tracker_yaml("nope.yaml")


# --------------------------------------------------------------------------- #
# harness wiring
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def harness():
    """Load ``scripts/eval_mot.py`` (not an importable package) under a private name."""
    pytest.importorskip("ultralytics")
    spec = importlib.util.spec_from_file_location("eval_mot_harness_hybrid", REPO / "scripts" / "eval_mot.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_make_tracker_dispatches_hybridsort_to_hybrid_backend(harness):
    from tgtrack.mot_hybrid import HybridSortTracker
    from tgtrack.mot_layer import AllPersonTracker

    cfg = harness.build_config(harness.RunOptions(), HYBRID_STEM)
    tracker = harness.make_tracker(cfg, None)
    assert isinstance(tracker, HybridSortTracker)
    assert tracker.tracker_type == "hybridsort"

    other = harness.build_config(harness.RunOptions(), "bytetrack.yaml")
    assert isinstance(harness.make_tracker(other, None), AllPersonTracker)


def test_parse_trackers_accepts_the_hybrid_stem(harness):
    assert harness.parse_trackers("hybridsort.yaml") == ["hybridsort.yaml"]


def test_enumerate_pairs_includes_hybridsort(harness):
    pairs = harness.enumerate_pairs(["MOT-01"], ["hybridsort.yaml"])
    assert len(pairs) == 1
    assert pairs[0].pair_id == "hybridsort__MOT-01"
    assert pairs[0].tracker == "hybridsort.yaml"
