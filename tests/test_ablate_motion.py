"""Tests for the motion-strategy ablation harness (:mod:`scripts.ablate_motion`).

Two levels:

* unit tests for the deterministic pieces -- scenario generators, the stub
  embedder, the config sweep, the JSON/CSV/Markdown writers;
* one end-to-end 2-config sweep that drives the real
  ``tgtrack.target_tracker.TargetTracker`` on a tiny scenario, so the wiring
  between the harness and the tracker/evaluator cannot silently rot.

The harness is imported as a plain module from ``scripts/`` (the repo has no
package there); ``src/`` comes in via the Lead's root ``conftest.py`` and is
also inserted defensively below.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "scripts", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ablate_motion as ab  # noqa: E402

# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #


def test_synthetic_frame_is_constant_and_wide() -> None:
    assert ab.FRAME.shape == (ab.FRAME_H, ab.FRAME_W, 3)
    assert ab.FRAME.dtype == np.uint8
    assert ab.FRAME.max() == 0, "the frame must never be rendered"
    assert ab.FRAME_W > 4000, "80 px/frame must not push the target out of frame"


def test_box_and_centre_helpers_agree() -> None:
    b = ab._box(100.0, 200.0)
    assert b.tolist() == pytest.approx([40.0, 80.0, 160.0, 320.0])
    assert ab.centre(b).tolist() == pytest.approx([100.0, 200.0])
    assert ab.centre(b.astype(np.float32)).tolist() == pytest.approx([100.0, 200.0])


# --------------------------------------------------------------------------- #
# Scenario generators
# --------------------------------------------------------------------------- #


def test_fast_linear_layout_and_detector_drop() -> None:
    drop = (4, 7)
    s = ab.fast_linear(step=10, n_frames=12, drop=drop, x0=260.0)

    assert s.name == "fast_linear_step10"
    assert s.family == "fast_linear"
    assert s.params == {"step": 10.0}
    assert s.n_frames == 12
    assert set(s.dets) == set(range(12))

    for f in range(12):
        # The target is always present in ground truth and to the embedder;
        # only the *detector* may miss it.
        assert s.gt_target[f] is not None
        assert [g.track_id for g in s.all_gt[f]] == [ab.TARGET_ID]
        assert len(s.person_boxes[f]) == 1
        if drop[0] <= f < drop[1]:
            assert s.dets[f] == [], f"frame {f} should be a detector drop"
        else:
            assert len(s.dets[f]) == 1


def test_fast_linear_is_exactly_constant_velocity() -> None:
    step, x0 = 40.0, 260.0
    s = ab.fast_linear(step, n_frames=20)
    centres = np.array([ab.centre(s.gt_target[f]) for f in range(20)])

    assert centres[:, 0].tolist() == pytest.approx([x0 + step * f for f in range(20)])
    assert centres[:, 1].tolist() == pytest.approx([ab.CY] * 20)

    deltas = np.diff(centres, axis=0)
    assert np.allclose(deltas, [step, 0.0], atol=1e-9)
    # A straight line has no curvature.
    assert float(np.abs(np.diff(centres, n=2, axis=0)).max()) < 1e-9


def test_circle_is_uniform_circular_motion() -> None:
    radius, speed, n = 60.0, 4.0, 40
    cx0, cy0 = 1200.0, 360.0
    s = ab.circle(radius, speed, n_frames=n, cx0=cx0, cy0=cy0)

    assert s.name == "circle_r60_v4"
    assert s.family == "nonlinear"
    assert s.params == {"radius": radius, "speed": speed}

    centres = np.array([ab.centre(s.gt_target[f]) for f in range(n)])
    radii = np.linalg.norm(centres - np.array([cx0, cy0]), axis=1)
    assert radii.tolist() == pytest.approx([radius] * n, abs=1e-6)

    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    assert steps.mean() == pytest.approx(speed, rel=0.05), "constant speed"
    assert steps.std() < 1e-6, "uniform in time as well as in arc length"

    # Curvature: the heading rotates, which no constant-velocity model can track.
    h0 = centres[5] - centres[4]
    h1 = centres[25] - centres[24]
    angle = math.acos(float(np.clip(np.dot(h0, h1) / (np.linalg.norm(h0) * np.linalg.norm(h1)), -1, 1)))
    assert angle > 0.5, f"heading must rotate (got {angle:.3f} rad)"
    curvature = float(np.abs(np.diff(centres, n=2, axis=0)).max())
    assert curvature > 0.01 * speed


def test_figure_eight_turns_both_ways() -> None:
    radius, speed, n = 80.0, 6.0, 60
    s = ab.figure_eight(radius, speed, n_frames=n)

    assert s.name == "figure8_r80_v6"
    assert s.family == "nonlinear"
    centres = np.array([ab.centre(s.gt_target[f]) for f in range(n)])
    assert np.abs(centres[:, 0] - 1200.0).max() == pytest.approx(radius, rel=0.05)
    assert np.abs(centres[:, 1] - 360.0).max() <= 0.5 * radius + 1e-6

    dy = np.diff(centres[:, 1])
    assert dy.min() < -1e-6 and dy.max() > 1e-6, "the path must turn both ways"


def test_occlusion_gap_semantics_inplace() -> None:
    gap, n_pre, n_post = 15, 6, 10
    s = ab.occlusion(gap, distant=False, n_pre=n_pre, n_post=n_post)

    assert s.name == "occlusion_inplace_gap15"
    assert s.family == "occlusion"
    assert s.n_frames == n_pre + gap + n_post
    assert s.params == {"gap": gap, "distant": False}

    for f in range(n_pre, n_pre + gap):
        assert s.gt_target[f] is None, f"frame {f} must be absent from GT"
        assert s.dets[f] == []
        assert s.all_gt[f] == []
        assert s.person_boxes[f] == []

    for f in list(range(n_pre)) + list(range(n_pre + gap, s.n_frames)):
        assert s.gt_target[f] is not None

    # In-place reappearance continues the original motion.
    resumed = ab.centre(s.gt_target[n_pre + gap])
    expected = ab.centre(s.gt_target[n_pre - 1]) + np.array([6.0 * (gap + 1), 0.0])
    assert resumed.tolist() == pytest.approx(expected.tolist())


def test_occlusion_distant_reappearance_jumps() -> None:
    gap, n_pre = 15, 6
    near = ab.occlusion(gap, distant=False, n_pre=n_pre, n_post=5)
    far = ab.occlusion(gap, distant=True, n_pre=n_pre, n_post=5)

    assert far.name == "occlusion_distant_gap15"
    assert far.params["distant"] is True

    f = n_pre + gap
    d_near = ab.centre(near.gt_target[f])
    d_far = ab.centre(far.gt_target[f])
    assert d_far[0] - d_near[0] == pytest.approx(600.0)
    assert d_far[1] == pytest.approx(150.0)
    assert d_near[1] == pytest.approx(ab.CY)


@pytest.mark.parametrize("gap", [5, 15, 40, 90])
def test_occlusion_gap_lengths(gap: int) -> None:
    s = ab.occlusion(gap)
    absent = [f for f in range(s.n_frames) if s.gt_target[f] is None]
    assert len(absent) == gap
    assert absent == list(range(absent[0], absent[0] + gap))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if area <= 0 else inter / area


def test_blackout_hides_detections_but_keeps_the_target_in_ground_truth() -> None:
    """The decisive property: the frames stay visible, so success needs a real box."""
    gap, drop_at, n = 15, 20, 70
    s = ab.circle_blackout(60, 4, gap, drop_at=drop_at, n_frames=n)

    assert s.name == "circle_blackout_r60_v4_gap15"
    assert s.family == "nonlinear_blackout"
    assert s.params == {"radius": 60.0, "speed": 4.0, "shape": "circle", "gap": gap, "drop_at": drop_at}
    assert s.n_frames == n

    for f in range(n):
        assert s.gt_target[f] is not None, "GT must stay visible or the metric would ignore these frames"
        assert len(s.person_boxes[f]) == 1
        if drop_at <= f < drop_at + gap:
            assert s.dets[f] == [], f"frame {f} is inside the blind window"
        else:
            assert len(s.dets[f]) == 1

    centres = np.array([ab.centre(s.gt_target[f]) for f in range(n)])
    radii = np.linalg.norm(centres - np.array([1200.0, 360.0]), axis=1)
    assert radii.tolist() == pytest.approx([60.0] * n, abs=1e-6)


def test_blackout_gap_is_a_real_prediction_challenge() -> None:
    """A tangent extrapolation must actually miss the box, or the family proves nothing."""
    gap, drop_at = 15, 20
    s = ab.circle_blackout(60, 4, gap, drop_at=drop_at, n_frames=drop_at + gap + 20)
    centres = np.array([ab.centre(s.gt_target[f]) for f in range(s.n_frames)])

    last = centres[drop_at - 1]
    tangent = centres[drop_at - 1] - centres[drop_at - 2]
    predicted = last + tangent * gap
    truth = centres[drop_at + gap - 1]  # the last frame the detector was blind
    naive_offset = float(np.linalg.norm(predicted - truth))
    assert naive_offset > 20.0, f"tangent extrapolation is only {naive_offset:.1f} px off"

    # Over a longer blind window the naive prediction is not even a valid report.
    long_gap = 30
    s2 = ab.circle_blackout(60, 4, long_gap, drop_at=drop_at, n_frames=drop_at + long_gap + 20)
    c2 = np.array([ab.centre(s2.gt_target[f]) for f in range(s2.n_frames)])
    pred_box = ab._box(*(c2[drop_at - 1] + (c2[drop_at - 1] - c2[drop_at - 2]) * long_gap))
    assert _iou(pred_box, s2.gt_target[drop_at + long_gap - 1]) < 0.5


def test_linear_blackout_control_is_straight() -> None:
    gap, drop_at = 15, 20
    s = ab.linear_blackout(20, gap, drop_at=drop_at, n_frames=drop_at + gap + 20)

    assert s.name == "linear_blackout_step20_gap15"
    assert s.family == "linear_blackout"
    centres = np.array([ab.centre(s.gt_target[f]) for f in range(s.n_frames)])
    assert float(np.abs(np.diff(centres, n=2, axis=0)).max()) < 1e-9

    for f in range(s.n_frames):
        assert s.gt_target[f] is not None
        if drop_at <= f < drop_at + gap:
            assert s.dets[f] == []

    # On a straight path the naive extrapolation is exact -- that is the point of
    # the control.
    last = centres[drop_at - 1]
    tangent = centres[drop_at - 1] - centres[drop_at - 2]
    truth = centres[drop_at + gap - 1]  # the last frame the detector was blind
    assert float(np.linalg.norm(last + tangent * gap - truth)) < 1e-6


def test_figure8_blackout_reverses_curvature_inside_the_window() -> None:
    gap, drop_at = 15, 20
    s = ab.figure8_blackout(80, 6, gap, drop_at=drop_at, n_frames=drop_at + gap + 25)
    assert s.family == "nonlinear_blackout"
    assert s.params["shape"] == "figure8"

    centres = np.array([ab.centre(s.gt_target[f]) for f in range(s.n_frames)])
    window = centres[drop_at : drop_at + gap]
    dy = np.diff(window[:, 1])
    assert dy.min() < 0 < dy.max(), "the unobserved window must contain a curvature reversal"
    assert all(s.gt_target[f] is not None for f in range(s.n_frames))
    assert all(s.dets[f] == [] for f in range(drop_at, drop_at + gap))


def test_distractor_lookalike_shares_one_embedding() -> None:
    s = ab.distractor(lookalike=True)
    assert s.name == "distractor_lookalike"
    assert s.params["lookalike"] is True
    assert s.vectors[ab.TARGET_ID].tolist() == pytest.approx(ab.IDENT_A.tolist())
    assert s.vectors[ab.DISTRACTOR_ID].tolist() == pytest.approx(ab.IDENT_A.tolist())
    assert s.vectors[ab.TARGET_ID].tolist() == pytest.approx(s.vectors[ab.DISTRACTOR_ID].tolist())


def test_distractor_different_appearance_is_orthogonal_control() -> None:
    s = ab.distractor(lookalike=False)
    assert s.name == "distractor_different"
    a = s.vectors[ab.TARGET_ID]
    b = s.vectors[ab.DISTRACTOR_ID]
    assert float(np.dot(a, b)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.linalg.norm(a)) == pytest.approx(1.0)
    assert float(np.linalg.norm(b)) == pytest.approx(1.0)


def test_distractor_crosses_and_almost_coincides() -> None:
    s = ab.distractor(lookalike=True, n_frames=100, d_start=45, d_end=95)
    assert s.family == "distractor"

    present = [f for f in range(s.n_frames) if len(s.dets[f]) == 2]
    assert present == list(range(45, 95))

    gaps = []
    for f in present:
        target = ab.centre(s.gt_target[f])
        other = ab.centre([pb for pid, pb in s.person_boxes[f] if pid == ab.DISTRACTOR_ID][0])
        gaps.append(float(np.linalg.norm(other - target)))
    assert min(gaps) < 60.0, "the two people must genuinely cross"

    # Equal size, so size cost cannot separate them either.
    for f in present:
        for _, b in s.person_boxes[f]:
            w, h = b[2] - b[0], b[3] - b[1]
            assert (w, h) == pytest.approx((ab.BOX_W, ab.BOX_H))


def test_scenarios_are_deterministic() -> None:
    for build in (
        lambda: ab.fast_linear(40, n_frames=20),
        lambda: ab.circle(60, 4, n_frames=25),
        lambda: ab.figure_eight(80, 6, n_frames=25),
        lambda: ab.occlusion(15),
        lambda: ab.distractor(lookalike=True),
    ):
        a, b = build(), build()
        assert a.name == b.name and a.n_frames == b.n_frames
        assert a.click.tolist() == b.click.tolist()
        for f in range(a.n_frames):
            ga, gb = a.gt_target[f], b.gt_target[f]
            assert (ga is None) == (gb is None)
            if ga is not None:
                assert ga.tolist() == gb.tolist()
            assert [d.xyxy.tolist() for d in a.dets[f]] == [d.xyxy.tolist() for d in b.dets[f]]


def test_every_scenario_clicks_on_frame_zero_target() -> None:
    for s in ab.default_scenarios():
        assert s.click.tolist() == pytest.approx(s.gt_target[0].tolist())
        assert len(s.dets[0]) >= 1, "the click frame must contain the target detection"


def test_default_scenarios_cover_all_families() -> None:
    scenarios = ab.default_scenarios()
    assert len({s.name for s in scenarios}) == len(scenarios), "scenario names must be unique"
    assert {s.family for s in scenarios} <= set(ab.FAMILY_NOTES)

    # Assert the per-family shape rather than a bare total: the total moved when the
    # hard-overlap distractor family landed, and a count alone does not say what broke.
    families = [s.family for s in scenarios]
    expected = {
        "fast_linear": 6,
        "nonlinear": 4,
        "occlusion": 8,
        "distractor": 2,
        "nonlinear_blackout": 4,
        "linear_blackout": 2,
        "distractor_overlap": 3,
        "overlap_sameid": 1,
        "overlap_solo": 1,
        "unidentifiable_target": 3,
        "unidentifiable_solo": 1,
    }
    assert {family: families.count(family) for family in expected} == expected
    assert len(scenarios) == sum(expected.values())

    steps = sorted(s.params["step"] for s in scenarios if s.family == "fast_linear")
    assert steps == [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    gaps = sorted({s.params["gap"] for s in scenarios if s.family == "occlusion"})
    assert gaps == [5, 15, 40, 90]
    # The blackout family must cover more than one blind-window length.
    assert {s.params["gap"] for s in scenarios if s.family == "nonlinear_blackout"} >= {15, 20, 30}
    # The hard-overlap family must sample the similarity ladder where the bystander
    # overtakes the target's own pose offset (0.97), not just easy rows.
    sims = sorted(s.params["similarity"] for s in scenarios if s.family == "distractor_overlap")
    assert sims == [0.85, 0.92, 0.99]
    # The unidentifiable family must carry both appearance-failure modes and a rung
    # below the refusal cliff, where the cost gate still refuses the stranger.
    unid = [s for s in scenarios if s.family == "unidentifiable_target"]
    assert sorted(s.params["similarity"] for s in unid) == [0.0, 0.6, 0.6]
    assert {s.params["mode"] for s in unid} == {"missing", "orthogonal"}


def test_quick_scenarios_are_short() -> None:
    scenarios = ab.quick_scenarios()
    assert scenarios, "the smoke set must not be empty"
    assert max(s.n_frames for s in scenarios) <= 60
    assert {s.family for s in scenarios} <= set(ab.FAMILY_NOTES)
    for s in scenarios:
        assert s.gt_target[0] is not None


# --------------------------------------------------------------------------- #
# Stub embedder
# --------------------------------------------------------------------------- #


def _two_person_embedder(lookalike: bool = False) -> ab.ScenarioEmbedder:
    a = ab._box(400.0, 300.0)
    b = ab._box(1600.0, 300.0)
    vectors = {
        ab.TARGET_ID: ab.IDENT_A,
        ab.DISTRACTOR_ID: ab.IDENT_A if lookalike else ab.IDENT_B,
    }
    return ab.ScenarioEmbedder(vectors, {0: [(ab.TARGET_ID, a), (ab.DISTRACTOR_ID, b)]})


def test_stub_embedder_maps_query_box_to_nearest_person() -> None:
    e = _two_person_embedder()
    e.frame = 0
    out = e.encode(None, [ab._box(400.0, 300.0), ab._box(1600.0, 300.0)], parts=("full", "upper"))

    assert len(out) == 2
    assert set(out[0]) == {"full", "upper"}
    assert out[0]["full"].tolist() == pytest.approx(ab.IDENT_A.tolist())
    assert out[1]["full"].tolist() == pytest.approx(ab.IDENT_B.tolist())
    assert float(np.linalg.norm(out[0]["full"])) == pytest.approx(1.0)
    assert e.calls == 1


def test_stub_embedder_returns_identical_vectors_for_lookalikes() -> None:
    e = _two_person_embedder(lookalike=True)
    e.frame = 0
    out = e.encode(None, [ab._box(400.0, 300.0), ab._box(1600.0, 300.0)])
    assert out[0]["full"].tolist() == pytest.approx(out[1]["full"].tolist())


def test_stub_embedder_is_deterministic_and_nan_free() -> None:
    e = _two_person_embedder()
    e.frame = 0
    first = e.encode(None, [ab._box(420.0, 300.0)])
    second = e.encode(None, [ab._box(420.0, 300.0)])
    assert first[0]["full"].tolist() == second[0]["full"].tolist()
    assert bool(np.isfinite(first[0]["full"]).all())


def test_stub_embedder_tracks_the_frame_attribute() -> None:
    s = ab.fast_linear(10, n_frames=3)
    e = ab.ScenarioEmbedder(s.vectors, s.person_boxes)
    e.frame = 0
    at0 = e.encode(None, [s.gt_target[0]])[0]["full"]
    e.frame = 2
    at2 = e.encode(None, [s.gt_target[2]])[0]["full"]
    assert at0.tolist() == at2.tolist(), "one identity: the vector must not depend on position"


def test_stub_embedder_falls_back_when_nobody_is_visible() -> None:
    e = ab.ScenarioEmbedder({ab.TARGET_ID: ab.IDENT_A}, {0: []})
    e.frame = 0
    out = e.encode(None, [ab._box(100.0, 100.0)])
    assert out[0]["full"].tolist() == pytest.approx(ab.IDENT_A.tolist())
    assert e.unknown == 1


# --------------------------------------------------------------------------- #
# Config sweep
# --------------------------------------------------------------------------- #


def test_motion_spec_applies_every_swept_knob() -> None:
    from tgtrack.config import Config

    spec = ab.MotionSpec("custom", model="imm", nsa_adaptive=False, rollback_depth=0, imm_models=("cv", "ca"))
    cfg = spec.apply(Config())

    assert cfg.motion.model == "imm"
    assert cfg.motion.nsa_adaptive is False
    assert cfg.motion.rollback_depth == 0
    assert cfg.motion.imm_models == ("cv", "ca")

    d = spec.as_dict()
    assert d["config"] == "custom" and d["model"] == "imm" and d["imm_models"] == ["cv", "ca"]


def test_default_specs_cover_the_required_sweep() -> None:
    labels = [s.label for s in ab.DEFAULT_SPECS]
    assert len(labels) == len(set(labels)), "labels key the leaderboard"

    assert {"kf", "nsa_kf", "imm"} <= {s.model for s in ab.DEFAULT_SPECS}
    assert {s.rollback_depth for s in ab.DEFAULT_SPECS} >= {0, 3, 5}
    assert {s.nsa_adaptive for s in ab.DEFAULT_SPECS} == {True, False}
    imm_sets = {s.imm_models for s in ab.DEFAULT_SPECS if s.model == "imm"}
    assert ("cv",) in imm_sets and ("cv", "ca") in imm_sets and ("cv", "ca", "ct") in imm_sets


def test_motion_spec_does_not_mutate_the_dataclass_defaults() -> None:
    from tgtrack.config import Config, MotionCfg

    before = (MotionCfg().rollback_depth, MotionCfg().model, MotionCfg().imm_models)

    cfg = Config()
    ab.MotionSpec("probe", model="nsa_kf", rollback_depth=0, imm_models=("cv",)).apply(cfg)

    assert cfg.motion.model == "nsa_kf", "apply() edits the instance it is given"
    assert (MotionCfg().rollback_depth, MotionCfg().model, MotionCfg().imm_models) == before
    assert Config().motion.model == "imm", "each run must start from a fresh default Config"


# --------------------------------------------------------------------------- #
# End-to-end: the real TargetTracker, two motion configs, one tiny scenario
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_bundle() -> dict:
    scenarios = [ab.fast_linear(20, n_frames=16, drop=(8, 10))]
    specs = [ab.DEFAULT_SPECS[0], ab.DEFAULT_SPECS[5]]  # kf vs imm
    assert [s.label for s in specs] == ["kf", "imm"]
    return ab.run_sweep(scenarios, specs, progress=False)


def test_two_config_sweep_runs_both_configs(tiny_bundle: dict) -> None:
    meta = tiny_bundle["meta"]
    assert meta["n_scenarios"] == 1
    assert meta["n_configs"] == 2
    assert meta["n_runs"] == 2
    assert tiny_bundle["errors"] == []

    assert {r["config"] for r in tiny_bundle["results"]} == {"kf", "imm"}
    assert set(tiny_bundle["leaderboard"]) == {"kf", "imm"}
    assert set(tiny_bundle["by_family"]) == {"fast_linear"}


def test_run_scenario_drives_one_tracker_step_per_frame() -> None:
    spec = ab.DEFAULT_SPECS[0]
    scenario = ab.fast_linear(20, n_frames=16, drop=(8, 10))
    result = ab.run_scenario(scenario, spec)

    assert result["n_frames"] == scenario.n_frames
    assert sum(result["states"].values()) == scenario.n_frames
    assert result["embedder_calls"] >= 1
    assert result["unmapped_crops"] == 0, "every crop must map to a known person"


def test_metrics_come_from_evaluate_target_not_hand_rolled(tiny_bundle: dict) -> None:
    from tgtrack.evaluate import combine_target

    for row in tiny_bundle["results"]:
        pooled = combine_target([row["metrics"]])
        for key, value in tiny_bundle["leaderboard"][row["config"]].items():
            if isinstance(value, (int, float)) and key in pooled:
                if math.isnan(float(value)):
                    assert math.isnan(float(pooled[key]))
                else:
                    assert float(value) == pytest.approx(float(pooled[key]), rel=1e-9, abs=1e-12)


def test_sweep_is_deterministic(tiny_bundle: dict) -> None:
    scenarios = [ab.fast_linear(20, n_frames=16, drop=(8, 10))]
    specs = [ab.DEFAULT_SPECS[0], ab.DEFAULT_SPECS[5]]
    again = ab.run_sweep(scenarios, specs, progress=False)
    assert ab._jsonable(again["results"]) == ab._jsonable(tiny_bundle["results"])
    assert ab._jsonable(again["leaderboard"]) == ab._jsonable(tiny_bundle["leaderboard"])


def test_findings_are_number_bearing(tiny_bundle: dict) -> None:
    lines = tiny_bundle["findings"]
    assert lines, "a sweep must produce findings"
    assert any(line.startswith("OVERALL") for line in lines)
    assert all(any(ch.isdigit() for ch in line) for line in lines), lines
    assert not any("nan" in line or "n/a" == line for line in lines)


def test_run_sweep_records_errors_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(scenario: ab.Scenario, spec: ab.MotionSpec) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(ab, "run_scenario", boom)
    bundle = ab.run_sweep([ab.fast_linear(10, n_frames=6)], [ab.DEFAULT_SPECS[0]], progress=False)

    assert bundle["results"] == []
    assert len(bundle["errors"]) == 1
    assert bundle["errors"][0]["config"] == "kf"
    assert "boom" in bundle["errors"][0]["error"]
    assert bundle["leaderboard"] == {}
    assert bundle["findings"] == ["no results"]


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def test_jsonable_sanitises_non_finite_values() -> None:
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "array": np.array([1.0, float("nan")]),
        "int": np.int64(7),
        "bool": np.bool_(True),
        "nested": [{"x": np.float32(1.5)}],
    }
    out = ab._jsonable(payload)
    assert out["nan"] is None and out["inf"] is None
    assert out["array"] == [1.0, None]
    assert out["int"] == 7 and out["bool"] is True
    assert out["nested"][0]["x"] == pytest.approx(1.5)
    json.dumps(out, allow_nan=False)  # must not raise


def test_write_outputs_creates_clean_files(tmp_path: Path, tiny_bundle: dict) -> None:
    paths = ab.write_outputs(tiny_bundle, tmp_path)

    assert set(paths) == {"raw", "md", "csv", "readme"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    raw_text = paths["raw"].read_text(encoding="utf-8")
    assert "NaN" not in raw_text and "Infinity" not in raw_text
    raw = json.loads(raw_text)
    assert raw["meta"]["n_runs"] == 2
    assert len(raw["results"]) == 2
    assert set(raw["leaderboard"]) == {"kf", "imm"}
    assert isinstance(raw["findings"], list) and raw["findings"]

    rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))
    assert len(rows) == 2
    assert set(rows[0]) >= {"scenario", "family", "config", "model", "target_success_rate", "id_consistency"}

    md = paths["md"].read_text(encoding="utf-8")
    assert "# Motion-strategy ablation" in md
    assert "## Leaderboard" in md
    assert "## Findings" in md
    assert "| config |" in md
    assert "fast_linear" in md

    readme = paths["readme"].read_text(encoding="utf-8")
    assert "# Motion-strategy ablation" in readme
    assert "scripts/ablate_motion.py" in readme
    assert "evaluate_target" in readme


def test_nan_metrics_serialise_to_null(tmp_path: Path) -> None:
    # A clean run never loses the target, so evaluate_target returns NaN for the
    # re-acquisition latency -- it must land in JSON as null, not as a bare NaN.
    bundle = ab.run_sweep([ab.fast_linear(5, n_frames=12, drop=(99, 100))], [ab.DEFAULT_SPECS[0]], progress=False)
    metrics = bundle["results"][0]["metrics"]
    assert math.isnan(float(metrics["reacquire_latency"]))

    paths = ab.write_outputs(bundle, tmp_path)
    text = paths["raw"].read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["results"][0]["metrics"]["reacquire_latency"] is None


def test_csv_rows_mirror_results(tiny_bundle: dict) -> None:
    rows = ab.csv_rows(tiny_bundle)
    assert len(rows) == len(tiny_bundle["results"])
    for row in rows:
        assert row["imm_models"]
        assert row["model"] in {"kf", "nsa_kf", "imm"}
        assert row["n_frames"] == 16


def test_render_markdown_includes_every_family_and_scenario(tiny_bundle: dict) -> None:
    md = ab.render_markdown(tiny_bundle)
    for scenario in tiny_bundle["scenarios"]:
        assert scenario["name"] in md
    for finding in tiny_bundle["findings"]:
        assert finding[:40] in md


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_quick_no_write_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out and "FINDINGS" in out
    assert "tracker runs" in out


def test_main_writes_into_the_requested_directory(tmp_path: Path) -> None:
    assert ab.main(["--quick", "--outdir", str(tmp_path / "ablation")]) == 0
    for name in ("raw.json", "table.md", "table.csv"):
        assert (tmp_path / "ablation" / name).exists()


# --------------------------------------------------------------------------- #
# Per-frame diagnostics (--trace): the gate/state-machine interaction
# --------------------------------------------------------------------------- #

_KF_TRACE = ab.MotionSpec("kf", model="kf", rollback_depth=0)


def test_trace_has_one_row_per_frame_with_the_requested_columns() -> None:
    scenario = ab.occlusion(9, distant=True, n_pre=6, n_post=6)
    result = ab.run_scenario(scenario, _KF_TRACE, trace=True)
    rows = result["trace"]

    assert len(rows) == scenario.n_frames
    assert [r["frame"] for r in rows] == list(range(scenario.n_frames))
    for row in rows:
        assert set(ab.TRACE_COLUMNS) <= set(row)


def test_trace_observes_the_run_without_changing_it() -> None:
    scenario = ab.occlusion(9, distant=True, n_pre=6, n_post=6)
    plain = ab.run_scenario(scenario, _KF_TRACE)
    traced = ab.run_scenario(scenario, _KF_TRACE, trace=True)

    assert "trace" not in plain
    # NaN != NaN, so compare the serialised form (undefined metrics -> null).
    assert ab._jsonable(traced["metrics"]) == ab._jsonable(plain["metrics"])
    assert traced["states"] == plain["states"]


def test_short_gap_reappearance_is_reachable_and_still_runs_the_cost_gate() -> None:
    """A short-gap distant reappearance is gated in, not gated out.

    Before the canonical frame-scaled radius the reappearance sat ~650 px away
    against a ~175-210 px grown radius and was refused for the whole window.
    With search armed at ``lost_short`` (``fsu`` 15 > ``lost_short_after`` 12)
    the target is rebound on the very frame it reappears.

    Passing the gate is not the same as being accepted: acceptance still runs
    through appearance evidence and the distractor margin -- the refusal side of
    that decision is covered by the hard-overlap distractor tests.
    """
    scenario = ab.occlusion(15, distant=True, n_pre=8, n_post=20)
    result = ab.run_scenario(scenario, _KF_TRACE, trace=True)
    rows = result["trace"]
    reappear = 8 + 15
    blind = [r for r in rows if r["frame"] >= reappear]

    assert blind
    assert all(r["n_dets"] == 1 for r in blind), "the reappeared target is detected"
    assert any(int(r["gated"]) == 1 for r in blind), "the reappearance must pass the gate"

    accepted = [r for r in blind if r["accepted"]]
    assert accepted, "and it must be accepted on the appearance evidence"
    assert int(accepted[0]["frame"]) == reappear, "rebound on the first visible frame"
    assert str(accepted[0]["reason"]) == "ok"
    # The decision really ran: a numeric cost was scored against a threshold.
    assert accepted[0]["best_cost"] is not None
    assert float(accepted[0]["best_cost"]) < float(accepted[0]["thr"])
    assert result["metrics"]["reacquire_latency"] == pytest.approx(0.0, abs=1e-9)


def test_gate_radius_is_monotone_through_the_blind_window() -> None:
    """The grown radius must never peak and then decay while blind.

    It used to: the speed term came from the coasting velocity, which
    ``coast_velocity_damp`` shrinks every frame, so the radius rose and then
    fell until ``lost_long`` armed search -- exactly the frames where the target
    needed the widest gate.
    """
    scenario = ab.occlusion(15, distant=True, n_pre=8, n_post=20)
    rows = ab.run_scenario(scenario, _KF_TRACE, trace=True)["trace"]
    radii = [
        float(r["gate_r"])
        for r in rows
        if r["frame"] < 8 + 15 and r["gate_r"] is not None
    ]

    assert len(radii) >= 10
    assert all(later >= earlier - 1e-9 for earlier, later in zip(radii, radii[1:])), radii


def test_search_arms_at_lost_short_and_the_control_restores_the_old_rule() -> None:
    """Global search is armed at ``lost_short``, not only at ``lost_long``.

    The harness' own distant reappearance (600 px) lands just inside the
    canonical frame-scaled radius (``0.10 * hypot(6000, 720)`` ~= 604 px), so it
    cannot separate the wider radius from the earlier search.  A 900 px jump
    can, and it is the ablation control ``state.global_states=()`` -- the old
    "arm only at lost_long/gave_up/out_of_frame" rule -- that shows what the
    fix is worth: with kf, shipped -> success 1.0000 and re-acquire latency 0.0,
    control -> success 0.3443 and no re-acquisition inside the window.
    """
    scenario = ab.occlusion(15, distant=True, jump=900.0)
    shipped = ab.run_scenario(scenario, _KF_TRACE)
    control = ab.run_scenario(
        scenario, _KF_TRACE, cfg_overrides={"state.global_states": ()}
    )

    assert shipped["metrics"]["target_success_rate"] == pytest.approx(1.0, abs=1e-9)
    assert shipped["metrics"]["reacquire_latency"] == pytest.approx(0.0, abs=1e-9)
    assert control["metrics"]["target_success_rate"] == pytest.approx(0.3443, abs=5e-4)
    # The event is still raised (the target is seen far away) but it is never
    # completed inside the window, so the latency stays undefined.
    assert control["metrics"]["reacquire_rate"] == 0.0
    assert not math.isfinite(control["metrics"]["reacquire_latency"])


def test_cfg_overrides_reach_the_tracker_without_leaking_into_the_next_run() -> None:
    """``cfg_overrides`` patches a clone; the next un-overridden run is untouched."""
    scenario = ab.occlusion(9, distant=False, n_pre=4, n_post=4)
    plain = ab.run_scenario(scenario, _KF_TRACE)
    patched = ab.run_scenario(
        scenario, _KF_TRACE, cfg_overrides={"state.lost_long_after": 7}
    )

    assert patched["scenario"] == plain["scenario"]  # same run, different config
    again = ab.run_scenario(scenario, _KF_TRACE)
    assert ab._jsonable(again["metrics"]) == ab._jsonable(plain["metrics"])
    assert ab.Config().state.lost_long_after == 60


def test_render_trace_is_one_line_per_frame() -> None:
    scenario = ab.occlusion(5, distant=False, n_pre=4, n_post=4)
    rows = ab.run_scenario(scenario, _KF_TRACE, trace=True)["trace"]
    text = ab.render_trace(rows, scenario=scenario.name, spec="kf")
    lines = text.splitlines()

    assert lines[0] == f"trace: {scenario.name} | spec kf | {len(rows)} frames"
    assert len(lines) == 3 + len(rows)  # title + header + separator + rows


def test_trace_csv_round_trips(tmp_path: Path) -> None:
    scenario = ab.occlusion(5, distant=False, n_pre=4, n_post=4)
    rows = ab.run_scenario(scenario, _KF_TRACE, trace=True)["trace"]
    path = ab.write_trace_csv(tmp_path / "trace.csv", rows)

    with path.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == len(rows)
    assert list(got[0]) == list(ab.TRACE_COLUMNS)
    assert int(got[0]["frame"]) == 0


def test_main_trace_prints_the_table_and_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--trace", "occlusion_distant_gap15", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "trace: occlusion_distant_gap15" in out
    assert "final metrics:" in out


def test_main_trace_rejects_an_unknown_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--trace", "no_such_scenario", "--no-write"]) == 2
    assert "unknown scenario" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Hard-overlap distractor family: the margin gate's own experiment
# --------------------------------------------------------------------------- #


def test_correlated_pair_hits_the_requested_cosine() -> None:
    for cos in (0.0, 0.6, 0.85, 0.99, 1.0):
        first, second = ab.correlated_pair(cos)
        assert first.shape == (4,) and second.shape == (4,)
        assert float(np.linalg.norm(first)) == pytest.approx(1.0, abs=1e-6)
        assert float(np.linalg.norm(second)) == pytest.approx(1.0, abs=1e-6)
        assert float(first @ second) == pytest.approx(cos, abs=1e-6)

    assert np.array_equal(ab.correlated_pair(0.85)[0], ab.correlated_pair(0.85)[0])
    assert np.array_equal(ab.correlated_pair(0.85)[1], ab.correlated_pair(0.85)[1])


def test_overlap_distractor_boxes_overlap_hard_but_are_not_identical() -> None:
    """The geometry must not resolve the ambiguity -- that is the whole point."""
    scenario = ab.overlap_distractor(0.85)
    lo = int(scenario.params["n_pre"])
    hi = lo + int(scenario.params["n_overlap"])

    assert all(len(scenario.dets[f]) == 2 for f in range(lo, hi))
    assert all(scenario.gt_target[f] is not None for f in range(lo, hi))

    ious = []
    for f in range(lo, hi):
        boxes = dict(scenario.person_boxes[f])
        assert set(boxes) == {ab.TARGET_ID, ab.DISTRACTOR_ID}
        ious.append(ab.box_iou(boxes[ab.TARGET_ID], boxes[ab.DISTRACTOR_ID]))
    assert all(iou > 0.85 for iou in ious), ious
    assert all(iou < 1.0 for iou in ious), "the two candidates must not be the same box"


def test_overlap_similarity_ladder_is_the_crop_cosine() -> None:
    for cos in ab.DISTRACTOR_COSINES:
        scenario = ab.overlap_distractor(cos)
        assert scenario.params["similarity"] == pytest.approx(cos)
        template = scenario.vectors[ab.TARGET_ID][0]
        assert float(template @ scenario.vectors[ab.TARGET_ID][5]) == pytest.approx(0.97, abs=1e-5)
        assert float(template @ scenario.vectors[ab.DISTRACTOR_ID][5]) == pytest.approx(cos, abs=1e-5)


def test_overlap_controls_are_a_same_identity_and_a_solo_run() -> None:
    sameid = ab.overlap_sameid()
    assert sameid.params["same_identity"] is True
    assert sameid.family == "overlap_sameid"
    lo = int(sameid.params["n_pre"])
    hi = lo + int(sameid.params["n_overlap"])
    assert all(len(sameid.dets[f]) == 2 for f in range(lo, hi))
    # One identity only: a false acquisition is impossible by construction.
    assert all(len(boxes) == 1 for boxes in sameid.all_gt.values())

    solo = ab.overlap_solo()
    assert solo.family == "overlap_solo"
    assert all(len(dets) <= 1 for dets in solo.dets.values())
    assert all(len(boxes) == 1 for boxes in solo.all_gt.values())


def test_overlap_peel_direction_separates_the_bystander_and_marks_the_control() -> None:
    backwards = ab.overlap_distractor(0.99)
    forward = ab.overlap_distractor(0.99, peel=(14.0, 9.0))

    assert not backwards.name.endswith("_fwd")
    assert forward.name.endswith("_fwd")
    assert backwards.params["peel"] == [-14.0, -9.0]
    assert forward.params["peel"] == [14.0, 9.0]

    last = int(backwards.n_frames) - 1
    for scenario, expect_left in ((backwards, True), (forward, False)):
        boxes = dict(scenario.person_boxes[last])
        left = ab.centre(boxes[ab.DISTRACTOR_ID])[0] < ab.centre(boxes[ab.TARGET_ID])[0]
        assert bool(left) == expect_left


def test_overlap_window_is_not_a_motion_challenge() -> None:
    """The family isolates appearance: the target itself walks in a straight line."""
    scenario = ab.overlap_distractor(0.85)
    centres = np.asarray(
        [ab.centre(np.asarray(scenario.gt_target[f], dtype=float)) for f in range(scenario.n_frames)]
    )
    deltas = np.diff(centres, axis=0)
    assert np.allclose(deltas[:, 0], 6.0, atol=1e-9)
    assert np.allclose(deltas[:, 1], 0.0, atol=1e-9)
    assert np.allclose(np.diff(deltas, axis=0), 0.0, atol=1e-9)


def test_overlap_scenarios_are_deterministic_and_click_on_the_target() -> None:
    for first, second in (
        (ab.overlap_distractor(0.92), ab.overlap_distractor(0.92)),
        (ab.overlap_solo(), ab.overlap_solo()),
    ):
        assert first.name == second.name
        assert first.n_frames == second.n_frames
        for f in range(first.n_frames):
            assert np.allclose(
                np.asarray(first.gt_target[f], dtype=float),
                np.asarray(second.gt_target[f], dtype=float),
            )
            assert [d.xyxy.tolist() for d in first.dets.get(f, [])] == [
                d.xyxy.tolist() for d in second.dets.get(f, [])
            ]
        assert np.allclose(
            np.asarray(first.click, dtype=float), np.asarray(first.gt_target[0], dtype=float)
        )


# --------------------------------------------------------------------------- #
# The distractor sweep itself
# --------------------------------------------------------------------------- #


def test_distractor_specs_cover_the_model_by_margin_grid() -> None:
    specs = ab.distractor_specs(models=("kf", "imm"), margins=(0.0, 0.03, 0.06, 0.12))

    assert [s.label for s in specs] == [
        "kf-m00",
        "kf-m03",
        "kf-m06",
        "kf-m12",
        "imm-m00",
        "imm-m03",
        "imm-m06",
        "imm-m12",
    ]
    assert {s.rollback_depth for s in specs} == {0}
    assert {s.distractor_margin for s in specs} == {0.0, 0.03, 0.06, 0.12}
    assert {s.model for s in specs} == {"kf", "imm"}


def test_refusal_counts_separate_ambiguous_from_all_gated() -> None:
    rows = [
        {"frame": 0, "reason": "ok", "accepted": True, "best_cost": 0.1, "runner_up": 0.4},
        {
            "frame": 1,
            "reason": "ambiguous: margin 0.010 < 0.060",
            "accepted": False,
            "best_cost": 0.2,
            "runner_up": 0.21,
        },
        {
            "frame": 2,
            "reason": "ambiguous: margin 0.020 < 0.060",
            "accepted": False,
            "best_cost": 0.2,
            "runner_up": 0.22,
        },
        {
            "frame": 3,
            "reason": "all_gated:outside_radius",
            "accepted": False,
            "best_cost": None,
            "runner_up": None,
        },
    ]
    counts = ab._refusal_counts(rows)

    assert counts["frames"] == 4
    assert counts["ambiguous"] == 2
    assert counts["all_gated"] == 1
    assert counts["accepted"] == 1
    assert counts["margin_median"] == pytest.approx(0.02, abs=1e-9)
    assert counts["margin_min"] == pytest.approx(0.01, abs=1e-9)
    assert counts["reasons"]["ambiguous"] == 2

    windowed = ab._refusal_counts(rows, window=(1, 3))
    assert windowed["frames"] == 2
    assert windowed["ambiguous"] == 2
    assert windowed["accepted"] == 0


def _tiny_distractor_bundle() -> dict:
    return ab.run_distractor_sweep(
        [ab.overlap_distractor(1.0, n_pre=8, n_overlap=8, n_post=6)],
        cosines=(1.0,),
        margins=(0.0, 0.12),
        models=("kf",),
        progress=False,
    )


def test_distractor_sweep_aggregates_traces_into_refusal_summaries() -> None:
    bundle = _tiny_distractor_bundle()

    assert bundle["meta"]["n_runs"] == 2
    assert bundle["meta"]["n_errors"] == 0
    assert "distractor_margin" in bundle["meta"]["sweep"]
    for result in bundle["results"]:
        assert "trace" not in result, "traces are aggregated, never stored"
        assert "refusals" in result and "refusals_overlap" in result
        assert result["refusals_overlap"]["frames"] == 8
    assert bundle["findings"]


def test_distractor_grid_is_similarity_by_margin() -> None:
    bundle = _tiny_distractor_bundle()

    grid = ab.distractor_grid(bundle, column="false_acquisition_rate", model="kf")
    assert grid["source"] == "metrics"
    assert grid["model"] == "kf"
    assert grid["cosines"] == [1.0]
    assert grid["margins"] == [0.0, 0.12]
    assert len(grid["grid"]) == 1 and len(grid["grid"][0]) == 2
    assert all(math.isfinite(v) for row in grid["grid"] for v in row)

    refused = ab.distractor_grid(bundle, column="ambiguous", model="kf", source="refusals_overlap")
    assert refused["grid"][0][1] > refused["grid"][0][0], "a wider margin must refuse more"
    assert refused["grid"][0][1] > 0


def test_margin_gate_trades_success_for_false_acquisition() -> None:
    """The primary endpoint: a wider margin refuses more AND can cause a swap."""
    scenario = ab.overlap_distractor(1.00)
    specs = {s.label: s for s in ab.distractor_specs(models=("kf",), margins=(0.0, 0.06))}

    loose = ab.run_scenario(scenario, specs["kf-m00"], trace=True)
    tight = ab.run_scenario(scenario, specs["kf-m06"], trace=True)

    assert loose["metrics"]["target_success_rate"] == pytest.approx(1.0, abs=1e-9)
    assert loose["metrics"]["false_acquisition_rate"] == pytest.approx(0.0, abs=1e-9)
    assert ab._refusal_counts(loose["trace"])["ambiguous"] == 0

    assert tight["metrics"]["target_success_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert ab._refusal_counts(tight["trace"])["ambiguous"] == 50
    assert tight["metrics"]["false_acquisition_rate"] > 0.0
    assert tight["metrics"]["target_switch_count"] >= 1


def test_forward_peel_refuses_too_but_never_false_acquires() -> None:
    """2x2 control: refusal costs success in both peel directions, but the false
    acquisition needs the bystander to peel the way a damped coast drifts."""
    spec = ab.MotionSpec("kf-m06", model="kf", rollback_depth=0, distractor_margin=0.06)
    backwards = ab.run_scenario(ab.overlap_distractor(1.00), spec)
    forward = ab.run_scenario(ab.overlap_distractor(1.00, peel=(14.0, 9.0)), spec)

    assert backwards["metrics"]["target_success_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert forward["metrics"]["target_success_rate"] == pytest.approx(0.6667, abs=1e-3)
    assert backwards["metrics"]["false_acquisition_rate"] > 0.0
    assert forward["metrics"]["false_acquisition_rate"] == pytest.approx(0.0, abs=1e-9)


def test_render_distractor_markdown_covers_every_view() -> None:
    text = ab.render_distractor_markdown(_tiny_distractor_bundle())

    for _, _, _, title in ab.DISTRACTOR_VIEWS:
        assert title in text
    assert "kf" in text
    assert "1.00" in text


def test_distractor_findings_are_number_bearing() -> None:
    lines = ab.distractor_findings(_tiny_distractor_bundle())

    assert lines
    assert all(any(ch.isdigit() for ch in line) for line in lines), lines
    assert any("false" in line.lower() for line in lines)
    assert any("margin" in line.lower() for line in lines)


def test_distractor_outputs_and_readme_are_written(tmp_path: Path) -> None:
    bundle = _tiny_distractor_bundle()
    paths = ab.write_distractor_outputs(bundle, tmp_path)

    assert set(paths) == {"distractor_md", "distractor_csv"}
    assert paths["distractor_md"].name == "distractor_sweep.md"
    assert paths["distractor_md"].stat().st_size > 0
    assert paths["distractor_csv"].stat().st_size > 0

    everything = ab.write_outputs(bundle, tmp_path / "both", distractor=bundle)
    assert {"raw", "md", "csv", "readme", "distractor_md", "distractor_csv"} <= set(everything)
    readme = everything["readme"].read_text(encoding="utf-8")
    assert "distractor" in readme.lower()
    assert "scripts/ablate_motion.py" in readme


def test_csv_rows_include_the_margin_and_overlap_columns() -> None:
    rows = ab.csv_rows(_tiny_distractor_bundle())

    assert rows
    for key in (
        "distractor_margin",
        "overlap_frames",
        "refused_frames",
        "accepted_in_overlap",
        "margin_median",
    ):
        assert key in rows[0]


def test_main_distractor_flag_runs_only_that_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--distractor", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "HARD-OVERLAP DISTRACTOR SWEEP" in out
    assert "LEADERBOARD" not in out
    assert "distractor_margin" in out


def test_main_no_distractor_flag_skips_that_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--no-distractor", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "HARD-OVERLAP DISTRACTOR SWEEP" not in out


# --------------------------------------------------------------------------- #
# The unidentifiable-target family: requirement 4's failure mode
# --------------------------------------------------------------------------- #


def _tiny_unid(cos: float = 0.6, *, mode: str = "missing") -> ab.Scenario:
    return ab.unidentifiable_target(cos, mode=mode, n_pre=8, n_gap=8, n_blind=8, n_post=6)


def _unid_shipped() -> ab.MotionSpec:
    specs = {s.label: s for s in ab.unid_specs(models=("kf",), sweeps=(("shipped", None),))}
    return specs["kf-shipped"]


def _unid_bound(bound: float | None) -> ab.MotionSpec:
    """The shipped spec with ``lone_candidate_accept_cost`` overridden.

    ``None`` sets an EXPLICIT override to ``None`` (bound off) rather than leaving
    the shipped default alone: the shipped default is now a live bound, so "no
    override" would make the control a second copy of the treatment.
    """
    import dataclasses

    spec = _unid_shipped()
    overrides = tuple(dict(spec.cfg_overrides).items())
    overrides += (("cost.lone_candidate_accept_cost", None if bound is None else float(bound)),)
    return dataclasses.replace(spec, label=f"kf-l{bound}", cfg_overrides=overrides)


def _tiny_unid_bundle() -> dict:
    return ab.run_unid_sweep(
        [
            _tiny_unid(0.0),
            _tiny_unid(0.6),
            ab.unidentifiable_solo(n_pre=8, n_gap=8, n_blind=8, n_post=6),
        ],
        modes=("missing",),
        models=("kf",),
        sweeps=(("shipped", None),),
        progress=False,
    )


def test_unidentifiable_target_keeps_gt_alive_over_a_stale_prediction() -> None:
    """The decisive setup: the target never leaves the GT (so those frames are
    'visible'), it just stops being identifiable -- while an unoccluded
    bystander rides where a constant-velocity coast predicts the target to be."""
    scenario = _tiny_unid(0.6)
    lo, hi = ab._unid_window(scenario)
    n_pre = int(scenario.params["n_pre"])
    speed = float(scenario.params["speed"])
    offset = np.asarray(scenario.params["offset"], dtype=float)

    assert scenario.family == "unidentifiable_target"
    assert all(scenario.gt_target[f] is not None for f in range(scenario.n_frames))
    assert all(len(scenario.dets.get(f, [])) == 1 for f in range(n_pre, lo))
    assert all(len(scenario.dets.get(f, [])) == 2 for f in range(lo, hi))

    centres = [ab.centre(np.asarray(scenario.gt_target[f], dtype=float)) for f in range(scenario.n_frames)]
    halted = centres[n_pre - 1]
    assert np.allclose(centres[n_pre:], halted, atol=1e-9), "the target halts and never moves again"

    first = ab.centre(dict(scenario.person_boxes[n_pre])[ab.DISTRACTOR_ID])
    assert np.allclose(first, halted + np.array([speed, 0.0]) + offset, atol=1e-6)

    rides = np.asarray(
        [
            ab.centre(dict(scenario.person_boxes[f])[ab.DISTRACTOR_ID])
            for f in range(n_pre, lo)
        ]
    )
    assert np.allclose(np.diff(rides, axis=0), [speed, 0.0], atol=1e-6)

    twin = _tiny_unid(0.6)
    assert twin.name == scenario.name
    assert all(
        np.allclose(
            np.asarray(scenario.gt_target[f], dtype=float), np.asarray(twin.gt_target[f], dtype=float)
        )
        for f in range(scenario.n_frames)
    )


def test_unidentifiable_modes_make_the_target_crop_unusable() -> None:
    missing = _tiny_unid(0.6)
    lo, hi = ab._unid_window(missing)
    vectors = np.asarray(missing.vectors[ab.TARGET_ID], dtype=float)

    assert np.allclose(np.linalg.norm(vectors[lo:hi], axis=1), 0.0)
    assert np.linalg.norm(vectors[lo - 1]) == pytest.approx(1.0)

    embedder = ab.ScenarioEmbedder(missing.vectors, missing.person_boxes)
    embedder.frame = lo
    crop = embedder.encode(None, [np.asarray(missing.gt_target[lo], dtype=float)], parts=("full",))
    assert crop == [{}], "an all-zero descriptor must reach the tracker as 'no appearance'"

    orthogonal = _tiny_unid(0.6, mode="orthogonal")
    ovecs = np.asarray(orthogonal.vectors[ab.TARGET_ID], dtype=float)
    assert "orthogonal" in orthogonal.name
    assert np.allclose(np.linalg.norm(ovecs[lo:hi], axis=1), 1.0)
    assert np.allclose(ovecs[lo:hi] @ ovecs[0], 0.0, atol=1e-6)


def test_unidentifiable_similarity_ladder_is_the_bystander_cosine() -> None:
    for cos in (0.0, 0.4, 0.6, 0.85):
        scenario = _tiny_unid(cos)
        assert float(scenario.params["similarity"]) == pytest.approx(cos)
        template = np.asarray(scenario.vectors[ab.TARGET_ID][0], dtype=float)
        bystander = np.asarray(scenario.vectors[ab.DISTRACTOR_ID][0], dtype=float)
        assert float(template @ bystander) == pytest.approx(cos, abs=1e-5)

    solo = ab.unidentifiable_solo(n_pre=8, n_gap=8, n_blind=8, n_post=6)
    assert solo.family == "unidentifiable_solo"
    assert ab.DISTRACTOR_ID not in solo.vectors
    assert all(len(boxes) == 1 for boxes in solo.all_gt.values())


def test_unid_specs_are_the_model_by_arming_rule_grid() -> None:
    specs = ab.unid_specs()

    assert [s.label for s in specs] == [
        "kf-shipped",
        "kf-no_early_search",
        "imm-shipped",
        "imm-no_early_search",
    ]
    assert {s.rollback_depth for s in specs} == {0}
    assert dict(specs[0].cfg_overrides) == {}
    assert dict(specs[1].cfg_overrides) == {"state.global_states": ()}
    assert ab._unid_window(_tiny_unid(0.6)) == (16, 24)


def test_lone_bound_refuses_the_bystander_on_the_first_unobserved_frame() -> None:
    """The mechanism, before and after ``lone_candidate_accept_cost``.

    At f=``n_pre`` the target stops being detected and the bystander is the ONLY
    finite candidate, so ``select_candidate`` leaves ``margin = inf`` and the
    ambiguity test is vacuous: with the bound off the bystander is ACCEPTED while
    the tracker is still ``tracking`` (fsu=0), and neither the state machine, the
    arming rule nor the search guard ever runs. The lone bound is the only rule
    that can see this, and with it on the bystander is refused.
    """
    n_pre = int(_tiny_unid(0.6).params["n_pre"])

    def frame_row(bound: float | None) -> dict:
        trace = ab.run_scenario(_tiny_unid(0.6), _unid_bound(bound), trace=True)["trace"]
        at = [r for r in trace if r["frame"] == n_pre]
        assert len(at) == 1
        return at[0]

    # Shipped: the lone bystander is refused, and the refusal says why.
    refused = frame_row(0.5)
    assert refused["n_dets"] == 1, "the target is undetected; only the bystander remains"
    assert bool(refused["accepted"]) is False, refused
    assert refused["state"] == "tracking" and refused["fsu"] == 1, refused
    assert str(refused["reason"]).startswith("lone_candidate"), refused

    # Bound off: the documented handover is still exactly there.
    off = frame_row(None)
    assert bool(off["accepted"]) is True, off
    assert off["fsu"] == 0 and off["state"] == "tracking", off

    # ...and on this world the bound is a strict improvement, not a wash: the
    # refused frame-8 hijack becomes a PRICED comparison at frame 16, which the
    # tracker still loses but more slowly, so all three aggregate metrics improve.
    on_m = ab.run_scenario(_tiny_unid(0.6), _unid_bound(0.5))["metrics"]
    off_m = ab.run_scenario(_tiny_unid(0.6), _unid_bound(None))["metrics"]
    assert on_m["target_success_rate"] > off_m["target_success_rate"]
    assert on_m["false_acquisition_rate"] < off_m["false_acquisition_rate"]
    assert on_m["id_consistency"] > off_m["id_consistency"]


def test_arming_the_global_search_earlier_does_not_change_the_unid_outcome() -> None:
    """The handover happens while TRACKING, so the arming rule is inert here."""
    bundle = ab.run_unid_sweep(
        [_tiny_unid(0.6)],
        modes=("missing",),
        models=("kf",),
        sweeps=(("shipped", None), ("no_early_search", {"state.global_states": ()})),
        progress=False,
    )
    assert bundle["meta"]["n_runs"] == 2 and bundle["meta"]["n_errors"] == 0

    by_config = {r["config"]: r["metrics"] for r in bundle["results"]}
    shipped = by_config["kf-shipped"]
    control = by_config["kf-no_early_search"]

    assert shipped["false_acquisition_rate"] > 0.0
    for key in ("target_success_rate", "false_acquisition_rate", "id_consistency"):
        assert ab._jsonable(shipped[key]) == ab._jsonable(control[key]), key


def test_unid_sweep_aggregates_traces_and_counts_the_blind_window() -> None:
    bundle = _tiny_unid_bundle()

    assert bundle["meta"]["n_runs"] == 3
    assert bundle["meta"]["n_errors"] == 0
    assert "unidentifiable" in bundle["meta"]["sweep"]
    for result in bundle["results"]:
        assert "trace" not in result
        assert result["refusals_blind"]["frames"] == 8
        assert 0.0 <= result["blind_ratio"] <= 1.0
    assert bundle["findings"]
    assert all(any(ch.isdigit() for ch in line) for line in bundle["findings"]), bundle["findings"]


def test_unid_grid_is_similarity_by_arming_rule() -> None:
    bundle = ab.run_unid_sweep(
        [_tiny_unid(0.0), _tiny_unid(0.6)],
        modes=("missing",),
        models=("kf",),
        sweeps=(("shipped", None), ("no_early_search", {"state.global_states": ()})),
        progress=False,
    )

    grid = ab.unid_grid(bundle, column="false_acquisition_rate", mode="missing", model="kf")

    assert grid["cosines"] == [0.0, 0.6]
    assert grid["cols"] == ["no_early_search", "shipped"]
    assert len(grid["grid"]) == 2 and all(len(row) == 2 for row in grid["grid"])
    assert grid["grid"][1][1] > 0.0, "the 0.60 rung must hand the track over"
    assert grid["grid"][1][0] == pytest.approx(grid["grid"][1][1])


def test_unid_solo_control_never_false_acquires() -> None:
    solo = ab.run_scenario(
        ab.unidentifiable_solo(n_pre=8, n_gap=8, n_blind=8, n_post=6), _unid_shipped()
    )["metrics"]

    assert solo["false_acquisition_rate"] == pytest.approx(0.0, abs=1e-9)
    assert solo["target_switch_count"] == 0

    rows = ab.unid_solo_rows(_tiny_unid_bundle())
    assert len(rows) == 1
    assert rows[0]["false_acquisition_rate"] == pytest.approx(0.0, abs=1e-9)


def test_render_unid_markdown_covers_every_view() -> None:
    text = ab.render_unid_markdown(_tiny_unid_bundle())

    for _, _, _, title in ab.UNID_VIEWS:
        assert title in text
    assert "stale prediction" in text
    assert "shipped" in text and "no_early_search" in text


def test_unid_outputs_and_readme_are_written(tmp_path: Path) -> None:
    bundle = _tiny_unid_bundle()
    paths = ab.write_unid_outputs(bundle, tmp_path)

    assert set(paths) == {"unid_md", "unid_csv"}
    assert paths["unid_md"].name == "unid_sweep.md"
    assert paths["unid_md"].stat().st_size > 0
    assert paths["unid_csv"].stat().st_size > 0

    everything = ab.write_outputs(bundle, tmp_path / "both", unid=bundle)
    assert {"raw", "md", "csv", "readme", "unid_md", "unid_csv"} <= set(everything)
    readme = everything["readme"].read_text(encoding="utf-8")
    assert "unidentifiable" in readme.lower()


def test_csv_rows_include_the_blind_window_columns() -> None:
    rows = ab.csv_rows(_tiny_unid_bundle())

    assert rows
    for key in ("blind_frames", "accepted_in_blind", "refused_in_blind", "blind_accept_ratio"):
        assert key in rows[0]


def test_main_unid_flag_runs_only_that_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--unid", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "UNIDENTIFIABLE-TARGET SWEEP" in out
    assert "LEADERBOARD" not in out


def test_main_no_unid_flag_skips_that_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--no-unid", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "LEADERBOARD" in out
    assert "UNIDENTIFIABLE-TARGET SWEEP" not in out


# --------------------------------------------------------------------------- #
# The price of unreadable appearance: missing_appearance_cost x similarity
# --------------------------------------------------------------------------- #


def _tiny_cost_bundle(costs: Sequence[float] = (0.5, 1.0, 1.2)) -> dict:
    return ab.run_unid_cost_sweep(
        [
            _tiny_unid(0.0),
            _tiny_unid(0.6),
            ab.unidentifiable_solo(n_pre=8, n_gap=8, n_blind=8, n_post=6),
        ],
        modes=("missing",),
        models=("kf",),
        costs=costs,
        progress=False,
    )


def test_missing_costs_axis_covers_the_low_side() -> None:
    """The axis must include the LOW side, which is the only side that can work.

    Raising ``missing_appearance_cost`` makes the unreadable TRUE target more
    expensive than a readable stranger, so a sweep over {0.5, 1.0, 1.2} could only
    ever confirm "pricing does not help" -- which is what it did. The direction
    that can help is pricing a missing appearance near zero, so the unreadable
    true target is not outbid by whoever happens to be readable.
    """
    assert ab.MISSING_COSTS[0] == 0.0, "the low side must be on the axis"
    assert 0.5 in ab.MISSING_COSTS, "the shipped value stays as a reference point"
    assert ab.Config().cost.missing_appearance_cost == pytest.approx(0.5)


def test_unid_cost_specs_are_the_price_axis() -> None:
    specs = ab.unid_cost_specs(models=("kf", "imm"), costs=(0.5, 1.2))

    assert [s.label for s in specs] == ["kf-c050", "kf-c120", "imm-c050", "imm-c120"]
    for spec in specs:
        assert spec.rollback_depth == 0, "only the price may move"
        assert dict(spec.cfg_overrides)["cost.missing_appearance_cost"] in (0.5, 1.2)


def test_unid_cost_sweep_measures_the_price_axis() -> None:
    bundle = _tiny_cost_bundle()

    assert bundle["meta"]["n_runs"] == 9
    assert bundle["meta"]["n_errors"] == 0
    assert "missing_appearance_cost" in bundle["meta"]["sweep"]
    for result in bundle["results"]:
        assert "trace" not in result
        assert "handover" in result
        assert result["refusals_blind"]["frames"] == 8
    assert bundle["findings"]
    assert all(any(ch.isdigit() for ch in line) for line in bundle["findings"])


def test_handover_frame_is_the_first_accepted_report_on_the_bystander() -> None:
    """With the SHIPPED default (bound off) the handover is the first lone accept.

    `CostCfg.lone_candidate_accept_cost` is None by default because on real MOT17 it
    costs 3.6 pp of target success on the upper-body-only mask with no measurable
    false-acquisition benefit (`results/target_obs19_shipped` vs
    `results/target_obs19`). The rule itself is exercised through explicit overrides
    (`_unid_bound`, the `lone_specs` grid), so with the default the handover is the
    frame-`n_pre` lone accept that the bound would have refused.
    """
    bundle = _tiny_cost_bundle(costs=(0.5,))
    by_name = {r["scenario"]: r for r in bundle["results"]}
    n_pre = int(_tiny_unid(0.6).params["n_pre"])

    hand = by_name["unidentifiable_missing_s060"]["handover"]
    assert hand is not None
    assert hand["frame"] == n_pre, "the unverifiable accept lands on the first blind frame"
    assert hand["n_dets"] == 1, "a lone hijack: the true target is not even a candidate"
    assert hand["phase"] == "gap"
    assert hand["iou_bystander"] > hand["iou_target"]

    assert by_name["unidentifiable_missing_s000"]["handover"] is None
    assert by_name["unidentifiable_solo_missing"]["handover"] is None


def test_handover_split_counts_the_lone_hijack() -> None:
    """With the shipped default every handover is a LONE candidate.

    ``handover_split`` keys the split by how many finite candidates existed at the
    handover. This test pins the default behaviour so the difference stays
    measurable: with `lone_candidate_accept_cost=None` the handover key is
    ``n_dets=1`` and the priced key is absent. Arming the rule is what removes that
    key -- see ``_unid_bound`` / the ``lone_specs`` grid for the other side.
    """
    rows = ab.handover_split(_tiny_cost_bundle(costs=(0.5,)))
    row = next(r for r in rows if r["config"] == "kf-c050" and r["mode"] == "missing")

    assert row.get("n_dets=1", 0) == 1, "the handover is a lone candidate"
    assert row.get("n_dets=2", 0) == 0, "no priced comparison takes place"
    assert row["never"] == 1, "one scenario never hands over at all"


def test_pricing_the_absent_evidence_moves_recovery_not_the_handover() -> None:
    """Price alone cannot move the handover; it only moves the RECOVERY.

    At the handover the true target is not a candidate at all (``n_dets=1``), so
    ``missing_appearance_cost`` is never applied and every price hands over on the
    same frame. Price does govern whether the tracker later takes the track back,
    which is why the low rungs still measure better. Closing the handover itself
    needs the lone-candidate rule armed first (``_unid_bound(0.4)`` closes it at
    <= 0.25 -- see ``scripts/ablate_motion.py --unid-cost`` / ``--lone``).
    """
    bundle = _tiny_cost_bundle(costs=ab.MISSING_COSTS)
    by_name = {(r["scenario"], r["config"]): r for r in bundle["results"]}

    def cfg(cost: float) -> str:
        return f"kf-c{int(round(cost * 100)):03d}"

    tsr = {
        cost: by_name[("unidentifiable_missing_s060", cfg(cost))]["metrics"]["target_success_rate"]
        for cost in ab.MISSING_COSTS
    }
    far = {
        cost: by_name[("unidentifiable_missing_s060", cfg(cost))]["metrics"]["false_acquisition_rate"]
        for cost in ab.MISSING_COSTS
    }

    low = [c for c in ab.MISSING_COSTS if c <= 0.25]
    high = [c for c in ab.MISSING_COSTS if c > 0.25]
    assert low and high, "both sides of the price must be sampled"

    n_pre = int(_tiny_unid(0.6).params["n_pre"])

    # The handover itself is NOT moved by price, in either direction.
    for cost in ab.MISSING_COSTS:
        hand = by_name[("unidentifiable_missing_s060", cfg(cost))]["handover"]
        assert hand is not None, cost
        assert hand["frame"] == n_pre and hand["n_dets"] == 1, (cost, hand)

    # Price does move the recovery: cheaper missing evidence keeps the true target
    # competitive once it is detected again.
    cheap, dear = min(low), max(high)
    assert tsr[cheap] > tsr[dear], (cheap, dear, tsr)
    assert far[cheap] < far[dear], (cheap, dear, far)
    assert far[dear] > 0.0, "the dear rung hands the track over and does not take it back"

    solo = [
        r for r in bundle["results"] if r["family"] == "unidentifiable_solo"
    ]
    assert solo and all(r["metrics"]["false_acquisition_rate"] == pytest.approx(0.0, abs=1e-9) for r in solo)


def test_unid_cost_grid_is_similarity_by_price() -> None:
    bundle = _tiny_cost_bundle()
    grid = ab.unid_cost_grid(bundle, column="false_acquisition_rate", mode="missing", model="kf")

    assert grid["cosines"] == [0.0, 0.6]
    assert grid["cols"] == ["c050", "c100", "c120"]
    assert len(grid["grid"]) == 2 and all(len(row) == 3 for row in grid["grid"])
    assert grid["grid"][1][0] > 0.0, "the 0.60 rung hands over at the shipped price"
    assert np.allclose(grid["grid"][1], grid["grid"][1][0]), "and at every other price"


def test_render_unid_cost_markdown_covers_every_view() -> None:
    text = ab.render_unid_cost_markdown(_tiny_cost_bundle())

    for _, _, _, title in ab.UNID_COST_VIEWS:
        assert title in text
    assert "n_dets=1" in text and "margin = inf" in text
    assert "c050" in text and "c120" in text


def test_unid_cost_outputs_and_readme_are_written(tmp_path: Path) -> None:
    bundle = _tiny_cost_bundle()
    paths = ab.write_unid_cost_outputs(bundle, tmp_path)

    assert set(paths) == {"unid_cost_md", "unid_cost_csv"}
    assert paths["unid_cost_md"].name == "unid_cost.md"
    assert paths["unid_cost_md"].stat().st_size > 0
    assert paths["unid_cost_csv"].stat().st_size > 0

    everything = ab.write_outputs(bundle, tmp_path / "both", unid_cost=bundle)
    assert {"raw", "md", "csv", "readme", "unid_cost_md", "unid_cost_csv"} <= set(everything)


def test_csv_rows_include_the_price_and_handover_columns() -> None:
    rows = ab.csv_rows(_tiny_cost_bundle())
    keys: set[str] = set()
    for row in rows:
        keys |= set(row)

    assert rows
    for key in ("missing_appearance_cost", "handover_frame", "handover_n_dets", "handover_phase"):
        assert key in keys, f"{key} missing from the CSV columns"
    assert {r["missing_appearance_cost"] for r in rows} == {0.5, 1.0, 1.2}
    assert ab._csv_fields(rows)[:3] == ["scenario", "family", "params"]


def test_main_unid_cost_flag_runs_only_the_pricing_sweep(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--unid-cost", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "UNIDENTIFIABLE-TARGET PRICING SWEEP" in out
    assert "UNIDENTIFIABLE-TARGET SWEEP" not in out
    assert "LEADERBOARD" not in out


def test_main_unid_flag_runs_both_unid_sweeps(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--unid", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "UNIDENTIFIABLE-TARGET SWEEP" in out
    assert "UNIDENTIFIABLE-TARGET PRICING SWEEP" in out
    assert "LEADERBOARD" not in out
    assert "HARD-OVERLAP DISTRACTOR SWEEP" not in out


# --------------------------------------------------------------------------- #
# Lone-candidate guard (CostCfg.lone_candidate_accept_cost)
# --------------------------------------------------------------------------- #
def _lone_unid() -> ab.Scenario:
    return ab.unidentifiable_target(0.60, n_pre=8, n_gap=8, n_blind=8, n_post=6)


def _lone_solo() -> ab.Scenario:
    return ab.unidentifiable_solo(n_pre=8, n_gap=8, n_blind=8, n_post=6)


def _tiny_lone_bundle() -> dict:
    return ab.run_lone_sweep(
        [_lone_unid(), _lone_solo()],
        modes=("missing",),
        models=("kf",),
        bounds=(0.4, None),
        progress=False,
    )


def test_lone_specs_are_the_model_by_bound_grid() -> None:
    specs = ab.lone_specs(models=("kf", "imm"), bounds=(0.2, 0.5, None))
    assert [s.label for s in specs] == [
        "kf-l020",
        "kf-l050",
        "kf-lnone",
        "imm-l020",
        "imm-l050",
        "imm-lnone",
    ]
    by = {s.label: s for s in specs}
    # None is an EXPLICIT override to None (bound off), never "leave the shipped
    # default alone" -- the shipped default is a live bound, so a no-override
    # control would silently be a second copy of the treatment.
    assert by["kf-lnone"].cfg_overrides == (("cost.lone_candidate_accept_cost", None),)
    assert dict(by["kf-l020"].cfg_overrides) == {"cost.lone_candidate_accept_cost": 0.2}
    assert dict(by["imm-l050"].cfg_overrides) == {"cost.lone_candidate_accept_cost": 0.5}
    assert {s.rollback_depth for s in specs} == {0}


def test_lone_bound_turns_a_lone_hijack_into_a_priced_comparison() -> None:
    """Before/after on the tiny unid world, with the off control genuinely off.

    Off: the bystander is the ONLY finite candidate on the first unobserved frame,
    so ``margin`` is infinite, the ambiguity test is vacuous, and the report is
    accepted at frame 8 with ``n_dets=1``. On: that lone accept is refused and the
    surviving handover is a PRICED comparison at frame 16 with ``n_dets=2`` -- so
    the world is measurably better, not merely different.
    """
    bundle = _tiny_lone_bundle()
    rows = {r["config"]: r for r in bundle["results"] if r["family"] == "unidentifiable_target"}
    off, on = rows["kf-lnone"], rows["kf-l040"]

    assert "trace" not in off
    assert off["handover"] is not None and off["handover"]["n_dets"] == 1, "lone hijack"
    assert on["handover"] is not None and on["handover"]["n_dets"] == 2, "priced comparison"
    assert on["handover"]["frame"] > off["handover"]["frame"], "the hijack moved later"

    # The frame-level difference, checked directly on the first unobserved frame.
    n_pre = int(_tiny_unid(0.6).params["n_pre"])
    specs = {s.label: s for s in ab.lone_specs(("kf",), (0.4, None))}

    def frame_row(spec: ab.MotionSpec) -> dict:
        trace = ab.run_scenario(_lone_unid(), spec, trace=True)["trace"]
        at = [r for r in trace if r["frame"] == n_pre]
        assert len(at) == 1
        return at[0]

    accepted = frame_row(specs["kf-lnone"])
    assert bool(accepted["accepted"]) is True, accepted
    refused = frame_row(specs["kf-l040"])
    assert bool(refused["accepted"]) is False, refused
    assert str(refused["reason"]).startswith("lone_candidate"), refused

    # Three aggregate metrics, all better, and the solo control untouched.
    for key in ("target_success_rate", "id_consistency"):
        assert on["metrics"][key] > off["metrics"][key], key
    assert on["metrics"]["false_acquisition_rate"] < off["metrics"]["false_acquisition_rate"]
    controls = {
        r["config"]: r["metrics"]["false_acquisition_rate"]
        for r in bundle["results"]
        if r["family"] == "unidentifiable_solo"
    }
    assert set(controls.values()) == {0.0}, controls


def test_lone_guard_leaves_the_legitimate_lone_target_alone() -> None:
    """The no-bystander control: with nothing to refuse, every bound must keep
    accepting the tracker's own lone target."""
    bundle = _tiny_lone_bundle()
    solo = ab.unid_solo_rows(bundle)
    assert len(solo) == 2
    assert {r["false_acquisition_rate"] for r in solo} == {0.0}
    assert all(r["target_success_rate"] == pytest.approx(0.9667, abs=1e-4) for r in solo)


def test_lone_findings_name_the_handover_split_and_carry_numbers() -> None:
    bundle = _tiny_lone_bundle()
    findings = bundle["findings"]
    assert any("n_dets=1" in line for line in findings)
    assert any("control (no bystander)" in line for line in findings)
    assert all(any(ch.isdigit() for ch in line) for line in findings)


def test_lone_clean_sweep_prices_the_bounds_on_the_clean_families() -> None:
    bundle = ab.run_lone_clean_sweep(
        [ab.fast_linear(4.0, n_frames=24), ab.circle(60.0, 4.0, n_frames=24)],
        bounds=(0.4, None),
        model="kf",
        progress=False,
    )
    assert bundle["meta"]["n_errors"] == 0
    assert set(bundle["leaderboard"]) == {"kf-l040", "kf-lnone"}
    base = bundle["leaderboard"]["kf-lnone"]
    guarded = bundle["leaderboard"]["kf-l040"]
    assert base["target_success_rate"] >= guarded["target_success_rate"]
    assert all(any(ch.isdigit() for ch in line) for line in bundle["findings"])
    assert any("clean families" in line for line in bundle["findings"])


def test_clean_scenarios_exclude_the_unid_families() -> None:
    families = {s.family for s in ab.clean_scenarios()}
    assert not any(f.startswith("unidentifiable") for f in families)
    assert "unidentifiable_target" in {s.family for s in ab.default_scenarios()}


def test_lone_outputs_and_csv_columns(tmp_path: Path) -> None:
    bundle = _tiny_lone_bundle()
    paths = ab.write_lone_outputs(bundle, tmp_path)
    assert set(paths) == {"lone_md", "lone_csv"}
    assert paths["lone_md"].exists() and paths["lone_csv"].exists()
    text = paths["lone_md"].read_text(encoding="utf-8")
    assert "lone_candidate_accept_cost" in text
    assert "handover frame" in text
    fieldnames = paths["lone_csv"].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "lone_candidate_accept_cost" in fieldnames
    rows = [r for r in ab.csv_rows(bundle) if "lone_candidate_accept_cost" in r]
    # The off cell carries an EXPLICIT None override, so it shows up in the CSV as
    # None rather than being indistinguishable from "no row for that knob".
    assert {r["lone_candidate_accept_cost"] for r in rows} == {0.4, None}


def test_main_lone_flag_runs_only_the_lone_sweeps(capsys: pytest.CaptureFixture[str]) -> None:
    assert ab.main(["--lone", "--quick", "--no-write"]) == 0
    out = capsys.readouterr().out
    assert "LONE-CANDIDATE BOUND SWEEP" in out
    assert "on the clean families" in out
    assert "UNIDENTIFIABLE-TARGET SWEEP" not in out
    assert "LEADERBOARD" not in out
