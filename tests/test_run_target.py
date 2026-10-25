"""Tests for the clicked-target harness + strategy ablation grid (task-7).

Hermetic by construction: no dataset, no GPU, no detector weights and no ReID
checkpoint.  Two appearance sources are used:

* the **GT oracle detector** (``GtDetector``), which turns the masked-world
  ground truth into person detections, so ``--detector gt`` runs are fully
  deterministic, and
* ``Embedder(..., force_fallback=True)`` -- the dependency-free descriptor the
  appearance layer ships for exactly this situation.

Every sequence is a few frames long and lives in ``tmp_path`` (or in memory), so
the whole file runs in seconds on a CPU-only box.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT / "src"), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_target as rt  # noqa: E402
from tgtrack import Detection, GtBox, Sequence, load_config, load_mot_sequence  # noqa: E402
from tgtrack.appearance import Embedder  # noqa: E402

W, H, N = 320, 120, 8

# ---------------------------------------------------------------- fixtures --


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def embedder(cfg):
    """The dependency-free descriptor: real association, no weights, no GPU."""
    try:
        emb = Embedder(cfg.embedder, device="cpu", force_fallback=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fallback embedder unavailable: {type(exc).__name__}: {exc}")
    if getattr(emb, "info", {}).get("is_real"):
        pytest.skip("expected the fallback descriptor, got a real model")
    return emb


def _boxes(frame: int, *, two: bool = True, gap: tuple[int, int] = (0, 0)) -> list[GtBox]:
    """Two people: id 1 walks right, id 2 walks left and crosses it."""
    out: list[GtBox] = []
    if not (gap[0] <= frame < gap[1]):
        out.append(GtBox(frame=frame, track_id=1, xyxy=np.array([10.0 + 6 * frame, 20.0, 40.0 + 6 * frame, 90.0])))
    if two:
        out.append(GtBox(frame=frame, track_id=2, xyxy=np.array([210.0 - 5 * frame, 26.0, 240.0 - 5 * frame, 96.0])))
    return out


def _paint(boxes: list[GtBox]) -> np.ndarray:
    img = np.full((H, W, 3), 40, np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(float(v))) for v in b.xyxy)
        colour = (205, 200, 195) if int(b.track_id) == 1 else (120, 150, 185)
        img[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = colour
    return img


def _seq(n: int = N, *, two: bool = True, gap: tuple[int, int] = (0, 0), name: str = "clean") -> Sequence:
    gt = {i: _boxes(i, two=two, gap=gap) for i in range(n)}
    frames = [_paint(gt[i]) for i in range(n)]
    return Sequence(name=name, frames=frames, gt=gt, fps=25.0, width=W, height=H)


def _write_mot_dir(root: Path, *, n: int = N, name: str = "SEQ-01", two: bool = True) -> Path:
    """A minimal but real MOTChallenge sequence on disk (images + gt + seqinfo)."""
    import cv2

    seq_dir = Path(root) / name
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    rows = ["# minimal synthetic MOTChallenge ground truth"]
    for i in range(n):
        img = _paint(_boxes(i, two=two))
        assert cv2.imwrite(str(seq_dir / "img1" / f"{i + 1:06d}.jpg"), img)
        for b in _boxes(i, two=two):
            x0, y0, x1, y1 = (float(v) for v in b.xyxy)
            rows.append(
                "%d,%d,%.1f,%.1f,%.1f,%.1f,1,1,1" % (i + 1, int(b.track_id), x0, y0, x1 - x0, y1 - y0)
            )
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=%s\nimDir=img1\nframeRate=25\nseqLength=%d\nimWidth=%d\nimHeight=%d\nimExt=.jpg\n"
        % (name, n, W, H),
        encoding="utf-8",
    )
    return seq_dir


class ShiftedDetector:
    """GT-oracle detector whose boxes are deliberately in the wrong place."""

    def __init__(self, seq: Sequence, dx: float) -> None:
        self.seq = seq
        self.dx = float(dx)

    def detect(self, frame, frame_idx: int | None = None) -> list[Detection]:
        idx = 0 if frame_idx is None else int(frame_idx)
        out = []
        for b in self.seq.gt_at(idx):
            xyxy = np.asarray(b.xyxy, np.float64) + np.array([self.dx, 0.0, self.dx, 0.0])
            out.append(Detection(xyxy=xyxy, score=1.0))
        return out


def _spec(
    seq_dir: Path | str,
    *,
    target_id: int = 1,
    variant: str = "baseline",
    group: str = "baseline",
    mask: str = "none",
    overrides: dict | None = None,
    opts: dict | None = None,
    seq_name: str | None = None,
) -> dict:
    o = {
        "dry_run": False,
        "detector": "gt",
        "device": "cpu",
        "use_embedder": True,
        "embed_fallback": True,
        "quiet": True,
        "max_frames": None,
        "subsample": 1,
        "min_visible": 0.0,
        "min_conf": 1.0,
        "synth_frames": N,
        "margin": 0.02,
        "edge_frac": 0.5,
        "drop_below": 0.2,
        "fill": 0,
    }
    o.update(opts or {})
    return {
        "seq_dir": str(seq_dir),
        "seq_name": seq_name or Path(seq_dir).name,
        "target_id": int(target_id),
        "variant": variant,
        "group": group,
        "overrides": dict(overrides or {}),
        "mask": mask,
        "config_path": None,
        "opts": o,
    }


# ------------------------------------------------------------- the grid -----


def test_variant_grid_covers_every_required_switch():
    variants = rt.build_variants("all")
    names = [v.name for v in variants]
    assert names[0] == "baseline" and variants[0].is_baseline
    assert "baseline" in names
    by_name = {v.name: v for v in variants}

    # every ablation group required by the task description is present
    assert {v.group for v in variants} == {"baseline", *rt.GROUPS}

    # 1. partial-view part mechanism
    assert by_name["part_matching_off"].overrides == {"cost.part_matching": False}
    assert by_name["part_aware_off"].overrides == {"memory.part_aware": False}
    # 2. the anti-latch margin (the headline anti-false-acquisition claim)
    for m in rt.DEFAULT_MARGINS:
        assert by_name[f"margin_{m:.2f}"].overrides == {"cost.distractor_margin": float(m)}
    assert rt.DEFAULT_MARGINS == (0.0, 0.06, 0.12)
    # 3. distractor bank
    assert by_name["hard_negative_off"].overrides == {"memory.hard_negative_bank": False}
    # 4. short-term vs long-term ReID
    assert by_name["stm_only"].overrides == {"cost.stm_weight": 1.0, "cost.ltm_weight": 0.0}
    assert by_name["balanced"].overrides == {"cost.stm_weight": 0.6, "cost.ltm_weight": 0.4}
    assert by_name["ltm_only"].overrides == {"cost.stm_weight": 0.0, "cost.ltm_weight": 1.0}
    # 5. long-term memory capacity
    for cap in (1, 96, 384):
        assert by_name[f"ltm_cap_{cap}"].overrides == {"memory.ltm_capacity": cap}
    # 6. freezing the gallery while blind
    assert by_name["freeze_on"].overrides == {}
    assert by_name["freeze_off"].overrides == {"state.frozen_states": ()}
    # 7. cheap vs full part set
    assert by_name["parts_full_only"].overrides == {"embedder.target_parts": ("full",)}
    assert by_name["parts_all"].overrides == {"embedder.target_parts": ("full", "upper", "lower", "head")}

    # the overrides must all be real, settable configuration keys
    cfg = load_config()
    for v in variants:
        if v.overrides:
            cfg.clone(**v.overrides)  # raises on a typo


def test_variant_overrides_actually_change_the_tracker(cfg, embedder):
    """A flip must reach the object the tracker reads, not just look like one."""
    off = cfg.clone(**{"cost.part_matching": False})
    assert off.cost.part_matching is False and cfg.cost.part_matching is True
    cap = cfg.clone(**{"memory.ltm_capacity": 1})
    assert cap.memory.ltm_capacity == 1 and cfg.memory.ltm_capacity == 96
    frozen = cfg.clone(**{"state.frozen_states": ()})
    assert tuple(frozen.state.frozen_states) == () and cfg.state.frozen_states


def test_variant_selection_and_errors():
    assert [v.name for v in rt.build_variants(None)] == [v.name for v in rt.build_variants("all")]
    only = rt.build_variants("part_matching")
    assert [v.name for v in only] == ["baseline", "part_matching_off"]
    with pytest.raises(ValueError):
        rt.build_variants("part_matchng")  # typo must not silently drop a claim
    text = rt.variant_table(rt.build_variants("distractor_margin"))
    for name in ("baseline", "margin_0.00", "margin_0.12"):
        assert name in text
    assert "cost.distractor_margin=0.12" in text


def test_target_selection_prefers_the_stressed_identity():
    seq = _seq(14, gap=(5, 9))  # id 1 disappears for 4 frames; id 2 never does
    assert rt.choose_target_id(seq, "stressed") == 1
    assert rt.choose_target_id(seq, "longest") == 2
    assert rt.choose_target_id(seq, "first") == 1
    assert rt.choose_target_id(_seq(4, two=False), "most-occluded") == 1
    assert rt.choose_target_id(Sequence(name="empty", frames=[], gt={}, width=W, height=H)) is None
    with pytest.raises(ValueError):
        rt.choose_target_id(seq, "nope")


# --------------------------------------------------------- masking ----------


@pytest.mark.parametrize("mode", ["lower", "upper", "left", "right"])
def test_mask_plan_rewrites_the_target_box_to_its_visible_half(mode):
    seq = _seq(N)
    plan = rt.build_mask_plan(seq, 1, mode)
    full = seq.gt_at(0)[0].xyxy
    visible = plan.gt_target[0]
    hidden = plan.hidden[0]
    assert visible is not None and hidden is not None
    # the hidden rect covers the complementary half and stays in the frame
    assert hidden[0] >= 0 and hidden[1] >= 0 and hidden[2] <= W and hidden[3] <= H
    assert rt._area(rt._intersect(visible, hidden)) <= 1e-6
    # visible + hidden == full up to the configured pad
    assert visible[2] - visible[0] <= full[2] - full[0] + 1e-6
    assert plan.visibility[0] < 1.0
    # mask='lower' hides everything below the vertical midpoint
    if mode == "lower":
        assert hidden[1] == pytest.approx((full[1] + full[3]) / 2.0)
        assert visible[3] == pytest.approx(hidden[1])
    if mode == "upper":
        assert visible[1] == pytest.approx((full[1] + full[3]) / 2.0)


def test_mask_plan_none_is_the_identity_transform():
    seq = _seq(N)
    plan = rt.build_mask_plan(seq, 1, "none")
    assert plan.is_identity and plan.delta == (0, 0) and plan.hidden == {}
    for i in range(N):
        assert plan.visibility[i] == 1.0
        assert [b.track_id for b in plan.all_gt[i]] == [b.track_id for b in seq.gt_at(i)]
        # NOTE: `[0]` is only the target in this fixture (the synthetic sequence lists the
        # target first). On real data `gt_at(i)[0]` is a DIFFERENT person; resolve the target
        # with `tgtrack.data.make_target_gt`. Never copy this `[0]` convention into a script
        # (it produced the E020/E060 misattributions, retracted in E062).
        np.testing.assert_allclose(plan.gt_target[i], seq.gt_at(i)[0].xyxy)
        # verbatim: no clipping, no visibility edits
        np.testing.assert_allclose(plan.all_gt[i][0].xyxy, seq.gt_at(i)[0].xyxy)


def test_mask_plan_edge_translates_the_target_off_screen():
    seq = _seq(N)
    plan = rt.build_mask_plan(seq, 1, "edge", edge_frac=0.5)
    # the target starts nearer the left border, so the scene is pushed LEFT and
    # the target ends up straddling x=0
    assert plan.delta[0] < 0
    box0 = plan.gt_target[0]
    full0 = seq.gt_at(0)[0].xyxy
    assert box0 is not None and box0[0] == pytest.approx(0.0)
    assert box0[2] <= W
    assert box0[2] - box0[0] < full0[2] - full0[0]
    assert plan.visibility[0] == pytest.approx((box0[2] - box0[0]) / (full0[2] - full0[0]))
    assert plan.visibility[0] < 1.0
    # the whole scene moves: other people's boxes shift too, and are not clipped
    other = plan.all_gt[0][1]
    assert other.xyxy[0] == pytest.approx(seq.gt_at(0)[1].xyxy[0] + plan.delta[0])
    assert other.visibility == pytest.approx(1.0)


def test_mask_plan_paints_the_hidden_half_without_touching_the_cache():
    seq = _seq(N)
    plan = rt.build_mask_plan(seq, 1, "lower", fill=0)
    raw = seq.frame(0)
    before = raw.copy()
    masked = plan.apply(raw, 0)
    assert masked is not None and not np.array_equal(masked, before)
    # the loader's cached array is untouched (MaskPlan must copy before painting)
    assert np.array_equal(seq.frame(0), before)
    x0, y0, x1, y1 = (int(round(float(v))) for v in plan.hidden[0])
    assert masked[y0 : y1 - 1, x0 : x1 - 1].max() == 0  # painted over
    vis = plan.gt_target[0]
    vx0, vy0, vx1, vy1 = (int(round(float(v))) for v in vis)
    assert masked[vy0 + 1 : vy1 - 1, vx0 + 1 : vx1 - 1].min() > 0  # visible half kept


def test_mask_plan_rejects_unknown_modes_and_parses_mask_lists():
    seq = _seq(N)
    with pytest.raises(ValueError):
        rt.build_mask_plan(seq, 1, "diagonal")
    assert rt._parse_masks("none,lower") == ["none", "lower"]
    assert rt._parse_masks("") == ["none"]
    with pytest.raises(ValueError):
        rt._parse_masks("none,sideways")


def test_mask_plan_drops_people_pushed_out_of_frame():
    """The edge stress must remove anyone it pushes fully outside the image."""
    gt = {
        i: [
            GtBox(frame=i, track_id=1, xyxy=np.array([280.0, 20.0, 310.0, 90.0])),
            GtBox(frame=i, track_id=2, xyxy=np.array([305.0, 20.0, 320.0, 90.0])),
        ]
        for i in range(4)
    }
    seq = Sequence(
        name="border",
        frames=[np.zeros((H, W, 3), np.uint8) for _ in range(4)],
        gt=gt,
        fps=25.0,
        width=W,
        height=H,
    )
    plan = rt.build_mask_plan(seq, 1, "edge", edge_frac=0.5, drop_below=0.2)
    # target nearer the right border -> pushed right by ceil(0.5 * 30) = 15 px
    assert plan.delta[0] == 15
    ids = {int(b.track_id) for i in range(4) for b in plan.all_gt[i]}
    assert 1 in ids and 2 not in ids  # id 2 lands fully outside -> dropped
    assert plan.gt_target[0] is not None
    # the target survived but is only 25/30 visible
    assert plan.visibility[0] == pytest.approx(25.0 / 30.0)


# ------------------------------------------------------- single runs -------


def test_synthetic_sequence_is_self_consistent():
    seq = rt.synthetic_sequence(12)
    assert len(seq) == 12 and seq.frame(0) is not None and seq.frame(0).shape == (240, 320, 3)
    assert sorted(seq.gt_ids()) == [1, 2, 3]
    # the stressed target really is interrupted
    present = [i for i in range(12) if any(b.track_id == 1 for b in seq.gt_at(i))]
    assert present and len(present) < 12
    assert rt.choose_target_id(seq, "stressed") == 1


def test_run_sequence_reports_every_required_key(cfg, embedder):
    seq = _seq(N)
    run = rt.run_sequence(seq, 1, cfg, mask="none", detector_kind="gt", embedder=embedder, quiet=True)
    missing = [k for k in rt.REQUIRED_RUN_KEYS if k not in run]
    assert missing == []
    assert run["sequence"] == "clean" and run["target_id"] == 1 and run["mask"] == "none"
    assert run["clicked_id"] == 1 and run["frames_processed"] == N
    assert run["metrics"]["n_frames"] == N
    # a clean, fully visible single-target sequence must be tracked perfectly
    assert run["target_success_rate"] == pytest.approx(1.0)
    assert run["target_precision"] == pytest.approx(1.0)
    assert run["loss_rate"] == pytest.approx(0.0)
    assert run["false_acquisition_rate"] == pytest.approx(0.0)
    assert run["false_acquisition_events"] == 0
    assert run["id_consistency"] == pytest.approx(1.0)
    assert run["target_switch_count"] == 0
    assert run["longest_correct_streak"] == N
    assert run["ms_per_frame"] > 0.0
    emitted = sum(v for k, v in run["metrics"].items() if k.startswith("n_events_"))
    assert run["n_events"] == emitted


def test_wrong_position_run_scores_worse_than_the_correct_one(cfg, embedder):
    seq = _seq(N)
    good = rt.run_sequence(seq, 1, cfg, mask="none", detector_kind="gt", embedder=embedder, quiet=True)
    bad = rt.run_sequence(
        seq, 1, cfg, mask="none", detector=ShiftedDetector(seq, 150.0), embedder=embedder, quiet=True
    )
    assert good["target_success_rate"] == pytest.approx(1.0)
    assert bad["target_success_rate"] < good["target_success_rate"]
    assert bad["loss_rate"] > good["loss_rate"]
    # the metric layer agrees with the harness about what happened
    assert bad["metrics"]["n_correct_frames"] < bad["metrics"]["n_visible_frames"]


def test_run_sequence_handles_an_absent_target_and_the_full_gt_reading(cfg, embedder):
    seq = _seq(N, gap=(3, 6))
    run = rt.run_sequence(seq, 1, cfg, mask="lower", detector_kind="gt", embedder=embedder, quiet=True)
    assert run["metrics"]["n_absent_frames"] == 3
    assert "target_success_rate_fullgt" in run and 0.0 <= run["target_success_rate_fullgt"] <= 1.0
    # the masked GT is a strict subset of the full-body GT: the visible half is
    # what the run is scored against
    assert run["metrics"]["n_visible_frames"] == N - 3


def test_run_sequence_writes_a_debug_csv(cfg, embedder, tmp_path):
    seq = _seq(N)
    csv_path = tmp_path / "run.csv"
    run = rt.run_sequence(
        seq, 1, cfg, mask="none", detector_kind="gt", embedder=embedder, debug_csv=csv_path, quiet=True
    )
    assert csv_path.exists() and run["debug_csv"] == str(csv_path)
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == N
    assert {"frame", "reported", "pred_xyxy", "correct", "false_acquisition", "state"} <= set(rows[0])
    assert sum(1 for r in rows if r["correct"] == "True") == N


def test_save_video_prepares_its_directory_and_degrades_cleanly(cfg, embedder, tmp_path):
    """`--save-video` must create the parent dir; a missing codec must not crash."""
    seq = rt.synthetic_sequence(8)
    target = tmp_path / "runs" / "x.mp4"
    run = rt.run_sequence(
        seq, 1, cfg, mask="none", detector_kind="gt", embedder=embedder, video_path=target, quiet=True
    )
    assert target.parent.is_dir()  # always prepared, even if the codec is absent
    assert "metrics" in run
    if run.get("video"):
        assert Path(run["video"]).exists() and Path(run["video"]).stat().st_size > 0


def test_run_sequence_rejects_an_impossible_run(cfg, embedder):
    seq = _seq(2)
    with pytest.raises(ValueError):
        rt.run_sequence(seq, 99, cfg, mask="none", detector_kind="gt", embedder=embedder)
    with pytest.raises(ValueError):
        rt.run_sequence(seq, 1, cfg, mask="sideways", detector_kind="gt", embedder=embedder)


def test_partial_view_runs_stay_above_chance_for_every_mask(cfg, embedder):
    """Requirement (4): partial view must not be fatal to the target layer."""
    seq = _seq(N)
    results = {}
    for mask in ("none", "lower", "upper", "left", "right", "edge"):
        run = rt.run_sequence(seq, 1, cfg, mask=mask, detector_kind="gt", embedder=embedder, quiet=True)
        results[mask] = run["target_success_rate"]
        assert run["target_success_rate"] >= 0.5, (mask, run["target_success_rate"])
    # the un-stressed run must be at least as good as any stressed run
    assert results["none"] >= max(results.values()) - 1e-9


# ------------------------------------------------ grid + outputs -----------


def test_grid_pools_by_variant_and_mask_and_writes_outputs(cfg, tmp_path):
    seq_dir = _write_mot_dir(tmp_path, n=6)
    specs = [
        _spec(seq_dir, mask="none"),
        _spec(seq_dir, mask="lower"),
        _spec(seq_dir, variant="part_matching_off", group="part_matching", mask="none",
              overrides={"cost.part_matching": False}),
        _spec(seq_dir, variant="part_matching_off", group="part_matching", mask="lower",
              overrides={"cost.part_matching": False}),
        # an unloadable spec must be reported, not raised
        _spec(tmp_path / "does-not-exist", mask="none"),
    ]
    runs = rt.run_grid(specs, jobs=1, quiet=True)
    assert len(runs) == len(specs)
    assert sum(1 for r in runs if r.get("error")) == 1
    pooled = rt.pool_runs(runs)
    assert {(r["variant"], r["mask"]) for r in pooled} == {
        ("baseline", "none"),
        ("baseline", "lower"),
        ("part_matching_off", "none"),
        ("part_matching_off", "lower"),
    }
    for row in pooled:
        assert row["n_sequences"] == 1
        for key in ("target_success_rate", "false_acquisition_rate", "false_acquisition_events",
                    "id_consistency", "target_switch_count", "fragmentation", "reacquire_latency_p90",
                    "reacquire_latency", "ms_per_frame"):
            assert key in row, key

    written = rt.write_outputs(runs, pooled, tmp_path / "out", meta={"note": "unit-test"})
    for path in written.values():
        if isinstance(path, Path):
            assert path.exists() and path.stat().st_size > 0
    payload = json.loads((tmp_path / "out" / "per_run.json").read_text(encoding="utf-8"))
    assert payload["n_runs"] == len(specs) and payload["n_errors"] == 1
    assert payload["meta"]["note"] == "unit-test"
    md = (tmp_path / "out" / "table.md").read_text(encoding="utf-8")
    assert "Partial-view cross-tab" in md and "part_matching_off" in md
    # NaN must be rendered as '-', never as the string 'nan'
    assert "nan" not in md.lower()
    csv_text = (tmp_path / "out" / "table.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("group,variant,mask,seqs,frames")


def test_claims_and_partial_view_tables_answer_the_headline_question():
    pooled = [
        {"variant": "baseline", "group": "baseline", "mask": "lower", "target_success_rate": 0.80,
         "false_acquisition_rate": 0.20, "false_acquisition_events": 2, "target_precision": 0.8,
         "ms_per_frame": 5.0, "n_sequences": 1},
        {"variant": "part_matching_off", "group": "part_matching", "mask": "lower", "target_success_rate": 0.60,
         "false_acquisition_rate": 0.30, "false_acquisition_events": 3, "target_precision": 0.7,
         "ms_per_frame": 4.0, "n_sequences": 1},
    ]
    table = "\n".join(rt.partial_view_table(pooled))
    assert "part_matching" in table and "| lower |" in table
    # ON TSR = 0.80, OFF TSR = 0.60 -> dTSR column shows the loss
    assert "0.8000" in table and "0.6000" in table
    claims = "\n".join(rt.claims_lines(pooled))
    assert "part_matching_off" in claims and "TSR 0.6000 vs 0.8000 (-0.2000)" in claims
    assert "FA events 3 vs 2 (+1)" in claims


def test_run_spec_forwards_the_mask_geometry_options(tmp_path):
    seq_dir = _write_mot_dir(tmp_path, n=6)
    small = rt.run_spec(_spec(seq_dir, mask="edge", opts={"edge_frac": 0.1}))
    big = rt.run_spec(_spec(seq_dir, mask="edge", opts={"edge_frac": 0.8}))
    assert not small.get("error") and not big.get("error")
    assert abs(big["mask_delta"][0]) > abs(small["mask_delta"][0])


def test_run_spec_reports_errors_instead_of_raising(tmp_path):
    run = rt.run_spec(_spec(tmp_path / "nope", mask="none"))
    assert run["error"].startswith("FileNotFoundError") or run["error"]
    assert "metrics" not in run


# ------------------------------------------------------------- CLI ---------


def test_cli_list_variants_and_bad_data_root(capsys, tmp_path):
    assert rt.main(["--list-variants"]) == 0
    assert "cost.part_matching=False" in capsys.readouterr().out
    assert rt.main(["--data-root", str(tmp_path / "missing"), "--detector", "gt"]) == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_cli_dry_run_end_to_end(tmp_path, capsys):
    out = tmp_path / "res"
    code = rt.main(
        [
            "--dry-run",
            "--detector", "gt",
            "--embed-fallback",
            "--device", "cpu",
            "--max-frames", "8",
            "--ablate", "part_matching",
            "--mask", "none,lower",
            "--out", str(out),
            "--quiet",
        ]
    )
    assert code == 0
    for name in ("per_run.json", "table.md", "table.csv"):
        assert (out / name).exists(), name
    printed = capsys.readouterr().out
    assert "[result] runs=4 ok=4 errors=0" in printed
    assert "TSR=" in printed
    payload = json.loads((out / "per_run.json").read_text(encoding="utf-8"))
    assert payload["n_runs"] == 4 and payload["n_errors"] == 0
    assert all("metrics" in r for r in payload["runs"])
    # the table must never claim a real ReID model when the fallback was used
    assert payload["meta"]["embedder"] == "fallback"
    assert (out / "table.md").read_text(encoding="utf-8").count("part_matching_off") >= 2


def test_cli_jobs_uses_a_process_pool(tmp_path):
    out = tmp_path / "par"
    code = rt.main(
        [
            "--dry-run",
            "--detector", "gt",
            "--embed-fallback",
            "--device", "cpu",
            "--max-frames", "6",
            "--ablate", "part_matching",
            "--mask", "none,lower",
            "--jobs", "2",
            "--out", str(out),
            "--quiet",
        ]
    )
    assert code == 0
    payload = json.loads((out / "per_run.json").read_text(encoding="utf-8"))
    assert payload["n_runs"] == 4 and payload["n_errors"] == 0


# ------------------------------------------ observability guard (task-10) ----


def _tiny_seq(n: int = 6, *, name: str = "tiny") -> Sequence:
    """The MOT17-02 failure shape: one 38x54 px person -- 38x27 once half-masked.

    On MOT17-02 the shipped chooser picked an identity whose post-mask visible box
    was 38x54 px; no detection ever overlapped it, so the run measured an
    invisible target and its numbers said nothing about partial-view ReID.
    """
    gt = {i: [GtBox(frame=i, track_id=1, xyxy=np.array([100.0, 40.0, 138.0, 94.0]))] for i in range(n)}
    frames = [_paint(gt[i]) for i in range(n)]
    return Sequence(name=name, frames=frames, gt=gt, fps=25.0, width=W, height=H)


def test_observability_guard_rejects_the_mot17_02_failure_shape():
    seq = _tiny_seq()
    full = rt.visible_box_stats(seq, 1, "none")
    assert (full["median_w"], full["median_h"]) == pytest.approx((38.0, 54.0))
    half = rt.visible_box_stats(seq, 1, "lower")
    assert half["median_h"] == pytest.approx(27.0, abs=1.0)
    assert half["frames"] == 6

    dec = rt.pick_target(seq, "stressed", masks=("none", "lower"), limit=6)
    assert dec["target_id"] is None and dec["reason"] == "no_candidate"
    assert dec["guard_passed"] is False and dec["control"] == rt.CONTROL_PARTIAL_VIEW
    assert "no ground-truth identity survives" in dec["message"]
    assert "id 1 median 38x27" in dec["message"]
    assert "--allow-invisible-target" in dec["message"]

    # the bar is a knob: lowering it admits the very same identity again
    ok = rt.pick_target(seq, "stressed", masks=("none", "lower"), limit=6, min_visible_h=20.0)
    assert ok["target_id"] == 1 and ok["reason"] == "observable"
    assert ok["guard_passed"] is True and ok["control"] == rt.CONTROL_PARTIAL_VIEW
    # an area bar bites as well (38x27 = 1026 px^2)
    area = rt.pick_target(
        seq, "stressed", masks=("none", "lower"), limit=6, min_visible_h=20.0, min_visible_area=2000.0
    )
    assert area["reason"] == "no_candidate" and "px^2" in area["message"]
    # an impossible floor is refused rather than quietly relaxed
    assert rt.pick_target(seq, "stressed", masks=("lower",), limit=6, min_visible_h=1e6)["reason"] == "no_candidate"


def test_pick_target_control_paths_are_labelled_not_silent():
    seq = _tiny_seq()
    mask = ("none", "lower")
    ctl = rt.pick_target(seq, "stressed", masks=mask, limit=6, allow_invisible=True)
    assert ctl["reason"] == "invisible_allowed" and ctl["invisible"] is True
    assert ctl["control"] == rt.CONTROL_INVISIBLE and ctl["guard_passed"] is False
    assert "UNOBSERVABLE" in ctl["message"] and "chosen by the strategy" in ctl["message"]

    strat = rt.pick_target(seq, "invisible", masks=mask, limit=6)
    assert strat["reason"] == "invisible_strategy" and strat["invisible"] is True
    assert strat["control"] == rt.CONTROL_INVISIBLE

    # --target-id is an explicit override: it wins, but the run is still labelled
    forced = rt.pick_target(seq, "stressed", masks=mask, limit=6, forced_id=1)
    assert forced["reason"] == "forced_invisible" and forced["invisible"] is True
    assert "forced by --target-id" in forced["message"]

    good = rt.pick_target(seq, "stressed", masks=("none",), limit=6, forced_id=1, min_visible_h=20.0)
    assert good["reason"] == "forced" and good["invisible"] is False
    assert good["control"] == rt.CONTROL_PARTIAL_VIEW

    missing = rt.pick_target(seq, "stressed", masks=mask, limit=6, forced_id=99)
    assert missing["target_id"] is None and missing["reason"] == "forced_missing"
    assert "not a ground-truth identity" in missing["message"]

    empty = rt.pick_target(Sequence(name="empty", frames=[], gt={}, width=W, height=H), "stressed")
    assert empty["target_id"] is None and empty["reason"] == "no_identities"


def test_guard_accepts_the_synthetic_clip_under_every_mask():
    seq = rt.synthetic_sequence(24)
    obs = rt.observability_table(seq, ("none", "lower", "upper", "left", "right", "edge"))
    assert sorted(obs) == [1, 2, 3]
    for tid, rec in obs.items():
        assert rec["frames"] > 0
        # the clip is tall enough that no mask can push it under the default bar
        assert rec["median_h"] >= rt.DEFAULT_MIN_VISIBLE_H
    dec = rt.pick_target(seq, "stressed", masks=("none", "lower", "upper", "edge"), limit=24)
    assert dec["target_id"] == 1 and dec["reason"] == "observable"
    assert dec["guard_passed"] is True and dec["control"] == rt.CONTROL_PARTIAL_VIEW


def test_classify_wrong_lock_splits_confident_from_coasting():
    def row(frame, *, reported=True, locked=True, best_id=2, best_iou=0.9, tgt_id=1, correct=False, fa=False):
        return {
            "frame": frame,
            "reported": reported,
            "locked": locked,
            "best_gt_id": best_id,
            "best_gt_iou": best_iou,
            "target_person_id": tgt_id,
            "correct": correct,
            "false_acquisition": fa,
        }

    rows = [
        row(0, best_id=1, correct=True),  # confident, right person
        row(1, best_iou=0.22),  # confident wrong lock at IoU 0.22
        row(2, best_iou=0.9, fa=True),  # confident wrong lock the metric layer also books
        row(3, locked=False, best_id=1, correct=True),  # coasting, still on the target
        row(4, locked=False, best_iou=0.7),  # coasting onto somebody else
        row(5, locked=False, best_id=None, best_iou=0.0),  # blind, no overlap at all
        row(6, reported=False, locked=False),  # nothing reported
    ]
    out = rt.classify_wrong_lock(rows)
    assert out["wrong_identity_locked"] == 2
    assert out["wrong_identity_locked_frames"] == [1, 2]
    assert out["wrong_identity_locked_first_frame"] == 1
    assert out["wrong_identity_iou_bar"] == rt.WRONG_IDENTITY_IOU
    # the metric layer only books a false acquisition at IoU >= 0.5, so it sees one
    assert out["metric_locked_false_acquisition_frames"] == 1
    assert out["coasting_correct_frames"] == 1
    assert out["coasting_wrong_frames"] == 1
    assert out["hallucination_locked_frames"] == 0

    hall = rt.classify_wrong_lock([row(0, best_id=None, best_iou=0.0)])
    assert hall["wrong_identity_locked"] == 0 and hall["hallucination_locked_frames"] == 1


def test_run_sequence_attributes_the_wrong_lock_and_agrees_with_the_metrics(cfg, embedder):
    seq = _seq(N)
    good = rt.run_sequence(seq, 1, cfg, mask="none", detector_kind="gt", embedder=embedder, quiet=True)
    bad = rt.run_sequence(
        seq, 1, cfg, mask="none", detector=ShiftedDetector(seq, 150.0), embedder=embedder, quiet=True
    )
    for run in (good, bad):
        assert run["control"] == rt.CONTROL_PARTIAL_VIEW
        assert run["invisible_target"] is False
        assert run["guard_passed"] is None  # a direct call never ran the guard
        assert run["target_reason"] == "direct"
        assert run["onset_metric_agrees"] is True
        # the run may not claim a wrong lock the metrics do not see
        assert run["wrong_identity_locked"] >= run["metrics"]["false_acquisition_frames_locked"]
        attributed = (
            run["wrong_identity_locked"]
            + run["hallucination_locked_frames"]
            + run["coasting_correct_frames"]
            + run["coasting_wrong_frames"]
        )
        assert attributed <= run["frames_processed"]
    assert good["wrong_identity_locked"] == 0
    assert good["coasting_correct_frames"] == 0
    # the observability fields are measured for every run, guard or no guard
    for run in (good, bad):
        assert run["visible_frames"] > 0
        assert run["detected_frames"] >= 0
        assert 0.0 <= run["detection_rate"] <= 1.0
        assert run["visible_median_h"] > 0.0


def test_pool_and_outputs_keep_the_control_out_of_the_headline(tmp_path):
    seq_dir = _write_mot_dir(tmp_path)
    runs = [rt.run_spec(_spec(seq_dir, mask="none"))]
    ctl = dict(rt.run_spec(_spec(seq_dir, mask="lower")))
    ctl["control"] = rt.CONTROL_INVISIBLE
    ctl["invisible_target"] = True
    runs.append(ctl)

    pooled = rt.pool_runs(runs)
    control = rt.pool_runs(runs, control=rt.CONTROL_INVISIBLE)
    assert {r["mask"] for r in pooled} == {"none"}
    assert {r["mask"] for r in control} == {"lower"}
    assert pooled[0]["control"] == rt.CONTROL_PARTIAL_VIEW

    out = tmp_path / "out"
    written = rt.write_outputs(
        runs,
        pooled,
        out,
        meta={"note": "task-10", "guard": {"min_visible_h": rt.DEFAULT_MIN_VISIBLE_H, "min_visible_area": 0.0}},
        invisible=control,
    )
    for key in ("per_run.json", "summary.json", "table.csv", "table.md"):
        assert written[key].exists() and written[key].stat().st_size > 0
    assert written["n_runs"] == 2 and written["n_headline_runs"] == 1 and written["n_invisible_runs"] == 1

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_runs"] == 2 and summary["n_headline_runs"] == 1 and summary["n_invisible_runs"] == 1
    assert len(summary["headline"]) == 1 and len(summary["control"]) == 1
    assert summary["meta"]["guard"]["min_visible_h"] == pytest.approx(rt.DEFAULT_MIN_VISIBLE_H)
    picked = summary["targets"][Path(seq_dir).name]
    assert picked["target_id"] == 1 and picked["reason"] == "direct"
    assert picked["onsets_agree"] is True

    md = (out / "table.md").read_text(encoding="utf-8")
    assert "Invisible-target control (excluded from the headline)" in md
    assert "Observability guard:" in md
    assert "nan" not in md.lower()

    csv_text = (out / "table.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("group,variant,mask,seqs,frames")
    assert len(csv_text.splitlines()) == 2  # header + the single headline row
    assert "coast_ok" in csv_text.splitlines()[0] and "detect" in csv_text.splitlines()[0]


def test_run_spec_forwards_the_guard_decision_and_flags_a_disagreement(tmp_path):
    seq_dir = _write_mot_dir(tmp_path)
    spec = _spec(seq_dir, mask="lower")
    spec["target"] = {
        "reason": "invisible_allowed",
        "control": rt.CONTROL_INVISIBLE,
        "invisible": True,
        "guard_passed": False,
        "guard": {"min_visible_h": rt.DEFAULT_MIN_VISIBLE_H, "min_visible_area": 0.0},
        "selected": {"median_h": 35.0, "median_area": 1050.0, "frames": N},
        "message": "control run",
    }
    run = rt.run_spec(spec)
    assert "error" not in run
    assert run["control"] == rt.CONTROL_INVISIBLE and run["invisible_target"] is True
    assert run["target_reason"] == "invisible_allowed" and run["guard_passed"] is False
    assert run["guard_visible_median_h"] == pytest.approx(35.0)
    assert run["guard_visible_frames"] == N
    # the guard's numbers and the run's own measurement agree (30x70 box, lower half)
    assert run["guard_stats_agree"] is True
    assert run["visible_median_h"] == pytest.approx(35.0)

    # a decision that contradicts the measurement is reported, not hidden
    spec2 = _spec(seq_dir, mask="lower")
    spec2["target"] = {
        "reason": "observable",
        "control": rt.CONTROL_PARTIAL_VIEW,
        "invisible": False,
        "guard_passed": True,
        "selected": {"median_h": 999.0, "median_area": 1.0, "frames": 1},
        "message": "",
    }
    run2 = rt.run_spec(spec2)
    assert run2["guard_stats_agree"] is False


def test_cli_guard_fails_loudly_and_the_control_flag_labels_the_run(tmp_path, capsys):
    seq_root = tmp_path / "seqs"
    _write_mot_dir(seq_root, n=6, name="TINY-01")
    out = tmp_path / "blocked"
    code = rt.main(
        [
            "--data-root", str(seq_root),
            "--detector", "gt",
            "--embed-fallback",
            "--device", "cpu",
            "--max-frames", "6",
            "--ablate", "part_matching",
            "--mask", "lower",
            "--out", str(out),
            "--quiet",
        ]
    )
    assert code == 3
    err = capsys.readouterr().err
    assert "[guard]" in err and "no ground-truth identity survives" in err
    # nothing was measured, so nothing is written that could be mistaken for a result
    assert not (out / "table.csv").exists()
    assert not (out / "table.md").exists()

    ctl_out = tmp_path / "control"
    code = rt.main(
        [
            "--data-root", str(seq_root),
            "--detector", "gt",
            "--embed-fallback",
            "--device", "cpu",
            "--max-frames", "6",
            "--ablate", "part_matching",
            "--mask", "lower",
            "--allow-invisible-target",
            "--out", str(ctl_out),
            "--quiet",
        ]
    )
    assert code == 0
    summary = json.loads((ctl_out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_headline_runs"] == 0 and summary["n_invisible_runs"] == 2
    assert all(r["invisible_target"] is True for r in summary["control"])
    csv_text = (ctl_out / "table.csv").read_text(encoding="utf-8")
    assert len(csv_text.splitlines()) == 1  # header only: no headline rows
    md = (ctl_out / "table.md").read_text(encoding="utf-8")
    assert "Invisible-target control (excluded from the headline)" in md
    assert "nan" not in md.lower()


def test_cli_observability_lists_every_identity(tmp_path, capsys):
    out = tmp_path / "obs"
    code = rt.main(
        ["--dry-run", "--observability", "--mask", "none,lower", "--max-frames", "12", "--out", str(out), "--quiet"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "[obs]" in printed and "id=1" in printed and "mask=lower" in printed
    assert not (out / "table.csv").exists()  # a survey is not a result


def test_mot17_02_guard_regression_on_real_data_if_present():
    root = Path(__file__).resolve().parents[1]
    roots = [root / "data" / "sequences", root / "data" / "raw" / "mot17" / "MOT17" / "train"]
    candidates: list[Path] = []
    for r in roots:
        if r.is_dir():
            candidates += sorted(p for p in r.glob("*MOT17-02*") if p.is_dir())
    if not candidates:
        pytest.skip("no MOT17-02 sequence under data/sequences or data/raw/mot17")
    seq = load_mot_sequence(candidates[0], subsample=2, max_frames=60)
    masks = ("none", "upper", "lower", "edge")
    obs = rt.observability_table(seq, masks, limit=60)
    if 17 in obs:
        # the identity the unguarded chooser used to pick: 38x54 px, undetectable
        assert obs[17]["median_h"] < rt.DEFAULT_MIN_VISIBLE_H
    dec = rt.pick_target(seq, "stressed", masks=masks, limit=60)
    # the invariant under test: a selected target is observable under every mask,
    # and the invisible identity can never be selected
    assert dec["reason"] == "observable"
    assert dec["target_id"] != 17
    assert dec["guard_passed"] is True and dec["control"] == rt.CONTROL_PARTIAL_VIEW
    assert dec["selected"]["median_h"] >= rt.DEFAULT_MIN_VISIBLE_H
    assert dec["selected"]["frames"] > 0
