"""Tests for the pure parts of `scripts/gate_sweep.py` (E070).

The sweep itself needs the GPU, but the parts that decide *what gets measured and how it is
pooled* are pure functions and are where a silent mistake would be most expensive:

- `point_spec` must produce exactly one override per axis point, or a "sweep" silently
  measures the same configuration several times;
- `all_points` must enumerate the documented axes, and a typo'd point must fail loudly;
- the dev-split constants must actually contain the frozen sequences, because every
  "held-out" claim in the ledger rests on them being excluded;
- `_weighted` / `_pct` / `cluster_bootstrap` must pool by frames, not by sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import gate_sweep as gs  # noqa: E402


# ------------------------------------------------------------------ point specs
@pytest.mark.parametrize("margin", [0.0, 0.10, 0.15, 0.20, 0.30])
def test_margin_points_override_exactly_one_key(margin):
    spec = gs.point_spec(f"margin_{margin:.2f}")
    assert spec["overrides"] == {"cost.search_margin": pytest.approx(margin)}
    assert spec["axis"] == "cost.search_margin"


def test_only_the_shipped_margin_is_flagged_as_shipped():
    for m in gs.PRIMARY_MARGINS:
        spec = gs.point_spec(f"margin_{m:.2f}")
        assert spec["shipped"] is (abs(m - gs.SHIPPED_MARGIN) < 1e-9)


@pytest.mark.parametrize("k", [3.0, 4.0])
def test_band_points_override_the_adaptive_k(k):
    spec = gs.point_spec(f"band_k{k:.1f}")
    assert spec["overrides"] == {"cost.anchor_adaptive_k": pytest.approx(k)}
    assert spec["axis"] == "cost.anchor_adaptive_k"


def test_unknown_point_fails_loudly():
    # The property that matters is "fails loudly instead of silently defaulting".
    # NOTE: a malformed *numeric* suffix currently raises ValueError from float() rather
    # than the module's SystemExit("unknown point id") -- a cosmetic inconsistency in
    # scripts/gate_sweep.py, deliberately NOT fixed while the E070 sweep was executing
    # (editing the harness mid-sweep would mix code versions across the 110 pairs).
    with pytest.raises((SystemExit, ValueError)):
        gs.point_spec("margin_not_a_number")
    with pytest.raises(SystemExit) as exc:
        gs.point_spec("nonsense_point")
    assert "unknown point id" in str(exc.value)


# ------------------------------------------------------------------- enumeration
def test_primary_axis_is_the_documented_five_margins():
    assert gs.all_points("primary") == [f"margin_{m:.2f}" for m in gs.PRIMARY_MARGINS]
    assert len(gs.all_points("primary")) == 5
    assert all(p.startswith("margin_") for p in gs.all_points("primary"))


def test_secondary_axis_covers_the_shipped_band_k():
    pts = gs.all_points("secondary")
    assert pts == [f"band_k{k:.1f}" for k in gs.SECONDARY_BAND_KS]
    assert any(abs(k - gs.SHIPPED_BAND_K) < 1e-9 for k in gs.SECONDARY_BAND_KS), \
        "the shipped value must be inside the sweep or there is no reference point"


def test_every_enumerated_point_is_resolvable():
    for which in ("primary", "secondary"):
        for point in gs.all_points(which):
            gs.point_spec(point)  # must not raise


# ------------------------------------------------------------------ split guards
def test_dev_sequences_constant_holds_the_frozen_three():
    assert gs.DEV_SEQUENCES == {"dancetrack0004", "dancetrack0005", "dancetrack0007"}


def test_no_axis_point_can_reintroduce_dev_data():
    # the axes are pure config overrides; they must not name sequences at all
    for which in ("primary", "secondary"):
        for point in gs.all_points(which):
            spec = gs.point_spec(point)
            blob = repr(spec).lower()
            for dev in gs.DEV_SEQUENCES:
                assert dev.lower() not in blob
            assert "dancetrack" not in blob


# --------------------------------------------------------------------- pooling
def test_weighted_pooling_is_frame_weighted():
    # 1.0 over 100 frames and 0.0 over 300 frames -> 0.25, not the sequence mean 0.5
    assert gs._weighted([1.0, 0.0], [100.0, 300.0]) == pytest.approx(0.25)


def test_weighted_pooling_reports_nan_when_there_is_no_data():
    # nan is the correct signal: 0.0 would masquerade as a real measurement of zero,
    # and the ledger must never show a metric for an empty pool
    import math

    assert math.isnan(gs._weighted([0.7], [0.0]))


def test_bootstrap_is_deterministic_for_a_given_seed():
    rows = [{"tsr": 0.1 + 0.05 * i, "n_report_frames": 100} for i in range(8)]
    a = gs.cluster_bootstrap(rows, key="tsr", reps=200, seed=0)
    b = gs.cluster_bootstrap(rows, key="tsr", reps=200, seed=0)
    assert a == b, "a bootstrap that is not seeded cannot be reproduced from the ledger"


# ------------------------------------------------------- corrupt-summary resilience
def test_corrupt_summary_becomes_an_error_record_not_a_crash(tmp_path):
    """One corrupted summary must cost one pair, never the whole sweep.

    Observed once on this machine: a summary.json held a valid document plus a stray
    trailing brace (two writers racing), and `json.loads` aborted a 110-pair sweep at
    pair 13. The contract is now "return an ERROR record; the caller retries once".
    """
    bad = tmp_path / "summary.json"
    bad.write_text("this is not json at all", encoding="utf-8")
    specs = tmp_path / "specs.json"
    specs.write_text("[]", encoding="utf-8")
    rec = gs._parse_summary(bad, "dancetrack0058", None, specs)
    assert rec["status"] == "ERROR"
    assert "corrupt summary.json" in rec["reason"]


def test_missing_summary_is_an_error_record(tmp_path):
    rec = gs._parse_summary(tmp_path / "nope.json", "seq", None, tmp_path / "specs.json")
    assert rec["status"] == "ERROR"
    assert "no summary.json" in rec["reason"]
