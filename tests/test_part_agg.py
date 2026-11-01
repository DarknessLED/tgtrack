"""E106 -- the part-aggregation knob (`cost.part_agg`), and its mandatory control.

The shipped appearance cost keeps the single cheapest (query part, gallery part) pair.  E105 measured
offline, on the pre-veto candidate pool, that averaging the best match of *every* available query part
ranks the target better (rank@1 48.5 % vs 44.2 %, AUC 0.813 vs 0.779).  These tests pin the algebra of
the new mode and the fact that the shipped default is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from tgtrack.association import _aggregate_part_terms, appearance_cost_of_query
from tgtrack.config import Config


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / float(np.linalg.norm(a))


def _terms(*rows) -> list:
    return [(float(c), (qp, gp)) for c, qp, gp in rows]


# --------------------------------------------------------------------------- #
# The reduction itself
# --------------------------------------------------------------------------- #
def test_default_aggregation_is_the_shipped_min() -> None:
    assert Config().cost.part_agg == "min"


def test_empty_term_list_is_none_for_both_modes() -> None:
    assert _aggregate_part_terms([], "min") is None
    assert _aggregate_part_terms([], "mean") is None


def test_min_keeps_the_single_cheapest_pair() -> None:
    terms = _terms((0.10, "upper", "full"), (0.50, "full", "full"), (0.90, "lower", "full"))
    assert _aggregate_part_terms(terms, "min") == pytest.approx(0.10)


def test_mean_averages_one_best_match_per_query_part() -> None:
    # min is a single lucky pair at 0.10; mean must also charge the full-body part (0.50).
    terms = _terms((0.10, "upper", "full"), (0.50, "full", "full"))
    assert _aggregate_part_terms(terms, "mean") == pytest.approx(0.30)


def test_mean_takes_the_best_gallery_match_within_each_query_part() -> None:
    # upper matches two gallery parts (0.20 / 0.40) -> only 0.20 counts; head adds 0.60.
    terms = _terms((0.20, "upper", "upper"), (0.40, "upper", "full"), (0.60, "head", "head"))
    assert _aggregate_part_terms(terms, "mean") == pytest.approx((0.20 + 0.60) / 2.0)


def test_unknown_mode_behaves_like_min() -> None:
    terms = _terms((0.10, "upper", "full"), (0.50, "full", "full"))
    assert _aggregate_part_terms(terms, "whatever") == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# The knob reaches the appearance cost
# --------------------------------------------------------------------------- #
def _cost(part_agg: str) -> float:
    cfg = Config().cost
    cfg.part_agg = part_agg
    ref = _unit([1.0, 0.0, 0.0, 0.0])
    query = {"full": ref, "upper": _unit([0.0, 1.0, 0.0, 0.0])}
    cost, _pair, _hn = appearance_cost_of_query(query, ref, None, cfg, 0)
    return float(cost)


def test_mean_is_stricter_than_min_when_a_second_part_disagrees() -> None:
    from tgtrack.association import part_affinity

    cheapest = _cost("min")
    averaged = _cost("mean")
    assert cheapest < averaged
    # full agrees exactly (0.0); upper is orthogonal (cosine distance 1.0) plus the
    # part penalty the matcher charges for matching an upper crop to a full-body bed.
    penalty = 0.15 * (1.0 - float(part_affinity("upper", "full")))
    assert averaged == pytest.approx((0.0 + (1.0 + penalty)) / 2.0, abs=1e-5)


def test_min_is_the_default_cost_path() -> None:
    cfg = Config().cost
    ref = _unit([1.0, 0.0, 0.0, 0.0])
    query = {"full": ref, "upper": _unit([0.0, 1.0, 0.0, 0.0])}
    explicit = appearance_cost_of_query(query, ref, None, cfg, 0)[0]
    assert explicit == pytest.approx(_cost("min"))


# --------------------------------------------------------------------------- #
# Arm contract
# --------------------------------------------------------------------------- #
def test_ablation_group_emits_both_arms_and_pins_the_vote_off() -> None:
    from scripts import run_target

    variants = {v.name: v for v in run_target.build_variants("part_agg")}
    assert "partagg_mean" in variants
    control = variants["partagg_min"]
    assert control.overrides.get("cost.part_agg") == "min"
    for name, variant in variants.items():
        if name == "baseline":
            continue
        assert variant.overrides.get("cost.anchor_vote") == "off"


def test_sweep_arms_exist_and_are_ordered() -> None:
    from scripts import vote_sweep

    for arm in ("partagg_mean", "partagg_min"):
        assert arm in vote_sweep.ARMS
        assert arm in vote_sweep.ARM_ORDER
    assert vote_sweep.ARMS["partagg_mean"]["overrides"]["cost.part_agg"] == "mean"
    assert vote_sweep.ARMS["partagg_min"]["overrides"]["cost.part_agg"] == "min"
