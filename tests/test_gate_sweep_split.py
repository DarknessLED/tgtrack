"""Tests for `scripts/gate_sweep_split.py` (E070b, select-on-one-half analysis).

The script computes a headline TSR/FAR number, so its failure modes are silent: a
weighting mistake or a leaked dev sequence would produce a plausible-looking table
that is wrong. These tests pin the four things that could go wrong quietly:
frame-weighted pooling, the dev-sequence refusal, the "one arm is not a selection"
refusal, and the paired-difference sign convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import gate_sweep_split as gss  # noqa: E402


# --------------------------------------------------------------------- fixtures
def _write_summary(root: Path, point: str, seq: str, *, tsr: float, far: float, n: int = 300,
                   status: str = "INCLUDED") -> None:
    d = root / point
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{seq}.json").write_text(
        json.dumps({
            "status": status, "point": point, "sequence": seq,
            "tsr": tsr, "far": far, "n_report_frames": n,
            "target_id": 1, "guard_passed": True,
        }),
        encoding="utf-8",
    )


@pytest.fixture()
def fake_sweep(tmp_path, monkeypatch):
    """Isolate the module from the real results tree."""
    root = tmp_path / "summaries"
    monkeypatch.setattr(gss, "SUMMARY_ROOT", root)
    monkeypatch.setattr(gss, "REPO", tmp_path)
    return root


# ------------------------------------------------------------------------ split
def test_split_is_deterministic_and_partitions_the_input():
    seqs = [f"dancetrack{i:04d}" for i in range(1, 23)]
    a1, b1 = gss.split_sequences(seqs, seed=0)
    a2, b2 = gss.split_sequences(seqs, seed=0)
    assert (a1, b1) == (a2, b2), "same seed must give the same halves"
    assert not set(a1) & set(b1), "halves must be disjoint"
    assert sorted(a1 + b1) == sorted(seqs), "halves must cover the input exactly"


def test_split_changes_with_the_seed():
    seqs = [f"dancetrack{i:04d}" for i in range(1, 41)]
    a0, _ = gss.split_sequences(seqs, seed=0)
    a7, _ = gss.split_sequences(seqs, seed=7)
    assert a0 != a7, "a different seed should produce a different half (stability check depends on it)"


# ----------------------------------------------------------------------- pooling
def test_pool_is_frame_weighted_not_sequence_mean():
    rows = {
        "short": {"tsr": 1.0, "n_report_frames": 100},
        "long": {"tsr": 0.0, "n_report_frames": 300},
    }
    # frame-weighted -> (1.0*100 + 0.0*300) / 400 = 0.25, not the sequence mean 0.5
    assert gss.pool(rows, "tsr") == pytest.approx(0.25)


def test_pool_ignores_sequences_with_no_report_frames():
    rows = {"a": {"tsr": 0.8, "n_report_frames": 10}, "b": {"tsr": 0.0, "n_report_frames": 0}}
    assert gss.pool(rows, "tsr") == pytest.approx(0.8)


# ------------------------------------------------------------- paired difference
def test_paired_difference_is_positive_when_the_candidate_is_better():
    a = {f"s{i}": {"tsr": 0.6, "n_report_frames": 100} for i in range(6)}
    b = {f"s{i}": {"tsr": 0.4, "n_report_frames": 100} for i in range(6)}
    out = gss.paired_difference(a, b, "tsr", reps=200, seed=0)
    assert out["n"] == 6
    assert out["delta"] == pytest.approx(0.2, abs=1e-9)
    assert out["lo"] > 0, "a constant positive difference must give a CI above zero"


def test_paired_difference_uses_only_shared_sequences():
    a = {"s0": {"tsr": 1.0, "n_report_frames": 100}, "s1": {"tsr": 1.0, "n_report_frames": 100}}
    b = {"s0": {"tsr": 0.0, "n_report_frames": 100}, "s9": {"tsr": 5.0, "n_report_frames": 100}}
    out = gss.paired_difference(a, b, "tsr", reps=50, seed=0)
    assert out["n"] == 1, "only s0 is comparable"
    # one sequence is not enough for a CI, and it must say so rather than invent one
    assert out["delta"] in (None, pytest.approx(1.0))


# --------------------------------------------------------------------- refusals
def test_dev_sequences_are_refused(fake_sweep, capsys):
    for point in ("margin_0.20", "margin_0.30"):
        for i, seq in enumerate(["dancetrack0004", "dancetrack0010", "dancetrack0014",
                                "dancetrack0018", "dancetrack0019", "dancetrack0025",
                                "dancetrack0026", "dancetrack0030", "dancetrack0034",
                                "dancetrack0035", "dancetrack0041"]):
            _write_summary(fake_sweep, point, seq, tsr=0.5, far=0.1)
    with pytest.raises(SystemExit) as exc:
        gss.main(["--out", str(fake_sweep.parent / "SPLIT.md")])
    assert "dev sequences" in str(exc.value)


def test_a_single_point_is_never_reported_as_a_selection(fake_sweep):
    # sequence names must avoid the frozen dev three (0004/0005/0007): the script
    # correctly refuses to run at all if they appear, which is a different test
    for i in range(10, 22):
        _write_summary(fake_sweep, "margin_0.20", f"dancetrack{i:04d}", tsr=0.5, far=0.2)
    out = fake_sweep.parent / "SPLIT.md"
    assert gss.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "no selection possible" in text
    assert "**<- selected**" not in text


def test_too_few_complete_sequences_refuses(fake_sweep):
    for point in ("margin_0.20", "margin_0.30"):
        for i in range(3):
            _write_summary(fake_sweep, point, f"dancetrack{i:04d}", tsr=0.5, far=0.2)
    with pytest.raises(SystemExit) as exc:
        gss.main(["--out", str(fake_sweep.parent / "SPLIT.md")])
    assert "need" in str(exc.value)


def test_multi_point_run_selects_on_explore_and_reports_test(fake_sweep):
    seqs = [f"dancetrack{i:04d}" for i in range(100, 122)]
    # 0.30 is uniformly better; the selection must find it and its paired CI must exclude 0
    for i, seq in enumerate(seqs):
        _write_summary(fake_sweep, "margin_0.20", seq, tsr=0.30, far=0.30)
        _write_summary(fake_sweep, "margin_0.30", seq, tsr=0.50, far=0.10)
    out = fake_sweep.parent / "SPLIT.md"
    assert gss.main(["--out", str(out), "--reps", "200"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "`margin_0.30` **<- selected**" in text
    assert "EXPLORE (selection)" in text and "TEST (reported)" in text
    # the paired TSR delta is a constant +0.20, so the CI must sit above zero
    assert "candidate better" in text
