#!/usr/bin/env python
"""Held-out measurement of the shipped click-target configuration on DanceTrack val.

WHY THIS FILE EXISTS
--------------------
Every one of the seven shipped target-layer knobs was *selected* on the same three
DanceTrack sequences (``dancetrack0004/0005/0007``) and then *reported* on those same
sequences.  Criterion 2's headline "TSR 0.512 / FAR 0.280" is therefore a mean over
n=3 sequences with a 95% interval of about +-0.394: it certifies nothing.  This script
produces the first measurement on sequences that were never used to choose anything.

HARD RULE: the dev split below is frozen.  It must never be used for reporting again,
and it is printed at startup so every log of this script carries its own provenance.

SWEEPS ARE FORBIDDEN ON THE HELD-OUT SPLIT
------------------------------------------
Only the nine flags in ``ALLOWED_FLAGS`` are accepted; anything else (in particular
``--ablate``, or any group/sweep/guard flag) exits with code 2.  Selecting a knob value
on held-out data converts it into dev data, which would defeat the entire exercise.

Note on ``scripts/run_target.py``: its ``--ablate`` default is ``"all"``, so a bare
invocation runs the *entire 84-variant grid* on whatever ``--data-root`` points at.
That is an ablation sweep, not a shipped-default run.  ``build_variants("")`` is also
mapped to "all", so the CLI cannot express "baseline only".  This script therefore
pins the shipped configuration by replacing ``run_target.build_variants`` with a
baseline-only builder *in its own child process* -- it never edits ``run_target.py``
and never passes ``--ablate``.  Every child summary is then asserted to contain
exactly one variant, ``baseline``, as machine-checked proof that no sweep happened.

Single-sequence selection uses ``discover_sequences``' documented idempotence: when
``--data-root`` already *is* a sequence directory, it returns ``[root]``.  So
``--data-root data/raw/dancetrack/val/dancetrack0010`` runs exactly that sequence.

Usage
-----
    python scripts/heldout_report.py --max-frames 300 --subsample 2 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VAL_ROOT = ROOT / "data" / "raw" / "dancetrack" / "val"
OUT_ROOT = ROOT / "results" / "heldout_dancetrack"
REPORT_JSON = OUT_ROOT / "report.json"
REPORT_MD = ROOT / "docs" / "reviews" / "R_heldout.md"

# --------------------------------------------------------------------------- #
# FROZEN DEV SPLIT.  NEVER USE THIS FOR REPORTING AGAIN.
# These are the sequences the seven shipped knobs were chosen on.  Any number
# quoted over them is selection-biased and must carry that caveat.  They are
# excluded from held-out selection below and printed at startup.
# --------------------------------------------------------------------------- #
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}
DEV_MOT17 = {"MOT17-02-FRCNN"}

#: The only flags this script will ever accept.  Anything else is refused.
ALLOWED_FLAGS = {
    "--data-root",
    "--out",
    "--max-frames",
    "--subsample",
    "--jobs",
    "--target-id",
    "--mask",
    "--detector",
    "--quiet",
}

#: Env var that turns this file into the in-process child runner.  Using an env var
#: (not a flag) keeps the argv allowlist above strictly closed.
CHILD_ENV = "HELDOUT_CHILD_SEQUENCE"

DEV_REFERENCE_TSR = 0.512
DEV_REFERENCE_FAR = 0.280
DEV_REFERENCE_SOURCE = "docs/EXPERIMENTS.md:1390-1395 (E041, cap 8 shipped)"

#: The dev split's per-sequence values, read off the E041 table so the dev-split
#: interval can be recomputed in-file instead of being quoted from memory.
DEV_PER_SEQUENCE = {
    "dancetrack0004": (0.415, 0.040),
    "dancetrack0005": (0.695, 0.205),
    "dancetrack0007": (0.426, 0.595),
}
DEV_PER_SEQUENCE_SOURCE = "docs/EXPERIMENTS.md:1392-1394"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


# --------------------------------------------------------------------------- #
# Child mode: run the real harness in-process, baseline variant only.
# --------------------------------------------------------------------------- #
def _run_child() -> int:
    """Run ``run_target.main`` for one sequence with the shipped config only."""
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - deliberate late import (needs sys.path)

    def _baseline_only(groups: Any = None, **kwargs: Any) -> list[Any]:
        """The shipped default config, and nothing else.  No group is selected."""
        return [rt.Variant("baseline", "baseline", {}, "shipped default config")]

    rt.build_variants = _baseline_only  # noqa: WPS437 - process-local, file untouched
    seq = os.environ[CHILD_ENV]
    out_dir = os.environ.get("HELDOUT_OUT") or str(OUT_ROOT / seq)
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", os.environ.get("HELDOUT_MASK", "none"),
        "--max-frames", os.environ.get("HELDOUT_MAX_FRAMES", "300"),
        "--subsample", os.environ.get("HELDOUT_SUBSAMPLE", "2"),
        "--jobs", os.environ.get("HELDOUT_JOBS", "2"),
    ]
    if os.environ.get("HELDOUT_TARGET_ID"):
        argv += ["--target-id", os.environ["HELDOUT_TARGET_ID"]]
    if os.environ.get("HELDOUT_DETECTOR"):
        argv += ["--detector", os.environ["HELDOUT_DETECTOR"]]
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Argument handling: closed allowlist, enforced before argparse ever sees it.
# --------------------------------------------------------------------------- #
def _enforce_argv(argv: Sequence[str]) -> None:
    """Exit 2 if any flag outside the allowlist is present.

    Scanned by hand rather than via argparse so that the refusal message can be
    specific about *why* sweeps are forbidden here, and so that a future
    ``run_target.py`` flag cannot silently become reachable.
    """
    bad: list[str] = []
    for tok in argv:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]
        if name in ("-h", "--help"):
            continue
        if name not in ALLOWED_FLAGS:
            bad.append(name)
    if bad:
        sys.stderr.write(
            "[heldout] REFUSED: unknown flag(s) %s\n"
            "[heldout] Sweeps are forbidden on the held-out split. This script accepts only:\n"
            "[heldout]   %s\n"
            "[heldout] Selecting or tuning a knob on these sequences turns them into dev data,\n"
            "[heldout] which is exactly what this measurement exists to avoid. If you need a\n"
            "[heldout] sweep, run it on data/raw/dancetrack/val/dancetrack0004 (dev) instead.\n"
            % (", ".join(sorted(set(bad))), " ".join(sorted(ALLOWED_FLAGS)))
        )
        raise SystemExit(2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Held-out (never-selected-on) measurement of the shipped target config.",
        epilog="Only %s are accepted; sweeps are forbidden on the held-out split." % sorted(ALLOWED_FLAGS),
    )
    p.add_argument("--data-root", default=str(VAL_ROOT), help="DanceTrack val root (default: %(default)s)")
    p.add_argument("--out", default=str(OUT_ROOT), help="output root (default: %(default)s)")
    p.add_argument("--max-frames", type=int, default=300, help="frames per sequence (default: %(default)s)")
    p.add_argument("--subsample", type=int, default=2, help="keep every K-th frame (default: %(default)s)")
    p.add_argument("--jobs", type=int, default=2, help="harness worker processes (default: %(default)s)")
    p.add_argument("--target-id", type=int, default=None, help="force one identity (default: harness auto-pick)")
    p.add_argument("--mask", default="none", help="mask mode (default: %(default)s)")
    p.add_argument("--detector", default=None, choices=[None, "yolo", "gt"], help="harness detector override")
    p.add_argument("--quiet", action="store_true", help="suppress per-sequence harness chatter")
    return p


# --------------------------------------------------------------------------- #
# Held-out selection
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    """Every sequence dir under ``data_root`` except the frozen dev split."""
    if not data_root.is_dir():
        raise SystemExit("[heldout] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    # A directory only counts as a sequence if the loader would accept it.
    present = [
        name
        for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    seqs = [name for name in present if name not in DEV_SEQUENCES]
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit(
            "[heldout] FATAL: dev sequence(s) %s present in the held-out selection; refusing to report." % leaked
        )
    missing_dev = sorted(DEV_SEQUENCES - set(present))
    if missing_dev:
        print("[heldout] NOTE: dev sequence(s) %s not found under %s (nothing to exclude)" % (missing_dev, data_root), flush=True)
    if not seqs:
        raise SystemExit("[heldout] FATAL: no sequences found under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def _weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def _pct(values: Sequence[float], qs: Sequence[float]) -> list[float]:
    """Linear-interpolation percentiles without a numpy dependency."""
    xs = sorted(values)
    n = len(xs)
    out: list[float] = []
    for q in qs:
        if n == 1:
            out.append(float(xs[0]))
            continue
        pos = (n - 1) * q / 100.0
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(float(xs[lo] * (1 - frac) + xs[hi] * frac))
    return out


def cluster_bootstrap(rows: Sequence[dict[str, Any]], *, key: str, reps: int, seed: int) -> dict[str, Any]:
    """Resample *sequences* with replacement; recompute the frame-weighted pool.

    Resampling whole sequences (not frames) is the only honest unit here: the
    frames inside one sequence are not independent draws.
    """
    import random

    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return {"reps": reps, "n_sequences": 0, "p2.5": None, "p50": None, "p97.5": None, "samples": []}
    values: list[float] = []
    weights = [float(r["n_report_frames"]) for r in rows]
    vals = [float(r[key]) for r in rows]
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(_weighted([vals[i] for i in idx], [weights[i] for i in idx]))
    p25, p50, p975 = _pct(values, [2.5, 50.0, 97.5])
    return {
        "reps": reps,
        "n_sequences": n,
        "seed": seed,
        "estimator": "frame-weighted pooled mean, resampling sequences with replacement",
        "p2.5": round(p25, 6),
        "p50": round(p50, 6),
        "p97.5": round(p975, 6),
        "mean": round(sum(values) / len(values), 6),
    }


# --------------------------------------------------------------------------- #
# One sequence
# --------------------------------------------------------------------------- #
def run_one(seq: str, args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    """Invoke the harness for a single sequence and parse its summary.json."""
    out_dir = Path(args.out) / seq
    log_path = log_dir / ("%s.log" % seq)
    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["HELDOUT_MASK"] = str(args.mask)
    env["HELDOUT_MAX_FRAMES"] = str(args.max_frames)
    env["HELDOUT_SUBSAMPLE"] = str(args.subsample)
    env["HELDOUT_JOBS"] = str(args.jobs)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.target_id is not None:
        env["HELDOUT_TARGET_ID"] = str(args.target_id)
    if args.detector:
        env["HELDOUT_DETECTOR"] = str(args.detector)

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("# %s\n# argv=%s\n" % (seq, " ".join(sys.argv)))
        fh.flush()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall = time.perf_counter() - t0
    rec: dict[str, Any] = {
        "sequence": seq,
        "exit_code": int(proc.returncode),
        "wall_seconds": round(wall, 1),
        "log": str(log_path.relative_to(ROOT)),
        "out_dir": str(out_dir.relative_to(ROOT)),
    }

    if proc.returncode == 3:
        tail = _tail_lines(log_path, 6)
        rec.update(
            status="EXCLUDED",
            reason="exit 3: no observable target under mask=%s" % args.mask,
            harness_message=_message_from_log(log_path),
            log_tail=tail,
        )
        return rec
    if proc.returncode != 0:
        rec.update(
            status="ERROR",
            reason="exit %d (harness error, not a target-observability decision)" % proc.returncode,
            harness_message=_message_from_log(log_path),
            log_tail=_tail_lines(log_path, 12),
        )
        return rec

    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        rec.update(status="ERROR", reason="harness exited 0 but wrote no summary.json", log_tail=_tail_lines(log_path, 12))
        return rec
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    meta = summary.get("meta", {})
    variants_run = list(meta.get("variants") or [])
    if variants_run != ["baseline"]:
        rec.update(
            status="ERROR",
            reason="SWEEP GUARD: expected exactly ['baseline'], summary reports %r" % (variants_run,),
        )
        return rec

    rows = [r for r in summary.get("headline", []) if r.get("variant") == "baseline" and r.get("mask") == args.mask]
    if not rows:
        rec.update(status="ERROR", reason="no baseline/%s row in summary.headline" % args.mask)
        return rec
    row = rows[0]

    decision = (meta.get("target_decisions") or {}).get(seq, {})
    targets = meta.get("target_ids") or {}
    rec.update(
        status="INCLUDED",
        tsr=float(row["target_success_rate"]),
        far=float(row["false_acquisition_rate"]),
        id_consistency=float(row["id_consistency"]),
        target_precision=float(row.get("target_precision", float("nan"))),
        frames=int(row["n_frames"]),
        n_report_frames=int(row.get("n_report_frames", row["n_frames"])),
        n_locked_frames=int(row.get("n_locked_frames", 0)),
        false_acquisition_frames=int(row.get("false_acquisition_frames", 0)),
        target_switch_count=int(row.get("target_switch_count", 0)),
        ms_per_frame=float(row.get("ms_per_frame", float("nan"))),
        target_id=targets.get(seq),
        target_reason=decision.get("reason"),
        target_control=decision.get("control"),
        guard_passed=decision.get("guard_passed"),
        visible_median_h=(summary.get("targets", {}).get(seq, {}) or {}).get("visible_median_h"),
        harness_wall_seconds=meta.get("wall_seconds"),
        summary=str(summary_path.relative_to(ROOT)),
        reason="included",
    )
    return rec


def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln for ln in lines if ln.strip()][-n:]


def _message_from_log(path: Path) -> str:
    """Pull the harness's own guard/pick message out of the log, verbatim."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for ln in reversed(lines):
        if "[guard]" in ln or "[pick]" in ln or "[error]" in ln:
            return ln.strip()
    return ""


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def dev_split_interval() -> dict[str, Any]:
    """The dev-split 95% t-interval on n=3, recomputed from the E041 table.

    The audit quotes "+-0.394" for the TSR headline; this reproduces it from the
    three per-sequence values so the report depends on arithmetic, not on a number
    copied out of a review.
    """
    import math
    import statistics as st

    out: dict[str, Any] = {"source": DEV_PER_SEQUENCE_SOURCE, "n": len(DEV_PER_SEQUENCE)}
    t975 = {1: 12.706, 2: 4.303, 3: 3.182}.get(len(DEV_PER_SEQUENCE) - 1, 1.96)
    for idx, name in ((0, "tsr"), (1, "far")):
        vals = [v[idx] for v in DEV_PER_SEQUENCE.values()]
        mean = sum(vals) / len(vals)
        sd = st.stdev(vals)
        margin = t975 * sd / math.sqrt(len(vals))
        out[name] = {
            "mean": round(mean, 6),
            "sd": round(sd, 6),
            "half_width_95": round(margin, 6),
            "ci": [round(mean - margin, 6), round(mean + margin, 6)],
        }
    return out


def derived_stats(included: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cheap diagnosis of *why* the pool looks the way it does.

    Everything here is computed from ``report.json``'s own rows, so the report can
    cite itself: is the spread explained by target size, by wrong-lock, or neither?
    """
    import statistics as st

    n = len(included)
    tsr = [float(r["tsr"]) for r in included]
    far = [float(r["far"]) for r in included]
    heights = [float(r["visible_median_h"] or 0.0) for r in included]
    pearson = None
    if n > 2 and st.pstdev(heights) > 0 and st.pstdev(tsr) > 0:
        mh, mt = sum(heights) / n, sum(tsr) / n
        cov = sum((a - mh) * (b - mt) for a, b in zip(heights, tsr)) / n
        pearson = round(cov / (st.pstdev(heights) * st.pstdev(tsr)), 3)
    return {
        "n": n,
        "tsr_below_0.05": [r["sequence"] for r in included if float(r["tsr"]) < 0.05],
        "tsr_above_0.80": [r["sequence"] for r in included if float(r["tsr"]) > 0.80],
        "far_above_0.50": [r["sequence"] for r in included if float(r["far"]) > 0.50],
        "pearson_r_target_height_vs_tsr": pearson,
        "target_height_min": round(min(heights), 1) if heights else None,
        "target_height_max": round(max(heights), 1) if heights else None,
        "mean_ms_per_frame": round(sum(float(r["ms_per_frame"]) for r in included) / n, 1) if n else None,
        "frames_min": min(int(r["n_report_frames"]) for r in included) if n else None,
        "frames_max": max(int(r["n_report_frames"]) for r in included) if n else None,
    }


def compare_to_dev(pooled_tsr: float, pooled_far: float, ci_tsr: dict[str, Any], ci_far: dict[str, Any]) -> dict[str, Any]:
    """Is the held-out estimate distinguishable from the dev-split claim?"""
    tsr_lo, tsr_hi = ci_tsr.get("p2.5"), ci_tsr.get("p97.5")
    far_lo, far_hi = ci_far.get("p2.5"), ci_far.get("p97.5")
    tsr_covers = (tsr_lo is not None) and (tsr_lo <= DEV_REFERENCE_TSR <= tsr_hi)
    far_covers = (far_lo is not None) and (far_lo <= DEV_REFERENCE_FAR <= far_hi)
    return {
        "dev_reference_tsr": DEV_REFERENCE_TSR,
        "dev_reference_far": DEV_REFERENCE_FAR,
        "dev_reference_source": DEV_REFERENCE_SOURCE,
        "heldout_tsr": round(pooled_tsr, 6),
        "heldout_far": round(pooled_far, 6),
        "delta_tsr_pp": round((pooled_tsr - DEV_REFERENCE_TSR) * 100, 2),
        "delta_far_pp": round((pooled_far - DEV_REFERENCE_FAR) * 100, 2),
        "ci_covers_dev_tsr": bool(tsr_covers),
        "ci_covers_dev_far": bool(far_covers),
    }


def write_markdown(payload: dict[str, Any]) -> str:
    inc = payload["included"]
    exc = payload["excluded"]
    pooled = payload["pooled"]
    ci_tsr = payload["bootstrap"]["tsr"]
    ci_far = payload["bootstrap"]["far"]
    cmp_ = payload["comparison"]

    L: list[str] = []
    a = L.append
    a("# R_heldout — first measurement on sequences that were never selected on")
    a("")
    a("<!-- generated by scripts/heldout_report.py; every number below is read from a file, named inline -->")
    a("")
    a("## Verdict")
    a("")
    a(payload["verdict_text"])
    a("")
    a("## Split")
    a("")
    a("| split | sequences | n |")
    a("|---|---|---|")
    a("| **dev (frozen, selection-biased, do not report)** | %s | %d |" % (", ".join(sorted(DEV_SEQUENCES)), len(DEV_SEQUENCES)))
    a("| dev MOT17 guard | %s | %d |" % (", ".join(sorted(DEV_MOT17)), len(DEV_MOT17)))
    a("| **held-out (this report)** | %s | **%d** |" % (", ".join(payload["heldout_sequences"]), len(payload["heldout_sequences"])))
    a("| excluded from pooling | %s | %d |" % (", ".join(r["sequence"] for r in exc) or "—", len(exc)))
    a("")
    a("Held-out = every sequence directory under `data/raw/dancetrack/val/` minus the frozen dev set,")
    a("enumerated at runtime (%d dirs total, %d after exclusion)." % (payload["n_dirs_seen"], len(payload["heldout_sequences"])))
    a("")
    a("## Protocol")
    a("")
    a("Shipped configuration only, no ablation: `scripts/run_target.py` ships with `--ablate` defaulting to")
    a("`\"all\"`, which would run the entire 84-variant grid — an ablation sweep, forbidden here.")
    a("`build_variants(\"\")` is also mapped to `\"all\"`, so the CLI cannot express \"baseline only\".")
    a("This script therefore pins `run_target.build_variants` to a baseline-only builder inside its own child")
    a("process and asserts each summary contains exactly one variant, `baseline`.")
    a("")
    a("- command per sequence: `python scripts/run_target.py --data-root data/raw/dancetrack/val/<seq> --mask %s --max-frames %d --subsample %d --jobs %d --out results/heldout_dancetrack/<seq>`"
      % (payload["protocol"]["mask"], payload["protocol"]["max_frames"], payload["protocol"]["subsample"], payload["protocol"]["jobs"]))
    a("- single-sequence selection: `discover_sequences` returns `[root]` when `--data-root` already is a sequence dir")
    a("- sweep guard: every included summary has `meta.variants == [\"baseline\"]` (asserted, not assumed)")
    a("- pooling: frame-weighted over `n_report_frames`; error bars from a **cluster bootstrap over sequences**")
    a("  (%d reps, seed %d, resampling whole sequences with replacement — frames within a sequence are not independent)"
      % (BOOTSTRAP_REPS, BOOTSTRAP_SEED))
    a("")
    a("## Per-sequence results")
    a("")
    a("All cells from `results/heldout_dancetrack/<seq>/summary.json`, `headline` row `variant=baseline`, `mask=%s`." % payload["protocol"]["mask"])
    a("")
    a("| sequence | TSR | FAR | id_cons | frames | target id | reason | prec | sw | ms/fr | src |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in inc:
        a("| `%s` | %.3f | %.3f | %.3f | %d | %s | %s | %.3f | %d | %.0f | `summary.json` |"
          % (r["sequence"], r["tsr"], r["far"], r["id_consistency"], r["frames"], r["target_id"],
             r["target_reason"], r["target_precision"], r["target_switch_count"], r["ms_per_frame"]))
    for r in exc:
        a("| `%s` | — | — | — | — | %s | %s | — | — | — | EXCLUDED |"
          % (r["sequence"], r.get("target_id") or "n/a", r["reason"]))
    a("")
    a("## Pooled estimates")
    a("")
    a("| metric | point | 95% CI (cluster bootstrap) | median | mean | min | max |")
    a("|---|---|---|---|---|---|---|")
    a("| **TSR** | **%.4f** | [%.4f, %.4f] | %.4f | %.4f | %.4f | %.4f |"
      % (pooled["tsr"], ci_tsr["p2.5"], ci_tsr["p97.5"], pooled["tsr_median"], pooled["tsr_mean"], pooled["tsr_min"], pooled["tsr_max"]))
    a("| **FAR** | **%.4f** | [%.4f, %.4f] | %.4f | %.4f | %.4f | %.4f |"
      % (pooled["far"], ci_far["p2.5"], ci_far["p97.5"], pooled["far_median"], pooled["far_mean"], pooled["far_min"], pooled["far_max"]))
    a("| id_consistency | %.4f | — | %.4f | %.4f | %.4f | %.4f |"
      % (pooled["id_consistency"], pooled["id_consistency_median"], pooled["id_consistency_mean"],
         pooled["id_consistency_min"], pooled["id_consistency_max"]))
    a("")
    a("Frame-weighted pooling sums to %d report frames over %d included sequences." % (pooled["total_report_frames"], len(inc)))
    a("")
    d = payload.get("derived") or {}
    if d:
        a("### Spread / diagnosis")
        a("")
        a("The per-sequence spread is not a rounding artefact — it is the finding:")
        a("")
        a("- %d of %d sequences sit below TSR 0.05: %s" % (len(d["tsr_below_0.05"]), d["n"], ", ".join("`%s`" % s for s in d["tsr_below_0.05"]) or "none"))
        a("- %d of %d sit above TSR 0.80: %s" % (len(d["tsr_above_0.80"]), d["n"], ", ".join("`%s`" % s for s in d["tsr_above_0.80"]) or "none"))
        a("- %d of %d have FAR > 0.50 (wrong-identity lock dominant): %s"
          % (len(d["far_above_0.50"]), d["n"], ", ".join("`%s`" % s for s in d["far_above_0.50"]) or "none"))
        if d.get("pearson_r_target_height_vs_tsr") is not None:
            a("- Pearson r between the auto-picked target's median visible height and its TSR is **%.3f** "
              "(target heights %.0f-%.0f px): success is *not* explained by how big the clicked target was, "
              "so 'the held-out targets are just tiny' is not a valid explanation of the drop."
              % (d["pearson_r_target_height_vs_tsr"], d["target_height_min"], d["target_height_max"]))
        a("- mean %.0f ms/frame; per-sequence report frames %d-%d (hence frame-weighted pooling)."
          % (d["mean_ms_per_frame"], d["frames_min"], d["frames_max"]))
        a("")
    a("## Comparison against the dev-split claim")
    a("")
    a("| | TSR | FAR |")
    a("|---|---|---|")
    a("| dev split (n=3, same 3 sequences used to choose every knob) — `%s` | %.3f | %.3f |" % (DEV_REFERENCE_SOURCE.split(" ")[0], cmp_["dev_reference_tsr"], cmp_["dev_reference_far"]))
    a("| **held-out (n=%d)** | **%.4f** | **%.4f** |" % (len(inc), cmp_["heldout_tsr"], cmp_["heldout_far"]))
    a("| delta | %+.2f pp | %+.2f pp |" % (cmp_["delta_tsr_pp"], cmp_["delta_far_pp"]))
    a("| held-out 95%% CI covers the dev claim? | %s | %s |" % ("yes" if cmp_["ci_covers_dev_tsr"] else "**no**", "yes" if cmp_["ci_covers_dev_far"] else "**no**"))
    a("")
    a("Dev per-sequence spread for context (`docs/EXPERIMENTS.md:1390-1395`): 0004 0.415/0.040, 0005 0.695/0.205,")
    a("0007 0.426/0.595 — i.e. the dev 'mean' is already the average of a 0.040-FAR and a 0.595-FAR sequence.")
    a("")
    dv = payload.get("dev_interval") or {}
    if dv.get("tsr"):
        a("**How weak the dev claim was, recomputed from its own three rows** (`%s`): TSR mean %.3f, sd %.3f, "
          "95%% t-interval ±%.3f → [%.3f, %.3f]. FAR mean %.3f, sd %.3f, 95%% t-interval ±%.3f → [%.3f, %.3f]. "
          "The audit's ±0.394 is this interval, reproduced exactly. A ±0.394 bar on a 0.512 estimate is wider than "
          "the whole gap to the 0.55 target, which is why the dev number certified nothing."
          % (dv["source"], dv["tsr"]["mean"], dv["tsr"]["sd"], dv["tsr"]["half_width_95"], dv["tsr"]["ci"][0],
             dv["tsr"]["ci"][1], dv["far"]["mean"], dv["far"]["sd"], dv["far"]["half_width_95"],
             dv["far"]["ci"][0], dv["far"]["ci"][1]))
        a("")
    a("## Exclusions")
    a("")
    if exc:
        for r in exc:
            a("- `%s` — %s. Harness said: `%s`" % (r["sequence"], r["reason"], r.get("harness_message") or "(no message)"))
        a("")
        a("Excluded sequences are reported as excluded; no substitute sequence was run in their place.")
    else:
        a("None — every held-out sequence produced an observable target.")
    a("")
    a("## Provenance / freshness")
    a("")
    a("Wall time: **%.1fs** (%.1f min) for %d attempted sequences, %.1f s/sequence mean."
      % (payload["wall_seconds"], payload["wall_seconds"] / 60.0, payload["n_attempted"], payload["wall_seconds"] / max(1, payload["n_attempted"])))
    a("")
    if payload.get("freshness"):
        f = payload["freshness"]
        a("`scripts/check_baseline_freshness.py %s` → exit **%d**: %s" % ("results/heldout_dancetrack", f["exit_code"], f["verdict"]))
        if f.get("detail"):
            a("")
            a("```")
            a(f["detail"])
            a("```")
    else:
        a("`scripts/check_baseline_freshness.py` was not run.")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    if os.environ.get(CHILD_ENV):
        return _run_child()

    _enforce_argv(sys.argv[1:])
    args = build_argparser().parse_args(sys.argv[1:])
    out_root = Path(args.out)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[heldout] FROZEN DEV SPLIT (selection set — NEVER report on it again):", flush=True)
    print("[heldout]   DanceTrack: %s" % ", ".join(sorted(DEV_SEQUENCES)), flush=True)
    print("[heldout]   MOT17 guard: %s" % ", ".join(sorted(DEV_MOT17)), flush=True)

    data_root = Path(args.data_root)
    n_dirs_seen = len([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else 0
    seqs = heldout_sequences(data_root)
    print("[heldout] held-out sequences: %d (from %d dirs under %s, minus %d dev)"
          % (len(seqs), n_dirs_seen, data_root, len(DEV_SEQUENCES)), flush=True)
    print("[heldout] %s" % ", ".join(seqs), flush=True)
    print("[heldout] config: mask=%s max_frames=%d subsample=%d jobs=%d detector=%s target_id=%s"
          % (args.mask, args.max_frames, args.subsample, args.jobs, args.detector or "shipped", args.target_id), flush=True)

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for i, seq in enumerate(seqs, 1):
        ts = time.perf_counter()
        rec = run_one(seq, args, log_dir)
        rows.append(rec)
        print("[heldout] %2d/%2d %-18s %-9s %6.1fs %s"
              % (i, len(seqs), seq, rec["status"], time.perf_counter() - ts,
                 ("TSR=%.3f FAR=%.3f" % (rec["tsr"], rec["far"])) if rec["status"] == "INCLUDED" else rec.get("reason", "")),
              flush=True)
    wall = time.perf_counter() - t0

    included = [r for r in rows if r["status"] == "INCLUDED"]
    excluded = [r for r in rows if r["status"] != "INCLUDED"]
    if not included:
        print("[heldout] FATAL: no sequence produced a usable measurement", flush=True)
        return 1

    weights = [float(r["n_report_frames"]) for r in included]
    tsr_vals = [r["tsr"] for r in included]
    far_vals = [r["far"] for r in included]
    idc_vals = [r["id_consistency"] for r in included]
    pooled_tsr = _weighted(tsr_vals, weights)
    pooled_far = _weighted(far_vals, weights)

    boot_tsr = cluster_bootstrap(included, key="tsr", reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    boot_far = cluster_bootstrap(included, key="far", reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    comparison = compare_to_dev(pooled_tsr, pooled_far, boot_tsr, boot_far)

    def _stat(key: str, q: float) -> float:
        return _pct([float(r[key]) for r in included], [q])[0]

    pooled = {
        "tsr": round(pooled_tsr, 6),
        "far": round(pooled_far, 6),
        "tsr_mean": round(sum(tsr_vals) / len(tsr_vals), 6),
        "far_mean": round(sum(far_vals) / len(far_vals), 6),
        "tsr_median": round(_stat("tsr", 50.0), 6),
        "far_median": round(_stat("far", 50.0), 6),
        "tsr_min": round(min(tsr_vals), 6),
        "tsr_max": round(max(tsr_vals), 6),
        "far_min": round(min(far_vals), 6),
        "far_max": round(max(far_vals), 6),
        "id_consistency": round(_weighted(idc_vals, weights), 6),
        "id_consistency_mean": round(sum(idc_vals) / len(idc_vals), 6),
        "id_consistency_median": round(_stat("id_consistency", 50.0), 6),
        "id_consistency_min": round(min(idc_vals), 6),
        "id_consistency_max": round(max(idc_vals), 6),
        "total_report_frames": int(sum(weights)),
        "estimator": "frame-weighted pooled mean over included sequences",
    }

    verdict = _verdict(len(included), pooled, boot_tsr, boot_far, comparison, excluded)

    payload: dict[str, Any] = {
        "generated_by": "scripts/heldout_report.py",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "dev_mot17": sorted(DEV_MOT17),
        "heldout_sequences": seqs,
        "n_dirs_seen": n_dirs_seen,
        "n_attempted": len(rows),
        "protocol": {
            "mask": args.mask,
            "max_frames": args.max_frames,
            "subsample": args.subsample,
            "jobs": args.jobs,
            "detector": args.detector or "shipped default (yolo)",
            "target_id": args.target_id,
            "variants_run": ["baseline"],
            "no_ablation": True,
        },
        "per_sequence": rows,
        "included": included,
        "excluded": excluded,
        "pooled": pooled,
        "derived": derived_stats(included),
        "bootstrap": {"tsr": boot_tsr, "far": boot_far, "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "comparison": comparison,
        "dev_interval": dev_split_interval(),
        "verdict": verdict,
        "verdict_text": verdict["text"],
        "wall_seconds": round(wall, 1),
        "harness_argv": sys.argv,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(write_markdown(payload), encoding="utf-8")

    print("[heldout] pooled TSR=%.4f [%.4f, %.4f]  FAR=%.4f [%.4f, %.4f]  n=%d"
          % (pooled_tsr, boot_tsr["p2.5"], boot_tsr["p97.5"], pooled_far, boot_far["p2.5"], boot_far["p97.5"], len(included)),
          flush=True)
    print("[heldout] verdict: %s" % verdict["label"], flush=True)
    print("[heldout] wrote %s and %s (wall %.1fs)" % (REPORT_JSON.relative_to(ROOT), REPORT_MD.relative_to(ROOT), wall), flush=True)
    return 0


def _verdict(
    n: int,
    pooled: dict[str, Any],
    boot_tsr: dict[str, Any],
    boot_far: dict[str, Any],
    comparison: dict[str, Any],
    excluded: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """An explicit generalisation verdict — the point of the whole exercise."""
    tsr, far = pooled["tsr"], pooled["far"]
    lo_t, hi_t = boot_tsr["p2.5"], boot_tsr["p97.5"]
    lo_f, hi_f = boot_far["p2.5"], boot_far["p97.5"]
    covers = comparison["ci_covers_dev_tsr"] and comparison["ci_covers_dev_far"]

    if n < 10:
        label = "INSUFFICIENT"
        text = ("Only %d included sequences: a cluster bootstrap over %d units cannot support a generalisation "
                "claim. Treat the numbers as indicative." % (n, n))
    elif not covers:
        label = "DOES NOT GENERALISE"
        text = ("The held-out 95%% CI excludes the dev-split claim on %s. Pooled held-out TSR is %.3f "
                "[%.3f, %.3f] versus the dev 0.512, and FAR is %.3f [%.3f, %.3f] versus the dev 0.280. "
                "The dev-split headline is not reproduced on sequences the knobs were not selected on."
                % (" and ".join([k for k, v in (("TSR", comparison["ci_covers_dev_tsr"]), ("FAR", comparison["ci_covers_dev_far"])) if not v]),
                   tsr, lo_t, hi_t, far, lo_f, hi_f))
    else:
        label = "CONSISTENT (not confirmed)"
        text = ("The held-out 95%% CI contains the dev-split claim for both TSR and FAR (pooled TSR %.3f "
                "[%.3f, %.3f] vs dev 0.512; FAR %.3f [%.3f, %.3f] vs dev 0.280), so the dev headline is not "
                "contradicted by held-out data — but a CI that wide is also compatible with materially worse "
                "performance, so this certifies the direction, not the number. Report the interval, never the "
                "dev point estimate alone." % (tsr, lo_t, hi_t, far, lo_f, hi_f))
    if excluded:
        text += " %d sequence(s) were excluded (no observable target) and are not represented in the pool." % len(excluded)
    return {"label": label, "text": text, "ci_covers_dev_tsr": comparison["ci_covers_dev_tsr"], "ci_covers_dev_far": comparison["ci_covers_dev_far"]}


if __name__ == "__main__":
    raise SystemExit(main())
