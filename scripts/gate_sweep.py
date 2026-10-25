#!/usr/bin/env python
"""E070 -- gate calibration sweep on the HELD-OUT DanceTrack split.

WHY THIS FILE EXISTS
--------------------
``docs/NEXT_ROUND.md`` diagnosed the remaining gap as *gate calibration*: the same
acceptance gate that prevents a wrong-identity lock (E022 ``cost.search_margin=0.20``,
E029 adaptive band) is refusing correct re-acquisitions on held-out data (TSR 0.2102 /
FAR 0.2364 over 22 sequences, E068).  The one experiment it specifies is a sweep of the
search-side gate on the 22 held-out sequences only, with a pre-registered acceptance
rule.

``scripts/heldout_report.py`` deliberately refuses every sweep flag, so this file is the
narrow, explicit exception the spec allows: it drives ``run_target.main`` in a child
process and pins ``run_target.build_variants`` to a **single** variant carrying exactly
one override dict.  ``run_target.py`` is never edited, and ``--ablate`` never selects a
grid: every child run is machine-checked to contain exactly one variant with exactly the
intended overrides (``specs.json`` written by the wrapper around ``run_target.run_grid``
plus an assertion on ``meta.variants``).

HARD RULE: the frozen dev split (dancetrack0004/0005/0007) is printed at startup and any
run that would include it aborts.  A knob value selected on held-out data turns it into
dev data.

PRE-REGISTERED ACCEPTANCE RULE (verbatim from ``docs/NEXT_ROUND.md``)
---------------------------------------------------------------------
    ship a value only if pooled TSR rises by >= +5 pp **and** the upper bound of its FAR
    CI does not exceed the shipped FAR's upper bound.  Anything else is recorded as a
    negative and the default is left alone.

Usage
-----
    # the whole primary axis on all 22 held-out sequences
    python scripts/gate_sweep.py --points primary

    # one point / one sequence (smoke)
    python scripts/gate_sweep.py --points margin_0.20 --limit-seqs 1

    # partial fallback: 10 fixed random sequences, margin only
    python scripts/gate_sweep.py --points margin_0.00,margin_0.10,margin_0.20 --subset 10
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VAL_ROOT = ROOT / "data" / "raw" / "dancetrack" / "val"
OUT_ROOT = ROOT / "results" / "gate_sweep"
REPORT_JSON = OUT_ROOT / "report.json"
REPORT_MD = OUT_ROOT / "REPORT.md"

#: FROZEN selection split.  Never sweep or report on these again.
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}
DEV_MOT17 = {"MOT17-02-FRCNN"}

#: Shipped values under test (src/tgtrack/config.py, CostCfg).
SHIPPED_MARGIN = 0.20
SHIPPED_BAND_K = 4.0

PRIMARY_MARGINS = (0.0, 0.10, 0.15, 0.20, 0.30)
SECONDARY_BAND_KS = (3.0, 4.0)

#: Protocol fixed by NEXT_ROUND.md / the task statement.
PROTOCOL = {
    "max_frames": 300,
    "subsample": 2,
    "mask": "none",
    "jobs": 1,
    "detector": "yolo (shipped default)",
    "target_strategy": "stressed (shipped default)",
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0
TSR_ACCEPT_PP = 5.0

CHILD_ENV = "GATESWEEP_CHILD_SEQUENCE"


# --------------------------------------------------------------------------- #
# Axis points
# --------------------------------------------------------------------------- #
def point_spec(point_id: str) -> dict[str, Any]:
    """Map a point id to (overrides, human label).  Pure function, no state."""
    if point_id.startswith("margin_"):
        m = float(point_id.split("_", 1)[1])
        return {
            "point": point_id,
            "axis": "cost.search_margin",
            "value": m,
            "shipped": abs(m - SHIPPED_MARGIN) < 1e-9,
            "overrides": {"cost.search_margin": m},
            "variant": "sm_%.2f" % m,
            "label": "search_margin=%.2f%s" % (m, " (shipped)" if abs(m - SHIPPED_MARGIN) < 1e-9 else ""),
        }
    if point_id.startswith("band_k"):
        k = float(point_id.split("_k", 1)[1])
        return {
            "point": point_id,
            "axis": "cost.anchor_adaptive_k",
            "value": k,
            "shipped": abs(k - SHIPPED_BAND_K) < 1e-9,
            "overrides": {"cost.anchor_adaptive_k": k},
            "variant": "band_k%.1f" % k,
            "label": "anchor_adaptive_k=%.1f%s" % (k, " (shipped)" if abs(k - SHIPPED_BAND_K) < 1e-9 else ""),
        }
    raise SystemExit("[gate_sweep] unknown point id %r" % point_id)


def all_points(which: str) -> list[str]:
    if which == "primary":
        return ["margin_%.2f" % m for m in PRIMARY_MARGINS]
    if which == "secondary":
        return ["band_k%.1f" % k for k in SECONDARY_BAND_KS]
    ids = [p.strip() for p in which.split(",") if p.strip()]
    for p in ids:
        point_spec(p)  # validate
    return ids


# --------------------------------------------------------------------------- #
# Child mode: one sequence, one pinned variant, no grid.
# --------------------------------------------------------------------------- #
def _run_child() -> int:
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - late import needs sys.path

    seq = os.environ[CHILD_ENV]
    spec = point_spec(os.environ["GATESWEEP_POINT"])
    overrides = json.loads(os.environ.get("GATESWEEP_OVERRIDES") or "{}")
    variant_name = os.environ["GATESWEEP_VARIANT"]
    if overrides != spec["overrides"]:
        raise SystemExit("[gate_sweep] child override mismatch: %r vs %r" % (overrides, spec["overrides"]))

    def _pinned(groups: Any = None, **kwargs: Any) -> list[Any]:
        """Exactly one variant: the shipped config plus this point's overrides."""
        return [rt.Variant(variant_name, "gate_sweep", dict(overrides), "E070 sweep point " + spec["point"])]

    rt.build_variants = _pinned  # noqa: WPS437 - process-local, file untouched

    specs_path = Path(os.environ["GATESWEEP_SPECS"])
    _orig_run_grid = rt.run_grid

    def _checked_run_grid(specs: Any, *, jobs: int = 1, quiet: bool = False) -> Any:
        rows = [
            {
                "seq_name": s["seq_name"],
                "variant": s["variant"],
                "overrides": dict(s["overrides"]),
                "mask": s["mask"],
                "target_id": s["target_id"],
            }
            for s in specs
        ]
        specs_path.parent.mkdir(parents=True, exist_ok=True)
        specs_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        # A sweep would put >1 spec here.  One sequence/MASK/point == one row.
        if len(rows) != 1:
            raise SystemExit("[gate_sweep] GUARD: expected exactly 1 spec, got %d" % len(rows))
        if rows[0]["overrides"] != overrides or rows[0]["variant"] != variant_name:
            raise SystemExit("[gate_sweep] GUARD: spec carries %r / %r, intended %r / %r"
                             % (rows[0]["overrides"], rows[0]["variant"], overrides, variant_name))
        return _orig_run_grid(specs, jobs=jobs, quiet=quiet)

    rt.run_grid = _checked_run_grid

    out_dir = os.environ["GATESWEEP_OUT"]
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", PROTOCOL["mask"],
        "--max-frames", str(PROTOCOL["max_frames"]),
        "--subsample", str(PROTOCOL["subsample"]),
        "--jobs", str(PROTOCOL["jobs"]),
        # Recorded, never used for selection: build_variants is pinned above.
        "--ablate", "gate_sweep",
    ]
    if os.environ.get("GATESWEEP_TARGET_ID"):
        argv += ["--target-id", os.environ["GATESWEEP_TARGET_ID"]]
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise SystemExit("[gate_sweep] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    present = [
        name for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    seqs = [name for name in present if name not in DEV_SEQUENCES]
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[gate_sweep] FATAL: dev sequence(s) %s in held-out selection" % leaked)
    if not seqs:
        raise SystemExit("[gate_sweep] FATAL: no sequences under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats (same estimator as scripts/heldout_report.py, for comparability)
# --------------------------------------------------------------------------- #
def _weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return float("nan")
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def _pct(values: Sequence[float], qs: Sequence[float]) -> list[float]:
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


def cluster_bootstrap(rows: Sequence[dict[str, Any]], *, key: str, reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Resample whole *sequences* with replacement; recompute the frame-weighted pool."""
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return {"reps": reps, "n_sequences": 0, "p2.5": None, "p50": None, "p97.5": None}
    vals = [float(r[key]) for r in rows]
    weights = [float(r["n_report_frames"]) for r in rows]
    samples: list[float] = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(_weighted([vals[i] for i in idx], [weights[i] for i in idx]))
    p25, p50, p975 = _pct(samples, [2.5, 50.0, 97.5])
    return {
        "reps": reps,
        "n_sequences": n,
        "seed": seed,
        "estimator": "frame-weighted pooled mean, resampling sequences with replacement",
        "p2.5": round(p25, 6),
        "p50": round(p50, 6),
        "p97.5": round(p975, 6),
        "mean": round(sum(samples) / len(samples), 6),
    }


# --------------------------------------------------------------------------- #
# Running one (point, sequence)
# --------------------------------------------------------------------------- #
def run_pair(point_id: str, seq: str, *, target_id: int | None, force: bool, log_dir: Path) -> dict[str, Any]:
    spec = point_spec(point_id)
    out_dir = OUT_ROOT / "runs" / point_id / seq
    summary_path = out_dir / "summary.json"
    rec_path = OUT_ROOT / "summaries" / point_id / ("%s.json" % seq)
    log_path = log_dir / ("%s__%s.log" % (point_id, seq))
    specs_path = out_dir / "specs.json"

    base: dict[str, Any] = {
        "point": point_id,
        "axis": spec["axis"],
        "value": spec["value"],
        "shipped": spec["shipped"],
        "sequence": seq,
        "target_id_forced": target_id,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
    }

    owner_path = out_dir / ".mine"

    if summary_path.is_file() and not force and owner_path.is_file():
        cached = _parse_summary(summary_path, seq, target_id, specs_path)
        # A stale summary from a crashed/errored attempt must never be cached as a result.
        if cached.get("status") == "INCLUDED":
            cached.update(base)
            cached["from_cache"] = True
            return cached
    elif summary_path.is_file() and not force:
        # PROVENANCE GUARD.  2026-10-26: a second writer (a duplicate sweep started elsewhere in
        # this workspace) filled some runs/<point>/<seq>/ dirs.  A run dir without this sweep's
        # ``.mine`` stamp cannot be attributed to this protocol, so it is discarded and re-run
        # rather than reported -- a number whose producer is unknown is not a number.
        print(
            "[gate_sweep] PURGE foreign run dir (no .mine provenance): %s/%s" % (point_id, seq),
            flush=True,
        )

    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["GATESWEEP_POINT"] = point_id
    env["GATESWEEP_VARIANT"] = spec["variant"]
    env["GATESWEEP_OVERRIDES"] = json.dumps(spec["overrides"])
    env["GATESWEEP_OUT"] = str(out_dir)
    env["GATESWEEP_SPECS"] = str(specs_path)
    env["PYTHONIOENCODING"] = "utf-8"
    if target_id is not None:
        env["GATESWEEP_TARGET_ID"] = str(int(target_id))
    else:
        env.pop("GATESWEEP_TARGET_ID", None)

    log_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    rec: dict[str, Any] = {}
    for attempt in (1, 2):
        # Never let output from a failed attempt be parsed as this attempt's result.
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with log_path.open("a" if attempt > 1 else "w", encoding="utf-8") as fh:
            fh.write("# point=%s seq=%s attempt=%d\n# env=%s\n" % (
                point_id, seq, attempt,
                json.dumps({k: v for k, v in env.items() if k.startswith("GATESWEEP_")}, sort_keys=True)))
            fh.flush()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True,
            )
        wall = time.perf_counter() - t0
        code = int(proc.returncode)

        if code == 3:
            rec = {
                "status": "EXCLUDED",
                "reason": "exit 3: no observable target under mask=%s" % PROTOCOL["mask"],
                "harness_message": _message_from_log(log_path),
            }
        elif code == 0:
            rec = _parse_summary(summary_path, seq, target_id, specs_path)
        else:
            rec = {"status": "ERROR", "reason": "child exit %d" % code}
        rec["exit_code"] = code
        rec["wall_seconds"] = round(wall, 1)
        attempts.append({"attempt": attempt, "exit_code": code, "status": rec.get("status"),
                         "wall_seconds": round(wall, 1)})
        if code == 3 or rec.get("status") == "INCLUDED":
            break
        # One retry only, for infrastructure flakiness (a transient CUDA OOM from a co-tenant on
        # the shared GPU).  Every attempt is kept in `attempts`, so a retry cannot launder a real
        # failure into a success: a deterministic error fails twice.

    rec.update(base)
    rec["attempts"] = attempts
    rec["from_cache"] = False
    if rec.get("status") != "INCLUDED":
        rec["log_tail"] = _tail_lines(log_path, 8)

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    if rec.get("status") == "INCLUDED":
        owner_path.write_text(json.dumps({
            "point": point_id,
            "sequence": seq,
            "variant": spec["variant"],
            "overrides": spec["overrides"],
            "pid": os.getpid(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": rec.get("wall_seconds"),
            "attempt": attempts[-1]["attempt"] if attempts else None,
        }, indent=2) + "\n", encoding="utf-8")
    return rec


def _parse_summary(summary_path: Path, seq: str, forced_id: int | None, specs_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {"status": "ERROR", "reason": "harness exited 0 but wrote no summary.json"}
    # A malformed summary must never abort the whole sweep: it costs one pair, not 110.
    # Observed once on this machine ("Extra data: line 283 column 2"): the file held a
    # complete JSON document followed by a stray "}", i.e. two writers racing on the same
    # path. We try the strict parse, then salvage the first document, then give up -- the
    # caller retries the pair once before recording it as an error.
    text = summary_path.read_text(encoding="utf-8")
    recovered = False
    try:
        summary = json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            summary, end = json.JSONDecoder().raw_decode(text)
            recovered = True
            print(
                "[gate_sweep] WARNING %s: salvaged summary.json, ignored %d trailing chars "
                "(%s)" % (seq, len(text) - end, exc),
                flush=True,
            )
        except json.JSONDecodeError:
            return {
                "status": "ERROR",
                "reason": "corrupt summary.json (unparseable: %s)" % exc,
            }
    meta = summary.get("meta", {})


    variants = list(meta.get("variants") or [])
    if len(variants) != 1:
        return {"status": "ERROR", "reason": "GUARD: meta.variants=%r (expected exactly one point)" % (variants,)}

    spec_rows = json.loads(specs_path.read_text(encoding="utf-8")) if specs_path.is_file() else []
    if len(spec_rows) != 1:
        return {"status": "ERROR", "reason": "GUARD: specs.json rows=%d (expected 1)" % len(spec_rows)}

    # Protocol guards: the child must have run the protocol NEXT_ROUND.md fixes.
    problems = []
    if meta.get("max_frames") != PROTOCOL["max_frames"]:
        problems.append("max_frames=%r" % meta.get("max_frames"))
    if meta.get("subsample") != PROTOCOL["subsample"]:
        problems.append("subsample=%r" % meta.get("subsample"))
    if meta.get("jobs") != PROTOCOL["jobs"]:
        problems.append("jobs=%r" % meta.get("jobs"))
    if list(meta.get("masks") or []) != [PROTOCOL["mask"]]:
        problems.append("masks=%r" % meta.get("masks"))
    if problems:
        return {"status": "ERROR", "reason": "PROTOCOL GUARD failed: " + ", ".join(problems)}

    rows = [r for r in summary.get("headline", []) if r.get("mask") == PROTOCOL["mask"]]
    if len(rows) != 1:
        return {"status": "ERROR", "reason": "expected 1 headline row for mask=%s, got %d" % (PROTOCOL["mask"], len(rows))}
    row = rows[0]

    targets = meta.get("target_ids") or {}
    target_id = targets.get(seq)
    if forced_id is not None and target_id != int(forced_id):
        return {"status": "ERROR", "reason": "TARGET GUARD: meta.target_ids[%s]=%r, forced %r" % (seq, target_id, forced_id)}
    if target_id is None:
        return {"status": "ERROR", "reason": "TARGET GUARD: no meta.target_ids entry for %s" % seq}

    decision = (meta.get("target_decisions") or {}).get(seq, {})
    return {
        "status": "INCLUDED",
        "reason": "included",
        "json_recovered": recovered,
        "variant": variants[0],
        "overrides": dict(spec_rows[0].get("overrides") or {}),
        "tsr": float(row["target_success_rate"]),
        "far": float(row["false_acquisition_rate"]),
        "id_consistency": float(row["id_consistency"]),
        "n_report_frames": int(row.get("n_report_frames", row["n_frames"])),
        "n_frames": int(row["n_frames"]),
        "n_locked_frames": int(row.get("n_locked_frames", 0)),
        "false_acquisition_frames": int(row.get("false_acquisition_frames", 0)),
        "target_switch_count": int(row.get("target_switch_count", 0)),
        "ms_per_frame": float(row.get("ms_per_frame", float("nan"))),
        "target_id": target_id,
        "target_reason": decision.get("reason"),
        "guard_passed": decision.get("guard_passed"),
        "harness_wall_seconds": meta.get("wall_seconds"),
    }


def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln for ln in lines if ln.strip()][-n:]


def _message_from_log(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for ln in reversed(lines):
        if "[guard]" in ln or "[pick]" in ln or "[error]" in ln:
            return ln.strip()
    return ""


# --------------------------------------------------------------------------- #
# Pooling + acceptance rule
# --------------------------------------------------------------------------- #
def pool_point(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inc = [r for r in rows if r.get("status") == "INCLUDED"]
    exc = [r for r in rows if r.get("status") != "INCLUDED"]
    if not inc:
        return {"included": [], "excluded": exc, "pooled": None, "bootstrap": None}
    weights = [float(r["n_report_frames"]) for r in inc]
    tsr_vals = [float(r["tsr"]) for r in inc]
    far_vals = [float(r["far"]) for r in inc]
    boot_tsr = cluster_bootstrap(inc, key="tsr")
    boot_far = cluster_bootstrap(inc, key="far")
    pooled = {
        "tsr": round(_weighted(tsr_vals, weights), 6),
        "far": round(_weighted(far_vals, weights), 6),
        "tsr_mean": round(sum(tsr_vals) / len(inc), 6),
        "far_mean": round(sum(far_vals) / len(inc), 6),
        "tsr_median": round(_pct(tsr_vals, [50.0])[0], 6),
        "far_median": round(_pct(far_vals, [50.0])[0], 6),
        "tsr_min": round(min(tsr_vals), 6),
        "tsr_max": round(max(tsr_vals), 6),
        "far_min": round(min(far_vals), 6),
        "far_max": round(max(far_vals), 6),
        "total_report_frames": int(sum(weights)),
        "n_included": len(inc),
        "n_excluded": len(exc),
        "estimator": "frame-weighted pooled mean over included sequences",
    }
    return {"included": inc, "excluded": exc, "pooled": pooled,
            "bootstrap": {"tsr": boot_tsr, "far": boot_far, "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED}}


def acceptance(point_pool: dict[str, Any], shipped_pool: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered rule, applied verbatim and never re-tuned."""
    out: dict[str, Any] = {
        "point": spec["point"],
        "axis": spec["axis"],
        "value": spec["value"],
        "is_shipped": bool(spec["shipped"]),
        "rule": ("SHIP iff (pooled TSR rises >= +%.1f pp vs shipped) AND "
                 "(candidate FAR 95%% CI upper bound <= shipped FAR 95%% CI upper bound)" % TSR_ACCEPT_PP),
    }
    if spec["shipped"]:
        out.update(verdict="REFERENCE", tsr_rise_pp=None, far_ci_upper_ok=None,
                   text="Shipped value: this is the reference point, not a candidate.")
        return out
    if not point_pool.get("pooled") or not shipped_pool.get("pooled"):
        out.update(verdict="NO-SHIP", tsr_rise_pp=None, far_ci_upper_ok=None,
                   text="No complete pool (excluded/error runs); cannot satisfy the rule -> default stays.")
        return out

    tsr_rise = point_pool["pooled"]["tsr"] - shipped_pool["pooled"]["tsr"]
    far_hi = point_pool["bootstrap"]["far"]["p97.5"]
    shipped_far_hi = shipped_pool["bootstrap"]["far"]["p97.5"]
    ok_tsr = tsr_rise * 100.0 >= TSR_ACCEPT_PP
    ok_far = far_hi <= shipped_far_hi
    ship = bool(ok_tsr and ok_far)
    out.update(
        tsr_rise_pp=round(tsr_rise * 100.0, 2),
        tsr_rule_met=bool(ok_tsr),
        far_ci_upper=far_hi,
        shipped_far_ci_upper=shipped_far_hi,
        far_ci_upper_ok=bool(ok_far),
        far_ci_upper_delta=round(far_hi - shipped_far_hi, 6),
        verdict="SHIP" if ship else "NO-SHIP",
        text=("TSR %+.2f pp (needs >= +%.1f) and candidate FAR CI upper %.4f vs shipped %.4f."
              % (tsr_rise * 100.0, TSR_ACCEPT_PP, far_hi, shipped_far_hi)),
    )
    return out


def per_sequence_deltas(point_rows: Sequence[dict[str, Any]], shipped_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shipped = {r["sequence"]: r for r in shipped_rows if r.get("status") == "INCLUDED"}
    moved_tsr: list[dict[str, Any]] = []
    moved_far: list[dict[str, Any]] = []
    unpaired: list[str] = []
    for r in point_rows:
        seq = r["sequence"]
        if r.get("status") != "INCLUDED" or seq not in shipped:
            unpaired.append(seq)
            continue
        d_tsr = float(r["tsr"]) - float(shipped[seq]["tsr"])
        d_far = float(r["far"]) - float(shipped[seq]["far"])
        if abs(d_tsr) > 0.05:
            moved_tsr.append({"sequence": seq, "delta_tsr_pp": round(d_tsr * 100, 2),
                              "tsr": round(float(r["tsr"]), 6), "baseline_tsr": round(float(shipped[seq]["tsr"]), 6)})
        if abs(d_far) > 0.05:
            moved_far.append({"sequence": seq, "delta_far_pp": round(d_far * 100, 2)})
    return {
        "n_paired": sum(1 for r in point_rows if r.get("status") == "INCLUDED" and r["sequence"] in shipped),
        "unpaired_sequences": unpaired,
        "tsr_moved_gt_5pp": sorted(moved_tsr, key=lambda d: -abs(d["delta_tsr_pp"])),
        "n_tsr_moved_gt_5pp": len(moved_tsr),
        "far_moved_gt_5pp": sorted(moved_far, key=lambda d: -abs(d["delta_far_pp"])),
        "n_far_moved_gt_5pp": len(moved_far),
    }


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def write_markdown(payload: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a("# E070 -- gate calibration sweep on the 22 held-out DanceTrack sequences")
    a("")
    a("<!-- generated by scripts/gate_sweep.py; every number is read from results/gate_sweep/report.json -->")
    a("")
    a("## Verdict")
    a("")
    prim = payload["points"]
    for p in payload["order"]:
        if p in prim:
            v = prim[p]["acceptance"]["verdict"]
            a("- **%s: %s** -- %s" % (prim[p]["label"], v, prim[p]["acceptance"]["text"]))
    a("")
    a("## Split (printed at startup, asserted dev-free)")
    a("")
    a("- frozen dev (never swept, never reported): %s" % ", ".join(sorted(DEV_SEQUENCES)))
    a("- held-out: **%d** sequences -- %s" % (len(payload["heldout_sequences"]), ", ".join(payload["heldout_sequences"])))
    a("")
    a("## Protocol")
    a("")
    a("- 300 frames, subsample 2, `--mask none`, `--jobs 1`, shipped defaults elsewhere")
    a("- one pinned variant per run (`run_target.build_variants` replaced in the child); `specs.json` guard "
      "asserts exactly 1 spec with exactly the intended overrides, and each `summary.json` reports exactly 1 variant")
    a("- the shipped reference point is re-run in this sweep (not quoted from E068), so every comparison is paired")
    a("- target identity: the shipped-margin pass picks the target; every other point is pinned to that same "
      "`--target-id` and each run's own `meta.target_ids` is asserted against it")
    a("- pooling: frame-weighted over `n_report_frames`; 95%% CI = cluster bootstrap over sequences, %d reps, seed %d"
      % (BOOTSTRAP_REPS, BOOTSTRAP_SEED))
    a("")
    a("## Pooled results per axis point")
    a("")
    a("| axis | point | n incl | TSR | TSR 95% CI | FAR | FAR 95% CI | TSR vs shipped | FAR CI-hi vs shipped | sequences TSR>5pp | verdict |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for pid in payload["order"]:
        p = prim[pid]
        pooled = p["pooled"]
        if not pooled:
            a("| %s | %s | 0 | -- | -- | -- | -- | -- | -- | -- | %s |" % (p["axis"], pid, p["acceptance"]["verdict"]))
            continue
        b = p["bootstrap"]
        d = p["deltas"]
        acc = p["acceptance"]
        rose = "--" if acc.get("tsr_rise_pp") is None else "%+.2f pp" % acc["tsr_rise_pp"]
        farok = "--" if acc.get("far_ci_upper_ok") is None else ("yes" if acc["far_ci_upper_ok"] else "NO")
        a("| `%s` | %.2f | %d | %.4f | [%.4f, %.4f] | %.4f | [%.4f, %.4f] | %s | %s | %d | **%s** |"
          % (p["axis"], p["value"], pooled["n_included"], pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
             pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"], rose, farok, d["n_tsr_moved_gt_5pp"],
             acc["verdict"]))
    a("")
    a("## Sequences whose TSR moves by more than 5 pp")
    a("")
    any_moved = False
    for pid in payload["order"]:
        d = prim[pid]["deltas"]
        if d["n_tsr_moved_gt_5pp"] == 0:
            continue
        any_moved = True
        a("**%s** (%d/%d paired sequences)" % (pid, d["n_tsr_moved_gt_5pp"], d["n_paired"]))
        a("")
        for row in d["tsr_moved_gt_5pp"]:
            a("- `%s`: %+.2f pp (%.3f -> %.3f)" % (row["sequence"], row["delta_tsr_pp"], row["baseline_tsr"], row["tsr"]))
        a("")
    if not any_moved:
        a("None: no sequence moves its TSR by more than 5 pp at any measured point.")
        a("")
    a("## Per-sequence table")
    a("")
    a("| sequence | " + " | ".join("%s TSR/FAR" % pid for pid in payload["order"]) + " |")
    a("|---" * (len(payload["order"]) + 1) + "|")
    for seq in payload["heldout_sequences"]:
        cells = []
        for pid in payload["order"]:
            row = next((r for r in prim[pid]["rows"] if r["sequence"] == seq), None)
            if row and row.get("status") == "INCLUDED":
                cells.append("%.3f / %.3f" % (row["tsr"], row["far"]))
            else:
                cells.append("EXCL" if row else "--")
        a("| `%s` | %s |" % (seq, " | ".join(cells)))
    a("")
    a("## Exclusions")
    a("")
    n_exc = sum(len(prim[pid]["excluded"]) for pid in payload["order"])
    if n_exc:
        for pid in payload["order"]:
            for r in prim[pid]["excluded"]:
                a("- `%s` @ %s: %s -- %s" % (r["sequence"], pid, r["reason"], r.get("harness_message") or ""))
    else:
        a("None: every (sequence, point) pair produced an observable target; denominators are %d in every pool."
          % len(payload["heldout_sequences"]))
    a("")
    a("## Wall time and commands")
    a("")
    a("- total wall: **%.1f s** (%.1f min) for %d (sequence, point) pairs"
      % (payload["wall_seconds"], payload["wall_seconds"] / 60.0, payload["n_pairs_run"]))
    a("- exact parent command: `python scripts/gate_sweep.py --points %s%s%s`"
      % (payload["cli"]["points"], " --subset %d" % payload["cli"]["subset"] if payload["cli"]["subset"] else "",
         " --force" if payload["cli"]["force"] else ""))
    a("- per-pair child command (logged verbatim in `results/gate_sweep/logs/`): "
      "`python scripts/gate_sweep.py` with `GATESWEEP_CHILD_SEQUENCE=<seq> GATESWEEP_POINT=<point> "
      "GATESWEEP_TARGET_ID=<id>`")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process on this machine (Windows-safe, no psutil)."""
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return ("%d" % pid) in (out.stdout or "")


def acquire_single_instance_lock() -> bool:
    """Take ``results/gate_sweep/.sweep.lock``; return False if a live sweep already holds it.

    WHY: the whole experiment shares ONE 8 GB GPU.  Two concurrent workers killed a run with
    ``BFC arena OOM`` (269,746,176 B) earlier in this project, and on 2026-10-26 00:46 two
    parents writing the same ``runs/<point>/<seq>/summary.json`` produced a file with a
    complete JSON document followed by a stray ``}`` (JSONDecodeError: Extra data).  A
    second instance must therefore exit *before* it touches the GPU or any output file --
    a lock, not a retry.
    """
    lock_path = OUT_ROOT / ".sweep.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = int(lock_path.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            holder = -1
        if _pid_alive(holder):
            print(
                "[gate_sweep] REFUSING TO START: %s is held by live sweep pid %d. "
                "One GPU, one worker -- exiting without running anything." % (lock_path, holder),
                flush=True,
            )
            return False
        print("[gate_sweep] stale lock from dead pid %d -- taking over" % holder, flush=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
    os.write(fd, ("%d %s\n" % (os.getpid(), time.strftime("%Y-%m-%dT%H:%M:%S"))).encode("ascii"))
    os.close(fd)

    def _release() -> None:
        try:
            holder = int(lock_path.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            return
        if holder == os.getpid():
            lock_path.unlink(missing_ok=True)

    atexit.register(_release)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(CHILD_ENV):
        return _run_child()

    if not acquire_single_instance_lock():
        return 0

    p = argparse.ArgumentParser(description="E070 gate calibration sweep on the held-out split.")
    p.add_argument("--points", default="primary", help="'primary', 'secondary', or comma-separated point ids")
    p.add_argument("--subset", type=int, default=None, help="run on this many fixed random sequences (partial fallback)")
    p.add_argument("--subset-seed", type=int, default=0, help="seed for --subset selection (default %(default)s)")
    p.add_argument("--limit-seqs", type=int, default=None, help="debug: only the first N held-out sequences")
    p.add_argument("--force", action="store_true", help="re-run pairs that already have a summary")
    p.add_argument("--run-only", default=None, help="debug: restrict to one sequence name")
    args = p.parse_args(list(argv) if argv is not None else None)

    print("[gate_sweep] FROZEN DEV SPLIT (selection set -- NEVER swept, never reported):", flush=True)
    print("[gate_sweep]   DanceTrack: %s" % ", ".join(sorted(DEV_SEQUENCES)), flush=True)
    print("[gate_sweep]   MOT17 guard: %s" % ", ".join(sorted(DEV_MOT17)), flush=True)

    all_seqs = heldout_sequences(VAL_ROOT)
    print("[gate_sweep] held-out sequences: %d" % len(all_seqs), flush=True)
    print("[gate_sweep] %s" % ", ".join(all_seqs), flush=True)

    seqs = list(all_seqs)
    partial = False
    if args.subset:
        rng = random.Random(int(args.subset_seed))
        seqs = sorted(rng.sample(all_seqs, int(args.subset)))
        partial = len(seqs) < len(all_seqs)
        print("[gate_sweep] PARTIAL: %d of %d sequences, seed %d -> %s"
              % (len(seqs), len(all_seqs), args.subset_seed, ", ".join(seqs)), flush=True)
    if args.run_only:
        if args.run_only not in all_seqs:
            raise SystemExit("[gate_sweep] FATAL: %r is not a held-out sequence" % args.run_only)
        seqs = [args.run_only]
        partial = True
    if args.limit_seqs:
        seqs = seqs[: int(args.limit_seqs)]
        partial = len(seqs) < len(all_seqs)

    # HARD GUARD: no dev sequence may appear anywhere in the run set.
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[gate_sweep] FATAL: dev sequence(s) %s in the run set; refusing." % leaked)
    assert not (DEV_SEQUENCES & set(seqs)), "dev split leaked into the sweep"

    points = all_points(args.points)
    # The shipped point must run first: it picks the target every other point is pinned to.
    shipped_pid = "margin_%.2f" % SHIPPED_MARGIN
    if shipped_pid in points:
        points = [shipped_pid] + [x for x in points if x != shipped_pid]
    elif any(not point_spec(x)["shipped"] for x in points):
        raise SystemExit("[gate_sweep] the shipped point %s must be in --points to pin target ids" % shipped_pid)

    log_dir = OUT_ROOT / "logs"
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    target_ids: dict[str, int] = {}

    # Pass 1: shipped point (auto-pick).  Pass 2: every other point, target pinned.
    for pid in points:
        spec = point_spec(pid)
        for i, seq in enumerate(seqs, 1):
            tid = None if spec["shipped"] else target_ids.get(seq)
            if not spec["shipped"] and tid is None:
                raise SystemExit("[gate_sweep] no shipped target id for %s; run the shipped point first" % seq)
            ts = time.perf_counter()
            rec = run_pair(pid, seq, target_id=tid, force=bool(args.force), log_dir=log_dir)
            rows.append(rec)
            if spec["shipped"] and rec.get("status") == "INCLUDED":
                target_ids[seq] = int(rec["target_id"])
            print("[gate_sweep] %-12s %2d/%2d %-16s %-8s %6.1fs %s"
                  % (pid, i, len(seqs), seq, rec.get("status"), time.perf_counter() - ts,
                     ("TSR=%.3f FAR=%.3f tid=%s" % (rec["tsr"], rec["far"], rec.get("target_id")))
                     if rec.get("status") == "INCLUDED" else rec.get("reason", "")),
                  flush=True)
    wall = time.perf_counter() - t0

    points_payload: dict[str, Any] = {}
    for pid in points:
        spec = point_spec(pid)
        point_rows = [r for r in rows if r["point"] == pid]
        pool = pool_point(point_rows)
        entry = {
            "point": pid,
            "axis": spec["axis"],
            "value": spec["value"],
            "is_shipped": bool(spec["shipped"]),
            "label": spec["label"],
            "overrides": spec["overrides"],
            "variant": spec["variant"],
            "rows": point_rows,
            "included": pool["included"],
            "excluded": pool["excluded"],
            "pooled": pool["pooled"],
            "bootstrap": pool["bootstrap"],
        }
        points_payload[pid] = entry

    shipped_pool = points_payload.get(shipped_pid)
    if shipped_pool is None:
        raise SystemExit("[gate_sweep] shipped point missing from this run; cannot apply the rule")

    for pid in points:
        entry = points_payload[pid]
        entry["acceptance"] = acceptance(entry, shipped_pool, point_spec(pid))
        entry["deltas"] = per_sequence_deltas(entry["rows"], shipped_pool["rows"])

    payload: dict[str, Any] = {
        "generated_by": "scripts/gate_sweep.py",
        "experiment": "E070",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "dev_mot17": sorted(DEV_MOT17),
        "heldout_sequences": all_seqs,
        "sequences_run": seqs,
        "partial": bool(partial),
        "n_pairs_run": len(rows),
        "protocol": dict(PROTOCOL),
        "shipped": {"cost.search_margin": SHIPPED_MARGIN, "cost.anchor_adaptive_k": SHIPPED_BAND_K},
        "acceptance_rule": {
            "source": "docs/NEXT_ROUND.md:37-39",
            "text": ("ship a value only if pooled TSR rises by >= +5 pp and the upper bound of its FAR CI does not "
                     "exceed the shipped FAR's upper bound. Anything else is recorded as a negative and the default is "
                     "left alone."),
            "tsr_accept_pp": TSR_ACCEPT_PP,
            "far_comparison": "candidate FAR CI p97.5 <= shipped FAR CI p97.5",
        },
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "cli": {"points": args.points, "subset": args.subset, "subset_seed": args.subset_seed,
                "force": bool(args.force), "limit_seqs": args.limit_seqs},
        "order": points,
        "points": points_payload,
        "target_ids": target_ids,
        "wall_seconds": round(wall, 1),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(write_markdown(payload), encoding="utf-8")

    for pid in points:
        pooled = points_payload[pid]["pooled"]
        acc = points_payload[pid]["acceptance"]
        if pooled:
            b = points_payload[pid]["bootstrap"]
            print("[gate_sweep] %-12s TSR=%.4f [%.4f, %.4f] FAR=%.4f [%.4f, %.4f] -> %s"
                  % (pid, pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
                     pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"], acc["verdict"]), flush=True)
        else:
            print("[gate_sweep] %-12s no usable pool -> %s" % (pid, acc["verdict"]), flush=True)
    print("[gate_sweep] wall=%.1fs wrote %s and %s" % (wall, REPORT_JSON.relative_to(ROOT),
                                                       REPORT_MD.relative_to(ROOT)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
