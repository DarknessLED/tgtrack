#!/usr/bin/env python
"""E078 -- the absorb-ratio guard on the 22 held-out DanceTrack sequences.

WHAT IS BEING MEASURED
----------------------
E077-PREP measured that 1654 of 3152 `tracking`-reported frames (52.5 %) are wrong, and
that in 88.6 % of those our box matches a DIFFERENT real person at IoU >= 0.5 with median
area ratio 1.03: a confident identity swap, not a fat box.  The long-term gallery then
absorbs the impostor, so every later frame "confirms" the wrong person -- the memory that
would otherwise detect the drift is the thing being poisoned.

The mechanism (E078, `memory.ltm_absorb_anchor_only` / `ltm_absorb_anchor_ratio`): a crop
is appended to the long-term gallery only while its descriptor is inside a tight band
around the frozen click anchor,

    cos_dist(crop, anchor) <= ratio * TargetTracker._anchor_limit()

Hard negatives and the click crop itself are not gated.  Off by default.

ARMS (ratios derived from data, see results/absorb_sweep/ratio_derivation.json)
------------------------------------------------------------------------------
    absorb_off     flag off                       -- the shipped behaviour (reference)
    absorb_tight   ratio 0.90  (Youden-optimal cut of the measured ratio distribution)
    absorb_loose   ratio 1.09  (correct-frame p99, i.e. still engaged, barely)

PROTOCOL (identical to E070 / E071 so the numbers stay comparable)
-----------------------------------------------------------------
22 held-out DanceTrack-val sequences, dev 0004/0005/0007 excluded and asserted absent,
`--max-frames 300 --subsample 2 --mask none --jobs 1`, one `--ablate absorb` arm per run,
the reference arm picking the target id every other arm is pinned to, frame-weighted
pooling over `n_report_frames`, cluster bootstrap over sequences (2000 reps, seed 0), and
paired per-sequence deltas against the reference.

PRE-REGISTERED RULE (docs/EXPERIMENTS.md, E078; written before the run, never re-tuned)
---------------------------------------------------------------------------------------
    SHIP iff (pooled id_consistency rises >= +3 pp OR pooled TSR rises >= +3 pp)
             with the paired 95 % CI of that same metric excluding 0,
             AND pooled FAR rises by <= +3 pp.
    Otherwise KILL, and say why.

Usage::

    python scripts/absorb_sweep.py                                # all 3 arms, 22 sequences
    python scripts/absorb_sweep.py --arms absorb_off,absorb_tight
    python scripts/absorb_sweep.py --run-only dancetrack0019 --arms absorb_off,absorb_tight
    python scripts/absorb_sweep.py --limit-seqs 1                 # smoke

`E078_SRC=<dir>` puts `<dir>` in front of `sys.path` and pre-imports `tgtrack` from it, so
the sweep can measure a scratch copy of `src/` (the child asserts which file it loaded and
records it in `summary.json`).  `E078_OUT=<dir>` overrides the output root.
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
OUT_ROOT = Path(os.environ.get("E078_OUT") or (ROOT / "results" / "absorb_sweep"))
REPORT_JSON = OUT_ROOT / "summary.json"
REPORT_MD = OUT_ROOT / "REPORT.md"
DERIVATION_JSON = OUT_ROOT / "ratio_derivation.json"

#: FROZEN dev split (E070).  Never swept, never reported.
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}

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

#: The pre-registered rule.
GAIN_PP = 3.0
FAR_MAX_PP = 3.0

CHILD_ENV = "ABSORBSWEEP_CHILD_SEQUENCE"

METRICS = ("tsr", "far", "id_consistency")

ARMS: dict[str, dict[str, Any]] = {
    "absorb_off": {
        "overrides": {
            "memory.ltm_absorb_anchor_only": False,
            "memory.ltm_absorb_anchor_ratio": 1.25,
        },
        "variant": "absorb_off",
        "reference": True,
        "label": "absorb guard OFF (shipped behaviour)",
        "ratio": 1.25,
    },
    "absorb_tight": {
        "overrides": {
            "memory.ltm_absorb_anchor_only": True,
            "memory.ltm_absorb_anchor_ratio": 0.90,
        },
        "variant": "absorb_tight",
        "reference": False,
        "label": "gallery absorbs only inside 0.90 x the click-anchor band",
        "ratio": 0.90,
    },
    "absorb_loose": {
        "overrides": {
            "memory.ltm_absorb_anchor_only": True,
            "memory.ltm_absorb_anchor_ratio": 1.09,
        },
        "variant": "absorb_loose",
        "reference": False,
        "label": "gallery absorbs only inside 1.09 x the click-anchor band",
        "ratio": 1.09,
    },
}

ARM_ORDER = ["absorb_off", "absorb_tight", "absorb_loose"]
REFERENCE_ARM = "absorb_off"


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def arm_spec(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise SystemExit("[absorb_sweep] unknown arm %r; known: %s" % (arm_id, ARM_ORDER))
    spec = dict(ARMS[arm_id])
    spec["arm"] = arm_id
    spec["group"] = "absorb"
    return spec


def all_arms(which: str) -> list[str]:
    if which.strip().lower() in ("", "all"):
        return list(ARM_ORDER)
    ids = [a.strip() for a in which.split(",") if a.strip()]
    for a in ids:
        arm_spec(a)
    return ids


# --------------------------------------------------------------------------- #
# Child mode: one sequence, one pinned arm, no grid.
# --------------------------------------------------------------------------- #
def _run_child() -> int:
    src = os.environ.get("E078_SRC")
    if src:
        src_path = str(Path(src).resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        # Bind `tgtrack` to the scratch tree BEFORE run_target inserts `src/` at
        # sys.path[0]; otherwise the override would measure the shipped code.
        import tgtrack as _tg  # noqa: F401

        if Path(_tg.__file__).resolve().parents[1] != Path(src_path):
            raise SystemExit("[absorb_sweep] E078_SRC did not win: tgtrack is %s" % _tg.__file__)
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - late import needs sys.path
    import tgtrack as tg

    seq = os.environ[CHILD_ENV]
    spec = arm_spec(os.environ["ABSORBSWEEP_ARM"])
    overrides = json.loads(os.environ.get("ABSORBSWEEP_OVERRIDES") or "{}")
    variant_name = os.environ["ABSORBSWEEP_VARIANT"]
    if overrides != spec["overrides"]:
        raise SystemExit("[absorb_sweep] child override mismatch: %r vs %r"
                         % (overrides, spec["overrides"]))

    def _pinned(groups: Any = None, **kwargs: Any) -> list[Any]:
        """Exactly one variant: the arm's overrides on top of the shipped config."""
        return [rt.Variant(variant_name, "absorb_sweep", dict(overrides),
                           "E078 sweep arm " + spec["arm"])]

    rt.build_variants = _pinned  # noqa: WPS437 - process-local, file untouched

    specs_path = Path(os.environ["ABSORBSWEEP_SPECS"])
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
        if len(rows) != 1:
            raise SystemExit("[absorb_sweep] GUARD: expected exactly 1 spec, got %d" % len(rows))
        if rows[0]["overrides"] != overrides or rows[0]["variant"] != variant_name:
            raise SystemExit("[absorb_sweep] GUARD: spec carries %r / %r, intended %r / %r"
                             % (rows[0]["overrides"], rows[0]["variant"], overrides, variant_name))
        return _orig_run_grid(specs, jobs=jobs, quiet=quiet)

    rt.run_grid = _checked_run_grid

    out_dir = os.environ["ABSORBSWEEP_OUT"]
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", PROTOCOL["mask"],
        "--max-frames", str(PROTOCOL["max_frames"]),
        "--subsample", str(PROTOCOL["subsample"]),
        "--jobs", str(PROTOCOL["jobs"]),
        # Recorded, never used for selection: build_variants is pinned above.
        "--ablate", "absorb",
    ]
    if os.environ.get("ABSORBSWEEP_TARGET_ID"):
        argv += ["--target-id", os.environ["ABSORBSWEEP_TARGET_ID"]]
    print("[absorb_sweep] child tgtrack=%s arm=%s" % (tg.__file__, spec["arm"]), flush=True)
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise SystemExit("[absorb_sweep] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    present = [
        name for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    seqs = [name for name in present if name not in DEV_SEQUENCES]
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[absorb_sweep] FATAL: dev sequence(s) %s in held-out selection" % leaked)
    if not seqs:
        raise SystemExit("[absorb_sweep] FATAL: no sequences under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats (same estimator as scripts/gate_sweep.py / consensus_sweep.py)
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


def cluster_bootstrap(rows: Sequence[dict[str, Any]], *, key: str,
                      reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
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


def paired_bootstrap(arm_rows: Sequence[dict[str, Any]], ref_rows: Sequence[dict[str, Any]], *,
                     key: str, reps: int = BOOTSTRAP_REPS,
                     seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap the *paired* per-sequence delta of one metric."""
    ref = {r["sequence"]: float(r[key]) for r in ref_rows if r.get("status") == "INCLUDED"}
    pairs: list[tuple[str, float, float]] = []
    unpaired: list[str] = []
    for r in arm_rows:
        seq = r["sequence"]
        if r.get("status") != "INCLUDED" or seq not in ref:
            unpaired.append(seq)
            continue
        pairs.append((seq, float(r[key]) - ref[seq], float(r[key])))
    if not pairs:
        return {"reps": reps, "n_paired": 0, "unpaired_sequences": unpaired,
                "delta_mean": None, "p2.5": None, "p97.5": None}
    rng = random.Random(seed)
    deltas = [d for _, d, _ in pairs]
    n = len(deltas)
    samples: list[float] = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(deltas[i] for i in idx) / float(n))
    lo, mid, hi = _pct(samples, [2.5, 50.0, 97.5])
    up = sum(1 for d in deltas if d > 1e-12)
    down = sum(1 for d in deltas if d < -1e-12)
    return {
        "reps": reps,
        "seed": seed,
        "unit": "sequence",
        "estimator": "mean of paired per-sequence deltas, resampling sequences with replacement",
        "n_paired": n,
        "unpaired_sequences": unpaired,
        "delta_mean": round(sum(deltas) / float(n), 9),
        "p2.5": round(lo, 9),
        "p50": round(mid, 9),
        "p97.5": round(hi, 9),
        "sequences_up": up,
        "sequences_down": down,
        "sequences_flat": n - up - down,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "per_sequence": [
            {"sequence": seq, "delta_pp": round(d * 100.0, 3), "arm_value": round(v, 9),
             "reference_value": round(ref[seq], 9)}
            for seq, d, v in pairs
        ],
    }


# --------------------------------------------------------------------------- #
# Running one (arm, sequence)
# --------------------------------------------------------------------------- #
def run_pair(arm_id: str, seq: str, *, target_id: int | None, force: bool,
             log_dir: Path) -> dict[str, Any]:
    spec = arm_spec(arm_id)
    out_dir = OUT_ROOT / "runs" / arm_id / seq
    summary_path = out_dir / "summary.json"
    rec_path = OUT_ROOT / "summaries" / arm_id / ("%s.json" % seq)
    log_path = log_dir / ("%s__%s.log" % (arm_id, seq))
    specs_path = out_dir / "specs.json"
    owner_path = out_dir / ".mine"

    base: dict[str, Any] = {
        "arm": arm_id,
        "ratio": spec["ratio"],
        "reference": bool(spec["reference"]),
        "sequence": seq,
        "target_id_forced": target_id,
        "summary_path": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
        "log": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
    }

    if summary_path.is_file() and not force and owner_path.is_file():
        cached = _parse_summary(summary_path, seq, target_id, specs_path)
        if cached.get("status") == "INCLUDED":
            cached.update(base)
            cached["from_cache"] = True
            return cached
    elif summary_path.is_file() and not force:
        print("[absorb_sweep] PURGE foreign run dir (no .mine provenance): %s/%s"
              % (arm_id, seq), flush=True)

    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["ABSORBSWEEP_ARM"] = arm_id
    env["ABSORBSWEEP_VARIANT"] = spec["variant"]
    env["ABSORBSWEEP_OVERRIDES"] = json.dumps(spec["overrides"])
    env["ABSORBSWEEP_OUT"] = str(out_dir)
    env["ABSORBSWEEP_SPECS"] = str(specs_path)
    env["PYTHONIOENCODING"] = "utf-8"
    if target_id is not None:
        env["ABSORBSWEEP_TARGET_ID"] = str(int(target_id))
    else:
        env.pop("ABSORBSWEEP_TARGET_ID", None)

    log_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    rec: dict[str, Any] = {}
    for attempt in (1, 2):
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with log_path.open("a" if attempt > 1 else "w", encoding="utf-8") as fh:
            fh.write("# arm=%s seq=%s attempt=%d\n# env=%s\n" % (
                arm_id, seq, attempt,
                json.dumps({k: v for k, v in env.items() if k.startswith("ABSORBSWEEP_") or k == "E078_SRC"},
                           sort_keys=True)))
            fh.flush()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True,
            )
        wall = time.perf_counter() - t0
        code = int(proc.returncode)
        if code == 3:
            rec = {"status": "EXCLUDED",
                   "reason": "exit 3: no observable target under mask=%s" % PROTOCOL["mask"],
                   "harness_message": _message_from_log(log_path)}
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

    rec.update(base)
    rec["attempts"] = attempts
    rec["from_cache"] = False
    if rec.get("status") != "INCLUDED":
        rec["log_tail"] = _tail_lines(log_path, 8)

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    if rec.get("status") == "INCLUDED":
        owner_path.write_text(json.dumps({
            "arm": arm_id, "sequence": seq, "variant": spec["variant"],
            "overrides": spec["overrides"], "pid": os.getpid(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": rec.get("wall_seconds"),
        }, indent=2) + "\n", encoding="utf-8")
    return rec


def _record_stats(summary_path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    """``tracker_stats`` of the single run this summary covers (``per_run.json``)."""
    per_run = summary_path.parent / "per_run.json"
    if per_run.is_file():
        try:
            payload = json.loads(per_run.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        runs = payload.get("runs") or []
        if len(runs) == 1 and isinstance(runs[0].get("tracker_stats"), dict):
            return runs[0]["tracker_stats"]
    stats = row.get("tracker_stats")
    return stats if isinstance(stats, dict) else None


def _absorb_counters(stats: dict[str, Any] | None) -> dict[str, Any]:
    """The E078 engagement counters out of ``TargetTracker.stats()``.

    Missing counters are reported as -1, never as a silent zero: "the guard never
    refused anything" and "the guard never ran" must stay distinguishable.
    """
    if not isinstance(stats, dict):
        return {"available": False, "absorb_anchor_only": None, "absorb_anchor_ratio": None,
                "n_gated": -1, "n_blocked": -1}
    return {
        "available": True,
        "absorb_anchor_only": stats.get("absorb_anchor_only"),
        "absorb_anchor_ratio": stats.get("absorb_anchor_ratio"),
        "n_gated": int(stats.get("absorb_anchor_gated", -1)),
        "n_blocked": int(stats.get("absorb_anchor_blocked", -1)),
    }


def _parse_summary(summary_path: Path, seq: str, forced_id: int | None,
                   specs_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {"status": "ERROR", "reason": "harness exited 0 but wrote no summary.json"}
    text = summary_path.read_text(encoding="utf-8")
    recovered = False
    try:
        summary = json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            summary, end = json.JSONDecoder().raw_decode(text)
            recovered = True
            print("[absorb_sweep] WARNING %s: salvaged summary.json, ignored %d trailing chars (%s)"
                  % (seq, len(text) - end, exc), flush=True)
        except json.JSONDecodeError:
            return {"status": "ERROR", "reason": "corrupt summary.json (unparseable: %s)" % exc}
    meta = summary.get("meta", {})

    variants = list(meta.get("variants") or [])
    if len(variants) != 1:
        return {"status": "ERROR", "reason": "GUARD: meta.variants=%r (expected exactly one arm)" % (variants,)}
    spec_rows = json.loads(specs_path.read_text(encoding="utf-8")) if specs_path.is_file() else []
    if len(spec_rows) != 1:
        return {"status": "ERROR", "reason": "GUARD: specs.json rows=%d (expected 1)" % len(spec_rows)}

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
        return {"status": "ERROR", "reason": "expected 1 headline row for mask=%s, got %d"
                                            % (PROTOCOL["mask"], len(rows))}
    row = rows[0]

    targets = meta.get("target_ids") or {}
    target_id = targets.get(seq)
    if forced_id is not None and target_id != int(forced_id):
        return {"status": "ERROR", "reason": "TARGET GUARD: meta.target_ids[%s]=%r, forced %r"
                                            % (seq, target_id, forced_id)}
    if target_id is None:
        return {"status": "ERROR", "reason": "TARGET GUARD: no meta.target_ids entry for %s" % seq}

    decision = (meta.get("target_decisions") or {}).get(seq, {})
    counters = _absorb_counters(_record_stats(summary_path, row))
    rec = {
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
        "fragmentation": float(row.get("fragmentation", float("nan"))),
        "ms_per_frame": float(row.get("ms_per_frame", float("nan"))),
        "target_id": target_id,
        "target_reason": decision.get("reason"),
        "guard_passed": decision.get("guard_passed"),
        "harness_wall_seconds": meta.get("wall_seconds"),
    }
    rec.update(counters)
    return rec


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
# Pooling + the pre-registered rule
# --------------------------------------------------------------------------- #
def pool_arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inc = [r for r in rows if r.get("status") == "INCLUDED"]
    exc = [r for r in rows if r.get("status") != "INCLUDED"]
    if not inc:
        return {"included": [], "excluded": exc, "pooled": None, "bootstrap": None,
                "engagement": None}
    weights = [float(r["n_report_frames"]) for r in inc]
    pooled: dict[str, Any] = {
        "total_report_frames": int(sum(weights)),
        "n_included": len(inc),
        "n_excluded": len(exc),
        "estimator": "frame-weighted pooled mean over included sequences",
    }
    for key in METRICS:
        vals = [float(r[key]) for r in inc]
        pooled[key] = round(_weighted(vals, weights), 6)
        pooled["%s_mean" % key] = round(sum(vals) / len(inc), 6)
        pooled["%s_median" % key] = round(_pct(vals, [50.0])[0], 6)
    eng = {"n_gated": 0, "n_blocked": 0, "n_report_frames_gated_arm": 0,
           "available": bool(inc) and bool(inc[0].get("available", False))}
    for r in inc:
        eng["n_gated"] += max(0, int(r.get("n_gated", -1) or 0))
        eng["n_blocked"] += max(0, int(r.get("n_blocked", -1) or 0))
        if int(r.get("n_gated", -1) or 0) > 0:
            eng["n_report_frames_gated_arm"] += 1
    return {
        "included": inc,
        "excluded": exc,
        "pooled": pooled,
        "bootstrap": {key: cluster_bootstrap(inc, key=key) for key in METRICS},
        "engagement": eng,
    }


def verdict(arm_entry: dict[str, Any], ref_entry: dict[str, Any]) -> dict[str, Any]:
    """The E078 rule, applied verbatim and never re-tuned.

    SHIP iff (pooled id_consistency +3 pp OR pooled TSR +3 pp) with the paired 95 % CI
    of that same metric excluding 0, AND pooled FAR rises by <= +3 pp.
    """
    out: dict[str, Any] = {
        "arm": arm_entry["arm"],
        "ratio": arm_entry["ratio"],
        "rule": ("SHIP iff (pooled dID >= +%.1f pp OR pooled dTSR >= +%.1f pp) with the paired 95%% CI "
                 "of that metric excluding 0, AND pooled dFAR <= +%.1f pp; otherwise KILL"
                 % (GAIN_PP, GAIN_PP, FAR_MAX_PP)),
    }
    if arm_entry["arm"] == ref_entry["arm"]:
        out.update(verdict="REFERENCE", text="Reference arm: the shipped behaviour (guard off).")
        return out
    if not arm_entry.get("pooled") or not ref_entry.get("pooled"):
        out.update(verdict="KILL", text="KILL: no complete pool (excluded/error runs).")
        return out

    p = arm_entry["pooled"]
    r = ref_entry["pooled"]
    d = {k: (p[k] - r[k]) * 100.0 for k in METRICS}
    pb = arm_entry["paired"]
    ci = {k: ((pb[k]["p2.5"] or 0.0) * 100.0, (pb[k]["p97.5"] or 0.0) * 100.0) for k in METRICS}
    far_ci_hi = ci["far"][1]

    ok_id = d["id_consistency"] >= GAIN_PP and bool(pb["id_consistency"].get("excludes_zero")) \
        and ci["id_consistency"][0] > 0.0
    ok_tsr = d["tsr"] >= GAIN_PP and bool(pb["tsr"].get("excludes_zero")) and ci["tsr"][0] > 0.0
    ok_far = d["far"] <= FAR_MAX_PP
    ship = bool((ok_id or ok_tsr) and ok_far)

    eng = arm_entry.get("engagement") or {}
    inert = int(eng.get("n_gated", 0) or 0) == 0

    if ship:
        why = "; ".join(
            ["pooled dID %+.2f pp / dTSR %+.2f pp" % (d["id_consistency"], d["tsr"])]
            + (["dID CI [%+.2f, %+.2f] pp excludes 0" % ci["id_consistency"]] if ok_id else [])
            + (["dTSR CI [%+.2f, %+.2f] pp excludes 0" % ci["tsr"]] if ok_tsr else [])
            + ["pooled dFAR %+.2f pp <= +%.1f" % (d["far"], FAR_MAX_PP)])
        v = "SHIP"
    else:
        bad = []
        if not (ok_id or ok_tsr):
            bad.append("neither pooled dID %+.2f pp (CI [%+.2f, %+.2f]) nor pooled dTSR %+.2f pp "
                       "(CI [%+.2f, %+.2f]) clears +%.1f pp with a CI excluding 0"
                       % (d["id_consistency"], ci["id_consistency"][0], ci["id_consistency"][1],
                          d["tsr"], ci["tsr"][0], ci["tsr"][1], GAIN_PP))
        if not ok_far:
            bad.append("pooled dFAR %+.2f pp > +%.1f pp" % (d["far"], FAR_MAX_PP))
        v = "KILL"
        why = "; ".join(bad) + ("; the guard never engaged (0 crops gated)" if inert else "")

    out.update(
        verdict=v,
        text="%s: %s." % (v, why),
        pooled_dtsr_pp=round(d["tsr"], 3),
        pooled_dfar_pp=round(d["far"], 3),
        pooled_did_consistency_pp=round(d["id_consistency"], 3),
        paired_dtsr_mean_pp=round((pb["tsr"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_dtsr_ci_pp=[round(ci["tsr"][0], 3), round(ci["tsr"][1], 3)],
        paired_dfar_mean_pp=round((pb["far"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_dfar_ci_pp=[round(ci["far"][0], 3), round(ci["far"][1], 3)],
        paired_did_mean_pp=round((pb["id_consistency"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_did_ci_pp=[round(ci["id_consistency"][0], 3), round(ci["id_consistency"][1], 3)],
        paired_dtsr_excludes_zero=bool(pb["tsr"].get("excludes_zero")),
        paired_dfar_excludes_zero=bool(pb["far"].get("excludes_zero")),
        paired_did_excludes_zero=bool(pb["id_consistency"].get("excludes_zero")),
        far_ci_upper_pp=round(far_ci_hi, 3),
        far_ci_upper_ok=bool(far_ci_hi <= FAR_MAX_PP),
        id_ok=bool(ok_id), tsr_ok=bool(ok_tsr), far_ok=bool(ok_far),
        mechanism_inert=bool(inert),
        sequences_up_id=int(pb["id_consistency"]["sequences_up"]),
        sequences_down_id=int(pb["id_consistency"]["sequences_down"]),
        sequences_flat_id=int(pb["id_consistency"]["sequences_flat"]),
        sequences_up_tsr=int(pb["tsr"]["sequences_up"]),
        sequences_down_tsr=int(pb["tsr"]["sequences_down"]),
        sequences_flat_tsr=int(pb["tsr"]["sequences_flat"]),
        sequences_up_far=int(pb["far"]["sequences_up"]),
        sequences_down_far=int(pb["far"]["sequences_down"]),
        sequences_flat_far=int(pb["far"]["sequences_flat"]),
    )
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, boot: dict[str, Any] | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if not boot or boot.get("p2.5") is None:
        return "%.*f" % (digits, value)
    return "%.*f [%.*f, %.*f]" % (digits, value, digits, boot["p2.5"], digits, boot["p97.5"])


def write_markdown(payload: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a("# E078 -- the absorb-ratio guard on the held-out DanceTrack split")
    a("")
    a("<!-- generated by scripts/absorb_sweep.py; every number is read from %s -->"
      % REPORT_JSON.name)
    a("")
    verdicts = payload["verdicts"]
    a("## Verdict (pre-registered rule)")
    a("")
    a("> %s" % payload["rule_text"])
    a("")
    for arm in payload["order"][1:]:
        v = verdicts[arm]
        a("- **`%s`** (ratio %.2f) -- **%s**: %s" % (arm, v["ratio"], v["verdict"], v["text"]))
    ship = [arm for arm in payload["order"][1:] if verdicts[arm]["verdict"] == "SHIP"]
    if ship:
        a("")
        a("**Ship decision: `%s` clears the row. The default stays OFF until the change is "
          "applied by hand (E078 ships the knob, not the behaviour).**" % ", ".join(ship))
    else:
        a("")
        a("**Ship decision: kill the mechanism; the shipped default (`ltm_absorb_anchor_only = "
          "False`) stays.**")
    a("")

    a("## Protocol (identical to E070 / E071, so the numbers stay comparable)")
    a("")
    prot = payload["protocol"]
    a("- data: `data/raw/dancetrack/val`, the **%d** held-out sequences; frozen dev split %s "
      "asserted absent" % (payload["n_heldout"], ", ".join(sorted(DEV_SEQUENCES))))
    a("- `--max-frames %d --subsample %d --mask %s --jobs %d`, shipped defaults elsewhere"
      % (prot["max_frames"], prot["subsample"], prot["mask"], prot["jobs"]))
    a("- one `--ablate absorb` arm per run, `run_target.build_variants` replaced in the child; "
      "`specs.json` asserts exactly 1 spec with exactly the intended overrides and each "
      "`summary.json` reports exactly 1 variant")
    a("- target identity: the reference arm picks it (`stressed`), every other arm is pinned to "
      "the same id and the mismatch is a hard error")
    a("- pooling: frame-weighted mean over `n_report_frames`; CIs are cluster bootstraps over "
      "**sequences** (%d reps, seed %d); deltas are **paired** per sequence against the "
      "reference arm" % (payload["bootstrap"]["reps"], payload["bootstrap"]["seed"]))
    a("- runs: %d (including any retried pair); wall %.1f min" % (payload["n_pairs_run"],
                                                                payload["wall_seconds"] / 60.0))
    a("")

    a("## Ratio derivation (from data, before the sweep)")
    a("")
    der = payload.get("derivation")
    if not der:
        a("_`%s` missing: the ratios were NOT derived, so this sweep cannot be read as a test "
          "of a data-derived mechanism._" % DERIVATION_JSON.name)
    else:
        a("Measured on %d replayed frames (%d correct / %d wrong) from the E076 traces, with the "
          "real embedder and the shipped config. The knob is a *ratio* against the E029 band, so "
          "the derived quantity is `dist / band_hi`, not a raw distance."
          % (der["n_frames"], der["n_correct"], der["n_wrong"]))
        a("")
        a("| quantile | " + " | ".join(der["quantiles"]) + " |")
        a("|---" * (len(der["quantiles"]) + 1) + "|")
        for name, key in (("distance, correct", "dist_correct"), ("distance, wrong", "dist_wrong"),
                          ("ratio, correct", "ratio_correct"), ("ratio, wrong", "ratio_wrong")):
            a("| %s | %s |" % (name, " | ".join("%.3f" % der[key][q] for q in der["quantiles"])))
        a("")
        a("| ratio cut | correct absorbs kept | wrong absorbs refused |")
        a("|---|---|---|")
        for row in der["separation"]:
            a("| %.2f | %.3f | %.3f |" % (row["ratio"], row["keep_correct"], row["refuse_wrong"]))
        a("")
        a("- `absorb_tight` = **%.2f** -- %s" % (der["chosen"]["tight"], der["chosen"]["tight_why"]))
        a("- `absorb_loose` = **%.2f** -- %s" % (der["chosen"]["loose"], der["chosen"]["loose_why"]))
        a("- tracking-only slice (the confident-swap population E077-PREP measured, n=%d): "
          "correct ratio p50 %.3f / p90 %.3f; wrong ratio p50 %.3f / p90 %.3f -- separation is "
          "**weaker** inside that population, which bounds the effect this sweep can find."
          % (der["tracking"]["n"], der["tracking"]["ratio_correct"]["p50"],
             der["tracking"]["ratio_correct"]["p90"], der["tracking"]["ratio_wrong"]["p50"],
             der["tracking"]["ratio_wrong"]["p90"]))
    a("")

    a("## Per-arm results")
    a("")
    a("| arm | ratio | sequences | report frames | id_consistency | TSR | FAR |")
    a("|---|---|---|---|---|---|---|")
    for arm in payload["order"]:
        e = payload["arms"][arm]
        p = e["pooled"]
        if not p:
            a("| `%s` | %.2f | 0 | 0 | n/a | n/a | n/a |" % (arm, e["ratio"]))
            continue
        b = e["bootstrap"]
        a("| `%s` | %.2f | %d | %d | %s | %s | %s |"
          % (arm, e["ratio"], p["n_included"], p["total_report_frames"],
             _fmt(p["id_consistency"], b["id_consistency"]), _fmt(p["tsr"], b["tsr"]),
             _fmt(p["far"], b["far"])))
    a("")
    a("[value [95 %% CI]]: cluster bootstrap over sequences, %d reps, seed %d."
      % (payload["bootstrap"]["reps"], payload["bootstrap"]["seed"]))
    a("")

    a("## Paired deltas vs `%s`" % REFERENCE_ARM)
    a("")
    a("| arm | metric | pooled delta | paired mean | paired 95 % CI | up / down / flat | CI excludes 0 |")
    a("|---|---|---|---|---|---|---|")
    for arm in payload["order"][1:]:
        v = verdicts[arm]
        for key, label, dm, ci, up in (
            ("id_consistency", "dID", "paired_did_mean_pp", "paired_did_ci_pp", "id"),
            ("tsr", "dTSR", "paired_dtsr_mean_pp", "paired_dtsr_ci_pp", "tsr"),
            ("far", "dFAR", "paired_dfar_mean_pp", "paired_dfar_ci_pp", "far"),
        ):
            pooled_key = {"id_consistency": "pooled_did_consistency_pp",
                          "tsr": "pooled_dtsr_pp",
                          "far": "pooled_dfar_pp"}[key]
            a("| `%s` | %s | %+.2f pp | %+.2f pp | [%+.2f, %+.2f] pp | %d / %d / %d | %s |"
              % (arm, label, v[pooled_key], v[dm], v[ci][0], v[ci][1],
                 v["sequences_up_" + up], v["sequences_down_" + up], v["sequences_flat_" + up],
                 "yes" if v["paired_d%s_excludes_zero" % up] else "no"))
    a("")

    a("## Mechanism engagement (a guard that never fires is not a result)")
    a("")
    a("| arm | crops gated | crops refused | sequences with any gating |")
    a("|---|---|---|---|")
    for arm in payload["order"]:
        e = payload["arms"][arm]
        eng = e.get("engagement") or {}
        a("| `%s` | %d | %d | %d |" % (arm, eng.get("n_gated", -1), eng.get("n_blocked", -1),
                                        eng.get("n_report_frames_gated_arm", 0)))
    a("")

    a("## Per-sequence detail")
    a("")
    per: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in payload["order"]:
        for row in payload["arms"][arm]["included"]:
            per.setdefault(row["sequence"], {})[arm] = row
    a("| sequence | frames | " + " | ".join("`%s` id / TSR / FAR" % arm for arm in payload["order"]) + " |")
    a("|---|---|" + "---|" * len(payload["order"]))
    for seq in sorted(per):
        cells = []
        nfr = ""
        for arm in payload["order"]:
            row = per[seq].get(arm)
            if not row:
                cells.append("n/a")
                continue
            nfr = "%d" % row["n_report_frames"]
            cells.append("%.3f / %.3f / %.3f" % (row["id_consistency"], row["tsr"], row["far"]))
        a("| %s | %s | %s |" % (seq, nfr, " | ".join(cells)))
    a("")

    if payload.get("excluded"):
        a("## Excluded / errored pairs")
        a("")
        for row in payload["excluded"]:
            a("- `%s` / `%s`: %s (%s)" % (row["arm"], row["sequence"], row["status"],
                                          row.get("reason")))
        a("")

    a("## Notes")
    a("")
    for note in payload.get("notes", []):
        a("- %s" % note)
    a("")
    return "\n".join(L) + "\n"


def code_state_snapshot() -> dict[str, Any]:
    p = OUT_ROOT / "code_snapshot_start.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return ("%d" % pid) in (out.stdout or "")


def acquire_single_instance_lock() -> bool:
    """Take ``<OUT_ROOT>/.sweep.lock``; one GPU, one worker."""
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
            print("[absorb_sweep] REFUSING TO START: %s is held by live sweep pid %d. "
                  "One GPU, one worker." % (lock_path, holder), flush=True)
            return False
        print("[absorb_sweep] stale lock from dead pid %d -- taking over" % holder, flush=True)
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


def _derivation_block() -> dict[str, Any] | None:
    if not DERIVATION_JSON.is_file():
        return None
    try:
        raw = json.loads(DERIVATION_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    keys = ["p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    return {
        "n_frames": raw["n_frames"],
        "n_correct": raw["n_correct"],
        "n_wrong": raw["n_wrong"],
        "quantiles": keys,
        "dist_correct": raw["dist_correct"],
        "dist_wrong": raw["dist_wrong"],
        "ratio_correct": raw["ratio_correct"],
        "ratio_wrong": raw["ratio_wrong"],
        "separation": raw["separation"],
        "tracking": raw["tracking_slice"],
        "chosen": raw["chosen"],
        "source": str(DERIVATION_JSON.relative_to(ROOT)) if DERIVATION_JSON.is_relative_to(ROOT) else str(DERIVATION_JSON),
    }


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(CHILD_ENV):
        return _run_child()

    if not acquire_single_instance_lock():
        return 0

    p = argparse.ArgumentParser(description="E078 absorb-ratio guard sweep on the held-out split.")
    p.add_argument("--arms", default="all", help="'all' or comma-separated arm ids")
    p.add_argument("--limit-seqs", type=int, default=None, help="debug: only the first N sequences")
    p.add_argument("--run-only", default=None, help="debug: restrict to one sequence name")
    p.add_argument("--force", action="store_true", help="re-run pairs that already have a summary")
    args = p.parse_args(list(argv) if argv is not None else None)

    print("[absorb_sweep] FROZEN DEV SPLIT (never swept, never reported): %s"
          % ", ".join(sorted(DEV_SEQUENCES)), flush=True)

    all_seqs = heldout_sequences(VAL_ROOT)
    print("[absorb_sweep] held-out sequences: %d" % len(all_seqs), flush=True)
    seqs = list(all_seqs)
    partial = False
    if args.run_only:
        if args.run_only not in all_seqs:
            raise SystemExit("[absorb_sweep] FATAL: %r is not a held-out sequence" % args.run_only)
        seqs = [args.run_only]
        partial = True
    if args.limit_seqs:
        seqs = seqs[: int(args.limit_seqs)]
        partial = len(seqs) < len(all_seqs)
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[absorb_sweep] FATAL: dev sequence(s) %s in the run set; refusing." % leaked)
    assert not (DEV_SEQUENCES & set(seqs)), "dev split leaked into the sweep"

    arms = all_arms(args.arms)
    if REFERENCE_ARM not in arms:
        raise SystemExit("[absorb_sweep] the reference arm %s must be in --arms to pin target ids"
                         % REFERENCE_ARM)
    arms = [REFERENCE_ARM] + [a for a in arms if a != REFERENCE_ARM]

    log_dir = OUT_ROOT / "logs"
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    target_ids: dict[str, int] = {}

    for arm in arms:
        spec = arm_spec(arm)
        for i, seq in enumerate(seqs, 1):
            tid = None if spec["reference"] else target_ids.get(seq)
            if not spec["reference"] and tid is None:
                raise SystemExit("[absorb_sweep] no reference target id for %s; run %s first"
                                 % (seq, REFERENCE_ARM))
            ts = time.perf_counter()
            rec = run_pair(arm, seq, target_id=tid, force=bool(args.force), log_dir=log_dir)
            rows.append(rec)
            if spec["reference"] and rec.get("status") == "INCLUDED":
                target_ids[seq] = int(rec["target_id"])
            print("[absorb_sweep] %-13s %2d/%2d %-16s %-9s %6.1fs %s"
                  % (arm, i, len(seqs), seq, rec.get("status"), time.perf_counter() - ts,
                     ("ID=%.3f TSR=%.3f FAR=%.3f gate=%s/%s"
                      % (rec["id_consistency"], rec["tsr"], rec["far"],
                         rec.get("n_blocked"), rec.get("n_gated")))
                     if rec.get("status") == "INCLUDED" else rec.get("reason", "")), flush=True)
    wall = time.perf_counter() - t0

    arms_payload: dict[str, Any] = {}
    for arm in arms:
        spec = arm_spec(arm)
        arm_rows = [r for r in rows if r["arm"] == arm]
        pool = pool_arm(arm_rows)
        arms_payload[arm] = {
            "arm": arm,
            "ratio": spec["ratio"],
            "label": spec["label"],
            "overrides": spec["overrides"],
            "variant": spec["variant"],
            "reference": bool(spec["reference"]),
            "rows": arm_rows,
            "included": pool["included"],
            "excluded": pool["excluded"],
            "pooled": pool["pooled"],
            "bootstrap": pool["bootstrap"],
            "engagement": pool["engagement"],
        }

    ref_entry = arms_payload[REFERENCE_ARM]
    for arm in arms:
        e = arms_payload[arm]
        e["paired"] = {key: paired_bootstrap(e["rows"], ref_entry["rows"], key=key) for key in METRICS}
        e["verdict"] = verdict(e, ref_entry)

    notes = [
        "`absorb_off` spells out the shipped value (`ltm_absorb_anchor_only=False`, ratio 1.25) "
        "so the reference row is self-explanatory; with the flag off the ratio is inert.",
        "The guard multiplies `TargetTracker._absorb_band()`, which reuses the E029 adaptive "
        "band and falls back to `cost.anchor_adaptive_ceil` while that band is still in "
        "bootstrap. In the shipped config the band saturates at the 0.75 ceiling on %.0f %% of "
        "frames, so the guard is for most frames an absolute distance cut at "
        "`ratio x 0.75` (0.675 at 0.90, 0.818 at 1.09)." % (100.0 * 0.75),
        "Hard negatives (`ltm.add_hard_negative`) and the click crop (`force=True`) are not "
        "gated: the impostor must still be remembered as a negative, and the anchor must exist "
        "for the guard to have anything to measure against.",
        "The trace-based derivation re-encodes the *reported* (smoothed) box, while `_absorb` "
        "sees the accepted detection box; the smoothing residual is small and affects correct "
        "and wrong frames alike, so it moves the cut, not the separation.",
        "This is the second sweep in the programme whose arms are configurationally engaged but "
        "whose expected effect is bounded by the derivation: inside the tracking-only "
        "population the ratio distributions overlap substantially (Youden J 0.131 at 0.90), so "
        "a null is informative and a large effect would be surprising.",
    ]

    payload: dict[str, Any] = {
        "generated_by": "scripts/absorb_sweep.py",
        "experiment": "E078",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "heldout_sequences": all_seqs,
        "n_heldout": len(all_seqs),
        "sequences_run": seqs,
        "partial": bool(partial),
        "n_pairs_run": len(rows),
        "protocol": dict(PROTOCOL),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "rule_text": ("SHIP only if pooled `id_consistency` rises by >= +3 pp OR pooled TSR by "
                      ">= +3 pp with the paired 95 % CI excluding 0, AND pooled FAR rises by no "
                      "more than +3 pp. Otherwise KILL."),
        "rule": {"gain_pp": GAIN_PP, "far_max_pp": FAR_MAX_PP,
                 "source": "docs/EXPERIMENTS.md E078 (pre-registered)"},
        "order": arms,
        "arms": arms_payload,
        "verdicts": {arm: arms_payload[arm]["verdict"] for arm in arms},
        "derivation": _derivation_block(),
        "excluded": [r for r in rows if r.get("status") != "INCLUDED"],
        "target_ids": target_ids,
        "wall_seconds": round(wall, 1),
        "code_state": code_state_snapshot(),
        "env": {"E078_SRC": os.environ.get("E078_SRC"), "E078_OUT": os.environ.get("E078_OUT"),
                "tgtrack": _tgtrack_path()},
        "cli": {"arms": args.arms, "limit_seqs": args.limit_seqs, "run_only": args.run_only,
                "force": bool(args.force)},
        "notes": notes,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(write_markdown(payload), encoding="utf-8")

    for arm in arms:
        pooled = arms_payload[arm]["pooled"]
        v = arms_payload[arm]["verdict"]
        if pooled:
            b = arms_payload[arm]["bootstrap"]
            print("[absorb_sweep] %-13s ID=%.4f [%.4f, %.4f] TSR=%.4f [%.4f, %.4f] FAR=%.4f [%.4f, %.4f] -> %s"
                  % (arm, pooled["id_consistency"], b["id_consistency"]["p2.5"], b["id_consistency"]["p97.5"],
                     pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
                     pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"], v["verdict"]), flush=True)
            if v["verdict"] != "REFERENCE":
                print("[absorb_sweep] %-13s dID %+.2f pp %s | dTSR %+.2f pp %s | dFAR %+.2f pp %s"
                      % (arm, v["pooled_did_consistency_pp"], v["paired_did_ci_pp"],
                         v["pooled_dtsr_pp"], v["paired_dtsr_ci_pp"],
                         v["pooled_dfar_pp"], v["paired_dfar_ci_pp"]), flush=True)
        else:
            print("[absorb_sweep] %-13s no usable pool -> %s" % (arm, v["verdict"]), flush=True)
    print("[absorb_sweep] wall=%.1fs wrote %s and %s"
          % (wall, REPORT_JSON, REPORT_MD), flush=True)
    return 0


def _tgtrack_path() -> str | None:
    try:
        import tgtrack
        return str(tgtrack.__file__)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
