"""E071 -- temporal-consensus ablation on the 22 held-out DanceTrack sequences.

The E070 protocol, unchanged, with the handback rule as the only variable:

  * data root ``data/raw/dancetrack/val``, the 22 held-out sequences
    (the frozen dev split 0004 / 0005 / 0007 is asserted absent),
  * ``--max-frames 300 --subsample 2 --mask none --jobs 1``,
  * one ``--ablate consensus`` arm per run, the reference arm picking the
    target id that every other arm is pinned to,
  * frame-weighted pooling over ``n_report_frames`` and a cluster bootstrap
    over *sequences* (2000 reps, seed 0) for every confidence interval.

Why a separate sweep instead of ``gate_sweep.py --points``: that harness is
keyed on numeric axis points (``margin_0.10``) whose overrides it derives from
the point id, and its acceptance rule is the E070 margin rule.  This one is
keyed on named arms (``consensus_2f``) whose overrides are a table, and its
pre-registered rule is the E071 one (pooled TSR >= +3 pp with a paired-delta CI
that excludes 0, and a paired dFAR CI upper bound below +3 pp).  Everything
else -- the child process, the provenance stamps, the guards, the estimator --
is deliberately the same code, so the two sweeps stay comparable.

Usage::

    python scripts/consensus_sweep.py                              # all 4 arms
    python scripts/consensus_sweep.py --arms consensus_off,consensus_2f
    python scripts/consensus_sweep.py --run-only dancetrack0000 --arms consensus_off,consensus_2f
    python scripts/consensus_sweep.py --force

Each (sequence, arm) pair runs in its own child process (this file, re-entered
through ``CONSENSUSSWEEP_CHILD_SEQUENCE``) so one bad pair costs one pair and
not the sweep; the child pins ``run_target.build_variants`` to exactly one
variant and asserts the resulting ``specs.json`` carries the intended
overrides.
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
OUT_ROOT = ROOT / "results" / "consensus_sweep"
REPORT_JSON = OUT_ROOT / "summary.json"
SPECS_JSON = OUT_ROOT / "specs.json"
REPORT_MD = OUT_ROOT / "REPORT.md"

#: FROZEN dev split (E070).  Never swept, never reported.
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}

#: Protocol fixed by E070 -- do not vary it here, or the numbers stop being
#: comparable with results/gate_sweep/report.json.
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

#: The pre-registered rule (task statement, docs/E071_PLAN.md M1).
TSR_SHIP_PP = 3.0
FAR_MAX_PP = 3.0

CHILD_ENV = "CONSENSUSSWEEP_CHILD_SEQUENCE"

#: arm id -> the exact ``Config`` overrides plus the variant name that carries
#: them.  ``consensus_off`` is the reference: it spells out the shipped value
#: (``commit_consensus=False``) so the table is self-explanatory.
#:
#: NOTE ``consensus_2f`` and ``consensus_2f_pool`` are configurationally
#: identical: the shipped default of ``cost.consensus_pool`` is True, so the
#: pool arm's explicit ``True`` changes nothing.  They are kept because the
#: task fixes both names, and their agreement is a free determinism check on
#: the whole harness -- if the two rows differ, something is non-deterministic.
ARMS: dict[str, dict[str, Any]] = {
    "consensus_off": {
        "overrides": {"cost.commit_consensus": False},
        "variant": "consensus_off",
        "reference": True,
        "label": "consensus off (shipped handback rule)",
        "mechanism": "none",
    },
    "consensus_2f": {
        "overrides": {"cost.commit_consensus": True, "cost.consensus_frames": 2},
        "variant": "consensus_2f",
        "reference": False,
        "label": "two consecutive consistent proposals",
        "mechanism": "vote 2 frames",
    },
    "consensus_3f": {
        "overrides": {"cost.commit_consensus": True, "cost.consensus_frames": 3},
        "variant": "consensus_3f",
        "reference": False,
        "label": "three consecutive consistent proposals",
        "mechanism": "vote 3 frames",
    },
    "consensus_2f_pool": {
        "overrides": {
            "cost.commit_consensus": True,
            "cost.consensus_frames": 2,
            "cost.consensus_pool": True,
        },
        "variant": "consensus_2f_pool",
        "reference": False,
        "label": "two frames + pooled chain descriptor",
        "mechanism": "vote 2 frames, pool on",
    },
}

ARM_ORDER = ["consensus_off", "consensus_2f", "consensus_3f", "consensus_2f_pool"]
REFERENCE_ARM = "consensus_off"


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def arm_spec(arm_id: str) -> dict[str, Any]:
    """Map an arm id to (overrides, variant, label).  Pure function, no state."""
    if arm_id not in ARMS:
        raise SystemExit("[consensus_sweep] unknown arm %r; known: %s" % (arm_id, ARM_ORDER))
    spec = dict(ARMS[arm_id])
    spec["arm"] = arm_id
    spec["group"] = "consensus"
    return spec


def all_arms(which: str) -> list[str]:
    if which.strip().lower() in ("", "all"):
        return list(ARM_ORDER)
    ids = [a.strip() for a in which.split(",") if a.strip()]
    for a in ids:
        arm_spec(a)  # validate
    return ids


# --------------------------------------------------------------------------- #
# Child mode: one sequence, one pinned arm, no grid.
# --------------------------------------------------------------------------- #
def _run_child() -> int:
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - late import needs sys.path

    seq = os.environ[CHILD_ENV]
    spec = arm_spec(os.environ["CONSENSUSSWEEP_ARM"])
    overrides = json.loads(os.environ.get("CONSENSUSSWEEP_OVERRIDES") or "{}")
    variant_name = os.environ["CONSENSUSSWEEP_VARIANT"]
    if overrides != spec["overrides"]:
        raise SystemExit("[consensus_sweep] child override mismatch: %r vs %r"
                         % (overrides, spec["overrides"]))

    def _pinned(groups: Any = None, **kwargs: Any) -> list[Any]:
        """Exactly one variant: the arm's overrides on top of the shipped config."""
        return [rt.Variant(variant_name, "consensus_sweep", dict(overrides),
                           "E071 sweep arm " + spec["arm"])]

    rt.build_variants = _pinned  # noqa: WPS437 - process-local, file untouched

    specs_path = Path(os.environ["CONSENSUSSWEEP_SPECS"])
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
        # A sweep would put >1 spec here.  One sequence/mask/arm == one row.
        if len(rows) != 1:
            raise SystemExit("[consensus_sweep] GUARD: expected exactly 1 spec, got %d" % len(rows))
        if rows[0]["overrides"] != overrides or rows[0]["variant"] != variant_name:
            raise SystemExit("[consensus_sweep] GUARD: spec carries %r / %r, intended %r / %r"
                             % (rows[0]["overrides"], rows[0]["variant"], overrides, variant_name))
        return _orig_run_grid(specs, jobs=jobs, quiet=quiet)

    rt.run_grid = _checked_run_grid

    out_dir = os.environ["CONSENSUSSWEEP_OUT"]
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", PROTOCOL["mask"],
        "--max-frames", str(PROTOCOL["max_frames"]),
        "--subsample", str(PROTOCOL["subsample"]),
        "--jobs", str(PROTOCOL["jobs"]),
        # Recorded, never used for selection: build_variants is pinned above.
        "--ablate", "consensus_sweep",
    ]
    if os.environ.get("CONSENSUSSWEEP_TARGET_ID"):
        argv += ["--target-id", os.environ["CONSENSUSSWEEP_TARGET_ID"]]
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise SystemExit("[consensus_sweep] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    present = [
        name for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    seqs = [name for name in present if name not in DEV_SEQUENCES]
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[consensus_sweep] FATAL: dev sequence(s) %s in held-out selection" % leaked)
    if not seqs:
        raise SystemExit("[consensus_sweep] FATAL: no sequences under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats (same estimator as scripts/gate_sweep.py, for comparability)
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


def paired_bootstrap(arm_rows: Sequence[dict[str, Any]], ref_rows: Sequence[dict[str, Any]], *,
                     key: str, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap the *paired* delta of one metric: same sequence, both arms.

    The unit is still the sequence (the frame-weighted pool resamples sequences),
    but the statistic is now a difference of two paired measurements, which
    removes the sequence difficulty from the interval: a sequence that is hard
    for both arms contributes a small delta, not a large variance.
    """
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
            {"sequence": seq, "delta": round(d, 9), "arm_value": round(v, 9), "reference_value": round(ref[seq], 9),
             "delta_pp": round(d * 100.0, 3)}
            for seq, d, v in pairs
        ],
    }


# --------------------------------------------------------------------------- #
# Running one (arm, sequence)
# --------------------------------------------------------------------------- #
def run_pair(arm_id: str, seq: str, *, target_id: int | None, force: bool, log_dir: Path) -> dict[str, Any]:
    spec = arm_spec(arm_id)
    out_dir = OUT_ROOT / "runs" / arm_id / seq
    summary_path = out_dir / "summary.json"
    rec_path = OUT_ROOT / "summaries" / arm_id / ("%s.json" % seq)
    log_path = log_dir / ("%s__%s.log" % (arm_id, seq))
    specs_path = out_dir / "specs.json"

    base: dict[str, Any] = {
        "arm": arm_id,
        "variant": spec["variant"],
        "overrides": spec["overrides"],
        "label": spec["label"],
        "mechanism": spec["mechanism"],
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
        # PROVENANCE GUARD: a run dir without this sweep's ``.mine`` stamp cannot
        # be attributed to this protocol, so it is re-run rather than reported.
        print("[consensus_sweep] PURGE foreign run dir (no .mine provenance): %s/%s" % (arm_id, seq),
              flush=True)

    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["CONSENSUSSWEEP_ARM"] = arm_id
    env["CONSENSUSSWEEP_VARIANT"] = spec["variant"]
    env["CONSENSUSSWEEP_OVERRIDES"] = json.dumps(spec["overrides"])
    env["CONSENSUSSWEEP_OUT"] = str(out_dir)
    env["CONSENSUSSWEEP_SPECS"] = str(specs_path)
    env["PYTHONIOENCODING"] = "utf-8"
    if target_id is not None:
        env["CONSENSUSSWEEP_TARGET_ID"] = str(int(target_id))
    else:
        env.pop("CONSENSUSSWEEP_TARGET_ID", None)

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
            fh.write("# arm=%s seq=%s attempt=%d\n# env=%s\n" % (
                arm_id, seq, attempt,
                json.dumps({k: v for k, v in env.items() if k.startswith("CONSENSUSSWEEP_")}, sort_keys=True)))
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
        # One retry only, for infrastructure flakiness.  Every attempt is kept in
        # `attempts`, so a retry cannot launder a real failure into a success: a
        # deterministic error fails twice.

    rec.update(base)
    rec["attempts"] = attempts
    rec["from_cache"] = False
    if rec.get("status") != "INCLUDED":
        rec["log_tail"] = _tail_lines(log_path, 8)

    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    if rec.get("status") == "INCLUDED":
        owner_path.write_text(json.dumps({
            "arm": arm_id,
            "sequence": seq,
            "variant": spec["variant"],
            "overrides": spec["overrides"],
            "pid": os.getpid(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": rec.get("wall_seconds"),
            "attempt": attempts[-1]["attempt"] if attempts else None,
        }, indent=2) + "\n", encoding="utf-8")
    return rec


_CHALLENGE_KEYS = ("challenge_opened", "challenge_committed", "challenge_abandoned",
                   "challenge_restarted", "challenge_pending_frames")

def _challenge_counters(stats: dict[str, Any] | None) -> dict[str, Any]:
    """The E071 counters, out of ``TargetTracker.stats()``.

    ``run_target`` writes ``tracker_stats`` on the full run record (``per_run.json``)
    and the E071 wiring added the ``challenge_*`` counters there, so they survive
    into the harness output without any extra plumbing.  ``headline`` rows in
    ``summary.json`` are pooled copies that drop the nested dict, which is why the
    caller hands us the record from ``per_run.json`` rather than the headline row.
    Missing counters are reported as ``-1``, not as a silent zero: "never fired"
    and "never ran" must stay distinguishable.
    """
    if not isinstance(stats, dict):
        return {"available": False, "challenge_enabled": None, "n_opened": -1, "n_committed": -1,
                "n_abandoned": -1, "n_restarted": -1, "n_pending_frames": -1}
    out: dict[str, Any] = {"available": True, "challenge_enabled": stats.get("challenge_enabled")}
    for key in _CHALLENGE_KEYS:
        out["n_" + key[len("challenge_"):]] = int(stats.get(key, 0) or 0)
    return out


def _record_stats(summary_path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    """``tracker_stats`` of the single run this summary covers.

    ``summary.json``'s headline row is a pooled copy and does not carry the
    nested ``tracker_stats`` dict; ``per_run.json`` next to it keeps the full
    record (``runs`` has exactly one element, because the child pins one
    variant).  Both are tried so the sweep still reports counters if a future
    harness starts copying them into the headline.
    """
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


def _parse_summary(summary_path: Path, seq: str, forced_id: int | None, specs_path: Path) -> dict[str, Any]:
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
            print("[consensus_sweep] WARNING %s: salvaged summary.json, ignored %d trailing chars (%s)"
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

    # Protocol guards: the child must have run the E070 protocol.
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
    counters = _challenge_counters(_record_stats(summary_path, row))
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
        "ms_per_frame": float(row.get("ms_per_frame", float("nan"))),
        "target_id": target_id,
        "target_reason": decision.get("reason"),
        "guard_passed": decision.get("guard_passed"),
        "harness_wall_seconds": meta.get("wall_seconds"),
        "n_events": int(row.get("n_events", -1) or -1),
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
_METRICS = ("tsr", "far", "id_consistency", "ms_per_frame")


def pool_arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inc = [r for r in rows if r.get("status") == "INCLUDED"]
    exc = [r for r in rows if r.get("status") != "INCLUDED"]
    if not inc:
        return {"included": [], "excluded": exc, "pooled": None, "bootstrap": None,
                "engagement": None, "frame_counts": None}
    weights = [float(r["n_report_frames"]) for r in inc]
    pooled: dict[str, Any] = {
        "total_report_frames": int(sum(weights)),
        "n_included": len(inc),
        "n_excluded": len(exc),
        "estimator": "frame-weighted pooled mean over included sequences",
    }
    for key in _METRICS:
        vals = [float(r[key]) for r in inc]
        pooled[key] = round(_weighted(vals, weights), 6)
        pooled["%s_mean" % key] = round(sum(vals) / len(inc), 6)
        pooled["%s_median" % key] = round(_pct(vals, [50.0])[0], 6)
        pooled["%s_min" % key] = round(min(vals), 6)
        pooled["%s_max" % key] = round(max(vals), 6)
    # Mechanism engagement: a rule that never fires cannot have caused a delta.
    eng = {name: 0 for name in ("n_opened", "n_committed", "n_abandoned", "n_restarted",
                                "n_pending_frames")}
    eng["n_sequences_with_any_challenge"] = 0
    eng["n_sequences_with_a_commit"] = 0
    eng["available"] = bool(inc) and bool(inc[0].get("available", False))
    for r in inc:
        eng["n_opened"] += int(r.get("n_opened", 0) or 0)
        eng["n_committed"] += int(r.get("n_committed", 0) or 0)
        eng["n_abandoned"] += int(r.get("n_abandoned", 0) or 0)
        eng["n_restarted"] += int(r.get("n_restarted", 0) or 0)
        eng["n_pending_frames"] += int(r.get("n_pending_frames", 0) or 0)
        if int(r.get("n_opened", 0) or 0) > 0 or int(r.get("n_pending_frames", 0) or 0) > 0:
            eng["n_sequences_with_any_challenge"] += 1
        if int(r.get("n_committed", 0) or 0) > 0:
            eng["n_sequences_with_a_commit"] += 1
    return {
        "included": inc,
        "excluded": exc,
        "pooled": pooled,
        "bootstrap": {key: cluster_bootstrap(inc, key=key) for key in _METRICS},
        "engagement": eng,
        "frame_counts": {
            "n_sequences": len(inc),
            "n_report_frames": int(sum(weights)),
            "n_frames": int(sum(int(r["n_frames"]) for r in inc)),
        },
    }


def verdict(arm_entry: dict[str, Any], ref_entry: dict[str, Any]) -> dict[str, Any]:
    """The E071 rule, applied verbatim and never re-tuned.

    SHIP iff pooled TSR rises by >= +3 pp AND the 95% CI of the *paired* TSR
    delta excludes 0 AND the upper bound of the paired dFAR CI is < +3 pp.
    KILL if the pooled TSR delta is <= 0 (falls, or the mechanism is inert).
    Anything else is NO-SHIP -- it did not clear the bar, which is not the same
    as being refuted.
    """
    out: dict[str, Any] = {
        "arm": arm_entry["arm"],
        "rule": ("SHIP iff pooled dTSR >= +%.1f pp AND paired-dTSR 95%% CI excludes 0 AND "
                 "paired-dFAR 95%% CI upper bound < +%.1f pp; KILL if pooled dTSR <= 0" % (TSR_SHIP_PP, FAR_MAX_PP)),
    }
    if arm_entry["arm"] == ref_entry["arm"]:
        out.update(verdict="REFERENCE", text="Reference arm: the shipped handback rule.")
        return out
    if not arm_entry.get("pooled") or not ref_entry.get("pooled"):
        out.update(verdict="NO-SHIP", text="No complete pool (excluded/error runs); cannot clear the rule.")
        return out

    pooled = arm_entry["pooled"]
    ref_pooled = ref_entry["pooled"]
    d_tsr = (pooled["tsr"] - ref_pooled["tsr"]) * 100.0
    d_far = (pooled["far"] - ref_pooled["far"]) * 100.0
    d_id = (pooled["id_consistency"] - ref_pooled["id_consistency"]) * 100.0
    d_ms = pooled["ms_per_frame"] - ref_pooled["ms_per_frame"]
    pb = arm_entry["paired"]
    p_tsr = pb["tsr"]
    p_far = pb["far"]
    tsr_lo = (p_tsr["p2.5"] or 0.0) * 100.0
    tsr_hi = (p_tsr["p97.5"] or 0.0) * 100.0
    far_lo = (p_far["p2.5"] or 0.0) * 100.0
    far_hi = (p_far["p97.5"] or 0.0) * 100.0
    eng = arm_entry.get("engagement") or {}
    inert = (int(eng.get("n_pending_frames", 0) or 0) == 0
             and int(eng.get("n_committed", 0) or 0) == 0)

    ok_tsr_mag = d_tsr >= TSR_SHIP_PP
    ok_tsr_ci = bool(p_tsr.get("excludes_zero")) and tsr_lo > 0.0
    ok_far = far_hi < FAR_MAX_PP
    ship = bool(ok_tsr_mag and ok_tsr_ci and ok_far)
    if d_tsr <= 0.0:
        v = "KILL"
        why = ("pooled TSR does not rise (%+.2f pp)" % d_tsr
               + ("; the mechanism never engaged" if inert else ""))
    elif ship:
        v = "SHIP"
        why = ("pooled TSR %+.2f pp (>= +%.1f), paired CI [%+.2f, %+.2f] pp excludes 0, "
               "paired dFAR CI upper %+.2f pp < +%.1f" % (d_tsr, TSR_SHIP_PP, tsr_lo, tsr_hi, far_hi, FAR_MAX_PP))
    else:
        v = "NO-SHIP"
        bad = []
        if not ok_tsr_mag:
            bad.append("pooled dTSR %+.2f pp < +%.1f" % (d_tsr, TSR_SHIP_PP))
        if not ok_tsr_ci:
            bad.append("paired dTSR CI [%+.2f, %+.2f] pp includes 0" % (tsr_lo, tsr_hi))
        if not ok_far:
            bad.append("paired dFAR CI upper %+.2f pp >= +%.1f" % (far_hi, FAR_MAX_PP))
        why = "; ".join(bad)
    out.update(
        verdict=v,
        text=("%s: %s." % (v, why)),
        pooled_dtsr_pp=round(d_tsr, 3),
        pooled_dfar_pp=round(d_far, 3),
        pooled_did_consistency_pp=round(d_id, 3),
        pooled_dms_per_frame=round(d_ms, 3),
        paired_dtsr_mean_pp=round((p_tsr["delta_mean"] or 0.0) * 100.0, 3),
        paired_dtsr_ci_pp=[round(tsr_lo, 3), round(tsr_hi, 3)],
        paired_dfar_mean_pp=round((p_far["delta_mean"] or 0.0) * 100.0, 3),
        paired_dfar_ci_pp=[round(far_lo, 3), round(far_hi, 3)],
        paired_dtsr_excludes_zero=bool(p_tsr.get("excludes_zero")),
        paired_dfar_excludes_zero=bool(p_far.get("excludes_zero")),
        tsr_magnitude_ok=bool(ok_tsr_mag),
        tsr_ci_ok=bool(ok_tsr_ci),
        far_ci_ok=bool(ok_far),
        mechanism_inert=bool(inert),
        sequences_up=int(p_tsr["sequences_up"]),
        sequences_down=int(p_tsr["sequences_down"]),
        sequences_flat=int(p_tsr["sequences_flat"]),
    )
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def code_state_snapshot() -> dict[str, Any]:
    p = OUT_ROOT / "code_snapshot_start.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _fmt_ci(val: float | None, boot: dict[str, Any] | None, digits: int = 4) -> str:
    if val is None:
        return "--"
    if not boot:
        return ("%%.%df" % digits) % val
    return ("%%.%df [%%.%df, %%.%df]" % (digits, digits, digits)) % (val, boot["p2.5"], boot["p97.5"])


def write_markdown(payload: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    arms = payload["arms"]
    order = payload["order"]
    ref = arms[REFERENCE_ARM]
    a("# E071 -- temporal consensus on the handback, held-out sweep")
    a("")
    a("<!-- generated by scripts/consensus_sweep.py; every number is read from results/consensus_sweep/summary.json -->")
    a("")
    a("## Verdict (pre-registered rule)")
    a("")
    a("> SHIP only if pooled TSR rises by **>= +3 pp** with the 95% CI of the **paired** delta excluding 0, "
      "**and** the upper bound of the paired dFAR CI is **< +3 pp**. KILL if TSR is unchanged (inert) or falls.")
    a("")
    for arm_id in order:
        entry = arms[arm_id]
        v = entry["verdict"]
        if arm_id == REFERENCE_ARM:
            a("- **`%s`** -- %s (reference, not a candidate)" % (arm_id, v["text"]))
            continue
        a("- **`%s`: %s** -- %s" % (arm_id, v["verdict"], v["text"]))
    a("")
    shipped = [k for k in order if k != REFERENCE_ARM and arms[k]["verdict"]["verdict"] == "SHIP"]
    a("**Ship decision: %s**" % ("ship `%s`" % ", ".join("`%s`" % s for s in shipped)
                                if shipped else "ship nothing; the shipped default (consensus off) stays."))
    a("")
    a("## Protocol (identical to E070, so the two sweeps are comparable)")
    a("")
    a("- data: `data/raw/dancetrack/val`, the **%d** held-out sequences; frozen dev split %s asserted absent"
      % (len(payload["heldout_sequences"]), ", ".join(sorted(DEV_SEQUENCES))))
    a("- `--max-frames %d --subsample %d --mask %s --jobs %d`, shipped defaults elsewhere"
      % (PROTOCOL["max_frames"], PROTOCOL["subsample"], PROTOCOL["mask"], PROTOCOL["jobs"]))
    a("- one `--ablate consensus` arm per run, `run_target.build_variants` replaced in the child; "
      "`specs.json` asserts exactly 1 spec with exactly the intended overrides and each `summary.json` "
      "reports exactly 1 variant")
    a("- target identity: the `consensus_off` pass picks the target with the shipped strategy; every other "
      "arm is pinned to that `--target-id` and each run's own `meta.target_ids` is asserted against it")
    a("- pooling: frame-weighted over `n_report_frames`; 95%% CI = cluster bootstrap over sequences, "
      "%d reps, seed %d" % (BOOTSTRAP_REPS, BOOTSTRAP_SEED))
    a("- one `consensus_off` run per sequence is the paired reference for every delta below")
    a("")
    a("## What was actually run")
    a("")
    a("- pairs run: **%d** (%d arms x %d sequences); pairs included: **%d**, excluded: **%d**, errors: **%d**"
      % (payload["n_pairs_run"], len(order), len(payload["sequences_run"]),
         payload["n_pairs_included"], payload["n_pairs_excluded"], payload["n_pairs_errors"]))
    a("- sequences: **%d** of %d held-out; frames per arm (sum of `n_report_frames`): %s"
      % (len(payload["sequences_run"]), len(payload["heldout_sequences"]),
         ", ".join("`%s`=%d" % (k, (arms[k]["frame_counts"] or {}).get("n_report_frames") or 0) for k in order)))
    a("- total wall: **%.1f s** (%.1f min); summed per-pair wall %.1f min"
      % (payload["wall_seconds"], payload["wall_seconds"] / 60.0,
         (payload.get("wall_seconds_pair_sum") or 0.0) / 60.0))
    a("- exact command: `%s`" % payload["cli"]["command"])
    a("")
    a("## Results per arm")
    a("")
    a("| arm | frames | TSR [95% CI] | FAR [95% CI] | id_consistency [95% CI] | ms/frame | n incl |")
    a("|---|---|---|---|---|---|---|")
    for arm_id in order:
        e = arms[arm_id]
        p = e["pooled"]
        if not p:
            a("| `%s` | 0 | -- | -- | -- | -- | 0 |" % arm_id)
            continue
        b = e["bootstrap"]
        a("| `%s` | %d | %s | %s | %s | %.2f | %d |"
          % (arm_id, p["total_report_frames"],
             _fmt_ci(p["tsr"], b["tsr"]), _fmt_ci(p["far"], b["far"]),
             _fmt_ci(p["id_consistency"], b["id_consistency"]), p["ms_per_frame"], p["n_included"]))
    a("")
    a("## Paired deltas vs `%s`" % REFERENCE_ARM)
    a("")
    a("Every delta is computed on the *same* sequence in both arms, so sequence difficulty cancels. "
      "The CI resamples sequences (2000 reps, seed 0) and takes the mean of the paired per-sequence deltas.")
    a("")
    a("| arm | dTSR pp [95% CI] | sequences up/down/flat | dFAR pp [95% CI] | d id_consistency pp | d ms/frame | n paired |")
    a("|---|---|---|---|---|---|---|")
    for arm_id in order:
        if arm_id == REFERENCE_ARM:
            continue
        v = arms[arm_id]["verdict"]
        if "paired_dtsr_ci_pp" not in v:
            a("| `%s` | -- | -- | -- | -- | -- | 0 |" % arm_id)
            continue
        a("| `%s` | %+.2f [%+.2f, %+.2f] | %d/%d/%d | %+.2f [%+.2f, %+.2f] | %+.2f | %+.2f | %d |"
          % (arm_id, v["pooled_dtsr_pp"], v["paired_dtsr_ci_pp"][0], v["paired_dtsr_ci_pp"][1],
             v["sequences_up"], v["sequences_down"], v["sequences_flat"],
             v["pooled_dfar_pp"], v["paired_dfar_ci_pp"][0], v["paired_dfar_ci_pp"][1],
             v["pooled_did_consistency_pp"], v["pooled_dms_per_frame"],
             (arms[arm_id]["paired"]["tsr"]["n_paired"] if arms[arm_id].get("paired") else 0)))
    a("")
    a("## Did the mechanism engage?")
    a("")
    a("A vote that never fires is indistinguishable from a no-op, so the counters are reported before the "
      "metrics are read. `n_pending_frames` counts frames a candidate spent waiting for confirmation; "
      "`n_committed` counts handovers the vote actually authorised.")
    a("")
    a("| arm | n_opened | n_committed | n_pending_frames | n_abandoned | n_restarted | sequences with any challenge | sequences with a commit |")
    a("|---|---|---|---|---|---|---|---|")
    for arm_id in order:
        eng = arms[arm_id].get("engagement")
        if not eng:
            a("| `%s` | -- | -- | -- | -- | -- | -- | -- |" % arm_id)
            continue
        a("| `%s` | %d | %d | %d | %d | %d | %d | %d |"
          % (arm_id, eng["n_opened"], eng["n_committed"], eng["n_pending_frames"],
             eng["n_abandoned"], eng["n_restarted"],
             eng["n_sequences_with_any_challenge"], eng["n_sequences_with_a_commit"]))
    a("")
    a("## Per-sequence TSR")
    a("")
    a("| sequence | " + " | ".join("%s TSR/FAR" % arm for arm in order) + " |")
    a("|---" * (len(order) + 1) + "|")
    for seq in payload["sequences_run"]:
        cells = []
        for arm_id in order:
            row = next((r for r in arms[arm_id]["rows"] if r["sequence"] == seq), None)
            if row and row.get("status") == "INCLUDED":
                cells.append("%.3f / %.3f" % (row["tsr"], row["far"]))
            else:
                cells.append("EXCL" if row else "--")
        a("| `%s` | %s |" % (seq, " | ".join(cells)))
    a("")
    a("## Sequences that moved")
    a("")
    any_moved = False
    for arm_id in order:
        if arm_id == REFERENCE_ARM or not arms[arm_id].get("paired"):
            continue
        moved = [r for r in arms[arm_id]["paired"]["tsr"]["per_sequence"] if abs(r["delta_pp"]) > 5.0]
        if not moved:
            continue
        any_moved = True
        a("**`%s`** -- %d of %d paired sequences move TSR by more than 5 pp:" % (arm_id, len(moved),
                                                                                arms[arm_id]["paired"]["tsr"]["n_paired"]))
        a("")
        for row in sorted(moved, key=lambda r: -abs(r["delta_pp"])):
            a("- `%s`: %+.2f pp (%.3f -> %.3f)" % (row["sequence"], row["delta_pp"],
                                                   row["reference_value"], row["arm_value"]))
        a("")
    if not any_moved:
        a("None: no sequence moves its TSR by more than 5 pp in any arm.")
        a("")
    a("## Exclusions and errors")
    a("")
    n_bad = payload["n_pairs_excluded"] + payload["n_pairs_errors"]
    if n_bad:
        for arm_id in order:
            for r in arms[arm_id]["excluded"]:
                a("- `%s` @ `%s`: %s -- %s" % (r["sequence"], arm_id, r["reason"], r.get("harness_message") or ""))
    else:
        a("None: every (sequence, arm) pair produced an observable target; the denominators are %d in every pool."
          % len(payload["sequences_run"]))
    a("")
    a("## Provenance")
    a("")
    a("- one parent process at a time: `results/consensus_sweep/.sweep.lock` (live-pid checked)")
    a("- every included pair carries `runs/<arm>/<seq>/.mine`; a run dir without that stamp is purged and re-run")
    a("- a `summary.json` with trailing garbage is salvaged from its first JSON document and flagged "
      "`json_recovered` in the pair record instead of aborting the sweep")
    cs = payload.get("code_state") or {}
    if cs:
        a("")
        a("| file | sha256[:16] | mtime |")
        a("|---|---|---|")
        for f in sorted(cs):
            a("| `%s` | %s | %s |" % (f, cs[f]["sha256"], cs[f]["mtime"]))
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
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return ("%d" % pid) in (out.stdout or "")


def acquire_single_instance_lock() -> bool:
    """Take ``results/consensus_sweep/.sweep.lock``; False if a live sweep holds it.

    One GPU, one worker: two concurrent sweeps previously produced a summary.json
    with a complete JSON document followed by a stray brace (two writers on the
    same path), so a second instance must exit before it touches anything.
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
            print("[consensus_sweep] REFUSING TO START: %s is held by live sweep pid %d." % (lock_path, holder),
                  flush=True)
            return False
        print("[consensus_sweep] stale lock from dead pid %d -- taking over" % holder, flush=True)
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

    p = argparse.ArgumentParser(description="E071 temporal-consensus sweep on the held-out split.")
    p.add_argument("--arms", default="all", help="'all' or comma-separated arm ids: %s" % ",".join(ARM_ORDER))
    p.add_argument("--limit-seqs", type=int, default=None, help="debug: only the first N held-out sequences")
    p.add_argument("--force", action="store_true", help="re-run pairs that already have a summary")
    p.add_argument("--run-only", default=None, help="debug: restrict to one sequence name")
    args = p.parse_args(list(argv) if argv is not None else None)

    print("[consensus_sweep] FROZEN DEV SPLIT (selection set -- NEVER swept, never reported):", flush=True)
    print("[consensus_sweep]   DanceTrack: %s" % ", ".join(sorted(DEV_SEQUENCES)), flush=True)

    all_seqs = heldout_sequences(VAL_ROOT)
    print("[consensus_sweep] held-out sequences: %d" % len(all_seqs), flush=True)
    print("[consensus_sweep] %s" % ", ".join(all_seqs), flush=True)

    seqs = list(all_seqs)
    partial = False
    if args.run_only:
        if args.run_only not in all_seqs:
            raise SystemExit("[consensus_sweep] FATAL: %r is not a held-out sequence" % args.run_only)
        seqs = [args.run_only]
        partial = True
    if args.limit_seqs:
        seqs = seqs[: int(args.limit_seqs)]
        partial = len(seqs) < len(all_seqs)

    # HARD GUARD: no dev sequence may appear anywhere in the run set.
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[consensus_sweep] FATAL: dev sequence(s) %s in the run set; refusing." % leaked)
    assert not (DEV_SEQUENCES & set(seqs)), "dev split leaked into the sweep"

    arms = all_arms(args.arms)
    # The reference arm must run first: it picks the target every other arm is pinned to.
    if REFERENCE_ARM in arms:
        arms = [REFERENCE_ARM] + [x for x in arms if x != REFERENCE_ARM]
    elif any(not arm_spec(x)["reference"] for x in arms):
        raise SystemExit("[consensus_sweep] the reference arm %s must be in --arms to pin target ids" % REFERENCE_ARM)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SPECS_JSON.write_text(json.dumps({
        "generated_by": "scripts/consensus_sweep.py",
        "experiment": "E071",
        "protocol": dict(PROTOCOL),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "data_root": str(VAL_ROOT.relative_to(ROOT)),
        "dev_sequences": sorted(DEV_SEQUENCES),
        "arms": {arm_id: {"variant": arm_spec(arm_id)["variant"],
                          "overrides": arm_spec(arm_id)["overrides"],
                          "label": arm_spec(arm_id)["label"],
                          "mechanism": arm_spec(arm_id)["mechanism"]}
                 for arm_id in ARM_ORDER},
        "note": ("consensus_2f and consensus_2f_pool are configurationally identical because "
                 "cost.consensus_pool ships as True; their agreement is a determinism check."),
    }, indent=2) + "\n", encoding="utf-8")

    log_dir = OUT_ROOT / "logs"
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    target_ids: dict[str, int] = {}

    # Pass 1: reference arm (auto-pick).  Pass 2: every other arm, target pinned.
    for arm_id in arms:
        spec = arm_spec(arm_id)
        for i, seq in enumerate(seqs, 1):
            tid = None if spec["reference"] else target_ids.get(seq)
            if not spec["reference"] and tid is None:
                raise SystemExit("[consensus_sweep] no reference target id for %s; run the reference arm first" % seq)
            ts = time.perf_counter()
            rec = run_pair(arm_id, seq, target_id=tid, force=bool(args.force), log_dir=log_dir)
            rows.append(rec)
            if spec["reference"] and rec.get("status") == "INCLUDED":
                target_ids[seq] = int(rec["target_id"])
            print("[consensus_sweep] %-16s %2d/%2d %-16s %-8s %6.1fs %s"
                  % (arm_id, i, len(seqs), seq, rec.get("status"), time.perf_counter() - ts,
                     ("TSR=%.3f FAR=%.3f tid=%s pend=%s committed=%s"
                      % (rec["tsr"], rec["far"], rec.get("target_id"),
                         rec.get("n_pending_frames"), rec.get("n_committed")))
                     if rec.get("status") == "INCLUDED" else rec.get("reason", "")),
                  flush=True)
    wall = time.perf_counter() - t0

    arms_payload: dict[str, Any] = {}
    for arm_id in arms:
        spec = arm_spec(arm_id)
        arm_rows = [r for r in rows if r["arm"] == arm_id]
        pool = pool_arm(arm_rows)
        arms_payload[arm_id] = {
            "arm": arm_id,
            "variant": spec["variant"],
            "overrides": spec["overrides"],
            "label": spec["label"],
            "mechanism": spec["mechanism"],
            "is_reference": bool(spec["reference"]),
            "rows": arm_rows,
            "included": pool["included"],
            "excluded": pool["excluded"],
            "pooled": pool["pooled"],
            "bootstrap": pool["bootstrap"],
            "engagement": pool["engagement"],
            "frame_counts": pool["frame_counts"],
        }

    ref_entry = arms_payload.get(REFERENCE_ARM)
    if ref_entry is None:
        raise SystemExit("[consensus_sweep] reference arm missing from this run; cannot apply the rule")

    for arm_id in arms:
        entry = arms_payload[arm_id]
        entry["paired"] = {key: paired_bootstrap(entry["rows"], ref_entry["rows"], key=key)
                           for key in _METRICS}
        entry["verdict"] = verdict(entry, ref_entry)

    n_included = sum(1 for r in rows if r.get("status") == "INCLUDED")
    payload: dict[str, Any] = {
        "generated_by": "scripts/consensus_sweep.py",
        "experiment": "E071",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "heldout_sequences": all_seqs,
        "sequences_run": seqs,
        "partial": bool(partial),
        "n_pairs_run": len(rows),
        "n_pairs_included": n_included,
        "n_pairs_excluded": sum(1 for r in rows if r.get("status") == "EXCLUDED"),
        "n_pairs_errors": sum(1 for r in rows if r.get("status") == "ERROR"),
        "protocol": dict(PROTOCOL),
        "acceptance_rule": {
            "source": "task statement / docs/E071_PLAN.md",
            "text": ("SHIP only if pooled TSR rises by >= +3 pp with the 95% CI of the paired delta excluding 0, "
                     "and the upper bound of the paired dFAR CI is < +3 pp. KILL if TSR is unchanged (inert) or falls."),
            "tsr_ship_pp": TSR_SHIP_PP,
            "far_max_pp": FAR_MAX_PP,
            "statistic": "mean paired per-sequence delta vs %s, cluster bootstrap over sequences" % REFERENCE_ARM,
        },
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "cli": {"arms": args.arms, "force": bool(args.force), "limit_seqs": args.limit_seqs,
                "run_only": args.run_only,
                "command": "python scripts/consensus_sweep.py --arms %s%s%s"
                           % (args.arms, " --run-only %s" % args.run_only if args.run_only else "",
                              " --limit-seqs %d" % args.limit_seqs if args.limit_seqs else "")},
        "order": arms,
        "arms": arms_payload,
        "target_ids": target_ids,
        "wall_seconds": round(wall, 1),
        "wall_seconds_pair_sum": round(sum(float(r.get("wall_seconds") or 0.0) for r in rows), 1),
        "code_state": code_state_snapshot(),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(write_markdown(payload), encoding="utf-8")

    for arm_id in arms:
        entry = arms_payload[arm_id]
        pooled = entry["pooled"]
        v = entry["verdict"]
        if pooled:
            b = entry["bootstrap"]
            eng = entry["engagement"]
            print("[consensus_sweep] %-16s TSR=%.4f [%.4f, %.4f] FAR=%.4f [%.4f, %.4f] ms=%.1f "
                  "opened=%d committed=%d pending=%d -> %s"
                  % (arm_id, pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
                     pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"], pooled["ms_per_frame"],
                     eng["n_opened"], eng["n_committed"], eng["n_pending_frames"], v["verdict"]), flush=True)
        else:
            print("[consensus_sweep] %-16s no usable pool -> %s" % (arm_id, v["verdict"]), flush=True)
    print("[consensus_sweep] wall=%.1fs wrote %s and %s"
          % (wall, REPORT_JSON.relative_to(ROOT), REPORT_MD.relative_to(ROOT)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
