"""E077 held-out sweep -- channel-conditional fusion (B) and bounded-aspect coasting (C).

Pre-registered in ``docs/E077_PLAN.md``; the ship rule there is applied verbatim and
is NOT re-tuned here.

Structure is deliberately a clone of ``scripts/consensus_sweep.py`` and
``scripts/gate_sweep.py`` so the numbers are comparable with E068/E070/E071:

* one child process per (arm, sequence), the child monkey-patches
  ``run_target.build_variants`` to exactly ONE variant and then calls
  ``run_target.main``; ``specs.json`` asserts that exactly one spec with exactly
  the intended overrides reached the grid;
* the reference arm (``channel_off``) runs first and picks the target identity
  with the shipped strategy; every other arm is pinned to that ``--target-id`` and
  each run's own ``meta.target_ids`` is asserted against it;
* protocol frozen at ``--max-frames 300 --subsample 2 --mask none --jobs 1`` on the
  22 held-out DanceTrack-val sequences, with the frozen dev split
  (0004/0005/0007) asserted absent;
* pooling is frame-weighted, 95 % CIs are a cluster bootstrap over sequences
  (2000 reps, seed 0), and every delta is PAIRED against ``channel_off`` on the
  same sequence.

An extra channel-specific output: each child is run with ``--debug-csv``, so the
per-frame state of every pair is on disk.  That is what makes the pre-registered
mechanism check possible -- the rule requires the ``blind AND present`` share to
FALL, and the share can only be counted from the per-frame dump.  The same dump
gives the box-level answer for arm C, decoded from the per-frame ``refuse_reason``
and state mix.

    python scripts/channel_sweep.py --arms all
    python scripts/channel_sweep.py --run-only dancetrack0010 --arms channel_both   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
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
OUT_ROOT = ROOT / "results" / "channel_sweep"
REPORT_JSON = OUT_ROOT / "summary.json"
SPECS_JSON = OUT_ROOT / "specs.json"
REPORT_MD = OUT_ROOT / "REPORT.md"

#: FROZEN dev split (E070).  Never swept, never reported.
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}

#: Protocol fixed by E070 -- do not vary it here, or the numbers stop being
#: comparable with results/gate_sweep/report.json and results/consensus_sweep.
PROTOCOL = {
    "max_frames": 300,
    "subsample": 2,
    "mask": "none",
    "jobs": 1,
    "detector": "yolo (shipped default)",
    "target_strategy": "stressed (shipped default)",
    "debug_csv": True,
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0

#: The pre-registered rule (docs/E077_PLAN.md, written before the run).
TSR_SHIP_PP = 3.0
FAR_MAX_PP = 3.0

CHILD_ENV = "CHANNELSWEEP_CHILD_SEQUENCE"

#: When set (env ``CHANNELSWEEP_SRC``), every child imports ``tgtrack`` from this
#: tree instead of ``<repo>/src``.  Used for the scratch phase only.
_SRC_OVERRIDE = os.environ.get("CHANNELSWEEP_SRC") or ""

#: The tracker's blind states (mirrors `tgtrack.evaluate.BLIND_STATES`; the copy is
#: only a fallback -- `_blind_states()` prefers the imported constant).
_BLIND_FALLBACK = ("occluded_soft", "lost_short", "lost_long", "out_of_frame", "gave_up", "reacquiring")


def _blind_states() -> set[str]:
    try:
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from tgtrack.evaluate import BLIND_STATES  # noqa: WPS433

        return {str(s) for s in BLIND_STATES}
    except Exception:  # pragma: no cover - the fallback keeps the driver usable
        return set(_BLIND_FALLBACK)


#: arm id -> the exact ``Config`` overrides plus the variant name that carries them.
#: ``channel_off`` is the reference: it spells out the shipped values (both arms
#: off) so the table is self-explanatory.  Nothing else is overridden: E077 adds
#: exactly two thresholds (`cost.channel_geom_tau`, `cost.channel_app_tau`) and
#: they are left at their measured defaults in every arm.
ARMS: dict[str, dict[str, Any]] = {
    "channel_off": {
        "overrides": {"cost.channel_accept": False, "motion.aspect_clamp_coast": False},
        "variant": "channel_off",
        "reference": True,
        "label": "reference: both E077 arms off (shipped behaviour)",
        "mechanism": "none",
    },
    "channel_fusion": {
        "overrides": {"cost.channel_accept": True},
        "variant": "channel_fusion",
        "reference": False,
        "label": "arm B: channel-conditional acceptance (OR-branch on the bound)",
        "mechanism": "geometry OR appearance may carry acceptance",
    },
    "channel_aspect": {
        "overrides": {"motion.aspect_clamp_coast": True},
        "variant": "channel_aspect",
        "reference": False,
        "label": "arm C: bounded-aspect coasting",
        "mechanism": "coasted aspect clamped to the last observed band",
    },
    "channel_both": {
        "overrides": {"cost.channel_accept": True, "motion.aspect_clamp_coast": True},
        "variant": "channel_both",
        "reference": False,
        "label": "arm D: B + C",
        "mechanism": "both",
    },
}

ARM_ORDER = ["channel_off", "channel_fusion", "channel_aspect", "channel_both"]
REFERENCE_ARM = "channel_off"


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def arm_spec(arm_id: str) -> dict[str, Any]:
    """Map an arm id to (overrides, variant, label).  Pure function, no state."""
    if arm_id not in ARMS:
        raise SystemExit("[channel_sweep] unknown arm %r; known: %s" % (arm_id, ARM_ORDER))
    spec = dict(ARMS[arm_id])
    spec["arm"] = arm_id
    spec["group"] = "channel"
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
    # Optional: run against a different `tgtrack` tree.  `run_target` inserts
    # `<repo>/src` at sys.path[0] on import, so the ONLY way to measure a candidate
    # tree is to import `tgtrack` from it first and let `sys.modules` serve
    # `run_target`'s imports.  Used for the scratch phase, where `src/` must not be
    # touched because another sweep is reading it; the shipped run uses `src`.
    src_env = os.environ.get("CHANNELSWEEP_SRC")
    if src_env:
        src_path = Path(src_env).resolve()
        sys.path.insert(0, str(src_path))
        import tgtrack as _tg  # noqa: WPS433

        _where = str(Path(_tg.__file__).resolve())
        if not _where.startswith(str(src_path)):
            raise SystemExit("[channel_sweep] FATAL: CHANNELSWEEP_SRC=%s but tgtrack came from %s"
                             % (src_path, _where))
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - late import needs sys.path

    seq = os.environ[CHILD_ENV]
    spec = arm_spec(os.environ["CHANNELSWEEP_ARM"])
    overrides = json.loads(os.environ.get("CHANNELSWEEP_OVERRIDES") or "{}")
    variant_name = os.environ["CHANNELSWEEP_VARIANT"]
    if overrides != spec["overrides"]:
        raise SystemExit("[channel_sweep] child override mismatch: %r vs %r"
                         % (overrides, spec["overrides"]))

    def _pinned(groups: Any = None, **kwargs: Any) -> list[Any]:
        """Exactly one variant: the arm's overrides on top of the shipped config."""
        return [rt.Variant(variant_name, "channel", dict(overrides),
                           "E077 sweep arm " + spec["arm"])]

    rt.build_variants = _pinned  # noqa: WPS437 - process-local, file untouched

    specs_path = Path(os.environ["CHANNELSWEEP_SPECS"])
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
            raise SystemExit("[channel_sweep] GUARD: expected exactly 1 spec, got %d" % len(rows))
        if rows[0]["overrides"] != overrides or rows[0]["variant"] != variant_name:
            raise SystemExit("[channel_sweep] GUARD: spec carries %r / %r, intended %r / %r"
                             % (rows[0]["overrides"], rows[0]["variant"], overrides, variant_name))
        return _orig_run_grid(specs, jobs=jobs, quiet=quiet)

    rt.run_grid = _checked_run_grid

    out_dir = os.environ["CHANNELSWEEP_OUT"]
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", PROTOCOL["mask"],
        "--max-frames", str(PROTOCOL["max_frames"]),
        "--subsample", str(PROTOCOL["subsample"]),
        "--jobs", str(PROTOCOL["jobs"]),
        # Recorded, never used for selection: build_variants is pinned above.
        "--ablate", "channel",
    ]
    if PROTOCOL["debug_csv"]:
        argv.append("--debug-csv")
    if os.environ.get("CHANNELSWEEP_TARGET_ID"):
        argv += ["--target-id", os.environ["CHANNELSWEEP_TARGET_ID"]]
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise SystemExit("[channel_sweep] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    present = [
        name for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    seqs = [name for name in present if name not in DEV_SEQUENCES]
    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[channel_sweep] FATAL: dev sequence(s) %s in held-out selection" % leaked)
    if not seqs:
        raise SystemExit("[channel_sweep] FATAL: no sequences under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats (same estimator as scripts/consensus_sweep.py, for comparability)
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
                      seed: int = BOOTSTRAP_SEED,
                      weight_key: str = "n_report_frames") -> dict[str, Any]:
    """Resample whole *sequences* with replacement; recompute the frame-weighted pool."""
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return {"reps": reps, "n_sequences": 0, "p2.5": None, "p50": None, "p97.5": None}
    vals = [float(r[key]) for r in rows]
    weights = [float(r[weight_key]) for r in rows]
    samples: list[float] = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(_weighted([vals[i] for i in idx], [weights[i] for i in idx]))
    p25, p50, p975 = _pct(samples, [2.5, 50.0, 97.5])
    return {
        "reps": reps,
        "n_sequences": n,
        "seed": seed,
        "weight_key": weight_key,
        "estimator": "frame-weighted pooled mean, resampling sequences with replacement",
        "p2.5": round(p25, 6),
        "p50": round(p50, 6),
        "p97.5": round(p975, 6),
        "mean": round(sum(samples) / len(samples), 6),
    }


def paired_bootstrap(arm_rows: Sequence[dict[str, Any]], ref_rows: Sequence[dict[str, Any]], *,
                     key: str, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap the *paired* delta of one metric: same sequence, both arms."""
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
# Per-frame engagement (the pre-registered mechanism check)
# --------------------------------------------------------------------------- #
#: Refusal-reason buckets.  The raw reason carries numbers ("cost 1.160 > accept
#: 1.000"), so bucketing it is what makes the ledger readable and, more to the
#: point, comparable ACROSS arms: arm B is an exception to exactly one bucket
#: (`cost>bound`) and cannot touch the others, so the split is the arm's
#: opportunity space.
_REASON_BUCKETS = (
    ("all_gated", "all_gated (hard veto upstream: total=+inf)"),
    ("cost ", "cost>bound (the acceptance bound)"),
    ("ambiguous", "ambiguous (margin)"),
    ("search_margin", "search_margin"),
    ("lone_candidate", "lone_candidate"),
    ("search_needs_appearance", "search_needs_appearance"),
)


def bucket_reason(reason: str) -> str:
    r = str(reason or "").strip()
    for prefix, label in _REASON_BUCKETS:
        if r.startswith(prefix):
            return label
    return r.split("(")[0].strip()[:48] or "other"


def frame_engagement(csv_path: Path) -> dict[str, Any]:
    """``blind AND present`` share, state mix and box-level counters from one dump.

    The rule in ``docs/E077_PLAN.md`` ships an arm only if this share FALLS, so it
    is computed straight from the per-frame dump rather than inferred from the
    headline rates.  ``refuse_reason`` is decoded too: it is what distinguishes a
    refusal by the additive bound (which arm B can override) from a refusal by an
    upstream hard veto (``all_gated:*``, which it cannot).
    """
    if not csv_path.is_file():
        return {"available": False}
    blind = _blind_states()
    n = 0
    n_present = 0
    n_blind = 0
    n_blind_present = 0
    n_correct = 0
    states: dict[str, int] = {}
    reasons: dict[str, int] = {}
    chan: dict[str, int] = {}
    import csv as _csv

    with csv_path.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            n += 1
            st = str(row.get("state") or "")
            present = str(row.get("gt_present")).lower() == "true"
            is_blind = st in blind
            states[st] = states.get(st, 0) + 1
            if present:
                n_present += 1
            if is_blind:
                n_blind += 1
                if present:
                    n_blind_present += 1
            if str(row.get("correct")).lower() == "true":
                n_correct += 1
            rr = str(row.get("refuse_reason") or "")
            if rr:
                key = bucket_reason(rr)
                reasons[key] = reasons.get(key, 0) + 1
            ac = str(row.get("accept_channel") or "")
            if ac:
                chan[ac] = chan.get(ac, 0) + 1
    return {
        "available": True,
        "n_frames": n,
        "n_gt_present": n_present,
        "n_blind": n_blind,
        "n_blind_present": n_blind_present,
        "blind_share": round(n_blind / n, 6) if n else None,
        "blind_present_share": round(n_blind_present / n_present, 6) if n_present else None,
        "states": dict(sorted(states.items())),
        "refuse_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "accept_channels": dict(sorted(chan.items())),
        "csv_sha256_16": hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16],
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
    csv_path = out_dir / "runs" / ("%s__%s__%s.csv" % (seq, spec["variant"], PROTOCOL["mask"]))

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
        cached = _parse_summary(summary_path, seq, target_id, specs_path, csv_path)
        # A stale summary from a crashed/errored attempt must never be cached as a result.
        if cached.get("status") == "INCLUDED":
            cached.update(base)
            cached["from_cache"] = True
            return cached
    elif summary_path.is_file() and not force:
        # PROVENANCE GUARD: a run dir without this sweep's ``.mine`` stamp cannot
        # be attributed to this protocol, so it is re-run rather than reported.
        print("[channel_sweep] PURGE foreign run dir (no .mine provenance): %s/%s" % (arm_id, seq),
              flush=True)

    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["CHANNELSWEEP_ARM"] = arm_id
    env["CHANNELSWEEP_VARIANT"] = spec["variant"]
    env["CHANNELSWEEP_OVERRIDES"] = json.dumps(spec["overrides"])
    env["CHANNELSWEEP_OUT"] = str(out_dir)
    env["CHANNELSWEEP_SPECS"] = str(specs_path)
    env["PYTHONIOENCODING"] = "utf-8"
    if _SRC_OVERRIDE:
        env["CHANNELSWEEP_SRC"] = str(_SRC_OVERRIDE)
    if target_id is not None:
        env["CHANNELSWEEP_TARGET_ID"] = str(int(target_id))
    else:
        env.pop("CHANNELSWEEP_TARGET_ID", None)

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
                json.dumps({k: v for k, v in env.items() if k.startswith("CHANNELSWEEP_")}, sort_keys=True)))
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
            rec = _parse_summary(summary_path, seq, target_id, specs_path, csv_path)
        else:
            rec = {"status": "ERROR", "reason": "child exit %d" % code}
        rec["exit_code"] = code
        rec["wall_seconds"] = round(wall, 1)
        attempts.append({"attempt": attempt, "exit_code": code, "status": rec.get("status"),
                         "wall_seconds": round(wall, 1)})
        if code == 3 or rec.get("status") == "INCLUDED":
            break
        # One retry only, for infrastructure flakiness.  Every attempt is kept in
        # `attempts`, so a retry cannot launder a real failure into a success.

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


def _record_stats(summary_path: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    """``tracker_stats`` of the single run this summary covers (cf. consensus_sweep)."""
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


def _parse_summary(summary_path: Path, seq: str, forced_id: int | None, specs_path: Path,
                   csv_path: Path) -> dict[str, Any]:
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
            print("[channel_sweep] WARNING %s: salvaged summary.json, ignored %d trailing chars (%s)"
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
    stats = _record_stats(summary_path, row) or {}
    eng = frame_engagement(csv_path)
    if not eng.get("available"):
        # The mechanism clause of the ship rule needs the per-frame dump; without it
        # the pair cannot be scored against the rule.
        return {"status": "ERROR", "reason": "MECHANISM GUARD: no per-frame dump at %s"
                                            % csv_path.relative_to(ROOT)}
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
        # per-frame engagement (ship-rule input)
        "blind_present_share": eng["blind_present_share"],
        "blind_share": eng["blind_share"],
        "n_blind_present": eng["n_blind_present"],
        "n_gt_present": eng["n_gt_present"],
        "n_blind": eng["n_blind"],
        "states": eng["states"],
        "refuse_reasons": eng["refuse_reasons"],
        "accept_channels": eng["accept_channels"],
        "csv_sha256_16": eng["csv_sha256_16"],
        # mechanism counters from the tracker itself
        "channel_accept_geometry": int(stats.get("channel_accept_geometry", 0) or 0),
        "channel_accept_appearance": int(stats.get("channel_accept_appearance", 0) or 0),
        "aspect_clamp_steps": int(stats.get("aspect_clamp_steps", 0) or 0),
        "aspect_clamp_calls": int(stats.get("aspect_clamp_calls", 0) or 0),
    }
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
_METRICS = ("tsr", "far", "id_consistency", "ms_per_frame", "blind_present_share")


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
    # The mechanism check is counted over GT-PRESENT frames, not report frames.
    gt_w = [float(r["n_gt_present"]) for r in inc]
    pooled["blind_present_share_gtweighted"] = round(
        _weighted([float(r["blind_present_share"] or 0.0) for r in inc], gt_w), 6
    )
    pooled["n_blind_present"] = int(sum(int(r["n_blind_present"]) for r in inc))
    pooled["n_gt_present"] = int(sum(int(r["n_gt_present"]) for r in inc))

    eng = {
        "n_blind_present": int(sum(int(r["n_blind_present"]) for r in inc)),
        "n_gt_present": int(sum(int(r["n_gt_present"]) for r in inc)),
        "n_blind": int(sum(int(r["n_blind"]) for r in inc)),
        "blind_present_share": pooled["blind_present_share_gtweighted"],
        "channel_accept_geometry": int(sum(int(r["channel_accept_geometry"]) for r in inc)),
        "channel_accept_appearance": int(sum(int(r["channel_accept_appearance"]) for r in inc)),
        "aspect_clamp_steps": int(sum(int(r["aspect_clamp_steps"]) for r in inc)),
        "aspect_clamp_calls": int(sum(int(r["aspect_clamp_calls"]) for r in inc)),
        "n_sequences_with_channel_accept": int(
            sum(1 for r in inc if (r["channel_accept_geometry"] + r["channel_accept_appearance"]) > 0)),
        "n_sequences_with_aspect_clamp": int(
            sum(1 for r in inc if int(r["aspect_clamp_calls"]) > 0)),
        "refuse_reasons": _merge_counts(r["refuse_reasons"] for r in inc),
        "accept_channels": _merge_counts(r["accept_channels"] for r in inc),
        "states": _merge_counts(r["states"] for r in inc),
    }
    return {
        "included": inc,
        "excluded": exc,
        "pooled": pooled,
        "bootstrap": {
            **{key: cluster_bootstrap(inc, key=key) for key in _METRICS if key != "blind_present_share"},
            "blind_present_share": cluster_bootstrap(inc, key="blind_present_share", weight_key="n_gt_present"),
        },
        "engagement": eng,
        "frame_counts": {
            "n_sequences": len(inc),
            "n_report_frames": int(sum(weights)),
            "n_frames": int(sum(int(r["n_frames"]) for r in inc)),
        },
    }


def _merge_counts(dicts: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[str(k)] = out.get(str(k), 0) + int(v)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def verdict(arm_entry: dict[str, Any], ref_entry: dict[str, Any]) -> dict[str, Any]:
    """The E077 rule from ``docs/E077_PLAN.md``, applied verbatim.

    SHIP iff pooled TSR rises by >= +3 pp AND the 95 % CI of the *paired* TSR
    delta excludes 0 AND the upper bound of the paired dFAR CI is < +3 pp AND the
    ``blind AND present`` share FALLS (mechanism engaged).
    KILL if the arm is bit-identical to the reference (inert) or if TSR falls.
    Otherwise NO-SHIP: it did not clear the bar, which is not the same as refuted.
    """
    out: dict[str, Any] = {
        "arm": arm_entry["arm"],
        "rule": ("SHIP iff pooled dTSR >= +%.1f pp AND paired-dTSR 95%% CI excludes 0 AND "
                 "paired-dFAR 95%% CI upper bound < +%.1f pp AND blind-AND-present share falls; "
                 "KILL if bit-identical (inert) or dTSR < 0" % (TSR_SHIP_PP, FAR_MAX_PP)),
    }
    if arm_entry["arm"] == ref_entry["arm"]:
        out.update(verdict="REFERENCE", text="Reference arm: both E077 arms off (shipped behaviour).")
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
    ref_eng = ref_entry.get("engagement") or {}

    # Bit-identity: every included sequence produced a byte-identical per-frame
    # dump, i.e. the switch changed nothing at all.  This is the E032b failure and
    # the plan kills on it.
    ref_sha = {r["sequence"]: r.get("csv_sha256_16") for r in ref_entry["rows"]
               if r.get("status") == "INCLUDED"}
    arm_sha = {r["sequence"]: r.get("csv_sha256_16") for r in arm_entry["rows"]
               if r.get("status") == "INCLUDED"}
    common = sorted(set(ref_sha) & set(arm_sha))
    n_identical = sum(1 for s in common if ref_sha[s] and ref_sha[s] == arm_sha[s])
    bit_identical = bool(common) and n_identical == len(common)

    b_share = eng.get("blind_present_share")
    r_share = ref_eng.get("blind_present_share")
    d_blind = None if (b_share is None or r_share is None) else (b_share - r_share) * 100.0
    blind_falls = bool(d_blind is not None and d_blind < 0.0)
    engaged = bool(
        int(eng.get("channel_accept_geometry", 0) or 0) + int(eng.get("channel_accept_appearance", 0) or 0)
        + int(eng.get("aspect_clamp_steps", 0) or 0) > 0
    )

    ok_tsr_mag = d_tsr >= TSR_SHIP_PP
    ok_tsr_ci = bool(p_tsr.get("excludes_zero")) and tsr_lo > 0.0
    ok_far = far_hi < FAR_MAX_PP
    ship = bool(ok_tsr_mag and ok_tsr_ci and ok_far and blind_falls)

    if bit_identical or not engaged:
        v = "KILL"
        why = ("bit-identical on all %d paired sequences (the mechanism never fired)"
               % len(common)) if bit_identical else "the mechanism counters are all zero (never fired)"
    elif d_tsr < 0.0:
        v = "KILL"
        why = "pooled TSR falls (%+.2f pp)" % d_tsr
    elif ship:
        v = "SHIP"
        why = ("pooled TSR %+.2f pp (>= +%.1f), paired CI [%+.2f, %+.2f] pp excludes 0, "
               "paired dFAR CI upper %+.2f pp < +%.1f, blind-AND-present %+.2f pp"
               % (d_tsr, TSR_SHIP_PP, tsr_lo, tsr_hi, far_hi, FAR_MAX_PP, d_blind or 0.0))
    else:
        v = "NO-SHIP"
        bad = []
        if not ok_tsr_mag:
            bad.append("pooled dTSR %+.2f pp < +%.1f" % (d_tsr, TSR_SHIP_PP))
        if not ok_tsr_ci:
            bad.append("paired dTSR CI [%+.2f, %+.2f] pp includes 0" % (tsr_lo, tsr_hi))
        if not ok_far:
            bad.append("paired dFAR CI upper %+.2f pp >= +%.1f" % (far_hi, FAR_MAX_PP))
        if not blind_falls:
            bad.append("blind-AND-present share does not fall (%+.2f pp)" % (d_blind or 0.0))
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
        blind_present_falls=bool(blind_falls),
        pooled_dblind_present_pp=None if d_blind is None else round(d_blind, 3),
        mechanism_engaged=bool(engaged),
        bit_identical=bool(bit_identical),
        n_bit_identical=n_identical,
        n_paired_sequences=len(common),
        sequences_up=int(p_tsr["sequences_up"]),
        sequences_down=int(p_tsr["sequences_down"]),
        sequences_flat=int(p_tsr["sequences_flat"]),
    )
    return out


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
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
    a("# E077 -- channel-conditional fusion + bounded-aspect coasting, held-out sweep")
    a("")
    a("<!-- generated by scripts/channel_sweep.py; every number is read from results/channel_sweep/summary.json -->")
    a("")
    if payload.get("partial"):
        a("> **PARTIAL RUN -- read this first.** %s"
          % (payload.get("partial_note")
             or "Only part of the 4 x 22 grid was measured; the pools below cover the sequences "
                "that completed."))
        a(">")
        a("> The full grid needs ~7-10 h on this box (measured 244-464 s per pair while a second "
          "sweep shared the GPU), so it was stopped early; `sequences_run` is the set with a "
          "COMPLETE 4-arm block, and every delta below is paired within that set.")
        a("")
    a("## Verdict (pre-registered rule, fixed before the run)")
    a("")
    a("> SHIP only if pooled TSR rises by **>= +3 pp** with the 95 % CI of the **paired** delta excluding 0, "
      "**and** the upper bound of the paired dFAR CI is **< +3 pp**, **and** the `blind AND present` share "
      "falls (mechanism engaged). KILL if an arm is bit-identical to the reference (inert) or if TSR falls.")
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
                                if shipped else "ship nothing; both E077 arms stay OFF."))
    a("")
    a("## Protocol (identical to E070, so the numbers are comparable)")
    a("")
    a("- data: `data/raw/dancetrack/val`, the **%d** held-out sequences; frozen dev split %s asserted absent"
      % (len(payload["heldout_sequences"]), ", ".join(sorted(DEV_SEQUENCES))))
    a("- `--max-frames %d --subsample %d --mask %s --jobs %d --debug-csv`, shipped defaults elsewhere"
      % (PROTOCOL["max_frames"], PROTOCOL["subsample"], PROTOCOL["mask"], PROTOCOL["jobs"]))
    a("- one `--ablate channel` arm per run, `run_target.build_variants` replaced in the child; "
      "`specs.json` asserts exactly 1 spec with exactly the intended overrides and each `summary.json` "
      "reports exactly 1 variant")
    a("- target identity: the `channel_off` pass picks the target with the shipped strategy; every other "
      "arm is pinned to that `--target-id` and each run's own `meta.target_ids` is asserted against it")
    a("- pooling: frame-weighted over `n_report_frames`; 95 %% CI = cluster bootstrap over sequences, "
      "%d reps, seed %d. The `blind AND present` share is weighted by GT-present frames instead, and is "
      "read from each pair's per-frame dump (`refuse_reason`, `state`) rather than from the headline rates."
      % (BOOTSTRAP_REPS, BOOTSTRAP_SEED))
    a("- the two thresholds are the only new numbers: `cost.channel_geom_tau = %s`, "
      "`cost.channel_app_tau = %s` (medians over correct frames of the DEV split; see "
      "`results/e077_calib/thresholds.json`)"
      % (payload.get("thresholds", {}).get("channel_geom_tau", "?"),
         payload.get("thresholds", {}).get("channel_app_tau", "?")))
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
    a("| arm | dTSR pp [95% CI] | seq up/down/flat | dFAR pp [95% CI] | d id_consistency pp | d ms/frame | d blind&present pp | n paired |")
    a("|---|---|---|---|---|---|---|---|")
    for arm_id in order:
        if arm_id == REFERENCE_ARM:
            continue
        v = arms[arm_id]["verdict"]
        if "paired_dtsr_ci_pp" not in v:
            a("| `%s` | -- | -- | -- | -- | -- | -- | 0 |" % arm_id)
            continue
        a("| `%s` | %+.2f [%+.2f, %+.2f] | %d/%d/%d | %+.2f [%+.2f, %+.2f] | %+.2f | %+.2f | %s | %d |"
          % (arm_id, v["pooled_dtsr_pp"], v["paired_dtsr_ci_pp"][0], v["paired_dtsr_ci_pp"][1],
             v["sequences_up"], v["sequences_down"], v["sequences_flat"],
             v["pooled_dfar_pp"], v["paired_dfar_ci_pp"][0], v["paired_dfar_ci_pp"][1],
             v["pooled_did_consistency_pp"], v["pooled_dms_per_frame"],
             ("%+.2f" % v["pooled_dblind_present_pp"]) if v.get("pooled_dblind_present_pp") is not None else "--",
             v.get("n_paired_sequences", 0)))
    a("")
    a("## Did the mechanism engage? (`blind AND present`, per arm)")
    a("")
    a("The pre-registered engagement test. `blind AND present` is the share of GT-present frames on which "
      "the tracker is in one of its blind states (`occluded_soft`, `lost_short`, `lost_long`, "
      "`out_of_frame`, `gave_up`, `reacquiring`); it must FALL for the arm to be a candidate. "
      "The counters on the right are what the tracker itself recorded, and they are what separates "
      "\"the branch fired but did not help\" from \"the branch never fired\".")
    a("")
    a("| arm | blind & present [95% CI] | n blind&present / n present | d vs ref pp | channel_accept geom/app | aspect clamp steps / frames | seqs with channel accept |")
    a("|---|---|---|---|---|---|---|")
    for arm_id in order:
        e = arms[arm_id]
        eng = e.get("engagement")
        if not eng:
            a("| `%s` | -- | -- | -- | -- | -- | -- |" % arm_id)
            continue
        b = (e.get("bootstrap") or {}).get("blind_present_share")
        d = e["verdict"].get("pooled_dblind_present_pp")
        a("| `%s` | %s | %d / %d | %s | %d / %d | %d / %d | %d |"
          % (arm_id, _fmt_ci(eng["blind_present_share"], b),
             eng["n_blind_present"], eng["n_gt_present"],
             "--" if (d is None or arm_id == REFERENCE_ARM) else "%+.2f" % d,
             eng["channel_accept_geometry"], eng["channel_accept_appearance"],
             eng["aspect_clamp_steps"], eng["aspect_clamp_calls"],
             eng["n_sequences_with_channel_accept"]))
    a("")
    a("### Where the refusals come from (per arm, pooled frames)")
    a("")
    a("Arm B is an exception to the *acceptance bound* (`cost > bound`), so it can only engage on frames "
      "whose refusal arrived at that test. A refusal recorded as `all_gated:...` means every candidate was "
      "vetoed upstream (`total = +inf` by the E027/E029 appearance ceiling or the E032 motion band), which "
      "the plan explicitly keeps in force. This table says which of the two a sequence actually suffers from.")
    a("")
    for arm_id in order:
        eng = arms[arm_id].get("engagement") or {}
        rr = eng.get("refuse_reasons") or {}
        if not rr:
            a("- `%s`: no refusals recorded" % arm_id)
            continue
        top = list(rr.items())[:5]
        a("- `%s`: %s" % (arm_id, ", ".join("`%s` %d" % (k, v) for k, v in top)))
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
        a("**`%s`** -- %d of %d paired sequences move TSR by more than 5 pp:"
          % (arm_id, len(moved), arms[arm_id]["paired"]["tsr"]["n_paired"]))
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
        a("None: every (sequence, arm) pair produced an observable target and a per-frame dump; the "
          "denominators are %d in every pool." % len(payload["sequences_run"]))
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
    """Take ``results/channel_sweep/.sweep.lock``; False if a live sweep holds it."""
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
            print("[channel_sweep] REFUSING TO START: %s is held by live sweep pid %d."
                  % (lock_path, holder), flush=True)
            return False
        print("[channel_sweep] stale lock from dead pid %d -- taking over" % holder, flush=True)
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

    import atexit

    atexit.register(_release)
    return True


def thresholds_snapshot() -> dict[str, Any]:
    """The two derived numbers, read back from the calibration artifact."""
    p = ROOT / "results" / "e077_calib" / "thresholds.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(data.get("chosen") or {})


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(CHILD_ENV):
        return _run_child()

    if not acquire_single_instance_lock():
        return 0

    p = argparse.ArgumentParser(description="E077 channel-fusion / aspect-coast sweep on the held-out split.")
    p.add_argument("--arms", default="all", help="'all' or comma-separated arm ids: %s" % ",".join(ARM_ORDER))
    p.add_argument("--limit-seqs", type=int, default=None, help="debug: only the first N held-out sequences")
    p.add_argument("--force", action="store_true", help="re-run pairs that already have a summary")
    p.add_argument("--run-only", default=None, help="debug: restrict to one sequence name")
    p.add_argument("--by-sequence", action="store_true",
                   help="run sequence-outer / arm-inner, so a partial stop still leaves complete "
                        "4-arm blocks for the sequences that finished")
    args = p.parse_args(list(argv) if argv is not None else None)

    print("[channel_sweep] FROZEN DEV SPLIT (never swept, never reported): %s"
          % ", ".join(sorted(DEV_SEQUENCES)), flush=True)

    all_seqs = heldout_sequences(VAL_ROOT)
    print("[channel_sweep] held-out sequences: %d -- %s" % (len(all_seqs), ", ".join(all_seqs)), flush=True)

    seqs = list(all_seqs)
    partial = False
    if args.run_only:
        if args.run_only not in all_seqs:
            raise SystemExit("[channel_sweep] FATAL: %r is not a held-out sequence" % args.run_only)
        seqs = [args.run_only]
        partial = True
    if args.limit_seqs:
        seqs = seqs[: int(args.limit_seqs)]
        partial = len(seqs) < len(all_seqs)

    leaked = sorted(DEV_SEQUENCES & set(seqs))
    if leaked:
        raise SystemExit("[channel_sweep] FATAL: dev sequence(s) %s in the run set; refusing." % leaked)
    assert not (DEV_SEQUENCES & set(seqs)), "dev split leaked into the sweep"

    arms = all_arms(args.arms)
    if REFERENCE_ARM in arms:
        arms = [REFERENCE_ARM] + [x for x in arms if x != REFERENCE_ARM]
    elif any(not arm_spec(x)["reference"] for x in arms):
        raise SystemExit("[channel_sweep] the reference arm %s must be in --arms to pin target ids" % REFERENCE_ARM)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SPECS_JSON.write_text(json.dumps({
        "generated_by": "scripts/channel_sweep.py",
        "experiment": "E077",
        "protocol": dict(PROTOCOL),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "data_root": str(VAL_ROOT.relative_to(ROOT)),
        "dev_sequences": sorted(DEV_SEQUENCES),
        "thresholds": thresholds_snapshot(),
        "arms": {arm_id: {"variant": arm_spec(arm_id)["variant"],
                          "overrides": arm_spec(arm_id)["overrides"],
                          "label": arm_spec(arm_id)["label"],
                          "mechanism": arm_spec(arm_id)["mechanism"]}
                 for arm_id in ARM_ORDER},
        "ship_rule": ("pooled dTSR >= +3 pp AND paired-dTSR 95%% CI excludes 0 AND paired-dFAR CI "
                      "upper < +3 pp AND blind-AND-present share falls; KILL on bit-identity or dTSR < 0"),
    }, indent=2) + "\n", encoding="utf-8")

    log_dir = OUT_ROOT / "logs"
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    target_ids: dict[str, int] = {}

    # Ordering.  Default is the arm-outer order of the E070/E071 drivers (so the
    # reference pass finishes first and pins every target id).  `--by-sequence`
    # interleaves the arms per sequence instead, which costs extra model loads but
    # guarantees that a run stopped early still has COMPLETE 4-arm blocks -- the
    # only shape in which a partial sweep can still answer the paired question.
    if args.by_sequence:
        jobs = [(seq, arm_id) for seq in seqs for arm_id in arms]
    else:
        jobs = [(seq, arm_id) for arm_id in arms for seq in seqs]

    for i, (seq, arm_id) in enumerate(jobs, 1):
        spec = arm_spec(arm_id)
        tid = None if spec["reference"] else target_ids.get(seq)
        if not spec["reference"] and tid is None:
            # With --by-sequence the reference arm runs first for this sequence,
            # so this can only fire on a genuinely broken ordering.
            raise SystemExit("[channel_sweep] no reference target id for %s; run the reference arm first" % seq)
        ts = time.perf_counter()
        rec = run_pair(arm_id, seq, target_id=tid, force=bool(args.force), log_dir=log_dir)
        rows.append(rec)
        if spec["reference"] and rec.get("status") == "INCLUDED":
            target_ids[seq] = int(rec["target_id"])
        print("[channel_sweep] %3d/%3d %-16s %-16s %-8s %6.1fs %s"
              % (i, len(jobs), arm_id, seq, rec.get("status"), time.perf_counter() - ts,
                 ("TSR=%.3f FAR=%.3f b&p=%.3f ch=%d/%d clamp=%d"
                  % (rec["tsr"], rec["far"], rec["blind_present_share"] or 0.0,
                     rec["channel_accept_geometry"], rec["channel_accept_appearance"],
                     rec["aspect_clamp_steps"]))
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
        raise SystemExit("[channel_sweep] reference arm missing from this run; cannot apply the rule")

    for arm_id in arms:
        entry = arms_payload[arm_id]
        entry["paired"] = {key: paired_bootstrap(entry["rows"], ref_entry["rows"], key=key)
                           for key in _METRICS}
        entry["verdict"] = verdict(entry, ref_entry)

    n_included = sum(1 for r in rows if r.get("status") == "INCLUDED")
    payload: dict[str, Any] = {
        "generated_by": "scripts/channel_sweep.py",
        "experiment": "E077",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "heldout_sequences": all_seqs,
        "sequences_run": seqs,
        "partial": bool(partial),
        "n_pairs_run": len(rows),
        "n_pairs_included": n_included,
        "n_pairs_excluded": sum(1 for r in rows if r.get("status") == "EXCLUDED"),
        "n_pairs_errors": sum(1 for r in rows if r.get("status") == "ERROR"),
        "protocol": dict(PROTOCOL),
        "thresholds": thresholds_snapshot(),
        "acceptance_rule": {
            "source": "docs/E077_PLAN.md (ship rule written before the run; not edited)",
            "text": ("SHIP only if pooled TSR rises by >= +3 pp with the 95 % CI of the paired delta "
                     "excluding 0, and the upper bound of the paired dFAR CI is < +3 pp, and the "
                     "blind-AND-present share falls. KILL if bit-identical (inert) or TSR falls."),
            "tsr_ship_pp": TSR_SHIP_PP,
            "far_max_pp": FAR_MAX_PP,
            "statistic": "mean paired per-sequence delta vs %s, cluster bootstrap over sequences" % REFERENCE_ARM,
        },
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "cli": {"arms": args.arms, "force": bool(args.force), "limit_seqs": args.limit_seqs,
                "run_only": args.run_only,
                "command": "python scripts/channel_sweep.py --arms %s%s%s"
                           % (args.arms, " --run-only %s" % args.run_only if args.run_only else "",
                              " --limit-seqs %d" % args.limit_seqs if args.limit_seqs else "")},
        "order": arms,
        "arms": arms_payload,
        "target_ids": target_ids,
        "wall_seconds": round(wall, 1),
        "wall_seconds_pair_sum": round(sum(float(r.get("wall_seconds") or 0.0) for r in rows), 1),
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
            print("[channel_sweep] %-16s TSR=%.4f [%.4f, %.4f] FAR=%.4f [%.4f, %.4f] b&p=%.4f "
                  "ch=%d/%d clamp=%d -> %s"
                  % (arm_id, pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
                     pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"],
                     eng["blind_present_share"], eng["channel_accept_geometry"],
                     eng["channel_accept_appearance"], eng["aspect_clamp_steps"], v["verdict"]), flush=True)
        else:
            print("[channel_sweep] %-16s no usable pool -> %s" % (arm_id, v["verdict"]), flush=True)
    print("[channel_sweep] wall=%.1fs wrote %s and %s"
          % (wall, REPORT_JSON.relative_to(ROOT), REPORT_MD.relative_to(ROOT)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
