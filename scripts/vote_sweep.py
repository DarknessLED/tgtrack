#!/usr/bin/env python
"""E079 -- the three-reference anchor vote on the 22 held-out DanceTrack sequences.

WHAT IS BEING MEASURED
----------------------
E076 split the tracking failure into 0.511 blindness + 0.238 wrong-while-tracking +
0.019 other; E077 localised the refusal to `all_gated:anchor_ceiling` (61 of 70
refused frames on dancetrack0097); E078 showed the E029 band saturates at its 0.75
ceiling on about 75 % of frames.  So for most vetoed candidates the question is not
"how wide is the band" but "is this one reference the right judge?".

E079 gives the veto three references that answer the SAME question with the SAME
threshold (the E029 `TargetTracker._anchor_limit()` band -- no new constant):

    R1  the frozen click crop (`_anchor_feat`, unpoisonable)
    R2  an EMA-refreshed anchor (`cost.anchor_vote_alpha`, ~10-frame time constant)
        that may only move on frames where R1 AND R3 already accept the same box
    R3  the long-term gallery's nearest entry (cosine distance)

A candidate whose refusal is currently the anchor veto is admitted when >= k of the
three references pass it.  Every other refusal path (distractor margin, lone-candidate
bound, search margin, missing appearance while searching, blind-state transitions)
keeps applying afterwards.

ARMS
----
    vote_off        shipped behaviour: the click-crop veto, one reference (k = 0)
    vote_majority   k = 2 of 3
    vote_unanimous  k = 3 -- a CONTROL: a candidate only reaches the vote because R1
                    already rejected it, so k = 3 is unreachable and this arm must be
                    arithmetically identical to `vote_off`

PROTOCOL (identical to E070 / E071 / E078 so the numbers stay comparable)
-------------------------------------------------------------------------
22 held-out DanceTrack-val sequences, dev 0004/0005/0007 excluded and asserted absent,
`--max-frames 300 --subsample 2 --mask none --jobs 1 --debug-csv`, one `--ablate vote`
arm per run, the reference arm picking the target id every other arm is pinned to,
frame-weighted pooling over `n_report_frames`, cluster bootstrap over sequences
(2000 reps, seed 0), and paired per-sequence deltas against the reference.

PRE-REGISTERED RULE (docs/E079_PLAN.md; written before the run, never re-tuned)
------------------------------------------------------------------------------
    SHIP `vote_majority` iff ALL THREE hold:
      1. pooled TSR rises >= +3.0 pp with the paired 95 % CI excluding 0;
      2. the paired dFAR 95 % CI UPPER bound is < +3.0 pp;
      3. the mechanism engaged: the `blind AND present` frame share FALLS versus
         `vote_off`.
    Otherwise KILL, and say why.  `vote_unanimous` is a control, never a candidate.

`vote_off` must also reproduce `results/gate_sweep/summaries/margin_0.20/` exactly
(TSR / FAR / target id): that is the off-by-default no-op proof, checked per sequence
and recorded in `summary.json` under `reference_noop`.

Usage::

    python scripts/vote_sweep.py                                # all 3 arms, 22 sequences
    python scripts/vote_sweep.py --arms vote_off,vote_majority
    python scripts/vote_sweep.py --run-only dancetrack0019 --arms vote_off,vote_majority
    python scripts/vote_sweep.py --limit-seqs 1                 # smoke

`E079_SRC=<dir>` puts `<dir>` in front of `sys.path` and pre-imports `tgtrack` from it, so
the sweep can measure a scratch copy of `src/` (the child asserts which file it loaded and
records it in `summary.json`).  `E079_OUT=<dir>` overrides the output root.
"""
from __future__ import annotations

import argparse
import atexit
import csv
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
#: E085 -- cohort override.  Criterion 2's effect is sparse over 22 sequences (15 of
#: them bit-unchanged), which is why E082's pooled CI cannot certify it.  The only
#: honest way to buy power is a FRESH, pre-declared cohort, so the root is an
#: environment variable: the default epoch stays ``.../dancetrack/val``, and E085 runs
#: ``.../dancetrack/train`` with its own E079_OUT so the two epochs never pool.
VAL_ROOT = Path(os.environ.get("VOTE_DATA_ROOT") or (ROOT / "data" / "raw" / "dancetrack" / "val"))
#: E085 -- val and train share the ``dancetrack00XX`` numbering, so the val dev-split
#: name check cannot mean anything on the train cohort; the epoch sets this to 0 and
#: the run prints the fact instead of silently trusting a collision.
DEV_CHECK = os.environ.get("VOTE_DEV_CHECK", "1") != "0"
OUT_ROOT = Path(os.environ.get("E079_OUT") or (ROOT / "results" / "vote_sweep"))
REPORT_JSON = OUT_ROOT / "summary.json"
REPORT_MD = OUT_ROOT / "REPORT.md"

#: The E070 reference sweep `vote_off` has to reproduce bit for bit.
REFERENCE_DIR = ROOT / "results" / "gate_sweep" / "summaries" / "margin_0.20"

#: FROZEN dev split (E070).  Never swept, never reported.
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}

PROTOCOL = {
    "max_frames": 300,
    "subsample": 2,
    "mask": "none",
    "jobs": 1,
    "debug_csv": True,
    "detector": "yolo (shipped default)",
    "target_strategy": "stressed (shipped default)",
}

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0

#: The pre-registered rule.
GAIN_PP = 3.0
FAR_MAX_PP = 3.0

CHILD_ENV = "VOTESWEEP_CHILD_SEQUENCE"

METRICS = ("tsr", "far", "id_consistency")
#: Pooled / bootstrapped keys: the three metrics plus the pre-registered engagement share.
POOL_KEYS = METRICS + ("blind_present_share",)

ARMS: dict[str, dict[str, Any]] = {
    "vote_off": {
        # E089 shipped a non-1.0 default for `anchor_band_lost_scale`; the reference arm
        # pins the OLD value explicitly so the 22-sequence records of the E070/E079/E082
        # epoch (and the margin_0.20 no-op check) stay comparable bit for bit.
        "overrides": {
            "cost.anchor_vote": "off",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_band_lost_scale": 1.0,
        },
        "variant": "vote_off",
        "reference": True,
        "k": 0,
        "label": "anchor veto OFF, pre-E089 epoch (lost-scale pinned to 1.0)",
    },
    "shipped": {
        "overrides": {"cost.anchor_vote": "off"},
        "variant": "shipped_e089",
        "reference": False,
        "k": 0,
        "label": "the shipped default after E089 (lost-scale 1.25)",
    },
    "vote_majority": {
        "overrides": {"cost.anchor_vote": "majority", "cost.anchor_vote_alpha": 0.1},
        "variant": "vote_majority",
        "reference": False,
        "k": 2,
        "label": "admit an anchor-vetoed candidate when >= 2 of R1 / R2 / R3 pass it",
    },
    "vote_unanimous": {
        "overrides": {"cost.anchor_vote": "unanimous", "cost.anchor_vote_alpha": 0.1},
        "variant": "vote_unanimous",
        "reference": False,
        "k": 3,
        "label": "CONTROL: k = 3, unreachable for a vetoed candidate -> identical to vote_off",
    },
    # E081 (docs/E081_PLAN.md): the E079 vote run only while the tracker is in a
    # distressed regime.  `vote_adaptive_1.00` is the null control -- distress can
    # never exceed 1.0, so the gate never opens and the arm must reproduce
    # `vote_off`; that is the falsifier for the gate's plumbing.
    "vote_adaptive_0.20": {
        "overrides": {
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_distress_window": 50,
            "cost.anchor_vote_distress_tau": 0.20,
        },
        "variant": "vote_adaptive_0.20",
        "reference": False,
        "k": 2,
        "label": "E081: k = 2 vote armed only while trailing blindness share > 0.20",
    },
    "vote_adaptive_0.35": {
        "overrides": {
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_distress_window": 50,
            "cost.anchor_vote_distress_tau": 0.35,
        },
        "variant": "vote_adaptive_0.35",
        "reference": False,
        "k": 2,
        "label": "E081: k = 2 vote armed only while trailing blindness share > 0.35",
    },
    "vote_adaptive_0.50": {
        "overrides": {
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_distress_window": 50,
            "cost.anchor_vote_distress_tau": 0.50,
        },
        "variant": "vote_adaptive_0.50",
        "reference": False,
        "k": 2,
        "label": "E081: k = 2 vote armed only while trailing blindness share > 0.50",
    },
    "vote_adaptive_0.65": {
        "overrides": {
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_distress_window": 50,
            "cost.anchor_vote_distress_tau": 0.65,
        },
        "variant": "vote_adaptive_0.65",
        "reference": False,
        "k": 2,
        "label": "E081: k = 2 vote armed only while trailing blindness share > 0.65",
    },
    "vote_adaptive_1.00": {
        "overrides": {
            "cost.anchor_vote": "adaptive",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_distress_window": 50,
            "cost.anchor_vote_distress_tau": 1.00,
        },
        "variant": "vote_adaptive_1.00",
        "reference": False,
        "k": 2,
        "label": "E081 CONTROL: gate never opens -> must reproduce vote_off",
    },
    # E083 (docs/E083_PLAN.md + its AMENDMENT in docs/EXPERIMENTS.md): dual hypothesis tracking.
    # H_crop = shipped tracker (anchor_vote off); H_gallery = the E079 arm (anchor_vote majority).
    # `mh_off` is the structural falsifier (mode off builds a plain TargetTracker, so it must equal
    # `vote_off` bit-for-bit); `mh_dual_frozen` is the control that runs both hypotheses but pins
    # the reported box to H_crop, isolating "spends compute" from "posterior really switches".
    "mh_off": {
        "overrides": {"cost.mh_mode": "off"},
        "variant": "mh_off",
        "reference": False,
        "k": 0,
        "label": "E083 FALSIFIER: dual machinery off -> must equal vote_off bit-for-bit",
    },
    "mh_dual": {
        "overrides": {"cost.mh_mode": "dual"},
        "variant": "mh_dual",
        "reference": False,
        "k": 2,
        "label": "E083: H_crop + H_gallery both alive; posterior commits on a 0.15 margin x 3 frames",
    },
    "mh_dual_frozen": {
        "overrides": {"cost.mh_mode": "dual_frozen"},
        "variant": "mh_dual_frozen",
        "reference": False,
        "k": 2,
        "label": "E083 CONTROL: both hypotheses run, reported box pinned to H_crop",
    },
}

#: E082 (docs/E082_PLAN.md): the E079 majority vote is admitted only where the
#: candidate looks MORE like the refreshed references than like the frozen click
#: crop, i.e. where ``contrast = d_R1 - min(d_R2, d_R3) >= delta``.  A candidate
#: reaches the vote only because R1 already refused it, so contrast > 0 for every
#: candidate the vote would rescue; ``delta = 0`` therefore admits exactly what
#: `vote_majority` admits and is the bit-identical control that falsifies the
#: plumbing.  Two delta sets are registered, because the plan fixes
#: {0.10, 0.20, 0.30} while the contrast scale is unknown: the finer {0.02, 0.05}
#: probes the same axis near zero.  Both are registered BEFORE any E082 number
#: exists, so neither the plan nor the sharpened set can be chosen after the fact;
#: the report prints the measured retention curve next to the grid.
E082_DELTAS: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.20, 0.30)

for _delta in E082_DELTAS:  # noqa: WPS440 - loop-local name, module-level table
    _arm_id = "vote_contrast_%.2f" % _delta
    ARMS[_arm_id] = {
        "overrides": {
            "cost.anchor_vote": "contrast",
            "cost.anchor_vote_alpha": 0.1,
            "cost.anchor_vote_contrast": float(_delta),
        },
        "variant": _arm_id,
        "reference": False,
        "k": 2,
        "label": (
            "E082: k = 2 vote, admitted only while d_R1 - min(d_R2, d_R3) >= %.2f"
            % _delta
        ) + (
            "  (CONTROL: must be bit-identical to vote_majority)"
            if _delta == 0.0 else ""
        ),
    }

#: E084b -- k = 1, registered after the E084 step-0 audit (not after any k=1 number,
#: which does not exist): the audit found that among ceiling refusals that were the
#: TRUE target, R3 alone would rescue 18 and R2 alone 0, so the k = 2 majority refuses
#: real targets to satisfy a reference that never contributes.
ARMS["vote_any"] = {
    "overrides": {"cost.anchor_vote": "any", "cost.anchor_vote_alpha": 0.1},
    "variant": "vote_any",
    "reference": False,
    "k": 1,
    "label": "E084b: k = 1, admit when ANY of R1/R2/R3 passes (unmeasured corner of the axis)",
}

#: E086-r: the lone-candidate guard -- the OTHER knob in the handover hole.  Shipped
#: default is ``None`` (off); the armed values refuse an association whose only
#: candidate costs more than the bound.  The synthetic rig measured a 3/4 -> 1/4
#: handover cut at a 2.6 pp success price (``results/motion_ablation/lone_bound.md``),
#: but no controlled REAL-data pair has ever existed -- the claim that circulated in
#: the README compared against ``target_obs19_shipped``, which has no armed variant,
#: and was retracted.  See ``docs/E086_PLAN.md`` (revision) for the pre-registration.
for _lone in (0.30, 0.40, 0.50):
    ARMS["vote_lone_%.2f" % _lone] = {
        "overrides": {
            "cost.anchor_vote": "off",
            "cost.lone_candidate_accept_cost": float(_lone),
        },
        "variant": "vote_lone_%.2f" % _lone,
        "reference": False,
        # k = 0: no vote is involved here.  The field is required by the run record
        # (the vote arms carry k = 1/2/3) and 0 keeps "no vote" distinguishable from
        # "k = 1" in every downstream table.
        "k": 0,
        "lone_bound": float(_lone),
        "label": "E086-r: lone-candidate guard armed at %.2f (shipped: None)" % _lone,
    }

ARM_ORDER = [
    "vote_off",
    "vote_majority",
    "vote_any",
    "vote_unanimous",
    "vote_adaptive_0.20",
    "vote_adaptive_0.35",
    "vote_adaptive_0.50",
    "vote_adaptive_0.65",
    "vote_adaptive_1.00",
] + ["vote_contrast_%.2f" % _d for _d in E082_DELTAS] + ["vote_lone_%.2f" % _l for _l in (0.30, 0.40, 0.50)]

#: E089: widen (or drop) the identity ceiling while the target is LOST.  E088 measured
#: who the loss belongs to -- 91 % of it is a reported box that is neither the target nor
#: locked to anyone, 41 % of the missed frames are refused by `all_gated:anchor_ceiling`
#: while the tracker is already in lost_short / lost_long, and a candidate existed every
#: time.  The ceiling exists to stop a switch while watching; while lost there is nothing
#: left to protect.  `ceil_lost_1.00` is the plumbing control (must be bit-identical to
#: `vote_off`); 0.0 means "ceiling off while lost".
for _scale in (1.25, 1.50, 2.00, 0.0):
    _name = "ceil_lost_%s" % ("off" if _scale == 0.0 else ("%.2f" % _scale))
    ARMS[_name] = {
        "overrides": {
            "cost.anchor_vote": "off",
            "cost.anchor_band_lost_scale": float(_scale),
        },
        "variant": _name,
        "reference": False,
        "k": 0,
        "lost_scale": float(_scale),
        "label": "E089: anchor ceiling scale %s while lost (shipped: 1.00)" % (
            "OFF" if _scale == 0.0 else "%.2f" % _scale),
    }
ARM_ORDER += ["ceil_lost_%.2f" % _s for _s in (1.25, 1.50, 2.00)] + ["ceil_lost_off", "ceil_lost_1.00"]

#: E089's built-in plumbing control: scale 1.0 spells out the shipped behaviour, so it must
#: be bit-identical to `vote_off` on every sequence.  Same pattern as `vote_contrast_0.00`.
ARMS["ceil_lost_1.00"] = {
    "overrides": {"cost.anchor_vote": "off", "cost.anchor_band_lost_scale": 1.0},
    "variant": "ceil_lost_1.00",
    "reference": False,
    "k": 0,
    "lost_scale": 1.0,
    "label": "E089 control: scale 1.0, must be bit-identical to vote_off",
}
REFERENCE_ARM = "vote_off"


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def arm_spec(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise SystemExit("[vote_sweep] unknown arm %r; known: %s" % (arm_id, ARM_ORDER))
    spec = dict(ARMS[arm_id])
    spec["arm"] = arm_id
    spec["group"] = "vote"
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
    src = os.environ.get("E079_SRC")
    if src:
        src_path = str(Path(src).resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        # Bind `tgtrack` to the scratch tree BEFORE run_target inserts `src/` at
        # sys.path[0]; otherwise the override would measure the shipped code.
        import tgtrack as _tg  # noqa: F401

        if Path(_tg.__file__).resolve().parents[1] != Path(src_path):
            raise SystemExit("[vote_sweep] E079_SRC did not win: tgtrack is %s" % _tg.__file__)
    for p in (str(ROOT), str(SCRIPTS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_target as rt  # noqa: WPS433 - late import needs sys.path
    import tgtrack as tg

    seq = os.environ[CHILD_ENV]
    spec = arm_spec(os.environ["VOTESWEEP_ARM"])
    overrides = json.loads(os.environ.get("VOTESWEEP_OVERRIDES") or "{}")
    variant_name = os.environ["VOTESWEEP_VARIANT"]
    if overrides != spec["overrides"]:
        raise SystemExit("[vote_sweep] child override mismatch: %r vs %r"
                         % (overrides, spec["overrides"]))

    def _pinned(groups: Any = None, **kwargs: Any) -> list[Any]:
        """Exactly one variant: the arm's overrides on top of the shipped config."""
        return [rt.Variant(variant_name, "vote_sweep", dict(overrides),
                           "E079 sweep arm " + spec["arm"])]

    rt.build_variants = _pinned  # noqa: WPS437 - process-local, file untouched

    specs_path = Path(os.environ["VOTESWEEP_SPECS"])
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
            raise SystemExit("[vote_sweep] GUARD: expected exactly 1 spec, got %d" % len(rows))
        if rows[0]["overrides"] != overrides or rows[0]["variant"] != variant_name:
            raise SystemExit("[vote_sweep] GUARD: spec carries %r / %r, intended %r / %r"
                             % (rows[0]["overrides"], rows[0]["variant"], overrides, variant_name))
        return _orig_run_grid(specs, jobs=jobs, quiet=quiet)

    rt.run_grid = _checked_run_grid

    out_dir = os.environ["VOTESWEEP_OUT"]
    argv = [
        "--data-root", str(VAL_ROOT / seq),
        "--out", str(out_dir),
        "--mask", PROTOCOL["mask"],
        # E082b: the CHILD is a separate process, so a PROTOCOL override made in the
        # parent after parse_args does not reach it -- it must travel by env.
        "--max-frames", str(int(os.environ.get("E079_MAX_FRAMES") or PROTOCOL["max_frames"])),
        "--subsample", str(PROTOCOL["subsample"]),
        "--jobs", str(PROTOCOL["jobs"]),
        # Recorded, never used for selection: build_variants is pinned above.
        "--ablate", "vote",
    ]
    if PROTOCOL.get("debug_csv"):
        argv.append("--debug-csv")
    if os.environ.get("VOTESWEEP_TARGET_ID"):
        argv += ["--target-id", os.environ["VOTESWEEP_TARGET_ID"]]
    print("[vote_sweep] child tgtrack=%s arm=%s" % (tg.__file__, spec["arm"]), flush=True)
    return int(rt.main(argv))


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def heldout_sequences(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        raise SystemExit("[vote_sweep] no such directory: %s" % data_root)
    found = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    present = [
        name for name in found
        if (data_root / name / "gt" / "gt.txt").is_file() or (data_root / name / "img1").is_dir()
    ]
    dev = DEV_SEQUENCES if DEV_CHECK else set()
    seqs = [name for name in present if name not in dev]
    leaked = sorted(dev & set(seqs))
    if leaked:
        raise SystemExit("[vote_sweep] FATAL: dev sequence(s) %s in held-out selection" % leaked)
    if not seqs:
        raise SystemExit("[vote_sweep] FATAL: no sequences under %s" % data_root)
    return seqs


# --------------------------------------------------------------------------- #
# Stats (same estimator as scripts/gate_sweep.py / absorb_sweep.py)
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
    usable = [r for r in rows if r.get(key) is not None]
    n = len(usable)
    if n == 0:
        return {"reps": reps, "n_sequences": 0, "p2.5": None, "p50": None, "p97.5": None}
    vals = [float(r[key]) for r in usable]
    weights = [float(r["n_report_frames"]) for r in usable]
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
    ref = {r["sequence"]: float(r[key]) for r in ref_rows
           if r.get("status") == "INCLUDED" and r.get(key) is not None}
    pairs: list[tuple[str, float, float]] = []
    unpaired: list[str] = []
    for r in arm_rows:
        seq = r["sequence"]
        if r.get("status") != "INCLUDED" or seq not in ref or r.get(key) is None:
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
def _frame_dump_stats(csv_path: Path) -> dict[str, Any]:
    """Derive the pre-registered engagement share from the per-frame dump.

    ``blind AND present`` is the E076 / E077 failure population: the tracker makes no
    claim (``locked`` false) while the target is on screen.  The share is over *all*
    dumped frames, exactly like the 29.9 % E077-PREP figure.
    """
    out: dict[str, Any] = {"frame_dump": str(csv_path) if csv_path.is_file() else None}
    if not csv_path.is_file():
        out.update(rows=0, blind_present_frames=None, blind_present_share=None,
                   band_sat_share=None, vote_frame_share=None)
        return out
    rows = 0
    blind_present = 0
    sat = 0
    band_seen = 0
    vote_frames = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                present = str(row.get("gt_present", "")).strip().lower() == "true"
                locked = str(row.get("locked", "")).strip().lower() == "true"
                if present and not locked:
                    blind_present += 1
                sat_raw = str(row.get("anchor_band_sat", "")).strip().lower()
                if sat_raw in ("true", "false"):
                    band_seen += 1
                    sat += int(sat_raw == "true")
                try:
                    vote_frames += int(float(str(row.get("vote_n", "") or 0)))
                except ValueError:
                    pass
    except OSError:
        rows = 0
    out.update(
        rows=rows,
        blind_present_frames=blind_present if rows else None,
        blind_present_share=round(blind_present / rows, 9) if rows else None,
        band_sat_share=round(sat / band_seen, 9) if band_seen else None,
        band_frames_in_dump=band_seen,
        vote_n_in_dump=vote_frames,
        vote_frame_share=round(min(1.0, vote_frames / rows), 9) if rows else None,
    )
    return out


def run_pair(arm_id: str, seq: str, *, target_id: int | None, force: bool,
             log_dir: Path) -> dict[str, Any]:
    spec = arm_spec(arm_id)
    out_dir = OUT_ROOT / "runs" / arm_id / seq
    summary_path = out_dir / "summary.json"
    rec_path = OUT_ROOT / "summaries" / arm_id / ("%s.json" % seq)
    log_path = log_dir / ("%s__%s.log" % (arm_id, seq))
    specs_path = out_dir / "specs.json"
    owner_path = out_dir / ".mine"
    dump_path = out_dir / "runs" / ("%s__%s__%s.csv" % (seq, spec["variant"], PROTOCOL["mask"]))

    base: dict[str, Any] = {
        "arm": arm_id,
        "k": spec["k"],
        "reference": bool(spec["reference"]),
        "sequence": seq,
        "target_id_forced": target_id,
        "summary_path": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
        "log": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
    }

    if summary_path.is_file() and not force and owner_path.is_file():
        cached = _parse_summary(summary_path, seq, target_id, specs_path, arm_id, dump_path)
        if cached.get("status") == "INCLUDED":
            cached.update(base)
            cached["from_cache"] = True
            return cached
    elif summary_path.is_file() and not force:
        print("[vote_sweep] PURGE foreign run dir (no .mine provenance): %s/%s"
              % (arm_id, seq), flush=True)

    env = dict(os.environ)
    env[CHILD_ENV] = seq
    env["VOTESWEEP_ARM"] = arm_id
    env["VOTESWEEP_VARIANT"] = spec["variant"]
    env["VOTESWEEP_OVERRIDES"] = json.dumps(spec["overrides"])
    env["VOTESWEEP_OUT"] = str(out_dir)
    env["VOTESWEEP_SPECS"] = str(specs_path)
    env["PYTHONIOENCODING"] = "utf-8"
    if target_id is not None:
        env["VOTESWEEP_TARGET_ID"] = str(int(target_id))
    else:
        env.pop("VOTESWEEP_TARGET_ID", None)

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
                json.dumps({k: v for k, v in env.items() if k.startswith("VOTESWEEP_") or k == "E079_SRC"},
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
            rec = _parse_summary(summary_path, seq, target_id, specs_path, arm_id, dump_path)
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


def _vote_counters(stats: dict[str, Any] | None) -> dict[str, Any]:
    """The E079 engagement counters out of ``TargetTracker.stats()``.

    Missing counters are reported as -1, never as a silent zero: "the vote never saw a
    vetoed candidate" and "the vote was never installed" must stay distinguishable.
    """
    if not isinstance(stats, dict):
        return {"available": False, "vote_mode": None, "vote_alpha": None,
                "vote_frames": -1, "vote_n": -1, "vote_r1": -1, "vote_r2": -1, "vote_r3": -1,
                "vote_rescued": -1, "vote_rescued_accepted": -1, "vote_r2_updates": -1,
                "band_frames": -1, "band_saturated": -1, "band_saturation": None,
                "vote_contrast_n": -1, "vote_contrast_blocked": -1,
                "vote_contrast_mean": None}
    sat = stats.get("anchor_band_saturation")
    return {
        "available": True,
        "vote_mode": stats.get("anchor_vote"),
        "vote_alpha": stats.get("anchor_vote_alpha"),
        "vote_frames": int(stats.get("vote_frames", -1)),
        "vote_n": int(stats.get("vote_n", -1)),
        "vote_r1": int(stats.get("vote_r1", -1)),
        "vote_r2": int(stats.get("vote_r2", -1)),
        "vote_r3": int(stats.get("vote_r3", -1)),
        "vote_rescued": int(stats.get("vote_rescued", -1)),
        "vote_rescued_accepted": int(stats.get("vote_rescued_accepted", -1)),
        "vote_r2_updates": int(stats.get("vote_r2_updates", -1)),
        "band_frames": int(stats.get("anchor_band_frames", -1)),
        "band_saturated": int(stats.get("anchor_band_saturated", -1)),
        "band_saturation": (None if sat is None else round(float(sat), 6)),
        # E082: the contrast axis.  -1 means "this run predates the E082 counters"
        # (E079/E081 summaries on disk), which is NOT the same as "the gate never
        # bound" -- that is a 0.  The `vote_contrast_ge_*` retention counts are
        # carried through as a dynamic key family so the probe grid lives in one
        # place (`_CONTRAST_PROBES` in target_tracker.py).
        "vote_contrast_n": int(stats.get("vote_contrast_n", -1)),
        "vote_contrast_blocked": int(stats.get("vote_contrast_blocked", -1)),
        "vote_contrast_mean": (None if stats.get("vote_contrast_mean") is None
                               else round(float(stats["vote_contrast_mean"]), 6)),
        **{str(k): int(v) for k, v in stats.items()
           if str(k).startswith("vote_contrast_ge_")},
    }


def _reference_check(seq: str, tsr: float, far: float, target_id: int | None) -> dict[str, Any] | None:
    """Compare one `vote_off` run against the E070 `margin_0.20` reference numbers."""
    if os.environ.get("VOTE_DATA_ROOT"):
        # E085 -- a custom root is a different corpus; ``REFERENCE_DIR`` holds val
        # numbers under the same sequence NAMES, so comparing would be a name
        # collision dressed up as a reproduction check.
        return None
    if PROTOCOL["max_frames"] != 300:
        # E082b's 600-frame epoch has no 300-frame counterpart to reproduce, so a
        # comparison here would report a mismatch that means nothing.
        return None
    ref_path = REFERENCE_DIR / ("%s.json" % seq)
    if not ref_path.is_file():
        return None
    try:
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    r_tsr = ref.get("tsr")
    r_far = ref.get("far")
    r_id = ref.get("target_id")
    d_tsr = None if r_tsr is None else abs(float(tsr) - float(r_tsr))
    d_far = None if r_far is None else abs(float(far) - float(r_far))
    return {
        "source": str(ref_path.relative_to(ROOT)) if ref_path.is_relative_to(ROOT) else str(ref_path),
        "reference_tsr": r_tsr,
        "reference_far": r_far,
        "reference_target_id": r_id,
        "reference_id_consistency": ref.get("id_consistency"),
        "reference_n_report_frames": ref.get("n_report_frames"),
        "abs_dtsr": None if d_tsr is None else round(d_tsr, 12),
        "abs_dfar": None if d_far is None else round(d_far, 12),
        "tsr_match": None if d_tsr is None else bool(d_tsr == 0.0),
        "far_match": None if d_far is None else bool(d_far == 0.0),
        "target_id_match": None if r_id is None else bool(target_id == r_id),
    }


def _parse_summary(summary_path: Path, seq: str, forced_id: int | None,
                   specs_path: Path, arm_id: str, dump_path: Path) -> dict[str, Any]:
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
            print("[vote_sweep] WARNING %s: salvaged summary.json, ignored %d trailing chars (%s)"
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
    counters = _vote_counters(_record_stats(summary_path, row))
    dump = _frame_dump_stats(dump_path)
    rec: dict[str, Any] = {
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
    rec.update(dump)
    if arm_id == REFERENCE_ARM:
        rec["reference_check"] = _reference_check(seq, rec["tsr"], rec["far"], target_id)
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
    for key in POOL_KEYS:
        pairs = [(float(r[key]), float(r["n_report_frames"])) for r in inc
                 if r.get(key) is not None]
        if not pairs:
            pooled[key] = None
            continue
        vals = [v for v, _w in pairs]
        ws = [w for _v, w in pairs]
        pooled[key] = round(_weighted(vals, ws), 6)
        pooled["%s_mean" % key] = round(sum(vals) / len(vals), 6)
        pooled["%s_median" % key] = round(_pct(vals, [50.0])[0], 6)
    eng: dict[str, Any] = {
        "available": bool(inc) and bool(inc[0].get("available", False)),
        "sequences_with_any_veto": 0,
        "sequences_with_any_rescue": 0,
        "sequences_with_any_r2_update": 0,
    }
    for key in ("vote_frames", "vote_n", "vote_r1", "vote_r2", "vote_r3", "vote_rescued",
                "vote_rescued_accepted", "vote_r2_updates", "band_frames", "band_saturated"):
        eng[key] = sum(max(0, int(r.get(key, -1) or 0)) for r in inc)
    # E082 -- the contrast axis, pooled over the same population.  `contrast_n` is
    # the number of ELIGIBLE candidates (those the blanket vote would rescue), so
    # the `ge_*` counts divided by it are the retention curve the delta grid is
    # read against, and `blocked / n` is the share of E079's rescues the gate
    # removed.  blocked == 0 with a non-zero `vote_rescued` is the falsifier: the
    # gate never bound and the arm is E079 under another name.
    ge_keys = sorted({str(k) for r in inc for k in r
                      if str(k).startswith("vote_contrast_ge_")})
    eng["contrast_n"] = sum(max(0, int(r.get("vote_contrast_n", -1) or 0)) for r in inc)
    eng["contrast_blocked"] = sum(
        max(0, int(r.get("vote_contrast_blocked", -1) or 0)) for r in inc)
    for key in ge_keys:
        count = sum(max(0, int(r.get(key, 0) or 0)) for r in inc)
        eng[key] = count
        eng[key.replace("vote_contrast_ge_", "contrast_keep_")] = (
            round(count / eng["contrast_n"], 6) if eng["contrast_n"] > 0 else None)
    if eng["contrast_n"] > 0:
        eng["contrast_blocked_share"] = round(
            eng["contrast_blocked"] / eng["contrast_n"], 6)
    eng["sequences_with_any_veto"] = sum(1 for r in inc if int(r.get("vote_n", 0) or 0) > 0)
    eng["sequences_with_any_rescue"] = sum(1 for r in inc if int(r.get("vote_rescued", 0) or 0) > 0)
    eng["sequences_with_any_r2_update"] = sum(
        1 for r in inc if int(r.get("vote_r2_updates", 0) or 0) > 0)
    if eng["vote_n"] > 0:
        eng["r1_pass_rate"] = round(eng["vote_r1"] / eng["vote_n"], 6)
        eng["r2_pass_rate"] = round(eng["vote_r2"] / eng["vote_n"], 6)
        eng["r3_pass_rate"] = round(eng["vote_r3"] / eng["vote_n"], 6)
        eng["rescue_rate"] = round(eng["vote_rescued"] / eng["vote_n"], 6)
    if eng["band_frames"] > 0:
        eng["band_saturation"] = round(eng["band_saturated"] / eng["band_frames"], 6)
    return {
        "included": inc,
        "excluded": exc,
        "pooled": pooled,
        "bootstrap": {key: cluster_bootstrap(inc, key=key) for key in POOL_KEYS},
        "engagement": eng,
    }


def verdict(arm_entry: dict[str, Any], ref_entry: dict[str, Any]) -> dict[str, Any]:
    """The E079 rule, applied verbatim and never re-tuned.

    SHIP iff ALL of: (1) pooled TSR >= +3 pp with the paired 95 % CI excluding 0;
    (2) the paired dFAR 95 % CI UPPER bound < +3 pp; (3) the `blind AND present`
    share falls versus the reference arm.
    """
    out: dict[str, Any] = {
        "arm": arm_entry["arm"],
        "k": arm_entry["k"],
        "rule": ("SHIP iff (1) pooled TSR >= +%.1f pp with the paired 95%% CI excluding 0, "
                 "AND (2) paired dFAR 95%% CI upper < +%.1f pp, AND (3) the blind AND present "
                 "frame share falls versus %s; otherwise KILL"
                 % (GAIN_PP, FAR_MAX_PP, REFERENCE_ARM)),
    }
    if arm_entry["arm"] == ref_entry["arm"]:
        out.update(verdict="REFERENCE", text="Reference arm: the shipped one-reference veto.")
        return out
    if not arm_entry.get("pooled") or not ref_entry.get("pooled"):
        out.update(verdict="KILL", text="KILL: no complete pool (excluded/error runs).")
        return out

    p = arm_entry["pooled"]
    r = ref_entry["pooled"]
    d = {k: (float(p[k]) - float(r[k])) * 100.0 for k in METRICS
         if p.get(k) is not None and r.get(k) is not None}
    pb = arm_entry["paired"]
    ci = {k: ((pb[k]["p2.5"] or 0.0) * 100.0, (pb[k]["p97.5"] or 0.0) * 100.0) for k in METRICS}
    far_ci_hi = ci["far"][1]
    eng = arm_entry.get("engagement") or {}

    share_arm = p.get("blind_present_share")
    share_ref = r.get("blind_present_share")
    share_delta = None
    if share_arm is not None and share_ref is not None:
        share_delta = (float(share_arm) - float(share_ref)) * 100.0

    ok_tsr = bool(d.get("tsr", -1e9) >= GAIN_PP and pb["tsr"].get("excludes_zero")
                  and ci["tsr"][0] > 0.0)
    ok_far = bool(far_ci_hi < FAR_MAX_PP)
    ok_eng = bool(share_delta is not None and share_delta < 0.0)
    ship = bool(ok_tsr and ok_far and ok_eng)
    inert = int(eng.get("vote_n", 0) or 0) == 0 and int(eng.get("vote_rescued", 0) or 0) == 0

    out.update(
        verdict="SHIP" if ship else "KILL",
        pooled_dtsr_pp=round(d.get("tsr", float("nan")), 3),
        pooled_dfar_pp=round(d.get("far", float("nan")), 3),
        pooled_did_consistency_pp=round(d.get("id_consistency", float("nan")), 3),
        paired_dtsr_mean_pp=round((pb["tsr"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_dtsr_ci_pp=[round(ci["tsr"][0], 3), round(ci["tsr"][1], 3)],
        paired_dfar_mean_pp=round((pb["far"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_dfar_ci_pp=[round(ci["far"][0], 3), round(ci["far"][1], 3)],
        paired_did_mean_pp=round((pb["id_consistency"]["delta_mean"] or 0.0) * 100.0, 3),
        paired_did_ci_pp=[round(ci["id_consistency"][0], 3), round(ci["id_consistency"][1], 3)],
        paired_blind_share_mean_pp=(None if pb.get("blind_present_share", {}).get("delta_mean") is None
                                    else round(pb["blind_present_share"]["delta_mean"] * 100.0, 3)),
        paired_blind_share_ci_pp=(None if pb.get("blind_present_share", {}).get("p2.5") is None else
                                  [round(pb["blind_present_share"]["p2.5"] * 100.0, 3),
                                   round(pb["blind_present_share"]["p97.5"] * 100.0, 3)]),
        pooled_blind_share_pp=None if share_delta is None else round(share_delta, 3),
        blinded_share_arm=None if share_arm is None else round(float(share_arm) * 100.0, 3),
        blinded_share_reference=None if share_ref is None else round(float(share_ref) * 100.0, 3),
        paired_dtsr_excludes_zero=bool(pb["tsr"].get("excludes_zero")),
        paired_dfar_excludes_zero=bool(pb["far"].get("excludes_zero")),
        paired_did_excludes_zero=bool(pb["id_consistency"].get("excludes_zero")),
        far_ci_upper_pp=round(far_ci_hi, 3),
        far_ci_upper_ok=bool(ok_far),
        tsr_ok=bool(ok_tsr),
        engagement_ok=bool(ok_eng),
        mechanism_inert=bool(inert),
        sequences_up_tsr=int(pb["tsr"]["sequences_up"]),
        sequences_down_tsr=int(pb["tsr"]["sequences_down"]),
        sequences_flat_tsr=int(pb["tsr"]["sequences_flat"]),
        sequences_up_id=int(pb["id_consistency"]["sequences_up"]),
        sequences_down_id=int(pb["id_consistency"]["sequences_down"]),
        sequences_flat_id=int(pb["id_consistency"]["sequences_flat"]),
        sequences_up_far=int(pb["far"]["sequences_up"]),
        sequences_down_far=int(pb["far"]["sequences_down"]),
        sequences_flat_far=int(pb["far"]["sequences_flat"]),
    )
    if ship:
        why = ("pooled dTSR %+.2f pp with paired CI [%+.2f, %+.2f] pp excluding 0; paired dFAR "
               "CI upper %+.2f pp < +%.1f pp; blind AND present share %+.2f pp"
               % (d.get("tsr", 0.0), ci["tsr"][0], ci["tsr"][1], far_ci_hi, FAR_MAX_PP,
                  share_delta or 0.0))
    else:
        bad = []
        if not ok_tsr:
            bad.append("pooled dTSR %+.2f pp (CI [%+.2f, %+.2f]) fails the >= +%.1f pp bar with a "
                       "CI excluding 0"
                       % (d.get("tsr", 0.0), ci["tsr"][0], ci["tsr"][1], GAIN_PP))
        if not ok_far:
            bad.append("paired dFAR CI upper %+.2f pp >= +%.1f pp" % (far_ci_hi, FAR_MAX_PP))
        if not ok_eng:
            bad.append("blind AND present share did not fall (pooled %s pp)"
                       % ("n/a" if share_delta is None else "%+.2f" % share_delta))
        why = "; ".join(bad)
        if inert:
            why += ("; the vote never engaged (%d vetoed candidates, %d rescued)"
                    % (int(eng.get("vote_n", 0) or 0), int(eng.get("vote_rescued", 0) or 0)))
    out["text"] = "%s: %s." % (out["verdict"], why)
    return out


def control_check(arm_entry: dict[str, Any], ref_entry: dict[str, Any]) -> dict[str, Any]:
    """`vote_unanimous` must be arithmetically identical to `vote_off`, per sequence."""
    ref = {r["sequence"]: r for r in ref_entry["rows"] if r.get("status") == "INCLUDED"}
    diffs = []
    for r in arm_entry["rows"]:
        if r.get("status") != "INCLUDED" or r["sequence"] not in ref:
            continue
        o = ref[r["sequence"]]
        diffs.append({
            "sequence": r["sequence"],
            "dtsr": abs(float(r["tsr"]) - float(o["tsr"])),
            "dfar": abs(float(r["far"]) - float(o["far"])),
            "did": abs(float(r["id_consistency"]) - float(o["id_consistency"])),
            "target_id_same": bool(r["target_id"] == o["target_id"]),
        })
    return {
        "identical": bool(diffs) and all(
            d["dtsr"] == 0.0 and d["dfar"] == 0.0 and d["did"] == 0.0 and d["target_id_same"]
            for d in diffs),
        "n_compared": len(diffs),
        "max_abs_dtsr": max((d["dtsr"] for d in diffs), default=None),
        "max_abs_dfar": max((d["dfar"] for d in diffs), default=None),
        "max_abs_did": max((d["did"] for d in diffs), default=None),
        "per_sequence": diffs,
    }


def verdict_contrast(
    arm_entry: dict[str, Any],
    ref_entry: dict[str, Any],
    majority_entry: dict[str, Any] | None = None,
    *,
    sequences: Sequence[str] | None = None,
) -> dict[str, Any]:
    """E082's own pre-registered rule -- NOT the E079 rule above.

    ``docs/E082_PLAN.md`` asks a different question from E079 (when is the click crop
    the wrong reference, rather than when is the tracker uncertain), so it states its
    own bar, evaluated against ``vote_off``:

      1. paired dTSR > 0 with the paired 95 % CI excluding 0; and
      2. paired dFAR < 0 with the paired 95 % CI upper bound below +3 pp.

    Clause 1 is deliberately weaker than E079's ``+3 pp`` bar: E079 already
    established the point estimate (+5.88 pp) and missed only because its interval
    reached below zero, so E082 is a PRECISION experiment -- it wins by keeping the
    sign with a tighter interval, not by beating +3 pp a second time.

    ``sequences`` restricts the paired bootstrap to a named half (the TEST half is the
    one that may decide; the EXPLORE half is reported for the record).  ``None`` means
    every paired sequence.  ``majority_entry`` adds the mechanism falsifier: an arm
    bit-identical to ``vote_majority`` means the contrast never bound.
    """
    out: dict[str, Any] = {
        "arm": arm_entry["arm"],
        "delta": arm_entry["overrides"].get("cost.anchor_vote_contrast"),
        "subset": ("all" if sequences is None else "%d sequences" % len(sequences)),
        "rule": ("E082: SHIP iff (1) paired dTSR > 0 with the 95 %% CI excluding 0 AND "
                 "(2) paired dFAR < 0 with the paired 95 %% CI upper bound < +%.1f pp, "
                 "versus `%s`" % (FAR_MAX_PP, REFERENCE_ARM)),
    }
    if not arm_entry.get("pooled") or not ref_entry.get("pooled"):
        out.update(verdict="KILL", text="KILL: no complete pool (excluded/errored pairs).")
        return out
    want = None if sequences is None else {str(s) for s in sequences}
    arm_rows = [r for r in arm_entry["rows"] if want is None or r.get("sequence") in want]
    ref_rows = [r for r in ref_entry["rows"] if want is None or r.get("sequence") in want]
    # The identity check is a property of the whole arm, not of the subset, so it is
    # recorded even when the subset is too small to carry a verdict.
    if majority_entry is not None and majority_entry.get("pooled"):
        ident = control_check(arm_entry, majority_entry)
        out["identical_to_majority"] = bool(ident["identical"])
        out["n_compared_to_majority"] = int(ident["n_compared"])
        out["max_abs_dtsr_vs_majority"] = ident["max_abs_dtsr"]
        out["max_abs_dfar_vs_majority"] = ident["max_abs_dfar"]
        out["max_abs_did_vs_majority"] = ident["max_abs_did"]
        out["per_sequence_vs_majority"] = ident["per_sequence"]
    pb = {key: paired_bootstrap(arm_rows, ref_rows, key=key) for key in POOL_KEYS}
    if not pb["tsr"].get("n_paired"):
        out.update(verdict="KILL", n_paired=0,
                   text="KILL: no paired sequence in this subset.")
        return out
    if int(pb["tsr"]["n_paired"]) < 3:
        # A bootstrap over one or two sequences has no interval worth quoting; the
        # E081 close-out learned this the hard way (a 1-sequence "CI" printed as if it
        # were evidence).  Refuse to claim a verdict rather than overstate one.
        out.update(verdict="INSUFFICIENT", n_paired=int(pb["tsr"]["n_paired"]),
                   text="INSUFFICIENT: only %d paired sequence(s) in this subset; a verdict "
                        "needs at least 3." % int(pb["tsr"]["n_paired"]))
        return out
    tsr, far, dis = pb["tsr"], pb["far"], pb["id_consistency"]
    dtsr = float(tsr["delta_mean"]) * 100.0
    dfar = float(far["delta_mean"]) * 100.0
    tsr_ci = (float(tsr["p2.5"]) * 100.0, float(tsr["p97.5"]) * 100.0)
    far_ci = (float(far["p2.5"]) * 100.0, float(far["p97.5"]) * 100.0)
    ok_tsr = bool(dtsr > 0.0 and tsr_ci[0] > 0.0)
    ok_far = bool(dfar < 0.0 and far_ci[1] < FAR_MAX_PP)
    ship = bool(ok_tsr and ok_far)
    out.update(
        verdict="SHIP" if ship else "KILL",
        n_paired=int(tsr["n_paired"]),
        paired_dtsr_mean_pp=round(dtsr, 3),
        paired_dtsr_ci_pp=[round(tsr_ci[0], 3), round(tsr_ci[1], 3)],
        paired_dtsr_excludes_zero=bool(tsr.get("excludes_zero")),
        paired_dfar_mean_pp=round(dfar, 3),
        paired_dfar_ci_pp=[round(far_ci[0], 3), round(far_ci[1], 3)],
        paired_dfar_excludes_zero=bool(far.get("excludes_zero")),
        paired_did_mean_pp=round(float(dis["delta_mean"]) * 100.0, 3),
        paired_did_ci_pp=[round(float(dis["p2.5"]) * 100.0, 3),
                          round(float(dis["p97.5"]) * 100.0, 3)],
        tsr_ok=ok_tsr,
        far_ok=ok_far,
        sequences_up_tsr=int(tsr["sequences_up"]),
        sequences_down_tsr=int(tsr["sequences_down"]),
        sequences_flat_tsr=int(tsr["sequences_flat"]),
        sequences_up_far=int(far["sequences_up"]),
        sequences_down_far=int(far["sequences_down"]),
        sequences_flat_far=int(far["sequences_flat"]),
        per_sequence_tsr=tsr["per_sequence"],
    )
    if majority_entry is not None and majority_entry.get("pooled"):
        ident = control_check(arm_entry, majority_entry)
        out["identical_to_majority"] = bool(ident["identical"])
        out["n_compared_to_majority"] = int(ident["n_compared"])
        out["max_abs_dtsr_vs_majority"] = ident["max_abs_dtsr"]
        out["max_abs_dfar_vs_majority"] = ident["max_abs_dfar"]
        out["max_abs_did_vs_majority"] = ident["max_abs_did"]
        out["per_sequence_vs_majority"] = ident["per_sequence"]
    if ship:
        why = ("paired dTSR %+.2f pp [%+.2f, %+.2f] excludes 0, paired dFAR %+.2f pp "
               "[%+.2f, %+.2f] with upper bound %+.2f pp < +%.1f pp"
               % (dtsr, tsr_ci[0], tsr_ci[1], dfar, far_ci[0], far_ci[1], far_ci[1], FAR_MAX_PP))
    else:
        bad = []
        if not ok_tsr:
            bad.append("paired dTSR %+.2f pp [%+.2f, %+.2f] does not keep a positive sign with a "
                       "CI excluding 0" % (dtsr, tsr_ci[0], tsr_ci[1]))
        if not ok_far:
            bad.append("paired dFAR %+.2f pp [%+.2f, %+.2f] is not a gain with an upper bound "
                       "< +%.1f pp" % (dfar, far_ci[0], far_ci[1], FAR_MAX_PP))
        why = "; ".join(bad)
    out["text"] = "%s: %s." % (out["verdict"], why)
    return out


def contrast_split(seqs: Sequence[str]) -> tuple[list[str], list[str]]:
    """The EXPLORE / TEST halves, exactly as `scripts/vote_analyze.py` prints them.

    EXPLORE is the ODD positions of the sorted held-out list, TEST the EVEN ones;
    the same 11/11 split E081 selected on, so the E082 arms are scored on a half the
    deltas were not chosen from.
    """
    ordered = sorted(seqs)
    explore = [s for i, s in enumerate(ordered) if i % 2 == 1]
    test = [s for i, s in enumerate(ordered) if i % 2 == 0]
    return explore, test


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _num(value: Any, digits: int = 2) -> str:
    """A signed pp value, or ``n/a`` when the subset had no usable interval."""
    if value is None:
        return "n/a"
    v = float(value)
    if v != v:  # NaN -- an INSUFFICIENT subset has no delta to quote
        return "n/a"
    return "%+.*f" % (digits, v)


def _ci(lo: Any, hi: Any, digits: int = 2) -> str:
    if lo is None or hi is None:
        return ""
    return " [%s, %s]" % (_num(lo, digits), _num(hi, digits))


def _fmt(value: float | None, boot: dict[str, Any] | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if not boot or boot.get("p2.5") is None:
        return "%.*f" % (digits, value)
    return "%.*f [%.*f, %.*f]" % (digits, value, digits, boot["p2.5"], digits, boot["p97.5"])


def write_markdown(payload: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a("# E079 -- the three-reference anchor vote on the held-out DanceTrack split")
    a("")
    a("<!-- generated by scripts/vote_sweep.py; every number is read from %s -->"
      % REPORT_JSON.name)
    a("")
    verdicts = payload["verdicts"]
    a("## Verdict (pre-registered rule)")
    a("")
    a("> %s" % payload["rule_text"])
    a("")
    for arm in payload["order"][1:]:
        v = verdicts[arm]
        a("- **`%s`** (k = %d) -- **%s**: %s" % (arm, v["k"], v["verdict"], v["text"]))
    cand = [arm for arm in payload["order"][1:]
            if verdicts[arm]["verdict"] == "SHIP" and arm != "vote_unanimous"]
    a("")
    if cand:
        a("**Ship decision: `%s` clears the row. The default stays OFF until the change is "
          "applied by hand (E079 ships the knob, not the behaviour).**" % ", ".join(cand))
    else:
        a("**Ship decision: kill the mechanism; the shipped default (`anchor_vote = \"off\"`) "
          "stays.**")
    a("")
    cc = payload.get("control")
    if cc:
        a("Control: `vote_unanimous` is %s to `vote_off` on %d paired sequences "
          "(max |dTSR| %s, max |dFAR| %s, max |dID| %s). A candidate only reaches the vote "
          "because R1 already rejected it, so k = 3 is unreachable -- identical is the "
          "**expected** result, not a failed implementation."
          % ("bit-identical" if cc["identical"] else "NOT identical", cc["n_compared"],
             cc["max_abs_dtsr"], cc["max_abs_dfar"], cc["max_abs_did"]))
        a("")

    a("## Off-by-default no-op proof (`vote_off` vs the E070 `margin_0.20` reference)")
    a("")
    noop = payload.get("reference_noop") or {}
    if not noop or not noop.get("n_compared"):
        a("_Not checked: no `results/gate_sweep/summaries/margin_0.20/` reference was found "
          "for the sequences in this run._")
    else:
        a("%d of %d `vote_off` runs compared; **%d exact TSR matches, %d exact FAR matches, "
          "%d target-id matches**, max |dTSR| %s, max |dFAR| %s."
          % (noop["n_compared"], noop["n_sequences"],
             noop["n_tsr_match"], noop["n_far_match"], noop["n_target_id_match"],
             noop["max_abs_dtsr"], noop["max_abs_dfar"]))
        if noop.get("mismatches"):
            a("")
            for m in noop["mismatches"]:
                a("- MISMATCH `%s`: TSR %.6f vs %.6f, FAR %.6f vs %.6f, id %r vs %r"
                  % (m["sequence"], m["tsr"], m["reference_tsr"], m["far"], m["reference_far"],
                     m["target_id"], m["reference_target_id"]))
    a("")

    a("## Protocol (identical to E070 / E071 / E078, so the numbers stay comparable)")
    a("")
    prot = payload["protocol"]
    a("- data: `data/raw/dancetrack/val`, the **%d** held-out sequences; frozen dev split %s "
      "asserted absent" % (payload["n_heldout"], ", ".join(sorted(DEV_SEQUENCES))))
    a("- `--max-frames %d --subsample %d --mask %s --jobs %d --debug-csv`, shipped defaults "
      "elsewhere" % (prot["max_frames"], prot["subsample"], prot["mask"], prot["jobs"]))
    a("- one `--ablate vote` arm per run, `run_target.build_variants` replaced in the child; "
      "`specs.json` asserts exactly 1 spec with exactly the intended overrides and each "
      "`summary.json` reports exactly 1 variant")
    a("- target identity: the reference arm picks it (`stressed`), every other arm is pinned to "
      "the same id and the mismatch is a hard error")
    a("- pooling: frame-weighted mean over `n_report_frames`; CIs are cluster bootstraps over "
      "**sequences** (%d reps, seed %d); deltas are **paired** per sequence against the "
      "reference arm" % (payload["bootstrap"]["reps"], payload["bootstrap"]["seed"]))
    if payload.get("partial"):
        a("- **PARTIAL COVERAGE**: %d of %d sequences ran (%s)"
          % (len(payload["sequences_run"]), payload["n_heldout"], payload.get("partial_reason")))
        cov = payload.get("coverage") or {}
        if cov:
            a("- per-arm coverage: %s" % ", ".join(
                "%s %d/%d" % ("`%s`" % arm, cov[arm]["included"], cov[arm]["of"])
                for arm in cov))
        a("- each arm is pooled over its own coverage; paired deltas use the sequences it shares "
          "with the reference arm, so a partial arm's pooled row is NOT comparable to the "
          "reference pool")
    a("- runs: %d (including any retried pair); wall %.1f min" % (payload["n_pairs_run"],
                                                                payload["wall_seconds"] / 60.0))
    a("")

    a("## Per-arm results")
    a("")
    a("| arm | k | sequences | report frames | TSR | FAR | id_consistency | blind AND present |")
    a("|---|---|---|---|---|---|---|---|")
    for arm in payload["order"]:
        e = payload["arms"][arm]
        p = e["pooled"]
        if not p:
            a("| `%s` | %d | 0 | 0 | n/a | n/a | n/a | n/a |" % (arm, e["k"]))
            continue
        b = e["bootstrap"]
        a("| `%s` | %d | %d | %d | %s | %s | %s | %s |"
          % (arm, e["k"], p["n_included"], p["total_report_frames"],
             _fmt(p["tsr"], b["tsr"]), _fmt(p["far"], b["far"]),
             _fmt(p["id_consistency"], b["id_consistency"]),
             _fmt(None if p.get("blind_present_share") is None else p["blind_present_share"],
                  b.get("blind_present_share"), digits=3)))
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
            ("tsr", "dTSR", "paired_dtsr_mean_pp", "paired_dtsr_ci_pp", "tsr"),
            ("far", "dFAR", "paired_dfar_mean_pp", "paired_dfar_ci_pp", "far"),
            ("id_consistency", "dID", "paired_did_mean_pp", "paired_did_ci_pp", "id"),
        ):
            pooled_key = {"tsr": "pooled_dtsr_pp", "far": "pooled_dfar_pp",
                          "id_consistency": "pooled_did_consistency_pp"}[key]
            a("| `%s` | %s | %+.2f pp | %+.2f pp | [%+.2f, %+.2f] pp | %d / %d / %d | %s |"
              % (arm, label, v.get(pooled_key, float("nan")), v.get(dm, float("nan")),
                   *(v.get(ci) or [float("nan"), float("nan")]),
                 v.get("sequences_up_" + up, 0), v.get("sequences_down_" + up, 0),
                   v.get("sequences_flat_" + up, 0),
                 "yes" if v.get("paired_d%s_excludes_zero" % up) else "no"))
    a("")

    e082 = payload.get("e082")
    if e082:
        a("## E082 -- the contrast gate, judged by its own pre-registered rule")
        a("")
        a("> %s" % e082["rule_text"])
        a("")
        a("`vote_contrast_0.00` is a **CONTROL**: a candidate only reaches the vote because R1 "
          "refused it, so its contrast is > 0 and a delta = 0 gate removes nothing -- it must "
          "reproduce `vote_majority` bit-for-bit. If it does not, the gate is wired wrong and no "
          "non-zero delta means anything.")
        a("")
        a("| arm | delta | eligible candidates | gate blocked | TEST dTSR (11 seqs) | TEST dFAR "
          "(11 seqs) | TEST verdict | all-22 dTSR | dID (all) | identical to `vote_majority` |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for arm in payload["order"]:
            entry = e082["arms"].get(arm)
            if not entry:
                continue
            eng = entry.get("engagement") or {}
            t = entry["test"]
            u = entry["all"]
            a("| `%s` | %.2f | %d | %d | %s pp%s | %s pp%s | **%s** | %s pp%s | %s pp | %s |"
              % (arm, entry["delta"], int(eng.get("contrast_n", -1)),
                 int(eng.get("contrast_blocked", -1)),
                 _num(t.get("paired_dtsr_mean_pp")),
                 _ci(*(t.get("paired_dtsr_ci_pp") or [None, None])),
                 _num(t.get("paired_dfar_mean_pp")),
                 _ci(*(t.get("paired_dfar_ci_pp") or [None, None])),
                 t.get("verdict", "n/a"),
                 _num(u.get("paired_dtsr_mean_pp")),
                 _ci(*(u.get("paired_dtsr_ci_pp") or [None, None])),
                 _num(u.get("paired_did_mean_pp")),
                 ("YES" if u.get("identical_to_majority") else "no")
                 if "identical_to_majority" in u else "n/a"))
        a("")
        probe = next((e082["arms"][a_] for a_ in e082["arms"]
                      if int((e082["arms"][a_].get("engagement") or {}).get("contrast_n", 0)) > 0),
                     None)
        if probe:
            eng = probe["engagement"]
            keys = sorted(k for k in eng if str(k).startswith("contrast_keep_"))
            if keys:
                a("Retention curve of the gate, measured on the ELIGIBLE population (the "
                  "candidates the blanket vote would rescue), which is what the delta grid is "
                  "read against:")
                a("")
                a("| `contrast >=` | candidates kept | share of eligible |")
                a("|---|---|---|")
                for key in keys:
                    thr = key.replace("contrast_keep_", "")
                    a("| %s | %d | %.1f %% |"
                      % (thr, int(eng.get("vote_contrast_ge_" + thr, -1) or 0),
                         100.0 * float(eng[key] or 0.0)))
                a("")
        for arm in payload["order"]:
            entry = e082["arms"].get(arm)
            if not entry:
                continue
            ex = entry["explore"]
            a("- `%s` EXPLORE half: dTSR %s pp %s, dFAR %s pp %s (%s, reported, does not decide)."
              % (arm, _num(ex.get("paired_dtsr_mean_pp")),
                 _ci(*(ex.get("paired_dtsr_ci_pp") or [None, None])),
                 _num(ex.get("paired_dfar_mean_pp")),
                 _ci(*(ex.get("paired_dfar_ci_pp") or [None, None])), ex.get("verdict", "n/a")))
        a("")

    a("## Mechanism engagement (a veto that never fires is not a result)")
    a("")
    a("| arm | vetoed candidates | R1 pass | R2 pass | R3 pass | rescued | rescued+accepted | R2 updates | band frames | band at ceiling |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for arm in payload["order"]:
        e = payload["arms"][arm]
        eng = e.get("engagement") or {}
        a("| `%s` | %d | %d | %d | %d | %d | %d | %d | %d | %s |"
          % (arm, eng.get("vote_n", -1), eng.get("vote_r1", -1), eng.get("vote_r2", -1),
             eng.get("vote_r3", -1), eng.get("vote_rescued", -1),
             eng.get("vote_rescued_accepted", -1), eng.get("vote_r2_updates", -1),
             eng.get("band_frames", -1),
             "n/a" if eng.get("band_saturation") is None
             else "%.1f %%" % (100.0 * eng["band_saturation"])))
    a("")
    a("R1 pass is 0 by construction for a vetoed candidate (that is why it was vetoed); it is "
      "reported to prove the vote saw the right population. `rescued` counts candidates that "
      "would have been `+inf` and were left finite by the vote.")
    a("")

    a("## Per-sequence detail")
    a("")
    per: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in payload["order"]:
        for row in payload["arms"][arm]["included"]:
            per.setdefault(row["sequence"], {})[arm] = row
    a("| sequence | frames | " + " | ".join("`%s` TSR / FAR / b&p" % arm for arm in payload["order"]) + " |")
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
            cells.append("%.3f / %.3f / %s"
                         % (row["tsr"], row["far"],
                            "n/a" if row.get("blind_present_share") is None
                            else "%.3f" % row["blind_present_share"]))
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


def _reference_noop_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks = [r["reference_check"] for r in rows
              if r.get("arm") == REFERENCE_ARM and isinstance(r.get("reference_check"), dict)]
    mism = [
        {"sequence": r["sequence"], "tsr": r["tsr"], "far": r["far"],
         "target_id": r["target_id"],
         "reference_tsr": r["reference_check"].get("reference_tsr"),
         "reference_far": r["reference_check"].get("reference_far"),
         "reference_target_id": r["reference_check"].get("reference_target_id")}
        for r in rows
        if r.get("arm") == REFERENCE_ARM and isinstance(r.get("reference_check"), dict)
        and not (r["reference_check"].get("tsr_match")
                 and r["reference_check"].get("far_match")
                 and r["reference_check"].get("target_id_match"))
    ]
    return {
        "source": str(REFERENCE_DIR.relative_to(ROOT)) if REFERENCE_DIR.is_relative_to(ROOT)
        else str(REFERENCE_DIR),
        "n_compared": len(checks),
        "n_sequences": len(checks),
        "n_tsr_match": sum(1 for c in checks if c.get("tsr_match")),
        "n_far_match": sum(1 for c in checks if c.get("far_match")),
        "n_target_id_match": sum(1 for c in checks if c.get("target_id_match")),
        "max_abs_dtsr": max((c["abs_dtsr"] for c in checks if c.get("abs_dtsr") is not None),
                            default=None),
        "max_abs_dfar": max((c["abs_dfar"] for c in checks if c.get("abs_dfar") is not None),
                            default=None),
        "mismatches": mism,
    }


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
            print("[vote_sweep] REFUSING TO START: %s is held by live sweep pid %d. "
                  "One GPU, one worker." % (lock_path, holder), flush=True)
            return False
        print("[vote_sweep] stale lock from dead pid %d -- taking over" % holder, flush=True)
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

    p = argparse.ArgumentParser(description="E079 three-reference anchor vote sweep.")
    p.add_argument("--arms", default="all", help="'all' or comma-separated arm ids")
    p.add_argument("--limit-seqs", type=int, default=None, help="debug: only the first N sequences")
    p.add_argument("--run-only", default=None, help="debug: restrict to one sequence name")
    p.add_argument("--force", action="store_true", help="re-run pairs that already have a summary")
    p.add_argument("--max-frames", type=int, default=None,
                   help="E082b: override PROTOCOL max_frames (needs E079_OUT to a separate dir)")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild summary.json / REPORT.md from the cached per-pair records "
                        "without running anything (missing pairs are counted as NOT_RUN)")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.max_frames is not None:
        # E082b (docs/EXPERIMENTS.md): the power-buying run doubles the source
        # frames.  Overriding PROTOCOL here keeps the stamp, the validator and
        # the report header consistent.  The two epochs must never share an
        # output directory, or a 600-frame pair could be pooled with a
        # 300-frame one silently -- exactly the kind of cross-epoch arithmetic
        # that has already produced two false "improvements" in this campaign.
        if int(args.max_frames) != 300 and not os.environ.get("E079_OUT"):
            print("[vote_sweep] refusing --max-frames != 300 without E079_OUT set "
                  "to a separate output directory", flush=True)
            return 2
        PROTOCOL["max_frames"] = int(args.max_frames)
        # E082b: children re-import this module and would otherwise see the 300 default,
        # write max_frames=300 into their meta, and fail the parent's protocol guard.
        os.environ["E079_MAX_FRAMES"] = str(PROTOCOL["max_frames"])

    print("[vote_sweep] FROZEN DEV SPLIT (never swept, never reported): %s"
          % ", ".join(sorted(DEV_SEQUENCES)), flush=True)

    all_seqs = heldout_sequences(VAL_ROOT)
    print("[vote_sweep] held-out sequences: %d" % len(all_seqs), flush=True)
    seqs = list(all_seqs)
    partial = False
    partial_reason = ""
    if args.run_only:
        if args.run_only not in all_seqs:
            raise SystemExit("[vote_sweep] FATAL: %r is not a held-out sequence" % args.run_only)
        seqs = [args.run_only]
        partial = True
        partial_reason = "--run-only"
    if args.limit_seqs:
        seqs = seqs[: int(args.limit_seqs)]
        partial = len(seqs) < len(all_seqs)
        partial_reason = partial_reason or "--limit-seqs"
    if DEV_CHECK:
        leaked = sorted(DEV_SEQUENCES & set(seqs))
        if leaked:
            raise SystemExit("[vote_sweep] FATAL: dev sequence(s) %s in the run set; refusing." % leaked)
        assert not (DEV_SEQUENCES & set(seqs)), "dev split leaked into the sweep"
    else:
        print("[vote_sweep] dev-split NAME check DISABLED (VOTE_DEV_CHECK=0): this epoch is the "
              "train cohort, whose sequence names collide with the val dev split by number only",
              flush=True)

    arms = all_arms(args.arms)
    if REFERENCE_ARM not in arms:
        raise SystemExit("[vote_sweep] the reference arm %s must be in --arms to pin target ids"
                         % REFERENCE_ARM)
    arms = [REFERENCE_ARM] + [a for a in arms if a != REFERENCE_ARM]

    log_dir = OUT_ROOT / "logs"
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    target_ids: dict[str, int] = {}

    for arm in arms:
        spec = arm_spec(arm)
        for i, seq in enumerate(seqs, 1):
            if args.report_only:
                rec = _load_cached(arm, seq)
                if rec is not None and spec["reference"] and rec.get("status") == "INCLUDED":
                    target_ids[seq] = int(rec["target_id"])
                elif rec is None:
                    rec = {"status": "NOT_RUN", "arm": arm, "k": spec["k"], "sequence": seq,
                           "reason": "cached record missing (--report-only)"}
                rows.append(rec)
                continue
            tid = None if spec["reference"] else target_ids.get(seq)
            if not spec["reference"] and tid is None:
                raise SystemExit("[vote_sweep] no reference target id for %s; run %s first"
                                 % (seq, REFERENCE_ARM))
            ts = time.perf_counter()
            rec = run_pair(arm, seq, target_id=tid, force=bool(args.force), log_dir=log_dir)
            rows.append(rec)
            if spec["reference"] and rec.get("status") == "INCLUDED":
                target_ids[seq] = int(rec["target_id"])
            print("[vote_sweep] %-15s %2d/%2d %-16s %-9s %6.1fs %s"
                  % (arm, i, len(seqs), seq, rec.get("status"), time.perf_counter() - ts,
                     ("TSR=%.3f FAR=%.3f vote_n=%s rescued=%s r2upd=%s"
                      % (rec["tsr"], rec["far"], rec.get("vote_n"), rec.get("vote_rescued"),
                         rec.get("vote_r2_updates")))
                     if rec.get("status") == "INCLUDED" else rec.get("reason", "")), flush=True)
    wall = time.perf_counter() - t0
    not_run = [r for r in rows if r.get("status") == "NOT_RUN"]
    if not_run:
        partial = True
        partial_reason = (partial_reason + "; " if partial_reason else "") + (
            "%d pair(s) have no cached record yet" % len(not_run))

    arms_payload: dict[str, Any] = {}
    for arm in arms:
        spec = arm_spec(arm)
        arm_rows = [r for r in rows if r["arm"] == arm]
        pool = pool_arm(arm_rows)
        arms_payload[arm] = {
            "arm": arm,
            "k": spec["k"],
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
        e["paired"] = {key: paired_bootstrap(e["rows"], ref_entry["rows"], key=key)
                       for key in POOL_KEYS}
        e["verdict"] = verdict(e, ref_entry)

    control = control_check(arms_payload["vote_unanimous"], ref_entry) \
        if "vote_unanimous" in arms_payload else None
    if control is not None:
        arms_payload["vote_unanimous"]["verdict"]["verdict"] = "CONTROL"
        arms_payload["vote_unanimous"]["verdict"]["text"] = (
            "CONTROL (k = 3): %s to vote_off on %d paired sequences; it can never clear the "
            "rule by construction."
            % ("bit-identical" if control["identical"] else "NOT identical", control["n_compared"]))

    # E082 -- the contrast arms are judged by their OWN pre-registered rule.  E079's
    # rule (pooled TSR >= +3 pp) is still computed for them above so the two bars can
    # be read against each other, but it is not the bar `docs/E082_PLAN.md` set: E079
    # already established the point estimate, and E082 is a precision experiment.
    e082: dict[str, Any] | None = None
    contrast_arms = [arm for arm in arms if arm.startswith("vote_contrast_")]
    if contrast_arms:
        explore, test = contrast_split(all_seqs)
        majority = arms_payload.get("vote_majority")
        e082 = {
            "experiment": "E082",
            "source": "docs/E082_PLAN.md (pre-registered)",
            "rule_text": (
                "E082: SHIP `anchor_vote = \"contrast\"` at some delta iff, versus `vote_off` "
                "on the TEST half, (1) the paired dTSR is > 0 with the paired 95 %% CI "
                "excluding 0 AND (2) the paired dFAR is < 0 with the paired 95 %% CI upper "
                "bound below +%.1f pp. KILL if no delta achieves both, or if every delta is "
                "bit-identical to `vote_majority` (the contrast never binds: the axis is "
                "empty). `vote_contrast_0.00` is a control." % FAR_MAX_PP),
            "explore_sequences": explore,
            "test_sequences": test,
            "arms": {},
        }
        for arm in contrast_arms:
            e = arms_payload[arm]
            entry = {
                "delta": float(e["overrides"].get("cost.anchor_vote_contrast", -1.0)),
                "all": verdict_contrast(e, ref_entry, majority),
                "explore": verdict_contrast(e, ref_entry, majority, sequences=explore),
                "test": verdict_contrast(e, ref_entry, majority, sequences=test),
                "engagement": dict(e.get("engagement") or {}),
            }
            e082["arms"][arm] = entry
            e["verdict_e082"] = entry["test"]

    noop = _reference_noop_block(rows)

    notes = [
        "`vote_off` spells out the shipped values (`anchor_vote=\"off\"`, alpha 0.1) so the "
        "reference row is self-explanatory; with the vote off the extra code is arithmetically "
        "inert (`admitted` is all-False, so `veto == over`).",
        "The three references answer the SAME question with the SAME threshold: the E029 "
        "`_anchor_limit()` band, which saturates at its 0.75 ceiling on the majority of frames "
        "(see the engagement table).  No new constant was introduced.",
        "R2 only moves on a frame where R1 AND R3 already accept the same box, so a poisoned "
        "gallery cannot teach the fresh anchor: the anti-drift conjunction is the load-bearing "
        "part of the design, and `vote_r2_updates` is reported to show whether it ever moved.",
        "R1 pass rate is 0 by construction inside the vote population (a candidate is only "
        "counted once R1 rejected it), so a k = 2 admission means R2 AND R3 both passed: the "
        "majority arm is really 'the gallery and the refreshed anchor outvote the click crop'.",
        "`vote_unanimous` is a control, not a candidate: its k = 3 is unreachable for a vetoed "
        "candidate, so identical-to-reference is the expected identification result. If it were "
        "NOT identical, the vote block would be perturbing frames it does not own.",
        "The `blind AND present` share is computed from the `--debug-csv` per-frame dump "
        "(`gt_present and not locked` over all dumped frames), the same population E077-PREP "
        "measured at 29.9 %.",
    ]

    payload: dict[str, Any] = {
        "generated_by": "scripts/vote_sweep.py",
        "experiment": "E079",
        "dev_sequences": sorted(DEV_SEQUENCES),
        "heldout_sequences": all_seqs,
        "n_heldout": len(all_seqs),
        "sequences_run": seqs,
        "partial": bool(partial),
        "partial_reason": partial_reason,
        "n_pairs_run": len(rows),
        "protocol": dict(PROTOCOL),
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "unit": "sequence"},
        "rule_text": ("SHIP `vote_majority` iff ALL THREE: (1) pooled TSR rises >= +3.0 pp with "
                      "the paired 95 % CI excluding 0; (2) the paired dFAR 95 % CI upper bound is "
                      "< +3.0 pp; (3) the mechanism engaged -- the `blind AND present` frame share "
                      "falls versus `vote_off`. Otherwise KILL. `vote_unanimous` is a control."),
        "rule": {"gain_pp": GAIN_PP, "far_max_pp": FAR_MAX_PP,
                 "source": "docs/E079_PLAN.md (pre-registered)"},
        "order": arms,
        "arms": arms_payload,
        "verdicts": {arm: arms_payload[arm]["verdict"] for arm in arms},
        "control": control,
        "e082": e082,
        "reference_noop": noop,
        "excluded": [r for r in rows if r.get("status") != "INCLUDED"],
        "coverage": {arm: {"included": int(arms_payload[arm]["pooled"]["n_included"]) if
                           arms_payload[arm]["pooled"] else 0,
                           "of": len(seqs)}
                     for arm in arms},
        "target_ids": target_ids,
        "wall_seconds": round(wall, 1),
        "code_state": code_state_snapshot(),
        "env": {"E079_SRC": os.environ.get("E079_SRC"), "E079_OUT": os.environ.get("E079_OUT"),
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
            print("[vote_sweep] %-15s TSR=%.4f [%.4f, %.4f] FAR=%.4f [%.4f, %.4f] "
                  "b&p=%s -> %s"
                  % (arm, pooled["tsr"], b["tsr"]["p2.5"], b["tsr"]["p97.5"],
                     pooled["far"], b["far"]["p2.5"], b["far"]["p97.5"],
                     "n/a" if pooled.get("blind_present_share") is None
                     else "%.4f" % pooled["blind_present_share"], v["verdict"]), flush=True)
            if v["verdict"] not in ("REFERENCE", "CONTROL"):
                print("[vote_sweep] %-15s dTSR %+.2f pp %s | dFAR %+.2f pp %s | b&p %s pp"
                      % (arm, v["pooled_dtsr_pp"], v["paired_dtsr_ci_pp"],
                         v["pooled_dfar_pp"], v["paired_dfar_ci_pp"],
                         v["pooled_blind_share_pp"]), flush=True)
        else:
            print("[vote_sweep] %-15s no usable pool -> %s" % (arm, v["verdict"]), flush=True)
    if noop.get("n_compared"):
        print("[vote_sweep] reference no-op: %d/%d TSR, %d/%d FAR, %d/%d target-id, "
              "max |dTSR| %s max |dFAR| %s"
              % (noop["n_tsr_match"], noop["n_compared"], noop["n_far_match"], noop["n_compared"],
                 noop["n_target_id_match"], noop["n_compared"], noop["max_abs_dtsr"],
                 noop["max_abs_dfar"]), flush=True)
    print("[vote_sweep] wall=%.1fs wrote %s and %s"
          % (wall, REPORT_JSON, REPORT_MD), flush=True)
    return 0


def _load_cached(arm: str, seq: str) -> dict[str, Any] | None:
    """One previously finished (arm, sequence) record, or None when it never ran."""
    p = OUT_ROOT / "summaries" / arm / ("%s.json" % seq)
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rec.setdefault("arm", arm)
    rec.setdefault("sequence", seq)
    rec["from_cache"] = True
    return rec


def _tgtrack_path() -> str | None:
    try:
        import tgtrack
        return str(tgtrack.__file__)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
