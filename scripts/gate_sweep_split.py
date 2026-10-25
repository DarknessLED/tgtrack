#!/usr/bin/env python
"""E070b -- honest analysis of the gate sweep: select on one half, report on the other.

WHY THIS EXISTS
---------------
`scripts/gate_sweep.py` sweeps `cost.search_margin` over the 22 held-out DanceTrack
sequences. If we then declare "the best margin on those 22" we have simply moved the
dev set: E068 showed exactly that failure mode (a dev-selected configuration whose
held-out TSR CI did not contain the dev number). So this script refuses to report a
winner from one pool. It:

  1. splits the 22 held-out sequences into EXPLORE / TEST halves with a fixed seed
     (never the frozen dev three: dancetrack0004/0005/0007 are not in this data at all);
  2. ranks the margin points on EXPLORE only (max pooled TSR, tie-break min FAR -- the
     acceptance criterion is a TSR/FAR pair, so both axes are ranked);
  3. reports the chosen point on TEST, which is data the ranking never saw;
  4. reports the *selection penalty*: chosen-on-explore vs its TEST result vs the
     shipped reference (0.20) on the same TEST sequences;
  5. reports paired per-sequence differences against the reference with a cluster
     bootstrap (2000 reps, seed 0) -- paired because every point is run on every
     sequence, so sequence difficulty cancels.

Nothing here touches the GPU: it reads `results/gate_sweep/summaries/<point>/<seq>.json`.

Protocol notes
- pooling is frame-weighted over `n_report_frames` (long sequences are not outvoted);
- only `status == "INCLUDED"` summaries enter a pool, and the exclusion count is printed;
- a half with fewer than 5 included sequences is refused rather than reported;
- output is Markdown on stdout and `results/gate_sweep/SPLIT.md`.

Usage
-----
    python scripts/gate_sweep_split.py                     # primary axis, seed 0
    python scripts/gate_sweep_split.py --seed 7 --reps 5000
    python scripts/gate_sweep_split.py --points margin_0.10,margin_0.20,margin_0.30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP_ROOT = REPO / "results" / "gate_sweep"
SUMMARY_ROOT = SWEEP_ROOT / "summaries"

SHIPPED_REFERENCE = "margin_0.20"
MIN_SEQS_PER_HALF = 5
DEV_SEQUENCES = {"dancetrack0004", "dancetrack0005", "dancetrack0007"}  # must never appear


# --------------------------------------------------------------------------- io
def load_summaries(points: list[str]) -> dict[str, dict[str, dict]]:
    """{point: {sequence: summary}} for every summary that exists on disk."""
    out: dict[str, dict[str, dict]] = {}
    for point in points:
        d = SUMMARY_ROOT / point
        if not d.is_dir():
            continue
        seqs: dict[str, dict] = {}
        for f in sorted(d.glob("*.json")):
            try:
                seqs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[warn] unreadable summary {f}", file=sys.stderr)
        if seqs:
            out[point] = seqs
    return out


def included(summary_by_seq: dict[str, dict]) -> dict[str, dict]:
    return {s: r for s, r in summary_by_seq.items() if r.get("status") == "INCLUDED"}


# ----------------------------------------------------------------------- pooling
def pool(rows: dict[str, dict], field: str) -> float | None:
    """Frame-weighted pooled value of `field` over the given per-sequence rows."""
    num = 0.0
    den = 0.0
    for r in rows.values():
        w = float(r.get("n_report_frames") or 0)
        if w <= 0:
            continue
        num += float(r.get(field) or 0.0) * w
        den += w
    return (num / den) if den > 0 else None


def bootstrap_pooled(
    rows: dict[str, dict], field: str, *, reps: int, seed: int
) -> tuple[float | None, float | None, float | None]:
    """Pooled value plus 95% cluster-bootstrap CI (resampling sequences)."""
    seqs = sorted(rows)
    point = pool(rows, field)
    if point is None or len(seqs) < 2:
        return point, None, None
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(reps):
        sample = [seqs[rng.randrange(len(seqs))] for _ in seqs]
        num = den = 0.0
        for s in sample:
            r = rows[s]
            w = float(r.get("n_report_frames") or 0)
            if w <= 0:
                continue
            num += float(r.get(field) or 0.0) * w
            den += w
        if den > 0:
            draws.append(num / den)
    if not draws:
        return point, None, None
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return point, lo, hi


def paired_difference(
    a: dict[str, dict], b: dict[str, dict], field: str, *, reps: int, seed: int
) -> dict:
    """Frame-weighted pooled (a - b) on the shared sequences, with bootstrap CI.

    Paired by sequence, so differences in sequence difficulty cancel; the sign is
    reported as a-vs-b so positive means `a` is better for TSR and *worse* for FAR.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return {"n": len(shared), "delta": None, "lo": None, "hi": None}
    diffs: dict[str, dict] = {}
    for s in shared:
        wa = float(a[s].get("n_report_frames") or 0)
        wb = float(b[s].get("n_report_frames") or 0)
        w = min(wa, wb) if min(wa, wb) > 0 else max(wa, wb)
        diffs[s] = {
            "n_report_frames": w,
            "d": float(a[s].get(field) or 0.0) - float(b[s].get(field) or 0.0),
        }
    # reuse the pooled machinery by treating the difference as the "value"
    for s, r in diffs.items():
        r["_value"] = r["d"]
    num = sum(r["_value"] * r["n_report_frames"] for r in diffs.values())
    den = sum(r["n_report_frames"] for r in diffs.values())
    point = num / den if den else None
    rng = random.Random(seed)
    draws: list[float] = []
    keys = sorted(diffs)
    for _ in range(reps):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        dn = sum(diffs[s]["_value"] * diffs[s]["n_report_frames"] for s in sample)
        dd = sum(diffs[s]["n_report_frames"] for s in sample)
        if dd:
            draws.append(dn / dd)
    if not draws:
        return {"n": len(shared), "delta": point, "lo": None, "hi": None}
    draws.sort()
    return {
        "n": len(shared),
        "delta": point,
        "lo": draws[int(0.025 * (len(draws) - 1))],
        "hi": draws[int(0.975 * (len(draws) - 1))],
    }


# ------------------------------------------------------------------------- split
def split_sequences(seqs: list[str], *, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic, seed-controlled halving of the sequence list."""
    pool = sorted(seqs)
    rng = random.Random(seed)
    rng.shuffle(pool)
    half = len(pool) // 2
    return sorted(pool[:half]), sorted(pool[half:])


def fmt(v: float | None, *, pct: bool = False, nd: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.{nd - 1}f}%" if pct else f"{v:.{nd}f}"


def rank_key(points: dict[str, dict], explore: list[str]) -> list[tuple[str, float, float]]:
    """Rank points on the explore half: max pooled TSR, then min pooled FAR."""
    table: list[tuple[str, float, float]] = []
    for point, by_seq in points.items():
        rows = {s: by_seq[s] for s in explore if s in by_seq}
        tsr = pool(rows, "tsr")
        far = pool(rows, "far")
        if tsr is None or far is None:
            continue
        table.append((point, tsr, far))
    # TSR descending, FAR ascending, then name for determinism
    table.sort(key=lambda t: (-t[1], t[2], t[0]))
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="margin_0.00,margin_0.10,margin_0.15,margin_0.20,margin_0.30",
                    help="comma-separated sweep point ids (default: the E070 primary axis)")
    ap.add_argument("--seed", type=int, default=0, help="split seed (default 0)")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap repetitions (default 2000)")
    ap.add_argument("--out", default="results/gate_sweep/SPLIT.md")
    args = ap.parse_args(argv)

    points = [p.strip() for p in args.points.split(",") if p.strip()]
    data = load_summaries(points)
    if not data:
        raise SystemExit(f"[split] no summaries under {SUMMARY_ROOT} for points {points}")

    # a sequence participates only if it is included at EVERY point, so the
    # points are compared on the same sequences (no missing-data artefacts)
    per_point_incl = {p: set(included(v)) for p, v in data.items()}
    complete = set.intersection(*per_point_incl.values()) if per_point_incl else set()
    leaked = complete & DEV_SEQUENCES
    if leaked:
        raise SystemExit(f"[split] REFUSING: dev sequences present in sweep data: {sorted(leaked)}")
    if len(complete) < 2 * MIN_SEQS_PER_HALF:
        raise SystemExit(
            f"[split] only {len(complete)} complete sequences across {len(data)} points; "
            f"need {2 * MIN_SEQS_PER_HALF} to split. Run the sweep first."
        )

    # A selection needs at least two candidates. With one point on disk the only
    # honest output is "the sweep has not been run", never a ranking of one arm.
    if len(data) < 2:
        only = next(iter(data))
        rows = {s: data[only][s] for s in complete}
        tsr, lo, hi = bootstrap_pooled(rows, "tsr", reps=args.reps, seed=args.seed)
        far, flo, fhi = bootstrap_pooled(rows, "far", reps=args.reps, seed=args.seed)
        msg = (
            "# E070b -- no selection possible\n\n"
            f"Only one sweep point is on disk (`{only}`, {len(complete)} complete sequences).\n"
            "Ranking one arm is vacuous, so nothing is selected or claimed here.\n\n"
            f"- reference pool: TSR {fmt(tsr, pct=True, nd=4)} "
            f"[{fmt(lo, pct=True, nd=4)}, {fmt(hi, pct=True, nd=4)}], "
            f"FAR {fmt(far, pct=True, nd=4)} [{fmt(flo, pct=True, nd=4)}, "
            f"{fmt(fhi, pct=True, nd=4)}], n={len(rows)}\n\n"
            "Run the remaining points first:\n\n"
            "```powershell\npython scripts/gate_sweep.py --points primary\n```\n"
        )
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(msg, encoding="utf-8")
        print(msg)
        print(f"[split] wrote {out}")
        return 0

    points_data = {p: {s: v[s] for s in complete} for p, v in data.items()}
    explore, test = split_sequences(sorted(complete), seed=args.seed)

    ranked = rank_key(points_data, explore)
    if not ranked:
        raise SystemExit("[split] no point produced a poolable explore result")
    chosen, chosen_tsr_explore, chosen_far_explore = ranked[0]

    rows_test = points_data.get(chosen, {})
    rows_test = {s: rows_test[s] for s in test if s in rows_test}
    if len(rows_test) < MIN_SEQS_PER_HALF:
        raise SystemExit(
            f"[split] chosen point {chosen} has only {len(rows_test)} TEST sequences; refusing"
        )

    ref_rows = {s: points_data[SHIPPED_REFERENCE][s] for s in test
                if SHIPPED_REFERENCE in points_data and s in points_data[SHIPPED_REFERENCE]}

    tsr_test, tsr_lo, tsr_hi = bootstrap_pooled(rows_test, "tsr", reps=args.reps, seed=args.seed)
    far_test, far_lo, far_hi = bootstrap_pooled(rows_test, "far", reps=args.reps, seed=args.seed)
    d_tsr = paired_difference(rows_test, ref_rows, "tsr", reps=args.reps, seed=args.seed) if ref_rows else {}
    d_far = paired_difference(rows_test, ref_rows, "far", reps=args.reps, seed=args.seed) if ref_rows else {}
    ref_tsr = pool(ref_rows, "tsr") if ref_rows else None
    ref_far = pool(ref_rows, "far") if ref_rows else None

    # ---------------------------------------------------------------- report
    L: list[str] = []
    a = L.append
    a("# E070b -- select on one half, report on the other")
    a("")
    a(f"Generated by `scripts/gate_sweep_split.py --seed {args.seed} --reps {args.reps}`; "
      "every number is read from `results/gate_sweep/summaries/`.")
    a("")
    a("## Split (fixed seed, frozen dev excluded)")
    a("")
    a(f"- frozen dev (never swept, never reported): {', '.join(sorted(DEV_SEQUENCES))}")
    a(f"- complete held-out sequences: **{len(complete)}**")
    a(f"- EXPLORE ({len(explore)}): {', '.join(explore)}")
    a(f"- TEST    ({len(test)}): {', '.join(test)}")
    a("")
    a("## Ranking on EXPLORE only (max pooled TSR, tie-break min FAR)")
    a("")
    a("| rank | point | TSR | FAR |")
    a("|---|---|---|---|")
    for i, (p, tsr, far) in enumerate(ranked, 1):
        mark = " **<- selected**" if p == chosen else ""
        a(f"| {i} | `{p}`{mark} | {fmt(tsr, pct=True, nd=4)} | {fmt(far, pct=True, nd=4)} |")
    a("")
    a(f"## Chosen point on TEST (unseen by the ranking): `{chosen}`")
    a("")
    if ref_tsr is not None:
        a("| split | TSR | FAR | n seqs |")
        a("|---|---|---|---|")
        a(f"| EXPLORE (selection) | {fmt(chosen_tsr_explore, pct=True, nd=4)} | "
          f"{fmt(chosen_far_explore, pct=True, nd=4)} | {len(explore)} |")
        a(f"| **TEST (reported)** | **{fmt(tsr_test, pct=True, nd=4)}** "
          f"[{fmt(tsr_lo, pct=True, nd=4)}, {fmt(tsr_hi, pct=True, nd=4)}] | "
          f"**{fmt(far_test, pct=True, nd=4)}** [{fmt(far_lo, pct=True, nd=4)}, "
          f"{fmt(far_hi, pct=True, nd=4)}] | {len(rows_test)} |")
        a(f"| TEST, shipped reference `{SHIPPED_REFERENCE}` | {fmt(ref_tsr, pct=True, nd=4)} | "
          f"{fmt(ref_far, pct=True, nd=4)} | {len(ref_rows)} |")
        a("")
        a("### Paired per-sequence difference vs the shipped reference (TEST half only)")
        a("")
        a("| metric | delta (chosen - reference) | 95% CI | n | reads as |")
        a("|---|---|---|---|---|")
        if d_tsr.get("delta") is not None:
            verdict_tsr = ("candidate better" if d_tsr["lo"] is not None and d_tsr["lo"] > 0
                           else "reference better" if d_tsr["hi"] is not None and d_tsr["hi"] < 0
                           else "**indistinguishable**")
            a(f"| TSR | {fmt(d_tsr['delta'], pct=True, nd=4)} | "
              f"[{fmt(d_tsr['lo'], pct=True, nd=4)}, {fmt(d_tsr['hi'], pct=True, nd=4)}] | "
              f"{d_tsr['n']} | {verdict_tsr} |")
        if d_far.get("delta") is not None:
            verdict_far = ("candidate WORSE" if d_far["lo"] is not None and d_far["lo"] > 0
                           else "candidate better" if d_far["hi"] is not None and d_far["hi"] < 0
                           else "**indistinguishable**")
            a(f"| FAR | {fmt(d_far['delta'], pct=True, nd=4)} | "
              f"[{fmt(d_far['lo'], pct=True, nd=4)}, {fmt(d_far['hi'], pct=True, nd=4)}] | "
              f"{d_far['n']} | {verdict_far} |")
        a("")
        a("> Positive TSR delta = candidate better. Positive FAR delta = candidate worse.")
    else:
        a(f"Reference point `{SHIPPED_REFERENCE}` has no summaries; only the TEST result is shown: "
          f"TSR {fmt(tsr_test, pct=True, nd=4)} [{fmt(tsr_lo, pct=True, nd=4)}, "
          f"{fmt(tsr_hi, pct=True, nd=4)}], FAR {fmt(far_test, pct=True, nd=4)} "
          f"[{fmt(far_lo, pct=True, nd=4)}, {fmt(far_hi, pct=True, nd=4)}], n={len(rows_test)}.")
    a("")
    a("## Exclusions")
    a("")
    for p in sorted(data):
        inc = len(per_point_incl[p])
        dropped = sorted(set(data[p]) - per_point_incl[p])
        a(f"- `{p}`: {inc} included" + (f", dropped: {', '.join(dropped)}" if dropped else ""))
    a("")
    a("## Reading rules (the point of this document)")
    a("")
    a("1. **A selection number is not a result.** The EXPLORE row is where the winner was")
    a("   chosen; only the TEST row (and its CI) may be quoted as performance.")
    a("2. **Paired CIs, not point estimates.** If the TSR paired CI contains 0, the sweep has")
    a("   not earned the right to change the default, whatever the EXPLORE ranking says.")
    a("3. **Watch the FAR column.** It is the axis this knob moves in the opposite direction;")
    a("   a TSR gain bought with a FAR regression is a trade, not an improvement, and the")
    a("   objective requires FAR < 0.20.")
    a("4. **Selection on 11 sequences is noisy.** A reshuffle of `--seed` changes the halves;")
    a("   run at least two seeds before believing a ranking. If the winner flips between seeds,")
    a("   the sweep is measuring noise.")
    a("")
    a("## Reproduce")
    a("")
    a("```powershell")
    a("python scripts/gate_sweep.py --points primary        # 5 points x 22 sequences")
    a(f"python scripts/gate_sweep_split.py --seed {args.seed}")
    a("python scripts/gate_sweep_split.py --seed 7          # stability check")
    a("```")
    text = "\n".join(L) + "\n"

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[split] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
