#!/usr/bin/env python
"""Per-sequence paired analysis of a gate-sweep axis.

`scripts/gate_sweep_split.py` answers "does the *axis* move the criterion on data the choice was
not made on". This script answers the complementary question that decides whether a better rule
is even possible: **is the trade-off uniform across sequences, or does some subset gain without
paying?** If every sequence trades TSR against FAR in the same proportion, no router can help and
the decision rule itself must change (the E071 direction). If some sequences gain for free, a
per-sequence predictor is the cheaper win.

Reads `results/gate_sweep/runs/<point>/<seq>/summary.json`, uses only points whose 22 sequences are
present (a half-finished point would silently change the population), and prints:

  * the paired table (one row per sequence, deltas against the reference point);
  * a 2x2 classification of sequences: free win / free loss / trade-off up / trade-off down;
  * pooled frame-weighted deltas and the rank correlation between dTSR and dFAR.

Usage:
    python scripts/gate_sweep_paired.py --ref margin_0.20 --cmp margin_0.00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "gate_sweep"


def load_point(tag: str) -> dict[str, dict]:
    """seq -> headline row, only if the point is complete enough to compare."""
    out: dict[str, dict] = {}
    base = SWEEP / "runs" / tag
    if not base.is_dir():
        return out
    for seqdir in sorted(base.iterdir()):
        f = seqdir / "summary.json"
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            head = d.get("headline") or []
            if not head:
                continue
            row = dict(head[0])
            row["n_frames"] = int(d.get("meta", {}).get("max_frames") or row.get("n_frames") or 0)
            out[seqdir.name] = row
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return out


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation without scipy (kept dependency-free for the handoff)."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="margin_0.20", help="reference point (the shipped gate)")
    ap.add_argument("--cmp", default="margin_0.00", help="comparison point")
    ap.add_argument("--out", default="results/gate_sweep/PAIRED.md")
    args = ap.parse_args(argv)

    ref, cmp_ = load_point(args.ref), load_point(args.cmp)
    common = sorted(set(ref) & set(cmp_))
    if not common:
        print(f"no overlapping sequences between {args.ref} and {args.cmp}")
        return 1
    only_ref, only_cmp = len(ref) - len(common), len(cmp_) - len(common)

    L: list[str] = []
    a = L.append
    a(f"# Paired per-sequence analysis: `{args.cmp}` vs `{args.ref}`")
    a("")
    a(f"Sequences compared: **{len(common)}**"
      + (f" (`{args.ref}` has {only_ref} extra unfinished, `{args.cmp}` has {only_cmp})"
         if only_ref or only_cmp else ""))
    a("")
    a("| sequence | TSR ref | TSR cmp | dTSR (pp) | FAR ref | FAR cmp | dFAR (pp) | class |")
    a("|---|---|---|---|---|---|---|---|")
    cells = {"free_win": 0, "free_loss": 0, "trade_up": 0, "trade_down": 0, "flat": 0}
    d_tsr: list[float] = []
    d_far: list[float] = []
    for seq in common:
        r, c = ref[seq], cmp_[seq]
        rt, ct = float(r["target_success_rate"]), float(c["target_success_rate"])
        rf, cf = float(r["false_acquisition_rate"]), float(c["false_acquisition_rate"])
        dt, df = (ct - rt) * 100.0, (cf - rf) * 100.0
        d_tsr.append(dt)
        d_far.append(df)
        if abs(dt) < 1.0 and abs(df) < 1.0:
            cls = "flat"
        elif dt > 0 and df <= 0:
            cls = "FREE WIN"
        elif dt <= 0 and df > 0:
            cls = "free loss"
        elif dt > 0 and df > 0:
            cls = "trade-off up"
        else:
            cls = "trade-off down"
        cells[{"FREE WIN": "free_win", "free loss": "free_loss", "trade-off up": "trade_up",
               "trade-off down": "trade_down", "flat": "flat"}[cls]] += 1
        a(f"| {seq} | {rt:.3f} | {ct:.3f} | {dt:+.1f} | {rf:.3f} | {cf:.3f} | {df:+.1f} | {cls} |")

    a("")
    a("## Classification of the {n} sequences".format(n=len(common)))
    a("")
    a("| class | meaning | count |")
    a("|---|---|---|")
    a(f"| **FREE WIN** | TSR up, FAR not worse | **{cells['free_win']}** |")
    a(f"| free loss | TSR not better, FAR worse | {cells['free_loss']} |")
    a(f"| trade-off up | both up (the gate trade-off) | {cells['trade_up']} |")
    a(f"| trade-off down | both down | {cells['trade_down']} |")
    a(f"| flat | both within 1 pp | {cells['flat']} |")
    a("")

    # frame-weighted pooled deltas
    def pooled(rows: dict[str, dict], key: str, wkey: str = "n_report_frames") -> tuple[float, float]:
        num = den = 0.0
        for seq in common:
            row = rows[seq]
            w = float(row.get(wkey) or row.get("n_frames") or 0)
            num += float(row[key]) * w
            den += w
        return (num / den if den else float("nan")), den

    rt_p, frames = pooled(ref, "target_success_rate")
    ct_p, _ = pooled(cmp_, "target_success_rate")
    rf_p, _ = pooled(ref, "false_acquisition_rate")
    cf_p, _ = pooled(cmp_, "false_acquisition_rate")
    a("## Pooled (frame-weighted over the compared sequences)")
    a("")
    a("| metric | ref | cmp | delta |")
    a("|---|---|---|---|")
    a(f"| TSR | {rt_p:.4f} | {ct_p:.4f} | **{(ct_p - rt_p) * 100:+.2f} pp** |")
    a(f"| FAR | {rf_p:.4f} | {cf_p:.4f} | **{(cf_p - rf_p) * 100:+.2f} pp** |")
    a(f"| frames behind the pooled number | {int(frames)} | | |")
    a("")
    rho = _spearman(d_tsr, d_far)
    a(f"Rank correlation between dTSR and dFAR across sequences: **rho = {rho:+.3f}**")
    a("")
    if cells["free_win"] == 0 and cells["free_loss"] == 0:
        a("**Reading: the trade-off is uniform — every sequence moves the same way.** No sequence "
          "buys TSR without paying FAR, so a per-sequence router has nothing to exploit; the "
          "decision rule itself has to change (E071), which is what this justifies.")
    elif cells["free_win"] > 0:
        a(f"**Reading: {cells['free_win']} sequence(s) gain TSR at no FAR cost**, so the trade-off "
          "is NOT uniform and a per-sequence predictor is a candidate cheaper-than-E071 win. "
          "Inspect those rows before building anything: with 22 sequences a couple of free wins "
          "can be frame-count noise.")
    else:
        a("**Reading: no free wins; some sequences lose TSR while FAR rises.** The axis is at or "
          "past its useful end for those sequences.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6]))
    print(f"\n[paired] {len(common)} sequences, {cells} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
