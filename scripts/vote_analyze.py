"""Paired analysis of the vote-sweep arms (E079 / E081) -- read-only.

Reads the per-pair records written by `scripts/vote_sweep.py` and prints:
pooled TSR / FAR / id_consistency per arm, paired per-sequence deltas against the
reference arm with a cluster bootstrap over sequences, and the same deltas split
into the EXPLORE half (odd positions) and the TEST half (even positions) so a knob
can be chosen on one half and confirmed on the other.

    python scripts/vote_analyze.py
    python scripts/vote_analyze.py --arms vote_off,vote_majority,vote_adaptive_0.20
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/vote_sweep/runs")
REFERENCE = "vote_off"
ARMS = [
    "vote_off",
    "vote_majority",
    "vote_adaptive_0.20",
    "vote_adaptive_0.35",
    "vote_adaptive_0.50",
    "vote_adaptive_0.65",
    "vote_adaptive_1.00",
]
METRICS = ("target_success_rate", "false_acquisition_rate", "id_consistency")
REPS = 2000
SEED = 0


def load(arm: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    d = ROOT / arm
    if not d.is_dir():
        return out
    for seq_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        f = seq_dir / "summary.json"
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        hl = (data.get("headline") or [None])[0]
        if not hl:
            continue
        out[seq_dir.name] = {
            "n": int(hl.get("n_report_frames") or 0),
            **{m: float(hl[m]) for m in METRICS if hl.get(m) is not None},
        }
    return out


def pooled(rows: dict[str, dict]) -> dict[str, float]:
    num = {m: 0.0 for m in METRICS}
    den = 0.0
    for r in rows.values():
        n = r["n"]
        den += n
        for m in METRICS:
            num[m] += r.get(m, 0.0) * n
    return {m: (num[m] / den if den else float("nan")) for m in METRICS}


def paired(ref: dict[str, dict], arm: dict[str, dict]) -> dict[str, float]:
    """Mean of per-sequence deltas, bootstrapped over sequences."""
    seqs = sorted(set(ref) & set(arm))
    out: dict[str, float] = {"n_paired": len(seqs)}
    if len(seqs) < 3:
        return out
    deltas = {m: np.array([arm[s][m] - ref[s][m] for s in seqs]) for m in METRICS}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(seqs), size=(REPS, len(seqs)))
    for m, d in deltas.items():
        boots = d[idx].mean(axis=1)
        out[m] = float(d.mean())
        out[m + "_lo"] = float(np.percentile(boots, 2.5))
        out[m + "_hi"] = float(np.percentile(boots, 97.5))
        out[m + "_up"] = int(np.count_nonzero(d > 0))
        out[m + "_down"] = int(np.count_nonzero(d < 0))
    return out


def fmt(v: float, scale: float = 100.0) -> str:
    return f"{v * scale:+.2f}"


def main() -> int:
    arms = {}
    for a in ARMS:
        rows = load(a)
        if rows:
            arms[a] = rows
    seqs = sorted(arms.get(REFERENCE, {}))
    if not seqs:
        print("no reference pairs yet")
        return 1
    explore = [s for i, s in enumerate(seqs) if i % 2 == 1]
    test = [s for i, s in enumerate(seqs) if i % 2 == 0]

    print(f"sequences with a reference pair: {len(seqs)}")
    print()
    print("| arm | pairs | TSR | FAR | id_consistency |")
    print("|---|---|---|---|---|")
    for a, rows in arms.items():
        p = pooled(rows)
        print(f"| `{a}` | {len(rows)} | {p['target_success_rate']:.4f} | "
              f"{p['false_acquisition_rate']:.4f} | {p['id_consistency']:.4f} |")
    print()
    ref = arms[REFERENCE]
    for a, rows in arms.items():
        if a == REFERENCE:
            continue
        for label, subset in (("ALL", None), ("EXPLORE", explore), ("TEST", test)):
            r = {s: ref[s] for s in (subset or seqs) if s in ref}
            o = {s: rows[s] for s in (subset or seqs) if s in rows}
            d = paired(r, o)
            if d.get("n_paired", 0) < 3:
                print(f"| `{a}` {label} | paired {d.get('n_paired', 0)} | - | - | - |")
                continue
            print(
                f"| `{a}` {label} | {d['n_paired']} | "
                f"{fmt(d['target_success_rate'])} pp "
                f"[{fmt(d['target_success_rate_lo'])}, {fmt(d['target_success_rate_hi'])}] "
                f"({d['target_success_rate_up']}up/{d['target_success_rate_down']}dn) | "
                f"{fmt(d['false_acquisition_rate'])} pp "
                f"[{fmt(d['false_acquisition_rate_lo'])}, {fmt(d['false_acquisition_rate_hi'])}] | "
                f"{fmt(d['id_consistency'])} pp |"
            )
    print()
    print(f"explore half: {len(explore)} seqs | test half: {len(test)} seqs")
    print("explore:", ",".join(s.replace("dancetrack", "") for s in explore))
    print("test   :", ",".join(s.replace("dancetrack", "") for s in test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
