"""E076: state-based decomposition of the held-out single-target loss.

Answers one question: of the missing target-success, how much is BLINDNESS (the tracker is not
reported/locked) versus WRONG IDENTITY (locked, but on somebody else)?
"""
import csv, glob, json, pathlib, collections
import numpy as np

BLIND = {"occluded_soft", "lost_short", "lost_long", "out_of_frame"}
TRUTHY = {"1", "True", "true"}

def analyse(path):
    recs = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    if not recs:
        return None
    n = len(recs)
    states = collections.Counter(r.get("state", "") for r in recs)
    rep = [r for r in recs if r.get("reported") in TRUTHY]
    trk = [r for r in recs if r.get("state") == "tracking"]
    blind = sum(v for k, v in states.items() if k in BLIND)
    def tsr(rs):
        return (sum(1 for r in rs if r.get("correct") in TRUTHY) / len(rs)) if rs else float("nan")
    iou = np.array([float(r.get("iou_target") or 0.0) for r in recs])
    return dict(
        seq=pathlib.Path(path).stem.split("__")[0], n=n,
        tsr=tsr(rep), n_report=len(rep),
        blind_frac=blind / n, track_frac=len(trk) / n,
        tsr_tracking=tsr(trk),
        tsr_loose=float((iou >= 0.1).mean()),
        states={k: round(v / n, 3) for k, v in states.most_common()},
    )

files = sorted(glob.glob("results/e076_states/runs/*.csv"))
rows = [r for r in (analyse(f) for f in files) if r]
if not rows:
    raise SystemExit("no CSV yet")
rows.sort(key=lambda r: r["tsr"])
print(f"{'sequence':18s} {'TSR':>6s} {'loose':>6s} {'blind':>6s} {'track':>6s} {'TSR|trk':>7s}  states")
for r in rows:
    print(f"{r['seq']:18s} {r['tsr']:6.3f} {r['tsr_loose']:6.3f} {r['blind_frac']:6.3f} "
          f"{r['track_frac']:6.3f} {r['tsr_tracking']:7.3f}  "
          + " ".join(f"{k}:{v:.2f}" for k, v in list(r["states"].items())[:4]))

tot = sum(r["n"] for r in rows)
pooled = sum(r["tsr"] * r["n"] for r in rows) / tot
blind_pool = sum(r["blind_frac"] * r["n"] for r in rows) / tot
trk_pool = sum(r["track_frac"] * r["n"] for r in rows) / tot
tsr_trk_pool = sum(r["tsr_tracking"] * r["track_frac"] * r["n"] for r in rows if r["tsr_tracking"] == r["tsr_tracking"]) / max(1e-9, sum(r["track_frac"] * r["n"] for r in rows))
print(f"\nPOOLED over {len(rows)} seqs / {tot} frames: TSR {pooled:.4f}")
print(f"  blind share {blind_pool:.3f}; tracking share {trk_pool:.3f}; TSR while tracking {tsr_trk_pool:.4f}")
print(f"  loss {1 - pooled:.4f} =  blindness {(1 - 0.0) * blind_pool:.4f} + "
      f"wrong-while-tracking {(1 - tsr_trk_pool) * trk_pool:.4f} + other "
      f"{max(0.0, 1 - pooled - blind_pool - (1 - tsr_trk_pool) * trk_pool):.4f}")
pathlib.Path("results/e076_states/states.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
