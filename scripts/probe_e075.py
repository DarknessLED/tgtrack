"""E075: is the wrong-hold sequence invariant to the whole knob grid? (pure CSV analysis)"""
import csv, glob, pathlib, collections
rows = []
for f in sorted(glob.glob("results/e075_dt0097/runs/**/*.csv", recursive=True) + glob.glob("results/e075_dt0097/*.csv")):
    p = pathlib.Path(f)
    name = p.stem.split("__")[1] if "__" in p.stem else p.stem
    recs = list(csv.DictReader(open(f, encoding="utf-8", newline="")))
    if not recs: continue
    rep = [r for r in recs if r.get("reported") in ("1", "True", "true")]
    if not rep: continue
    n = len(rep)
    tsr = sum(1 for r in rep if r.get("correct") in ("1", "True", "true")) / n
    fa = sum(1 for r in rep if r.get("false_acquisition") in ("1", "True", "true")) / n
    followed = collections.Counter(r.get("followed_id") for r in rep)
    dom, domn = followed.most_common(1)[0]
    truth = collections.Counter(r.get("target_person_id") for r in rep)
    wrong_best = collections.Counter(r.get("best_gt_id") for r in rep
                                     if r.get("correct") in ("0", "False", "false"))
    rows.append(dict(variant=name, n=n, tsr=tsr, far=fa, dom_id=dom, dom_share=domn / n,
                     target_id=truth.most_common(1)[0][0], n_distinct_followed=len(followed),
                     top_wrong=wrong_best.most_common(2)))
rows.sort(key=lambda r: r["tsr"])
print(f"{'variant':26s} {'n':>3s} {'TSR':>6s} {'FAR':>6s} {'dom_followed':>12s} {'share':>5s} {'#ids':>4s} whose")
for r in rows:
    print(f"{r['variant']:26s} {r['n']:3d} {r['tsr']:6.3f} {r['far']:6.3f} {str(r['dom_id']):>12s} "
          f"{r['dom_share']:5.2f} {r['n_distinct_followed']:4d} gt_target={r['target_id']} "
          f"top_wrong={r['top_wrong']}")
print(f"\n{len(rows)} variants; TSR range {rows[0]['tsr']:.3f}..{rows[-1]['tsr']:.3f}")
