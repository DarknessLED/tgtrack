"""E075: is the low TSR box-quality or identity? Use the iou_target distribution."""
import csv, glob, numpy as np, pathlib
f = "results/e075_dt0097/dancetrack0097__baseline__none.csv"
recs = [r for r in csv.DictReader(open(f, encoding="utf-8", newline="")) if r.get("reported") in ("1","True","true")]
iou = np.array([float(r["iou_target"] or 0) for r in recs])
tgt = {r.get("target_person_id") for r in recs}
best = [r.get("best_gt_id") for r in recs]
print("rows", len(recs), "| target_person_id values:", tgt)
for t in (0.1, 0.2, 0.3, 0.4, 0.5):
    print(f"  TSR@{t}: {float((iou >= t).mean()):.3f}")
print("iou_target quantiles:", {q: round(float(np.quantile(iou, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
import collections
print("best_gt_id counts:", collections.Counter(best).most_common(5))
print("frac where best_gt_id == target:", float(np.mean([b in tgt for b in best])))
# state mix on the failures
st = collections.Counter(r["state"] for r, i in zip(recs, iou) if i < 0.5)
print("states on iou<0.5:", st.most_common())
fsu = np.array([float(r["frames_since_update"] or 0) for r, i in zip(recs, iou) if i < 0.5])
print("frames_since_update on failures: median", float(np.median(fsu)), "max", float(fsu.max()), "frac==0", float((fsu == 0).mean()))
