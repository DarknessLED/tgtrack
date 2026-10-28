"""E077-prep: how much of the target-success metric is a BOX-CONVENTION tax rather than identity?"""
import csv, glob, ast, numpy as np

def box(s):
    try:
        v = ast.literal_eval(s)
    except Exception:
        return None
    return [float(x) for x in v[:4]] if isinstance(v, (list, tuple)) and len(v) >= 4 else None

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0); y0 = max(ay0, by0); x1 = min(ax1, bx1); y1 = min(ay1, by1)
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0

def scale_box(b, sw, sh):
    cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2
    w = (b[2] - b[0]) * sw; h = (b[3] - b[1]) * sh
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

files = sorted(glob.glob("results/e076_states/runs/*.csv"))
print(f"analysing {len(files)} sequences")
grid = [(1.00, 1.00), (1.05, 1.00), (1.10, 1.00), (1.15, 1.00), (1.20, 1.00), (1.10, 1.05)]
rows = []
for f in files:
    recs = list(csv.DictReader(open(f, encoding="utf-8", newline="")))
    preds, gts, ok = [], [], []
    for r in recs:
        p, g = box(r["pred_xyxy"]), box(r["gt_xyxy"])
        if p and g:
            preds.append(p); gts.append(g); ok.append(1)
        else:
            ok.append(0)
    rows.append((f.split("__")[0].split("/")[-1], preds, gts))
print(f"{'scale w/h':12s} {'TSR@0.5':>8s} {'TSR@0.4':>8s} {'TSR@0.3':>8s}")
for sw, sh in grid:
    a = {0.5: [], 0.4: [], 0.3: []}
    for name, preds, gts in rows:
        if not preds: continue
        for p, g in zip(preds, gts):
            ps = scale_box(p, sw, sh)
            i = iou(ps, g)
            for t in a: a[t].append(i >= t)
    print(f"{sw:.2f}/{sh:.2f}    " + " ".join(f"{np.mean(a[t]):8.3f}" for t in (0.5, 0.4, 0.3)))
print("\n(report-frame weighted by availability, IoU against the target's own GT box)")
