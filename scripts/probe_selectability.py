"""E074 prep: does target SELECTABILITY at the click predict TSR? Pure GT, no GPU.

For each of E070's 22 held-out (sequence, target_id) rows, compute GT-only statistics over the
same evaluated window (frames 1..300 step 2) and correlate against the measured shipped TSR.
"""
import json, pathlib, statistics as st
import numpy as np

ROOT = pathlib.Path("data/raw/dancetrack/val")
rep = json.loads(pathlib.Path("results/gate_sweep/report.json").read_text(encoding="utf-8"))
rows = rep["points"]["margin_0.20"]["rows"]

def load_gt(seq):
    out = {}
    f = ROOT / seq / "gt" / "gt.txt"
    for line in f.read_text(encoding="utf-8").splitlines():
        p = line.strip().split(",")
        if len(p) < 6: continue
        fr, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        if w <= 0 or h <= 0: continue
        out.setdefault(fr, []).append((tid, x, y, w, h))
    return out

def iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax0 + aw, bx0 + bw); y1 = min(ay0 + ah, by0 + bh)
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0

def stats_for(seq, tid, frames=range(1, 301, 2)):
    gt = load_gt(seq)
    present, areas, heights, nb2, maxiou, crowd = [], [], [], [], [], []
    first_ident = None
    for fr in frames:
        boxes = gt.get(fr, [])
        me = [b for b in boxes if b[0] == tid]
        if not me:
            present.append(0); continue
        present.append(1)
        _, x, y, w, h = me[0]
        areas.append(w * h); heights.append(h)
        crowd.append(len(boxes))
        near = 0; mx = 0.0
        for b in boxes:
            if b[0] == tid: continue
            cx, cy = x + w / 2, y + h / 2
            bx, by, bw, bh = b[1], b[2], b[3], b[4]
            d = ((cx - (bx + bw / 2)) ** 2 + (cy - (by + bh / 2)) ** 2) ** 0.5
            if d <= 2.0 * max(h, bh): near += 1
            mx = max(mx, iou((x, y, w, h), (bx, by, bw, bh)))
        nb2.append(near); maxiou.append(mx)
        if first_ident is None:
            first_ident = dict(near=near, maxiou=mx, n_box=len(boxes), h=h, area=w * h)
    return dict(
        presence=float(np.mean(present)) if present else 0.0,
        n_eval=len(present),
        h_med=st.median(heights) if heights else 0.0,
        area_med=st.median(areas) if areas else 0.0,
        crowd_med=st.median(crowd) if crowd else 0.0,
        nb2_med=st.median(nb2) if nb2 else 0.0,
        maxiou_med=st.median(maxiou) if maxiou else 0.0,
        first=first_ident or {},
    )

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = (ra ** 2).sum() ** 0.5 * (rb ** 2).sum() ** 0.5
    return float((ra * rb).sum() / d) if d else 0.0

out = []
for r in rows:
    seq, tid = r["sequence"], int(r["target_id"])
    s = stats_for(seq, tid)
    s.update(seq=seq, tid=tid, tsr=r["tsr"], far=r["far"], idc=r["id_consistency"])
    out.append(s)

out.sort(key=lambda x: x["tsr"])
print(f"{'sequence':18s} {'tid':>3s} {'TSR':>6s} {'FAR':>6s} {'idc':>6s} {'pres':>5s} {'h_med':>6s} "
      f"{'area':>7s} {'crowd':>5s} {'nb2':>4s} {'maxIoU':>6s} {'f_near':>6s} {'f_ioU':>6s}")
for s in out:
    fi = s["first"]
    print(f"{s['seq']:18s} {s['tid']:3d} {s['tsr']:6.3f} {s['far']:6.3f} {s['idc']:6.3f} "
          f"{s['presence']:5.2f} {s['h_med']:6.1f} {s['area_med']:7.0f} {s['crowd_med']:5.1f} "
          f"{s['nb2_med']:4.1f} {s['maxiou_med']:6.3f} {fi.get('near',-1):6d} {fi.get('maxiou',-1):6.3f}")

print("\nSpearman vs TSR:")
for k in ["presence", "h_med", "area_med", "crowd_med", "nb2_med", "maxiou_med"]:
    print(f"  {k:12s} {spearman([s['tsr'] for s in out], [s[k] for s in out]):+.3f}")
print("  first.near  ", f"{spearman([s['tsr'] for s in out], [s['first'].get('near',0) for s in out]):+.3f}")
print("  first.maxiou", f"{spearman([s['tsr'] for s in out], [s['first'].get('maxiou',0) for s in out]):+.3f}")
pathlib.Path("results/selectability.json").parent.mkdir(exist_ok=True)
pathlib.Path("results/selectability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
