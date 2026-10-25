import json, numpy as np
from pathlib import Path
# E045 candidate 3: inter-person NEIGHBOURHOOD structure as an identity signal.
# Uses GT boxes only (no detector) -- an upper bound on what boxes-based structure can give.
SEQ = Path("data/raw/_root_dancetrack0007/dancetrack0007"); TARGET = 1; STEP, MAXF, K = 5, 200, 4

def load(seq):
    out = {}
    for ln in (seq/"gt"/"gt.txt").read_text().splitlines():
        f,i,x,y,w,h = ln.split(",")[:6]; f,i = int(float(f)), int(float(i))
        if f > MAXF: continue
        out.setdefault(f,{})[i] = np.array([float(x),float(y),float(x)+float(w),float(y)+float(h)])
    return out

def desc(box, others):
    """Neighbour-set descriptor relative to `box`: k nearest neighbours as (dx,dy,dw,dh)/h."""
    w = box[2]-box[0]; h = box[3]-box[1]; cx = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
    cand = []
    for b in others:
        ncx = (b[0]+b[2])/2; ncy = (b[1]+b[3])/2
        d = np.hypot(ncx-cx, ncy-cy)/max(h,1e-6)
        if 0.05 < d <= 3.0: cand.append((d, (ncx-cx)/h, (ncy-cy)/h, (b[2]-b[0])/h, (b[3]-b[1])/h))
    cand.sort(key=lambda t: t[0])
    v = np.zeros(4*K, dtype=np.float32)
    for j,(_,ax,ay,aw,ah) in enumerate(cand[:K]):
        v[4*j:4*j+4] = (ax,ay,aw,ah)
    return v, len(cand)

def cosd(a,b):
    na,nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na<1e-6 or nb<1e-6 else float(1.0-(a@b)/(na*nb))

gt = load(SEQ); frames = sorted(f for f in gt if f % STEP == 1 and TARGET in gt[f])
rows, click = [], None
for f in frames:
    T = gt[f][TARGET]; others = [b for i,b in gt[f].items() if i != TARGET]
    if len(others) < 2: continue
    if click is None: click, _ = desc(T, others); continue
    # impostor: the person whose box is closest to T (the realistic confusion case)
    imp = min(others, key=lambda b: np.hypot((b[0]+b[2]-T[0]-T[2])/2, (b[1]+b[3]-T[1]-T[3])/2))
    rest_t = [b for b in others if b is not imp]
    rest_i = [b for b in others if b is not imp]
    dA,_ = desc(T, rest_t); dB,_ = desc(imp, rest_i)
    cdA, cdB = cosd(click, dA), cosd(click, dB)
    if cdA is None or cdB is None: continue
    rows.append(dict(f=f, dA=round(cdA,4), dB=round(cdB,4), sep=round(cdB-cdA,4)))
n=len(rows); wins=sum(1 for r in rows if r["dB"]>r["dA"])
print(f"frames={n}  dist(target)={np.mean([r['dA'] for r in rows]):.4f}  dist(impostor)={np.mean([r['dB'] for r in rows]):.4f}  "
      f"wins={wins}/{n}={wins/max(n,1):.3f}  mean_margin={np.mean([r['sep'] for r in rows]):+.4f}")
Path("results/e045_neighbour_probe.json").write_text(json.dumps(dict(n=n,rows=rows),indent=1))
