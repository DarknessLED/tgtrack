import json, numpy as np
from pathlib import Path
from ultralytics import YOLO

# ---- E045 candidate 1 feasibility probe: is POSE STRUCTURE an independent identity signal?
# DanceTrack forces heavy overlap; GT gives us the target's box and a distractor's box per frame.
SEQ = Path("data/raw/_root_dancetrack0007/dancetrack0007")
TARGET = 1
STEP, MAXF = 5, 200

def load_gt(seq, tid):
    out = {}
    for ln in (seq/"gt"/"gt.txt").read_text().splitlines():
        f, i, x, y, w, h = ln.split(",")[:6]
        f, i = int(float(f)), int(float(i))
        if f > MAXF: continue
        out.setdefault(f, {})[i] = np.array([float(x),float(y),float(x)+float(w),float(y)+float(h)])
    return {f:v for f,v in out.items() if tid in v}

def iou(a, b):
    ix = max(0.0, min(a[2],b[2])-max(a[0],b[0])); iy = max(0.0, min(a[3],b[3])-max(a[1],b[1]))
    inter = ix*iy; ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def kp_desc(kps, box, conf=0.30):
    """Structural descriptor: box-normalised keypoints (aspect-free x rescaled by height)."""
    w = box[2]-box[0]; h = box[3]-box[1]
    v = np.zeros(34, dtype=np.float32)
    for j,(x,y,c) in enumerate(kps):
        if c >= conf:
            v[2*j] = (x - box[0])/max(h,1e-6); v[2*j+1] = (y - box[1])/max(h,1e-6)
    return v

def cosd(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6: return None
    return float(1.0 - (a@b)/(na*nb))

gt = load_gt(SEQ, TARGET)
frames = sorted(f for f in gt if f % STEP == 1)
model = YOLO(str(Path("data/weights/yolo26n-pose.pt")))
rows, click = [], None
for f in frames:
    img = str(SEQ/"img1"/f"{f:08d}.jpg")
    r = model.predict(img, imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None: continue
    boxes = r.boxes.xyxy.cpu().numpy(); kps = r.keypoints.data.cpu().numpy()
    tgt_box = gt[f][TARGET]
    others = [(i,b) for i,b in gt[f].items() if i != TARGET]
    iA = int(np.argmax([iou(b, tgt_box) for b in boxes]))
    if click is None:
        click = kp_desc(kps[iA], boxes[iA]); continue
    if not others: continue
    # the distractor that is spatially closest to the target's box = the likely impostor
    dist_b = min(others, key=lambda kv: np.linalg.norm(kv[1][:2]-tgt_box[:2]))[1]
    iB = int(np.argmax([iou(b, dist_b) for b in boxes]))
    if iB == iA: continue
    dA, dB = cosd(click, kp_desc(kps[iA], boxes[iA])), cosd(click, kp_desc(kps[iB], boxes[iB]))
    if dA is None or dB is None: continue
    rows.append(dict(f=f, dA=round(dA,4), dB=round(dB,4), iouA=round(iou(boxes[iA],tgt_box),3),
                     sep=(dB-dA)))
n = len(rows); wins = sum(1 for r in rows if r["dB"] > r["dA"])
print(f"frames_used={n}  pose_dist(target)={np.mean([r['dA'] for r in rows]):.4f}  "
      f"pose_dist(impostor)={np.mean([r['dB'] for r in rows]):.4f}  "
      f"separation_wins={wins}/{n} = {wins/max(n,1):.3f}")
print("mean margin (dB-dA) =", round(float(np.mean([r['sep'] for r in rows])),4),
      " median iou_A =", round(float(np.median([r['iouA'] for r in rows])),3))
Path("results").mkdir(exist_ok=True)
Path("results/e045_pose_probe.json").write_text(json.dumps(dict(n=n, rows=rows), indent=1))


