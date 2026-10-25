"""E054: box-level union of the general detector (precise boxes) and the crowd specialist (recall).

E051 diagnosed the split: crowdhuman_yolov8n finds people the general model misses but anchors looser
boxes; yolo26m anchors tight boxes but misses occluded people. Union + NMS should raise recall without
losing box quality. Measured at the detection level first (same protocol as E003/E046).
"""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_recall import load_frames, recall_for, TRAIN
from ultralytics import YOLO

SEQS = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
PAIRS = {"general": Path("results/weights/yolo26m.pt"), "crowd": Path("data/weights/crowdhuman_yolov8n_best.pt")}

def nms(boxes, scores, iou_thr=0.55):
    if len(boxes) == 0: return np.zeros((0, 4), np.float32)
    order = np.argsort(-scores); keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None)
        a = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]); b = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])
        iou = inter / np.maximum(a+b-inter, 1e-9)
        order = rest[iou <= iou_thr]
    return boxes[keep]

models, cache, rows = {}, {}, []
for tag, path in PAIRS.items():
    m = YOLO(str(path)); models[tag] = (m, None if len(m.names) == 1 else [0])
res = {k: dict(m=0, g=0, n=0) for k in ("general", "crowd", "union")}
t0 = time.perf_counter(); nf = 0
for seq_dir in SEQS:
    if not seq_dir.exists(): continue
    if str(seq_dir) not in cache: cache[str(seq_dir)] = load_frames(seq_dir)
    _name, frames, gts = cache[str(seq_dir)]
    for img, gt in zip(frames, gts):
        boxes, allb = {}, []
        for tag, (m, cls) in models.items():
            r = m.predict(img, imgsz=1280, conf=0.25, iou=0.7, classes=cls, max_det=300, verbose=False)[0]
            b = r.boxes.xyxy.cpu().numpy().astype(np.float32) if (r.boxes is not None and len(r.boxes)) else np.zeros((0,4), np.float32)
            s = r.boxes.conf.cpu().numpy().astype(np.float32) if (r.boxes is not None and len(r.boxes)) else np.zeros((0,), np.float32)
            boxes[tag] = b
            mt, tt = recall_for(b, gt); res[tag]["m"] += mt; res[tag]["g"] += tt; res[tag]["n"] += len(b)
            allb.append((b, s))
        ub = np.vstack([b for b, _ in allb]); us = np.concatenate([s for _, s in allb])
        fus = nms(ub, us, 0.55)
        mt, tt = recall_for(fus, gt); res["union"]["m"] += mt; res["union"]["g"] += tt; res["union"]["n"] += len(fus)
        nf += 1
out = []
for k, v in res.items():
    out.append(dict(model=k, recall=round(v["m"]/max(v["g"],1), 4), matched=v["m"], gt=v["g"], det_per_frame=round(v["n"]/max(nf,1), 1)))
    print(f'{k:8s} recall={out[-1]["recall"]:.4f} ({v["m"]}/{v["g"]})  {out[-1]["det_per_frame"]:5.1f} det/frame')
print(f"frames={nf}  wall={time.perf_counter()-t0:.0f}s")
Path("results/e054_union.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
