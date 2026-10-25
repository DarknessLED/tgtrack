"""E046: param-matched detector swap. Same protocol as E003 (5 MOT17 seqs, 30 frames, subsample 5,
VIS_MIN 0.25, greedy 1-1 IoU>=0.5) so numbers are directly comparable to yolo26m@1280 = 0.733."""
import sys, time, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_recall import load_frames, recall_for, TRAIN
from ultralytics import YOLO

SEQS = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
MODELS = [
    ("yolo26m",  Path("results/weights/yolo26m.pt"),              1280),
    ("ch_yolov8n", Path("data/weights/crowdhuman_yolov8n_best.pt"), 1280),
    ("ch_yolov8n@1536", Path("data/weights/crowdhuman_yolov8n_best.pt"), 1536),
]
cache, rows = {}, []
for tag, path, imgsz in MODELS:
    if not path.exists():
        print(f"{tag}: MISSING"); continue
    try:
        model = YOLO(str(path))
    except Exception as e:
        print(f"{tag}: LOAD FAILED {type(e).__name__}: {str(e)[:160]}"); continue
    n_par = sum(p.numel() for p in model.model.parameters())
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    person_only = len(names) == 1
    cls = None if person_only else [0]
    m_tot = g_tot = n_det = 0; t0 = time.perf_counter(); nf = 0
    for seq_dir in SEQS:
        if not seq_dir.exists(): continue
        if str(seq_dir) not in cache: cache[str(seq_dir)] = load_frames(seq_dir)
        name, frames, gts = cache[str(seq_dir)]
        for img, gt in zip(frames, gts):
            r = model.predict(img, imgsz=imgsz, conf=0.25, iou=0.7, classes=cls, max_det=300, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.zeros((0,4), np.float32)
            m, t = recall_for(boxes.astype(np.float32), gt)
            m_tot += m; g_tot += t; n_det += len(boxes); nf += 1
    ms = (time.perf_counter() - t0) / max(nf,1) * 1000
    row = dict(model=tag, imgsz=imgsz, params=n_par, classes=len(names), person_only=person_only,
               frames=nf, recall=round(m_tot/max(g_tot,1),4), matched=m_tot, gt=g_tot,
               det_per_frame=round(n_det/max(nf,1),1), ms_per_frame=round(ms,1))
    rows.append(row)
    print(f"{tag:12s} imgsz={imgsz} params={n_par/1e6:5.2f}M classes={len(names):3d} "
          f"recall={row['recall']:.3f} ({m_tot}/{g_tot})  {row['det_per_frame']:5.1f} det/f  {ms:6.1f} ms")
Path("results").mkdir(exist_ok=True)
Path("results/e046_detector_swap.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

