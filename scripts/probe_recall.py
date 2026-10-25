"""How much of a MOT17 frame does our detector actually find?

perception-dev measured DetA 0.249-0.264 across all six MOT trackers, i.e. the
tracker is not the bottleneck - the DETECTOR is.  This probe isolates that:
same frames, same GT, only the model and the input resolution change.

Reported per setting: recall at IoU 0.5 against GT boxes that are actually
visible, mean detections/frame, and latency.  Run from the repo root.

    python scripts/probe_recall.py            # full sweep on MOT17-02
    python scripts/probe_recall.py --confirm  # n@640 vs m@1280 on 5 sequences
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tgtrack.config import Config  # noqa: E402
from tgtrack.data import load_mot_sequence  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

TRAIN = Path("data/raw/mot17/MOT17/train")
N_FRAMES = 30
SUBSAMPLE = 5
WEIGHTS = {
    "n": Path("yolo26n.pt").resolve(),
    "s": Path("results/weights/yolo26s.pt").resolve(),
    "m": Path("results/weights/yolo26m.pt").resolve(),
    "l": Path("data/weights/yolo26l.pt").resolve(),
    "x": Path("data/weights/yolo26x.pt").resolve(),
}
FULL_SWEEP = [("n", 640), ("n", 1280), ("s", 640), ("s", 1280), ("m", 640), ("m", 1280)]
CONFIRM = [("n", 640), ("m", 1280)]
# Does a bigger backbone buy the recall that thresholds could not? (E003 headroom check)
BIG = [("m", 1280), ("l", 1280)]
VIS_MIN = 0.25  # a box that is 75% hidden is not a detector failure


def recall_for(det_boxes: np.ndarray, gt_boxes: np.ndarray) -> tuple[int, int]:
    """Greedy 1-1 matching at IoU >= 0.5. Returns (matched, total)."""
    if len(gt_boxes) == 0:
        return 0, 0
    if len(det_boxes) == 0:
        return 0, len(gt_boxes)
    iou = iou_matrix(det_boxes, gt_boxes)
    matched = 0
    used_r: set[int] = set()
    used_c: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for r, c in order:
        if iou[r, c] < 0.5:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r))
        used_c.add(int(c))
        matched += 1
    return matched, len(gt_boxes)


def load_frames(seq_dir: Path) -> tuple[str, list[np.ndarray], list[np.ndarray]]:
    seq = load_mot_sequence(seq_dir, subsample=SUBSAMPLE, max_frames=N_FRAMES, min_conf=1.0)
    frames: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for i in range(len(seq)):
        gt = [g for g in seq.gt_at(i) if float(getattr(g, "visibility", 1.0)) >= VIS_MIN]
        if not gt:
            continue
        frames.append(seq.frame(i))
        gts.append(np.asarray([g.xyxy for g in gt], dtype=np.float32))
    return seq.name, frames, gts


def run(seq_dir: Path, setting: tuple[str, int], cache: dict) -> dict | None:
    tag, imgsz = setting
    path = WEIGHTS[tag]
    if not path.exists():
        print(f"  yolo26{tag}: MISSING {path}")
        return None
    key = str(seq_dir)
    if key not in cache:
        cache[key] = load_frames(seq_dir)
    name, frames, gts = cache[key]

    cfg = Config().clone(**{"detector.weights": str(path), "detector.imgsz": int(imgsz)})
    det = PersonDetector(cfg.detector)
    det.warmup()
    matched = total = ndet = 0
    t0 = time.perf_counter()
    for img, gt in zip(frames, gts):
        dets = det.detect(img)
        boxes = np.asarray([d.xyxy for d in dets], dtype=np.float32).reshape(-1, 4)
        m, t = recall_for(boxes, gt)
        matched += m
        total += t
        ndet += len(boxes)
    dt = time.perf_counter() - t0
    ms = dt / max(len(frames), 1) * 1000.0
    return {
        "sequence": name,
        "model": f"yolo26{tag}",
        "imgsz": imgsz,
        "frames": len(frames),
        "recall": matched / max(total, 1),
        "matched": matched,
        "gt": total,
        "det_per_frame": ndet / max(len(frames), 1),
        "ms_per_frame": ms,
        "fps": 1000.0 / ms,
    }


def main() -> int:
    confirm = "--confirm" in sys.argv
    big = "--big" in sys.argv
    if big:
        settings = BIG
    elif confirm:
        settings = CONFIRM
    else:
        settings = FULL_SWEEP
    if confirm or big:
        seq_dirs = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
    else:
        seq_dirs = [TRAIN / "MOT17-02-FRCNN"]

    rows: list[dict] = []
    cache: dict = {}
    for seq_dir in seq_dirs:
        if not seq_dir.exists():
            print(f"{seq_dir.name}: MISSING")
            continue
        print(f"\n{seq_dir.name}")
        for setting in settings:
            r = run(seq_dir, setting, cache)
            if r is None:
                continue
            rows.append(r)
            print(f"  yolo26{setting[0]}@{setting[1]:<4} recall {r['recall']:.3f}  "
                  f"{r['det_per_frame']:6.1f} det/frame  {r['ms_per_frame']:7.2f} ms  "
                  f"{r['fps']:6.1f} FPS  (n={r['frames']}f, {r['gt']} GT)")

    out = Path("results/recall_probe")
    out.mkdir(parents=True, exist_ok=True)
    fname = "recall_confirm.json" if confirm else "recall.json"
    (out / fname).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if len(seq_dirs) > 1:
        print("\n--- pooled over sequences ---")
        for setting in settings:
            sel = [r for r in rows if r["model"] == f"yolo26{setting[0]}" and r["imgsz"] == setting[1]]
            if not sel:
                continue
            m = sum(r["matched"] for r in sel)
            g = sum(r["gt"] for r in sel)
            ms = float(np.mean([r["ms_per_frame"] for r in sel]))
            print(f"  yolo26{setting[0]}@{setting[1]:<4} pooled recall {m / max(g, 1):.3f}  "
                  f"({m}/{g} boxes)  {ms:7.2f} ms avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
