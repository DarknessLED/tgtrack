"""Does slicing the frame beat upscaling it?

`scripts/probe_recall.py` established the detector is the bottleneck (recall 0.733 at
`yolo26m@1280`), and that the wall-clock is dominated by *preprocessing*, not the
network.  That leaves an obvious untried lever: instead of feeding one 1920x1080 frame
to a 1280-letterboxed network (which SHRINKS it), feed several overlapping slices, each
of which is *upscaled* instead.  Small people then occupy more pixels, which is exactly
what the miss analysis says we lose.

This is the SAHI idea (slicing-aided hyper inference, aerial detection) applied to a
level-view MOT benchmark, and it costs zero training -- it is pure inference plumbing.

Reported per setting: recall at IoU 0.5 against GT boxes that are actually visible,
detections/frame (a recall gain that doubles the detection count is not a gain),
and latency.  Run from the repo root.

    python scripts/probe_tiling.py                       # MOT17-02, grid sweep
    python scripts/probe_tiling.py --confirm              # best grids, 5 sequences
    python scripts/probe_tiling.py --grid 1x1,2x2 --model m
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
}
VIS_MIN = 0.25
NMS_IOU = 0.55

# (model, imgsz, grid)
SWEEP = [
    ("m", 1280, "1x1"),
    ("m", 1280, "2x2"),
    ("m", 1280, "3x2"),
    ("m", 1280, "4x3"),
    ("n", 1280, "2x2"),
    ("m", 640, "2x2"),
]
CONFIRM = [("m", 1280, "1x1"), ("m", 1280, "2x2"), ("m", 1280, "3x2")]


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


def parse_grid(grid: str) -> tuple[int, int]:
    nx, ny = grid.lower().split("x")
    return int(nx), int(ny)


def slice_rects(width: int, height: int, grid: tuple[int, int],
                overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """SAHI-style overlapping slices; the last row/column is shifted to stay in frame."""
    nx, ny = grid
    if nx <= 1 and ny <= 1:
        return [(0, 0, int(width), int(height))]

    def spans(total: int, n: int) -> list[tuple[int, int]]:
        if n <= 1:
            return [(0, total)]
        span = int(np.ceil(total / (n - overlap * (n - 1))))
        span = min(span, total)
        starts = np.linspace(0, total - span, n)
        return [(int(round(s)), int(round(s)) + span) for s in starts]

    rects: list[tuple[int, int, int, int]] = []
    for y0, y1 in spans(int(height), ny):
        for x0, x1 in spans(int(width), nx):
            rects.append((x0, y0, x1, y1))
    return rects


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = NMS_IOU) -> np.ndarray:
    """Greedy NMS over (N,4) xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    boxes = boxes.astype(np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iou = iou_matrix(boxes[i][None, :], boxes[rest])[0]
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def pool_boxes(det: PersonDetector, img: np.ndarray, grid: tuple[int, int],
               overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the detector on every slice, map back to frame coords, fuse by NMS."""
    h, w = img.shape[:2]
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    for (x0, y0, x1, y1) in slice_rects(w, h, grid, overlap):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        dets = det.detect(crop)
        for d in dets:
            b = np.asarray(d.xyxy, dtype=np.float32) + np.asarray([x0, y0, x0, y0], np.float32)
            all_boxes.append(b)
            all_scores.append(float(d.score))
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes = np.stack(all_boxes).astype(np.float32)
    scores = np.asarray(all_scores, dtype=np.float32)
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep]


def load_frames(seq_dir: Path) -> tuple[str, list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    seq = load_mot_sequence(seq_dir, subsample=SUBSAMPLE, max_frames=N_FRAMES, min_conf=1.0)
    frames: list[np.ndarray] = []
    gts: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(seq)):
        marks = [g for g in seq.gt_at(i) if float(getattr(g, "visibility", 1.0)) >= VIS_MIN]
        if not marks:
            continue
        frames.append(seq.frame(i))
        gts.append((
            np.asarray([g.xyxy for g in marks], dtype=np.float32),
            np.asarray([g.xyxy for g in seq.gt_at(i)], dtype=np.float32),
        ))
    return seq.name, frames, gts


def run(seq_dir: Path, setting: tuple[str, int, str], cache: dict,
        overlap: float) -> dict | None:
    tag, imgsz, grid_s = setting
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
    grid = parse_grid(grid_s)

    matched = total = ndet = 0
    t0 = time.perf_counter()
    for img, (gt_vis, gt_all) in zip(frames, gts):
        boxes, _ = pool_boxes(det, img, grid, overlap)
        m, t = recall_for(boxes, gt_vis)
        matched += m
        total += t
        ndet += len(boxes)
    dt = time.perf_counter() - t0
    ms = dt / max(len(frames), 1) * 1000.0
    return {
        "sequence": name,
        "model": f"yolo26{tag}",
        "imgsz": imgsz,
        "grid": grid_s,
        "overlap": overlap,
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
    settings = CONFIRM if confirm else SWEEP
    overlap = 0.2
    if "--overlap" in sys.argv:
        overlap = float(sys.argv[sys.argv.index("--overlap") + 1])
    if "--grid" in sys.argv:
        grids = sys.argv[sys.argv.index("--grid") + 1].split(",")
        model = "m"
        if "--model" in sys.argv:
            model = sys.argv[sys.argv.index("--model") + 1]
        imgsz = 1280
        if "--imgsz" in sys.argv:
            imgsz = int(sys.argv[sys.argv.index("--imgsz") + 1])
        settings = [(model, imgsz, g) for g in grids]

    if confirm:
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
            r = run(seq_dir, setting, cache, overlap)
            if r is None:
                continue
            rows.append(r)
            print(f"  {r['model']}@{r['imgsz']:<4} grid {r['grid']:<4} recall {r['recall']:.3f}  "
                  f"{r['det_per_frame']:6.1f} det/frame  {r['ms_per_frame']:7.2f} ms  "
                  f"{r['fps']:6.1f} FPS  (n={r['frames']}f, {r['gt']} GT)")

    out = Path("results/recall_probe")
    out.mkdir(parents=True, exist_ok=True)
    fname = "tiling_confirm.json" if confirm else "tiling.json"
    (out / fname).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if len(seq_dirs) > 1:
        print("\n--- pooled over sequences ---")
        for setting in settings:
            sel = [r for r in rows if r["model"] == f"yolo26{setting[0]}"
                   and r["imgsz"] == setting[1] and r["grid"] == setting[2]]
            if not sel:
                continue
            m = sum(r["matched"] for r in sel)
            g = sum(r["gt"] for r in sel)
            ms = float(np.mean([r["ms_per_frame"] for r in sel]))
            print(f"  yolo26{setting[0]}@{setting[1]} grid {setting[2]:<4} pooled recall "
                  f"{m / max(g, 1):.3f}  ({m}/{g} boxes)  {ms:7.2f} ms avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
