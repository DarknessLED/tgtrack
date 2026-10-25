"""Where exactly does the missing recall go?

`scripts/probe_recall.py` measured recall 0.733 at `yolo26m@1280`; `scripts/probe_tiling.py`
then showed slicing the frame does NOT recover the rest (2x2 identical, 3x2 and 4x3 worse),
so the misses are not simply "people too small for one letterbox".  This probe asks the two
questions that decide what to do next:

1. **Is it the threshold?**  Recall is measured against a *confidence cutoff*.  If a large
   part of the missing 27% is sitting at conf 0.02-0.20, then the fix is not a better
   detector, it is carrying low-score detections into the association stage (ByteTrack's
   second pass, but for the *target* layer where a single person is at stake).  That is
   cheap and high-upside, so measure it before building anything.

2. **Is it genuine failure?**  For every visible GT box that is missed, report its height
   in pixels, its visibility label, and the best IoU any detection achieves against it.  If
   best IoU is ~0 the detector never saw them (scale/occlusion limit); if best IoU is
   0.3-0.5 they are found but mis-localised (a box-refinement problem, different fix).

Also sweeps `imgsz=1920` (native resolution for these 1920x1080 sequences) because 1280
letterboxing DOWNSCALES them, and drops conf to 0.01 to see the detector's actual ceiling.

    python scripts/probe_miss.py                 # MOT17-02
    python scripts/probe_miss.py --confirm       # 5 sequences, conf sweep only
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
CONFS = [0.25, 0.15, 0.10, 0.05, 0.02, 0.01]
SETTINGS = [("m", 1280), ("m", 1920)]


def load_frames(seq_dir: Path) -> tuple[str, list[np.ndarray], list[list]]:
    seq = load_mot_sequence(seq_dir, subsample=SUBSAMPLE, max_frames=N_FRAMES, min_conf=1.0)
    frames: list[np.ndarray] = []
    gts: list[list] = []
    for i in range(len(seq)):
        marks = [g for g in seq.gt_at(i) if float(getattr(g, "visibility", 1.0)) >= VIS_MIN]
        if not marks:
            continue
        frames.append(seq.frame(i))
        gts.append(marks)
    return seq.name, frames, gts


def match(det_boxes: np.ndarray, gt_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy 1-1 matching at IoU>=0.5. Returns (matched mask per GT, best IoU per GT)."""
    n_gt = len(gt_boxes)
    matched = np.zeros(n_gt, dtype=bool)
    best = np.zeros(n_gt, dtype=np.float32)
    if n_gt == 0 or len(det_boxes) == 0:
        return matched, best
    iou = iou_matrix(det_boxes, gt_boxes)
    best[:] = iou.max(axis=0)
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
        matched[int(c)] = True
    return matched, best


def detect_at(det: PersonDetector, img: np.ndarray) -> np.ndarray:
    dets = det.detect(img)
    return np.asarray([d.xyxy for d in dets], dtype=np.float32).reshape(-1, 4)


def main() -> int:
    confirm = "--confirm" in sys.argv
    if confirm:
        seq_dirs = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
    else:
        seq_dirs = [TRAIN / "MOT17-02-FRCNN"]

    rows: list[dict] = []
    miss_rows: list[dict] = []
    for seq_dir in seq_dirs:
        if not seq_dir.exists():
            print(f"{seq_dir.name}: MISSING")
            continue
        name, frames, gts = load_frames(seq_dir)
        print(f"\n{name}  ({len(frames)} frames, {sum(len(g) for g in gts)} visible GT)")

        for tag, imgsz in SETTINGS:
            path = WEIGHTS[tag]
            if not path.exists():
                print(f"  yolo26{tag}@{imgsz}: MISSING {path}")
                continue
            for conf in CONFS:
                cfg = Config().clone(**{
                    "detector.weights": str(path),
                    "detector.imgsz": int(imgsz),
                    "detector.conf": float(conf),
                })
                det = PersonDetector(cfg.detector)
                det.warmup()
                matched = total = ndet = 0
                detail: list[tuple[np.ndarray, np.ndarray, list]] = []
                t0 = time.perf_counter()
                for img, marks in zip(frames, gts):
                    boxes = detect_at(det, img)
                    gt_boxes = np.asarray([g.xyxy for g in marks], dtype=np.float32)
                    mm, bb = match(boxes, gt_boxes)
                    matched += int(mm.sum())
                    total += len(mm)
                    ndet += len(boxes)
                    detail.append((boxes, mm, marks))
                ms = (time.perf_counter() - t0) / max(len(frames), 1) * 1000.0
                rec = matched / max(total, 1)
                rows.append({
                    "sequence": name, "model": f"yolo26{tag}", "imgsz": imgsz,
                    "conf": conf, "frames": len(frames), "recall": rec,
                    "matched": matched, "gt": total,
                    "det_per_frame": ndet / max(len(frames), 1), "ms_per_frame": ms,
                })
                print(f"  yolo26{tag}@{imgsz:<4} conf {conf:<5} recall {rec:.3f}  "
                      f"{ndet / max(len(frames), 1):6.1f} det/frame  {ms:7.2f} ms")

                # Miss anatomy is only interesting at the shipped threshold / setting.
                if abs(conf - 0.25) < 1e-9 and imgsz == 1280:
                    heights: list[float] = []
                    vises: list[float] = []
                    best_ious: list[float] = []
                    for boxes, mm, marks in detail:
                        for i, g in enumerate(marks):
                            if mm[i]:
                                continue
                            x1, y1, x2, y2 = [float(v) for v in g.xyxy]
                            heights.append(max(y2 - y1, 1.0))
                            vises.append(float(getattr(g, "visibility", 1.0)))
                            if len(boxes):
                                best_ious.append(float(iou_matrix(boxes, g.xyxy.reshape(1, 4)).max()))
                            else:
                                best_ious.append(0.0)
                    if heights:
                        h = np.asarray(heights)
                        v = np.asarray(vises)
                        b = np.asarray(best_ious)
                        miss_rows.append({
                            "sequence": name, "model": f"yolo26{tag}", "imgsz": imgsz,
                            "conf": conf, "missed": int(h.size),
                            "h_p50": float(np.percentile(h, 50)),
                            "h_p90": float(np.percentile(h, 90)),
                            "frac_h_lt40": float((h < 40).mean()),
                            "frac_h_lt80": float((h < 80).mean()),
                            "vis_p50": float(np.percentile(v, 50)),
                            "frac_vis_lt50": float((v < 0.5).mean()),
                            "best_iou_p50": float(np.percentile(b, 50)),
                            "frac_best_iou_lt0.1": float((b < 0.1).mean()),
                            "frac_best_iou_0.1_0.5": float(((b >= 0.1) & (b < 0.5)).mean()),
                        })
                        mr = miss_rows[-1]
                        print(f"    missed {mr['missed']:4d} boxes | height p50 {mr['h_p50']:5.1f} "
                              f"px ({(mr['frac_h_lt40']) * 100:4.0f}% <40px) | vis p50 "
                              f"{mr['vis_p50']:.2f} ({mr['frac_vis_lt50'] * 100:4.0f}% <0.5) | "
                              f"bestIoU p50 {mr['best_iou_p50']:.2f} "
                              f"({mr['frac_best_iou_lt0.1'] * 100:4.0f}% never touched)")

    out = Path("results/recall_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "miss.json").write_text(
        json.dumps({"conf_sweep": rows, "miss_anatomy": miss_rows}, indent=2), encoding="utf-8")

    if len(seq_dirs) > 1:
        print("\n--- pooled over sequences ---")
        for setting in SETTINGS:
            for conf in CONFS:
                sel = [r for r in rows if r["model"] == f"yolo26{setting[0]}"
                       and r["imgsz"] == setting[1] and abs(r["conf"] - conf) < 1e-9]
                if not sel:
                    continue
                m = sum(r["matched"] for r in sel)
                g = sum(r["gt"] for r in sel)
                print(f"  yolo26{setting[0]}@{setting[1]:<4} conf {conf:<5} pooled recall "
                      f"{m / max(g, 1):.3f}  ({m}/{g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
