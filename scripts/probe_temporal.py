"""How much of the remaining detection loss is recoverable from *time*?

Three probes have now closed the frame-level routes to more recall on MOT17 with
`yolo26m@1280`:

  * `probe_tiling.py`  - slicing (SAHI) does not help: 2x2 == 1x1, 3x2 and 4x3 are worse.
  * `probe_recall.py`  - native resolution 1920 is worse than 1280; `yolo26l` is worse than
                         `yolo26m` (pooled recall 0.707 vs 0.733).
  * `probe_miss.py`     - lowering `conf` does raise recall (+19 pp to conf 0.01) but the
                         marginal detections are only ~55 % true, so the pipeline-level
                         trade is near zero-sum (HOTA +/-0.3 pp).

The question this script answers is therefore: **is there any recall left that a
temporal method could take?**  It partitions the false negatives of the shipped
pipeline (conf 0.25) into three disjoint classes:

  A. *same-frame*  - a low-confidence (<0.25) box already overlaps the missed person in
                     the SAME frame at IoU >= 0.5.  A threshold change would take it, and
                     E002 showed that trade is near zero-sum.
  B. *temporal-only* - nothing in this frame, but a low-confidence detection overlaps the
                     person in a neighbouring frame (t +/- K).  Only a track-before-detect /
                     tubelet method can use this.
  C. *untouched*    - nothing in the window at <=0.01 confidence.  No method that starts
                     from this detector can recover it.

Class B is the ceiling for H4 (track-before-detect).  If B is small, H4 is dead and the
honest conclusion is that the detector is saturated for this task and the contribution
has to be in association / identity, not detection.

    python scripts/probe_temporal.py                  # MOT17-02, +/-2 frames
    python scripts/probe_temporal.py --confirm         # 5 sequences
"""

from __future__ import annotations

import json
import sys
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
DETECTOR = Path("results/weights/yolo26m.pt").resolve()
SHIPPED_CONF = 0.25
PROBE_CONF = 0.01
WINDOW = 2
IOS = 0.5


def match_mask(det_boxes: np.ndarray, gt_boxes: np.ndarray, thr: float = IOS) -> np.ndarray:
    """Greedy 1-1 matching; returns a bool mask over GT of what got matched."""
    n_gt = len(gt_boxes)
    matched = np.zeros(n_gt, dtype=bool)
    if n_gt == 0 or len(det_boxes) == 0:
        return matched
    iou = iou_matrix(det_boxes, gt_boxes)
    used_r: set[int] = set()
    used_c: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for r, c in order:
        if iou[r, c] < thr:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r))
        used_c.add(int(c))
        matched[int(c)] = True
    return matched


def main() -> int:
    confirm = "--confirm" in sys.argv
    if confirm:
        seq_dirs = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
    else:
        seq_dirs = [TRAIN / "MOT17-02-FRCNN"]

    rows: list[dict] = []
    pooled = {"fn": 0, "A": 0, "B": 0, "C": 0, "gt": 0, "tp_high": 0}

    for seq_dir in seq_dirs:
        if not seq_dir.exists():
            print(f"{seq_dir.name}: MISSING")
            continue
        seq = load_mot_sequence(seq_dir, subsample=SUBSAMPLE, max_frames=N_FRAMES, min_conf=1.0)

        # Per-frame: GT boxes and the low-confidence detection pool (>= 0.01).
        gts: list[np.ndarray] = []
        dets_low: list[np.ndarray] = []
        dets_high: list[np.ndarray] = []
        for i in range(len(seq)):
            marks = seq.gt_at(i)
            gts.append(np.asarray([g.xyxy for g in marks], dtype=np.float32).reshape(-1, 4))

        cfg = Config().clone(**{
            "detector.weights": str(DETECTOR),
            "detector.imgsz": 1280,
            "detector.conf": PROBE_CONF,
        })
        det = PersonDetector(cfg.detector)
        det.warmup()
        for i in range(len(seq)):
            found = det.detect(seq.frame(i))
            boxes = np.asarray([d.xyxy for d in found], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray([d.score for d in found], dtype=np.float32)
            dets_low.append(boxes)
            dets_high.append(boxes[scores >= SHIPPED_CONF])

        n_gt = fn = a = b = c = tp_high = 0
        for t, gt in enumerate(gts):
            if len(gt) == 0:
                continue
            n_gt += len(gt)
            ok_high = match_mask(dets_high[t], gt)
            tp_high += int(ok_high.sum())
            for gi in np.nonzero(~ok_high)[0]:
                fn += 1
                box = gt[gi][None, :]
                # A: a sub-threshold box in THIS frame already overlaps it.
                if len(dets_low[t]) and iou_matrix(dets_low[t], box).max() >= IOS:
                    a += 1
                    continue
                # B: nothing here, but a low-confidence box overlaps it nearby in time.
                hit = False
                for dt in range(1, WINDOW + 1):
                    for tj in (t - dt, t + dt):
                        if 0 <= tj < len(dets_low) and len(dets_low[tj]):
                            if iou_matrix(dets_low[tj], box).max() >= IOS:
                                hit = True
                                break
                    if hit:
                        break
                if hit:
                    b += 1
                else:
                    c += 1

        row = {
            "sequence": seq.name, "frames": len(seq),
            "gt": n_gt, "tp_at_conf025": tp_high, "fn": fn,
            "A_same_frame_lowscore": a, "B_temporal_only": b, "C_untouched": c,
            "window": WINDOW, "probe_conf": PROBE_CONF,
            "A_frac_of_fn": a / max(fn, 1), "B_frac_of_fn": b / max(fn, 1),
            "C_frac_of_fn": c / max(fn, 1),
            "recall_now": tp_high / max(n_gt, 1),
            "recall_if_A": (tp_high + a) / max(n_gt, 1),
            "recall_if_A_B": (tp_high + a + b) / max(n_gt, 1),
        }
        rows.append(row)
        pooled["gt"] += row["gt"]
        pooled["tp_high"] += row["tp_at_conf025"]
        pooled["fn"] += row["fn"]
        pooled["A"] += row["A_same_frame_lowscore"]
        pooled["B"] += row["B_temporal_only"]
        pooled["C"] += row["C_untouched"]
        print(f"{seq.name}: GT {n_gt}  TP@0.25 {tp_high}  FN {fn}")
        print(f"   A same-frame low-score {a:5d} ({a / max(fn, 1) * 100:5.1f}% of FN)"
              f"   B temporal-only {b:5d} ({b / max(fn, 1) * 100:5.1f}%)"
              f"   C untouched {c:5d} ({c / max(fn, 1) * 100:5.1f}%)")
        print(f"   recall now {row['recall_now']:.3f} -> if A {row['recall_if_A']:.3f}"
              f" -> if A+B {row['recall_if_A_B']:.3f}")

    out = Path("results/recall_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "temporal.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if len(seq_dirs) > 1:
        gt = pooled["gt"]
        print("\n--- pooled ---")
        print(f"  GT {gt}  TP@0.25 {pooled['tp_high']}  FN {pooled['fn']}")
        for k, label in (("A", "A same-frame low-score"), ("B", "B temporal-only"),
                         ("C", "C untouched")):
            print(f"  {label:<24} {pooled[k]:6d}  ({pooled[k] / max(pooled['fn'], 1) * 100:5.1f}% of FN)")
        base = pooled["tp_high"] / max(gt, 1)
        print(f"  recall now {base:.3f} -> if A {(pooled['tp_high'] + pooled['A']) / max(gt, 1):.3f}"
              f" -> if A+B {(pooled['tp_high'] + pooled['A'] + pooled['B']) / max(gt, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
