"""E010: where does recall die between the detector and the tracker output?

E004 measured `DetRe`-level recall in the harness at **0.51** while the detector alone fires on
~0.72 of GT in the probes, and `probe_temporal` reproduced the same 0.47-0.51 on a different
subset. That gap has to be explained before any more strategy work, because `DetA` is the
dominant term of HOTA.

`AllPersonTracker.update` sets `track_id = None` on a detection the tracker did not activate
(low-score second-stage miss / failed re-identification), and `pred_boxes_from_dets` only
emits activated boxes. So a detection can be perfectly good and still never reach the output.
This probe measures exactly that, per score band:

  * `det_recall`  — GT boxes matched (IoU >= 0.5) by *any* detection at `conf`;
  * `out_recall`  — GT boxes matched by *emitted* (activated) boxes;
  * `lost`        — detections that match a real GT box but were **not activated**;
  * activation rate per score band.

If `lost` is large, the recall deficit is tracker gating and is fixable in config
(`track_high_thresh` / `track_low_thresh` / `new_track_thresh`) rather than by a better
detector. If it is small, the detector really is the limit and DetA is capped.

    python scripts/probe_gating.py --sequences MOT17-02-FRCNN,MOT17-04-FRCNN --max-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from tgtrack.config import Config  # noqa: E402
from tgtrack.data import iter_frames, load_mot_sequence  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.mot_layer import AllPersonTracker  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

BANDS = ((0.25, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 1.01))


def band_of(score: float) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= score < hi:
            return i
    return -1


def _matched(det_boxes: np.ndarray, gt_boxes: np.ndarray, thr: float = 0.5) -> int:
    """How many GT boxes are matched by at least one detection (one-to-one is not needed)."""
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return 0
    m = iou_matrix(det_boxes, gt_boxes)
    return int((m.max(axis=0) >= thr).sum())


def _on_real_person(box: np.ndarray, gt_boxes: np.ndarray, thr: float = 0.5) -> bool:
    if len(gt_boxes) == 0:
        return False
    return bool(float(iou_matrix(box.reshape(1, 4), gt_boxes).max()) >= thr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--sequences", default="MOT17-02-FRCNN,MOT17-04-FRCNN")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--conf", type=float, default=None, help="detector conf (default: config)")
    ap.add_argument("--tracker", default="tracktrack.yaml")
    ap.add_argument("--out", default="results/recall_probe/gating.json")
    opts = ap.parse_args()

    cfg = Config()
    if opts.conf is not None:
        cfg.detector.conf = float(opts.conf)
    cfg.mot.tracker = opts.tracker
    detector = PersonDetector(cfg.detector)
    detector.warmup()
    tracker = AllPersonTracker(cfg.mot, detector)

    summary = []
    for name in [s.strip() for s in opts.sequences.split(",") if s.strip()]:
        seq_dir = Path(opts.data_root) / name
        if not seq_dir.is_dir():
            print(f"missing {seq_dir}")
            continue
        seq = load_mot_sequence(seq_dir, subsample=opts.subsample, max_frames=opts.max_frames)
        tracker.reset()

        agg = {
            "frames": 0, "gt": 0,
            "det_boxes": 0, "out_boxes": 0,
            "det_matched": 0, "det_gt_matched": 0, "out_matched": 0,
            "lost_boxes": 0,
            "bands": [{"n": 0, "activated": 0, "on_gt": 0, "lost_on_gt": 0} for _ in BANDS],
        }
        try:
            for frame_idx, img, gt_boxes in iter_frames(seq, decode=True):
                if img is None:
                    continue
                gt = (np.asarray([np.asarray(g.xyxy, np.float64).reshape(4) for g in gt_boxes], np.float64)
                      if gt_boxes else np.zeros((0, 4)))
                dets = detector.detect(img)
                tracker.update(img, dets)
                agg["frames"] += 1
                agg["gt"] += len(gt)

                for d in dets:
                    box = np.asarray(d.xyxy, np.float64).reshape(4)
                    b = band_of(float(d.score))
                    on_gt = _on_real_person(box, gt)
                    agg["det_boxes"] += 1
                    if on_gt:
                        agg["det_matched"] += 1
                    if d.track_id is not None:
                        agg["out_boxes"] += 1
                        if b >= 0:
                            agg["bands"][b]["activated"] += 1
                    else:
                        if on_gt:
                            agg["lost_boxes"] += 1
                            if b >= 0:
                                agg["bands"][b]["lost_on_gt"] += 1
                    if b >= 0:
                        agg["bands"][b]["n"] += 1
                        if on_gt:
                            agg["bands"][b]["on_gt"] += 1

                all_boxes = np.asarray([np.asarray(d.xyxy, np.float64).reshape(4) for d in dets],
                                       np.float64) if dets else np.zeros((0, 4))
                agg["det_gt_matched"] += _matched(all_boxes, gt)
                emitted = np.asarray(
                    [np.asarray(d.xyxy, np.float64).reshape(4) for d in dets if d.track_id is not None],
                    np.float64) if any(d.track_id is not None for d in dets) else np.zeros((0, 4))
                agg["out_matched"] += _matched(emitted, gt)
        finally:
            seq.clear_cache()

        n_gt = max(agg["gt"], 1)
        row = {
            "sequence": name, **agg,
            "gt_per_frame": agg["gt"] / max(agg["frames"], 1),
            "det_per_frame": agg["det_boxes"] / max(agg["frames"], 1),
            "out_per_frame": agg["out_boxes"] / max(agg["frames"], 1),
            "det_recall": agg["det_gt_matched"] / n_gt,   # GT boxes matched by ANY detection
            "out_recall": agg["out_matched"] / n_gt,
            "activation_rate": agg["out_boxes"] / max(agg["det_boxes"], 1),
            "lost_on_gt": agg["lost_boxes"],
            "bands": agg["bands"],
        }
        summary.append(row)
        print(f"\n=== {name}  ({agg['frames']} frames, {agg['gt']} GT) ===")
        print(f"  GT/frame {row['gt_per_frame']:.2f}  dets/frame {row['det_per_frame']:.2f}  "
              f"emitted/frame {row['out_per_frame']:.2f}  activation {row['activation_rate']:.1%}")
        print(f"  det_recall {row['det_recall']:.4f}   out_recall {row['out_recall']:.4f}   "
              f"LOST ON GT {row['lost_on_gt']}")
        print(f"  {'band':<14}{'dets':>7}{'act%':>7}{'onGT':>7}{'lostOnGT':>10}")
        for (lo, hi), b in zip(BANDS, agg["bands"]):
            act = b["activated"] / max(b["n"], 1)
            print(f"  {f'{lo:.2f}-{hi:.2f}':<14}{b['n']:>7}{act:>6.1%}{b['on_gt']:>7}{b['lost_on_gt']:>10}")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opts": vars(opts), "sequences": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
