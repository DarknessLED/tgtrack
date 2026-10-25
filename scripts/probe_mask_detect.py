"""Does the partial-view mask break the DETECTOR, or the tracker?

The partial-view ablation paints half of the target's box black and then rewrites
the target's ground truth to only the still-visible half.  If the detector can no
longer find the person at all under that mask, then the resulting
`target_success_rate` is a DETECTOR number wearing a tracker's name, and quoting
it as evidence for the part-matching mechanism would be dishonest.

This probe answers that directly, with no tracker involved:
  * how many person detections survive the mask,
  * how often some detection overlaps the REWRITTEN (visible-half) GT by IoU>=0.5,
  * how often some detection overlaps the ORIGINAL full-body GT by IoU>=0.5.

If the second number collapses while the third stays high, the detector still
sees the person and merely reports a full-body box -- a metric-definition
artefact, not a perception failure.  If both collapse, the mask removed the
person from the image and the ablation cannot test the tracker at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402

import tgtrack  # noqa: E402
from tgtrack.config import Config  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

import run_target as RT  # noqa: E402

SEQ = "data/raw/mot17/MOT17/train/MOT17-02-FRCNN"
N = 60
MODES = ("none", "lower", "upper", "left", "right")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=SEQ)
    ap.add_argument("--max-frames", type=int, default=N)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--target-id", type=int, default=None)
    args = ap.parse_args()

    seq = tgtrack.load_mot_sequence(
        args.sequence, subsample=args.subsample, max_frames=args.max_frames
    )
    n = len(seq)

    counts: dict[int, int] = {}
    for f in range(n):
        for b in seq.gt_at(f):
            counts[int(b.track_id)] = counts.get(int(b.track_id), 0) + 1
    tid = args.target_id if args.target_id is not None else max(counts, key=lambda k: counts[k])
    print(f"sequence={seq.name} frames={n} subsample={args.subsample} target_id={tid} gt_frames={counts[tid]}")

    cfg = Config()
    print(f"detector={Path(cfg.detector.weights).name} imgsz={cfg.detector.imgsz} conf={cfg.detector.conf}")
    det = PersonDetector(cfg.detector)
    det.warmup()

    out: dict[str, dict] = {}
    for mode in MODES:
        plan = RT.build_mask_plan(seq, tid, mode, limit=N)
        n_det: list[int] = []
        hit_vis: list[float] = []
        hit_full: list[float] = []
        iou_vis: list[float] = []
        iou_full: list[float] = []
        for f in range(N):
            gv = plan.gt_target.get(f)
            gf = plan.full_gt_target.get(f)
            if gv is None:
                continue
            frame = plan.apply(seq.frame(f), f)
            if frame is None:
                continue
            dets = det.detect(frame)
            n_det.append(len(dets))
            if not dets:
                hit_vis.append(0.0)
                hit_full.append(0.0)
                iou_vis.append(0.0)
                iou_full.append(0.0)
                continue
            boxes = np.asarray([d.xyxy for d in dets], dtype=np.float64)
            iv = float(iou_matrix(np.asarray(gv, np.float64).reshape(1, 4), boxes)[0].max())
            iff = float(iou_matrix(np.asarray(gf, np.float64).reshape(1, 4), boxes)[0].max())
            iou_vis.append(iv)
            iou_full.append(iff)
            hit_vis.append(1.0 if iv >= 0.5 else 0.0)
            hit_full.append(1.0 if iff >= 0.5 else 0.0)

        row = {
            "gt_frames": len(iou_vis),
            "mean_n_det": round(float(np.mean(n_det)), 2) if n_det else 0.0,
            "det_rate_vs_visible_gt": round(float(np.mean(hit_vis)), 4) if hit_vis else 0.0,
            "det_rate_vs_full_gt": round(float(np.mean(hit_full)), 4) if hit_full else 0.0,
            "mean_best_iou_visible_gt": round(float(np.mean(iou_vis)), 4) if iou_vis else 0.0,
            "mean_best_iou_full_gt": round(float(np.mean(iou_full)), 4) if iou_full else 0.0,
            "mean_visible_frac": round(float(plan.visible_fraction()), 4),
        }
        out[mode] = row
        print(f"{mode:6s} {row}")

    dest = Path("results/probe_mask_detect.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
