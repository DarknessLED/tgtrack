"""E062: with the CORRECT target GT, was the true target detected on the frames where we
tracked someone else?

E060's premise came from `seq.gt_at(i)[0]`, which is an arbitrary person (GtBox.track_id is
an internal dense id, not the MOT id) -- so its "true target never detected" conclusion is
retracted.  This probe uses `make_target_gt` (the same resolver the harness's mask plan uses)
and pairs every detection with the target's real box.

The decision this settles:
  * if the true target IS detected on the wrong frames -> an out-of-gate recovery channel is
    implementable and is the shortest path to FAR without paying TSR;
  * if it is NOT detected -> the only way is to stop the initial latch (E061's axis).

Usage:
    python scripts/probe_recovery7.py [--sequence dancetrack0007] [--target-id 1]
                                      [--max-frames 200] [--subsample 2] [--out results/e062]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgtrack.config import Config  # noqa: E402
from tgtrack.data import load_mot_sequence  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.data import make_target_gt  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "raw" / "dancetrack" / "val"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--sequence", default="dancetrack0007")
    ap.add_argument("--target-id", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "e062"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    seq = load_mot_sequence(Path(args.data_root) / args.sequence, subsample=args.subsample,
                            max_frames=args.max_frames)
    target_gt = make_target_gt(seq, args.target_id)
    cfg = Config()
    det = PersonDetector(cfg.detector)
    det.warmup()

    frame = seq.frame(0)
    h, w = frame.shape[:2]
    first = next((i for i in range(len(seq)) if target_gt.get(i) is not None), None)
    if first is None:
        print("no target GT in the window")
        return 3
    box = np.asarray(target_gt[first], float)
    # initialize exactly as the harness does: the click box is the GT of the first frame
    tracker = TargetTracker.load(cfg, device=args.device)
    tracker.initialize(frame, box, frame_idx=first)

    rows = []
    for i in range(len(seq)):
        f = seq.frame(i)
        dets = det.detect(f)
        st = tracker.step(f, dets, frame_idx=i, frame_shape=f.shape[:2])
        g = target_gt.get(i)
        if g is None:
            continue
        g = np.asarray(g, float)
        pred = np.asarray(st.xyxy, float)
        iou = float(iou_matrix(pred[None], g[None])[0, 0])
        best_iou = 0.0
        for d in dets:
            best_iou = max(best_iou, float(iou_matrix(np.asarray(d.xyxy, float)[None], g[None])[0, 0]))
        area_ratio = float(((pred[2] - pred[0]) * (pred[3] - pred[1]))
                           / max(1.0, (g[2] - g[0]) * (g[3] - g[1])))
        rows.append({
            "i": int(i),
            "state": str(st.state),
            "iou_target": round(iou, 4),
            "correct": bool(iou >= 0.5),
            "det_iou_on_true_target": round(best_iou, 4),
            "true_target_detected": bool(best_iou >= 0.5),
            "area_ratio": round(area_ratio, 3),
            "gt_h": round(float(g[3] - g[1]), 1),
            "fsu": int(st.frames_since_update),
        })

    wrong = [r for r in rows if not r["correct"]]
    hit = [r for r in wrong if r["true_target_detected"]]
    near = [r for r in wrong if r["det_iou_on_true_target"] >= 0.3]
    summary = {
        "sequence": args.sequence,
        "target_id": args.target_id,
        "frames_scored": len(rows),
        "tsr": round(1.0 - len(wrong) / max(1, len(rows)), 4),
        "wrong_frames": len(wrong),
        "wrong_frames_true_target_detected": len(hit),
        "wrong_frames_true_target_iou_ge_0_3": len(near),
        "det_hit_rate_all_frames": round(
            float(np.mean([r["true_target_detected"] for r in rows])), 4),
        "wrong_state_counts": {s: sum(1 for r in wrong if r["state"] == s)
                               for s in sorted({r["state"] for r in wrong})},
        "wrong_det_iou_p50": round(float(np.median([r["det_iou_on_true_target"] for r in wrong])), 4)
        if wrong else None,
        "wrong_area_ratio_p50": round(float(np.median([r["area_ratio"] for r in wrong])), 3)
        if wrong else None,
        "correct_area_ratio_p50": round(
            float(np.median([r["area_ratio"] for r in rows if r["correct"]])), 3)
        if any(r["correct"] for r in rows) else None,
        "wrong_gt_h_p50": round(float(np.median([r["gt_h"] for r in wrong])), 1) if wrong else None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))

    if wrong:
        print("\nfirst wrong frames (i, state, iou_target, det_iou_on_true, area_ratio):")
        for r in wrong[:12]:
            print(f"  {r['i']:3d} {r['state']:>14s} iou={r['iou_target']:.3f} "
                  f"det_iou={r['det_iou_on_true_target']:.3f} area={r['area_ratio']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


