"""E018: on DanceTrack the target is DETECTED on ~100% of frames yet the tracker
follows someone else 80% of the time -- so the failure is in the decision rule.

This probe replays the shipped `TargetTracker` frame by frame (same detector, same
embedder, same config as `scripts/run_target.py` would use with `--mask none`) and, for
every frame where the reported box is wrong while the true target IS detected, dumps the
*cost breakdown* of the competing candidates:

    frame  state  fsu  acc_idx  true_idx  cost(acc)  motion(acc)  app(acc)  cost(true)  ...

That separates the three possible causes:

  a. the true target is not among the candidates at all  -> gate/selection bug;
  b. the true target is a candidate but its appearance is NOT better -> descriptor;
  c. the true target has better appearance but worse total  -> the fusion weights let
     geometry outvote identity.

Usage:
    python scripts/probe_decision.py --seq-dir data/raw/dancetrack/val/dancetrack0005 \
        --target-id 0 --max-frames 200 --subsample 2 --out results/probe_decision
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

IOU_HIT = 0.5


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--target-id", type=int, required=True)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/probe_decision")
    ap.add_argument("--click-frame", type=int, default=0)
    args = ap.parse_args()

    from tgtrack.config import Config
    from tgtrack.data import load_mot_sequence, make_target_gt
    from tgtrack.detector import PersonDetector
    from tgtrack.target_tracker import TargetTracker

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seq = load_mot_sequence(args.seq_dir, subsample=args.subsample, max_frames=args.max_frames,
                            load_frames=False, target_id=args.target_id)
    cfg = Config()
    det = PersonDetector(cfg.detector)
    det.warmup()
    tracker = TargetTracker.load(cfg, device=args.device, with_embedder=True)

    # The target's OWN ground-truth box. This used to be `seq.gt_at(i)[0].xyxy`, i.e.
    # whichever person happened to be first in that frame -- a DIFFERENT person in 200/200
    # frames on dancetrack0004 (target 3) and dancetrack0005 (target 2) and 199/200 on
    # dancetrack0007, because `GtBox.track_id` is the loader's internal dense id, not the MOT
    # id. Every conclusion drawn from that probe was retracted in E062. `make_target_gt` is
    # the authoritative resolver -- the same helper `evaluate_target` uses.
    target_gt = make_target_gt(seq, args.target_id)

    def gt_box(i):
        return target_gt.get(i)

    first = -1
    for i in range(len(seq)):
        if gt_box(i) is not None:
            first = i
            break
    if first < 0:
        print("no GT for that id")
        return 2
    print(f"{seq.name}: target {args.target_id}, first GT frame {first}, "
          f"{len(seq)} frames (subsample {args.subsample})")

    frame0 = seq.frame(first)
    status = tracker.initialize(frame0, gt_box(first), frame_idx=first)
    rows = []
    for i in range(first + 1, len(seq)):
        raw = seq.frame(i)
        dets = list(det.detect(raw))
        status = tracker.step(raw, dets, frame_idx=i)
        gt_xyxy = gt_box(i)
        if gt_xyxy is None:
            continue
        ious = [iou(d.xyxy, gt_xyxy) for d in dets]
        true_idx = int(np.argmax(ious)) if ious else -1
        true_ok = bool(true_idx >= 0 and ious[true_idx] >= IOU_HIT)
        correct = iou(status.xyxy, gt_xyxy) >= IOU_HIT
        bd = tracker._last_breakdown
        # `_last_breakdown` indexes the *encoded* subset of detections, not `dets`;
        # map both ways by object identity so the dump lines up with the GT match.
        enc = list(getattr(tracker, "_encoded_dets", None) or [])
        enc_of_det = {j: k for k, e in enumerate(enc) for j, d in enumerate(dets) if d is e}
        det_of_enc = {k: j for j, k in enc_of_det.items()}
        # TargetTracker exposes no accepted-index attribute; the accepted detection IS
        # the reported box, so recover it by maximum overlap.
        acc_iou = [iou(status.xyxy, d.xyxy) for d in dets] if (not status.frames_since_update and dets) else []
        acc_best = int(np.argmax(acc_iou)) if acc_iou else -1
        accepted = acc_best if acc_best >= 0 and acc_iou[acc_best] >= 0.90 else None
        acc_enc = enc_of_det.get(accepted) if accepted is not None else None
        true_enc = enc_of_det.get(true_idx) if true_ok else None
        rec = {
            "frame": i,
            "state": str(status.state),
            "fsu": int(status.frames_since_update),
            "n_dets": len(dets),
            "correct": correct,
            "true_detected": true_ok,
            "true_idx": true_idx,
            "accepted": accepted,
            "accepted_is_true": bool(accepted is not None and accepted == true_idx),
            "iou_target": round(iou(status.xyxy, gt_xyxy), 4),
        }
        if bd is not None and len(bd) > 0:
            def row(j):
                if j is None or j < 0 or j >= len(bd):
                    return None
                return {
                    "total": round(float(bd.total[j]), 4),
                    "motion": round(float(bd.motion[j]), 4),
                    "app": round(float(bd.appearance[j]), 4),
                    "size": round(float(bd.size[j]), 4),
                    "iou_pred": round(float(bd.iou[j]), 4),
                    "app_valid": bool(bd.appearance_valid[j]),
                }
            rec["acc"] = row(acc_enc)
            rec["true"] = row(true_enc) if true_enc is not None else None
            # if the true target is admissible, rank by appearance alone
            if true_enc is not None and bool(bd.appearance_valid[true_enc]):
                app = np.asarray(bd.appearance, dtype=float)
                valid = np.asarray(bd.appearance_valid, dtype=bool)
                cand = np.where(valid & np.isfinite(np.asarray(bd.total, dtype=float)), app, np.inf)
                rec["app_rank_of_true"] = int(np.sum(cand < cand[true_enc]))
        rows.append(rec)

    wrong = [r for r in rows if not r["correct"]]
    det_but_wrong = [r for r in wrong if r["true_detected"]]
    print(f"\nframes {len(rows)}  wrong {len(wrong)} ({len(wrong)/max(1,len(rows)):.3f})  "
          f"wrong-although-target-detected {len(det_but_wrong)}")
    print(f"accepted_is_true on correct frames: "
          f"{sum(1 for r in rows if r['correct'] and r['accepted_is_true'])}"
          f"/{sum(1 for r in rows if r['correct'])}")
    # cause split
    cause = {"not_detected": 0, "accepted_other": 0, "no_candidate": 0, "no_accept": 0}
    for r in wrong:
        if not r["true_detected"]:
            cause["not_detected"] += 1
        elif r["accepted"] is None:
            cause["no_accept"] += 1
        elif not r["accepted_is_true"]:
            cause["accepted_other"] += 1
    print("cause:", cause)

    print(f"\n{'f':>4} {'st':<13}{'fsu':>3} {'nd':>3} {'acc':>4} {'tru':>4} "
          f"{'c_acc':>6}{'mot':>6}{'app':>6} | {'c_tru':>6}{'mot':>6}{'app':>6} {'appRk':>5}")
    for r in det_but_wrong[:24]:
        a, t = r.get("acc") or {}, r.get("true") or {}
        print(f"{r['frame']:>4} {r['state']:<13}{r['fsu']:>3} {r['n_dets']:>3} "
              f"{str(r['accepted']):>4} {r['true_idx']:>4} "
              f"{a.get('total',float('nan')):>6.3f}{a.get('motion',float('nan')):>6.3f}{a.get('app',float('nan')):>6.3f} | "
              f"{t.get('total',float('nan')):>6.3f}{t.get('motion',float('nan')):>6.3f}{t.get('app',float('nan')):>6.3f} "
              f"{str(r.get('app_rank_of_true')):>5}")

    ranks = [r["app_rank_of_true"] for r in det_but_wrong if "app_rank_of_true" in r]
    if ranks:
        print(f"\nappearance rank of the true target on the {len(ranks)} analysable wrong frames: "
              f"rank0 {sum(1 for x in ranks if x == 0)}  rank1 {sum(1 for x in ranks if x == 1)}  "
              f"rank>=2 {sum(1 for x in ranks if x >= 2)}  median {np.median(ranks):.0f}")
    (out / "trace.json").write_text(json.dumps({"rows": rows, "cause": cause}, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'trace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
