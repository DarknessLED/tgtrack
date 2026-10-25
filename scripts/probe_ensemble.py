"""E011: can a multi-model / multi-scale ensemble lift detection recall where single models cannot?

E003 measured `yolo26m@1280` as the best single detector on MOT17-train FRCNN (pooled recall 0.733)
and found that *bigger models at the same 1280* were worse (`yolo26l@1280` 0.707) while `imgsz=1920`
was worse as well. E010 then showed the association layer gives back only 1-3 % of what the detector
finds, so `DetA` — and therefore HOTA — is capped by this one stage.

The single-model sweep therefore does not answer the question this experiment asks: do different
models/scales fail on **different** people? If yes, fusing them (weighted box fusion, no training)
recovers recall that no single operating point can reach. This is standard detection-ensemble
practice and is exactly the "optimise the strategy, not the weights" latitude the project has.

Implementation: WBF-lite. Detections from every member are pooled, clustered greedily by IoU against
the running fused box, and each cluster becomes one box (score-weighted coordinate average) whose
score is the mean of its members' scores times a small cluster-size bonus. Members that contribute
nothing still dilute the score, which is what keeps WBF precision-honest.

    python scripts/probe_ensemble.py --sequences MOT17-02-FRCNN --max-frames 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from tgtrack.config import Config, resolve_weight  # noqa: E402
from tgtrack.data import iter_frames, load_mot_sequence  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

MEMBERS = {
    "n1280": ("yolo26n.pt", 1280),
    "m1280": ("yolo26m.pt", 1280),
    "l1280": ("yolo26l.pt", 1280),
    "l1536": ("yolo26l.pt", 1536),
}


def wbf(boxes_list, scores_list, iou_thr: float = 0.55, size_bonus: float = 0.02):
    """Weighted box fusion across members. Returns (boxes (N,4), scores (N,))."""
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros((0,))
    n_models = len(boxes_list)
    all_b = np.concatenate([np.asarray(b, np.float64).reshape(-1, 4) for b in boxes_list])
    all_s = np.concatenate([np.asarray(s, np.float64).reshape(-1) for s in scores_list])
    if len(all_b) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    order = np.argsort(-all_s)
    clusters: list[dict] = []
    for i in order:
        b, s = all_b[i], float(all_s[i])
        hit = None
        for cl in clusters:
            if float(iou_matrix(b.reshape(1, 4), cl["box"].reshape(1, 4))[0, 0]) >= iou_thr:
                hit = cl
                break
        if hit is None:
            clusters.append({"box": b.copy(), "items": [(b, s, i)]})
            continue
        hit["items"].append((b, s, i))
        w = np.asarray([it[1] for it in hit["items"]], np.float64)
        C = np.stack([it[0] for it in hit["items"]])
        hit["box"] = (C * w[:, None]).sum(0) / max(w.sum(), 1e-9)

    out_b = np.stack([cl["box"] for cl in clusters])
    out_s = np.asarray([
        float(np.mean([it[1] for it in cl["items"]])) + size_bonus * (len(cl["items"]) - 1)
        for cl in clusters
    ], np.float64)
    return out_b, np.clip(out_s, 0.0, 0.999)


def matched_gt(boxes, gt, thr: float = 0.5) -> int:
    if len(gt) == 0 or len(boxes) == 0:
        return 0
    return int((iou_matrix(boxes, gt).max(axis=0) >= thr).sum())


def precision(boxes, gt, thr: float = 0.5) -> int:
    """How many boxes sit on a real GT box (TP count for a precision estimate)."""
    if len(gt) == 0 or len(boxes) == 0:
        return 0
    return int((iou_matrix(boxes, gt).max(axis=1) >= thr).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--sequences", default="MOT17-02-FRCNN")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--out", default="results/recall_probe/ensemble.json")
    opts = ap.parse_args()

    seqs = []
    for name in [s.strip() for s in opts.sequences.split(",") if s.strip()]:
        d = Path(opts.data_root) / name
        if d.is_dir():
            seqs.append(load_mot_sequence(d, subsample=3, max_frames=opts.max_frames))
        else:
            print(f"missing {d}")

    # one detector per distinct (weights, imgsz)
    detectors: dict[str, PersonDetector] = {}
    for key, (w, sz) in MEMBERS.items():
        cfg = Config()
        cfg.detector.weights = resolve_weight(w)
        cfg.detector.imgsz = sz
        cfg.detector.conf = opts.conf
        cfg.detector.max_det = opts.max_det
        detectors[key] = PersonDetector(cfg.detector)

    per_member_boxes: dict[str, list] = {k: [] for k in MEMBERS}
    per_member_scores: dict[str, list] = {k: [] for k in MEMBERS}
    frames_gt: list = []

    for seq in seqs:
        try:
            for _fi, img, gt_boxes in iter_frames(seq, decode=True):
                if img is None:
                    continue
                gt = (np.asarray([np.asarray(g.xyxy, np.float64).reshape(4) for g in gt_boxes], np.float64)
                      if gt_boxes else np.zeros((0, 4)))
                frames_gt.append(gt)
                for key, det in detectors.items():
                    ds = det.detect(img)
                    b = np.asarray([np.asarray(d.xyxy, np.float64).reshape(4) for d in ds], np.float64) \
                        if ds else np.zeros((0, 4))
                    s = np.asarray([float(d.score) for d in ds], np.float64) if ds else np.zeros((0,))
                    per_member_boxes[key].append(b)
                    per_member_scores[key].append(s)
        finally:
            seq.clear_cache()

    n_frames = len(frames_gt)
    gts = frames_gt
    n_gt = sum(len(g) for g in gts)
    print(f"{n_frames} frames, {n_gt} GT boxes, conf {opts.conf}")

    def score_member(keys) -> dict:
        rec = out = det = 0
        for i, gt in enumerate(gts):
            bl = [per_member_boxes[k][i] for k in keys]
            sl = [per_member_scores[k][i] for k in keys]
            if len(keys) == 1:
                b, s = bl[0], sl[0]
            else:
                b, s = wbf(bl, sl, iou_thr=opts.iou_thr)
            s = np.asarray(s, np.float64); b = np.asarray(b, np.float64)
            keep = s >= opts.conf
            b = b[keep]
            rec += matched_gt(b, gt)
            det += len(b)
            out += precision(b, gt)
        return {"recall": rec / max(n_gt, 1), "det_per_frame": det / max(n_frames, 1),
                "precision": out / max(det, 1)}

    rows = []
    keys_all = list(MEMBERS)
    for r in (1, 2, 3, 4):
        for combo in itertools.combinations(keys_all, r):
            st = score_member(list(combo))
            rows.append({"members": "+".join(combo), **st})
            print(f"  {'+'.join(combo):<22} recall {st['recall']:.4f}  prec {st['precision']:.3f}  "
                  f"dets/f {st['det_per_frame']:6.2f}")

    best = max(rows, key=lambda x: x["recall"])
    print(f"\nBEST by recall: {best['members']}  recall {best['recall']:.4f}  prec {best['precision']:.3f}")
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opts": vars(opts), "n_frames": n_frames, "n_gt": n_gt,
                               "rows": rows, "best": best}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
