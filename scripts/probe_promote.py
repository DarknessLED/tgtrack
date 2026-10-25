"""Does temporal promotion convert the class-A false negatives into *precise* recall?

E002 showed that simply lowering `conf` to 0.10 takes 244 false negatives off MOT17 but
adds 197 false positives (marginal precision 0.553), leaving HOTA flat.  E004 showed that
**49.6 %** of the remaining false negatives have a sub-threshold detection overlapping them
in the same frame — so the boxes are there, and the question is whether time can tell the
true ones from the noise.

This script prices three front-ends against ALL GT (the evaluation's semantics, not the
vis>=0.25 probe semantics):

  * ``high``  - detections at conf >= 0.25, i.e. what the shipped pipeline feeds the tracker;
  * ``naive`` - detections at conf >= 0.05, i.e. the E002 threshold change;
  * ``sprt``  - ``high`` (>= 0.25) plus boxes promoted by
                :class:`tgtrack.promote.TemporalPromoter` from the [0.01, 0.25) band.

If `sprt` does not beat `naive` on precision at similar recall, the idea is dead and the
contribution has to go somewhere else.

    python scripts/probe_promote.py               # MOT17-02
    python scripts/probe_promote.py --confirm      # 5 sequences
    python scripts/probe_promote.py --alpha 0.02   # stricter false-alarm bound
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
from tgtrack.promote import PromoteCfg, TemporalPromoter  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

TRAIN = Path("data/raw/mot17/MOT17/train")
N_FRAMES = 30
SUBSAMPLE = 5
DETECTOR = Path("results/weights/yolo26m.pt").resolve()
IMGSZ = 1280
CONF_LO = 0.01
CONF_HIGH = 0.25
CONF_NAIVE = 0.05
IOS = 0.5


def counts(det_boxes: np.ndarray, gt_boxes: np.ndarray) -> tuple[int, int, int]:
    """Greedy 1-1 matching at IoU>=0.5. Returns (tp, fp, fn)."""
    n_det, n_gt = len(det_boxes), len(gt_boxes)
    if n_gt == 0:
        return 0, n_det, 0
    if n_det == 0:
        return 0, 0, n_gt
    iou = iou_matrix(det_boxes, gt_boxes)
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_r: set[int] = set()
    used_c: set[int] = set()
    tp = 0
    for r, c in order:
        if iou[r, c] < IOS:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(int(r))
        used_c.add(int(c))
        tp += 1
    return tp, n_det - tp, n_gt - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall,
            "precision": precision, "f1": f1}


def main() -> int:
    confirm = "--confirm" in sys.argv
    alpha = 0.05
    if "--alpha" in sys.argv:
        alpha = float(sys.argv[sys.argv.index("--alpha") + 1])
    conf_lo = CONF_LO
    if "--conf-lo" in sys.argv:
        conf_lo = float(sys.argv[sys.argv.index("--conf-lo") + 1])
    if confirm:
        seq_dirs = [TRAIN / f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 11, 13)]
    else:
        seq_dirs = [TRAIN / "MOT17-02-FRCNN"]

    rows: list[dict] = []
    pooled = {arm: {"tp": 0, "fp": 0, "fn": 0} for arm in ("high", "naive", "sprt")}
    pooled_prom = {"promoted": 0, "promoted_tp": 0, "spawned": 0}

    for seq_dir in seq_dirs:
        if not seq_dir.exists():
            print(f"{seq_dir.name}: MISSING")
            continue
        seq = load_mot_sequence(seq_dir, subsample=SUBSAMPLE, max_frames=N_FRAMES, min_conf=1.0)
        gts = [np.asarray([g.xyxy for g in seq.gt_at(i)], dtype=np.float32).reshape(-1, 4)
               for i in range(len(seq))]

        cfg = Config().clone(**{
            "detector.weights": str(DETECTOR),
            "detector.imgsz": IMGSZ,
            "detector.conf": CONF_LO,
        })
        det = PersonDetector(cfg.detector)
        det.warmup()

        raw: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(seq)):
            found = det.detect(seq.frame(i))
            raw.append((
                np.asarray([d.xyxy for d in found], dtype=np.float32).reshape(-1, 4),
                np.asarray([d.score for d in found], dtype=np.float32),
            ))

        promoter = TemporalPromoter(PromoteCfg(
            conf_high=CONF_HIGH, conf_lo=conf_lo, alpha=alpha))
        acc = {arm: {"tp": 0, "fp": 0, "fn": 0} for arm in ("high", "naive", "sprt")}
        n_promoted = n_promoted_tp = 0
        for t, (boxes, scores) in enumerate(raw):
            gt = gts[t]
            arms = {
                "high": (boxes[scores >= CONF_HIGH], gt),
                "naive": (boxes[scores >= CONF_NAIVE], gt),
            }
            pb, ps = promoter.step(boxes, scores)
            prom_boxes = promoter.last_promoted
            n_prom = len(prom_boxes)
            n_promoted += n_prom
            if n_prom and len(gt):
                # how many promoted boxes actually landed on a person
                n_promoted_tp += int((iou_matrix(prom_boxes, gt).max(axis=1) >= IOS).sum())
            arms["sprt"] = (pb, gt)
            for arm, (bx, g) in arms.items():
                tp, fp, fn = counts(bx, g)
                acc[arm]["tp"] += tp
                acc[arm]["fp"] += fp
                acc[arm]["fn"] += fn

        row = {"sequence": seq.name, "frames": len(seq), "alpha": alpha, "conf_lo": conf_lo,
               "promoted": n_promoted, "promoted_tp": n_promoted_tp,
               "spawned": promoter.n_spawned_total}
        for arm in ("high", "naive", "sprt"):
            row[arm] = prf(**acc[arm])
            for k in ("tp", "fp", "fn"):
                pooled[arm][k] += acc[arm][k]
        pooled_prom["promoted"] += n_promoted
        pooled_prom["promoted_tp"] += n_promoted_tp
        pooled_prom["spawned"] += promoter.n_spawned_total
        rows.append(row)

        print(f"\n{seq.name}  ({len(seq)} frames, {sum(len(g) for g in gts)} GT)")
        for arm in ("high", "naive", "sprt"):
            m = row[arm]
            print(f"  {arm:<6} recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
                  f"F1 {m['f1']:.3f}   (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
        if n_promoted:
            print(f"  promoted {n_promoted} boxes, {n_promoted_tp} of them on a person "
                  f"({n_promoted_tp / max(n_promoted, 1) * 100:.0f}% precision)")

    out = Path("results/recall_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "promote.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n--- pooled (alpha={alpha}) ---")
    for arm in ("high", "naive", "sprt"):
        m = prf(**pooled[arm])
        print(f"  {arm:<6} recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"F1 {m['f1']:.3f}   (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
    if pooled_prom["promoted"]:
        print(f"  promoted {pooled_prom['promoted']} boxes, "
              f"{pooled_prom['promoted_tp']} on a person "
              f"({pooled_prom['promoted_tp'] / max(pooled_prom['promoted'], 1) * 100:.0f}%)"
              f"  | tracklets spawned {pooled_prom['spawned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
