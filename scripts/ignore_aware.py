"""E009: ignore-region-aware MOT evaluation (MOTChallenge / TrackEval convention).

E008 measured that 33.5 % of our MOT17 "false positives" are actually detections on
`conf=0` **ignore regions** (distractor pedestrians, static persons, reflections), which
MOTChallenge and TrackEval do not score. `tgtrack.data.load_mot_sequence` drops those rows
(`min_conf=1.0`), so our harness counts them as false positives and our tables are not
comparable with the literature.

This script re-evaluates a finished run both ways on the same saved tracks:
  * `raw`   — current harness behaviour (every non-TP box is FP);
  * `ignore`— a detection is excused when it matches no valid GT box at `--iou` **and**
    matches an ignore region at `--iou`. That is exactly what TrackEval's ignore handling
    removes from FP accounting, and it cannot turn a false positive into a true positive.

    python scripts/ignore_aware.py
    python scripts/ignore_aware.py --tracks-dir results/mot_stitch_base/tracks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from stitch_tracks import read_tracks, to_pred_dict  # noqa: E402

from tgtrack.data import read_mot_gt  # noqa: E402
from tgtrack.evaluate import GtBox, PredBox, combine_mot, evaluate_mot  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

KEYS = ("HOTA", "DetA", "AssA", "DetPr", "DetRe", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN")


def _gt_split(gt_path: Path, subsample: int, max_frames: int | None):
    """Return (valid, ignore) GT dicts keyed by *subsampled* 0-based frame index."""
    valid: dict[int, list[GtBox]] = {}
    ignore: dict[int, list[np.ndarray]] = {}
    for r in read_mot_gt(gt_path):
        original = int(r["frame"]) - 1        # gt.txt frame ids are 1-based
        if original % subsample:
            continue
        f = original // subsample             # renumbered 0-based run frame
        if max_frames is not None and f >= max_frames:
            continue
        if float(r["conf"]) >= 1.0:
            valid.setdefault(f, []).append(GtBox(frame=f, track_id=int(r["track_id"]),
                                                 xyxy=np.asarray(r["xyxy"], np.float64)))
        else:
            ignore.setdefault(f, []).append(np.asarray(r["xyxy"], np.float64))
    return valid, ignore


def _excuse_ignore(pred: dict[int, list[PredBox]], ignore: dict[int, list[np.ndarray]],
                   iou_thr: float) -> tuple[dict[int, list[PredBox]], int]:
    """Drop predictions that land on an ignore region (they are not scored at all)."""
    out: dict[int, list[PredBox]] = {}
    dropped = 0
    for f, boxes in pred.items():
        ib = np.asarray(ignore[f], np.float64) if ignore.get(f) else np.zeros((0, 4))
        for p in boxes:
            if len(ib) and float(iou_matrix(p.xyxy[None, :], ib).max()) >= iou_thr:
                dropped += 1
                continue
            out.setdefault(f, []).append(p)
    return out, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks-dir", default="results/mot_stitch_base/tracks")
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="results/mot_stitch_base/ignore_aware.json")
    opts = ap.parse_args()

    files = sorted(Path(opts.tracks_dir).glob("*.txt"))
    if not files:
        print(f"no track files under {opts.tracks_dir}")
        return 2

    raw_res, ign_res, rows = [], [], []
    for path in files:
        stem = path.stem
        if "__" not in stem:
            continue
        seq_name = stem.split("__", 1)[1]
        gt_path = Path(opts.data_root) / seq_name / "gt" / "gt.txt"
        if not gt_path.is_file():
            continue
        arr, _ = read_tracks(path)
        if len(arr) == 0:
            continue
        valid, ignore = _gt_split(gt_path, opts.subsample, opts.max_frames)
        if not valid:
            continue
        pred = to_pred_dict(arr, 1)  # MOT track files are 1-based

        # a prediction that is already a TP must never be excused
        pruned: dict[int, list[PredBox]] = {}
        dropped = 0
        for f, boxes in pred.items():
            vb = np.asarray([g.xyxy for g in valid.get(f, [])], np.float64)
            vb = vb if len(valid.get(f, [])) else np.zeros((0, 4))
            ib = np.asarray(ignore[f], np.float64) if ignore.get(f) else np.zeros((0, 4))
            for p in boxes:
                if ib.shape[0] and float(iou_matrix(p.xyxy[None, :], ib).max()) >= opts.iou:
                    if not (vb.shape[0] and float(iou_matrix(p.xyxy[None, :], vb).max()) >= opts.iou):
                        dropped += 1
                        continue
                pruned.setdefault(f, []).append(p)

        r_raw = evaluate_mot(valid, pred, return_state=True)
        r_ign = evaluate_mot(valid, pruned, return_state=True)
        raw_res.append(r_raw)
        ign_res.append(r_ign)
        row = {"pair": stem, "excused_boxes": dropped,
               "raw": {k: float(r_raw[k]) for k in KEYS},
               "ignore": {k: float(r_ign[k]) for k in KEYS}}
        rows.append(row)
        print(f"{stem:<30} HOTA {r_raw['HOTA']:.4f} -> {r_ign['HOTA']:.4f}  "
              f"DetPr {r_raw['DetPr']:.4f} -> {r_ign['DetPr']:.4f}  "
              f"FP {int(r_raw['CLR_FP'])} -> {int(r_ign['CLR_FP'])}  excused {dropped}")

    if not raw_res:
        return 2
    a, b = combine_mot(raw_res), combine_mot(ign_res)
    print(f"\n--- pooled ({len(raw_res)} pairs) ---")
    print(f"{'metric':<8}{'raw':>10}{'ignore-aware':>14}{'delta':>10}")
    for k in KEYS:
        print(f"{k:<8}{a[k]:>10.4f}{b[k]:>14.4f}{b[k] - a[k]:>+10.4f}")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "opts": vars(opts), "pairs": rows,
        "pooled_raw": {k: float(a[k]) for k in KEYS},
        "pooled_ignore": {k: float(b[k]) for k in KEYS},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
