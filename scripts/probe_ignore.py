"""How much of our MOT17 false-positive load is really *ignore regions*?

E008. MOT17's public GT carries two kinds of rows: `conf=1` real pedestrians, and `conf=0`
rows marking **distractor / static-person / reflection** regions that a detector may
rightfully fire on but that must not be scored. The MOTChallenge convention (and TrackEval's
implementation) is that a detection matching an ignore region is **neither TP nor FP**.

`tgtrack.data.load_mot_sequence` defaults to `min_conf=1.0`, i.e. it *drops* the ignore rows
entirely, so every detection that lands on a distractor is currently counted as a false
positive. That inflates `CLR_FP`, depresses `CLR_Pr`, and therefore depresses both `DetA` and
HOTA — on a metric whose FP count we measured at 4128 over 7 sequences. This probe measures
the size of that effect before anything is changed:

    FP_saved = boxes with (max IoU over valid GT < 0.5) and (max IoU over ignore regions >= 0.5)

Reported per sequence and pooled, with the implied `DetPr`/`DetA`/HOTA if those boxes were
correctly excused.

    python scripts/probe_ignore.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from stitch_tracks import read_tracks, to_pred_dict  # noqa: E402

from tgtrack.data import read_mot_gt  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks-dir", default="results/mot_stitch_base/tracks")
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.5)
    opts = ap.parse_args()

    files = sorted(Path(opts.tracks_dir).glob("*.txt"))
    tot = {"pred": 0, "tp": 0, "fp": 0, "fp_on_ignore": 0}
    print(f"{'sequence':<28} {'pred':>6} {'FP':>6} {'FP@ignore':>10} {'share':>7}")
    for path in files:
        stem = path.stem
        if "__" not in stem:
            continue
        seq_name = stem.split("__", 1)[1]
        seq_dir = Path(opts.data_root) / seq_name
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.is_file():
            continue
        arr, _ = read_tracks(path)
        if len(arr) == 0:
            continue
        offset = 1 if arr[:, 0].min() >= 1 else 1  # MOT files are 1-based
        pred = to_pred_dict(arr, offset)

        rows = read_mot_gt(gt_path)
        valid: dict[int, list[np.ndarray]] = {}
        ignore: dict[int, list[np.ndarray]] = {}
        for r in rows:
            original = int(r["frame"]) - 1        # gt.txt frame ids are 1-based
            if original % opts.subsample:
                continue
            f = original // opts.subsample        # renumbered 0-based run frame
            if f >= opts.max_frames:
                continue
            (valid if float(r["conf"]) >= 1.0 else ignore).setdefault(f, []).append(
                np.asarray(r["xyxy"], np.float64))

        n_pred = n_tp = n_fp = n_ign = 0
        for f, boxes in pred.items():
            vb = np.asarray(valid.get(f, []), np.float64) if valid.get(f) else np.zeros((0, 4))
            ib = np.asarray(ignore.get(f, []), np.float64) if ignore.get(f) else np.zeros((0, 4))
            for p in boxes:
                n_pred += 1
                pb = p.xyxy[None, :]
                if len(vb) and float(iou_matrix(pb, vb).max()) >= opts.thresh:
                    n_tp += 1
                    continue
                n_fp += 1
                if len(ib) and float(iou_matrix(pb, ib).max()) >= opts.thresh:
                    n_ign += 1
        share = n_ign / max(n_fp, 1)
        print(f"{seq_name:<28} {n_pred:>6} {n_fp:>6} {n_ign:>10} {share:>6.1%}")
        for k, v in (("pred", n_pred), ("tp", n_tp), ("fp", n_fp), ("fp_on_ignore", n_ign)):
            tot[k] += v

    fp2 = tot["fp"] - tot["fp_on_ignore"]
    denom = tot["tp"] + fp2
    print(f"\n{'pooled':<28} {tot['pred']:>6} {tot['fp']:>6} {tot['fp_on_ignore']:>10} "
          f"{tot['fp_on_ignore'] / max(tot['fp'], 1):>6.1%}")
    print(f"\nprecision now      : {tot['tp'] / max(tot['tp'] + tot['fp'], 1):.4f}")
    print(f"precision w/ ignore: {tot['tp'] / max(denom, 1):.4f}  "
          f"({tot['tp'] / max(denom, 1) - tot['tp'] / max(tot['tp'] + tot['fp'], 1):+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
