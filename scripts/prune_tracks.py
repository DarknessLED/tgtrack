"""Track-level evidence pruning: are false-positive tracklets poisoning the association?

E006. The offline stitching experiment (E007) showed fragmentation is *not* the problem:
bridging dead tracklets over 30-frame gaps moved pooled HOTA by +0.0005. What is left is the
opposite hypothesis — that the identity layer is being *polluted* rather than *fragmented*.
The evidence points that way: on real MOT17 the all-person layer emits 1803-2431 FP per 400
frames while `IDSW` stays low (104-132). A false-positive tracklet that survives a few frames
gets an identity, collects IDFP, and can steal the association of a real neighbour.

So this tests the mirror image of temporal promotion (E005, which failed): instead of
*admitting* weak detections, **evict whole tracklets whose evidence is weak**. It is a
track-level prior, not a detection-level one, and it needs no GPU time: the per-frame track
files from `eval_mot.py --save-tracks` already hold everything.

A tracklet is dropped when it is short *and* its mean box score is low. Swept over
`(min_len, mean_conf)`; `DetA` and `AssA` are both reported because pruning FPs should raise
DetPr (hence DetA) *and* remove IDFP (hence AssA), whereas pruning real fragments would lower
recall and show up immediately as a DetA drop.

    python scripts/prune_tracks.py                     # default grid
    python scripts/prune_tracks.py --tracks-dir results/mot_stitch_base/tracks
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

from stitch_tracks import gt_dict_for, read_tracks, to_pred_dict  # noqa: E402

from tgtrack.evaluate import PredBox, combine_mot, evaluate_mot  # noqa: E402

# (min_len, mean_conf) — a tracklet survives only if it is long enough OR confident enough.
GRID: list[tuple[int, float]] = [
    (0, 0.0),          # control: no pruning
    (1, 0.50),
    (2, 0.35),
    (2, 0.30),
    (3, 0.30),
    (3, 0.25),
    (5, 0.25),
    (5, 0.20),
    (10, 0.15),
]


def tracklet_stats(pred: dict[int, list[PredBox]]) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for frame in sorted(pred):
        for p in pred[frame]:
            s = stats.setdefault(p.track_id, {"n": 0, "sum": 0.0, "max": 0.0, "frames": 0})
            s["n"] += 1
            s["sum"] += float(p.score)
            s["max"] = max(s["max"], float(p.score))
            s["frames"] = max(s["frames"], frame)
    for s in stats.values():
        s["mean"] = s["sum"] / max(s["n"], 1)
    return stats


def prune(pred: dict[int, list[PredBox]], min_len: int, mean_thr: float) -> tuple[dict, dict]:
    stats = tracklet_stats(pred)
    drop = {t for t, s in stats.items() if s["n"] < min_len and s["mean"] < mean_thr}
    out: dict[int, list[PredBox]] = {}
    kept = dropped = 0
    for frame, boxes in pred.items():
        for p in boxes:
            if p.track_id in drop:
                dropped += 1
                continue
            kept += 1
            out.setdefault(frame, []).append(p)
    return out, {"tracklets": len(stats), "dropped_tracklets": len(drop),
                 "dropped_boxes": dropped, "kept_boxes": kept}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks-dir", default="results/mot_stitch_base/tracks")
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", default="results/mot_stitch_base/prune.json")
    opts = ap.parse_args()

    files = sorted(Path(opts.tracks_dir).glob("*.txt"))
    if not files:
        print(f"no track files under {opts.tracks_dir}")
        return 2

    # load once
    loaded = []
    for path in files:
        stem = path.stem
        if "__" not in stem:
            continue
        seq_name = stem.split("__", 1)[1]
        seq_dir = Path(opts.data_root) / seq_name
        if not seq_dir.is_dir():
            continue
        arr, _ = read_tracks(path)
        if len(arr) == 0:
            continue
        gt = gt_dict_for(seq_dir, opts.subsample, opts.max_frames)
        offset = 1 if (arr[:, 0].min() >= 1 and 0 in gt) else 0
        pred = to_pred_dict(arr, offset)
        loaded.append((stem, seq_name, gt, pred))
    if not loaded:
        return 2

    rows: list[dict] = []
    for min_len, mean_thr in GRID:
        results, extra = [], {}
        for stem, seq_name, gt, pred in loaded:
            pruned, stats = prune(pred, min_len, mean_thr)
            r = evaluate_mot(gt, pruned, return_state=True)
            results.append(r)
            extra[stem] = stats
        pooled = combine_mot(results)
        row = {"min_len": min_len, "mean_conf": mean_thr,
               **{k: float(pooled[k]) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN")},
               "dropped_boxes": int(sum(s["dropped_boxes"] for s in extra.values()))}
        rows.append(row)
        print(f"min_len>={min_len:<3} conf<{mean_thr:<5.2f}  HOTA {row['HOTA']:.4f}  DetA {row['DetA']:.4f}  "
              f"AssA {row['AssA']:.4f}  MOTA {row['MOTA']:.4f}  IDF1 {row['IDF1']:.4f}  "
              f"FP {int(row['CLR_FP'])}  FN {int(row['CLR_FN'])}  dropped {row['dropped_boxes']}")

    base = rows[0]
    print(f"\n--- delta vs no-pruning (pooled, {len(loaded)} pairs) ---")
    for r in rows[1:]:
        print(f"min_len>={r['min_len']:<3} conf<{r['mean_conf']:<5.2f}  "
              f"dHOTA {r['HOTA'] - base['HOTA']:+.4f}  dDetA {r['DetA'] - base['DetA']:+.4f}  "
              f"dAssA {r['AssA'] - base['AssA']:+.4f}  dIDF1 {r['IDF1'] - base['IDF1']:+.4f}")

    best = max(rows[1:], key=lambda r: r["HOTA"], default=base)
    print(f"\nbest: min_len>={best['min_len']} conf<{best['mean_conf']}  "
          f"HOTA {best['HOTA']:.4f} ({best['HOTA'] - base['HOTA']:+.4f})")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opts": vars(opts), "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
