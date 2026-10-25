"""E014a: is the upper-mask TSR deficit a DETECTOR failure or an ASSOCIATION failure?

`results/target_obs19` shows the target-success rate collapses under the upper/lower half-mask
(TSR 0.8988 none -> 0.7101 upper, loss_rate 0.10 -> 0.29) and — the tell — the collapse is
**identical for seven different association strategies** (baseline, `gate_growth_unfrozen`,
`gate_search_early`, `margin_0.06`, `part_matching_off`, `parts_all` all land on exactly 0.753 for
MOT17-02). A deficit that no cost-function change can move is probably not coming from the cost
function.

This probe prices the other side of the ledger: with the same half-box paint the harness applies,
how often does the detector fire on the still-visible half at all? No tracker, no embedder — just
`PersonDetector` — so it costs seconds and cannot be confounded by association.

If the detection rate under `visible=upper` is far below the no-paint control, the honest
conclusion is that the measured "partial view" deficit is largely "the detector cannot see a
half-painted person", and no association work fixes it.

    python scripts/probe_occluded_detect.py --sequence MOT17-02-FRCNN --target-id 19
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
from tgtrack.data import load_mot_sequence  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402


def visible_subbox(box: np.ndarray, mode: str) -> np.ndarray:
    """The part of `box` that stays visible when the OTHER half is painted over."""
    x1, y1, x2, y2 = [float(v) for v in box]
    if mode == "full":
        return np.asarray(box, np.float64)
    if mode == "upper":          # lower half painted -> top half visible
        return np.asarray([x1, y1, x2, y1 + 0.5 * (y2 - y1)], np.float64)
    if mode == "lower":          # upper half painted -> bottom half visible
        return np.asarray([x1, y1 + 0.5 * (y2 - y1), x2, y2], np.float64)
    raise ValueError(mode)


def hidden_rect(box: np.ndarray, mode: str) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    if mode == "full":
        return None
    if mode == "upper":
        return np.asarray([x1, y1 + 0.5 * (y2 - y1), x2, y2], np.float64)
    return np.asarray([x1, y1, x2, y1 + 0.5 * (y2 - y1)], np.float64)


def paint(frame: np.ndarray, rect: np.ndarray | None, fill: str) -> np.ndarray:
    if rect is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = int(max(0, np.floor(rect[0])))
    y1 = int(max(0, np.floor(rect[1])))
    x2 = int(min(w, np.ceil(rect[2])))
    y2 = int(min(h, np.ceil(rect[3])))
    if x2 <= x1 or y2 <= y1:
        return out
    if fill == "black":
        out[y1:y2, x1:x2] = 0
    else:  # mean colour of the surrounding frame, a less tell-tale occluder
        out[y1:y2, x1:x2] = np.median(out.reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--sequence", default="MOT17-02-FRCNN")
    ap.add_argument("--target-id", type=int, default=19)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--modes", default="full,upper,lower")
    ap.add_argument("--fill", default="black", choices=("black", "mean"))
    ap.add_argument("--out", default="results/recall_probe/occluded_detect.json")
    opts = ap.parse_args()

    seq = load_mot_sequence(Path(opts.data_root) / opts.sequence,
                            subsample=opts.subsample, max_frames=opts.max_frames)
    det = PersonDetector(Config().detector)
    det.warmup()

    modes = [m.strip() for m in opts.modes.split(",") if m.strip()]
    stats = {m: {"frames": 0, "detected": 0, "n_dets": 0, "best_iou_sum": 0.0} for m in modes}
    try:
        for idx in range(seq.num_frames):
            gts = seq.gt_at(idx)
            box = None
            for g in gts:
                if int(g.track_id) == opts.target_id:
                    box = np.asarray(g.xyxy, np.float64).reshape(4)
                    break
            if box is None:
                continue
            raw = seq.frame(idx)
            if raw is None:
                continue
            for mode in modes:
                vis = visible_subbox(box, mode)
                frame = paint(raw, hidden_rect(box, mode), opts.fill)
                ds = det.detect(frame)
                stats[mode]["frames"] += 1
                stats[mode]["n_dets"] += len(ds)
                if ds:
                    m = iou_matrix(np.asarray([d.xyxy for d in ds], np.float64).reshape(-1, 4),
                                   vis.reshape(1, 4))[:, 0]
                    best = float(m.max())
                else:
                    best = 0.0
                stats[mode]["best_iou_sum"] += best
                if best >= 0.5:
                    stats[mode]["detected"] += 1
    finally:
        seq.clear_cache()

    print(f"{opts.sequence}  target {opts.target_id}  fill={opts.fill}")
    print(f"{'mode':<8}{'frames':>7}{'det_rate':>10}{'dets/f':>8}{'mean_best_iou':>15}")
    for mode in modes:
        s = stats[mode]
        n = max(s["frames"], 1)
        print(f"{mode:<8}{s['frames']:>7}{s['detected'] / n:>10.3f}{s['n_dets'] / n:>8.2f}"
              f"{s['best_iou_sum'] / n:>15.3f}")

    base = stats["full"]["detected"] / max(stats["full"]["frames"], 1)
    for mode in modes:
        if mode == "full":
            continue
        r = stats[mode]["detected"] / max(stats[mode]["frames"], 1)
        print(f"  detection-rate drop {mode}: {base - r:+.3f}  ({base:.3f} -> {r:.3f})")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opts": vars(opts), "stats": stats}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
