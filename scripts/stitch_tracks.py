"""Offline tracklet stitching: can association be repaired after the fact?

The first phase's evidence chain (see `docs/EXPERIMENTS.md`) closes every *detection*
route to a higher HOTA: slicing (E001), native resolution and bigger backbones (E003),
confidence thresholds (E002) and temporal promotion (E005) are all measured dead ends.
What is left is **association**: on real MOT17 `DetA` is flat at 0.44 while `AssA` ranges
0.45-0.53, and `HOTA = sqrt(DetA * AssA)`. Raising `AssA` from 0.53 to 0.69 alone would put
HOTA at 0.55 with the *same* detector — so that is where the remaining headroom is.

The cheapest association experiment is an offline one: take the per-frame track files a
finished run already wrote (`eval_mot.py --save-tracks`), and stitch fragments that are
almost certainly the same person — a track that dies and a track that is born a few frames
later, nearby, at the same scale, with nothing else in the way. It costs no GPU time, so it
is the fastest way to find out whether the remaining `AssA` loss is fragmentation or
something deeper.

Reported per sequence and pooled: HOTA/DetA/AssA/IDF1/IDSW/MOTA before and after stitching.
`DetA` cannot change (the boxes are identical); only identity bookkeeping moves.

    python scripts/stitch_tracks.py                      # default paths, gap 30
    python scripts/stitch_tracks.py --gap 60 --dist 1.2
    python scripts/stitch_tracks.py --tracks-dir results/mot_stitch_base/tracks --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tgtrack.data import load_mot_sequence  # noqa: E402
from tgtrack.evaluate import GtBox, PredBox, combine_mot, evaluate_mot  # noqa: E402


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_tracks(path: Path) -> tuple[np.ndarray, list[int]]:
    """Read a MOT-format track file -> (N,7) array [frame,id,x,y,w,h,conf], col names."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 7:
            continue
        try:
            rows.append([float(v) for v in parts[:7]])
        except ValueError:
            continue
    if not rows:
        return np.zeros((0, 7), np.float32), ["frame", "id", "x", "y", "w", "h", "conf"]
    return np.asarray(rows, dtype=np.float64), ["frame", "id", "x", "y", "w", "h", "conf"]


def to_pred_dict(arr: np.ndarray, frame_offset: int) -> dict[int, list[PredBox]]:
    out: dict[int, list[PredBox]] = {}
    for frame, tid, x, y, w, h, conf in arr:
        f = int(frame) - frame_offset
        out.setdefault(max(f, 0), []).append(
            PredBox(frame=max(f, 0), track_id=int(tid),
                    xyxy=np.asarray([x, y, x + w, y + h], np.float64),
                    score=float(conf) if conf > 0 else 1.0))
    return out


def gt_dict_for(seq_dir: Path, subsample: int, max_frames: int | None) -> dict[int, list[GtBox]]:
    seq = load_mot_sequence(seq_dir, subsample=subsample, max_frames=max_frames, min_conf=1.0)
    out: dict[int, list[GtBox]] = {}
    for i in range(len(seq)):
        out[i] = [GtBox(frame=i, track_id=int(g.track_id), xyxy=g.xyxy,
                        conf=float(getattr(g, "conf", 1.0)),
                        visibility=float(getattr(g, "visibility", 1.0)))
                  for g in seq.gt_at(i)]
    return out


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #
class _Fragment:
    __slots__ = ("tid", "frames", "boxes", "first", "last", "vx", "vy", "h")

    def __init__(self, tid: int) -> None:
        self.tid = tid
        self.frames: list[int] = []
        self.boxes: list[np.ndarray] = []
        self.first = 0
        self.last = 0
        self.vx = 0.0
        self.vy = 0.0
        self.h = 1.0

    def finish(self) -> None:
        order = np.argsort(self.frames)
        self.frames = [self.frames[i] for i in order]
        self.boxes = [self.boxes[i] for i in order]
        self.first, self.last = self.frames[0], self.frames[-1]
        b = self.boxes[-1]
        self.h = max(float(b[3] - b[1]), 1.0)
        if len(self.boxes) >= 2:
            n = min(5, len(self.boxes))
            b0, b1 = self.boxes[-n], self.boxes[-1]
            dt = max(self.frames[-1] - self.frames[-n], 1)
            self.vx = float(((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0 / dt)
            self.vy = float(((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0 / dt)

    def predicted_center(self, frame: int) -> tuple[float, float]:
        b = self.boxes[-1]
        dt = frame - self.last
        cx = (b[0] + b[2]) / 2.0 + self.vx * dt
        cy = (b[1] + b[3]) / 2.0 + self.vy * dt
        return float(cx), float(cy)

    def center(self) -> tuple[float, float]:
        b = self.boxes[-1]
        return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)


def build_fragments(pred: dict[int, list[PredBox]]) -> dict[int, _Fragment]:
    frags: dict[int, _Fragment] = {}
    for frame in sorted(pred):
        for p in pred[frame]:
            f = frags.get(p.track_id)
            if f is None:
                f = frags[p.track_id] = _Fragment(p.track_id)
            f.frames.append(frame)
            f.boxes.append(np.asarray(p.xyxy, np.float64))
    for f in frags.values():
        f.finish()
    return frags


def stitch(pred: dict[int, list[PredBox]], *, gap: int, dist: float, size: float,
           size_ratio: float = 1.6) -> tuple[dict[int, list[PredBox]], dict]:
    """Merge fragments that are almost certainly the same person. Returns (pred2, stats)."""
    frags = build_fragments(pred)
    ids = sorted(frags)
    frames_of: dict[int, set[int]] = {t: set(f.frames) for t, f in frags.items()}
    parent: dict[int, int] = {t: t for t in ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # candidate merges, cheapest (closest/soonest) first
    cands: list[tuple[float, int, int]] = []
    for i, ta in enumerate(ids):
        fa = frags[ta]
        ca = fa.center()
        for tb in ids:
            if tb == ta:
                continue
            fb = frags[tb]
            dt = fb.first - fa.last
            if dt <= 0 or dt > gap:
                continue
            px, py = fa.predicted_center(fb.first)
            cb = fb.center()
            d = float(np.hypot(px - cb[0], py - cb[1]) / max(fa.h, 1.0))
            if d > dist:
                continue
            ratio = max(fb.h, 1.0) / max(fa.h, 1.0)
            if not (1.0 / size_ratio <= ratio <= size_ratio):
                continue
            # someone else already occupies b's start position -> do not bridge it
            occupied = False
            for p in pred.get(fb.first, []):
                if p.track_id in (ta, tb):
                    continue
                other = frags.get(p.track_id)
                if other is None:
                    continue
                oc = other.center()
                if float(np.hypot(oc[0] - cb[0], oc[1] - cb[1]) / max(fb.h, 1.0)) < 0.5:
                    occupied = True
                    break
            if occupied:
                continue
            cands.append((d + 0.02 * dt, ta, tb))

    merged = 0
    for _, ta, tb in sorted(cands):
        ra, rb = find(ta), find(tb)
        if ra == rb:
            continue
        if frames_of[ra] & frames_of[rb]:      # would create a duplicate id in one frame
            continue
        # keep the older id (longer history wins, which is also more stable to read)
        parent[rb] = ra
        frames_of[ra] |= frames_of[rb]
        frames_of.pop(rb, None)
        merged += 1

    out: dict[int, list[PredBox]] = {}
    for frame, boxes in pred.items():
        for p in boxes:
            rid = find(p.track_id)
            out.setdefault(frame, []).append(
                PredBox(frame=p.frame, track_id=rid, xyxy=p.xyxy, score=p.score))
    stats = {"fragments_before": len(ids), "ids_after": len(set(find(t) for t in ids)),
             "merges": merged, "candidates": len(cands)}
    return out, stats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks-dir", default="results/mot_stitch_base/tracks")
    ap.add_argument("--data-root", default="data/raw/mot17/MOT17/train")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--gap", type=int, default=30, help="max frames a fragment may be dead")
    ap.add_argument("--dist", type=float, default=1.0, help="max miss distance, in box heights")
    ap.add_argument("--size", type=float, default=1.6, help="tolerated height ratio")
    ap.add_argument("--out", default="results/mot_stitch_base/stitch.json")
    ap.add_argument("--dry-run", action="store_true")
    opts = ap.parse_args()

    tracks_dir = Path(opts.tracks_dir)
    files = sorted(tracks_dir.glob("*.txt"))
    if not files:
        print(f"no track files under {tracks_dir}")
        return 2

    base_res, stitched_res = [], []
    rows: list[dict] = []
    for path in files:
        stem = path.stem
        if "__" not in stem:
            print(f"{stem}: cannot parse '<tracker>__<sequence>'")
            continue
        tracker, seq_name = stem.split("__", 1)
        seq_dir = Path(opts.data_root) / seq_name
        if not seq_dir.is_dir():
            print(f"{stem}: missing sequence {seq_dir}")
            continue

        arr, _ = read_tracks(path)
        if len(arr) == 0:
            continue
        gt = gt_dict_for(seq_dir, opts.subsample, opts.max_frames)
        # frame numbering: MOT files are 1-based; the loader renumbers to 0-based
        offset = 1 if (arr[:, 0].min() >= 1 and 0 in gt) else 0
        pred = to_pred_dict(arr, offset)

        r0 = evaluate_mot(gt, pred, return_state=True)
        stitched, stats = stitch(pred, gap=opts.gap, dist=opts.dist, size=opts.size)
        r1 = evaluate_mot(gt, stitched, return_state=True)

        base_res.append(r0)
        stitched_res.append(r1)
        row = {
            "pair": stem, "tracker": tracker, "sequence": seq_name, **stats,
            "base": {k: r0[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")},
            "stitched": {k: r1[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")},
        }
        for k in ("HOTA", "AssA", "IDF1"):
            row[f"d_{k}"] = row["stitched"][k] - row["base"][k]
        rows.append(row)
        print(f"{stem:<44} HOTA {r0['HOTA']:.4f} -> {r1['HOTA']:.4f} "
              f"({row['d_HOTA']:+.4f})  AssA {r0['AssA']:.4f} -> {r1['AssA']:.4f}  "
              f"IDSW {int(r0['IDSW'])} -> {int(r1['IDSW'])}  "
              f"merges {stats['merges']}/{stats['fragments_before']}")

    if not base_res:
        return 2
    b = combine_mot(base_res)
    s = combine_mot(stitched_res)
    print(f"\n--- pooled ({len(base_res)} pairs, gap {opts.gap}, dist {opts.dist}) ---")
    for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"):
        print(f"  {k:<6} {b[k]:9.4f} -> {s[k]:9.4f}   ({s[k] - b[k]:+.4f})")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "opts": vars(opts), "pairs": rows,
        "pooled_base": {k: float(b[k]) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")},
        "pooled_stitched": {k: float(s[k]) for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")},
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
