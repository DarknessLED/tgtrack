"""Convert VisDrone2019-MOT (aerial/UAV) into the MOTChallenge layout our loader reads.

Why this exists: MOT17 and DanceTrack are both level, pedestrian-scale views.  The
reference product for this project is a *drone*, so the dataset set has to contain at
least one looking-down-at-people sequence, and VisDrone is the standard one.  Its
annotations are not MOTChallenge, so they are converted here rather than by teaching
the loader a second format.

VisDrone annotation row (10 comma-separated fields, no header):

    frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
    score, object_category, truncation, occlusion

  * ``score`` 0 marks regions to IGNORE (they are dropped; MOTChallenge expresses the
    same thing as conf=0, which our loader also drops by default).
  * ``object_category`` 1 = pedestrian, 2 = people, 3 = bicycle, 4 = car, ...  Only the
    two person categories are kept, because this project tracks people.
  * ``target_id`` 0 means untracked; those rows cannot be used as ground truth.
  * ``occlusion`` 0..10 is the only visibility signal VisDrone offers, so it is mapped
    to MOTChallenge's ``visibility`` as ``1 - occlusion/10``.  This is an approximation:
    MOT17's visibility is a fraction of the box that is visible, VisDrone's is an
    ordinal 0-10 label.  Visibility-dependent ablations should therefore be run on
    MOT17 and only cross-checked here.

Output (``<out>/<sequence>/``):

    img1/000001.jpg ...   hardlinks, not copies (the images are already on disk)
    gt/gt.txt             MOTChallenge 9-column: frame,id,x,y,w,h,conf,class,visibility
    seqinfo.ini           so ``discover_sequences`` sees a normal sequence

Usage:
    python scripts/convert_visdrone.py \
        --src data/raw/visdrone/VisDrone2019-MOT-val \
        --out data/sequences/visdrone
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PERSON_CATEGORIES = (1, 2)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/raw/visdrone/VisDrone2019-MOT-val")
    p.add_argument("--out", default="data/sequences/visdrone")
    p.add_argument("--fps", type=int, default=30, help="VisDrone ships no frame rate; 30 is the usual reading")
    p.add_argument("--min-id", type=int, default=1, help="drop rows whose target_id is below this (0 = untracked)")
    p.add_argument("--keep-ignored", action="store_true", help="keep score==0 rows as conf=0")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def read_annotations(path: Path, min_id: int, keep_ignored: bool) -> tuple[list[tuple], dict]:
    """Return (rows, stats). Rows are MOTChallenge columns as ints/floats."""
    rows: list[tuple] = []
    stats = {"total": 0, "kept": 0, "drop_score": 0, "drop_category": 0, "drop_id": 0, "drop_malformed": 0}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            parts = line.split(",")
            if len(parts) < 10:
                stats["drop_malformed"] += 1
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = (float(parts[i]) for i in (2, 3, 4, 5))
                score = int(float(parts[6]))
                category = int(float(parts[7]))
                occlusion = float(parts[9])
            except (TypeError, ValueError):
                stats["drop_malformed"] += 1
                continue

            if score == 0 and not keep_ignored:
                stats["drop_score"] += 1
                continue
            if category not in PERSON_CATEGORIES:
                stats["drop_category"] += 1
                continue
            if tid < min_id:
                stats["drop_id"] += 1
                continue
            if w <= 0 or h <= 0:
                stats["drop_malformed"] += 1
                continue

            visibility = float(np.clip(1.0 - occlusion / 10.0, 0.0, 1.0))
            # frame, id, x, y, w, h, conf, class, visibility
            rows.append((frame, tid, x, y, w, h, 1, 1, visibility))
            stats["kept"] += 1
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, stats


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot decode {path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def link_images(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Hardlink every image into ``dst_dir`` under a 6-digit MOTChallenge name.

    Hardlinks rather than copies: the same bytes, one lot of disk.  Falls back to a
    copy on filesystems that refuse links, so this never half-finishes.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    out = []
    for i, src in enumerate(images, start=1):
        dst = dst_dir / f"{i:06d}{src.suffix.lower()}"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        out.append(dst)
    return out


def convert_sequence(seq_dir: Path, ann_path: Path, out_root: Path, fps: int, args) -> dict | None:
    images = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        return None
    rows, stats = read_annotations(ann_path, args.min_id, args.keep_ignored)
    if not rows:
        return None

    out = out_root / seq_dir.name
    linked = link_images(seq_dir, out / "img1")

    # Keep a row only if its frame actually exists: a GT row for a missing frame
    # would silently shift or drop in the loader.
    n_img = len(linked)
    kept = [r for r in rows if 1 <= int(r[0]) <= n_img]
    dropped_oor = len(rows) - len(kept)

    (out / "gt").mkdir(parents=True, exist_ok=True)
    with (out / "gt" / "gt.txt").open("w", encoding="utf-8") as fh:
        for frame, tid, x, y, w, h, conf, cls, vis in kept:
            fh.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},{cls},{vis:.3f}\n")

    width, height = image_size(linked[0])
    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name": seq_dir.name,
        "imDir": "img1",
        "frameRate": str(fps),
        "seqLength": str(n_img),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": linked[0].suffix.lower(),
    }
    with (out / "seqinfo.ini").open("w", encoding="utf-8") as fh:
        ini.write(fh)

    stats.update(frames=n_img, kept=len(kept), dropped_out_of_range=dropped_oor,
                 width=width, height=height, ids=len({int(r[1]) for r in kept}))
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.src)
    seq_root = src / "sequences"
    ann_root = src / "annotations"
    if not seq_root.is_dir() or not ann_root.is_dir():
        print(f"error: {src} does not look like a VisDrone MOT split "
              f"(expected sequences/ and annotations/)", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    total_rows = total_frames = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        ann = ann_root / f"{seq_dir.name}.txt"
        if not ann.exists():
            print(f"  {seq_dir.name}: no annotation, skipped")
            continue
        stats = convert_sequence(seq_dir, ann, out_root, args.fps, args)
        if stats is None:
            print(f"  {seq_dir.name}: nothing usable, skipped")
            continue
        total_rows += stats["kept"]
        total_frames += stats["frames"]
        if not args.quiet:
            print(f"  {seq_dir.name}: {stats['frames']:5d} frames  {stats['kept']:6d} GT rows  "
                  f"{stats['ids']:3d} ids  {stats['width']}x{stats['height']}  "
                  f"(dropped: score {stats['drop_score']}, category {stats['drop_category']}, "
                  f"id {stats['drop_id']}, oob {stats['dropped_out_of_range']})")

    print(f"converted {len(list(out_root.iterdir()))} sequences -> {total_frames} frames, "
          f"{total_rows} GT rows, out={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
