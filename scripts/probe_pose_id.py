"""E013: is skeletal geometry an identity cue that the ReID embedding lacks — especially on
partial views?

Requirement 4 fails in a specific way: when only the torso and head are visible, our ReID
descriptor mistakes the target for other people (upper-mask TSR 0.7527 vs 0.9236 full-body on
MOT17). The plan's H3 asks whether **pose** is an orthogonal cue that breaks those ties.

This probe measures the only thing that matters before any wiring: **discriminability**, with GT
identities as the label source so tracking quality cannot confound it. On real footage we sample

  * SAME-identity pairs (different frames, >= `--min-gap` apart), and
  * DIFFERENT-identity pairs (same frame),

and report ROC-AUC for four channels plus one fusion:

  `app_full`   shipped ReID descriptor on the whole box           (current shipped cue)
  `app_upper`  the same descriptor on the UPPER crop               (the partial-view scenario)
  `pose_full`  torso-normalised skeleton distance, all 17 joints
  `pose_upper` skeleton distance over upper-body joints only
  `fuse`       z-normalised `app_upper + pose_upper`               (the H3 hypothesis)

AUC is reported as P(same-identity distance < different-identity distance), so 0.5 = useless and
1.0 = perfect.

    python scripts/probe_pose_id.py --sequences dancetrack0004 --max-frames 80
    python scripts/probe_pose_id.py --data-root data/raw/mot17/MOT17/train --sequences MOT17-02-FRCNN --max-frames 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from tgtrack.config import Config, resolve_weight  # noqa: E402
from tgtrack.data import load_mot_sequence  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
# 11/12 hips, 13/14 knees, 15/16 ankles.
UPPER = list(range(0, 11))
LOWER = list(range(11, 17))
FULL = list(range(17))


def _mid(a, b):
    return 0.5 * (a + b)


def normalise(kpts: np.ndarray, conf: np.ndarray, min_conf: float = 0.25):
    """Scale/translation-normalise a skeleton around the torso -> (coords (17,2), mask (17,))."""
    mask = conf >= min_conf
    if mask[5] and mask[6] and mask[11] and mask[12]:
        centre = _mid(kpts[11], kpts[12])
        scale = float(np.linalg.norm(_mid(kpts[5], kpts[6]) - centre))
    elif mask[5] and mask[6]:
        centre = _mid(kpts[5], kpts[6])
        scale = float(np.linalg.norm(kpts[5] - kpts[6]))
    else:
        vis = kpts[mask]
        if len(vis) < 2:
            return np.zeros((17, 2)), np.zeros(17, dtype=bool)
        centre = vis.mean(0)
        scale = float(np.linalg.norm(vis.max(0) - vis.min(0)))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros((17, 2)), np.zeros(17, dtype=bool)
    out = (kpts - centre) / scale
    out[~mask] = 0.0
    return out, mask


def pose_distance(a, b, joints) -> float:
    """Masked normalised L2 over joints visible in BOTH skeletons; NaN if < 3 such joints."""
    ca, ma = a
    cb, mb = b
    j = np.asarray(joints)
    m = ma[j] & mb[j]
    if int(m.sum()) < 3:
        return float("nan")
    d = ca[j][m] - cb[j][m]
    return float(np.sqrt((d ** 2).sum(1).mean()))


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(1.0 - float(np.dot(a, b) / (na * nb)))


def auc(same: np.ndarray, diff: np.ndarray) -> float:
    """P(same-pair distance < different-pair distance). 0.5 = chance, 1.0 = perfect."""
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if len(same) == 0 or len(diff) == 0:
        return float("nan")
    allv = np.concatenate([same, diff])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ties
        j = i + 1
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    r_same = ranks[: len(same)].sum()
    p_same_greater = (r_same - len(same) * (len(same) + 1) / 2.0) / (len(same) * len(diff))
    return float(1.0 - p_same_greater)


def collect(root: Path, names, subsample: int, max_frames: int):
    from ultralytics import YOLO
    from tgtrack.appearance import Embedder

    pose_model = YOLO(resolve_weight("yolo26n-pose.pt"))
    embedder = Embedder(Config().embedder, device="cuda", warmup=True)

    records = []
    for name in names:
        seq_dir = root / name
        if not seq_dir.is_dir():
            print(f"missing {seq_dir}")
            continue
        seq = load_mot_sequence(seq_dir, subsample=subsample, max_frames=max_frames)
        try:
            n_used = 0
            for idx in range(seq.num_frames):
                img = seq.frame(idx)
                gts = seq.gt_at(idx)
                if img is None or not gts:
                    continue
                gt = np.asarray([np.asarray(g.xyxy, np.float64).reshape(4) for g in gts])
                pred = pose_model.predict(img, verbose=False, conf=0.25, imgsz=1280, max_det=300)
                if not pred or pred[0].boxes is None or len(pred[0].boxes) == 0:
                    continue
                r = pred[0]
                pb = r.boxes.xyxy.cpu().numpy().astype(np.float64)
                kp = r.keypoints.xy.cpu().numpy().astype(np.float64)
                kc = (r.keypoints.conf.cpu().numpy().astype(np.float64)
                      if r.keypoints.conf is not None else np.ones(kp.shape[:2]))
                m = iou_matrix(pb, gt)
                picked: dict[int, int] = {}  # person index -> gt index
                for gi in range(len(gt)):
                    if m[:, gi].max() < 0.5:
                        continue
                    pi = int(np.argmax(m[:, gi]))
                    if pi in picked:
                        continue
                    picked[pi] = gi
                if len(picked) < 2:
                    continue
                pis = sorted(picked)
                feats = embedder.encode(img, [pb[p] for p in pis], parts=("full", "upper"))
                for k, p in enumerate(pis):
                    f = feats[k] if k < len(feats) else {}
                    if f.get("full") is None or f.get("upper") is None:
                        continue
                    records.append({
                        "seq": name, "frame": idx, "gid": int(gts[picked[p]].track_id),
                        "pose": normalise(kp[p], kc[p]),
                        "app_full": np.asarray(f["full"], np.float32),
                        "app_upper": np.asarray(f["upper"], np.float32),
                    })
                n_used += 1
            print(f"  {name}: {n_used} frames used, {len(records)} records so far")
        finally:
            seq.clear_cache()
    return records


def pairs(records, min_gap: int, max_pairs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    by_frame, by_id = {}, {}
    for rec in records:
        by_frame.setdefault(rec["frame"], []).append(rec)
        by_id.setdefault(rec["gid"], []).append(rec)

    same_d, diff_d = [], []
    ids = [g for g, v in by_id.items() if len(v) >= 2]
    tries = 0
    while len(same_d) < max_pairs and tries < max_pairs * 40 and ids:
        tries += 1
        lst = by_id[int(rng.choice(ids))]
        i, j = rng.choice(len(lst), 2, replace=False)
        if abs(lst[i]["frame"] - lst[j]["frame"]) < min_gap:
            continue
        same_d.append((lst[i], lst[j]))
    frames = [f for f, v in by_frame.items() if len(v) >= 2]
    tries = 0
    while len(diff_d) < max_pairs and tries < max_pairs * 40 and frames:
        tries += 1
        lst = by_frame[int(rng.choice(frames))]
        i, j = rng.choice(len(lst), 2, replace=False)
        if lst[i]["gid"] == lst[j]["gid"]:
            continue
        diff_d.append((lst[i], lst[j]))
    return same_d, diff_d


def channel_channel(pairlist, fn) -> np.ndarray:
    return np.asarray([fn(a, b) for a, b in pairlist], np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/raw/dancetrack/val")
    ap.add_argument("--sequences", default="")
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--min-gap", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--out", default="results/pose_id/pose_id.json")
    opts = ap.parse_args()

    root = Path(opts.data_root)
    names = [s.strip() for s in opts.sequences.split(",") if s.strip()]
    if not names:
        names = sorted(d.name for d in root.iterdir() if d.is_dir())[:2]

    records = collect(root, names, opts.subsample, opts.max_frames)
    if not records:
        print("no records")
        return 1

    report = {}
    for name in sorted({r["seq"] for r in records}):
        sub = [r for r in records if r["seq"] == name]
        same, diff = pairs(sub, opts.min_gap, opts.max_pairs)
        if not same or not diff:
            continue
        ch = {
            "app_full": (channel_channel(same, lambda a, b: cos_dist(a["app_full"], b["app_full"])),
                         channel_channel(diff, lambda a, b: cos_dist(a["app_full"], b["app_full"]))),
            "app_upper": (channel_channel(same, lambda a, b: cos_dist(a["app_upper"], b["app_upper"])),
                          channel_channel(diff, lambda a, b: cos_dist(a["app_upper"], b["app_upper"]))),
            "pose_full": (channel_channel(same, lambda a, b: pose_distance(a["pose"], b["pose"], FULL)),
                          channel_channel(diff, lambda a, b: pose_distance(a["pose"], b["pose"], FULL))),
            "pose_upper": (channel_channel(same, lambda a, b: pose_distance(a["pose"], b["pose"], UPPER)),
                           channel_channel(diff, lambda a, b: pose_distance(a["pose"], b["pose"], UPPER))),
        }
        res = {f"{k}_auc": auc(s, d) for k, (s, d) in ch.items()}
        for k, (s, d) in ch.items():
            res[f"{k}_med"] = [float(np.nanmedian(s)) if np.isfinite(s).any() else None,
                               float(np.nanmedian(d)) if np.isfinite(d).any() else None]

        # ---- fusion: z-normalise each channel on the POOLED same+diff distribution, then add.
        # Normalising same and diff separately would erase the between-population difference and
        # drive every fused AUC to 0.5 (that bug was caught here, in the first run).
        def zpool(v_same, v_diff):
            vs = np.asarray(v_same, np.float64).copy()
            vd = np.asarray(v_diff, np.float64).copy()
            pooled = np.concatenate([vs, vd])
            fin = pooled[np.isfinite(pooled)]
            mu = float(fin.mean()) if len(fin) else 0.0
            sd = float(fin.std()) if len(fin) else 1.0
            sd = sd if sd > 1e-9 else 1.0
            worst = float(fin.max()) if len(fin) else mu + 3.0 * sd
            vs[~np.isfinite(vs)] = worst
            vd[~np.isfinite(vd)] = worst
            return (vs - mu) / sd, (vd - mu) / sd

        def fuse(sa, da, sb, db, w=1.0):
            zs_a, zd_a = zpool(sa, da)
            zs_b, zd_b = zpool(sb, db)
            return auc(zs_a + w * zs_b, zd_a + w * zd_b)

        res["fuse_upper_auc"] = fuse(ch["app_upper"][0], ch["app_upper"][1],
                                     ch["pose_upper"][0], ch["pose_upper"][1])
        res["fuse_full_auc"] = fuse(ch["app_full"][0], ch["app_full"][1],
                                    ch["pose_full"][0], ch["pose_full"][1])
        res["fuse_upper_w_sweep"] = {str(w): fuse(ch["app_upper"][0], ch["app_upper"][1],
                                                 ch["pose_upper"][0], ch["pose_upper"][1], w)
                                     for w in (0.25, 0.5, 1.0, 2.0)}

        report[name] = {"n_records": len(sub), "n_ids": len({r["gid"] for r in sub}),
                        "n_same": len(same), "n_diff": len(diff), **res}
        print(f"\n=== {name}: {len(sub)} records, {report[name]['n_ids']} ids, "
              f"{len(same)} same-pairs, {len(diff)} diff-pairs")
        for k in ("app_full", "app_upper", "pose_full", "pose_upper"):
            print(f"  {k:<11} AUC {res[k + '_auc']:.4f}   median same/diff {res[k + '_med']}")
        print(f"  {'FUSE upper':<11} AUC {res['fuse_upper_auc']:.4f}   (app_upper + pose_upper)")
        print(f"  {'FUSE full':<11} AUC {res['fuse_full_auc']:.4f}   (app_full + pose_full)")
        print(f"  fuse w-sweep {({k: round(v, 4) for k, v in res['fuse_upper_w_sweep'].items()})}")

    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opts": vars(opts), "sequences": report}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
