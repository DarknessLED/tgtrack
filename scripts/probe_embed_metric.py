"""E017: is the ReID embedding metric itself the DanceTrack bottleneck -- and can a
*training-free* post-hoc transform (whitening / power normalization) fix it?

Why this probe exists
---------------------
E013 measured the shipped `yolo26n-reid.onnx` descriptor on DanceTrack and found a
verification AUC of only 0.59-0.67, versus 0.97 on MOT17-02.  That is the stated root
cause of the target layer's false-acquisition rate on DanceTrack (0.49 with `mask=none`),
and *no* gating rule can recover information the descriptor does not carry: tightening the
margin only trades FAR for TSR along the same ROC.

Classical face-verification and image-retrieval practice says a large part of a weak
in-domain verification score is not the embedding but the *metric*: raw cosine on an
embedding whose covariance is badly conditioned collapses the distances and hides the
signal.  Whitening (``C^{-1/2}``), mean subtraction, power normalization and their
combinations are all training-free -- they use only the statistics of the scene being
tracked, which is exactly the regime the project is restricted to (no model fine-tuning).

This script measures, on real detections of ONE sequence:

  * pairwise verification AUC (same-id vs different-id, cosine distance),
  * nearest-neighbour rank-1 (is the closest other crop the same person?), and
  * the *tracking-style* separation that actually drives FAR: for each query, the best
    same-id distance and the best different-id distance, and the fraction of queries
    where the same-id crop is strictly closer.

...for the raw embedding and for several training-free transforms, so the decision
"does the metric need fixing?" is made on numbers, not on intuition.

Usage
-----
    python scripts/probe_embed_metric.py \
        --seq-dir data/raw/dancetrack/val/dancetrack0005 \
        --seq-dir data/raw/mot17/MOT17/train/MOT17-02-FRCNN \
        --max-frames 120 --subsample 2 --out results/embed_metric
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

IOU_MATCH = 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(a random positive scores better than a random negative), ties = 0.5."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    allv = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rp = ranks[: len(scores_pos)].sum()
    return float((rp - len(scores_pos) * (len(scores_pos) + 1) / 2.0) / (len(scores_pos) * len(scores_neg)))


def collect_embeddings(seq_dir: str, max_frames: int, subsample: int, device: str) -> dict:
    from tgtrack.appearance import Embedder
    from tgtrack.config import Config, resolve_weight
    from tgtrack.data import load_mot_sequence
    from tgtrack.detector import PersonDetector

    seq = load_mot_sequence(seq_dir, subsample=subsample, max_frames=max_frames, load_frames=False)
    cfg = Config()
    det = PersonDetector(cfg.detector)
    det.warmup()
    emb = Embedder(cfg.embedder, device=device, warmup=True)

    X: list[np.ndarray] = []
    ids: list[int] = []
    frames: list[int] = []
    t0 = time.perf_counter()
    for i in range(len(seq)):
        if i >= max_frames:
            break
        gt = seq.gt.get(i, [])
        if len(gt) < 2:
            continue
        frame = seq.frame(i)
        dets = det.detect(frame)
        if len(dets) < 2:
            continue
        boxes = np.stack([d.xyxy for d in dets]).astype(np.float32)
        gt_boxes = np.stack([g.xyxy for g in gt]).astype(np.float32)
        # match each detection to the best GT box (>= IOU_MATCH), keeping the GT id
        keep_box, keep_id = [], []
        for k, b in enumerate(boxes):
            best_iou, best_id = 0.0, None
            for g, gb in zip(gt, gt_boxes):
                v = iou(b, gb)
                if v > best_iou:
                    best_iou, best_id = v, g.track_id
            if best_iou >= IOU_MATCH:
                keep_box.append(b)
                keep_id.append(int(best_id))
        if len(keep_box) < 2:
            continue
        feats = emb.encode(frame, np.stack(keep_box), parts=("full",))
        for f, tid in zip(feats, keep_id):
            v = f.get("full")
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v))
            if n <= 1e-6:
                continue
            X.append(v / n)
            ids.append(tid)
            frames.append(i)
    return {
        "sequence": Path(seq_dir).name,
        "X": np.stack(X).astype(np.float32),
        "ids": np.asarray(ids, dtype=np.int64),
        "frames": np.asarray(frames, dtype=np.int64),
        "seconds": time.perf_counter() - t0,
    }


def power_norm(X: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.sign(X) * np.abs(X) ** alpha


def whiten_fit(X: np.ndarray, shrinkage: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Mean + inverse-sqrt covariance with shrinkage toward the identity."""
    mu = X.mean(axis=0, keepdims=True)
    Z = X - mu
    d = Z.shape[1]
    C = (Z.T @ Z) / max(1, Z.shape[0] - 1)
    lam = float(shrinkage)
    C = (1.0 - lam) * C + lam * (np.trace(C) / d) * np.eye(d, dtype=np.float32)
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-8)
    W = (V * (1.0 / np.sqrt(w))) @ V.T
    return mu.astype(np.float32), W.astype(np.float32)


def l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n <= 1e-9] = 1.0
    return X / n


def evaluate_metric(X: np.ndarray, ids: np.ndarray, frames: np.ndarray, protocol: str) -> dict:
    """Pairwise verification + tracking-style separation for one feature matrix."""
    Xn = l2(np.asarray(X, dtype=np.float32))
    S = Xn @ Xn.T  # cosine similarity
    D = 1.0 - S
    n = len(ids)
    iu = np.triu_indices(n, k=1)
    same = (ids[:, None] == ids[None, :])[iu]
    d = D[iu]
    pos, neg = d[same], d[~same]

    # tracking-style: for each query, best same-id distance vs best different-id
    better = 0
    total = 0
    margins = []
    for i in range(n):
        m_same = np.where(ids == ids[i], True, False)
        m_same[i] = False
        if not m_same.any():
            continue
        best_pos = float(D[i][m_same].min())
        m_neg = ids != ids[i]
        if not m_neg.any():
            continue
        best_neg = float(D[i][m_neg].min())
        total += 1
        if best_pos < best_neg:
            better += 1
        margins.append(best_neg - best_pos)

    return {
        "protocol": protocol,
        "n": int(n),
        "n_pairs": int(len(d)),
        "auc": auc(-pos, -neg),
        "same_mean": float(pos.mean()),
        "diff_mean": float(neg.mean()),
        "same_p95": float(np.percentile(pos, 95)),
        "best_vs_best_win": float(better / total) if total else float("nan"),
        "margin_median": float(np.median(margins)) if margins else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", action="append", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/embed_metric")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for sd in args.seq_dir:
        data = collect_embeddings(sd, args.max_frames, args.subsample, args.device)
        X, ids, frames = data["X"], data["ids"], data["frames"]
        name = data["sequence"]
        print(f"\n== {name}: {len(ids)} matched crops from {len(set(frames))} frames "
              f"({len(set(ids.tolist()))} identities), collect {data['seconds']:.1f}s")

        variants: dict[str, np.ndarray] = {}
        variants["raw_l2"] = X
        variants["pnorm_l2"] = power_norm(X, 0.5)
        mu, W = whiten_fit(X, shrinkage=0.10)
        variants["whiten"] = l2((X - mu) @ W)
        variants["pnorm_whiten"] = l2((power_norm(X, 0.5) - power_norm(X, 0.5).mean(0, keepdims=True)) @ W)
        mu2, W2 = whiten_fit(X, shrinkage=0.01)
        variants["whiten_s001"] = l2((X - mu2) @ W2)
        mu3, W3 = whiten_fit(X, shrinkage=0.50)
        variants["whiten_s050"] = l2((X - mu3) @ W3)

        rows = []
        for key, mat in variants.items():
            r = evaluate_metric(mat, ids, frames, key)
            rows.append(r)
            print(f"  {key:16s} AUC={r['auc']:.4f}  win={r['best_vs_best_win']:.3f}  "
                  f"same={r['same_mean']:.3f} diff={r['diff_mean']:.3f}  margin_p50={r['margin_median']:+.4f}")
        report[name] = {"stats": {"n": int(len(ids)), "n_frames": int(len(set(frames))),
                                  "n_ids": int(len(set(ids.tolist()))),
                                  "seconds": data["seconds"]}, "rows": rows}

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# E017 — does the ReID *metric* need fixing? (training-free transforms)\n",
             "Verification on real YOLO26m detections matched to GT (IoU >= 0.5). "
             "`win` = fraction of queries whose best same-id crop is strictly closer than "
             "their best different-id crop (the quantity that actually drives FAR).\n"]
    for name, blk in report.items():
        lines.append(f"## {name}  ({blk['stats']['n']} crops, {blk['stats']['n_ids']} ids, "
                     f"{blk['stats']['n_frames']} frames)\n")
        lines.append("| transform | AUC | win | same mean | diff mean | margin p50 |")
        lines.append("|---|---|---|---|---|---|")
        for r in blk["rows"]:
            lines.append(f"| `{r['protocol']}` | **{r['auc']:.4f}** | {r['best_vs_best_win']:.3f} | "
                         f"{r['same_mean']:.3f} | {r['diff_mean']:.3f} | {r['margin_median']:+.4f} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out / 'report.json'} and {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
