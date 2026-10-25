"""Does the part-aware appearance mechanism actually earn its complexity?

The project's central claim for requirement (4) — "the selected target must still
be tracked when only PART of them is visible" — rests on one mechanism: matching
a *part* descriptor (upper body / head) instead of the whole-body descriptor when
the rest of the person is occluded.  That claim is testable without any tracking,
any ground truth and any training, as an **identity-retrieval** experiment.

Protocol
--------
1. Run the detector + all-person MOT layer over a source, collecting person crops
   labelled by their all-person track id (so identities are free — no GT needed).
   Identities with too few samples are dropped; crops must be tall enough for the
   part split to be meaningful.
2. First half of each identity's samples -> GALLERY, second half -> QUERY pool.
   This is the one-shot / long-term ReID setting the project operates in.
3. For every query crop and every occlusion mask (lower half painted out, upper
   half, left, right, ...) re-encode the crop and rank all gallery identities.
4. Compare matching STRATEGIES:
     full          whole-body descriptor only (the naive baseline)
     mean_parts    mean cosine over all parts
     max_part      best single part
     quality_w     cosine weighted by each region's measured image activity
     visible_gate  mean over only the parts that still show real image structure
   `quality_w` / `visible_gate` are the honest versions of the mechanism: the
   tracker is NOT told which mask was applied, it estimates which regions still
   contain structure from the image alone (Laplacian energy), exactly as the real
   tracker does via `parts.crop_quality`.

Reported per (mask, strategy): Rank-1 / Rank-5 identity accuracy and the mean
margin between the correct identity and the best wrong one.  The margin is what
the tracker's refusal gate consumes, so a strategy that raises margin can raise
accuracy *and* reduce false acquisition at the same time.

Outputs: ``results/partial_view/{summary.json, table.md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tgtrack import Config, ensure_ort_cuda  # noqa: E402
from tgtrack.appearance import Embedder  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.mot_layer import AllPersonTracker  # noqa: E402
from tgtrack.types import PART_X_RANGES, PART_Y_RANGES  # noqa: E402

CROP_W, CROP_H = 128, 256
DEFAULT_PARTS = ("full", "upper", "lower", "head")
MASKS = ("none", "lower", "upper", "lower60", "upper60", "left", "right")


# --------------------------------------------------------------------------- #
# crops, masks, and an honest visibility estimator
# --------------------------------------------------------------------------- #
def crop_person(img: np.ndarray, box, out=(CROP_W, CROP_H), pad: float = 0.06) -> np.ndarray | None:
    """Crop a person with a little context, resized to a fixed canvas."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    x1, x2 = x1 - pad * bw, x2 + pad * bw
    y1, y2 = y1 - pad * bh, y2 + pad * bh
    # Reflect-pad rather than clamp: clamping a box at the frame edge would
    # squeeze the person and corrupt the part geometry we are trying to measure.
    pad_l, pad_t = max(0, int(round(-x1))), max(0, int(round(-y1)))
    pad_r, pad_b = max(0, int(round(x2 - w))), max(0, int(round(y2 - h)))
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    x2i, y2i = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2i - x1i < 2 or y2i - y1i < 2:
        return None
    patch = img[y1i:y2i, x1i:x2i]
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return cv2.resize(patch, out, interpolation=cv2.INTER_LINEAR)


def apply_mask(crop: np.ndarray, kind: str) -> np.ndarray:
    """Paint out a region, imitating a real occluder (arm, desk, another person).

    Painting a flat colour is the hardest case for an appearance descriptor, which
    is the point: if the mechanism survives this it survives a textured occluder.
    """
    if kind == "none":
        return crop
    out = crop.copy()
    h, w = crop.shape[:2]
    if kind == "lower":
        out[h // 2 :, :] = 0
    elif kind == "upper":
        out[: h // 2, :] = 0
    elif kind == "lower60":
        out[int(h * 0.4) :, :] = 0
    elif kind == "upper60":
        out[: int(h * 0.6), :] = 0
    elif kind == "left":
        out[:, : w // 2] = 0
    elif kind == "right":
        out[:, w // 2 :] = 0
    else:
        raise ValueError(f"unknown mask {kind!r}")
    return out


def region_activity(img: np.ndarray, box_xyxy) -> float:
    """Fraction of a region that still contains real image structure, in [0, 1].

    This is the tracker's only honest visibility signal: it needs no knowledge of
    what occluded the person.  A blacked-out or flat region yields ~0, textured
    clothing or skin yields ~0.8-1.0, and a half-covered region yields ~half —
    the measure is monotone in how much of the region is actually visible.

    An earlier version averaged the Laplacian magnitude, which SATURATED: a
    region that was 90% black but had 10% of high-contrast edge still scored
    ~1.0.  Counting structured pixels instead cannot do that.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return float(np.mean(lap > 3.0))


def part_regions(shape) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape[:2]
    out = {}
    for p in DEFAULT_PARTS:
        ya, yb = PART_Y_RANGES[p]
        xa, xb = PART_X_RANGES[p]
        out[p] = (int(xa * w), int(ya * h), int(xb * w), int(yb * h))
    return out


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


# --------------------------------------------------------------------------- #
def encode_crops(embedder: Embedder, crops: list[np.ndarray], parts, per_row: int = 8,
                 gutter: int = 4) -> list[dict]:
    """Encode pre-cropped person images in batches.

    ``Embedder.encode`` takes ONE image plus N boxes, so independent crops are
    packed side by side into a single canvas and passed as N boxes.  Part crops
    are derived from each box, so the mosaic is equivalent to encoding each crop
    alone — and it keeps the real batching the GPU wants.

    Part bits are simply absent when a crop was degenerate, so callers must use
    ``.get()``.
    """
    out: list[dict] = [{} for _ in crops]
    for start in range(0, len(crops), per_row):
        chunk = crops[start : start + per_row]
        if not chunk:
            continue
        hh = max(c.shape[0] for c in chunk)
        ww = sum(c.shape[1] for c in chunk) + gutter * (len(chunk) - 1)
        canvas = np.zeros((hh, ww, 3), dtype=np.uint8)
        boxes, x = [], 0
        for c in chunk:
            ch, cw = c.shape[:2]
            canvas[:ch, x : x + cw] = c
            boxes.append([x, 0, x + cw, ch])
            x += cw + gutter
        res = embedder.encode(canvas, np.asarray(boxes, dtype=np.float32), parts=parts)
        for i, r in enumerate(res):
            out[start + i] = r if r is not None else {}
    return out


def collect_identities(args, cfg, detector, mot) -> dict[int, list[np.ndarray]]:
    from demo_click import open_source  # same directory, keeps the source logic in one place

    gen, _seq = open_source(args.source, args.max_frames, args.subsample)
    crops: dict[int, list[np.ndarray]] = {}
    t0 = time.perf_counter()
    n = 0
    for idx, frame in gen:
        n += 1
        dets = detector.detect(frame)
        mot.update(frame, dets)
        for d in dets:
            if d.track_id is None:
                continue
            if (d.xyxy[3] - d.xyxy[1]) < args.min_box_h:
                continue
            c = crop_person(frame, d.xyxy)
            if c is not None:
                crops.setdefault(int(d.track_id), []).append(c)
        if n % 50 == 0:
            print(f"  ... {n} frames, {len(crops)} candidate identities", flush=True)
    el = time.perf_counter() - t0
    print(f"collected {n} frames in {el:.1f}s ({n/el:.1f} FPS), {len(crops)} raw identities")
    return crops


def build_sets(crops: dict[int, list[np.ndarray]], min_samples: int, gallery_per_id: int):
    """Split each identity's crops into gallery / query halves."""
    gallery: dict[int, list[np.ndarray]] = {}
    queries: dict[int, list[np.ndarray]] = {}
    for tid, samples in crops.items():
        if len(samples) < min_samples:
            continue
        samples = sorted(
            samples, key=lambda c: region_activity(c, (0, 0, c.shape[1], c.shape[0])), reverse=True
        )
        gallery[tid] = samples[:gallery_per_id]
        rest = samples[gallery_per_id:]
        if not rest:
            continue
        queries[tid] = rest[: max(1, len(samples) // 2)]
    return {k: v for k, v in gallery.items() if k in queries}, queries


# --------------------------------------------------------------------------- #
# matching strategies — the actual comparison
# --------------------------------------------------------------------------- #
def make_scorer(name: str, vis_thr: float = 0.25):
    if name == "full":
        def score(q, q_act, gal, parts):
            qf = q.get("full")
            if qf is None:
                return -1.0
            return max((float(np.dot(qf, e["full"])) for e in gal if e.get("full") is not None),
                       default=-1.0)
        return score

    if name == "mean_parts":
        def score(q, q_act, gal, parts):
            vals = []
            for p in parts:
                qp = q.get(p)
                if qp is None:
                    continue
                best = max((float(np.dot(qp, e[p])) for e in gal if e.get(p) is not None),
                           default=None)
                if best is not None:
                    vals.append(best)
            return float(np.mean(vals)) if vals else -1.0
        return score

    if name == "max_part":
        def score(q, q_act, gal, parts):
            vals = []
            for p in parts:
                qp = q.get(p)
                if qp is None:
                    continue
                best = max((float(np.dot(qp, e[p])) for e in gal if e.get(p) is not None),
                           default=None)
                if best is not None:
                    vals.append(best)
            return float(max(vals)) if vals else -1.0
        return score

    if name == "quality_w":
        def score(q, q_act, gal, parts):
            num = den = 0.0
            for p in parts:
                qp = q.get(p)
                if qp is None:
                    continue
                best = [float(np.dot(qp, e[p])) for e in gal if e.get(p) is not None]
                if not best:
                    continue
                w = max(0.0, q_act.get(p, 0.0))
                num += w * max(best)
                den += w
            return float(num / den) if den > 1e-6 else -1.0
        return score

    if name == "visible_gate":
        def score(q, q_act, gal, parts):
            vis = [p for p in parts if q_act.get(p, 0.0) >= vis_thr and q.get(p) is not None]
            if not vis:
                vis = [p for p in parts if q.get(p) is not None]
            vals = []
            for p in vis:
                best = max((float(np.dot(q[p], e[p])) for e in gal if e.get(p) is not None),
                           default=None)
                if best is not None:
                    vals.append(best)
            return float(np.mean(vals)) if vals else -1.0
        return score

    raise ValueError(f"unknown strategy {name!r}")


STRATEGIES = ("full", "mean_parts", "max_part", "quality_w", "visible_gate")


def part_discriminability(embedder, gallery_crops, query_crops, gal_enc, parts) -> dict:
    """Control: does each part carry identity evidence at all?

    For every clean query crop, compare its per-part descriptor against the
    correct identity's gallery entries and against every other identity.  A part
    whose same/different gap is ~0 carries no identity evidence (a degenerate
    embedding) and must not be averaged into a fused score.  Without this control
    a part that returns the same vector for everybody would look harmless in a
    mean-parts score while silently diluting every decision.
    """
    ids = sorted(gallery_crops)
    stats: dict[str, dict[str, list[float]]] = {p: {"same": [], "diff": []} for p in parts}
    for tid in ids:
        for enc in encode_crops(embedder, query_crops[tid], parts):
            for p in parts:
                qp = enc.get(p)
                if qp is None:
                    continue
                same = [float(np.dot(qp, e[p])) for e in gal_enc[tid] if e.get(p) is not None]
                diff = [
                    float(np.dot(qp, e[p]))
                    for gid in ids
                    if gid != tid
                    for e in gal_enc[gid]
                    if e.get(p) is not None
                ]
                if same:
                    stats[p]["same"].append(max(same))
                if diff:
                    stats[p]["diff"].append(max(diff))
    out = {}
    for p, d in stats.items():
        s = float(np.mean(d["same"])) if d["same"] else float("nan")
        f = float(np.mean(d["diff"])) if d["diff"] else float("nan")
        gap = s - f if (s == s and f == f) else float("nan")
        out[p] = {"same_mean": s, "diff_mean": f, "gap": gap, "n": len(d["same"])}
    return out


def evaluate(args, embedder, gallery_crops, query_crops, parts):
    ids = sorted(gallery_crops)
    gal_enc = {tid: encode_crops(embedder, gallery_crops[tid], parts) for tid in ids}
    print(f"gallery: {len(ids)} identities, "
          f"{sum(len(v) for v in gal_enc.values())} entries")

    disc = part_discriminability(embedder, gallery_crops, query_crops, gal_enc, parts)
    print("part discriminability (same-identity minus different-identity cosine):")
    for p in parts:
        d = disc[p]
        flag = "  <-- ZERO EVIDENCE" if d["gap"] == d["gap"] and d["gap"] < 0.05 else ""
        print(f"  {p:6s} same={d['same_mean']:+.4f} diff={d['diff_mean']:+.4f} "
              f"gap={d['gap']:+.4f}{flag}")

    results: dict[str, dict[str, dict]] = {}
    region_activity_log: dict[str, dict] = {}

    for mask in MASKS:
        scorers = {name: make_scorer(name, args.vis_threshold) for name in STRATEGIES}
        ranks: dict[str, list[int]] = {name: [] for name in STRATEGIES}
        margins: dict[str, list[float]] = {name: [] for name in STRATEGIES}
        act_log: dict[str, list[float]] = {p: [] for p in parts}

        for tid in ids:
            for crop in query_crops[tid]:
                masked = apply_mask(crop, mask)
                enc = encode_crops(embedder, [masked], parts)[0]
                regions = part_regions(masked.shape)
                q_act = {p: region_activity(masked, regions[p]) for p in parts}
                for p in parts:
                    act_log[p].append(q_act[p])
                if not enc:
                    continue
                for name, scorer in scorers.items():
                    scored = []
                    for gid in ids:
                        s = scorer(enc, q_act, gal_enc[gid], parts)
                        scored.append((s, gid))
                    scored.sort(key=lambda t: -t[0])
                    order = [gid for _s, gid in scored]
                    rank = order.index(tid) + 1 if tid in order else len(order) + 1
                    ranks[name].append(rank)
                    # margin = correct identity's score minus the best WRONG one
                    correct = next(s for s, g in scored if g == tid)
                    wrong = max((s for s, g in scored if g != tid), default=-1.0)
                    margins[name].append(float(correct - wrong))

        results[mask] = {}
        for name in STRATEGIES:
            r = np.asarray(ranks[name], dtype=float)
            m = np.asarray(margins[name], dtype=float)
            if r.size == 0:
                continue
            results[mask][name] = {
                "n": int(r.size),
                "rank1": float(np.mean(r <= 1)),
                "rank5": float(np.mean(r <= 5)),
                "mrr": float(np.mean(1.0 / r)),
                "margin_mean": float(m.mean()),
                "margin_p10": float(np.percentile(m, 10)),
            }
        region_activity_log[mask] = {p: float(np.mean(v)) if v else 0.0 for p, v in act_log.items()}

    return results, region_activity_log, disc, len(ids)


def render_table(results, region_activity_log, disc, n_ids: int) -> str:
    lines = [
        "# Partial-view appearance experiment",
        "",
        f"Gallery identities: **{n_ids}**.  Mask = the region painted out of the query crop.",
        "The tracker is *not* told which mask was applied; `quality_w`/`visible_gate` estimate",
        "the still-visible regions from image structure alone (Laplacian energy).",
        "",
        "## Control: does each part carry identity evidence?",
        "",
        "Same-identity vs different-identity mean cosine on **clean** crops. A gap near",
        "zero means that part is zero-evidence and must be kept out of any fused score.",
        "",
        "| part | same-id cos | diff-id cos | gap | n |",
        "|---|---|---|---|---|",
    ]
    for p in DEFAULT_PARTS:
        d = disc.get(p, {})
        gap = d.get("gap", float("nan"))
        flag = " **ZERO EVIDENCE**" if gap == gap and gap < 0.05 else ""
        lines.append(
            f"| {p} | {d.get('same_mean', float('nan')):+.4f} | "
            f"{d.get('diff_mean', float('nan')):+.4f} | {gap:+.4f}{flag} | {d.get('n', 0)} |"
        )

    lines += [
        "",
        "## Measured region activity (0 = flat/occluded, 1 = fully structured)",
        "",
        "| mask | " + " | ".join(DEFAULT_PARTS) + " |",
        "|---|" + "---|" * len(DEFAULT_PARTS),
    ]
    for mask in MASKS:
        row = region_activity_log.get(mask, {})
        lines.append(
            f"| {mask} | " + " | ".join(f"{row.get(p, 0.0):.3f}" for p in DEFAULT_PARTS) + " |"
        )

    for metric, title, fmt in (
        ("rank1", "Rank-1 identity accuracy", "{:.3f}"),
        ("margin_mean", "Mean margin (correct minus best wrong)", "{:+.4f}"),
    ):
        lines += ["", f"## {title}", "",
                  "| mask | " + " | ".join(STRATEGIES) + " |",
                  "|---|" + "---|" * len(STRATEGIES)]
        for mask in MASKS:
            row = results.get(mask, {})
            cells = []
            for s in STRATEGIES:
                v = row.get(s, {}).get(metric)
                cells.append("n/a" if v is None else fmt.format(v))
            lines.append(f"| {mask} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video, image dir, or MOT sequence dir")
    ap.add_argument("--out", default="results/partial_view")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--min-box-h", type=float, default=90.0,
                    help="minimum person height in pixels for the part split to be meaningful")
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--gallery-per-id", type=int, default=3)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--vis-threshold", type=float, default=0.25,
                    help="region activity below this counts as not visible")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override the ReID input size (the model was TRAINED at 448; "
                         "the shipped default is 224)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fallback-embedder", action="store_true")
    args = ap.parse_args(argv)

    ensure_ort_cuda()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if args.imgsz:
        cfg = cfg.clone(**{"embedder.imgsz": args.imgsz})
    detector = PersonDetector(cfg.detector)
    detector.warmup()
    mot = AllPersonTracker(cfg.mot, detector=detector)
    embedder = Embedder(cfg.embedder, device="cpu" if args.cpu else "cuda",
                        force_fallback=args.fallback_embedder)
    print(f"embedder: mode={embedder.mode} dim={embedder.dim} "
          f"provider={embedder.provider} imgsz={cfg.embedder.imgsz}")
    if not args.cpu and embedder.provider != "CUDAExecutionProvider":
        print(f"WARNING: expected the CUDA execution provider, got {embedder.provider!r} — "
              f"timings below are CPU timings")

    parts = tuple(DEFAULT_PARTS)
    crops = collect_identities(args, cfg, detector, mot)
    gallery_crops, query_crops = build_sets(crops, args.min_samples, args.gallery_per_id)
    print(f"usable identities: {len(gallery_crops)}  "
          f"(queries: {sum(len(v) for v in query_crops.values())})")
    if len(gallery_crops) < 2:
        print("not enough identities — use a longer or more crowded source")
        return 2

    results, act_log, disc, n_ids = evaluate(args, embedder, gallery_crops, query_crops, parts)

    (out / "summary.json").write_text(
        json.dumps({"n_identities": n_ids, "results": results,
                    "region_activity": act_log, "discriminability": disc}, indent=2),
        encoding="utf-8")
    table = render_table(results, act_log, disc, n_ids)
    (out / "table.md").write_text(table, encoding="utf-8")
    print()
    print(table)

    # Headline: the claim is specifically about the occluded cases.
    occluded = [m for m in MASKS if m != "none"]
    print("=" * 78)
    for s in STRATEGIES:
        base = np.mean([results[m][s]["rank1"] for m in occluded if s in results.get(m, {})])
        full_clean = results.get("none", {}).get("full", {}).get("rank1", float("nan"))
        print(f"{s:14s}  mean Rank-1 over occluded masks = {base:.3f}   "
              f"(full-only on clean crops = {full_clean:.3f})")
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
