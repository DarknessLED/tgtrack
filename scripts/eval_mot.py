#!/usr/bin/env python3
"""Parallel MOT evaluation harness for the all-person tracking layer.

Runs every ``(sequence, tracker)`` pair through the real pipeline
``PersonDetector`` -> optional part-aware ``Embedder`` -> ``AllPersonTracker``,
scores the accumulating track boxes against the MOT ground truth with
:func:`tgtrack.evaluate.evaluate_mot` (a TrackEval port: HOTA/DetA/AssA/LocA,
CLEAR-MOT, Identity) and writes per-pair JSON plus combined CSV/Markdown tables
sorted by HOTA.

Usage
-----
    # plan only, needs no data and imports no torch:
    python scripts/eval_mot.py --data-root data/sequences --dry-run

    # full sweep over the six ultralytics backends, 4 worker processes:
    python scripts/eval_mot.py --data-root data/sequences \
        --trackers botsort.yaml,ocsort.yaml,deepocsort.yaml,fasttrack.yaml,tracktrack.yaml,bytetrack.yaml \
        --out results/mot_eval --jobs 4

    # with real ReID features (512-d ONNX, CUDA) fed into the ReID-capable backends:
    python scripts/eval_mot.py --data-root data/sequences --embed --jobs 4

    # how the sweep scales (re-runs the same plan at several --jobs values):
    python scripts/eval_mot.py --data-root data/sequences --max-frames 200 \
        --jobs-sweep 1,2,4,6 --out results/mot_eval_scaling

Design notes
------------
* **Pair independence.** Every pair gets a *fresh* ``PersonDetector`` and
  ``AllPersonTracker`` (the tracker is also ``reset()`` immediately after
  construction and again at teardown), so no Kalman/ID state and no cached
  appearance gallery can leak from one sequence into the next, and no GPU
  allocation survives a pair: teardown drops the references, runs
  ``gc.collect()`` and (on CUDA) ``torch.cuda.empty_cache()``.
* **Parallelism.** Pairs are embarrassingly parallel and are dispatched to a
  spawn-based ``ProcessPoolExecutor`` capped at ``--jobs``. Setup cost (weight
  load, ONNX session, tracker construction) is timed separately from the frame
  loop, so ``ms/frame`` is a pure pipeline number. The observed speedup
  ``sum(pair wall) / wall elapsed`` is written to ``run.json`` and to
  ``summary.md`` -- that is the measured scaling evidence, not an estimate.
  The ReID ``Embedder`` is cached **per worker process** (one ONNX session per
  process) because it is stateless; the detector and tracker are not.
* **Memory.** ``Sequence`` caches every decoded frame, so a 1000-frame 1080p
  MOT sequence would pin several GB per worker. The loop drops the decode cache
  every ``--cache-flush`` frames (default 50) to bound RSS.
* **Failure isolation.** A pair that raises is recorded as ``ok: false`` with
  its traceback and is excluded from the aggregate instead of killing the run.

This script imports ``tgtrack`` (light: numpy only). ``torch``/``ultralytics``
are imported lazily inside the detector/embedder factories, which is why
``--dry-run`` works with no weights, no datasets and no GPU.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tgtrack  # noqa: E402  (light import: no torch / ultralytics)
from tgtrack.config import Config, EvalCfg, load_config  # noqa: E402
from tgtrack.data import (  # noqa: E402
    discover_sequences,
    iter_frames,
    load_mot_sequence,
    read_mot_gt,
    sequence_summary,
)
from tgtrack.evaluate import PredBox, combine_mot, evaluate_mot  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402

#: The six backends shipped as ``ultralytics/cfg/trackers/*.yaml``.
DEFAULT_TRACKERS: tuple[str, ...] = (
    "botsort.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
    "fasttrack.yaml",
    "tracktrack.yaml",
    "bytetrack.yaml",
)

#: Backends that consume appearance features (mirrors ``tgtrack.mot_layer.REID_CAPABLE``).
REID_CAPABLE: tuple[str, ...] = ("botsort", "deepocsort", "tracktrack")

#: Metrics carried into the tables, in report order.
METRIC_KEYS: tuple[str, ...] = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "FP",
    "FN",
    "IDSW",
    "Frag",
    "MT",
    "ML",
)

PAIR_COLUMNS: tuple[str, ...] = (
    "tracker",
    "sequence",
    *METRIC_KEYS,
    "num_frames",
    "num_gt_dets",
    "num_pred_dets",
    "num_pred_ids",
    "ms_per_frame",
    "det_ms",
    "emb_ms",
    "track_ms",
    "pipeline_fps",
    "wall_s",
    "setup_ms",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "tracker",
    "num_sequences",
    "num_frames",
    *METRIC_KEYS,
    "num_gt_dets",
    "num_pred_dets",
    "num_pred_ids",
    "ms_per_frame",
    "pipeline_fps",
    "wall_s",
)

_EMBEDDER_CACHE: dict[tuple[Any, ...], Any] = {}


# --------------------------------------------------------------------------- #
# Plan / options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairSpec:
    """One unit of work: score ``tracker`` on ``seq_dir``."""

    sequence: str
    seq_dir: str
    tracker: str

    @property
    def pair_id(self) -> str:
        return f"{safe_name(Path(self.tracker).stem)}__{safe_name(self.sequence)}"

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "seq_dir": self.seq_dir, "tracker": self.tracker, "pair_id": self.pair_id}


@dataclass
class RunOptions:
    """Picklable run options (sent to each worker process)."""

    out_dir: str = "results/mot_eval"
    subsample: int = 1
    max_frames: int | None = None
    min_conf: float = 1.0
    min_visible: float = 0.0
    config: str | None = None
    weights: str | None = None
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    max_det: int | None = None
    device: str | None = None
    half: bool = True
    embed: bool = False
    embed_model: str | None = None
    embed_fallback: bool = False
    with_reid: bool = False
    cache_flush: int = 50
    save_tracks: bool = False
    ignore_regions: bool = False
    iou_thresh: float = 0.5
    name: str = "mot_eval"
    layout: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_name(text: str) -> str:
    """Filesystem-safe token for pair ids (keeps results portable)."""
    out = []
    for ch in str(text):
        out.append(ch if (ch.isalnum() or ch in "-_.") else "_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unnamed"


def parse_trackers(text: str | None) -> list[str]:
    """``"botsort.yaml, ocsort"`` -> ``["botsort.yaml", "ocsort.yaml"]``.

    Blank/``None`` means the six defaults. A bare stem is accepted and gets the
    ``.yaml`` suffix (``resolve_tracker_yaml`` accepts either, but the tables and
    pair ids should be uniform).
    """
    raw = (text or "").strip()
    if not raw:
        return list(DEFAULT_TRACKERS)
    out: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        name = chunk.strip()
        if not name:
            continue
        if not name.lower().endswith((".yaml", ".yml")):
            name = f"{name}.yaml"
        if name not in out:
            out.append(name)
    return out or list(DEFAULT_TRACKERS)


def enumerate_pairs(seq_dirs: Sequence[str], trackers: Sequence[str]) -> list[PairSpec]:
    """Cartesian product of sequences x trackers, in caller order (deterministic)."""
    specs: list[PairSpec] = []
    for seq_dir in seq_dirs:
        name = Path(str(seq_dir)).name
        for tracker in trackers:
            specs.append(PairSpec(sequence=name, seq_dir=str(seq_dir), tracker=str(tracker)))
    return specs


def clamp_jobs(jobs: int, n_pairs: int) -> int:
    """Cap workers by the available work and by the machine's CPU count."""
    if n_pairs <= 0:
        return 0
    cpu = os.cpu_count() or 1
    want = int(jobs) if int(jobs) > 0 else cpu
    return max(1, min(want, n_pairs, cpu))


def discover_in(data_root: str | os.PathLike[str], layout: str = "auto") -> tuple[list[str], list[str]]:
    """Sorted sequence directories under ``data_root`` plus non-fatal warnings."""
    root = Path(data_root)
    if not root.exists():
        return [], [f"data root does not exist: {root}"]
    try:
        found = [str(p) for p in discover_sequences(str(root), layout=layout)]
    except Exception as exc:  # a malformed layout must not kill a dry run
        return [], [f"discover_sequences failed ({type(exc).__name__}: {exc})"]
    if not found:
        return [], [f"no MOT sequences found under {root} (looked for img1/ and gt/gt.txt)"]
    return sorted(found), []


def filter_sequences(
    seq_dirs: Sequence[str],
    *,
    only: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Apply ``--sequences`` substring filter and ``--limit-sequences``."""
    out = list(seq_dirs)
    if only:
        needles = [n.strip().lower() for n in str(only).replace(";", ",").split(",") if n.strip()]
        if needles:
            out = [d for d in out if any(n in Path(d).name.lower() for n in needles)]
    if limit is not None and int(limit) >= 0:
        out = out[: int(limit)]
    return out


# --------------------------------------------------------------------------- #
# Config / resource factories  (torch-free until actually called)
# --------------------------------------------------------------------------- #
def build_config(opts: RunOptions, tracker: str) -> Config:
    """Base config + CLI overrides, with ``mot.tracker`` set to this pair's backend."""
    overrides: dict[str, Any] = {"name": opts.name}
    if opts.weights:
        overrides["detector.weights"] = opts.weights
    if opts.imgsz:
        overrides["detector.imgsz"] = int(opts.imgsz)
    if opts.conf is not None:
        overrides["detector.conf"] = float(opts.conf)
    if opts.iou is not None:
        overrides["detector.iou"] = float(opts.iou)
    if opts.max_det:
        overrides["detector.max_det"] = int(opts.max_det)
    if opts.device:
        overrides["detector.device"] = str(opts.device)
    if opts.half is not None:
        overrides["detector.half"] = bool(opts.half)
    if opts.embed_model:
        overrides["embedder.model"] = str(opts.embed_model)
    cfg = load_config(opts.config, **overrides)
    cfg.mot.tracker = str(tracker)
    # Appearance is only consumed by REID_CAPABLE backends; enabling with_reid for
    # them is what makes the feats argument matter at all (see mot_layer.py:276).
    cfg.mot.with_reid = bool(opts.with_reid or opts.embed)
    return cfg


def make_detector(cfg: Config) -> Any:
    """Fresh ``PersonDetector`` (imports torch/ultralytics on first use)."""
    from tgtrack.detector import PersonDetector

    return PersonDetector(cfg.detector)


def make_tracker(cfg: Config, detector: Any) -> Any:
    """Fresh tracker bound to ``detector`` (hybridsort -> ``HybridSortTracker``).

    ``hybridsort.yaml`` is our own 7th backend (E012, ``tgtrack.mot_hybrid``); every other
    stem goes to ``AllPersonTracker``, which dispatches on the six ultralytics yamls.
    """
    stem = Path(str(cfg.mot.tracker)).stem
    if stem == "hybridsort":
        from tgtrack.mot_hybrid import HybridSortTracker

        return HybridSortTracker(cfg.mot, detector=detector)
    from tgtrack.mot_layer import AllPersonTracker

    return AllPersonTracker(cfg.mot, detector=detector)


def make_embedder(cfg: Config, opts: RunOptions) -> Any:
    """Real part-aware ``Embedder``; ``ensure_ort_cuda()`` first so ONNX really uses the GPU."""
    tgtrack.ensure_ort_cuda()  # must precede the first onnxruntime.InferenceSession
    from tgtrack.appearance import Embedder

    return Embedder(
        cfg.embedder,
        device=str(cfg.detector.device or "cuda"),
        force_fallback=bool(opts.embed_fallback),
        warmup=True,
    )


def get_embedder(cfg: Config, opts: RunOptions) -> Any:
    """Per-process cached embedder (stateless: safe to share across pairs)."""
    key = (
        str(getattr(cfg.embedder, "model", "")),
        str(cfg.detector.device),
        bool(getattr(cfg.embedder, "fp16", False)),
        bool(opts.embed_fallback),
    )
    emb = _EMBEDDER_CACHE.get(key)
    if emb is None:
        emb = make_embedder(cfg, opts)
        _EMBEDDER_CACHE[key] = emb
    return emb


def feats_from_embeddings(embs: Sequence[dict[str, np.ndarray]], dim: int | None, n: int) -> tuple[np.ndarray | None, int]:
    """Pack per-box part dicts into the ``(N, D)`` array ``AllPersonTracker.update`` wants.

    ``Embedder.encode`` omits a part when its crop was degenerate, so a row can be
    missing. The tracker's feature hook requires **every** row to be present and
    finite (``mot_layer._resolve_feats``) and silently drops the whole array
    otherwise, which would turn an embedding ablation into a no-op. Missing rows
    are therefore filled with the mean of the valid ones (L2-normalized), and the
    count is reported so the run can be audited.

    Returns ``(array_or_None, n_missing)``.
    """
    if dim is None or n <= 0:
        return None, n
    rows: list[np.ndarray | None] = []
    for i in range(n):
        vec = None
        if i < len(embs):
            vec = embs[i].get("full")
        arr = None if vec is None else np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr is not None and (arr.size != int(dim) or not np.isfinite(arr).all()):
            arr = None
        rows.append(arr)
    missing = sum(1 for r in rows if r is None)
    if missing == len(rows):
        return None, missing
    valid = np.stack([r for r in rows if r is not None])
    fill = valid.mean(axis=0)
    norm = float(np.linalg.norm(fill))
    fill = fill / norm if norm > 0 else fill
    out = np.stack([r if r is not None else fill for r in rows]).astype(np.float32)
    return out, missing


# --------------------------------------------------------------------------- #
# Core: one (sequence, tracker) pair
# --------------------------------------------------------------------------- #
def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """mean/p50/p95/min/max of a latency series (ms)."""
    v = sorted(float(x) for x in values)
    if not v:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "n": len(v),
        "mean": sum(v) / len(v),
        "p50": v[len(v) // 2],
        "p95": v[min(len(v) - 1, round(0.95 * (len(v) - 1)))],
        "min": v[0],
        "max": v[-1],
        "sum": sum(v),
    }


def pred_boxes_from_dets(frame_idx: int, dets: Sequence[Any]) -> list[PredBox]:
    """Convert the tracker's per-frame ``Detection`` list into evaluable ``PredBox``es.

    Detections the tracker did not activate (``track_id is None``, e.g. second-stage
    low-score misses) contribute no box: MOT metrics must see only boxes with an id.
    """
    out: list[PredBox] = []
    for det in dets:
        tid = getattr(det, "track_id", None)
        if tid is None:
            continue
        out.append(
            PredBox(
                frame=int(frame_idx),
                track_id=int(tid),
                xyxy=np.asarray(det.xyxy, dtype=np.float64).reshape(4),
                score=float(getattr(det, "score", 1.0) or 0.0),
            )
        )
    return out


def write_mot_tracks(path: str | os.PathLike[str], pred: dict[int, list[PredBox]]) -> Path:
    """Dump predictions in MOTChallenge format (1-based frame, ``-1`` marker columns)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for frame in sorted(pred):
            for box in pred[frame]:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy)
                fh.write(
                    f"{frame + 1},{box.track_id},{x0:.2f},{y0:.2f},{x1 - x0:.2f},{y1 - y0:.2f},"
                    f"{float(box.score):.4f},-1,-1,-1\n"
                )
    return p


def load_ignore_boxes(
    gt_path: str | os.PathLike[str],
    subsample: int = 1,
    max_frames: int | None = None,
) -> dict[int, np.ndarray]:
    """MOTChallenge ``conf=0`` rows, keyed by this run's *renumbered* 0-based frame index.

    MOT17's public GT marks distractor pedestrians, static persons and reflections with
    ``conf=0``. MOTChallenge and TrackEval never score a detection that lands on one of
    those regions and matches no valid box. :func:`tgtrack.data.load_mot_sequence` drops
    those rows (``min_conf=1.0``), which is correct for *recall* accounting but makes the
    harness over-count false positives by a measured 33.5 % on MOT17 (E008). This loader
    brings them back so :func:`excuse_ignore_regions` can subtract them.

    Frame numbering is a classic trap here: ``gt.txt`` frame ids are **1-based**, while the
    run's frames are **0-based and renumbered** by ``subsample``. ``load_mot_sequence`` keeps
    original frame ``o`` when ``o % subsample == 0`` and renumbers it to ``o // subsample``;
    the file carries ``o + 1``, so ``index = (file_frame - 1) // subsample``.
    """
    path = Path(gt_path)
    if not path.is_file():
        return {}
    sub = max(int(subsample), 1)
    out: dict[int, list[np.ndarray]] = {}
    for row in read_mot_gt(path):
        if float(row["conf"]) >= 1.0:
            continue
        original = int(row["frame"]) - 1
        if original % sub:
            continue
        index = original // sub
        if max_frames is not None and index >= int(max_frames):
            continue
        out.setdefault(index, []).append(np.asarray(row["xyxy"], np.float64))
    return {f: np.asarray(v, np.float64) for f, v in out.items()}


def excuse_ignore_regions(
    pred: dict[int, list[PredBox]],
    gt: dict[int, list[Any]],
    ignore: dict[int, np.ndarray],
    iou_thresh: float = 0.5,
) -> tuple[dict[int, list[PredBox]], int]:
    """Drop predictions that are scored against nothing (MOTChallenge ignore convention).

    A prediction is excused only when it matches **no valid GT box** at ``iou_thresh`` and
    **does** match an ignore region there. That ordering matters: a box that is a true
    positive can never be removed, so this cannot inflate recall -- it only stops
    distractor detections from being charged as false positives.
    """
    if not ignore:
        return pred, 0
    out: dict[int, list[PredBox]] = {}
    excused = 0
    for frame, boxes in pred.items():
        valid = gt.get(frame) or []
        vb = (np.asarray([np.asarray(b.xyxy, np.float64).reshape(4) for b in valid], np.float64)
              if valid else np.zeros((0, 4)))
        ib = ignore.get(frame, np.zeros((0, 4)))
        for box in boxes:
            pb = np.asarray(box.xyxy, np.float64).reshape(1, 4)
            if ib.shape[0] and float(iou_matrix(pb, ib).max()) >= iou_thresh:
                if not (vb.shape[0] and float(iou_matrix(pb, vb).max()) >= iou_thresh):
                    excused += 1
                    continue
            out.setdefault(frame, []).append(box)
    return out, excused


def _warmup_detector(detector: Any) -> bool:
    """Best-effort one-shot warmup so cold-start cost is not charged to frame 1.

    Returns whether a warmup ran. A detector without ``warmup()`` (test double) or a
    warmup that raises is not an error: the pair still runs, only the timing is noisier.
    """
    fn = getattr(detector, "warmup", None)
    if not callable(fn):
        return False
    try:
        fn()
    except Exception:  # noqa: BLE001 - a warmup failure must never kill the pair
        return False
    return True


def gpu_memory_mb() -> dict[str, float] | None:
    """This process's CUDA allocation in MB, or ``None`` when torch/CUDA is absent.

    Reads ``sys.modules`` rather than importing torch, so a CPU-only or dry run never
    pays the import. Sampling it after every pair (post-teardown) is the evidence that
    ``release_gpu()`` keeps a worker flat instead of leaking VRAM across sequences.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 2**20,
            "reserved_mb": torch.cuda.memory_reserved() / 2**20,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
    except Exception:  # noqa: BLE001 - diagnostics must never fail a pair
        return None


def run_pair(
    spec: PairSpec,
    opts: RunOptions,
    *,
    detector_factory: Callable[[Config], Any] | None = None,
    tracker_factory: Callable[[Config, Any], Any] | None = None,
    embedder_factory: Callable[[Config, RunOptions], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Score one pair. Returns a JSON-serialisable record (``ok: False`` on failure).

    The factories are injectable so the whole loop -- GT accumulation, metric call,
    timing bookkeeping -- is unit-testable without torch (see tests/test_eval_mot.py).
    """
    det_factory = detector_factory or make_detector
    trk_factory = tracker_factory or make_tracker
    emb_factory = embedder_factory or get_embedder

    warnings: list[str] = []
    t_wall0 = clock()
    seq = load_mot_sequence(
        spec.seq_dir,
        subsample=int(opts.subsample),
        max_frames=opts.max_frames,
        min_conf=float(opts.min_conf),
        min_visible=float(opts.min_visible),
    )
    cfg = build_config(opts, spec.tracker)

    t_setup0 = clock()
    detector = det_factory(cfg)
    tracker = trk_factory(cfg, detector)
    tracker.reset()  # fresh object: makes the "no leaked state" contract explicit
    embedder = emb_factory(cfg, opts) if opts.embed else None
    # Warm the detector inside setup: the first CUDA inference pays cuDNN autotune /
    # kernel-JIT (measured ~4.9 s for yolo26n) which would otherwise be charged to
    # frame 1 and wreck the mean ms/frame and pipeline FPS.
    warmed_up = _warmup_detector(detector)
    setup_ms = (clock() - t_setup0) * 1000.0

    gt: dict[int, list[Any]] = {}
    pred: dict[int, list[PredBox]] = {}
    det_ms: list[float] = []
    emb_ms: list[float] = []
    trk_ms: list[float] = []
    frame_ms: list[float] = []
    n_frames = 0
    n_dets = 0
    orphan_dets = 0
    missing_feats = 0

    try:
        for frame_idx, img, gt_boxes in iter_frames(seq, decode=True):
            if img is None:  # GT-only entry (no pixels on disk)
                continue
            gt[int(frame_idx)] = list(gt_boxes)
            n_frames += 1

            t0 = clock()
            dets = detector.detect(img)
            t1 = clock()

            feats = None
            if embedder is not None and dets:
                boxes = np.stack([np.asarray(d.xyxy, dtype=np.float32).reshape(4) for d in dets])
                embs = embedder.encode(img, boxes, parts=tuple(getattr(cfg.embedder, "global_parts", ("full",))))
                feats, missing = feats_from_embeddings(embs, getattr(embedder, "dim", None), len(dets))
                missing_feats += int(missing)
            t2 = clock()

            tracker.update(img, dets, feats=feats)
            t3 = clock()

            boxes = pred_boxes_from_dets(int(frame_idx), dets)
            if boxes:
                pred.setdefault(int(frame_idx), []).extend(boxes)
            n_dets += len(dets)
            orphan_dets += len(dets) - len(boxes)

            det_ms.append((t1 - t0) * 1000.0)
            emb_ms.append((t2 - t1) * 1000.0)
            trk_ms.append((t3 - t2) * 1000.0)
            frame_ms.append((t3 - t0) * 1000.0)

            if opts.cache_flush and (frame_idx + 1) % int(opts.cache_flush) == 0:
                seq.clear_cache()  # bound RSS: Sequence.frame() caches every decoded frame
    finally:
        seq.clear_cache()
        # Reset first (explicit contract), then drop the caller's references too:
        # otherwise torch keeps the CUDA allocations alive for the whole worker
        # lifetime (empty_cache() only releases *unused* blocks).
        try:
            tracker.reset()
        except Exception:
            pass
        tracker = None
        detector = None
        release_gpu(cfg)

    loop_ms = frame_ms
    ignored_boxes = 0
    if getattr(opts, "ignore_regions", False):
        ignore = load_ignore_boxes(Path(spec.seq_dir) / "gt" / "gt.txt",
                                   subsample=int(opts.subsample),
                                   max_frames=opts.max_frames)
        pred, ignored_boxes = excuse_ignore_regions(
            pred, gt, ignore, float(opts.iou_thresh))
    result = evaluate_mot(gt, pred, EvalCfg(iou_thresh=float(opts.iou_thresh)), return_state=True)
    wall_s = clock() - t_wall0

    if opts.save_tracks:
        try:
            write_mot_tracks(Path(opts.out_dir) / "tracks" / f"{spec.pair_id}.txt", pred)
        except Exception as exc:  # never fail a measured pair over a debug dump
            warnings.append(f"save-tracks failed: {type(exc).__name__}: {exc}")

    det_stats = latency_stats(det_ms)
    emb_stats = latency_stats(emb_ms)
    trk_stats = latency_stats(trk_ms)
    frame_stats = latency_stats(loop_ms)
    fps = 1000.0 / frame_stats["mean"] if frame_stats["mean"] > 0 else 0.0

    return {
        "ok": True,
        "pair_id": spec.pair_id,
        "sequence": spec.sequence,
        "tracker": spec.tracker,
        "seq_dir": spec.seq_dir,
        "num_frames": n_frames,
        "num_detections": n_dets,
        "num_orphan_detections": orphan_dets,
        "num_missing_features": missing_feats,
        "num_ignored_boxes": ignored_boxes,
        "timing": {
            "setup_ms": setup_ms,
            "wall_s": wall_s,
            "warmed_up": warmed_up,
            "frame_ms": frame_stats,
            "detector_ms": det_stats,
            "embedder_ms": emb_stats,
            "tracker_ms": trk_stats,
            "pipeline_fps": fps,
            "wall_fps": (n_frames / wall_s) if wall_s > 0 else 0.0,
        },
        "metrics": {k: result.get(k) for k in METRIC_KEYS},
        "result": result,
        "config": {
            "tracker": spec.tracker,
            "with_reid": bool(cfg.mot.with_reid),
            "weights": str(cfg.detector.weights),
            "imgsz": int(cfg.detector.imgsz),
            "conf": float(cfg.detector.conf),
            "iou": float(cfg.detector.iou),
            "max_det": int(cfg.detector.max_det),
            "half": bool(cfg.detector.half),
            "device": str(cfg.detector.device),
            "embed": bool(opts.embed),
            "ignore_regions": bool(getattr(opts, "ignore_regions", False)),
        },
        "embedder": (dict(embedder.info) if embedder is not None else None),
        "gpu_mem_mb": gpu_memory_mb(),
        "warnings": warnings,
    }


def release_gpu(cfg: Config) -> None:
    """Collect garbage and hand unused CUDA blocks back after a pair finishes.

    Called *after* the caller has dropped its own references to the detector and
    tracker, which is what makes ``empty_cache()`` actually able to release them.
    Never raises: a teardown hiccup must not fail a measured pair.
    """
    gc.collect()
    if str(getattr(cfg.detector, "device", "")).startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def pair_worker(payload: tuple[PairSpec, RunOptions]) -> dict[str, Any]:
    """Process-pool entry point: never raises, always returns a record."""
    spec, opts = payload
    try:
        return run_pair(spec, opts)
    except Exception as exc:
        return {
            "ok": False,
            "pair_id": spec.pair_id,
            "sequence": spec.sequence,
            "tracker": spec.tracker,
            "seq_dir": spec.seq_dir,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "metrics": {},
            "timing": {},
        }


def run_pairs(
    specs: Sequence[PairSpec],
    opts: RunOptions,
    *,
    jobs: int = 1,
    recycle_workers: int = 0,
    quiet: bool = False,
    worker: Callable[[tuple[PairSpec, RunOptions]], dict[str, Any]] | None = None,
    stream: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute every pair (serially or in a spawn pool) and return records + parallel meta.

    Returns ``(records, meta)`` where ``meta`` carries the measured parallel
    efficiency: ``wall_s``, ``sum_pair_wall_s`` and ``speedup`` (the honest
    serial-estimate ratio, not a theoretical one).
    """
    stream = stream or sys.stdout
    work = worker or pair_worker
    n_pairs = len(specs)
    eff_jobs = clamp_jobs(jobs, n_pairs)
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def _emit(done: int) -> None:
        if quiet:
            return
        rec = records[-1]
        tag = "ok " if rec.get("ok") else "ERR"
        metrics = rec.get("metrics") or {}
        hota = metrics.get("HOTA")
        hota_txt = f"HOTA={hota:.4f}" if isinstance(hota, (int, float)) else (rec.get("error") or "")
        print(f"[{done}/{n_pairs}] {tag} {rec.get('pair_id')} {hota_txt}", file=stream, flush=True)

    if eff_jobs <= 1:
        for spec in specs:
            records.append(work((spec, opts)))
            _emit(len(records))
    else:
        ctx = multiprocessing.get_context("spawn")
        kwargs: dict[str, Any] = {"max_workers": eff_jobs, "mp_context": ctx}
        if recycle_workers and recycle_workers > 0:
            kwargs["max_tasks_per_child"] = int(recycle_workers)
        with ProcessPoolExecutor(**kwargs) as ex:
            futures = {ex.submit(work, (spec, opts)): spec for spec in specs}
            for fut in as_completed(futures):
                records.append(fut.result())
                _emit(len(records))

    wall_s = time.perf_counter() - t0
    pair_wall = sum(float((r.get("timing") or {}).get("wall_s", 0.0)) for r in records if r.get("ok"))
    records.sort(key=lambda r: (r.get("tracker", ""), r.get("sequence", "")))
    meta = {
        "jobs": eff_jobs,
        "requested_jobs": int(jobs),
        "num_pairs": n_pairs,
        "num_failed": sum(1 for r in records if not r.get("ok")),
        "wall_s": wall_s,
        "sum_pair_wall_s": pair_wall,
        "speedup": (pair_wall / wall_s) if wall_s > 0 else 0.0,
        "mode": "serial" if eff_jobs <= 1 else "process-pool",
        "cpu_count": os.cpu_count() or 1,
        "recycle_workers": int(recycle_workers or 0),
    }
    return records, meta


# --------------------------------------------------------------------------- #
# Aggregation / reports
# --------------------------------------------------------------------------- #
def _row_from_pair(rec: dict[str, Any]) -> dict[str, Any]:
    metrics = rec.get("metrics") or {}
    timing = rec.get("timing") or {}
    frame = timing.get("frame_ms") or {}
    row: dict[str, Any] = {"tracker": rec.get("tracker"), "sequence": rec.get("sequence")}
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    row.update(
        {
            "num_frames": rec.get("num_frames"),
            "num_gt_dets": (rec.get("result") or {}).get("num_gt_dets"),
            "num_pred_dets": (rec.get("result") or {}).get("num_pred_dets"),
            "num_pred_ids": (rec.get("result") or {}).get("num_pred_ids"),
            "ms_per_frame": frame.get("mean"),
            "det_ms": (timing.get("detector_ms") or {}).get("mean"),
            "emb_ms": (timing.get("embedder_ms") or {}).get("mean"),
            "track_ms": (timing.get("tracker_ms") or {}).get("mean"),
            "pipeline_fps": timing.get("pipeline_fps"),
            "wall_s": timing.get("wall_s"),
            "setup_ms": timing.get("setup_ms"),
        }
    )
    return row


def _row_from_summary(tracker: str, result: dict[str, Any], per_seq: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frames = sum(float((r.get("timing") or {}).get("frame_ms", {}).get("n", 0) or 0) for r in per_seq)
    frame_sum = sum(float((r.get("timing") or {}).get("frame_ms", {}).get("sum", 0.0) or 0.0) for r in per_seq)
    mean_ms = (frame_sum / frames) if frames else 0.0
    row: dict[str, Any] = {"tracker": tracker, "num_sequences": result.get("num_sequences")}
    row["num_frames"] = result.get("num_frames")
    for key in METRIC_KEYS:
        row[key] = result.get(key)
    row.update(
        {
            "num_gt_dets": result.get("num_gt_dets"),
            "num_pred_dets": result.get("num_pred_dets"),
            "num_pred_ids": result.get("num_pred_ids"),
            "ms_per_frame": mean_ms,
            "pipeline_fps": (1000.0 / mean_ms) if mean_ms > 0 else 0.0,
            "wall_s": sum(float((r.get("timing") or {}).get("wall_s", 0.0) or 0.0) for r in per_seq),
        }
    )
    return row


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return path


def _md_table(columns: Sequence[str], rows: Sequence[dict[str, Any]], nd: dict[str, int] | None = None) -> str:
    nd = nd or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c), nd.get(c, 4)) for c in columns) + " |")
    return "\n".join(lines)


def aggregate(
    records: Sequence[dict[str, Any]],
    out_dir: str | os.PathLike[str],
    *,
    opts: RunOptions | None = None,
    parallel: dict[str, Any] | None = None,
    trackers: Sequence[str] | None = None,
    sequences: Sequence[str] | None = None,
    scaling: Sequence[dict[str, Any]] | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write per-pair JSON, ``pairs.csv``, ``summary.csv``/``summary.md`` and ``run.json``.

    Tracker-level rows come from :func:`combine_mot` over every successful sequence
    for that tracker (counts summed, AssA/LocA weighted by HOTA_TP, DetA/HOTA
    recomputed) -- never a mean of per-sequence HOTA. Both tables are sorted by
    HOTA descending.
    """
    out = Path(out_dir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    ok_records = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]

    for rec in ok_records:
        payload = {k: v for k, v in rec.items() if k != "result"}
        payload["result"] = rec.get("result")
        with (out / "pairs" / f"{rec.get('pair_id')}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)

    pair_rows = [_row_from_pair(r) for r in ok_records]
    pair_rows.sort(key=lambda r: (-(r.get("HOTA") or 0.0), str(r.get("tracker")), str(r.get("sequence"))))
    _write_csv(out / "pairs.csv", PAIR_COLUMNS, pair_rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in ok_records:
        grouped.setdefault(str(rec.get("tracker")), []).append(rec)
    order = list(trackers) if trackers else sorted(grouped)
    summary_rows: list[dict[str, Any]] = []
    for tracker in order:
        group = grouped.get(str(tracker))
        if not group:
            continue
        try:
            combined = combine_mot([r["result"] for r in group], EvalCfg(iou_thresh=float((opts or RunOptions()).iou_thresh)))
        except Exception as exc:  # a tracker whose pairs all failed must not kill reporting
            failed.append({"pair_id": f"{tracker} (aggregate)", "error": f"{type(exc).__name__}: {exc}", "tracker": tracker})
            continue
        summary_rows.append(_row_from_summary(str(tracker), combined, group))
    summary_rows.sort(key=lambda r: -(r.get("HOTA") or 0.0))
    _write_csv(out / "summary.csv", SUMMARY_COLUMNS, summary_rows)

    md: list[str] = []
    md.append("# MOT evaluation")
    md.append("")
    md.append(f"- pairs: {len(records)} ({len(ok_records)} ok, {len(failed)} failed)")
    if sequences:
        md.append(f"- sequences: {len(sequences)} ({', '.join(Path(s).name for s in sequences)})")
    if parallel:
        md.append(
            f"- workers: jobs={parallel.get('jobs')} mode={parallel.get('mode')} "
            f"wall={_fmt(parallel.get('wall_s'), 1)}s"
        )
        md.append(
            f"- measured scaling: sum(pair wall)={_fmt(parallel.get('sum_pair_wall_s'), 1)}s / "
            f"wall={_fmt(parallel.get('wall_s'), 1)}s -> speedup {_fmt(parallel.get('speedup'), 2)}x "
            f"on {parallel.get('cpu_count')} logical CPUs"
        )
    if opts is not None:
        md.append(
            f"- pipeline: weights={opts.weights or 'config default'}, imgsz={opts.imgsz or 'default'}, "
            f"conf={opts.conf if opts.conf is not None else 'default'}, half={opts.half}, "
            f"embed={opts.embed}, subsample={opts.subsample}, max_frames={opts.max_frames}"
        )
    md.append("")
    md.append("## By tracker (combine_mot over sequences, sorted by HOTA)")
    md.append("")
    md.append(
        _md_table(
            ("tracker", "num_sequences", "HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDP", "IDR", "FP", "FN", "IDSW", "Frag", "MT", "ML", "ms/frame", "FPS"),
            [{**r, "ms/frame": r.get("ms_per_frame"), "FPS": r.get("pipeline_fps")} for r in summary_rows],
            nd={"ms/frame": 2, "FPS": 2},
        )
    )
    md.append("")
    md.append("## Per (sequence, tracker), sorted by HOTA")
    md.append("")
    md.append(
        _md_table(
            ("tracker", "sequence", "HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "FP", "FN", "IDSW", "Frag", "ms/frame", "FPS"),
            [{**r, "ms/frame": r.get("ms_per_frame"), "FPS": r.get("pipeline_fps")} for r in pair_rows],
            nd={"ms/frame": 2, "FPS": 2},
        )
    )
    if scaling:
        md.append("")
        md.append("## Parallel scaling (same plan, different worker counts)")
        md.append("")
        md.append(_md_table(("jobs", "pairs", "wall_s", "sum_pair_wall_s", "speedup"), scaling, nd={"wall_s": 1, "sum_pair_wall_s": 1, "speedup": 2}))
    if failed:
        md.append("")
        md.append("## Failures")
        md.append("")
        for rec in failed:
            md.append(f"- {rec.get('pair_id')}: {rec.get('error')}")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    run_meta = {
        "argv": list(argv or []),
        "options": (opts or RunOptions()).to_dict(),
        "trackers": list(trackers or []),
        "sequences": [Path(s).name for s in (sequences or [])],
        "parallel": parallel or {},
        "scaling": list(scaling or []),
        "summary": summary_rows,
        "pairs": pair_rows,
        "failures": [{"pair_id": r.get("pair_id"), "error": r.get("error")} for r in failed],
        "env": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "tgtrack_version": getattr(tgtrack, "__version__", ""),
        },
        "artifacts": {
            "pairs_csv": str(out / "pairs.csv"),
            "summary_csv": str(out / "summary.csv"),
            "summary_md": str(out / "summary.md"),
        },
    }
    with (out / "run.json").open("w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, sort_keys=True, default=str)
    return run_meta


def scaling_table(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise ``{jobs, meta}`` sweep entries into table rows for reports."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        meta = entry.get("meta") or {}
        rows.append(
            {
                "jobs": entry.get("jobs", meta.get("jobs")),
                "pairs": meta.get("num_pairs"),
                "wall_s": meta.get("wall_s"),
                "sum_pair_wall_s": meta.get("sum_pair_wall_s"),
                "speedup": meta.get("speedup"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_mot.py",
        description="Parallel MOT evaluation over MOT-format sequences (HOTA/MOTA/IDF1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="data/sequences", help="root holding MOT sequences (or one sequence dir)")
    p.add_argument("--out", default="results/mot_eval", help="output directory (JSON/CSV/MD)")
    p.add_argument("--trackers", default=",".join(DEFAULT_TRACKERS), help="comma list; bare stems get .yaml")
    p.add_argument("--jobs", type=int, default=4, help="worker processes (0 = cpu_count)")
    p.add_argument("--recycle-workers", type=int, default=0, help="0 = keep workers; N = recycle after N pairs")
    p.add_argument("--jobs-sweep", default="", help="e.g. 1,2,4: re-run the plan once per worker count and report scaling")
    p.add_argument("--sequences", default="", help="comma list of sequence-name substrings to keep")
    p.add_argument("--limit-sequences", type=int, default=None, help="keep only the first N sequences")
    p.add_argument("--layout", default="auto", choices=("auto", "mot", "yolo", "coco"), help="discover_sequences layout")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per sequence")
    p.add_argument("--subsample", type=int, default=1, help="keep every K-th frame (MOT frame ids are renumbered)")
    p.add_argument("--min-conf", type=float, default=1.0, help="min GT confidence (0 rows are 'ignore', dropped)")
    p.add_argument("--min-visible", type=float, default=0.0, help="min GT visibility")
    p.add_argument("--iou-thresh", type=float, default=0.5, help="matching IoU for CLEAR + Identity (TrackEval default)")
    p.add_argument("--config", default=None, help="YAML/JSON config for the detector/tracker")
    p.add_argument("--weights", default=None, help="detector weights (default: config / local yolo26n.pt)")
    p.add_argument("--imgsz", type=int, default=None, help="detector input size")
    p.add_argument("--conf", type=float, default=None, help="detector confidence threshold")
    p.add_argument("--iou", type=float, default=None, help="detector NMS IoU")
    p.add_argument("--max-det", type=int, default=None, help="detector max detections")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu")
    p.add_argument("--half", dest="half", action="store_true", default=True, help="FP16 on CUDA")
    p.add_argument("--no-half", dest="half", action="store_false", help="FP32")
    p.add_argument("--embed", action="store_true", help="encode real ReID features and feed them to the tracker")
    p.add_argument("--no-embed", dest="embed", action="store_false", help="detector-only appearance (default)")
    p.add_argument("--embed-model", default=None, help="ReID model (ONNX/PT) for --embed")
    p.add_argument("--embed-fallback", action="store_true", help="force the cheap non-neural descriptor")
    p.add_argument("--with-reid", action="store_true", help="enable the tracker's appearance term even without --embed")
    p.add_argument("--cache-flush", type=int, default=50, help="decoded-frame cache flush period (0 = never)")
    p.add_argument("--save-tracks", action="store_true", help="also dump per-pair MOT-format track files")
    p.add_argument("--ignore-regions", action="store_true",
                   help="MOTChallenge ignore convention: do not score detections that land on a "
                        "conf=0 distractor region (recommended for MOT17; E008/E009)")
    p.add_argument("--dry-run", action="store_true", help="only enumerate the plan (no torch, no data needed)")
    p.add_argument("--quiet", action="store_true", help="no per-pair progress lines")
    return p


def options_from_args(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        out_dir=str(args.out),
        subsample=int(args.subsample),
        max_frames=(None if args.max_frames in (None, 0) else int(args.max_frames)),
        min_conf=float(args.min_conf),
        min_visible=float(args.min_visible),
        config=(str(args.config) if args.config else None),
        weights=(str(args.weights) if args.weights else None),
        imgsz=(int(args.imgsz) if args.imgsz else None),
        conf=(float(args.conf) if args.conf is not None else None),
        iou=(float(args.iou) if args.iou is not None else None),
        max_det=(int(args.max_det) if args.max_det else None),
        device=(str(args.device) if args.device else None),
        half=bool(args.half),
        embed=bool(args.embed),
        embed_model=(str(args.embed_model) if args.embed_model else None),
        embed_fallback=bool(args.embed_fallback),
        with_reid=bool(args.with_reid),
        cache_flush=int(args.cache_flush),
        save_tracks=bool(args.save_tracks),
        ignore_regions=bool(getattr(args, "ignore_regions", False)),
        iou_thresh=float(args.iou_thresh),
        layout=str(args.layout),
    )


def plan_sequences(seq_dirs: Sequence[str], out_dir: Path) -> list[dict[str, Any]]:
    """Cheap per-sequence description for the dry run (GT parsing only, no pixels)."""
    rows: list[dict[str, Any]] = []
    for seq_dir in seq_dirs:
        try:
            seq = load_mot_sequence(str(seq_dir))
            summary = sequence_summary(seq)
            summary["seq_dir"] = str(seq_dir)
            rows.append(summary)
        except Exception as exc:
            rows.append({"name": Path(str(seq_dir)).name, "seq_dir": str(seq_dir), "error": f"{type(exc).__name__}: {exc}"})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    opts = options_from_args(args)
    trackers = parse_trackers(args.trackers)
    out_dir = Path(opts.out_dir)

    seq_dirs, notes = discover_in(args.data_root, layout=str(args.layout))
    seq_dirs = filter_sequences(seq_dirs, only=args.sequences, limit=args.limit_sequences)
    specs = enumerate_pairs(seq_dirs, trackers)

    for note in notes:
        print(f"warning: {note}", file=sys.stderr)

    if args.dry_run:
        plan = {
            "data_root": str(args.data_root),
            "out_dir": str(out_dir),
            "trackers": trackers,
            "jobs": clamp_jobs(int(args.jobs), len(specs)),
            "num_sequences": len(seq_dirs),
            "num_pairs": len(specs),
            "sequences": plan_sequences(seq_dirs, out_dir),
            "pairs": [s.to_dict() for s in specs],
            "options": opts.to_dict(),
            "warnings": notes,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "plan.json").open("w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2, sort_keys=True)
        print(f"dry run: {len(seq_dirs)} sequence(s) x {len(trackers)} tracker(s) = {len(specs)} pair(s)")
        print(f"trackers: {', '.join(trackers)}")
        print(f"plan written: {out_dir / 'plan.json'}")
        for row in plan["sequences"]:
            if "error" in row:
                print(f"  ! {row['name']}: {row['error']}")
            else:
                print(f"  - {row['name']}: {row.get('num_frames')} frames, {row.get('num_gt_ids')} gt ids, {row.get('num_gt_boxes')} boxes")
        return 0

    if not specs:
        print("nothing to evaluate: no sequences found (see warnings)", file=sys.stderr)
        aggregate([], out_dir, opts=opts, trackers=trackers, sequences=seq_dirs, argv=sys.argv)
        return 1

    sweep = [int(x) for x in str(args.jobs_sweep).replace(";", ",").split(",") if x.strip()]
    scaling: list[dict[str, Any]] = []
    if sweep:
        for jobs in sweep:
            records, meta = run_pairs(specs, opts, jobs=jobs, recycle_workers=int(args.recycle_workers), quiet=bool(args.quiet))
            scaling.append({"jobs": int(jobs), "meta": meta, "records": records})
            print(f"jobs={jobs}: wall={meta['wall_s']:.1f}s speedup={meta['speedup']:.2f}x", flush=True)
        last = scaling[-1]
        records, meta = last["records"], last["meta"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "scaling.json").open("w", encoding="utf-8") as fh:
            json.dump(scaling_table(scaling), fh, indent=2)
    else:
        records, meta = run_pairs(specs, opts, jobs=int(args.jobs), recycle_workers=int(args.recycle_workers), quiet=bool(args.quiet))

    run_meta = aggregate(
        records,
        out_dir,
        opts=opts,
        parallel=meta,
        trackers=trackers,
        sequences=seq_dirs,
        scaling=scaling_table(scaling) if scaling else None,
        argv=sys.argv,
    )
    summary = run_meta.get("summary") or []
    print(f"pairs: {len(records)} ({meta['num_failed']} failed), wall {meta['wall_s']:.1f}s, speedup {meta['speedup']:.2f}x")
    if summary:
        head = summary[0]
        print(
            f"best: {head.get('tracker')} HOTA={_fmt(head.get('HOTA'))} MOTA={_fmt(head.get('MOTA'))} "
            f"IDF1={_fmt(head.get('IDF1'))} ({_fmt(head.get('ms_per_frame'), 2)} ms/frame)"
        )
    print(f"reports: {out_dir / 'summary.md'}, {out_dir / 'summary.csv'}, {out_dir / 'run.json'}")
    return 0 if meta["num_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
