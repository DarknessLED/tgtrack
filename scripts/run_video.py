#!/usr/bin/env python
"""End-to-end demo/eval runner: YOLO26 person detection + an all-person MOT layer.

Pipeline per frame:
    PersonDetector.detect(frame)                 -> list[Detection] (person only)
    [optional] Embedder.encode(frame, boxes)     -> det.feat / det.part_feats
    AllPersonTracker.update(frame, detections)   -> det.track_id filled IN PLACE

Outputs (under ``--out``):
    detections.jsonl   one JSON object per processed frame (streaming, no memory blow-up)
    summary.json       aggregate metrics + effective config/backends
    annotated.mp4      only with ``--save-video`` (``mp4v`` fourcc: no ffmpeg needed)

Examples:
    python scripts/run_video.py --source data/clip.mp4 --out results/run1 --save-video
    python scripts/run_video.py --source data/frames --tracker deepocsort.yaml --out results/run2
    python scripts/run_video.py --source data/bus.jpg --no-embed --out results/run3
    python scripts/run_video.py --source data/clip.mp4 --all-trackers --out results/sweep

Notes:
    * Console encoding on this machine is GBK, so every message is ASCII-only.
    * ``--no-embed`` (or a failed Embedder import) degrades gracefully: the MOT
      layer then runs IoU/motion-only, which ``AllPersonTracker`` supports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tgtrack.config import load_config  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.mot_layer import TRACKER_YAMLS, AllPersonTracker  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}

#: github.com is blocked on this machine; look for checkpoints in these directories
#: before letting ultralytics attempt (and slowly fail) a download.
WEIGHT_DIRS = (Path("weights"), Path("data") / "weights", Path("results") / "weights")


def resolve_local_weights(name: str) -> str:
    """Return a local path for ``name`` when one exists, else ``name`` unchanged.

    Handles both the detector checkpoint and the ReID ONNX, e.g.
    ``yolo26n-reid.onnx`` -> ``data/weights/yolo26n-reid.onnx``.
    ``$TGTRACK_WEIGHTS_DIR`` is checked last so a shared cache can be pointed at.
    """
    p = Path(name)
    if p.exists() or p.is_absolute():
        return str(p)
    for d in WEIGHT_DIRS:
        cand = d / p.name
        if cand.exists():
            return str(cand)
    env_dir = os.environ.get("TGTRACK_WEIGHTS_DIR")
    if env_dir and (Path(env_dir) / p.name).exists():
        return str(Path(env_dir) / p.name)
    return name


# --------------------------------------------------------------------------- IO
class SourceReader:
    """Uniform frame source: single image, image directory, or video file."""

    def __init__(self, source: str | Path, fps: float = 30.0) -> None:
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f"source not found: {self.source}")
        self.cap = None
        self._frames: list[Path] = []
        self.warn: list[str] = []

        if self.source.is_dir():
            self.kind = "images"
            self._frames = sorted(p for p in self.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self._frames:
                raise FileNotFoundError(f"no images ({sorted(IMAGE_EXTS)}) in directory {self.source}")
            first = _imread(self._frames[0])
            self.height, self.width = first.shape[:2]
            self.count = len(self._frames)
            self.fps = float(fps)  # images have no intrinsic rate; used for the mp4 only
            self.name = self.source.name
        elif self.source.suffix.lower() in IMAGE_EXTS:
            self.kind = "image"
            img = _imread(self.source)
            self.height, self.width = img.shape[:2]
            self.count = 1
            self.fps = float(fps)
            self.name = self.source.name
        else:
            self.kind = "video"
            import cv2

            self.cap = cv2.VideoCapture(str(self.source))
            if not self.cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture could not open {self.source}")
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(fps)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.count = n if n > 0 else None
            self.name = self.source.name

    def frames(self, stride: int = 1, max_frames: int | None = None):
        """Yield ``(frame_idx, timestamp_s, bgr)`` with frame_idx = source frame number."""
        stride = max(1, int(stride))
        emitted = 0
        if self.kind == "video":
            import cv2

            idx = -1
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                yield idx, idx / self.fps, frame
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
        else:
            paths = [self.source] if self.kind == "image" else self._frames
            for i, p in enumerate(paths):
                if i % stride:
                    continue
                img = _imread(p)
                if img is None:
                    self.warn.append(f"unreadable image skipped: {p}")
                    continue
                yield i, i / self.fps, img
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
        self.close()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _imread(path: Path) -> np.ndarray:
    import cv2

    # imread() mangles non-ASCII paths on Windows; imdecode does not.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode image: {path}")
    return img


# ---------------------------------------------------------------------- drawing
_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def _id_color(track_id) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)
    return _PALETTE[int(track_id) % len(_PALETTE)]


def annotate(frame: np.ndarray, dets, thickness: int = 2) -> np.ndarray:
    """Draw person boxes with their track id + score (BGR copy, input untouched)."""
    import cv2

    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(float(v))) for v in d.xyxy)
        color = _id_color(d.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        tag = f"id{d.track_id}" if d.track_id is not None else "id-"
        label = f"{tag} {float(d.score):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - base - 2)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------------ stats
class TrackStats:
    """Per-id lifetime bookkeeping -> unique ids + an ID-switch-ish heuristic."""

    def __init__(self) -> None:
        self.first: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.count: dict[int, int] = {}
        self.gaps: dict[int, int] = {}
        self.per_frame: list[int] = []
        self.max_concurrent = 0
        self._absent = 0

    def observe(self, frame_idx: int, dets) -> None:
        seen = {int(d.track_id) for d in dets if d.track_id is not None}
        for tid in seen:
            if tid not in self.first:
                self.first[tid] = frame_idx
            elif frame_idx - self.last[tid] > 1:
                self.gaps[tid] = self.gaps.get(tid, 0) + 1  # lost then reappeared
            self.last[tid] = frame_idx
            self.count[tid] = self.count.get(tid, 0) + 1
        self.max_concurrent = max(self.max_concurrent, len(seen))
        self.per_frame.append(len(seen))

    def summary(self, first_frame: int) -> dict:
        ids = sorted(self.first)
        base = [i for i in ids if self.first[i] == first_frame]
        later = [i for i in ids if self.first[i] != first_frame]
        aside = max(1, self.max_concurrent)
        lives = [self.count[i] for i in ids] or [0]
        return {
            "unique_track_ids": len(ids),
            "ids_on_first_frame": len(base),
            "ids_created_after_first_frame": len(later),
            "id_gap_events": int(sum(self.gaps.values())),  # id vanished then came back
            "short_tracks_lt3_frames": int(sum(1 for c in lives if c < 3)),
            "max_concurrent_tracks": self.max_concurrent,
            "mean_tracks_per_frame": round(float(np.mean(self.per_frame)), 3) if self.per_frame else 0.0,
            "id_switch_proxy": round(len(later) / aside, 4),  # extra ids minted per concurrent track
            "track_lifetimes": {str(i): self.count[i] for i in ids},
            "first_seen_frame": {str(i): self.first[i] for i in ids},
        }


# ------------------------------------------------------------------------- main
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_video.py",
        description="YOLO26 person detection + all-person tracking, with JSONL/MP4 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, help="video file, image file, or directory of images")
    p.add_argument("--out", default="results/run1", help="output directory (created if missing)")
    p.add_argument("--tracker", default=None, help=f"tracker yaml name or path; one of {list(TRACKER_YAMLS)}")
    p.add_argument("--all-trackers", action="store_true", help="run every direct-API backend into per-tracker subdirs")
    p.add_argument("--save-video", action="store_true", help="write annotated.mp4 (mp4v fourcc, no ffmpeg)")
    p.add_argument("--config", default=None, help="optional YAML/JSON config (Config schema)")
    p.add_argument("--weights", default=None, help="override detector weights, e.g. yolo26n.pt")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (falls back to cpu if no CUDA build)")
    p.add_argument("--half", dest="half", action="store_true", default=None, help="FP16 (CUDA only)")
    p.add_argument("--no-half", dest="half", action="store_false", help="disable FP16")
    p.add_argument("--all-classes", action="store_true", help="keep every class (default: persons only)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    p.add_argument("--fps", type=float, default=30.0, help="frame rate for image sources / the mp4 writer")
    p.add_argument("--no-embed", action="store_true", help="skip appearance embeddings (IoU/motion only)")
    p.add_argument("--embed-model", default=None, help="override EmbedderCfg.model (e.g. yolo26n-reid.onnx)")
    p.add_argument("--embed-fallback", action="store_true", help="force the dependency-free descriptor")
    p.add_argument("--quiet", action="store_true", help="only print the final summary line")
    return p


def _apply_overrides(cfg, args) -> None:
    if args.weights is not None:
        cfg.detector.weights = args.weights
    cfg.detector.weights = resolve_local_weights(cfg.detector.weights)
    if args.imgsz is not None:
        cfg.detector.imgsz = int(args.imgsz)
    if args.conf is not None:
        cfg.detector.conf = float(args.conf)
    if args.iou is not None:
        cfg.detector.iou = float(args.iou)
    if args.max_det is not None:
        cfg.detector.max_det = int(args.max_det)
    if args.device is not None:
        cfg.detector.device = args.device
    if args.half is not None:
        cfg.detector.half = bool(args.half)
    if args.all_classes:
        cfg.detector.person_class = None
    if args.tracker is not None:
        cfg.mot.tracker = args.tracker
    if args.embed_model is not None:
        cfg.embedder.model = args.embed_model
    cfg.embedder.model = resolve_local_weights(cfg.embedder.model)


def build_embedder(cfg, args, log):
    """Guarded construction: any failure downgrades to 'no appearance features'."""
    if args.no_embed:
        log("[embed] disabled by --no-embed")
        return None
    try:
        from tgtrack.appearance import Embedder
    except Exception as exc:  # pragma: no cover - depends on the workspace state
        log(f"[embed] WARN appearance.Embedder unavailable ({type(exc).__name__}: {exc}); continuing without it")
        return None
    try:
        emb = Embedder(cfg.embedder, device=args.device or cfg.detector.device, force_fallback=args.embed_fallback)
    except Exception as exc:  # pragma: no cover - depends on the environment
        log(f"[embed] WARN Embedder construction failed ({type(exc).__name__}: {exc}); continuing without it")
        return None
    log(f"[embed] {emb.info}")
    return emb


def run_once(args, cfg, tracker_name: str, outdir: Path) -> dict:
    """Detect + track one source with one tracker backend; write JSONL/MP4/summary."""
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "detections.jsonl"
    summary_path = outdir / "summary.json"
    video_path = outdir / "annotated.mp4"
    quiet = bool(args.quiet)

    def log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    cfg.mot.tracker = tracker_name
    reader = SourceReader(args.source, fps=args.fps)
    detector = PersonDetector(cfg.detector)
    tracker = AllPersonTracker(cfg.mot)  # args mirror ultralytics/trackers/track.py:90
    embedder = build_embedder(cfg, args, log)
    emb_parts = tuple(getattr(cfg.embedder, "global_parts", ("full",)) or ("full",))

    log(f"[source] kind={reader.kind} name={reader.name} size={reader.width}x{reader.height} "
        f"frames={reader.count} fps={reader.fps:.2f}")
    log(f"[detector] {detector.model_info()}")
    log(f"[tracker] {tracker.describe()}")

    writer = None
    want_video = bool(args.save_video)
    stats = TrackStats()
    times: list[float] = []
    n_frames = n_dets = frames_with_person = 0
    first_frame_idx = None
    t_start = time.perf_counter()

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for frame_idx, t_sec, frame in reader.frames(stride=args.stride, max_frames=args.max_frames):
            if first_frame_idx is None:
                first_frame_idx = frame_idx
            t0 = time.perf_counter()

            dets = detector.detect(frame)

            if embedder is not None and dets:
                boxes = np.stack([d.xyxy for d in dets]).astype(np.float32)
                enc = embedder.encode(frame, boxes, parts=emb_parts)
                for d, per_part in zip(dets, enc):
                    if per_part:
                        d.part_feats = {k: v for k, v in per_part.items()}
                        d.feat = per_part.get("full", next(iter(per_part.values())))

            tracked = tracker.update(frame, dets)  # fills det.track_id in place
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

            n_frames += 1
            n_dets += len(tracked)
            frames_with_person += int(bool(tracked))
            stats.observe(frame_idx, tracked)

            fh.write(json.dumps({
                "frame": frame_idx,
                "t": round(float(t_sec), 4),
                "n": len(tracked),
                "ms": round(dt, 3),
                "tracks": [
                    {
                        "id": (int(d.track_id) if d.track_id is not None else None),
                        "xyxy": [round(float(v), 2) for v in d.xyxy],
                        "score": round(float(d.score), 4),
                        "quality": round(float(d.quality), 4),
                        "visible_ratio": round(float(d.visible_ratio), 4),
                    }
                    for d in tracked
                ],
            }) + "\n")

            if want_video:
                import cv2

                canvas = annotate(frame, tracked)
                if writer is None:
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps, (w, h))
                    if not writer.isOpened():
                        log("[video] WARN cv2.VideoWriter could not open annotated.mp4; continuing without video")
                        writer = None
                        want_video = False
                if writer is not None:
                    if canvas.shape[:2] != (reader.height, reader.width):
                        canvas = cv2.resize(canvas, (reader.width, reader.height))
                    writer.write(canvas)

    if writer is not None:
        writer.release()
    wall = time.perf_counter() - t_start

    t_arr = np.asarray(times, dtype=np.float64) if times else np.zeros(1)
    summary = {
        "source": str(args.source),
        "source_kind": reader.kind,
        "source_name": reader.name,
        "source_frames": reader.count,
        "stride": int(args.stride),
        "tracker": tracker_name,
        "detector": detector.model_info(),
        "tracker_cfg": tracker.describe(),
        "embedder": (embedder.info if embedder is not None else {"mode": "disabled"}),
        "frames_processed": n_frames,
        "frames_with_person": frames_with_person,
        "total_detections": n_dets,
        "ms_mean": round(float(t_arr.mean()), 3),
        "ms_p95": round(float(np.percentile(t_arr, 95)), 3),
        "ms_min": round(float(t_arr.min()), 3),
        "pipeline_fps": round(1000.0 / float(t_arr.mean()), 2) if t_arr.mean() > 0 else 0.0,
        "wall_seconds": round(wall, 3),
        "end_to_end_fps": round(n_frames / wall, 2) if wall > 0 else 0.0,
        "video_written": str(video_path) if writer is not None else None,
        "jsonl": str(jsonl_path),
        "warnings": reader.warn,
    }
    summary.update(stats.summary(first_frame_idx if first_frame_idx is not None else 0))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "[result] tracker={} frames={} persons={} total_dets={} dets/frame={:.2f} "
        "unique_ids={} ids_after_frame0={} gap_events={} max_concurrent={} "
        "ms_mean={:.2f} pipeline_fps={:.2f} e2e_fps={:.2f} id_switch_proxy={:.3f} out={}".format(
            tracker_name,
            n_frames,
            frames_with_person,
            n_dets,
            (n_dets / n_frames) if n_frames else 0.0,
            summary["unique_track_ids"],
            summary["ids_created_after_first_frame"],
            summary["id_gap_events"],
            summary["max_concurrent_tracks"],
            summary["ms_mean"],
            summary["pipeline_fps"],
            summary["end_to_end_fps"],
            summary["id_switch_proxy"],
            outdir,
        ),
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    out_root = Path(args.out)

    if args.all_trackers:
        names = list(TRACKER_YAMLS)
    else:
        name = args.tracker or cfg.mot.tracker
        names = [Path(name).name if str(name).endswith((".yaml", ".yml")) else f"{name}.yaml"]

    results = {}
    for name in names:
        outdir = out_root / Path(name).stem if args.all_trackers else out_root
        try:
            results[name] = run_once(args, cfg, name, outdir)
        except Exception as exc:
            print(f"[error] tracker={name} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    if args.all_trackers:
        table = [
            {
                "tracker": k,
                "unique_ids": v.get("unique_track_ids"),
                "ids_after_frame0": v.get("ids_created_after_first_frame"),
                "gap_events": v.get("id_gap_events"),
                "max_concurrent": v.get("max_concurrent_tracks"),
                "ms_mean": v.get("ms_mean"),
                "pipeline_fps": v.get("pipeline_fps"),
                "id_switch_proxy": v.get("id_switch_proxy"),
                "error": v.get("error"),
            }
            for k, v in results.items()
        ]
        (out_root / "tracker_comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
        print("\n[comparison] " + json.dumps(table), flush=True)

    if not results or all("error" in v for v in results.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
