"""Click-to-track demo — the product behaviour, end to end.

This is the front door of the deliverable.  It does what the user asked for:

1. detect **every** person in the frame (YOLO26n);
2. give the other people *identity separation only* (an all-person MOT layer);
3. let the operator **click** one person and then follow exactly that person,
   reporting an explicit state instead of silently drifting onto a bystander.

Usage
-----
    # auto-pick the largest person in frame 0 and follow them
    python scripts/demo_click.py --source results/_assets/persons_pan.mp4 --save-video

    # click a specific point (x,y) in frame 0
    python scripts/demo_click.py --source clip.mp4 --click 640,300

    # a MOTChallenge sequence directory (img1/ + gt/gt.txt): also scores the run
    python scripts/demo_click.py --source data/sequences/MOT17-02-FRCNN --click 812,417

    # compare the anti-latch margin on the same clip
    python scripts/demo_click.py --source clip.mp4 --distractor-margin 0.0
    python scripts/demo_click.py --source clip.mp4 --distractor-margin 0.12

Outputs (under ``--out``): ``target.jsonl``, ``events.json``, ``summary.json``,
``annotated.mp4`` and, when ground truth exists, ``metrics.json``.

Note: ``scripts/`` is not a package, so ``src/`` is put on ``sys.path`` here
rather than relying on an installed distribution.
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

from tgtrack import Config, Detection, TargetTracker, ensure_ort_cuda, load_config  # noqa: E402
from tgtrack.appearance import Embedder  # noqa: E402
from tgtrack.detector import PersonDetector  # noqa: E402
from tgtrack.mot_layer import AllPersonTracker  # noqa: E402

# BGR.  Green = the target is confidently ours, red = we admit we lost them.
# A tracker that draws the same colour in every state is lying about its
# confidence, which is exactly the failure mode this project exists to avoid.
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "init": (80, 220, 80),
    "tracking": (80, 220, 80),
    "reacquiring": (0, 200, 255),
    "occluded_soft": (0, 165, 255),
    "lost_short": (0, 80, 255),
    "lost_long": (0, 0, 255),
    "out_of_frame": (255, 0, 200),
    "gave_up": (128, 128, 128),
}
BYSTANDER_COLOR = (200, 200, 200)
HUD_BG = (32, 32, 32)


# --------------------------------------------------------------------------- #
# frame sources
# --------------------------------------------------------------------------- #
def _as_bgr(item) -> np.ndarray | None:
    """A source item may already be a decoded array or a path to decode."""
    if isinstance(item, np.ndarray):
        return item
    if isinstance(item, (str, Path)):
        return cv2.imread(str(item))
    return None


def is_mot_sequence(path: Path) -> bool:
    return path.is_dir() and (path / "img1").is_dir()


def open_source(source: str, max_frames: int | None, subsample: int):
    """Yield ``(frame_idx, frame_bgr)`` from a video, an image dir or a MOT seq.

    Returns ``(generator, sequence_or_None)`` — the :class:`Sequence` is returned
    when the source is a MOT directory, so the caller can score the run.
    """
    path = Path(source)

    if is_mot_sequence(path):
        from tgtrack.data import load_mot_sequence

        seq = load_mot_sequence(path, subsample=subsample, max_frames=max_frames)

        def gen():
            for i, item in enumerate(seq.frames):
                frame = _as_bgr(item)
                if frame is not None:
                    yield i, frame

        return gen(), seq

    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)[::subsample]
        if max_frames:
            files = files[:max_frames]

        def gen():
            for i, f in enumerate(files):
                frame = cv2.imread(str(f))
                if frame is not None:
                    yield i, frame

        return gen(), None

    if not path.exists():
        raise SystemExit(f"source not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")

    def gen():
        i = 0
        emitted = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if subsample <= 1 or i % subsample == 0:
                yield i, frame
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            i += 1
        cap.release()

    return gen(), None


# --------------------------------------------------------------------------- #
# picking the target (the "click")
# --------------------------------------------------------------------------- #
def pick_by_click(dets: list[Detection], click: tuple[int, int]) -> Detection | None:
    """The smallest detection containing the click — most specific wins.

    Mirrors :func:`tgtrack.data.simulate_click`, but works on detections so the
    demo does not need ground truth.
    """
    cx, cy = click
    hits = [
        d
        for d in dets
        if d.xyxy[0] <= cx <= d.xyxy[2] and d.xyxy[1] <= cy <= d.xyxy[3]
    ]
    if not hits:
        return None
    return min(hits, key=lambda d: float(d.area))


def pick_largest(dets: list[Detection]) -> Detection | None:
    """Auto-select: the biggest person, who is usually the intended subject."""
    return max(dets, key=lambda d: float(d.area)) if dets else None


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def followed_track_id(status, dets: list[Detection]) -> int | None:
    """Which all-person track id we are currently believed to be following.

    Prefers the tracker's own diagnosis; otherwise falls back to the best-IoU
    detection, so the summary can report identity changes honestly.
    """
    diag = getattr(status, "diag", None) or {}
    for key in ("track_id", "matched_track_id", "target_track_id"):
        if diag.get(key) is not None:
            try:
                return int(diag[key])
            except (TypeError, ValueError):
                pass
    if status.xyxy is None:
        return None
    best, best_iou = None, 0.3
    for d in dets:
        if d.track_id is None:
            continue
        v = iou_xyxy(np.asarray(status.xyxy, np.float32), np.asarray(d.xyxy, np.float32))
        if v > best_iou:
            best, best_iou = d.track_id, v
    return int(best) if best is not None else None


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw_frame(frame, dets, status, hud: list[str]) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), BYSTANDER_COLOR, 1)
        if d.track_id is not None:
            cv2.putText(
                out,
                f"id{d.track_id}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                BYSTANDER_COLOR,
                1,
                cv2.LINE_AA,
            )

    if status is not None and status.xyxy is not None:
        state = getattr(status.state, "value", str(status.state))
        color = STATE_COLORS.get(state, (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in status.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        # Corner ticks make the target obvious even over a busy background.
        arm = max(10, (x2 - x1) // 6)
        for px, py, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(out, (px, py), (px + dx * arm, py), color, 3)
            cv2.line(out, (px, py), (px, py + dy * arm), color, 3)
        cv2.putText(
            out,
            f"TARGET {state}",
            (x1, max(16, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    for i, line in enumerate(hud):
        y = 22 + i * 20
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (8, y - 15), (16 + tw, y + 6), HUD_BG, -1)
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return out


# --------------------------------------------------------------------------- #
def build_config(args) -> Config:
    import torch

    cuda = bool(torch.cuda.is_available()) and not args.cpu
    overrides: dict = {
        "detector.half": cuda,
        "mot.tracker": args.tracker,
        "cost.distractor_margin": args.distractor_margin,
    }
    if args.no_parts:
        overrides["embedder.target_parts"] = ("full",)
    if args.no_freeze:
        overrides["state.frozen_states"] = ()
    # `load_config` already applies overrides, so do not clone on top of it too.
    return load_config(args.config, **overrides) if args.config else Config().clone(**overrides)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, image dir, or MOT sequence dir")
    ap.add_argument("--out", default="results/click_demo")
    ap.add_argument("--click", default=None, help="click point 'x,y'; default = largest person")
    ap.add_argument("--frame", type=int, default=0, help="frame index to take the click on")
    ap.add_argument("--tracker", default="botsort.yaml", help="all-person tracker yaml")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--config", default=None, help="optional yaml config to load as the base")
    ap.add_argument("--distractor-margin", type=float, default=None,
                    help="anti-latch margin; 0.0 disables refusal")
    ap.add_argument("--no-parts", action="store_true", help="use only the 'full' appearance bank")
    ap.add_argument("--no-freeze", action="store_true", help="do NOT freeze memory while blind")
    ap.add_argument("--fallback-embedder", action="store_true", help="force the non-neural fallback")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.distractor_margin is None:
        args.distractor_margin = Config().cost.distractor_margin

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The ReID CUDA provider only loads if torch's bundled CUDA-12 DLLs are on the
    # DLL search path.  Without this the encoder silently runs on CPU (~8x slower).
    ort_info = ensure_ort_cuda()

    cfg = build_config(args)
    detector = PersonDetector(cfg.detector)
    detector.warmup()
    mot = AllPersonTracker(cfg.mot, detector=detector)
    embedder = Embedder(
        cfg.embedder,
        device="cpu" if args.cpu else "cuda",
        force_fallback=args.fallback_embedder,
    )
    tracker = TargetTracker(cfg, embedder=embedder)

    info = detector.model_info()
    if not args.quiet:
        print("=" * 78)
        print("click-to-track demo")
        print("=" * 78)
        print(f"detector   : {info['weights_path']}  ({info['n_params_M']:.2f} M params, "
              f"{info['device']}, half={info['half']})")
        print(f"embedder   : mode={embedder.mode} dim={embedder.dim} "
              f"provider={embedder.provider or 'n/a'} device={embedder.device}")
        if embedder.fallback_reason:
            print(f"             fallback reason: {embedder.fallback_reason}")
        print(f"tracker    : {cfg.mot.tracker}   margin={cfg.cost.distractor_margin}")
        print(f"target rts : {cfg.embedder.target_parts}  freeze_while_blind="
              f"{bool(cfg.state.frozen_states)}")
        print(f"ORT        : {ort_info.get('available', 'n/a')}")

    frames_gen, seq = open_source(args.source, args.max_frames, args.subsample)

    click = None
    if args.click:
        parts = args.click.replace(" ", "").split(",")
        click = (int(float(parts[0])), int(float(parts[1])))

    writer: cv2.VideoWriter | None = None
    rows: list[dict] = []
    statuses: dict[int, object] = {}
    started = False
    target_gt_id: int | None = None
    init_track_id: int | None = None
    init_box: np.ndarray | None = None
    t0 = time.perf_counter()
    n_frames = 0

    for idx, frame in frames_gen:
        n_frames += 1
        h, w = frame.shape[:2]
        dets = detector.detect(frame)
        mot.update(frame, dets)

        status = None
        if not started and idx >= args.frame and dets:
            picked = pick_by_click(dets, click) if click else pick_largest(dets)
            if picked is not None:
                init_box = np.asarray(picked.xyxy, np.float32)
                init_track_id = picked.track_id
                status = tracker.initialize(frame, init_box, frame_idx=idx, score=float(picked.score))
                started = True
                if seq is not None and seq.gt:
                    # map the click to a ground-truth identity for scoring
                    best, best_iou = None, 0.0
                    for gf, boxes in seq.gt.items():
                        if gf != idx:
                            continue
                        for g in boxes:
                            v = iou_xyxy(init_box, np.asarray(g.xyxy, np.float32))
                            if v > best_iou:
                                best, best_iou = int(g.track_id), v
                    target_gt_id = best

        if started and status is None:
            status = tracker.step(frame, dets, frame_idx=idx)

        if status is not None:
            statuses[int(idx)] = status
            state = getattr(status.state, "value", str(status.state))
            tid = followed_track_id(status, dets)
            rows.append(
                {
                    "frame": int(idx),
                    "state": state,
                    "box": None if status.xyxy is None else [float(v) for v in status.xyxy],
                    "score": float(status.score or 0.0),
                    "frames_since_update": int(status.frames_since_update),
                    "age": int(status.age),
                    "reacquired": bool(status.reacquired),
                    "followed_track_id": tid,
                    "n_persons": len(dets),
                }
            )

        if args.save_video:
            elapsed = time.perf_counter() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0.0
            state = (
                getattr(status.state, "value", str(status.state)) if status is not None else "no-click-yet"
            )
            hud = [
                f"frame {idx}  {w}x{h}  {fps:5.1f} FPS",
                f"persons {len(dets):3d}   state {state}",
            ]
            if status is not None:
                hud.append(
                    f"age {int(status.age):4d}  fsu {int(status.frames_since_update):4d}"
                )
            canvas = draw_frame(frame, dets, status, hud)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out / "annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (w, h),
                )
            writer.write(canvas)

    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0

    (out / "target.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8"
    )
    events = [
        {"frame": int(e.frame_idx), "kind": e.kind, "detail": e.detail}
        for e in getattr(tracker, "events", [])
    ]
    (out / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    state_hist: dict[str, int] = {}
    for r in rows:
        state_hist[r["state"]] = state_hist.get(r["state"], 0) + 1

    followed_ids = [r["followed_track_id"] for r in rows if r["followed_track_id"] is not None]
    id_changes = sum(1 for a, b in zip(followed_ids, followed_ids[1:]) if a != b)

    summary = {
        "source": str(args.source),
        "frames": n_frames,
        "click": None if click is None else list(click),
        "click_frame": int(args.frame),
        "init_box": None if init_box is None else [float(v) for v in init_box],
        "init_track_id": init_track_id,
        "target_gt_id": target_gt_id,
        "wall_s": round(elapsed, 3),
        "pipeline_fps": round(fps, 2),
        "ms_per_frame": round(1000.0 / fps, 2) if fps > 0 else None,
        "state_histogram": state_hist,
        "n_events": len(events),
        "event_kinds": sorted({e["kind"] for e in events}),
        "followed_track_id_changes": int(id_changes),
        "config": {
            "tracker": cfg.mot.tracker,
            "distractor_margin": cfg.cost.distractor_margin,
            "target_parts": list(cfg.embedder.target_parts),
            "frozen_states": [getattr(s, "value", str(s)) for s in cfg.state.frozen_states],
            "with_reid": bool(cfg.mot.with_reid),
        },
        "embedder": {
            "mode": embedder.mode,
            "dim": embedder.dim,
            "provider": embedder.provider,
            "device": embedder.device,
        },
        "ort": {k: v for k, v in ort_info.items() if k != "preferred"},
    }

    # If we have ground truth, score the run with the same code path the
    # experiments use, so the demo and the benchmark cannot disagree.
    if seq is not None and seq.gt and target_gt_id is not None:
        try:
            from tgtrack.data import make_target_gt
            from tgtrack.evaluate import evaluate_target

            gt_target = make_target_gt(seq, target_gt_id)
            metrics = evaluate_target(
                gt_target, statuses, list(getattr(tracker, "events", [])), seq.gt, cfg.eval,
                debug_csv=out / "frames.csv",
            )
            summary["metrics"] = {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating, str))
            }
            (out / "metrics.json").write_text(
                json.dumps(summary["metrics"], indent=2), encoding="utf-8"
            )
        except Exception as exc:  # scoring is best-effort; never hide the tracking result
            summary["metrics_error"] = f"{type(exc).__name__}: {exc}"

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print("-" * 78)
        print(f"frames            : {n_frames}   ({fps:.1f} FPS, {1000.0/fps:.1f} ms/frame)"
              if fps > 0 else f"frames            : {n_frames}")
        print(f"click             : {click or 'auto (largest)'} at frame {args.frame}")
        print(f"initial box       : {init_box}")
        print(f"state histogram   : {state_hist}")
        print(f"events            : {len(events)} {sorted({e['kind'] for e in events})}")
        print(f"followed-id churn : {id_changes}")
        if "metrics" in summary:
            m = summary["metrics"]
            for key in (
                "target_success_rate",
                "target_precision",
                "loss_rate",
                "reacquire_rate",
                "reacquire_latency_mean",
                "false_acquisition_rate",
                "id_consistency",
            ):
                if key in m:
                    print(f"  {key:26s}: {m[key]}")
        if args.save_video:
            print(f"video             : {out / 'annotated.mp4'}")
        print(f"outputs           : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
