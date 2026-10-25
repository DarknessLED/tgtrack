"""Scratch: find out why the target tracker refuses the TRUE target under the
partial-view mask, and what it accepts instead.

Why this exists: the aggregate TSR/FAR numbers say a run failed but not why,
and `run_target.py`'s per-frame CSV records neither the refusal reason nor the
detections offered to the tracker.  Those two facts decide whether the failure
is "the detector stopped seeing the person", "the tracker refused a real
observation", or "the tracker accepted somebody else" -- three completely
different bugs with three completely different fixes.

This is READ-ONLY over `src/tgtrack/*` and `scripts/run_target.py`: it imports
their public helpers and replays `run_sequence`'s exact loop.

Usage:
    python scripts/probe_why_refuse.py [--mask upper] [--frames 25] [--seq 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tgtrack.config import Config  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402
from tgtrack.types import iou_matrix  # noqa: E402


def _load_run_target():
    """Import scripts/run_target.py by path (it is not an installed module)."""
    path = ROOT / "scripts" / "run_target.py"
    spec = importlib.util.spec_from_file_location("_probe_run_target", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves annotations through sys.modules[cls.__module__], so the
    # module must be registered BEFORE exec_module runs the @dataclass decorators.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fmt_box(b) -> str:
    if b is None:
        return "-"
    return "[" + ",".join(f"{float(v):7.1f}" for v in np.asarray(b).reshape(-1)[:4]) + "]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="upper")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip the ReID encoder; NOTE this charges the constant "
                         "missing_appearance_cost (0.5) x w_appearance (1.6) = 0.8 "
                         "against every candidate and is NOT what the grid does")
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--target-id", type=int, default=None)
    ap.add_argument("--strategy", default="stressed",
                    choices=["stressed", "longest", "most-occluded", "first"])
    args = ap.parse_args()

    rt = _load_run_target()
    cfg = Config()
    # Build the args through run_target's OWN parser so the probe sees exactly the
    # same defaults the grid does (a hand-made Namespace silently missed dry_run
    # and cost a run).
    argv = [
        "--data-root", str(ROOT / "data" / "raw" / "mot17" / "MOT17" / "train"),
        "--limit-sequences", str(args.seq + 1),
        "--max-frames", str(args.frames),
        "--subsample", str(max(1, args.subsample)),
        "--target-strategy", args.strategy,
    ]
    ns = rt.build_argparser().parse_args(argv)
    seqs = rt._load_sequences(ns)
    seq = seqs[args.seq]
    target_id = args.target_id if args.target_id is not None else rt.choose_target_id(seq, "stressed")
    print(f"sequence {seq.name}  target_id {target_id}  frames {len(seq)}")

    plan = rt.build_mask_plan(seq, int(target_id), args.mask, limit=args.frames)
    detector = rt.get_detector(cfg, "yolo")
    print(f"mask={args.mask}  detector={detector.info()}")

    embedder = None
    if not args.no_embed:
        embedder = rt.get_embedder(cfg, device=args.device)
        print("embedder:", rt._embedder_info(embedder))
    tracker = TargetTracker(cfg, embedder)
    first = plan.frames()[0]
    print(f"\n{'fr':>3} {'state':<14} {'fsu':>3} {'reason':<26} "
          f"{'reported':<34} {'gt_vis':<34} {'best_IoU':>8} {'n_det':>5}  dets(score|IoU_vis|IoU_full)")
    prev_state = ""
    for i in range(first, min(first + args.frames, len(seq))):
        raw = seq.frame(i)
        assert raw is not None
        frame = plan.apply(raw, i)
        assert frame is not None
        dets = list(detector.detect(frame, i))

        if i == first:
            status = tracker.initialize(frame, plan.gt_target[i], frame_idx=i)
        else:
            status = tracker.step(frame, dets, frame_idx=i)

        vis = plan.gt_target.get(i)
        full = plan.full_gt_target.get(i)
        if i == first:
            # The mask reduces the target's GROUND TRUTH to its visible rectangle.
            # If that rectangle is a sliver then the "partial view" experiment is
            # really an "invisible target" experiment, and the two must not be
            # reported with the same number.
            def _wh(b):
                return "-" if b is None else f"{float(b[2]) - float(b[0]):.0f}x{float(b[3]) - float(b[1]):.0f}"
            print(f"      click box {_fmt_box(vis)} ({_wh(vis)})   full-body gt {_fmt_box(full)} ({_wh(full)})")
        best = 0.0
        if vis is not None and dets:
            best = float(np.max(iou_matrix(np.asarray([vis], float),
                                           np.asarray([d.xyxy for d in dets], float))))
        elif vis is not None:
            best = float(np.max(np.asarray([d.xyxy for d in dets], float)[:, 0] * 0)) if dets else 0.0

        diag = status.diag or {}
        print(f"{i:>3} {str(status.state):<14} {status.frames_since_update:>3} "
              f"{str(diag.get('reason', diag.get('init', '-'))):<26} "
              f"{_fmt_box(status.xyxy):<34} {_fmt_box(vis):<34} {best:>8.3f} {len(dets):>5}")
        # Which detection actually covers the target, how big is it, and what did
        # the tracker think of it?  The tracker's box height vs the detection's box
        # height is the decisive number: the mask reduces the TARGET's ground truth
        # to its visible half while the detector keeps boxing the whole person.
        #
        # Only expand interesting frames: the first two (to show the seeding cost),
        # every refusal, every state change and the last frame.  Printing all 60
        # buries the one frame that matters.
        interesting = (
            i <= first + 1
            or not bool(diag.get("accepted", True))
            or str(status.state) != prev_state
            or i == min(first + args.frames, len(seq)) - 1
        )
        prev_state = str(status.state)
        if interesting and dets:
            arr = np.asarray([d.xyxy for d in dets], float)
            iv = iou_matrix(np.asarray([vis], float), arr)[0]
            order = np.argsort(-iv)[:3]
            parts = []
            for k in order:
                b = arr[k]
                parts.append(f"#{int(k)} s={dets[k].score:.2f} "
                             f"h={b[3] - b[1]:6.1f} IoUvis={iv[k]:.2f} "
                             f"IoUfull={float(iou_matrix(np.asarray([full], float), b[None, :])[0, 0]):.2f} "
                             f"{_fmt_box(b)}")
            print("      best-det:", " | ".join(parts))
        if interesting:
            print("      diag:", {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in diag.items() if k != "init"})

    print("\nevents:", [(e.kind, getattr(e, 'detail', None)) for e in tracker.events][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
