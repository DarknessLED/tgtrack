"""Which MOT17 identities stay VISIBLE enough to be detectable after masking?

`scripts/run_target.py` chooses the target identity from track statistics alone
(`choose_target_id`: stressed / longest / most-occluded / first).  None of those
look at whether the target is still DETECTABLE once the mask has blacked out half
of it.  On MOT17-02-FRCNN with `--frames 60 --subsample 2` the harness picks id 17,
whose visible rectangle collapses to 38x54 px -- no detection overlaps it on ANY
frame -- so that run measures an *invisible* target, not partial-view ReID.

This scratch tool ranks identities by the size of their post-mask visible box so a
masked experiment can be pointed at a target that is genuinely still observable.

Read-only.  Writes nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_scratch(name: str, path: Path):
    """Import a scripts/*.py module by path.

    The module must be registered in sys.modules BEFORE exec_module: its module
    level dataclasses resolve their annotations through sys.modules[__module__],
    and without the registration that lookup returns None and blows up with
    AttributeError: 'NoneType' object has no attribute '__dict__'.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=r"data\raw\mot17\MOT17\train\MOT17-02-FRCNN")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--min-presence", type=float, default=0.5,
                    help="fraction of frames an id must appear in to be a candidate")
    args = ap.parse_args()

    import tgtrack

    rt = load_scratch("rt_probe", ROOT / "scripts" / "run_target.py")

    seq_dir = ROOT / args.sequence
    seq = tgtrack.load_mot_sequence(
        str(seq_dir), subsample=args.subsample, max_frames=args.max_frames
    )
    n = len(seq)
    print(f"sequence {seq.name}  frames {n}  subsample {args.subsample}  {seq.width}x{seq.height}")

    # presence map from the raw GT
    presence: dict[int, int] = {}
    for fi in range(n):
        for b in seq.gt_at(fi) or []:
            presence[int(b.track_id)] = presence.get(int(b.track_id), 0) + 1
    candidates = sorted(
        (i for i, c in presence.items() if c >= args.min_presence * n),
        key=lambda i: -presence[i],
    )
    print(f"candidate identities present in >= {args.min_presence:.0%} of frames: {candidates}")

    for mode in ("none", "upper", "lower"):
        print(f"\n=== mask={mode} ===")
        print(f"{'id':>6} {'seen':>5} {'med h':>6} {'med w':>6} {'med area':>9} {'frames w/ vis':>13}")
        rows = []
        for tid in candidates:
            plan = rt.build_mask_plan(seq, tid, mode=mode, limit=n)
            hs, ws, areas, nvis = [], [], [], 0
            for fi in range(n):
                box = plan.gt_target.get(fi)
                if box is None:
                    continue
                if plan.full_gt_target.get(fi) is None:
                    continue
                w = float(box[2]) - float(box[0])
                h = float(box[3]) - float(box[1])
                if w <= 0 or h <= 0:
                    continue
                nvis += 1
                hs.append(h)
                ws.append(w)
                areas.append(w * h)
            if not areas:
                continue
            rows.append((statistics.median(areas), tid, presence[tid], statistics.median(hs),
                         statistics.median(ws), nvis))
        for area, tid, seen, mh, mw, nvis in sorted(rows, reverse=True):
            print(f"{tid:>6} {seen:>5} {mh:>6.0f} {mw:>6.0f} {area:>9.0f} {nvis:>13}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
