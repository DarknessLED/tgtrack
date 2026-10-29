"""E095 -- the second decision channel: bounded-lag refutation.

The identity layer's four measured mechanisms (E082 contrast vote, E089 widened
lost ceiling, E092/E092b temporal hold, E093 stack) all lie on one Pareto surface
with a ~1:1 TSR/FAR exchange rate, because :class:`TargetTracker` commits one
identity per frame from one candidate set and that commit is final: a wrong
commit is a false acquisition, a refusal is a miss, and both are scored against
the same visible frames.

This module implements the one structural way out that does not re-weight
within-frame evidence: after a sequence is tracked, the reports are re-examined
with the evidence the online decision did not have -- the frames that came
*after* it.  A report that is an isolated excursion (the tracker jumped to a
different candidate for one or two frames and jumped back) is re-attributed to
the candidate that the surrounding window agrees on, when that candidate exists
in the frame.

What it may NOT do:

* delete a report (that is the metric gaming the ``blind & present`` share
  exists to expose, and it is what makes a FAR win look free);
* touch a frame that reported nothing (a miss stays a miss);
* rewrite a frame whose report already agrees with the window consensus;
* run at all when ``state.decision_lag == 0`` (the shipped path is then
  bit-identical by construction -- :func:`refute_statuses` is not called).

The lag is a *delivery* latency, not a memory of the answer key: the pass only
ever reads candidate boxes the tracker itself produced in those frames.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

__all__ = ["refute_statuses", "consensus_index"]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    inter = ix * iy
    area = max((ax2 - ax1) * (ay2 - ay1), 1e-9) + max((bx2 - bx1) * (by2 - by1), 1e-9)
    return float(inter / (area - inter + 1e-9))


def _centre(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0], float
    )


def _best_index(box: np.ndarray, cands: list[list[float]]) -> int | None:
    """Index of the candidate that best explains ``box`` (IoU first, then distance)."""
    if not cands:
        return None
    best, best_key = None, None
    c = _centre(box)
    for i, cand in enumerate(cands):
        arr = np.asarray(cand, float)
        iou = _iou(box, arr)
        dist = float(np.linalg.norm(_centre(arr) - c))
        key = (-iou, dist)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def consensus_index(
    frames: list[int],
    candidates: dict[int, list[list[float]]],
    reports: dict[int, np.ndarray],
    i: int,
    window: int,
    min_share: float = 0.5,
) -> int | None:
    """Candidate index the window around position ``i`` mostly agrees on.

    Position ``i`` is skipped, so a single-frame excursion cannot vote for itself.
    """
    n = len(frames)
    lo, hi = max(0, i - window), min(n, i + window + 1)
    votes: dict[int, int] = {}
    for j in range(lo, hi):
        if j == i:
            continue
        frame = frames[j]
        box = reports.get(frame)
        if box is None:
            continue
        idx = _best_index(box, candidates.get(frame) or [])
        if idx is None:
            continue
        votes[idx] = votes.get(idx, 0) + 1
    if not votes:
        return None
    total = sum(votes.values())
    idx, count = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
    if count / total < min_share:
        return None
    return idx


def refute_statuses(
    statuses: dict,
    candidates: dict[int, list[list[float]]],
    window: int = 8,
    min_share: float = 0.5,
) -> dict:
    """Return a revised status mapping (never deletes reports, never edits misses).

    Args:
        statuses: ``{frame_idx: TargetStatus}`` as produced by the tracker.
        candidates: ``{frame_idx: [[x1, y1, x2, y2], ...]}`` -- the candidate boxes
            the tracker saw in that frame (``TargetTracker.refute_candidates()``).
        window: half-width of the evidence window in frames.
        min_share: fraction of the window that must agree on a candidate before an
            isolated excursion is re-attributed.

    Returns:
        A new mapping. Frames whose report already agrees with the window consensus,
        frames with no report, and frames without candidates are passed through
        unchanged (the same object), so a sequence the pass does nothing on is
        byte-identical to the input.
    """
    if not candidates or not statuses or int(window) <= 0:
        return statuses
    frames = sorted(statuses)
    reports: dict[int, np.ndarray] = {}
    for frame in frames:
        st = statuses[frame]
        if st.xyxy is not None:
            reports[frame] = np.asarray(st.xyxy, float).ravel()[:4]
    if not reports:
        return statuses
    runs: list[list[int]] = []
    cur: list[int] = []
    for frame in frames:
        if frame in reports:
            cur.append(frame)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    out = dict(statuses)
    revised = 0
    for run in runs:
        if len(run) < 3:
            continue
        tip = {frame: _best_index(reports[frame], candidates.get(frame) or []) for frame in run}
        for i, frame in enumerate(run):
            own = tip.get(frame)
            cons = consensus_index(run, candidates, reports, i, int(window), float(min_share))
            if cons is None or own is None or cons == own:
                continue
            cands = candidates.get(frame) or []
            if cons >= len(cands):
                continue
            new_box = np.asarray(cands[cons], float)
            if _iou(new_box, reports[frame]) >= 0.5:
                continue  # already the same person; nothing to fix
            # the revision must actually reduce the jump to the neighbours that stay
            jump_old = jump_new = 0.0
            for step in (-1, 1):
                j = i + step
                if 0 <= j < len(run):
                    nb = reports[run[j]]
                    jump_old += float(np.linalg.norm(_centre(reports[frame]) - _centre(nb)))
                    jump_new += float(np.linalg.norm(_centre(new_box) - _centre(nb)))
            if jump_new >= jump_old:
                continue
            out[frame] = replace(
                statuses[frame],
                xyxy=new_box.copy(),
                diag={**dict(statuses[frame].diag or {}), "refuted_from": reports[frame].tolist()},
            )
            revised += 1
    if revised == 0:
        return statuses
    return out
