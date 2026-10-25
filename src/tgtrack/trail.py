"""Motion-signature ("trail") evidence for crowded single-target tracking.

Why this exists
---------------
On DanceTrack the appearance channel is nearly useless for telling two dancers
apart: the pairwise AUC of the YOLO26-ReID descriptor is 0.43-0.58 there
(`probe_embed_metric.py`, E013/E017), and every attempt to squeeze a decision out
of "distance to the target's own past" failed (E019-E032).  The literature says
what to do when appearance collapses: use motion/gait, which beats appearance for
long-term re-identification (GAE-Net, *Neural Networks* 2025; gait+HAR
multi-year ReID, *Pattern Recognition* 2025).

The transferable part is not a network.  It is a descriptor of how a person
moves: gait work uses silhouette periodicity, and here the cheap analogue is the
*shape of the trajectory* -- per-step turning angle plus speed normalised by body
height.  That descriptor is invariant to where the person is, which way they
face, and how fast they walk, which is exactly what survives being compared
between a target we lost and a stranger we are considering.

It is deliberately NOT a learned feature: no fine-tuning is allowed, and the
experiment is meant to answer "is there a second, orthogonal evidence channel at
all", so a 16-number analytic descriptor is the cleanest possible probe.  If the
channel carries signal, a learned one can replace it later.

Design notes
------------
* ``TrailLinker`` links detections across frames by IoU.  The target tracker is
  fed detections only (the all-person MOT layer is a separate product), so the
  short tubes have to be reconstructed here.  It is a greedy one-frame matcher,
  which is enough for 8-16 frame tubes and costs microseconds.
* ``descriptor()`` returns turning angles and log speeds, so distances are
  scale-free.  Missing history returns ``None`` and every caller treats that as
  "no evidence", never as "bad".
* ``trail_distance()`` is a plain mean absolute difference.  With ~2x8 numbers a
  fitted metric would be noise-fitting; the experiment asks whether the *raw*
  signal separates identities at all.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .types import Detection, iou_matrix

__all__ = [
    "TrailLinker",
    "trail_descriptor",
    "trail_distance",
    "box_trail",
    "velocity_disagreement",
]


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def trail_descriptor(points: np.ndarray, steps: int = 8) -> np.ndarray | None:
    """Turning angles + log speeds over the last ``steps`` frames.

    Parameters
    ----------
    points
        ``(N, 3)`` array of ``(cx, cy, h)`` ordered oldest -> newest.
    steps
        How many displacement steps to describe.  Fewer available -> ``None``.

    Returns
    -------
    ``(2 * steps,)`` float32 vector: ``[log1p(speed)..., turning/pi...]``, or
    ``None`` when the trail is too short or degenerate (zero height).
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < steps + 1:
        return None
    p = p[-(steps + 1):]
    heights = p[1:, 2]
    if not np.all(np.isfinite(p)) or float(np.min(heights)) <= 0.0:
        return None
    d = np.diff(p[:, :2], axis=0)
    speed = np.linalg.norm(d, axis=1) / heights
    if not np.all(np.isfinite(speed)):
        return None
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = _wrap_angle(np.diff(ang)) if len(ang) > 1 else np.zeros(1)
    if len(turn) < steps:  # pad the leading turning angles with zero
        turn = np.concatenate([np.zeros(steps - len(turn)), turn])
    turn = turn[-steps:]
    out = np.concatenate([np.log1p(speed), turn / np.pi]).astype(np.float32)
    return out if np.all(np.isfinite(out)) else None


def trail_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean absolute difference between two descriptors (inf when either is absent)."""
    if a is None or b is None:
        return float("inf")
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    return float(np.mean(np.abs(a - b)))


def box_trail(boxes: Sequence[np.ndarray]) -> np.ndarray | None:
    """Turn an ascending sequence of ``xyxy`` boxes into ``(N, 3)`` ``(cx, cy, h)``."""
    if not boxes:
        return None
    out = np.zeros((len(boxes), 3), dtype=np.float64)
    for i, b in enumerate(boxes):
        bb = np.asarray(b, dtype=np.float64).ravel()[:4]
        out[i, 0] = 0.5 * (bb[0] + bb[2])
        out[i, 1] = 0.5 * (bb[1] + bb[3])
        out[i, 2] = max(bb[3] - bb[1], 1e-6)
    return out


class TrailLinker:
    """Greedy IoU tracker that keeps a short centre/height history per detection."""

    def __init__(self, max_len: int = 16, min_len: int = 8, iou_link: float = 0.25) -> None:
        self.max_len = max(2, int(max_len))
        self.min_len = max(2, int(min_len))
        self.iou_link = float(iou_link)
        self._tracks: list[dict[str, Any]] = []
        self._frame = -1
        # E061: per-detection tube velocity, aligned to the frame last passed to
        # update() (mean of the last few centre deltas, normalised by height).
        self.vels: list[np.ndarray | None] = []

    def reset(self) -> None:
        self._tracks = []
        self._frame = -1
        self.vels = []

    def __len__(self) -> int:
        return len(self._tracks)

    def update(self, dets: Sequence[Detection], frame_idx: int) -> list[np.ndarray | None]:
        """Advance one frame and return one descriptor (or ``None``) per detection."""
        boxes = [np.asarray(d.xyxy, dtype=np.float64).ravel()[:4] for d in dets]
        n = len(boxes)
        assign = [-1] * n
        # Tracks that survived a gap of more than one frame are stale: the tube
        # must be contiguous or the turning angles are meaningless.
        if self._frame >= 0 and frame_idx - self._frame > 1:
            self._tracks = []
        if self._tracks and n:
            prev = np.asarray([t["box"] for t in self._tracks], dtype=np.float64)
            cur = np.asarray(boxes, dtype=np.float64)
            m = iou_matrix(prev, cur)
            pairs = sorted(
                ((m[i, j], i, j) for i in range(m.shape[0]) for j in range(m.shape[1])),
                key=lambda t: -t[0],
            )
            taken_t: set[int] = set()
            for iou, i, j in pairs:
                if iou < self.iou_link:
                    break
                if i in taken_t or assign[j] >= 0:
                    continue
                assign[j] = i
                taken_t.add(i)
        kept: list[dict[str, Any]] = []
        out: list[np.ndarray | None] = [None] * n
        for j, box in enumerate(boxes):
            bb = np.asarray(box, dtype=np.float64)
            cx = 0.5 * (bb[0] + bb[2])
            cy = 0.5 * (bb[1] + bb[3])
            h = max(bb[3] - bb[1], 1e-6)
            ti = assign[j]
            if ti >= 0:
                tr = self._tracks[ti]
                tr["hist"].append((cx, cy, h))
                del tr["hist"][: -self.max_len]
            else:
                tr = {"hist": [(cx, cy, h)], "box": bb}
                self._tracks.append(tr)
            tr["box"] = bb
            kept.append(tr)
            if len(tr["hist"]) >= self.min_len + 1:
                out[j] = trail_descriptor(np.asarray(tr["hist"], dtype=np.float64), steps=min(self.min_len, len(tr["hist"]) - 1))
        self._tracks = kept
        self._frame = int(frame_idx)
        # E061: velocity of each tube over its last few steps, normalised by height
        # so it is directly comparable with ``MotionModel.velocity()``.
        vels: list[np.ndarray | None] = []
        for tr in kept:
            hist = np.asarray(tr["hist"], dtype=np.float64)
            if len(hist) >= 3:
                d = np.diff(hist[-3:, :2], axis=0) / max(hist[-1, 2], 1e-6)
                v = np.mean(d, axis=0)
                vels.append(v if np.all(np.isfinite(v)) else None)
            elif len(hist) == 2:
                d = (hist[1, :2] - hist[0, :2]) / max(hist[1, 2], 1e-6)
                vels.append(d if np.all(np.isfinite(d)) else None)
            else:
                vels.append(None)
        self.vels = vels
        return out


def velocity_disagreement(
    tube_v: np.ndarray | None, target_v: np.ndarray | None, min_speed: float = 0.004
) -> float:
    """How much a tube's heading disagrees with the target's, in ``[0, 1]``.

    ``0.0`` = same direction (or no usable evidence, which must never punish a
    candidate), ``1.0`` = exactly opposite.  Both vectors are normalised by body
    height per frame, so the magnitudes are comparable; a heading is only trusted
    above ``min_speed`` because the angle of a nearly stationary tube is noise.

    This is the E061 decision rule on its own so it can be tested exactly: the
    end-to-end behaviour depends on which box the local tube linker attached to a
    detection, which is not something a unit test can pin down honestly.
    """
    if tube_v is None or target_v is None:
        return 0.0
    a = np.asarray(tube_v, dtype=np.float64).ravel()[:2]
    b = np.asarray(target_v, dtype=np.float64).ravel()[:2]
    if a.size < 2 or b.size < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    sa = float(np.hypot(a[0], a[1]))
    sb = float(np.hypot(b[0], b[1]))
    if sa <= min_speed or sb <= min_speed:
        return 0.0
    cos = float(np.dot(a, b) / (sa * sb))
    return float(0.5 * (1.0 - max(-1.0, min(1.0, cos))))
