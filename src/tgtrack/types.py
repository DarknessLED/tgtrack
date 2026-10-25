"""Core data types — THE SHARED CONTRACT for the whole package.

Every module in `src/tgtrack/` must use these types. Do not redefine them.
`Detection` is produced by the detector, enriched by the appearance encoder,
tagged with `track_id` by the all-person MOT layer, and consumed by the
target tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Canonical names for part crops. Keyed by these strings everywhere.
PART_KEYS = ("full", "upper", "lower", "head", "torso")
# Vertical extents (y0_frac, y1_frac) of the person box used per part.
# Chosen so that `upper` covers a torso+head view, `lower` covers legs,
# `head` is a tight face/head region, `torso` is the chest band.
PART_Y_RANGES: dict[str, tuple[float, float]] = {
    "full": (0.00, 1.00),
    "upper": (0.00, 0.60),
    "lower": (0.45, 1.00),
    "head": (0.00, 0.28),
    "torso": (0.15, 0.55),
}
# Horizontal extents (x0_frac, x1_frac), typically inset for parts.
PART_X_RANGES: dict[str, tuple[float, float]] = {
    "full": (0.00, 1.00),
    "upper": (0.00, 1.00),
    "lower": (0.00, 1.00),
    "head": (0.15, 0.85),
    "torso": (0.05, 0.95),
}


@dataclass
class Detection:
    """One person detection in one frame, with optional appearance features.

    Coordinates are absolute pixels, `xyxy` order, in the ORIGINAL frame
    resolution (never letterboxed coordinates).
    """

    xyxy: np.ndarray
    score: float
    cls: int = 0
    #: L2-normalized global (full-body) embedding, or None if not encoded.
    feat: np.ndarray | None = None
    #: L2-normalized per-part embeddings keyed by PART_KEYS.
    part_feats: dict[str, np.ndarray] = field(default_factory=dict)
    #: Filled in by the all-person MOT layer.
    track_id: int | None = None
    #: Crop quality signals used by the long-term gallery.
    quality: float = 1.0
    sharpness: float = 0.0
    #: Fraction of the crop that is inside the frame (1.0 = fully inside).
    visible_ratio: float = 1.0

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(4)

    @property
    def xywh(self) -> np.ndarray:
        x0, y0, x1, y1 = self.xyxy
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dtype=np.float32)

    @property
    def wh(self) -> tuple[float, float]:
        return float(self.xyxy[2] - self.xyxy[0]), float(self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        w, h = self.wh
        return max(0.0, w) * max(0.0, h)

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2], dtype=np.float32
        )


@dataclass
class FrameMeta:
    """Per-frame context handed to the tracker."""

    frame_idx: int
    width: int
    height: int
    timestamp: float | None = None


@dataclass
class TargetStatus:
    """What the target tracker reports for one frame.

    `state` is one of TargetState values (see target_tracker.TargetState).
    `xyxy` is the target's box in this frame, or None when the tracker
    declines to report a location (e.g. hard lost).
    """

    frame_idx: int
    state: str
    xyxy: np.ndarray | None = None
    score: float = 0.0
    #: Consecutive frames since the last accepted observation.
    frames_since_update: int = 0
    #: Total frames since target initialization.
    age: int = 0
    #: Set when the reported box came from a re-acquisition, not continuous tracking.
    reacquired: bool = False
    #: Diagnostics: component costs for the accepted match this frame.
    diag: dict[str, Any] = field(default_factory=dict)

    @property
    def is_reporting(self) -> bool:
        return self.xyxy is not None


@dataclass
class TargetEvent:
    """A discrete event in the target's lifetime — used by the evaluator."""

    frame_idx: int
    kind: str  # 'init' | 'lost' | 'reacquire' | 'confirm' | 'out_of_frame' | 'reentered' | 'swap_risk'
    detail: dict[str, Any] = field(default_factory=dict)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        a: (N, 4) xyxy float array.
        b: (M, 4) xyxy float array.

    Returns:
        (N, M) float32 IoU matrix.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def cosine_distance_matrix(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos sim) between one query and a bank of features.

    Args:
        query: (D,) or (N, D) L2-normalized features.
        bank: (M, D) L2-normalized features.

    Returns:
        (M,) when query is 1-D, else (N, M), float32 in [0, 2].
    """
    q = np.asarray(query, dtype=np.float32)
    b = np.asarray(bank, dtype=np.float32)
    if b.size == 0:
        return np.zeros((0,), dtype=np.float32) if q.ndim == 1 else np.zeros((q.shape[0], 0), dtype=np.float32)
    squeeze = q.ndim == 1
    if squeeze:
        q = q[None, :]
    if b.ndim == 1:  # a single feature is a valid 1-entry bank
        b = b[None, :]
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    d = 1.0 - np.clip(qn @ bn.T, -1.0, 1.0)
    return d[0].astype(np.float32) if squeeze else d.astype(np.float32)


def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a safe denominator."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def clip_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp a box to image bounds (may produce a degenerate box)."""
    b = np.asarray(box, dtype=np.float32).copy()
    b[0] = np.clip(b[0], 0, width - 1)
    b[1] = np.clip(b[1], 0, height - 1)
    b[2] = np.clip(b[2], 0, width - 1)
    b[3] = np.clip(b[3], 0, height - 1)
    return b
