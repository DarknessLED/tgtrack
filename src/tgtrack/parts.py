"""Part crop extraction — the geometry half of partial-view robustness.

A person detector gives one axis-aligned box. When the target is only partly
visible (behind a desk, cut by the frame edge, occluded by another person) the
*full-body* appearance descriptor is useless, but the sub-region that IS visible
still carries identity: a jacket, a face, a pair of trousers.

This module turns one person box into the canonical set of part boxes
(``PART_KEYS`` in :mod:`tgtrack.types`), extracts the corresponding pixels, and
measures how trustworthy a crop is. The memory banks in
:mod:`tgtrack.memory` consume both.

Conventions used by every function here:
    * boxes are absolute-pixel ``xyxy`` float arrays in the ORIGINAL frame;
    * ``out_size`` is ``(width, height)`` — the OpenCV convention (cv2.resize),
      NOT the numpy/torch ``(height, width)`` convention;
    * a crop that falls partly outside the frame is *not* squeezed: the visible
      part is placed on a zero canvas at its correct offset so part geometry
      (leg length, head width) is preserved.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .types import PART_KEYS, PART_X_RANGES, PART_Y_RANGES, clip_xyxy

__all__ = [
    "PART_CROP_PARTS",
    "estimate_partial_state",
    "crop",
    "crop_quality",
    "part_boxes",
    "visible_ratio",
]

#: Default part set for :func:`part_boxes` (kept in sync with ``types.PART_KEYS``).
PART_CROP_PARTS: tuple[str, ...] = tuple(PART_KEYS)

#: Pixel area at which the area term of :func:`crop_quality` saturates (64x128).
_AREA_REF = 64.0 * 128.0
#: Laplacian variance at which the sharpness term saturates.
_SHARPNESS_REF = 250.0
#: Typical standing-person height/width ratio, used to estimate how much of the
#: person is inside the frame when the box is short or clipped.
_PERSON_ASPECT = 2.7


def _as_box(xyxy) -> np.ndarray | None:
    """Coerce to a finite (4,) float32 ``xyxy`` array, or None."""
    try:
        b = np.asarray(xyxy, dtype=np.float32).reshape(4)
    except (ValueError, TypeError):
        return None
    if not np.all(np.isfinite(b)):
        return None
    return b


def _check_parts(parts: Sequence[str]) -> tuple[str, ...]:
    parts = tuple(parts)
    bad = [p for p in parts if p not in PART_Y_RANGES]
    if bad:
        raise KeyError(f"unknown part key(s) {bad}; valid keys: {sorted(PART_Y_RANGES)}")
    if not parts:
        raise ValueError("`parts` must contain at least one part key")
    return parts


def part_boxes(
    xyxy: np.ndarray,
    parts: Sequence[str] = PART_CROP_PARTS,
    frame_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Absolute ``xyxy`` boxes for each requested part of one person box.

    The part extents come from ``PART_Y_RANGES`` / ``PART_X_RANGES`` in
    :mod:`tgtrack.types`, applied as fractions of the person box, so the result
    is independent of image resolution.

    Args:
        xyxy: Person box, absolute pixels, ``(x0, y0, x1, y1)``.
        parts: Part keys to compute. Must be a subset of ``PART_Y_RANGES``.
        frame_size: Optional ``(width, height)``; when given, every part box is
            clamped to the image bounds with :func:`tgtrack.types.clip_xyxy`.

    Returns:
        ``{part_key: (4,) float32 xyxy}`` — same iteration order as ``parts``.

    Raises:
        KeyError: An unknown part key was requested.
        ValueError: ``parts`` is empty, or ``xyxy`` is not a finite 4-vector.
    """
    parts = _check_parts(parts)
    b = _as_box(xyxy)
    if b is None:
        raise ValueError(f"`xyxy` must be a finite 4-vector, got {xyxy!r}")
    x0, y0, x1, y1 = (float(v) for v in b)
    w = x1 - x0
    h = y1 - y0

    out: dict[str, np.ndarray] = {}
    for key in parts:
        fy0, fy1 = PART_Y_RANGES[key]
        fx0, fx1 = PART_X_RANGES[key]
        box = np.array(
            [x0 + fx0 * w, y0 + fy0 * h, x0 + fx1 * w, y0 + fy1 * h],
            dtype=np.float32,
        )
        if frame_size is not None:
            fw, fh = int(frame_size[0]), int(frame_size[1])
            box = clip_xyxy(box, fw, fh)
        out[key] = box
    return out


def visible_ratio(box_xyxy: np.ndarray, width: int, height: int) -> float:
    """Fraction of the box area that lies inside a ``width`` x ``height`` frame.

    Returns 1.0 for a box fully inside, 0.0 for a box fully outside (or a
    degenerate box). The box area is the *unclipped* area, so a box extending
    past the frame edge yields < 1.
    """
    b = _as_box(box_xyxy)
    if b is None or width <= 0 or height <= 0:
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in b)
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return 0.0
    iw = max(0.0, min(x1, float(width)) - max(x0, 0.0))
    ih = max(0.0, min(y1, float(height)) - max(y0, 0.0))
    return float(np.clip((iw * ih) / (bw * bh), 0.0, 1.0))


def crop(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    out_size: tuple[int, int] | None = None,
    pad: float = 0.0,
) -> np.ndarray | None:
    """Crop one region, preserving geometry when the box leaves the frame.

    The box is optionally expanded by ``pad`` (a fraction of its width/height,
    applied on every side), then the visible part is copied onto a zero canvas
    of the box's full extent at the correct offset. A box hanging off the left
    edge therefore produces a crop with black pixels on the left — *not* a
    horizontally squeezed person, which would destroy part proportions.

    Args:
        img_bgr: ``(H, W, C)`` image (any dtype; the result keeps the dtype).
        box_xyxy: Absolute-pixel ``xyxy`` box.
        out_size: Optional ``(width, height)``; resized with ``INTER_LINEAR``.
        pad: Fractional padding added on each side (0.1 = +10 % of w and h).

    Returns:
        The crop, or None when the box is degenerate, non-finite, entirely
        outside the frame, or ``out_size`` is degenerate.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim not in (2, 3):
        return None
    ih, iw = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    if ih <= 0 or iw <= 0:
        return None
    b = _as_box(box_xyxy)
    if b is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in b)
    if x1 <= x0 or y1 <= y0:
        return None
    if pad:
        px = float(pad) * (x1 - x0)
        py = float(pad) * (y1 - y0)
        x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py

    # Full canvas extent (the box may stick out of the frame).
    cx0, cy0 = int(np.floor(x0)), int(np.floor(y0))
    cx1, cy1 = int(np.ceil(x1)), int(np.ceil(y1))
    cw, ch = cx1 - cx0, cy1 - cy0
    if cw <= 0 or ch <= 0:
        return None
    # Visible intersection.
    sx0, sy0 = max(cx0, 0), max(cy0, 0)
    sx1, sy1 = min(cx1, iw), min(cy1, ih)
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    patch = img_bgr[sy0:sy1, sx0:sx1]
    if patch.size == 0:
        return None
    chan = (img_bgr.shape[2],) if img_bgr.ndim == 3 else ()
    canvas = np.zeros((ch, cw) + chan, dtype=img_bgr.dtype)
    dx, dy = sx0 - cx0, sy0 - cy0
    canvas[dy : dy + patch.shape[0], dx : dx + patch.shape[1]] = patch

    if out_size is not None:
        ow, oh = int(out_size[0]), int(out_size[1])
        if ow <= 0 or oh <= 0:
            return None
        canvas = cv2.resize(canvas, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas)


def crop_quality(img_bgr: np.ndarray, box_xyxy: np.ndarray) -> tuple[float, float]:
    """Trustworthiness of a crop: ``(quality in [0,1], sharpness)``.

    ``sharpness`` is the variance of the Laplacian (motion blur and
    out-of-focus crops score low). ``quality`` is the geometric mean of a
    saturating area term (real pixels only, so zero padding does not inflate it)
    and a saturating sharpness term, hence 0 when either is 0.

    The detection score is deliberately *not* folded in — the caller multiplies
    it in, so detection confidence and crop quality stay separable.

    Args:
        img_bgr: ``(H, W, C)`` image.
        box_xyxy: Absolute-pixel ``xyxy`` box.

    Returns:
        ``(quality, sharpness)``; ``(0.0, 0.0)`` for a degenerate box or an
        empty intersection with the frame.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim not in (2, 3):
        return 0.0, 0.0
    ih, iw = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    if ih <= 0 or iw <= 0:
        return 0.0, 0.0
    b = _as_box(box_xyxy)
    if b is None:
        return 0.0, 0.0
    sx0 = max(int(np.floor(float(b[0]))), 0)
    sy0 = max(int(np.floor(float(b[1]))), 0)
    sx1 = min(int(np.ceil(float(b[2]))), iw)
    sy1 = min(int(np.ceil(float(b[3]))), ih)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0.0, 0.0
    patch = img_bgr[sy0:sy1, sx0:sx1]
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    gray = np.asarray(gray, dtype=np.float32)
    if gray.size == 0:
        return 0.0, 0.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if not np.isfinite(sharpness) or sharpness < 0:
        sharpness = 0.0
    area_score = float(np.clip((sx1 - sx0) * (sy1 - sy0) / _AREA_REF, 0.0, 1.0))
    sharp_score = float(np.clip(sharpness / _SHARPNESS_REF, 0.0, 1.0))
    quality = float(np.sqrt(area_score * sharp_score))
    return quality, sharpness


def estimate_partial_state(
    xyxy: np.ndarray,
    width: int,
    height: int,
    frame_edges: float | Sequence[float] = 2.0,
) -> dict:
    """Infer whether a person box is truncated by the frame, and how.

    Args:
        xyxy: Person box, absolute pixels. It MAY extend beyond the frame.
        width, height: Frame size in pixels.
        frame_edges: Edge margin in pixels — a box within this margin of an edge
            counts as clipped. Either one float applied to all four sides, or a
            ``(left, top, right, bottom)`` sequence.

    Returns:
        dict with:
            ``truncated`` (bool): any side is clipped.
            ``sides`` (list[str]): subset of ``("top", "bottom", "left",
                "right")``, always in that order.
            ``kind`` (str): ``"none"``, a single side name, or ``"corner"`` when
                both a vertical (top/bottom) and a horizontal (left/right) side
                are clipped. With top AND bottom clipped the side with the
                larger overflow wins.
            ``est_visible_frac`` (float in [0,1]): visible box area divided by
                the area a standing person of this width would occupy
                (``_PERSON_ASPECT * w``); 0.5 therefore means "about half the
                body is in view". 1.0 when the box is at least a full body tall
                and fully inside.
            ``upper_only`` (bool): likely only the upper body is visible
                (bottom clipped, or a short box in the top half of the frame).
            ``lower_only`` (bool): likely only the lower body is visible.
            ``visible`` (tuple[float, float, float, float]): the clipped box.
    """
    if isinstance(frame_edges, (int, float, np.floating)):
        ml = mt = mr = mb = float(frame_edges)
    else:
        vals = [float(v) for v in frame_edges]
        if len(vals) != 4:
            raise ValueError("`frame_edges` must be a float or a 4-sequence (left, top, right, bottom)")
        ml, mt, mr, mb = vals

    b = _as_box(xyxy)
    fw, fh = float(width), float(height)
    if b is None or fw <= 0 or fh <= 0:
        return {
            "truncated": False,
            "sides": [],
            "kind": "none",
            "est_visible_frac": 0.0,
            "upper_only": False,
            "lower_only": False,
            "visible": (0.0, 0.0, 0.0, 0.0),
        }
    x0, y0, x1, y1 = (float(v) for v in b)

    overflow = {
        "top": mt - y0,  # >0 => the box crosses (or touches) the top edge
        "bottom": y1 - (fh - 1.0 - mb),
        "left": ml - x0,
        "right": x1 - (fw - 1.0 - mr),
    }
    sides = [s for s in ("top", "bottom", "left", "right") if overflow[s] > 0.0]

    vert = [s for s in ("top", "bottom") if s in sides]
    horz = [s for s in ("left", "right") if s in sides]
    if not sides:
        kind = "none"
    elif vert and horz:
        kind = "corner"
    elif len(vert) > 1:
        kind = max(vert, key=lambda s: overflow[s])
    elif vert:
        kind = vert[0]
    else:
        kind = horz[0]

    bw = x1 - x0
    bh = y1 - y0
    vx0, vy0 = max(x0, 0.0), max(y0, 0.0)
    vx1, vy1 = min(x1, fw), min(y1, fh)
    visible = (vx0, vy0, max(vx0, vx1), max(vy0, vy1))
    vis_area = max(0.0, vx1 - vx0) * max(0.0, vy1 - vy0)
    expected = _PERSON_ASPECT * max(bw, 1e-6) ** 2
    est = float(np.clip(vis_area / expected, 0.0, 1.0))

    # A short box (h/w well below a standing person) is a partial body view.
    short = bh < 1.8 * max(bw, 1e-6)
    cy = 0.5 * (y0 + y1)
    upper_only = ("bottom" in sides) or (short and cy < 0.5 * fh)
    lower_only = ("top" in sides) or (short and cy >= 0.5 * fh)
    if upper_only and lower_only:  # box spans the whole frame height: both ends cut
        upper_only = "top" not in sides
        lower_only = not upper_only

    return {
        "truncated": bool(sides),
        "sides": sides,
        "kind": kind,
        "est_visible_frac": est,
        "upper_only": bool(upper_only),
        "lower_only": bool(lower_only),
        "visible": visible,
    }
