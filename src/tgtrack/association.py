"""Association for the *selected target*.

This is the decision core that answers one question every frame:

    "among the persons detected in this frame, which one (if any) is my target?"

It is deliberately decoupled from the all-person MOT layer. Design choices, and
why they matter for the DJI-ActiveTrack use case:

1. **Multi-cue cost.** Motion (IoU + adaptive center distance + Mahalanobis),
   appearance (short-term EMA blended with a long-term part-aware gallery),
   direction consistency (OC-SORT-style momentum), and scale. Every term has a
   configurable weight so the strategy itself is what gets optimized.

2. **Part-aware appearance matching.** A query crop that only contains an upper
   body is compared against *every* gallery part bank, and the part-to-part
   affinity is derived from the vertical overlap of the part definitions. So an
   upper-body-only observation naturally prefers the `upper` bank, and is only
   mildly penalized against the `full` bank. This is the mechanism that lets a
   half-visible target stay locked.

3. **Distractor margin.** Even a low cost is rejected if the runner-up is nearly
   as good. Sensor-free consumer trackers fail by latching onto a
   similar-looking bystander; requiring a decisive margin is the cheapest
   defence and it is measured (`false_acquisition_rate`).

4. **Hard negatives.** Confirmed distractors push up the cost of candidates that
   look like them.

All functions are pure NumPy and side-effect free, so they can be unit-tested
without torch, a GPU, or a video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .config import CostCfg
from .types import PART_X_RANGES, PART_Y_RANGES, Detection, cosine_distance_matrix, iou_matrix

# --------------------------------------------------------------------------- #
# Part-to-part affinity
# --------------------------------------------------------------------------- #


def _range_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """1-D IoU of two (lo, hi) intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 1e-9 else 0.0


def build_part_affinity(part_keys: Sequence[str] | None = None) -> np.ndarray:
    """Affinity in [0, 1] between every pair of part banks.

    Derived from the vertical overlap of `PART_Y_RANGES` (and the horizontal
    overlap of `PART_X_RANGES`), so it stays consistent if the part definitions
    are ever changed. Identical parts have affinity 1.0; `upper` vs `lower`
    has low affinity.

    Returns:
        (P, P) float32 matrix; rows = query part, cols = gallery part.
    """
    keys = list(part_keys) if part_keys is not None else list(PART_Y_RANGES)
    n = len(keys)
    out = np.zeros((n, n), dtype=np.float32)
    for i, j in product(range(n), repeat=2):
        ky, kg = keys[i], keys[j]
        out[i, j] = _range_iou(PART_Y_RANGES[ky], PART_Y_RANGES[kg]) * _range_iou(
            PART_X_RANGES[ky], PART_X_RANGES[kg]
        )
    return out


#: Default affinity over `PART_KEYS` order.
PART_AFFINITY: np.ndarray = build_part_affinity()


def part_affinity(query_part: str, gallery_part: str) -> float:
    """Scalar affinity between two part keys (1.0 for identical parts)."""
    if query_part == gallery_part:
        return 1.0
    if query_part not in PART_Y_RANGES or gallery_part not in PART_Y_RANGES:
        return 0.0
    return _range_iou(PART_Y_RANGES[query_part], PART_Y_RANGES[gallery_part]) * _range_iou(
        PART_X_RANGES[query_part], PART_X_RANGES[gallery_part]
    )


# --------------------------------------------------------------------------- #
# Cost breakdown
# --------------------------------------------------------------------------- #


@dataclass
class CostBreakdown:
    """Per-detection cost terms. All arrays are (N,) over the candidate list."""

    motion: np.ndarray
    appearance: np.ndarray
    direction: np.ndarray
    size: np.ndarray
    total: np.ndarray
    #: Boolean: appearance was actually measurable for this detection.
    appearance_valid: np.ndarray
    #: Boolean gate: candidate is plausible at all (motion gate + spatial radius).
    gate: np.ndarray
    #: Raw IoU with the motion prediction, for diagnostics/acceptance.
    iou: np.ndarray
    #: Mahalanobis distance (inf when no motion model was supplied).
    maha: np.ndarray
    #: Distractor penalty actually applied from the hard-negative bank.
    hard_negative: np.ndarray
    #: Per-detection winning (query_part, gallery_part) pair, for diagnostics.
    winning_parts: list[tuple[str, str] | None] = field(default_factory=list)
    #: Why a candidate was excluded, per index ("" when admissible).
    excluded: list[str] = field(default_factory=list)
    #: E077 -- height-normalised centre distance ``|det_c - pred_c| / gate_radius``
    #: (the same normalisation ``motion_cost`` applies to its distance term).
    #: ``+inf`` when the motion prediction carries no information this frame
    #: (``apply_motion`` False: the global-search path and every blind state), so
    #: the geometry channel can never carry a decision where geometry is not
    #: evidence -- that would be E022's latch, re-opened.
    norm_centre: np.ndarray | None = None
    #: E077 -- ``log(det_h / pred_h)`` per candidate. ``+inf`` under the same
    #: condition as ``norm_centre``.
    log_h_ratio: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.total.shape[0])


@dataclass
class MatchResult:
    """Outcome of `select_candidate`."""

    index: int | None
    cost: float
    accepted: bool
    reason: str
    margin: float = 0.0
    diag: dict[str, Any] = field(default_factory=dict)
    #: E077 -- which conclusive channel admitted this candidate on the OR-branch:
    #: ``""`` (the additive bound was satisfied, i.e. no new behaviour), or
    #: ``"geometry"`` / ``"appearance"``.
    channel: str = ""


# --------------------------------------------------------------------------- #
# Individual cost terms
# --------------------------------------------------------------------------- #


def motion_cost(pred_xyxy: np.ndarray, det_boxes: np.ndarray, gate_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Blend box-overlap and gate-normalized center distance.

    IoU alone collapses to zero under fast motion (no overlap => no gradient),
    so a normalized center distance is mixed in. That is what keeps a sprinting
    target associable.

    Returns:
        (cost, iou) both (N,) float32; cost in [0, 1].
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    if det_boxes.shape[0] == 0:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    iou = iou_matrix(pred_xyxy.reshape(1, 4), det_boxes)[0]
    pc = np.array([(pred_xyxy[0] + pred_xyxy[2]) / 2, (pred_xyxy[1] + pred_xyxy[3]) / 2], dtype=np.float32)
    dc = np.stack(
        [(det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2], axis=1
    )
    dist = np.linalg.norm(dc - pc[None, :], axis=1)
    dist_term = np.clip(dist / max(float(gate_radius), 1e-6), 0.0, 1.0)
    cost = 0.6 * (1.0 - iou) + 0.4 * dist_term
    return cost.astype(np.float32), iou.astype(np.float32)


def size_cost(pred_xyxy: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """Log-ratio scale disagreement in [0, 1]. Guards against ID swaps between
    a near adult and a far child, which pure IoU can tolerate."""
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    if det_boxes.shape[0] == 0:
        return np.zeros((0,), np.float32)
    ph = max(float(pred_xyxy[3] - pred_xyxy[1]), 1e-6)
    dh = np.maximum(det_boxes[:, 3] - det_boxes[:, 1], 1e-6)
    ratio = np.abs(np.log(dh / ph))
    return (1.0 - np.exp(-ratio)).astype(np.float32)


def direction_cost(velocity: np.ndarray | None, last_box: np.ndarray | None, det_boxes: np.ndarray,
                   weight: float = 0.2) -> np.ndarray:
    """How much a candidate violates the target's recent direction of travel.

    Implements the OC-SORT momentum idea in closed form: compare the momentum
    vector (from `last_box` backwards along `velocity`) with the vector from
    `last_box` to each candidate. Reversing direction is penalized; continuing
    straight is free. Robust when there is no usable history (returns zeros).

    Returns:
        (N,) float32 in [0, 1], already scaled by `weight`.
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    if det_boxes.shape[0] == 0:
        return np.zeros((0,), np.float32)
    if velocity is None or last_box is None:
        return np.zeros((det_boxes.shape[0],), np.float32)

    lb = np.array([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2], dtype=np.float32)
    momentum = np.asarray(velocity[:2], dtype=np.float32).copy()
    mnorm = float(np.linalg.norm(momentum))
    if not np.isfinite(mnorm) or mnorm < 1e-6:
        return np.zeros((det_boxes.shape[0],), np.float32)

    dc = np.stack(
        [(det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2], axis=1
    )
    obs = dc - lb[None, :]
    onorm = np.linalg.norm(obs, axis=1)
    valid = onorm > 1e-6
    cos = np.zeros((det_boxes.shape[0],), dtype=np.float32)
    if np.any(valid):
        cos[valid] = (obs[valid] @ momentum) / (onorm[valid] * mnorm)
    cos = np.clip(cos, -1.0, 1.0)
    # cos = +1 => continuing straight (cost 0); cos = -1 => exact reversal (cost 1).
    cost = 0.5 * (1.0 - cos)
    return (weight * cost).astype(np.float32)


def appearance_cost_of_query(
    query_feats: Mapping[str, np.ndarray],
    stm_feat: np.ndarray | None,
    ltm,
    cfg: CostCfg,
    frame_idx: int,
    *,
    stm_part_feats: Mapping[str, np.ndarray] | None = None,
) -> tuple[float, tuple[str, str] | None, float]:
    """Appearance cost for ONE candidate against STM + LTM.

    Part-aware: every (query part, gallery part) pair is scored as
    ``cosine_distance + part_penalty * (1 - affinity)`` and the best pair wins.
    The penalty keeps an upper-body query from being matched to a leg bank just
    because the raw cosine happened to be low.

    Args:
        query_feats: {part: L2-normalized embedding} for this detection.
        stm_feat: short-term EMA embedding (full-body), or None.
        ltm: object exposing ``best(feat, part=..., frame_idx=...)`` and
            ``hard_negative_penalty(feat)``, or None.
        cfg: cost configuration.
        frame_idx: current frame, forwarded to the gallery for recency weighting.
        stm_part_feats: optional per-part STM embeddings.

    Returns:
        (cost, winning_pair, hard_negative_penalty). `cost` is 0.5 when nothing
        was measurable.
    """
    if not query_feats:
        return 0.5, None, 0.0

    penalty_scale = 0.15
    best: float | None = None
    best_pair: tuple[str, str] | None = None

    # --- STM term (short-term memory: "how the target looks right now") ------
    stm_terms: list[tuple[float, tuple[str, str]]] = []
    if stm_feat is not None:
        for qp, qf in query_feats.items():
            aff = part_affinity(qp, "full")
            if aff <= 0.0:
                continue
            d = float(cosine_distance_matrix(qf, np.asarray(stm_feat, dtype=np.float32)))
            stm_terms.append((d + penalty_scale * (1.0 - aff), (qp, "full")))
    if stm_part_feats:
        for qp, qf in query_feats.items():
            for gp, gf in stm_part_feats.items():
                aff = part_affinity(qp, gp)
                if aff <= 0.0:
                    continue
                d = float(cosine_distance_matrix(qf, np.asarray(gf, dtype=np.float32)))
                stm_terms.append((d + penalty_scale * (1.0 - aff), (qp, gp)))

    # --- LTM term (long-term memory: "how the target looked before") ---------
    ltm_terms: list[tuple[float, tuple[str, str]]] = []
    if ltm is not None:
        for qp, qf in query_feats.items():
            try:
                d, entry = ltm.best(qf, part=qp, frame_idx=frame_idx)
            except TypeError:  # gallery without the part/frame_idx kwargs
                d, entry = ltm.best(qf)
            if entry is None or not np.isfinite(d):
                continue
            gp = getattr(entry, "part", qp)
            aff = part_affinity(qp, gp)
            if aff <= 0.0 and qp != gp:
                continue
            ltm_terms.append((float(d) + penalty_scale * (1.0 - aff), (qp, gp)))

    # --- blend -------------------------------------------------------------
    stm_d = min(stm_terms)[0] if stm_terms else None
    ltm_d = min(ltm_terms)[0] if ltm_terms else None
    pair_candidates: list[tuple[float, tuple[str, str]]] = list(stm_terms) + list(ltm_terms)

    if stm_d is not None and ltm_d is not None:
        wsum = cfg.stm_weight + cfg.ltm_weight
        best = (cfg.stm_weight * stm_d + cfg.ltm_weight * ltm_d) / max(wsum, 1e-6)
    elif stm_d is not None:
        best = stm_d
    elif ltm_d is not None:
        best = ltm_d

    if best is None:
        return 0.5, None, 0.0

    if pair_candidates:
        best_pair = min(pair_candidates)[1]

    hn = 0.0
    if ltm is not None and hasattr(ltm, "hard_negative_penalty"):
        try:
            qlist = [f for f in query_feats.values() if f is not None]
            if qlist:
                hn = float(max(ltm.hard_negative_penalty(f) for f in qlist))
        except Exception:
            hn = 0.0
    return float(best), best_pair, float(np.clip(hn, 0.0, 1.0))


def appearance_costs(
    det_feats: list[Mapping[str, np.ndarray]],
    stm_feat: np.ndarray | None,
    ltm,
    cfg: CostCfg,
    frame_idx: int,
    *,
    stm_part_feats: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str] | None]]:
    """Vectorized-over-candidates appearance cost.

    Returns:
        (cost, valid, hard_negative, winning_parts) — each (N,), plus a list of
        the winning part pairs. `valid` is False for candidates with no features,
        so the caller can renormalize the weights instead of inventing a cost.
    """
    n = len(det_feats)
    cost = np.full((n,), 0.5, dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    hn = np.zeros((n,), dtype=np.float32)
    pairs: list[tuple[str, str] | None] = [None] * n
    for i, feats in enumerate(det_feats):
        if not feats:
            continue
        c, pair, h = appearance_cost_of_query(feats, stm_feat, ltm, cfg, frame_idx, stm_part_feats=stm_part_feats)
        cost[i] = c
        valid[i] = True
        hn[i] = h
        pairs[i] = pair
    return cost, valid, hn, pairs


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #


def gate_radius(velocity: np.ndarray | None, base: float, speed_gain: float, frame_diag: float | None = None) -> float:
    """Adaptive search radius in pixels.

    `base` and `speed_gain` are fractions, so the radius scales with the frame.
    A fast target gets a proportionally larger gate - this is what stops a
    sprinting person from being dropped on the very frame they accelerate.
    """
    ref = float(frame_diag) if frame_diag and frame_diag > 0 else 640.0
    r = float(base) * ref
    if velocity is not None:
        v = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if v.size >= 2:
            speed = float(np.hypot(v[0], v[1]))
            if np.isfinite(speed):
                r += float(speed_gain) * speed
    return float(max(r, 1.0))


def motion_gate_mask(
    det_boxes: np.ndarray,
    pred_xyxy: np.ndarray,
    maha: np.ndarray | None,
    radius: float,
    maha_thresh: float,
    frame_w: int | None = None,
    frame_h: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Boolean gate + human-readable exclusion reasons.

    A candidate must be both within the Mahalanobis ellipse (when a motion model
    exists) and within the adaptive pixel radius of the prediction.
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 4)
    n = det_boxes.shape[0]
    mask = np.ones((n,), dtype=bool)
    reasons = [""] * n
    if n == 0:
        return mask, reasons

    pc = np.array([(pred_xyxy[0] + pred_xyxy[2]) / 2, (pred_xyxy[1] + pred_xyxy[3]) / 2], dtype=np.float32)
    dc = np.stack(
        [(det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2], axis=1
    )
    dist = np.linalg.norm(dc - pc[None, :], axis=1)
    too_far = dist > radius
    mask &= ~too_far
    for i in np.nonzero(too_far)[0]:
        reasons[int(i)] = f"outside_radius(dist={dist[i]:.1f}>{radius:.1f})"

    if maha is not None:
        maha = np.asarray(maha, dtype=np.float32).reshape(-1)
        if maha.shape[0] == n:
            bad = ~np.isfinite(maha) | (maha > maha_thresh)
            # Only exclude on Mahalanobis when the pixel radius also disagrees;
            # a single-model Mahalanobis is over-confident for abrupt motion.
            bad = bad & too_far
            mask &= ~bad
            for i in np.nonzero(bad)[0]:
                reasons[int(i)] = f"maha_gate(maha={maha[i]:.2f}>{maha_thresh:.2f})"
    return mask, reasons


# --------------------------------------------------------------------------- #
# Full cost assembly
# --------------------------------------------------------------------------- #


def compute_costs(
    dets: Sequence[Detection],
    pred_xyxy: np.ndarray | None,
    cfg: CostCfg,
    *,
    velocity: np.ndarray | None = None,
    last_box: np.ndarray | None = None,
    motion=None,
    stm_feat: np.ndarray | None = None,
    stm_part_feats: Mapping[str, np.ndarray] | None = None,
    ltm=None,
    frame_idx: int = 0,
    frame_shape: tuple[int, int] | None = None,
    apply_gate: bool = True,
    apply_motion: bool = True,
    gate_radius_override: float | None = None,
    direction_weight: float | None = None,
    missing_appearance_cost: float | None = None,
) -> CostBreakdown:
    """Assemble the combined association cost for every detection.

    Args:
        dets: candidate detections (already carrying `feat`/`part_feats` when available).
        pred_xyxy: motion prediction for this frame; ``None`` disables motion terms.
        cfg: `CostCfg`.
        motion: optional motion model exposing ``mahalanobis(boxes)``.
        apply_gate: set False for global re-detection (no spatial prior).
        apply_motion: set False when the prediction carries no information —
            i.e. after a long loss, when the target could be anywhere. Charging
            a motion cost against a 60-frame-old prediction would make
            re-acquisition impossible, so those terms are zeroed rather than
            merely gated. `size` is kept: scale is still a real cue.
        gate_radius_override: force a radius (used while LOST / out-of-frame).
        missing_appearance_cost: cost used for candidates with no features.
            ``None`` reads `CostCfg.missing_appearance_cost`.

    Returns:
        `CostBreakdown`, with `total` = +inf for excluded candidates.
    """
    n = len(dets)
    if missing_appearance_cost is None:
        missing_appearance_cost = float(getattr(cfg, "missing_appearance_cost", 0.5))
    boxes = np.stack([d.xyxy for d in dets], axis=0).astype(np.float32) if n else np.zeros((0, 4), np.float32)
    frame_diag = float(np.hypot(*frame_shape)) if frame_shape else None

    if pred_xyxy is None or not apply_motion:
        m_cost = np.zeros((n,), np.float32)
        iou = np.zeros((n,), np.float32)
    else:
        m_cost, iou = motion_cost(np.asarray(pred_xyxy, np.float32), boxes, gate_radius_override or 1e9)

    s_cost = size_cost(np.asarray(pred_xyxy, np.float32), boxes) if pred_xyxy is not None else np.zeros((n,), np.float32)

    if apply_motion:
        d_cost = direction_cost(
            velocity,
            last_box if last_box is not None else pred_xyxy,
            boxes,
            weight=direction_weight if direction_weight is not None else cfg.w_direction,
        )
    else:
        d_cost = np.zeros((n,), np.float32)

    feats = [d.feat if d.feat is not None or d.part_feats else None for d in dets]
    # Prefer per-detection part feats; fall back to the global feat as 'full'.
    feat_dicts: list[dict[str, np.ndarray]] = []
    for d in dets:
        fd: dict[str, np.ndarray] = {}
        if d.part_feats:
            fd.update({k: v for k, v in d.part_feats.items() if v is not None})
        if d.feat is not None:
            fd.setdefault("full", d.feat)
        feat_dicts.append(fd)
    del feats

    a_cost, a_valid, hn, pairs = appearance_costs(
        feat_dicts, stm_feat, ltm, cfg, frame_idx, stm_part_feats=stm_part_feats
    )
    if not cfg.part_matching:
        # Fall back to full-body only matching.
        a_cost = np.where(np.array([bool(f.get("full") is not None) for f in feat_dicts], dtype=bool),
                          a_cost, missing_appearance_cost).astype(np.float32)

    a_cost = np.where(a_valid, a_cost, missing_appearance_cost).astype(np.float32)

    # Renormalize weights so a missing cue does not silently scale the total.
    w_m, w_a, w_d, w_s = cfg.w_motion, cfg.w_appearance, cfg.w_direction, cfg.w_size
    total = w_m * m_cost + w_a * a_cost + w_d * d_cost + w_s * s_cost
    total = total + cfg.hard_negative_penalty * hn

    # --- E077: the two channel costs the OR-branch of acceptance reads --------
    # Computed from the same prediction and the same radius that gate this frame,
    # and NEVER folded into `total`: the ranking score is exactly what it was.
    #
    # They are ``+inf`` only when there IS no prediction (no motion model yet).
    # Deliberately NOT also gated on `apply_motion`: the whole point of the arm is
    # the frame where the tracker refuses a candidate that sits exactly where it
    # predicted, and on this tracker those frames are the `lost_short` ones, where
    # `apply_motion` is False.  Neutering the channel there would make the arm
    # structurally inert (measured: byte-identical to the reference on
    # dancetrack0097).  What keeps it honest is the bar itself -- with
    # `channel_geom_tau` at the correct-frame median (0.0278) the candidate must
    # sit within ~3 % of the current gate RADIUS of the prediction, i.e. about
    # 6-12 px on these sequences, which is a far tighter claim than the loose
    # spatial latch E022 removed -- plus the refusal rules that still run on top
    # (search needs readable appearance, search margin, ambiguous margin).
    if pred_xyxy is None:
        norm_centre = np.full((n,), np.inf, dtype=np.float32)
        log_h_ratio = np.full((n,), np.inf, dtype=np.float32)
    else:
        rad = gate_radius_override or gate_radius(
            velocity, cfg.gate_base, cfg.gate_speed_gain, frame_diag
        )
        p4 = np.asarray(pred_xyxy, dtype=np.float32).reshape(4)
        ph = max(abs(float(p4[3] - p4[1])), 1e-6)
        pc = np.array([(p4[0] + p4[2]) / 2.0, (p4[1] + p4[3]) / 2.0], dtype=np.float32)
        if n:
            dc = np.stack(
                [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1
            )
            dist = np.linalg.norm(dc - pc[None, :], axis=1)
            norm_centre = (dist / max(float(rad), 1e-6)).astype(np.float32)
            dh = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
            log_h_ratio = np.log(dh / ph).astype(np.float32)
            norm_centre = np.nan_to_num(norm_centre, nan=np.inf, posinf=np.inf, neginf=np.inf)
            log_h_ratio = np.nan_to_num(log_h_ratio, nan=np.inf, posinf=np.inf, neginf=np.inf)
        else:
            norm_centre = np.zeros((0,), np.float32)
            log_h_ratio = np.zeros((0,), np.float32)

    maha = np.full((n,), np.inf, dtype=np.float32)
    if motion is not None and n:
        try:
            maha = np.asarray(motion.mahalanobis(boxes), dtype=np.float32).reshape(-1)
        except Exception:
            maha = np.full((n,), np.inf, dtype=np.float32)

    if apply_gate and pred_xyxy is not None:
        radius = gate_radius_override or gate_radius(
            velocity, cfg.gate_base, cfg.gate_speed_gain, frame_diag
        )
        gate, reasons = motion_gate_mask(
            boxes,
            np.asarray(pred_xyxy, np.float32),
            maha if np.isfinite(maha).any() else None,
            radius,
            cfg.motion_gate,
        )
    else:
        gate = np.ones((n,), dtype=bool)
        reasons = [""] * n

    total = total.astype(np.float32)
    total = np.where(gate, total, np.inf).astype(np.float32)

    return CostBreakdown(
        motion=m_cost,
        appearance=a_cost,
        direction=np.asarray(d_cost, np.float32) / max(w_d if w_d else 1.0, 1e-6),
        size=s_cost,
        total=total,
        appearance_valid=a_valid,
        gate=gate,
        iou=iou,
        maha=maha,
        hard_negative=hn,
        winning_parts=pairs,
        excluded=reasons,
        norm_centre=norm_centre,
        log_h_ratio=log_h_ratio,
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_candidate(
    breakdown: CostBreakdown,
    cfg: CostCfg,
    *,
    lost: bool = False,
    require_margin: bool = True,
) -> MatchResult:
    """Pick the winner, or refuse.

    Refusal conditions, all of which are logged:
      * no candidate passes the gate;
      * the best cost exceeds the (state-dependent) acceptance threshold;
      * the best candidate does not beat the runner-up by `distractor_margin`;
      * the best candidate is the ONLY finite one and exceeds
        `lone_candidate_accept_cost` (see below).

    The margin test is the anti-latch mechanism: when two people look alike and
    are near each other, the correct behaviour for a consumer tracker is to keep
    coasting, not to gamble on one of them.

    The margin test is however **vacuous for a lone candidate**: with a single
    finite entry there is no runner-up, `margin` stays ``inf`` and
    ``margin < margin_thr`` is false for every threshold. The identity handover
    measured by motion-dev happens exactly there -- the bystander is the only
    detection on the frame the target stops being detected -- so
    `CostCfg.lone_candidate_accept_cost` supplies the absolute bound the margin
    test cannot. It applies in BOTH directions of the state machine: exempting
    the lost path reduced it to a two-frame deferral (the state goes blind, the
    exemption kicks in, the same lone bystander is re-accepted), so lone
    handovers fell from 3/4 runs to 0/4 once the exemption was removed. It also
    only applies when the lone candidate's appearance was actually READABLE:
    the bound is a bound on evidence, and when nothing could be read the cost is
    the constant ``missing_appearance_cost``, so refusing on it would be
    arbitrary and would turn any transient encoder failure into a lost target.
    """
    thr = cfg.accept_cost_lost if lost else cfg.accept_cost
    total = np.asarray(breakdown.total, dtype=np.float32)
    n = total.shape[0]
    if n == 0:
        return MatchResult(None, float("inf"), False, "no_detections")

    finite = np.isfinite(total)
    if not np.any(finite):
        # Distinguish "gated out" from "no detections" for diagnostics.
        reason = "all_gated"
        if breakdown.excluded and any(breakdown.excluded):
            reason = "all_gated:" + ";".join(sorted({e for e in breakdown.excluded if e})[:3])
        return MatchResult(None, float("inf"), False, reason)

    order = np.argsort(np.where(finite, total, np.inf))
    best = int(order[0])
    best_cost = float(total[best])

    margin = float("inf")
    if len(order) > 1:
        second = int(order[1])
        if np.isfinite(total[second]):
            margin = float(total[second] - best_cost)

    # ------------------------------------------------------------------ #
    # E077 arm B -- the additive bound gets an OR-branch.
    #
    # `total` remains the RANKING score above (argsort untouched) and the bound
    # below is untouched too; what changes is only what happens when it fails.
    # A candidate whose ADDITIVE sum is too high may still be admitted when a
    # single channel is individually conclusive: geometry (it is exactly where we
    # predicted, at the predicted height) or appearance (it is at the median
    # distance of a genuine same-person pair AND it still beats the runner-up).
    # Both legs are only reachable when the channel is informative at all
    # (`norm_centre`/`log_h_ratio` are +inf while searching, so geometry cannot
    # carry a rebind), and every refusal BELOW still runs: ambiguous margin,
    # lone-candidate bound, and the caller's search-needs-appearance rule.
    # ------------------------------------------------------------------ #
    channel = ""
    if best_cost > thr:
        channel = _conclusive_channel(breakdown, best, cfg, margin=margin,
                                      require_margin=require_margin, lost=lost)
        if not channel:
            return MatchResult(
                best,
                best_cost,
                False,
                f"cost {best_cost:.3f} > accept {thr:.3f}",
                margin=margin,
                diag=_diag(breakdown, best),
            )

    # While the motion prior is dead the gate is off, so appearance is the only
    # evidence left and the discriminative requirement goes UP even though
    # accept_cost_lost lets the absolute cost go up as well.
    margin_thr = float(cfg.distractor_margin)
    if lost:
        override = float(getattr(cfg, "distractor_margin_lost", 0.0) or 0.0)
        if override > 0:
            margin_thr = override

    if require_margin and margin_thr > 0 and margin < margin_thr:
        return MatchResult(
            best,
            best_cost,
            False,
            f"ambiguous: margin {margin:.3f} < {margin_thr:.3f}",
            margin=margin,
            diag=_diag(breakdown, best),
        )

    # A lone candidate has no runner-up, so the margin test above is vacuous and
    # accepts it at any threshold. Give it the absolute bound instead -- in BOTH
    # directions of the state machine. Exempting the lost path is what reduced a
    # swept bound of 0.20 to a two-frame deferral (results/motion_ablation/
    # lone_bound.md): the guard refused the lone bystander for
    # `soft_occlusion_after` frames, the state then went blind, and the lost path
    # re-accepted it unguarded (cost 0.83 against `accept_cost_lost` 1.25).
    # A genuine re-acquisition is not penalised: measured genuine rebinds arrive
    # at cost ~0.00, because there the appearance channel really does recognise
    # the target. The bound therefore tests EVIDENCE, not elapsed time -- which
    # is also why it only fires when the lone candidate's appearance was
    # readable at all. When nothing could be read the cost is the constant
    # `missing_appearance_cost`, and refusing on a placeholder would be
    # arbitrary (it would make any transient encoder failure lose the target).
    if require_margin:
        lone_bound = getattr(cfg, "lone_candidate_accept_cost", None)
        a_valid = np.asarray(breakdown.appearance_valid, dtype=bool)
        if (
            lone_bound is not None
            and int(np.count_nonzero(finite)) == 1
            and bool(a_valid[best])
            and best_cost > float(lone_bound)
        ):
            return MatchResult(
                best,
                best_cost,
                False,
                f"lone_candidate: cost {best_cost:.3f} > {float(lone_bound):.3f}",
                margin=margin,
                diag=_diag(breakdown, best),
            )

    return MatchResult(best, best_cost, True, "ok", margin=margin, diag=_diag(breakdown, best),
                       channel=channel)


def _conclusive_channel(
    b: CostBreakdown,
    i: int,
    cfg: CostCfg,
    *,
    margin: float,
    require_margin: bool,
    lost: bool,
) -> str:
    """E077 arm B: is channel ``i`` individually conclusive enough to carry the frame?

    Returns ``"geometry"``, ``"appearance"`` or ``""``.  Inert unless
    ``CostCfg.channel_accept`` is set, so the shipped decision is unchanged.

    geometry   ``norm_centre <= channel_geom_tau AND |log h_ratio| <= 0.25``
    appearance ``app_cost <= channel_app_tau AND margin >= distractor_margin``

    The 0.25 on the height ratio is a physical bound (a person's height does not
    change by more than ~28 % between two frames of the same shot), not a tuned
    weight; the plan fixes it in code.  ``appearance`` additionally requires that
    the appearance was actually READABLE (``appearance_valid``): the constant
    placeholder charged for an unreadable crop is not evidence, which is the same
    argument that keeps the lone-candidate bound conditional on it.
    """
    if not bool(getattr(cfg, "channel_accept", False)):
        return ""

    g_tau = float(getattr(cfg, "channel_geom_tau", 0.0) or 0.0)
    nc = getattr(b, "norm_centre", None)
    lh = getattr(b, "log_h_ratio", None)
    if g_tau > 0.0 and nc is not None and lh is not None and i < len(nc) and i < len(lh):
        nc_i = float(np.asarray(nc)[i])
        lh_i = float(np.asarray(lh)[i])
        if np.isfinite(nc_i) and np.isfinite(lh_i) and nc_i <= g_tau and abs(lh_i) <= 0.25:
            return "geometry"

    a_tau = float(getattr(cfg, "channel_app_tau", 0.0) or 0.0)
    if a_tau > 0.0:
        valid = bool(np.asarray(b.appearance_valid)[i])
        a_cost = float(np.asarray(b.appearance)[i])
        margin_thr = float(cfg.distractor_margin)
        if lost:
            override = float(getattr(cfg, "distractor_margin_lost", 0.0) or 0.0)
            if override > 0:
                margin_thr = override
        margin_ok = (not require_margin) or (margin >= margin_thr)
        if valid and np.isfinite(a_cost) and a_cost <= a_tau and margin_ok:
            return "appearance"
    return ""


def _diag(b: CostBreakdown, i: int) -> dict[str, Any]:
    d: dict[str, Any] = {
        "motion": float(b.motion[i]),
        "appearance": float(b.appearance[i]),
        "direction": float(b.direction[i]),
        "size": float(b.size[i]),
        "iou": float(b.iou[i]),
        "hard_negative": float(b.hard_negative[i]),
        "appearance_valid": bool(b.appearance_valid[i]),
    }
    if np.isfinite(b.maha[i]):
        d["maha"] = float(b.maha[i])
    if i < len(b.winning_parts):
        d["parts"] = b.winning_parts[i]
    # E077: the two channel costs, so a run can show WHY the branch did or did not
    # fire (the E032b failure mode is a branch that never runs).
    nc = getattr(b, "norm_centre", None)
    if nc is not None and i < len(nc):
        d["norm_centre"] = float(np.asarray(nc)[i])
    lh = getattr(b, "log_h_ratio", None)
    if lh is not None and i < len(lh):
        d["log_h_ratio"] = float(np.asarray(lh)[i])
    return d


def assign_hungarian(cost: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Standard rectangular Hungarian assignment above a cost threshold.

    Provided for the all-person layer and for experiments; the single-target
    path uses `select_candidate` instead (it needs refusal semantics, which
    assignment does not express).
    """
    from scipy.optimize import linear_sum_assignment

    c = np.asarray(cost, dtype=np.float64)
    if c.size == 0:
        return []
    c = np.where(np.isfinite(c), c, 1e6)
    rows, cols = linear_sum_assignment(c)
    return [(int(r), int(cc)) for r, cc in zip(rows, cols) if c[r, cc] <= thresh]
