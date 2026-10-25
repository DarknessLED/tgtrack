"""Hybrid-SORT-style association backend (7th all-person tracker, E012).

Why this file exists
--------------------
``docs/research/R1_mot_association.md`` records that Hybrid-SORT (AAAI 2024) reaches
**+7.6 HOTA over OC-SORT on DanceTrack with identical association inputs** using three
"weak cues" that cost nothing:

1. ``HMIoU = HIoU * IoU`` with ``HIoU = h_over / h_union`` — the one geometric cue that
   survives *vertical* truncation, i.e. exactly the upper-body-only / lower-body-only
   partial-view case this project cares about.
2. A **tracklet-confidence state** that is linearly extrapolated while the track is
   unobserved, instead of trusting a single frame's detection score.
3. A **height state**, so scale is modelled (predicted) rather than differenced.

None of the six shipped ultralytics backends uses HMIoU, so this backend is the one place
the all-person layer can still move AssA with the detector held fixed.

Source evidence (installed ultralytics 8.4.185)
-----------------------------------------------
* ``ultralytics/trackers/track_tracker.py:31-42`` — ``_hmiou_distance``::

      h_over  = min(a.y2, b.y2) - max(a.y1, b.y1)
      h_union = max(a.y2, b.y2) - min(a.y1, b.y1)
      h_iou   = clip(h_over / (h_union + 1e-9), 0, 1)
      return iou_sim, 1.0 - h_iou * iou_sim

  :func:`hmiou_matrix` implements that formula verbatim on ``(N, 4)`` arrays.
* ``ultralytics/trackers/track_tracker.py:67-75`` — ``_confidence_distance`` projects a
  track's score with first-order extrapolation ``proj = curr + (curr - prev)`` and costs
  ``|proj - det_score|``; ``:45-64`` ``_angle_distance`` is the 4-corner angle momentum
  over ``delta_t``. Both are re-implemented here against this module's own track state
  (:func:`angle_distance_matrix`, :meth:`HybridTrack.conf_projection`).
* ``ultralytics/trackers/utils/kalman_filter.py:8-205`` — ``KalmanFilterXYAH`` is reused
  as-is: its 8-D state ``[cx, cy, a, h, vx, vy, va, vh]`` *is* the height state (``h``
  with velocity ``vh``), so the prediction is a proper motion model.
* ``ultralytics/trackers/utils/matching.py:18-59`` — ``linear_assignment(cost, thresh)``
  (lap/Hungarian with ``cost_limit``) is the assignment used by all six baselines; we call
  the same function so the assignment semantics are directly comparable (scipy's
  ``linear_sum_assignment`` is the fallback when ultralytics is unavailable).
* ``ultralytics/trackers/utils/gmc.py:45-122`` — ``GMC.apply(frame) -> (2, 3)`` warp, reused
  with the same ``sparseOptFlow`` default as the baseline, so the camera-motion budget is
  identical and any metric difference comes from association only.

Interface
---------
Duck-type identical to :class:`tgtrack.mot_layer.AllPersonTracker` —
``__init__(cfg: MotCfg | None = None, detector=None)``, ``update(img_bgr, dets, feats=None)``,
``reset()``, plus ``started`` / ``num_tracks`` / ``names`` / ``describe()``.
``scripts/eval_mot.py`` consumes only that surface. ``feats`` is accepted and ignored:
this backend has no appearance branch (Hybrid-SORT *without* ReID is the measured recipe).
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import MotCfg
from .detector import PersonDetector
from .types import Detection

__all__ = [
    "HybridSortTracker",
    "HybridTrack",
    "hmiou_matrix",
    "iou_matrix",
    "angle_distance_matrix",
    "xyxy_to_xyah",
    "xyah_to_xyxy",
    "linear_assignment",
    "resolve_hybrid_yaml",
    "HYBRID_TRACKER_STEM",
    "DEFAULT_HYBRID_YAML",
]

HYBRID_TRACKER_STEM = "hybridsort"
DEFAULT_HYBRID_YAML = "configs/trackers/hybridsort.yaml"

#: Cost used for pairs a stage has already decided are invalid. ``lap.lapjv`` needs a
#: finite value; the caller thresholds afterwards, exactly like TrackTrack does.
_BIG = 1e5

#: Corner order used by the ROCM angle term: TL, TR, BR, BL.
_CORNER_IDX = ((0, 1), (2, 1), (2, 3), (0, 3))


# ---------------------------------------------------------------------------
# geometry helpers (pure, unit-testable without ultralytics / torch)
# ---------------------------------------------------------------------------
def xyxy_to_xyah(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4)`` xyxy -> ``(N, 4)`` ``[cx, cy, aspect, height]`` (KalmanFilterXYAH space)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w = np.maximum(b[:, 2] - b[:, 0], 1e-6)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    return np.stack([b[:, 0] + w / 2.0, b[:, 1] + h / 2.0, w / h, h], axis=1)


def xyah_to_xyxy(states: np.ndarray) -> np.ndarray:
    """``(N, 4)`` ``[cx, cy, aspect, height]`` -> ``(N, 4)`` xyxy."""
    s = np.asarray(states, dtype=np.float64).reshape(-1, 4)
    h = np.maximum(s[:, 3], 1e-6)
    w = np.maximum(s[:, 2] * h, 1e-6)
    return np.stack(
        [s[:, 0] - w / 2.0, s[:, 1] - h / 2.0, s[:, 0] + w / 2.0, s[:, 1] + h / 2.0], axis=1
    )


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """``(N, M)`` IoU between two xyxy box sets (vectorised, no ultralytics import)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    iw = np.minimum(a[:, 2:3], b[:, 2:3].T) - np.maximum(a[:, 0:1], b[:, 0:1].T)
    ih = np.minimum(a[:, 3:4], b[:, 3:4].T) - np.maximum(a[:, 1:2], b[:, 1:2].T)
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def hmiou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(iou_sim, 1 - HMIoU)`` with ``HMIoU = HIoU * IoU``.

    Verbatim port of ``ultralytics/trackers/track_tracker.py:31-42`` (``_hmiou_distance``):
    ``HIoU`` is the vertical overlap divided by the vertical union, so a track whose legs
    are occluded still scores well against the matching upper-body detection.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64), np.ones((n, m), dtype=np.float64)
    iou = iou_matrix(a, b)
    h_over = np.minimum(a[:, 3:4], b[:, 3:4].T) - np.maximum(a[:, 1:2], b[:, 1:2].T)
    h_union = np.maximum(a[:, 3:4], b[:, 3:4].T) - np.minimum(a[:, 1:2], b[:, 1:2].T)
    h_iou = np.clip(h_over / (h_union + 1e-9), 0, 1)
    return iou, 1.0 - h_iou * iou


def corners(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4)`` xyxy -> ``(N, 4, 2)`` corners in TL, TR, BR, BL order."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, [cx, cy]] for cx, cy in _CORNER_IDX], axis=1)


def angle_distance_matrix(
    box_from: np.ndarray,
    box_to: np.ndarray,
    det_boxes: np.ndarray,
    det_scores: np.ndarray | None = None,
) -> np.ndarray:
    """ROCM corner-angle cost of one track (``box_from -> box_to``) against all detections.

    The track's momentum is the per-corner displacement ``box_to - box_from``; the
    observation direction is ``det - box_to``. The cost is the mean absolute angle between
    them over the four corners, divided by ``pi`` (0 = perfectly consistent, 1 = reversal),
    scaled by the detection score exactly like ``track_tracker.py:64``.
    """
    src = corners(box_from)[0]  # (4, 2)
    dst = corners(box_to)[0]  # (4, 2)
    momentum = dst - src  # (4, 2)
    d = corners(det_boxes)  # (M, 4, 2)
    obs = d - dst[None, :, :]  # (M, 4, 2)
    num = (obs * momentum[None]).sum(-1)  # (M, 4)
    mnorm = np.linalg.norm(momentum, axis=-1)  # (4,)
    den = np.linalg.norm(obs, axis=-1) * mnorm[None] + 1e-5
    dot = np.clip(num / den, -1.0, 1.0)
    # A corner with no observed motion carries no direction evidence: contribute 0 rather
    # than arccos(0) = pi/2 (which would be a spurious half-penalty for a fresh track).
    angle = np.where(mnorm[None, :] > 1e-6, np.abs(np.arccos(dot)), 0.0)
    dist = angle.mean(axis=-1) / np.pi  # (M,)
    if det_scores is not None:
        dist = dist * np.asarray(det_scores, dtype=np.float64).reshape(-1)
    return dist


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> tuple[list[list[int]], list[int], list[int]]:
    """Thresholded optimal assignment, identical semantics to the ultralytics backends.

    Uses ultralytics' ``matching.linear_assignment`` (``lap.lapjv`` with ``cost_limit``)
    when available, else scipy's ``linear_sum_assignment`` plus a post-hoc threshold.
    Returns ``(matches, unmatched_rows, unmatched_cols)`` with plain int indices.
    """
    cm = np.asarray(cost_matrix, dtype=np.float64)
    if cm.size == 0:
        return [], list(range(cm.shape[0])), list(range(cm.shape[1]))
    try:
        from ultralytics.trackers.utils.matching import linear_assignment as _la

        matches, unmatched_a, unmatched_b = _la(cm, thresh)
        return (
            [[int(a), int(b)] for a, b in matches],
            [int(a) for a in unmatched_a],
            [int(b) for b in unmatched_b],
        )
    except Exception:
        from scipy.optimize import linear_sum_assignment as _lsa

        rows, cols = _lsa(cm)
        matches = [[int(a), int(b)] for a, b in zip(rows, cols) if cm[a, b] <= thresh]
        used_a = {a for a, _ in matches}
        used_b = {b for _, b in matches}
        return (
            matches,
            [i for i in range(cm.shape[0]) if i not in used_a],
            [j for j in range(cm.shape[1]) if j not in used_b],
        )


def _warp_boxes(matrix: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Apply a ``(2, 3)`` GMC warp to ``(N, 4)`` xyxy boxes (via their 4 corners)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if b.size == 0 or matrix is None:
        return b
    pts = corners(b).reshape(-1, 2).astype(np.float64)  # (4N, 2)
    warped = pts @ np.asarray(matrix, dtype=np.float64)[:, :2].T + np.asarray(matrix, dtype=np.float64)[:, 2]
    warped = warped.reshape(-1, 4, 2)
    return np.stack(
        [warped[:, :, 0].min(1), warped[:, :, 1].min(1), warped[:, :, 0].max(1), warped[:, :, 1].max(1)], axis=1
    )


def resolve_hybrid_yaml(name: str) -> Path:
    """Resolve this backend's yaml: explicit path, cwd ``configs/trackers``, or repo root."""
    raw = Path(str(name))
    if raw.exists():
        return raw
    stem = raw.stem or HYBRID_TRACKER_STEM
    candidates = [
        Path.cwd() / "configs" / "trackers" / f"{stem}.yaml",
        Path(__file__).resolve().parents[2] / "configs" / "trackers" / f"{stem}.yaml",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"hybrid tracker yaml {stem}.yaml not found; looked in {[str(c) for c in candidates]}")


# ---------------------------------------------------------------------------
# tracklet
# ---------------------------------------------------------------------------
class HybridTrack:
    """One tracklet: Kalman state + confidence state + height state + observation history."""

    def __init__(
        self,
        box_xyxy: np.ndarray,
        score: float,
        cls: float,
        track_id: int,
        frame_id: int,
        kf: Any,
        tp: dict[str, Any],
    ) -> None:
        self.id = int(track_id)
        self.kf = kf
        self.cls = float(cls)
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box_xyxy)[0])
        self.score = float(score)
        self.prev_score = float(score)
        #: Hybrid-SORT tracklet-confidence STATE (EMA on hits, linear decay on misses).
        self.conf_state = float(score)
        self.conf_state_prev = float(score)
        self.alpha = float(tp.get("conf_alpha", 0.6))
        self.decay = float(tp.get("conf_decay", 0.04))
        self.h_beta = float(tp.get("height_beta", 0.5))
        self.h_clip = float(tp.get("h_clip", 1.5))
        self.hist_len = int(tp.get("history_len", 64))
        self.n_init = int(tp.get("n_init", 1))
        h = float(np.asarray(box_xyxy, dtype=np.float64).reshape(4)[3] - np.asarray(box_xyxy, dtype=np.float64).reshape(4)[1])
        self.h_state = max(h, 1.0)
        self.history: list[tuple[int, np.ndarray]] = [(int(frame_id), np.asarray(box_xyxy, dtype=np.float64).reshape(4).copy())]
        self.first_frame = int(frame_id)
        self.frame_last_obs = int(frame_id)
        self.age = 1
        self.hits = 1
        self.hits_streak = 1
        self.time_since_update = 0
        self.is_activated = bool(tp.get("activate_on_creation", False)) or (
            bool(tp.get("activate_first_frame", True)) and int(frame_id) == 1
        )
        self._pred = np.asarray(box_xyxy, dtype=np.float64).reshape(4).copy()

    # ------------------------------------------------------------------ state
    @property
    def xyxy(self) -> np.ndarray:
        """Current prediction (KF, plus any camera-motion warp applied this frame)."""
        return self._pred.copy()

    @property
    def conf_projection(self) -> float:
        """First-order extrapolation of the confidence state (``curr + (curr - prev)``)."""
        return float(np.clip(self.conf_state + (self.conf_state - self.conf_state_prev), 0.0, 1.5))

    def predict(self) -> None:
        """Kalman prediction for the next frame + confidence-state linear extrapolation.

        The predicted **height** is bounded by the height state: ``h_pred`` is clipped to
        ``h_state * [1/h_clip, h_clip]`` (centre and aspect ratio kept, the KF state itself
        untouched). A single abrupt height change -- exactly what a truncated or partially
        occluded box produces -- can leave ``vh`` badly wrong, and the next prediction then
        shrinks/grows so much that a good detection falls outside ``match_thresh``. Bounding
        the extrapolation by the state we already track is cheaper and less ambiguous than
        also scoring against the last observation.

        Scope note: the original end-to-end repro for this (h 300 -> 200 -> predicted 47 vs
        detected 80 -> cost 0.908) turned out to be caused by the spawn path applying its
        first observation twice, not by the clamp -- see the ``spawned`` guard in
        :meth:`HybridSortTracker._step`. With that fixed the truncation scenario matches
        without the clamp, so ``h_clip`` is a defensive bound on KF overshoot, pinned by
        ``tests/test_mot_hybrid.py::test_height_clamp_bounds_the_predicted_height``.
        """
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hits_streak = 0
        state = self.mean[:4].copy()
        if self.h_clip > 1.0:
            lo = self.h_state / self.h_clip
            hi = self.h_state * self.h_clip
            state[3] = float(np.clip(state[3], lo, hi))
        self._pred = xyah_to_xyxy(state)[0]
        self.conf_state = max(0.0, self.conf_state - self.decay)

    def apply_obs(self, box_xyxy: np.ndarray, score: float, frame_id: int) -> None:
        """Kalman correction with a real detection, then update the weak-cue states."""
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(box)[0], confidence=float(score))
        self.conf_state_prev = self.conf_state
        self.conf_state = self.alpha * self.conf_state + (1.0 - self.alpha) * float(score)
        h = max(float(box[3] - box[1]), 1.0)
        self.h_state = (1.0 - self.h_beta) * self.h_state + self.h_beta * h
        self.prev_score, self.score = self.score, float(score)
        self.history.append((int(frame_id), box.copy()))
        if len(self.history) > self.hist_len:
            self.history = self.history[-self.hist_len :]
        self.time_since_update = 0
        self.hits += 1
        self.hits_streak += 1
        self.frame_last_obs = int(frame_id)
        self._pred = box.copy()
        if not self.is_activated and self.hits > self.n_init:
            self.is_activated = True

    # ---------------------------------------------------------------- helpers
    def history_box(self, frame_id: int) -> np.ndarray | None:
        """Last observed box at or before ``frame_id`` (None when the history is too short)."""
        for fid, box in reversed(self.history):
            if fid <= frame_id:
                return box
        return None

    def last_obs_box(self) -> np.ndarray:
        """The most recent observed box (used for long-occlusion re-binding)."""
        return self.history[-1][1].copy()

    def warp(self, matrix: np.ndarray) -> None:
        """Camera-motion compensation: warp the state centre, prediction and history."""
        self._pred = _warp_boxes(matrix, self._pred[None, :])[0]
        self.mean[:4] = xyxy_to_xyah(_warp_boxes(matrix, xyah_to_xyxy(self.mean[:4])[None, :])[0])[0]
        self.history = [(fid, _warp_boxes(matrix, box[None, :])[0]) for fid, box in self.history]


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------
class HybridSortTracker:
    """Hybrid-SORT-style all-person tracker (HMIoU + confidence state + height state).

    Example:
        >>> mot = HybridSortTracker(MotCfg(tracker="hybridsort.yaml"), detector=det)
        >>> dets = det.detect(frame)
        >>> mot.update(frame, dets)      # det.track_id filled in place
        >>> mot.reset()                  # between videos / sequences
    """

    def __init__(self, cfg: MotCfg | None = None, detector: PersonDetector | None = None) -> None:
        self.cfg = cfg or MotCfg(tracker=f"{HYBRID_TRACKER_STEM}.yaml")
        self.detector = detector
        self.tracker_yaml = resolve_hybrid_yaml(self.cfg.tracker)
        self.tracker_type = HYBRID_TRACKER_STEM
        self.args = _load_hybrid_args(self.cfg)
        tp = dict(vars(self.args))
        self._tp = {
            **tp,
            "conf_alpha": float(tp.get("conf_alpha", 0.6)),
            "conf_decay": float(tp.get("conf_decay", 0.04)),
            "n_init": int(tp.get("n_init", 1)),
            "activate_first_frame": bool(tp.get("activate_first_frame", True)),
            "activate_on_creation": bool(tp.get("activate_on_creation", False)),
        }
        self.track_high_thresh = float(tp.get("track_high_thresh", 0.25))
        self.track_low_thresh = float(tp.get("track_low_thresh", 0.10))
        self.new_track_thresh = float(tp.get("new_track_thresh", 0.25))
        self.track_buffer = int(tp.get("track_buffer", 30))
        self.match_thresh = float(tp.get("match_thresh", 0.8))
        self.match_thresh_low = float(tp.get("match_thresh_low", 0.5))
        self.fuse_score = bool(tp.get("fuse_score", False))
        self.w_hmiou = float(tp.get("w_hmiou", 1.0))
        self.w_vel = float(tp.get("w_vel", 0.2))
        self.w_conf = float(tp.get("w_conf", 0.1))
        self.w_height = float(tp.get("w_height", 0.1))
        self.vel_deltas = tuple(int(d) for d in (tp.get("vel_deltas") or (1, 2, 3)))
        self.conf_floor = float(tp.get("conf_floor", 0.05))
        self.height_gate = float(tp.get("height_gate", 2.5))
        self.rebind_thresh = float(tp.get("rebind_thresh", 0.9))
        self.rebind_max_frames = int(tp.get("rebind_max_frames", 60))
        self.rebind_lost_penalty = float(tp.get("rebind_lost_penalty", 0.15))
        self.h_clip = float(tp.get("h_clip", 1.5))
        self.rescue_ratio = float(tp.get("rescue_ratio", 1.15) or 1.0)
        if self.cfg.with_reid:
            warnings.warn(
                "hybridsort has no appearance branch; with_reid=True is ignored",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

            self.kf = KalmanFilterXYAH()
        except Exception as exc:  # pragma: no cover - ultralytics is a hard dependency in practice
            raise ImportError("HybridSortTracker needs ultralytics' KalmanFilterXYAH") from exc
        method = getattr(self.args, "gmc_method", None)
        self.gmc = None
        if method and str(method).lower() not in {"none", "null"}:
            try:
                from ultralytics.trackers.utils.gmc import GMC

                self.gmc = GMC(method=str(method), downscale=int(tp.get("gmc_downscale", 2) or 2))
            except Exception:
                self.gmc = None
        self.tracks: list[HybridTrack] = []
        self.frame_id = 0
        self.n_updates = 0
        self._ids = itertools.count(1)
        self._warned_empty = False

    # ---------------------------------------------------------------- public
    @property
    def started(self) -> bool:
        """True once at least one frame has been processed."""
        return self.frame_id > 0

    @property
    def num_tracks(self) -> int:
        """Number of live (non-removed) tracklets."""
        return len(self.tracks)

    @property
    def names(self) -> dict:
        """Effective tracker configuration as a plain dict."""
        return dict(vars(self.args))

    def update(
        self,
        img_bgr: np.ndarray,
        dets: list[Detection],
        feats: np.ndarray | None = None,
    ) -> list[Detection]:
        """Assign stable ids to ``dets`` in place and return the same list object.

        Args:
            img_bgr: Current BGR frame (only used for GMC).
            dets: ``PersonDetector`` detections for this frame.
            feats: Accepted for interface parity and ignored (no appearance branch).

        Returns:
            ``dets`` with ``track_id`` set (``None`` for detections whose track is not
            activated yet — same convention as the ultralytics backends).
        """
        dets = list(dets)
        for d in dets:
            d.track_id = None
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
            raise ValueError(f"update() expects an (H, W, 3) BGR ndarray, got {type(img_bgr)}")

        self.frame_id += 1
        for t in self.tracks:
            t.predict()
        matrix = self._gmc_matrix(img_bgr)
        if matrix is not None:
            for t in self.tracks:
                t.warp(matrix)

        if dets:
            boxes = np.asarray([np.asarray(d.xyxy, dtype=np.float64).reshape(4) for d in dets], dtype=np.float64)
            scores = np.asarray([float(d.score) for d in dets], dtype=np.float64)
            classes = np.asarray([float(d.cls) for d in dets], dtype=np.float64)
            self._step(boxes, scores, classes, dets)
        else:
            self._prune()
        self.n_updates += 1
        return dets

    def reset(self) -> None:
        """Clear every tracklet, the ID counter and the GMC state.

        Must be called between sequences: leaked ids/state corrupt MOTA/IDF1/HOTA.
        """
        self.tracks = []
        self.frame_id = 0
        self.n_updates = 0
        self._ids = itertools.count(1)
        self._warned_empty = False
        if self.gmc is not None:
            try:
                self.gmc.reset_params()
            except Exception:
                pass

    def describe(self) -> dict[str, Any]:
        """Reproducibility record of the effective configuration (mirrors mot_layer)."""
        keys = (
            "tracker_type",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "match_thresh_low",
            "fuse_score",
            "with_reid",
            "gmc_method",
            "w_hmiou",
            "w_vel",
            "w_conf",
            "w_height",
            "vel_deltas",
            "conf_alpha",
            "conf_decay",
            "conf_floor",
            "height_gate",
            "h_clip",
            "rebind_thresh",
            "rebind_max_frames",
            "rebind_lost_penalty",
            "n_init",
            "activate_first_frame",
            "activate_on_creation",
        )
        return {k: getattr(self.args, k, None) for k in keys} | {
            "tracker_yaml": str(self.tracker_yaml),
            "device": getattr(self.args, "device", None),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HybridSortTracker(tracks={len(self.tracks)}, frame={self.frame_id})"

    # --------------------------------------------------------------- internals
    def _gmc_matrix(self, img_bgr: np.ndarray) -> np.ndarray | None:
        if self.gmc is None:
            return None
        try:
            matrix = np.asarray(self.gmc.apply(img_bgr), dtype=np.float64)
        except Exception:
            return None
        if matrix.shape != (2, 3) or np.allclose(matrix, np.eye(2, 3)):
            return None
        return matrix

    def _conf_dist(self, tracks: list[HybridTrack], det_scores: np.ndarray) -> np.ndarray:
        """``|projected confidence state - detection score|`` (Hybrid-SORT C_Conf)."""
        proj = np.asarray([t.conf_projection for t in tracks], dtype=np.float64)
        return np.abs(proj[:, None] - np.asarray(det_scores, dtype=np.float64)[None, :])

    def _height_cost(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        """Scale consistency ``|log(h_pred / h_det)|``, with an optional hard gate."""
        tb = np.asarray(track_boxes, dtype=np.float64).reshape(-1, 4)
        db = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        h_t = np.maximum(tb[:, 3] - tb[:, 1], 1e-6)
        h_d = np.maximum(db[:, 3] - db[:, 1], 1e-6)
        ratio = h_t[:, None] / h_d[None, :]
        cost = np.clip(np.abs(np.log(ratio)), 0.0, 1.0)
        if self.height_gate and self.height_gate > 0:
            bad = (ratio > self.height_gate) | (ratio < 1.0 / self.height_gate)
            cost = np.where(bad, _BIG, cost)
        return cost

    def _rocm_cost(self, tracks: list[HybridTrack], det_boxes: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
        """ROCM: min over ``vel_deltas`` of the 4-corner angle-momentum cost."""
        n, m = len(tracks), len(np.asarray(det_boxes).reshape(-1, 4))
        out = np.zeros((n, m), dtype=np.float64)
        if n == 0 or m == 0:
            return out
        for i, t in enumerate(tracks):
            best: np.ndarray | None = None
            for dt in self.vel_deltas:
                hist = t.history_box(self.frame_id - dt)
                prev = t.history_box(self.frame_id - dt - 1)
                if hist is None or prev is None:
                    continue
                dist = angle_distance_matrix(prev, hist, det_boxes, det_scores)
                best = dist if best is None else np.minimum(best, dist)
            if best is not None:
                out[i] = best
        return out

    def _geo_cost(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Blended geometry cost ``w_hmiou*(1 - HMIoU) + (1 - w_hmiou)*(1 - IoU)``.

        ``w_hmiou=1`` reproduces TrackTrack's pure HMIoU cost; ``0`` is plain IoU. The
        blend matters because ``1 - HMIoU`` is *stricter* than ``1 - IoU`` on a genuinely
        truncated pair (a half-visible box scores ``HMIoU=0.25`` but ``IoU=0.5``), so pure
        HMIoU would push some true truncation matches past ``match_thresh``. Blending keeps
        such a match (cost ``0.625`` < ``0.8``) while still letting vertical alignment break
        ties between candidates of similar IoU.
        """
        iou, c_hmiou = hmiou_matrix(boxes_a, boxes_b)
        w = float(np.clip(self.w_hmiou, 0.0, 1.0))
        return w * c_hmiou + (1.0 - w) * (1.0 - iou)

    def _stage_cost(
        self,
        tracks: list[HybridTrack],
        boxes_for_geo: np.ndarray,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
    ) -> np.ndarray:
        """One full stage-1 cost grid for a given geometry source (prediction or last obs)."""
        det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        cost = self._geo_cost(boxes_for_geo, det_boxes)
        if self.w_vel:
            # Direction evidence is only meaningful where the boxes actually overlap:
            # `track_tracker.py` computes its angle term for IoU-supported pairs and leaves the
            # rest at the geometry cost. Adding it everywhere penalised pairs with no overlap
            # for a momentum that says nothing about them.
            iou = iou_matrix(np.asarray(boxes_for_geo, dtype=np.float64).reshape(-1, 4), det_boxes)
            cost = cost + self.w_vel * self._rocm_cost(tracks, det_boxes, det_scores) * (iou > 1e-6)
        if self.w_conf:
            cost = cost + self.w_conf * self._conf_dist(tracks, det_scores)
        if self.w_height:
            cost = cost + self.w_height * self._height_cost(boxes_for_geo, det_boxes)
        if self.fuse_score:
            cost = 1.0 - (1.0 - np.clip(cost, 0.0, 1.0)) * np.asarray(det_scores, dtype=np.float64)[None, :]
        return np.where(np.isfinite(cost), cost, _BIG)

    def _rescue_pairs(
        self,
        cost: np.ndarray,
        rows: Sequence[int],
        cols: Sequence[int],
        ceiling: float,
    ) -> list[tuple[int, int]]:
        """Greedy mutually-nearest second round for pairs just above the matching ceiling.

        ``lap`` optimises the *global* assignment, so when two tracks compete for one detection a
        correct pair can be left unmatched purely because it was the loser of that contest -- and
        an unmatched track plus an unmatched detection is exactly how an id switch is born.
        TrackTrack's iterative associate handles the same failure mode by re-running greedy
        mutually-nearest matching with a shrinking ceiling; this is that round, restricted to
        pairs whose cost is at most ``ceiling`` and where each side is the other's nearest
        remaining candidate (so it cannot steal a better match).
        """
        pairs: list[tuple[int, int]] = []
        if not len(rows) or not len(cols):
            return pairs
        sub = np.asarray(cost, dtype=np.float64)[np.ix_(list(rows), list(cols))]
        work = np.where(np.isfinite(sub), sub, _BIG).copy()
        while work.size:
            i, j = np.unravel_index(int(np.argmin(work)), work.shape)
            if not np.isfinite(work[i, j]) or work[i, j] > ceiling:
                break
            if int(np.argmin(work[i, :])) != j or int(np.argmin(work[:, j])) != i:
                work[i, j] = _BIG  # one side has a cheaper option: let that round settle it
                continue
            pairs.append((int(rows[i]), int(cols[j])))
            work[i, :] = _BIG
            work[:, j] = _BIG
        return pairs

    def _cost_matrix(self, tracks: list[HybridTrack], det_boxes: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
        """Hybrid-SORT cost for the stage-1 (high-score) pass.

        The geometry is the track's Kalman prediction, whose height is already bounded by
        the height state in :meth:`HybridTrack.predict` -- that bound is what stops the
        post-truncation KF overshoot from mis-gating a frame (see the ``h_clip`` knob).
        """
        pred_boxes = np.asarray([t.xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)
        return self._stage_cost(tracks, pred_boxes, det_boxes, det_scores)

    def _iou_cost(self, track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        return 1.0 - iou_matrix(track_boxes, det_boxes)

    def _rebind_cost(self, tracks: list[HybridTrack], det_boxes: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
        """Relaxed cost for LOST tracks: best of last-observation / KF-prediction geometry + weak cues + lost-age.

        Both anchors are stale for a lost track, so scoring each pair against the better-supported
        one only *widens* the re-bind window. The elementwise min is deliberately confined to this
        stage: in the tracked stage the same trick let two crossing tracks exchange identities
        (measured), but here the candidate set is already reduced to tracks nobody has claimed and
        detections nobody has matched, and :attr:`rebind_lost_penalty` still orders by lost age.
        """
        last = np.asarray([t.last_obs_box() for t in tracks], dtype=np.float64).reshape(-1, 4)
        pred = np.asarray([t.xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)
        det_boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
        cost = np.minimum(self._geo_cost(last, det_boxes), self._geo_cost(pred, det_boxes))
        if self.w_vel:
            iou = np.maximum(iou_matrix(last, det_boxes), iou_matrix(pred, det_boxes))
            cost = cost + self.w_vel * self._rocm_cost(tracks, det_boxes, det_scores) * (iou > 1e-6)
        if self.w_conf:
            cost = cost + self.w_conf * self._conf_dist(tracks, det_scores)
        if self.w_height:
            cost = cost + self.w_height * np.minimum(
                self._height_cost(last, det_boxes), self._height_cost(pred, det_boxes)
            )
        age = np.asarray(
            [min(1.0, (self.frame_id - t.frame_last_obs) / max(1, self.rebind_max_frames)) for t in tracks],
            dtype=np.float64,
        )
        cost = cost + self.rebind_lost_penalty * age[:, None]
        return np.where(np.isfinite(cost), cost, _BIG)

    def _step(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        dets: list[Detection],
    ) -> None:
        """One association frame: high-score, low-score recovery, re-bind, spawn, prune."""
        high = np.where(scores >= self.track_high_thresh)[0]
        low = np.where((scores >= self.track_low_thresh) & (scores < self.track_high_thresh))[0]
        active = [i for i, t in enumerate(self.tracks) if t.time_since_update <= 1]
        lost = [i for i, t in enumerate(self.tracks) if t.time_since_update > 1]
        matches: dict[int, int] = {}

        # stage 1 — tracked tracks vs high-score detections (Hybrid-SORT cost)
        if active and high.size:
            cost = self._cost_matrix([self.tracks[i] for i in active], boxes[high], scores[high])
            pairs, ua, ub = linear_assignment(cost, self.match_thresh)
            rem_a, rem_b = list(ua), list(ub)
            if self.rescue_ratio > 1.0:
                for a, b in self._rescue_pairs(cost, rem_a, rem_b, self.match_thresh * self.rescue_ratio):
                    pairs.append((a, b))
                    rem_a.remove(a)
                    rem_b.remove(b)
            for a, b in pairs:
                matches[int(high[b])] = active[a]
            rem_active = [active[a] for a in rem_a]
            rem_high = [int(high[b]) for b in rem_b]
        else:
            rem_active = list(active)
            rem_high = [int(i) for i in high]

        # stage 2 — unmatched active tracks vs low-score detections (ByteTrack recovery)
        if rem_active and low.size:
            cost = self._iou_cost(
                np.asarray([self.tracks[i].xyxy for i in rem_active], dtype=np.float64), boxes[low]
            )
            pairs, ua, _ub = linear_assignment(cost, self.match_thresh_low)
            for a, b in pairs:
                matches[int(low[b])] = rem_active[a]
            rem_active = [rem_active[a] for a in ua]

        # stage 3 — re-bind LOST tracks (relaxed, last-observation geometry)
        rebindable = [
            i
            for i in lost
            if (self.frame_id - self.tracks[i].frame_last_obs) <= self.rebind_max_frames
            and self.tracks[i].conf_state >= self.conf_floor
        ]
        if rebindable and rem_high:
            cost = self._rebind_cost([self.tracks[i] for i in rebindable], boxes[rem_high], scores[rem_high])
            pairs, _ua, ub = linear_assignment(cost, self.rebind_thresh)
            for a, b in pairs:
                matches[rem_high[b]] = rebindable[a]
            rem_high = [rem_high[b] for b in ub]

        # stage 3b -- still-unmatched LOST tracks vs leftover LOW-score detections
        # A person whose detections dip under `track_high_thresh` for a stretch (very common with
        # the FRCNN detections) otherwise has no pathway back: the track is lost, so it is not
        # eligible for the active/low pass, and the re-bind pass only sees high-score detections.
        # TrackTrack/ByteTrack keep their lost tracks alive on weak detections for the same reason.
        still_lost = [i for i in rebindable if i not in set(matches.values())]
        rem_low = [int(i) for i in low if int(i) not in matches]
        if still_lost and rem_low:
            cost = self._rebind_cost([self.tracks[i] for i in still_lost], boxes[rem_low], scores[rem_low])
            pairs, _ua, _ub = linear_assignment(cost, self.match_thresh_low)
            for a, b in pairs:
                matches[rem_low[b]] = still_lost[a]

        # stage 4 — spawn new tracks from leftover high-score detections
        spawned: set[int] = set()
        for j in rem_high:
            if scores[j] < self.new_track_thresh:
                continue
            track = HybridTrack(boxes[j], scores[j], classes[j], next(self._ids), self.frame_id, self.kf, self._tp)
            self.tracks.append(track)
            matches[j] = len(self.tracks) - 1
            spawned.add(int(j))

        # stage 5 — apply matches and expose ids
        for det_idx, track_idx in matches.items():
            track = self.tracks[track_idx]
            if det_idx in spawned:
                # A track spawned after frame 1 has already consumed its first observation
                # in __init__; the ultralytics baselines also leave it unconfirmed until a
                # later match (``track_tracker.py``: ``is_activated = frame_id == 1``), which
                # is what keeps DetA comparable instead of inventing one-frame ids. A
                # frame-1 spawn is already activated and is output immediately.
                if track.is_activated:
                    dets[det_idx].track_id = int(track.id)
                continue
            track.apply_obs(boxes[det_idx], scores[det_idx], self.frame_id)
            if track.is_activated:
                dets[det_idx].track_id = int(track.id)
        self._prune()

    def _prune(self) -> None:
        """Drop tracks that have been unobserved for longer than ``track_buffer``."""
        keep = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        if len(keep) != len(self.tracks):
            self.tracks = keep
        elif not self.tracks and not self._warned_empty:
            self._warned_empty = True  # nothing to warn about; keeps the flag meaningful


def _load_hybrid_args(cfg: MotCfg) -> Any:
    """Load the hybrid yaml and apply the same ``MotCfg`` overrides as ``mot_layer``."""
    from ultralytics.utils import YAML, IterableSimpleNamespace

    path = resolve_hybrid_yaml(cfg.tracker)
    data: dict[str, Any] = dict(YAML.load(path) or {})
    data.setdefault("tracker_type", HYBRID_TRACKER_STEM)
    overrides: dict[str, Any] = {
        "track_high_thresh": cfg.track_high_thresh,
        "track_low_thresh": cfg.track_low_thresh,
        "new_track_thresh": cfg.new_track_thresh,
        "track_buffer": cfg.track_buffer,
        "match_thresh": cfg.match_thresh,
        "with_reid": cfg.with_reid,
        "gmc_method": cfg.gmc_method,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    data.update(dict(cfg.extra or {}))
    data["tracker_yaml"] = str(path)
    args = IterableSimpleNamespace(**data)
    try:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        args.device = "cpu"
    return args
