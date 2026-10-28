"""Evaluation harness — MOT metrics for the all-person layer + target-centric metrics.

Two metric families live here:

1. **MOT metrics** (all-person layer), implemented from scratch in NumPy:
   CLEAR-MOT (``MOTA``/``MOTP``/``IDSW``/``Frag``/``MT``/``PT``/``ML``),
   Identity (``IDF1``/``IDP``/``IDR``) and HOTA (``HOTA``/``DetA``/``AssA``/``LocA``
   with the standard alpha sweep 0.05..0.95).
2. **Target-centric metrics** (the real deliverable): success rate, precision,
   loss rate, re-acquisition rate/latency, *false acquisition rate* and identity
   consistency for the clicked DJI-style target.

Correctness contract
--------------------
The three MOT metrics mirror the reference implementations in
``JonathonLuiten/TrackEval`` (``trackeval/metrics/{clear,identity,hota}.py``)
operation for operation, so numbers produced here are directly comparable with
published MOTChallenge tables:

* CLEAR-MOT — Bernardin & Stiefelhagen, EURASIP JIVP 2008.
* IDF1 — Ristani et al., ECCV-W 2016.
* HOTA — Luiten et al., "HOTA: A Higher Order Metric for Evaluating
  Multi-Object Tracking", IJCV 2020.

Places where a naive implementation silently diverges from TrackEval and this
module follows TrackEval instead:

* CLEAR matching maximises ``1000 * (tracker_id == previous_tracker_id_of_this_gt)
  + IoU``, i.e. it *prefers continuity* before maximising IoU. A pure IoU-max
  Hungarian under-counts ID switches.
* CLEAR's ``Frag`` counts transitions untracked->tracked per GT id, then
  ``sum(runs - 1)`` over ids that were ever tracked (not "number of gaps").
* ``MT`` uses ``tracked_ratio > 0.8`` while ``PT`` uses ``>= 0.2``.
* IDF1 solves a *global* assignment over the whole sequence via a
  ``(num_gt_ids + num_tracker_ids)`` square cost matrix of FN+FP counts.
* HOTA computes a per-frame matching score
  ``global_alignment_score[gt_id, tr_id] * IoU`` where the global alignment is a
  Jaccard score over *similarity-count* accumulations, and the per-frame
  Hungarian is then thresholded at each alpha. ``AssA`` averages the
  whole-sequence per-pair association accuracy over matched detections.
* All rates/means are natural units in ``[0, 1]`` (no ``x100``); undefined
  quantities are ``nan`` and every one of them ships with its count denominator.

Units
-----
Every metric is returned in **natural units**: ``MOTA`` in ``[-1, 1]``,
``IDF1``/``HOTA``/``AssA``/``DetA``/``LocA`` in ``[0, 1]``, ``MOTP`` = mean IoU
of matched pairs in ``[0, 1]`` (higher is better, as in TrackEval),
``FP``/``FN``/``IDSW``/``Frag``/``MT``/``ML``/``TP`` as integer counts. Use
:func:`to_percent` to render a TrackEval-style percentage table.

Box format
----------
GT and prediction boxes are keyed by **0-based frame index**; each box carries
``(frame, track_id, xyxy)`` in absolute pixels. ``GtBox`` additionally carries
``conf``/``visibility`` (MOTChallenge ``gt.txt`` columns 7 and 10) and
``PredBox`` a detection ``score``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import EvalCfg, StateCfg
from .types import TargetEvent, TargetStatus, iou_matrix

__all__ = [
    "HOTA_ALPHAS",
    "BLIND_STATES",
    "GtBox",
    "PredBox",
    "combine_mot",
    "combine_target",
    "evaluate_mot",
    "evaluate_target",
    "mot_frame_data",
    "target_frame_table",
    "to_percent",
    "write_frame_csv",
]

#: HOTA alpha sweep — identical to TrackEval's ``np.arange(0.05, 0.99, 0.05)``.
HOTA_ALPHAS: np.ndarray = np.arange(0.05, 0.99, 0.05)

#: IoU required for a GT box in ``all_gt`` to be identified as the target's own
#: identity when only the target's box (not its id) is known.
_GT_ID_IOU = 0.9

#: The tracker's own "blind" states — while in one of these the reported box is
#: being extrapolated and the tracker makes **no claim** to be locked on anyone.
#: Read from :class:`~tgtrack.config.StateCfg` (the same tuple
#: ``TargetTracker.memory_frozen`` uses; ``TargetTracker.is_blind`` compares
#: against the equivalent ``BLIND_STATES`` frozenset in ``target_tracker.py``) so
#: the metric can never drift from the state machine.
BLIND_STATES: tuple[str, ...] = tuple(str(s) for s in StateCfg().frozen_states)

_BLIND_STATE_SET = frozenset(BLIND_STATES)

#: Keys checked, in order, in ``TargetStatus.diag`` for a tracker-reported
#: all-person track id of the followed person.
_DIAG_ID_KEYS = ("track_id", "matched_track_id", "gt_id", "target_track_id")


# --------------------------------------------------------------------------- #
# Box records
# --------------------------------------------------------------------------- #
@dataclass
class GtBox:
    """One ground-truth box.

    Attributes:
        frame: 0-based frame index.
        track_id: GT identity (MOTChallenge ``id`` column).
        xyxy: absolute-pixel ``[x0, y0, x1, y1]``.
        conf: MOTChallenge ``conf`` column (1 = valid/considered).
        visibility: MOTChallenge ``z`` column, the visible fraction in ``[0, 1]``;
            ``1.0`` when the format carries no visibility channel (MOT15/16).
    """

    frame: int
    track_id: int
    xyxy: np.ndarray
    conf: float = 1.0
    visibility: float = 1.0

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float64).reshape(4)

    @classmethod
    def from_xywh(
        cls,
        frame: int,
        track_id: int,
        xywh: Sequence[float],
        conf: float = 1.0,
        visibility: float = 1.0,
    ) -> "GtBox":
        """Build from MOTChallenge/COCO ``(left, top, w, h)``."""
        x, y, w, h = (float(v) for v in xywh)
        return cls(frame, track_id, [x, y, x + w, y + h], conf, visibility)

    @property
    def area(self) -> float:
        w = max(0.0, self.xyxy[2] - self.xyxy[0])
        h = max(0.0, self.xyxy[3] - self.xyxy[1])
        return w * h


@dataclass
class PredBox:
    """One tracker output box (same conventions as :class:`GtBox`)."""

    frame: int
    track_id: int
    xyxy: np.ndarray
    score: float = 1.0

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float64).reshape(4)

    @property
    def area(self) -> float:
        w = max(0.0, self.xyxy[2] - self.xyxy[0])
        h = max(0.0, self.xyxy[3] - self.xyxy[1])
        return w * h


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clean(obj: Any) -> Any:
    """Recursively coerce NumPy scalars/arrays to plain Python values."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _box_array(boxes: Sequence[Any]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.stack([np.asarray(b.xyxy, dtype=np.float64).reshape(4) for b in boxes])


def _pair_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """IoU of two single boxes, 0.0 when either is missing/degenerate."""
    if a is None or b is None:
        return 0.0
    return float(iou_matrix(np.asarray(a, dtype=np.float64).reshape(1, 4),
                            np.asarray(b, dtype=np.float64).reshape(1, 4))[0, 0])


def _nanmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _p90(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(vals, 90)) if vals else float("nan")


def _rate(num: int, den: int) -> float:
    """num/den, or nan when the denominator is empty."""
    return float(num) / float(den) if den > 0 else float("nan")


def to_percent(result: dict[str, Any], keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Scale rate-like metrics by 100 (TrackEval/MOTChallenge table style)."""
    names = list(keys) if keys is not None else [
        "MOTA", "MOTP", "MODA", "sMOTA", "MOTAL", "CLR_Re", "CLR_Pr", "CLR_F1",
        "MTR", "PTR", "MLR", "IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "LocA",
        "DetRe", "DetPr", "AssRe", "AssPr", "OWTA",
    ]
    out = dict(result)
    for k in names:
        v = out.get(k)
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            out[k] = float(v) * 100.0
    return out


# --------------------------------------------------------------------------- #
# Frame-wise data assembly
# --------------------------------------------------------------------------- #
def mot_frame_data(
    gt: dict[int, list[GtBox]],
    pred: dict[int, list[PredBox]],
) -> dict[str, Any]:
    """Pack frame-keyed boxes into TrackEval's per-sequence arrays.

    Returns a dict with ``num_timesteps``, ``num_gt_ids``, ``num_tracker_ids``,
    ``num_gt_dets``, ``num_tracker_dets``, ``gt_ids``/``tracker_ids`` (lists of
    int arrays of *compact* ids per frame), ``similarity_scores`` (list of
    ``(n_gt_t, n_tr_t)`` IoU matrices), plus ``frames``, ``gt_id_list``,
    ``pred_id_list`` and the raw box arrays for reuse.
    """
    frames = sorted(set(gt) | set(pred))
    gt_id_map: dict[int, int] = {}
    pred_id_map: dict[int, int] = {}

    gt_ids: list[np.ndarray] = []
    tr_ids: list[np.ndarray] = []
    sims: list[np.ndarray] = []
    gt_boxes: list[np.ndarray] = []
    pred_boxes: list[np.ndarray] = []
    gt_rows: list[list[GtBox]] = []
    pred_rows: list[list[PredBox]] = []
    num_gt_dets = 0
    num_tr_dets = 0

    for f in frames:
        g = sorted(gt.get(f, []), key=lambda b: (b.track_id, b.xyxy[0]))
        p = sorted(pred.get(f, []), key=lambda b: (b.track_id, b.xyxy[0]))
        gb = _box_array(g)
        pb = _box_array(p)
        gid = np.empty(len(g), dtype=np.int64)
        pid = np.empty(len(p), dtype=np.int64)
        for i, b in enumerate(g):
            key = int(b.track_id)
            if key not in gt_id_map:
                gt_id_map[key] = len(gt_id_map)
            gid[i] = gt_id_map[key]
        for i, b in enumerate(p):
            key = int(b.track_id)
            if key not in pred_id_map:
                pred_id_map[key] = len(pred_id_map)
            pid[i] = pred_id_map[key]
        if len(g) and len(p):
            sim = np.asarray(iou_matrix(gb.astype(np.float32), pb.astype(np.float32)), dtype=np.float64)
        else:
            sim = np.zeros((len(g), len(p)), dtype=np.float64)
        gt_ids.append(gid)
        tr_ids.append(pid)
        sims.append(sim)
        gt_boxes.append(gb)
        pred_boxes.append(pb)
        gt_rows.append(list(g))
        pred_rows.append(list(p))
        num_gt_dets += len(g)
        num_tr_dets += len(p)

    return {
        "frames": frames,
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(pred_id_map),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tr_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tr_ids,
        "similarity_scores": sims,
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "gt_rows": gt_rows,
        "pred_rows": pred_rows,
        "gt_id_list": [k for k, _ in sorted(gt_id_map.items(), key=lambda kv: kv[1])],
        "pred_id_list": [k for k, _ in sorted(pred_id_map.items(), key=lambda kv: kv[1])],
    }


# --------------------------------------------------------------------------- #
# CLEAR-MOT
# --------------------------------------------------------------------------- #
def _clear_final(res: dict[str, float]) -> dict[str, float]:
    """Derived CLEAR fields — mirrors TrackEval ``CLEAR._compute_final_fields``."""
    n_ids = res["MT"] + res["ML"] + res["PT"]
    res["MTR"] = res["MT"] / max(1.0, n_ids)
    res["MLR"] = res["ML"] / max(1.0, n_ids)
    res["PTR"] = res["PT"] / max(1.0, n_ids)
    res["CLR_Re"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["CLR_Pr"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FP"])
    res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
    res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["CLR_F1"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"])
    res["FP_per_frame"] = res["CLR_FP"] / max(1.0, res["CLR_Frames"])
    safe_log_idsw = math.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
    res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / max(1.0, res["CLR_TP"] + res["CLR_FN"])
    return res


def _clear_metrics(data: dict[str, Any], threshold: float = 0.5) -> dict[str, float]:
    """CLEAR-MOT accounting identical to TrackEval ``metrics/clear.py``."""
    res: dict[str, float] = {k: 0 for k in (
        "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag",
        "MOTP_sum", "CLR_Frames",
    )}
    num_gt_ids = data["num_gt_ids"]
    if data["num_tracker_dets"] == 0:
        res["CLR_FN"] = data["num_gt_dets"]
        res["ML"] = num_gt_ids
        res["MLR"] = 1.0
        return _clear_final(res)
    if data["num_gt_dets"] == 0:
        res["CLR_FP"] = data["num_tracker_dets"]
        res["MLR"] = 1.0
        return _clear_final(res)

    gt_id_count = np.zeros(num_gt_ids, dtype=np.float64)          # for MT/ML/PT
    gt_matched_count = np.zeros(num_gt_ids, dtype=np.float64)     # for MT/ML/PT
    gt_frag_count = np.zeros(num_gt_ids, dtype=np.float64)        # for Frag
    prev_tracker_id = np.full(num_gt_ids, np.nan)                 # for IDSW (last ever)
    prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)        # for matching + Frag

    for gt_ids_t, tr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        if len(gt_ids_t) == 0:
            res["CLR_FP"] += len(tr_ids_t)
            continue
        if len(tr_ids_t) == 0:
            res["CLR_FN"] += len(gt_ids_t)
            gt_id_count[gt_ids_t] += 1
            continue

        # Prefer the tracker id matched to this GT id in the previous timestep,
        # then maximise IoU (this ordering is what makes IDSW correct).
        score_mat = (tr_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]).astype(np.float64)
        score_mat = 1000.0 * score_mat + sim
        score_mat[sim < threshold - np.finfo(float).eps] = 0.0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        keep = score_mat[match_rows, match_cols] > 0.0 + np.finfo(float).eps
        match_rows = match_rows[keep]
        match_cols = match_cols[keep]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tr_ids = tr_ids_t[match_cols]

        prev_matched_tr = prev_tracker_id[matched_gt_ids]
        is_idsw = np.logical_not(np.isnan(prev_matched_tr)) & (matched_tr_ids != prev_matched_tr)
        res["IDSW"] += float(np.sum(is_idsw))

        gt_id_count[gt_ids_t] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tr_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tr_ids
        currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

        num_matches = len(matched_gt_ids)
        res["CLR_TP"] += float(num_matches)
        res["CLR_FN"] += float(len(gt_ids_t) - num_matches)
        res["CLR_FP"] += float(len(tr_ids_t) - num_matches)
        if num_matches > 0:
            res["MOTP_sum"] += float(np.sum(sim[match_rows, match_cols]))

    tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
    res["MT"] = float(np.sum(tracked_ratio > 0.8))
    res["PT"] = float(np.sum(tracked_ratio >= 0.2)) - res["MT"]
    res["ML"] = float(num_gt_ids) - res["MT"] - res["PT"]
    res["Frag"] = float(np.sum(gt_frag_count[gt_frag_count > 0] - 1))
    res["CLR_Frames"] = float(data["num_timesteps"])
    return _clear_final(res)


# --------------------------------------------------------------------------- #
# Identity (IDF1)
# --------------------------------------------------------------------------- #
def _identity_final(res: dict[str, float]) -> dict[str, float]:
    """Derived ID fields — mirrors TrackEval ``Identity._compute_final_fields``."""
    res["IDR"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFN"])
    res["IDP"] = res["IDTP"] / max(1.0, res["IDTP"] + res["IDFP"])
    res["IDF1"] = res["IDTP"] / max(1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"])
    return res


def _identity_metrics(data: dict[str, Any], threshold: float = 0.5) -> dict[str, float]:
    """IDF1 via the global FN+FP assignment — TrackEval ``metrics/identity.py``."""
    res: dict[str, float] = {"IDTP": 0.0, "IDFN": 0.0, "IDFP": 0.0}
    if data["num_tracker_dets"] == 0:
        res["IDFN"] = float(data["num_gt_dets"])
        return _identity_final(res)
    if data["num_gt_dets"] == 0:
        res["IDFP"] = float(data["num_tracker_dets"])
        return _identity_final(res)

    num_gt_ids = data["num_gt_ids"]
    num_tr_ids = data["num_tracker_ids"]
    potential = np.zeros((num_gt_ids, num_tr_ids), dtype=np.float64)
    gt_id_count = np.zeros(num_gt_ids, dtype=np.float64)
    tr_id_count = np.zeros(num_tr_ids, dtype=np.float64)

    for gt_ids_t, tr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        matches = sim >= threshold
        mi, mt = np.nonzero(matches)
        if len(mi):
            potential[gt_ids_t[mi], tr_ids_t[mt]] += 1
        gt_id_count[gt_ids_t] += 1
        tr_id_count[tr_ids_t] += 1

    n = num_gt_ids + num_tr_ids
    fp_mat = np.zeros((n, n), dtype=np.float64)
    fn_mat = np.zeros((n, n), dtype=np.float64)
    fp_mat[num_gt_ids:, :num_tr_ids] = 1e10
    fn_mat[:num_gt_ids, num_tr_ids:] = 1e10
    for g in range(num_gt_ids):
        fn_mat[g, :num_tr_ids] = gt_id_count[g]
        fn_mat[g, num_tr_ids + g] = gt_id_count[g]
    for tr in range(num_tr_ids):
        fp_mat[:num_gt_ids, tr] = tr_id_count[tr]
        fp_mat[num_gt_ids + tr, tr] = tr_id_count[tr]
    fn_mat[:num_gt_ids, :num_tr_ids] -= potential
    fp_mat[:num_gt_ids, :num_tr_ids] -= potential

    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    res["IDFN"] = float(fn_mat[rows, cols].sum())
    res["IDFP"] = float(fp_mat[rows, cols].sum())
    res["IDTP"] = float(gt_id_count.sum() - res["IDFN"])
    return _identity_final(res)


# --------------------------------------------------------------------------- #
# HOTA
# --------------------------------------------------------------------------- #
def _hota_final(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, assa: np.ndarray,
                assre: np.ndarray, asspr: np.ndarray, loca: np.ndarray) -> dict[str, Any]:
    """Per-alpha HOTA curves and their means (TrackEval ``_compute_final_fields``)."""
    det_re = tp / np.maximum(1.0, tp + fn)
    det_pr = tp / np.maximum(1.0, tp + fp)
    deta = tp / np.maximum(1.0, tp + fn + fp)
    hota = np.sqrt(deta * assa)
    owta = np.sqrt(det_re * assa)
    curves = {
        "alphas": HOTA_ALPHAS.tolist(),
        "HOTA_curve": hota.tolist(),
        "DetA_curve": deta.tolist(),
        "AssA_curve": assa.tolist(),
        "LocA_curve": loca.tolist(),
        "DetRe_curve": det_re.tolist(),
        "DetPr_curve": det_pr.tolist(),
        "AssRe_curve": assre.tolist(),
        "AssPr_curve": asspr.tolist(),
        "HOTA_TP_curve": tp.tolist(),
        "HOTA_FN_curve": fn.tolist(),
        "HOTA_FP_curve": fp.tolist(),
    }
    return {
        "HOTA": float(np.mean(hota)),
        "DetA": float(np.mean(deta)),
        "AssA": float(np.mean(assa)),
        "LocA": float(np.mean(loca)),
        "DetRe": float(np.mean(det_re)),
        "DetPr": float(np.mean(det_pr)),
        "AssRe": float(np.mean(assre)),
        "AssPr": float(np.mean(asspr)),
        "OWTA": float(np.mean(owta)),
        "HOTA(0)": float(hota[0]),
        "LocA(0)": float(loca[0]),
        "HOTALocA(0)": float(hota[0] * loca[0]),
        "curves": curves,
    }


def _hota_metrics(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """HOTA / DetA / AssA / LocA — a faithful port of TrackEval ``metrics/hota.py``.

    Returns ``(scalars, state)``; ``state`` holds the per-alpha accumulators that
    :func:`combine_mot` needs for correct cross-sequence aggregation.
    """
    alphas = HOTA_ALPHAS
    n_a = len(alphas)
    zero = np.zeros(n_a, dtype=np.float64)
    num_gt_ids = data["num_gt_ids"]
    num_tr_ids = data["num_tracker_ids"]
    num_gt_dets = int(data["num_gt_dets"])
    num_tr_dets = int(data["num_tracker_dets"])

    def _state(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, assa: np.ndarray,
               assre: np.ndarray, asspr: np.ndarray, loca: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        scalars = _hota_final(tp, fn, fp, assa, assre, asspr, loca)
        state = {
            "alphas": alphas.copy(),
            "hota_tp": tp, "hota_fn": fn, "hota_fp": fp,
            "assa": assa, "assre": assre, "asspr": asspr, "loca": loca,
        }
        return scalars, state

    # TrackEval returns early for empty sequences: HOTA/DetA/AssA stay 0 while
    # LocA is set to 1.0 by the ``1e-10 / 1e-10`` convention of its final field
    # computation (no matched pair can *lower* the localisation accuracy).
    if num_tr_dets == 0:
        loca = np.full(n_a, 1.0)
        return _state(zero.copy(), np.full(n_a, float(num_gt_dets)), zero.copy(), zero.copy(), zero.copy(), zero.copy(), loca)
    if num_gt_dets == 0:
        return _state(zero.copy(), zero.copy(), np.full(n_a, float(num_tr_dets)), zero.copy(), zero.copy(), zero.copy(), np.full(n_a, 1.0))

    # --- global alignment (ID-level) -------------------------------------- #
    potential_matches_count = np.zeros((num_gt_ids, num_tr_ids), dtype=np.float64)
    gt_id_count = np.zeros((num_gt_ids, 1), dtype=np.float64)
    tracker_id_count = np.zeros((1, num_tr_ids), dtype=np.float64)

    for gt_ids_t, tr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        # NOTE: no ``sim.size == 0`` fast path here.  TrackEval accumulates the
        # per-id detection counts on *every* timestep, including frames where
        # only one side has detections; skipping those frames under-counts
        # ``gt_id_count``/``tracker_id_count`` and therefore inflates AssA.
        # The fancy-indexed adds below are no-ops for an empty index array.
        denom = sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] - sim
        sim_iou = np.zeros_like(sim)
        mask = denom > np.finfo(np.float64).eps
        sim_iou[mask] = sim[mask] / denom[mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tr_ids_t[np.newaxis, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tr_ids_t] += 1

    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in alphas]

    # --- per-frame matching + accumulation -------------------------------- #
    hota_tp = np.zeros(n_a, dtype=np.float64)
    hota_fn = np.zeros(n_a, dtype=np.float64)
    hota_fp = np.zeros(n_a, dtype=np.float64)
    loca_sum = np.zeros(n_a, dtype=np.float64)

    for gt_ids_t, tr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        if len(gt_ids_t) == 0:
            hota_fp += len(tr_ids_t)
            continue
        if len(tr_ids_t) == 0:
            hota_fn += len(gt_ids_t)
            continue

        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tr_ids_t[np.newaxis, :]] * sim
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched_sim = sim[match_rows, match_cols]
        for a, alpha in enumerate(alphas):
            keep = matched_sim >= alpha - np.finfo(np.float64).eps
            mr = match_rows[keep]
            mc = match_cols[keep]
            num_matches = len(mr)
            hota_tp[a] += num_matches
            hota_fn[a] += len(gt_ids_t) - num_matches
            hota_fp[a] += len(tr_ids_t) - num_matches
            if num_matches > 0:
                loca_sum[a] += float(np.sum(sim[mr, mc]))
                matches_counts[a][gt_ids_t[mr], tr_ids_t[mc]] += 1

    # --- association accuracy --------------------------------------------- #
    assa = np.zeros(n_a, dtype=np.float64)
    assre = np.zeros(n_a, dtype=np.float64)
    asspr = np.zeros(n_a, dtype=np.float64)
    for a in range(n_a):
        mc = matches_counts[a]
        a_denom = gt_id_count + tracker_id_count - mc
        assa[a] = float(np.sum(mc * (mc / np.maximum(1.0, a_denom)))) / max(1.0, hota_tp[a])
        assre[a] = float(np.sum(mc * (mc / np.maximum(1.0, gt_id_count)))) / max(1.0, hota_tp[a])
        asspr[a] = float(np.sum(mc * (mc / np.maximum(1.0, tracker_id_count)))) / max(1.0, hota_tp[a])

    loca = np.maximum(1e-10, loca_sum) / np.maximum(1e-10, hota_tp)
    return _state(hota_tp, hota_fn, hota_fp, assa, assre, asspr, loca)


# --------------------------------------------------------------------------- #
# Public MOT entry point
# --------------------------------------------------------------------------- #
def evaluate_mot(
    gt: dict[int, list[GtBox]],
    pred: dict[int, list[PredBox]],
    cfg: EvalCfg | None = None,
    return_curves: bool = False,
    return_state: bool = False,
) -> dict[str, Any]:
    """Evaluate the all-person MOT layer.

    Args:
        gt: GT boxes keyed by 0-based frame index.
        pred: tracker boxes keyed by 0-based frame index.
        cfg: :class:`EvalCfg`; ``iou_thresh`` is used for CLEAR and Identity
            matching (TrackEval's default is 0.5 for both).
        return_curves: also attach the per-alpha HOTA curves under ``"curves"``.
        return_state: also attach the raw accumulators under ``"state"`` so the
            result can be aggregated with :func:`combine_mot`.

    Returns:
        Flat ``dict[str, float]`` with ``MOTA``/``MOTP``/``MODA``/``sMOTA``/
        ``MOTAL``/``MOTP_sum``/``IDSW``/``Frag``/``MT``/``PT``/``ML``/``MTR``/
        ``PTR``/``MLR``/``CLR_TP``/``CLR_FN``/``CLR_FP``/``CLR_Re``/``CLR_Pr``/
        ``CLR_F1``/``FP_per_frame``/``CLR_Frames`` (aliases ``TP``/``FP``/``FN``),
        ``IDF1``/``IDP``/``IDR``/``IDTP``/``IDFP``/``IDFN``,
        ``HOTA``/``DetA``/``AssA``/``LocA``/``DetRe``/``DetPr``/``AssRe``/
        ``AssPr``/``OWTA``/``HOTA(0)``/``LocA(0)``/``HOTALocA(0)`` and the
        dataset counters ``num_gt_dets``/``num_pred_dets``/``num_frames``/
        ``num_gt_ids``/``num_pred_ids``/``iou_thresh``.
    """
    cfg = cfg or EvalCfg()
    thr = float(cfg.iou_thresh)
    data = mot_frame_data(gt, pred)

    clear = _clear_metrics(data, thr)
    identity = _identity_metrics(data, thr)
    hota, hota_state = _hota_metrics(data)

    out: dict[str, Any] = {}
    out.update(clear)
    out.update(identity)
    out.update({k: v for k, v in hota.items() if k != "curves"})
    # Convenience aliases with the names most reports use.
    out["TP"] = int(clear["CLR_TP"])
    out["FP"] = int(clear["CLR_FP"])
    out["FN"] = int(clear["CLR_FN"])
    out["num_gt_dets"] = int(data["num_gt_dets"])
    out["num_pred_dets"] = int(data["num_tracker_dets"])
    out["num_frames"] = int(data["num_timesteps"])
    out["num_gt_ids"] = int(data["num_gt_ids"])
    out["num_pred_ids"] = int(data["num_tracker_ids"])
    out["iou_thresh"] = thr

    if return_curves:
        out["curves"] = hota["curves"]
    if return_state:
        out["state"] = {
            **hota_state,
            "clear": {k: float(clear[k]) for k in (
                "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag",
                "MOTP_sum", "CLR_Frames",
            )},
            "identity": {k: float(identity[k]) for k in ("IDTP", "IDFN", "IDFP")},
            "num_gt_dets": float(data["num_gt_dets"]),
            "num_pred_dets": float(data["num_tracker_dets"]),
            "num_frames": float(data["num_timesteps"]),
            "num_gt_ids": float(data["num_gt_ids"]),
            "num_pred_ids": float(data["num_tracker_ids"]),
            "iou_thresh": thr,
        }
    return _clean(out)


def combine_mot(results: Sequence[dict[str, Any]], cfg: EvalCfg | None = None) -> dict[str, Any]:
    """Aggregate ``evaluate_mot(..., return_state=True)`` results across sequences.

    Mirrors TrackEval's ``combine_sequences``: count fields are summed, HOTA
    ``AssA``/``AssRe``/``AssPr``/``LocA`` are weighted by ``HOTA_TP``, ``DetA``
    and ``HOTA`` are recomputed from the summed TP/FN/FP. Averaging per-sequence
    HOTA/IDF1 instead would be wrong.
    """
    states = [r["state"] for r in results if "state" in r]
    if not states:
        raise ValueError("combine_mot needs results from evaluate_mot(..., return_state=True)")
    cfg = cfg or EvalCfg()

    def _sum(field: str) -> float:
        return float(sum(float(s[field]) for s in states))

    clear = {k: float(sum(s["clear"][k] for s in states)) for k in (
        "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "MOTP_sum", "CLR_Frames",
    )}
    identity = {k: float(sum(s["identity"][k] for s in states)) for k in ("IDTP", "IDFN", "IDFP")}
    tp = sum(np.asarray(s["hota_tp"], dtype=np.float64) for s in states)
    fn = sum(np.asarray(s["hota_fn"], dtype=np.float64) for s in states)
    fp = sum(np.asarray(s["hota_fp"], dtype=np.float64) for s in states)
    w = np.maximum(1e-10, tp)

    def _wav(field: str) -> np.ndarray:
        num = sum(np.asarray(s[field], dtype=np.float64) * np.asarray(s["hota_tp"], dtype=np.float64) for s in states)
        return np.maximum(1e-10, num) / w

    hota = _hota_final(tp, fn, fp, _wav("assa"), _wav("assre"), _wav("asspr"), _wav("loca"))
    out: dict[str, Any] = {}
    out.update(_clear_final(dict(clear)))
    out.update(_identity_final(dict(identity)))
    out.update({k: v for k, v in hota.items() if k != "curves"})
    out["TP"] = int(clear["CLR_TP"])
    out["FP"] = int(clear["CLR_FP"])
    out["FN"] = int(clear["CLR_FN"])
    out["num_gt_dets"] = int(sum(s["num_gt_dets"] for s in states))
    out["num_pred_dets"] = int(sum(s["num_pred_dets"] for s in states))
    out["num_frames"] = int(sum(s["num_frames"] for s in states))
    out["num_gt_ids"] = int(sum(s["num_gt_ids"] for s in states))
    out["num_pred_ids"] = int(sum(s["num_pred_ids"] for s in states))
    out["iou_thresh"] = float(cfg.iou_thresh)
    out["num_sequences"] = len(states)
    return _clean(out)


# --------------------------------------------------------------------------- #
# Target-centric metrics
# --------------------------------------------------------------------------- #
def _diag_track_id(status: TargetStatus | None) -> int | None:
    """Tracker-reported followed track id, if the tracker put one in ``diag``."""
    if status is None or not status.diag:
        return None
    for key in _DIAG_ID_KEYS:
        if key in status.diag:
            value = status.diag[key]
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _run_lengths(flags: Sequence[bool]) -> tuple[list[list[int]], list[list[int]]]:
    """Split a boolean sequence into runs of True/False as index lists."""
    runs_t: list[list[int]] = []
    runs_f: list[list[int]] = []
    cur: list[int] = []
    cur_val: bool | None = None
    for i, flag in enumerate(flags):
        flag = bool(flag)
        if cur_val is None or flag == cur_val:
            cur.append(i)
        else:
            (runs_t if cur_val else runs_f).append(cur)
            cur = [i]
        cur_val = flag
    if cur:
        (runs_t if cur_val else runs_f).append(cur)
    return runs_t, runs_f


def target_frame_table(
    gt_target: dict[int, np.ndarray | None],
    statuses: dict[int, TargetStatus],
    all_gt: dict[int, list[GtBox]] | None = None,
    cfg: EvalCfg | None = None,
) -> list[dict[str, Any]]:
    """Per-frame debug table behind :func:`evaluate_target`.

    One row per frame in the union of ``gt_target``/``statuses`` keys, with:
    ``frame``, ``gt_present``, ``gt_xyxy``, ``target_person_id``,
    ``target_person_id_iou``, ``reported``, ``state``, ``locked``,
    ``pred_xyxy``, ``iou_target``, ``best_gt_id``, ``best_gt_iou``, ``correct``,
    ``false_acquisition``, ``followed_id``, ``reacquired``,
    ``frames_since_update``, ``age``.

    ``locked`` is ``True`` when the tracker claims to hold the target this frame,
    i.e. its ``state`` is not one of :data:`BLIND_STATES`.  A frame with no
    recorded status carries no such claim, so it is **not** counted as a
    confident lock (it still counts in the total rates).
    """
    cfg = cfg or EvalCfg()
    all_gt = all_gt or {}
    thr = float(cfg.iou_thresh)
    frames = sorted(set(gt_target) | set(statuses))
    rows: list[dict[str, Any]] = []

    for f in frames:
        gt_box = gt_target.get(f)
        gt_arr = None if gt_box is None else np.asarray(gt_box, dtype=np.float64).reshape(4)
        st = statuses.get(f)
        rep = None
        if st is not None and st.xyxy is not None:
            rep = np.asarray(st.xyxy, dtype=np.float64).reshape(4)

        boxes = list(all_gt.get(f, []))
        target_pid: int | None = None
        target_pid_iou: float = float("nan")
        if gt_arr is not None and boxes:
            box_arr = _box_array(boxes)
            ious_gt = iou_matrix(gt_arr.reshape(1, 4), box_arr)[0]
            k = int(np.argmax(ious_gt))
            target_pid_iou = float(ious_gt[k])
            if target_pid_iou >= _GT_ID_IOU:
                target_pid = int(boxes[k].track_id)

        best_gt_id: int | None = None
        best_gt_iou = 0.0
        matched_id: int | None = None
        if rep is not None and boxes:
            box_arr = _box_array(boxes)
            ious_rep = iou_matrix(rep.reshape(1, 4), box_arr)[0]
            k = int(np.argmax(ious_rep))
            best_gt_iou = float(ious_rep[k])
            best_gt_id = int(boxes[k].track_id)
            if best_gt_iou >= thr:
                matched_id = best_gt_id

        iou_target = _pair_iou(rep, gt_arr) if (rep is not None and gt_arr is not None) else float("nan")
        correct = bool(rep is not None and gt_arr is not None and iou_target >= thr)

        # False acquisition: latched onto a real, different person.
        fa = False
        if rep is not None:
            if gt_arr is None:
                fa = matched_id is not None
            elif matched_id is not None and target_pid is not None and matched_id != target_pid \
                    and best_gt_iou > max(iou_target, thr):
                fa = True
            elif matched_id is not None and target_pid is None and best_gt_iou > max(iou_target, thr):
                fa = True

        diag_id = _diag_track_id(st)
        followed_id = diag_id if diag_id is not None else matched_id

        state = None if st is None else str(st.state)
        # A confident lock needs an explicit non-blind state: a frame with no
        # status recorded at all makes no claim about who is being followed.
        locked = state is not None and state not in _BLIND_STATE_SET

        # E077: the per-frame CHANNEL costs of the candidate the tracker selected
        # (whether or not it was accepted), plus which conclusive channel carried
        # an otherwise-refused acceptance.  They are what makes the two arms
        # auditable after the fact: the pre-registered engagement check is the
        # `blind AND gt_present` share, and the arm-C mechanism is only visible in
        # the box geometry, which this dump carries.  Absent diag keys become
        # empty CSV fields, so a dump from a build without E077 still parses.
        dg = dict(getattr(st, "diag", {}) or {}) if st is not None else {}

        def _num(key: str) -> float | None:
            v = dg.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        rows.append({
            "frame": int(f),
            "gt_present": bool(gt_arr is not None),
            "gt_xyxy": None if gt_arr is None else [float(v) for v in gt_arr],
            "target_person_id": target_pid,
            "target_person_id_iou": target_pid_iou,
            "reported": bool(rep is not None),
            "state": state,
            "locked": bool(locked),
            "pred_xyxy": None if rep is None else [float(v) for v in rep],
            "iou_target": iou_target,
            "best_gt_id": best_gt_id,
            "best_gt_iou": best_gt_iou,
            "correct": correct,
            "false_acquisition": bool(fa),
            "followed_id": followed_id,
            "followed_id_from_diag": diag_id is not None,
            "ch_norm_centre": _num("ch_norm_centre"),
            "ch_log_h_ratio": _num("ch_log_h_ratio"),
            "ch_app_cost": _num("ch_app_cost"),
            "ch_app_valid": None if "ch_app_valid" not in dg else bool(dg.get("ch_app_valid")),
            "accept_channel": "" if dg.get("accept_channel") is None else str(dg.get("accept_channel")),
            "refuse_reason": "" if dg.get("refuse_reason") is None else str(dg.get("refuse_reason")),
            # E079: the E029 band actually used this frame, whether it was pinned at
            # its ceiling, and what the reference vote did -- so the `blind AND present`
            # share can be attributed to the veto rather than only counted.
            "anchor_band": _num("anchor_band"),
            "anchor_band_sat": None if "anchor_band_sat" not in dg else bool(dg.get("anchor_band_sat")),
            "vote_n": _num("vote_n"),
            "vote_r2": _num("vote_r2"),
            "vote_r3": _num("vote_r3"),
            "vote_rescued": _num("vote_rescued"),
            "vote_rescued_accepted": _num("vote_rescued_accepted"),
            "reacquired": bool(st.reacquired) if st is not None else False,
            "frames_since_update": int(st.frames_since_update) if st is not None else -1,
            "age": int(st.age) if st is not None else -1,
        })
    return rows


def write_frame_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write a :func:`target_frame_table` dump to CSV (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "gt_present", "gt_xyxy", "target_person_id", "target_person_id_iou",
        "reported", "state", "locked", "pred_xyxy", "iou_target", "best_gt_id", "best_gt_iou",
        "correct", "false_acquisition", "followed_id", "followed_id_from_diag",
        "ch_norm_centre", "ch_log_h_ratio", "ch_app_cost", "ch_app_valid", "accept_channel",
        "refuse_reason",
        "anchor_band", "anchor_band_sat",
        "vote_n", "vote_r2", "vote_r3", "vote_rescued", "vote_rescued_accepted",
        "reacquired", "frames_since_update", "age",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    return path


def _first_correct(rows: Sequence[dict[str, Any]], start: int, window: int) -> int | None:
    """First frame index in ``[start, start+window]`` where the tracker is correct."""
    stop = min(len(rows), start + window + 1)
    for i in range(max(0, start), stop):
        if rows[i]["correct"]:
            return i
    return None


def evaluate_target(
    gt_target: dict[int, np.ndarray | None],
    statuses: dict[int, TargetStatus],
    events: list[TargetEvent],
    all_gt: dict[int, list[GtBox]],
    cfg: EvalCfg | None = None,
    debug_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Target-centric metrics for the clicked (DJI-style) target.

    Args:
        gt_target: per-frame target GT box (``xyxy``) or ``None`` when the target
            is not visible — see :func:`tgtrack.data.make_target_gt`.
        statuses: per-frame :class:`TargetStatus`; missing frames count as
            "reported nothing".
        events: :class:`TargetEvent` list from the tracker. Used for the
            ``n_*_events`` diagnostics, for the initial clicked identity and as a
            cross-check — the quantitative rates are computed geometrically so
            they cannot be gamed by tracker self-reporting.
        all_gt: all-person GT for the sequence, used to decide *which* identity a
            report latched onto (identity consistency and false acquisition).
        cfg: :class:`EvalCfg`; ``iou_thresh`` for a correct match,
            ``reacquire_window`` for the re-acquisition latency budget.
        debug_csv: when given, dump :func:`target_frame_table` there.

    Returns:
        Flat ``dict[str, float]``. Undefined quantities are ``nan``; every rate
        ships with its integer denominator so callers can mask/filter:

        * ``target_success_rate`` = correct frames / visible frames
          (``target_success_frames`` over ``n_visible_frames``).
        * ``loss_rate`` = (visible frames with no report **or** a wrong box) /
          visible frames, decomposed into ``no_report_rate`` and
          ``wrong_box_rate``.
        * ``target_precision`` = mean IoU over frames with a report (absent
          target counts as IoU 0); ``target_precision_visible`` restricts to
          frames where the target is visible.
        * ``reacquire_rate`` over ``reacquire_events`` gap events (pooled GT
          absence gaps and tracker-lost-while-visible gaps, reported separately
          as ``*_gt_gap``/``*_tracker_lost``), with ``reacquire_latency``
          (mean), ``reacquire_latency_p90`` and ``reacquire_latency_max`` in
          frames from re-appearance to first correct lock.
        * ``false_acquisition_rate`` = frames latched onto a different person /
          frames with a report (``false_acquisition_frames``), plus the count of
          distinct latch-on runs ``false_acquisition_events``;
          ``false_acquisition_rate_all_frames`` uses every frame as denominator.
        * Confident (locked) vs blind-coast failures — the same failure counted
          against the frames where the tracker *claimed* to hold the target
          (``state`` not in :data:`BLIND_STATES`), so the final report can state
          both numbers without conflating them:

          - ``false_acquisition_rate_locked`` = ``false_acquisition_frames_locked``
            / ``n_reported_locked`` (reports made while locked), with
            ``false_acquisition_events_locked`` counting distinct locked
            latch-on runs. The headline ``false_acquisition_rate`` still counts
            **every** latch, locked or not, so this split can never excuse a
            confident wrong lock.
          - ``loss_rate_locked`` = ``loss_frames_locked`` / visible frames:
            visible frames where the tracker was locked yet produced no box or a
            wrong one. ``loss_rate`` remains the total.
        * ``id_consistency`` = reporting frames still on the initially clicked
          identity / reporting frames with a determined identity, plus
          ``target_switch_count`` and ``distinct_followed_ids``.
        * ``fragmentation``, ``longest_correct_streak``, ``longest_lost_streak``,
          ``mean_correct_streak``.
    """
    cfg = cfg or EvalCfg()
    rows = target_frame_table(gt_target, statuses, all_gt, cfg)
    if debug_csv is not None:
        write_frame_csv(debug_csv, rows)
    n_frames = len(rows)
    thr = float(cfg.iou_thresh)
    window = int(cfg.reacquire_window)

    visible = [bool(r["gt_present"]) for r in rows]
    reported = [bool(r["reported"]) for r in rows]
    correct = [bool(r["correct"]) for r in rows]
    fa = [bool(r["false_acquisition"]) for r in rows]
    locked = [bool(r["locked"]) for r in rows]

    n_visible = sum(visible)
    n_reported = sum(reported)
    n_correct = sum(correct)
    n_fa = sum(fa)
    n_absent = n_frames - n_visible

    # --- confident (locked) vs blind-coast failures ------------------------ #
    locked_fa = [bool(fa[i] and locked[i]) for i in range(n_frames)]
    n_locked = sum(locked)
    n_reported_locked = sum(1 for i in range(n_frames) if reported[i] and locked[i])
    n_fa_locked = sum(locked_fa)
    fa_locked_runs, _ = _run_lengths(locked_fa)
    n_loss_locked = sum(
        1 for i in range(n_frames)
        if visible[i] and locked[i] and (not reported[i] or not correct[i])
    )

    # --- success / loss ---------------------------------------------------- #
    tsr = _rate(n_correct, n_visible)
    n_no_report = sum(1 for i in range(n_frames) if visible[i] and not reported[i])
    n_wrong_box = sum(1 for i in range(n_frames) if visible[i] and reported[i] and not correct[i])
    loss_rate = _rate(n_no_report + n_wrong_box, n_visible)

    # Mean IoU over frames with a report; a report while the target is absent has
    # no target box to score against, so it counts as IoU 0 (that is the penalty
    # for reporting someone who is not the target).
    prec_all: list[float] = []
    prec_vis: list[float] = []
    for i in range(n_frames):
        if not reported[i]:
            continue
        value = rows[i]["iou_target"]
        value = 0.0 if (value is None or math.isnan(float(value))) else float(value)
        prec_all.append(value)
        if visible[i]:
            prec_vis.append(value)

    # --- re-acquisition ---------------------------------------------------- #
    def _first_correct_index(start: int) -> int | None:
        """Row index of the first correct frame in the re-acquisition window."""
        return _first_correct(rows, start, window)

    def _score(event_start: int) -> tuple[bool, float, bool]:
        """``(succeeded, latency_frames, censored)`` for one gap event."""
        hit = _first_correct_index(event_start)
        censored = event_start + window >= n_frames
        if hit is None:
            return False, float("nan"), censored
        return True, float(hit - event_start), False

    # Event type A — the GT target leaves the scene and comes back.
    # The latency clock starts at the first frame the target is visible again.
    _, absent_runs = _run_lengths(visible)
    gt_gap_events: list[int] = []
    for run in absent_runs:
        a, b = run[0], run[-1]
        if a > 0 and b < n_frames - 1 and visible[a - 1] and visible[b + 1]:
            gt_gap_events.append(b + 1)

    # Event type B — the target is still visible but the tracker stops reporting
    # (gave up / hard lost / occluded-while-visible). The latency clock starts at
    # the first frame of the run with no report.
    lost_flags = [visible[i] and not reported[i] for i in range(n_frames)]
    lost_runs, _ = _run_lengths(lost_flags)
    lost_events: list[int] = [run[0] for run in lost_runs]

    def _summarize(starts: list[int]) -> dict[str, Any]:
        ok, latencies, censored = 0, [], 0
        for start in starts:
            good, latency, cens = _score(start)
            if good:
                ok += 1
                latencies.append(latency)
            elif cens:
                censored += 1
        return {
            "events": len(starts),
            "ok": ok,
            "rate": _rate(ok, len(starts)),
            "latency": _nanmean(latencies),
            "latency_p90": _p90(latencies),
            "latency_max": float(max(latencies)) if latencies else float("nan"),
            "censored": censored,
            "latencies": latencies,
        }

    s_gt = _summarize(gt_gap_events)
    s_lost = _summarize(lost_events)
    all_latencies = s_gt["latencies"] + s_lost["latencies"]
    n_reacq_events = s_gt["events"] + s_lost["events"]
    n_reacq_ok = s_gt["ok"] + s_lost["ok"]

    # --- false acquisition ------------------------------------------------- #
    fa_runs_true, _ = _run_lengths(fa)
    # --- identity consistency ---------------------------------------------- #
    followed = [r["followed_id"] for r in rows if r["reported"]]
    determined = [fid for fid in followed if fid is not None]
    initial_id: int | None = None
    for ev in events:
        if ev.kind == "init":
            for key in _DIAG_ID_KEYS:
                if ev.detail.get(key) is not None:
                    try:
                        initial_id = int(ev.detail[key])
                    except (TypeError, ValueError):
                        initial_id = None
                    break
        if initial_id is not None:
            break
    if initial_id is None and determined:
        initial_id = determined[0]

    n_on_initial = sum(1 for fid in determined if fid == initial_id)
    switches = sum(1 for a, b in zip(determined, determined[1:]) if a != b)

    # --- structure --------------------------------------------------------- #
    correct_runs, _ = _run_lengths(correct)
    longest_correct = max((len(r) for r in correct_runs), default=0)
    mean_correct_streak = float(np.mean([len(r) for r in correct_runs])) if correct_runs else float("nan")

    # A frame is "in play" when the target is visible *or* the tracker claims to
    # be tracking something.  Frames where the target is gone and nothing is
    # reported carry no information about the lock and are skipped, but a
    # hallucinated report while the target is absent IS part of a broken lock.
    in_play_rows = [r for r in rows if r["gt_present"] or r["reported"]]
    vis_correct = [bool(r["correct"]) for r in in_play_rows]
    _, vis_broken = _run_lengths(vis_correct)
    longest_lost = max((len(r) for r in vis_broken), default=0)
    # Target fragmentation: interior runs of "in play but not correctly tracked"
    # (a broken lock that the tracker later recovers from) — same convention as
    # CLEAR-MOT's Frag, restricted to the target.
    fragmentation = sum(
        1 for run in vis_broken if run[0] > 0 and run[-1] < len(vis_correct) - 1
    )

    event_counts: dict[str, int] = {}
    for ev in events:
        event_counts[str(ev.kind)] = event_counts.get(str(ev.kind), 0) + 1

    out: dict[str, Any] = {
        # counters
        "n_frames": n_frames,
        "n_visible_frames": n_visible,
        "n_absent_frames": n_absent,
        "n_report_frames": n_reported,
        "n_correct_frames": n_correct,
        "n_false_acquisition_frames": n_fa,
        "initial_target_id": initial_id,
        # success / loss
        "target_success_rate": tsr,
        "target_success_frames": n_correct,
        "loss_rate": loss_rate,
        "no_report_rate": _rate(n_no_report, n_visible),
        "wrong_box_rate": _rate(n_wrong_box, n_visible),
        "no_report_frames": n_no_report,
        "wrong_box_frames": n_wrong_box,
        "target_precision": _nanmean(prec_all),
        "target_precision_visible": _nanmean(prec_vis),
        # Kept so :func:`combine_target` can pool the mean IoU exactly
        # (sum of IoUs / sum of reporting frames) instead of averaging means.
        "target_precision_sum": float(sum(prec_all)),
        # re-acquisition
        "reacquire_events": n_reacq_events,
        "reacquire_ok": n_reacq_ok,
        "reacquire_rate": _rate(n_reacq_ok, n_reacq_events),
        "reacquire_fail": n_reacq_events - n_reacq_ok,
        "reacquire_censored": s_gt["censored"] + s_lost["censored"],
        "reacquire_latency": _nanmean(all_latencies),
        "reacquire_latency_p90": _p90(all_latencies),
        "reacquire_latency_max": float(max(all_latencies)) if all_latencies else float("nan"),
        "reacquire_events_gt_gap": s_gt["events"],
        "reacquire_ok_gt_gap": s_gt["ok"],
        "reacquire_rate_gt_gap": s_gt["rate"],
        "reacquire_latency_gt_gap": s_gt["latency"],
        "reacquire_events_tracker_lost": s_lost["events"],
        "reacquire_ok_tracker_lost": s_lost["ok"],
        "reacquire_rate_tracker_lost": s_lost["rate"],
        "reacquire_latency_tracker_lost": s_lost["latency"],
        # false acquisition (the DJI-critical failure mode)
        "false_acquisition_rate": _rate(n_fa, n_reported),
        "false_acquisition_rate_all_frames": _rate(n_fa, n_frames),
        "false_acquisition_frames": n_fa,
        "false_acquisition_events": len(fa_runs_true),
        # The same failure restricted to the frames where the tracker claimed to
        # hold the target (state not in BLIND_STATES).  A refusal that then
        # coasts onto a bystander shows up in the total but not here; a
        # confident wrong lock shows up in both.  Keep both readings.
        "false_acquisition_rate_locked": _rate(n_fa_locked, n_reported_locked),
        "false_acquisition_frames_locked": n_fa_locked,
        "false_acquisition_events_locked": len(fa_locked_runs),
        "n_locked_frames": n_locked,
        "n_reported_locked": n_reported_locked,
        "loss_frames_locked": n_loss_locked,
        "loss_rate_locked": _rate(n_loss_locked, n_visible),
        # A report that matches nobody in the all-person GT (IoU below threshold)
        # and is not a latch onto a different real person is a pure hallucination.
        "hallucination_frames": sum(
            1 for r in rows if r["reported"] and r["best_gt_iou"] < thr and not r["false_acquisition"]
        ),
        "hallucination_rate": _rate(
            sum(1 for r in rows if r["reported"] and r["best_gt_iou"] < thr and not r["false_acquisition"]),
            n_reported,
        ),
        # identity
        "id_consistency": _rate(n_on_initial, len(determined)),
        "id_consistency_frames": n_on_initial,
        "id_consistency_denominator": len(determined),
        "target_switch_count": switches,
        "distinct_followed_ids": len(set(determined)),
        "n_undetermined_id_frames": n_reported - len(determined),
        # structure
        "fragmentation": fragmentation,
        "longest_correct_streak": longest_correct,
        "longest_lost_streak": longest_lost,
        "mean_correct_streak": mean_correct_streak,
        "n_correct_streaks": len(correct_runs),
        "iou_thresh": thr,
        "reacquire_window": window,
    }
    for kind, count in sorted(event_counts.items()):
        out[f"n_events_{kind}"] = count
    return _clean(out)


def combine_target(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Frame-weighted aggregation of :func:`evaluate_target` results.

    Rates are pooled as ``sum(numerator counts) / sum(denominator counts)`` —
    the only correct way to merge sequences of different length — and latencies
    are averaged over successful events.  The confident-vs-blind false-acquisition
    split is pooled the same way: counts are summed and the locked rates are
    recomputed, never averaged.
    """
    out: dict[str, Any] = {}
    n_frames = sum(int(r["n_frames"]) for r in results)
    n_visible = sum(int(r["n_visible_frames"]) for r in results)
    n_reported = sum(int(r["n_report_frames"]) for r in results)
    n_correct = sum(int(r["n_correct_frames"]) for r in results)
    n_fa = sum(int(r["n_false_acquisition_frames"]) for r in results)
    n_ok = sum(int(r["reacquire_ok"]) for r in results)
    n_ev = sum(int(r["reacquire_events"]) for r in results)
    lat = []
    for r in results:
        if r["reacquire_ok"] > 0 and not math.isnan(float(r["reacquire_latency"])):
            lat.extend([float(r["reacquire_latency"])] * int(r["reacquire_ok"]))
    out["n_sequences"] = len(results)
    out["n_frames"] = n_frames
    out["n_visible_frames"] = n_visible
    out["n_report_frames"] = n_reported
    out["target_success_rate"] = _rate(n_correct, n_visible)
    out["loss_rate"] = _rate(n_visible - n_correct, n_visible)
    if n_reported > 0 and all("target_precision_sum" in r for r in results):
        # Exact pooled mean IoU — averaging per-sequence means would be wrong.
        out["target_precision"] = float(sum(float(r["target_precision_sum"]) for r in results)) / float(n_reported)
    else:
        out["target_precision"] = _nanmean(
            [r["target_precision"] for r in results if not math.isnan(float(r["target_precision"]))]
        )
    out["reacquire_events"] = n_ev
    out["reacquire_ok"] = n_ok
    out["reacquire_rate"] = _rate(n_ok, n_ev)
    out["reacquire_latency"] = _nanmean(lat)
    out["false_acquisition_rate"] = _rate(n_fa, n_reported)
    out["false_acquisition_frames"] = n_fa
    out["false_acquisition_events"] = sum(int(r["false_acquisition_events"]) for r in results)
    # Confident (locked) vs blind-coast split: sum the counts, recompute the
    # rates.  Averaging the per-sequence rates here would be wrong (different
    # reporting denominators).
    n_locked = sum(int(r.get("n_locked_frames", 0)) for r in results)
    n_reported_locked = sum(int(r.get("n_reported_locked", 0)) for r in results)
    n_fa_locked = sum(int(r.get("false_acquisition_frames_locked", 0)) for r in results)
    n_loss_locked = sum(int(r.get("loss_frames_locked", 0)) for r in results)
    out["n_locked_frames"] = n_locked
    out["n_reported_locked"] = n_reported_locked
    out["false_acquisition_frames_locked"] = n_fa_locked
    out["false_acquisition_rate_locked"] = _rate(n_fa_locked, n_reported_locked)
    out["false_acquisition_events_locked"] = sum(
        int(r.get("false_acquisition_events_locked", 0)) for r in results
    )
    out["loss_frames_locked"] = n_loss_locked
    out["loss_rate_locked"] = _rate(n_loss_locked, n_visible)
    if all("id_consistency_denominator" in r for r in results):
        n_on_init = sum(int(r["id_consistency_frames"]) for r in results)
        n_determined = sum(int(r["id_consistency_denominator"]) for r in results)
        out["id_consistency"] = _rate(n_on_init, n_determined)
        out["id_consistency_frames"] = n_on_init
        out["id_consistency_denominator"] = n_determined
    else:
        out["id_consistency"] = _nanmean(
            [r["id_consistency"] for r in results if not math.isnan(float(r["id_consistency"]))]
        )
    out["target_switch_count"] = sum(int(r["target_switch_count"]) for r in results)
    out["fragmentation"] = sum(int(r["fragmentation"]) for r in results)
    return _clean(out)
