"""Temporal promotion of low-confidence detections (Wald's SPRT, borrowed from radar).

Why this exists
---------------
`docs/EXPERIMENTS.md` E002/E004 measured the situation precisely on real MOT17:

* at the shipped `conf=0.25` the pipeline gets recall **0.602** over all GT;
* **49.6 %** of the remaining false negatives *do* have a sub-threshold detection
  overlapping them **in the same frame** (class A), so a threshold change would take them
  — but doing that costs precision: the marginal precision is only **0.553**, which is why
  E002's `conf` sweep moved HOTA by ±0.3 pp;
* a further **17.8 %** are only visible in neighbouring frames (class B), and **32.6 %**
  are untouched by this detector at any confidence (class C).

Class A is the prize, and E002 proves it is a **selection** problem, not a threshold
problem: the extra boxes are ~55 % true in aggregate, but some of them are *consistently*
true over time while others are single-frame noise. So the question is not "how low can I
push the threshold" but "which low-score boxes does time vouch for".

The rule
--------
A person who is real leaves a low-confidence detection in *the same place* in consecutive
frames; a false positive rarely does. That is a sequential decision, and the classical
optimal test for it is Wald's **sequential probability ratio test**: accumulate

    LLR += log(p1 / p0)                 on a hit
    LLR += log((1 - p1) / (1 - p0))     on a miss

and promote when ``LLR >= log((1 - beta) / alpha)``, give up when
``LLR <= log(beta / (1 - alpha))``.  With ``alpha`` the tolerated false-alarm probability,
this replaces a hand-tuned "keep 3 frames in a row" heuristic with a rule that has a
*stated* error rate — the same shape of claim as the calibrated-refusal hypothesis (H5) in
`docs/RESEARCH_PLAN.md`.

Cost is negligible: it never touches the network, only a few hundred boxes per frame.

Nothing here is wired into the shipped pipeline yet — `scripts/probe_promote.py` prices it
first, per the protocol in `docs/RESEARCH_PLAN.md` §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import iou_matrix


@dataclass
class PromoteCfg:
    """Knobs for :class:`TemporalPromoter`.

    Attributes:
        conf_high: detections at or above this score are already accepted by the tracker
            and are only used as anchors for linking.
        conf_lo: detections below this are ignored entirely (noise in practice).
        link_iou: a low-score box may only continue a tracklet if it overlaps the
            prediction at least this much.
        link_dist: relaxed alternative to ``link_iou``, in units of the tracklet's box
            height, used when the overlap test fails (fast motion between frames).
        p_hit_true: probability that a *real* low-confidence person is detected each frame.
        p_hit_false: probability that a spurious low-confidence box recurs where it was
            last seen.  Must be well below ``p_hit_true`` or the LLR never accumulates.
        alpha: tolerated false-alarm probability (promoting noise), upper LLR boundary.
        beta: tolerated miss probability, lower LLR boundary.
        max_age: frames a tracklet may go unmatched before it is dropped regardless of LLR.
        max_tracklets: hard cap so a pathological scene cannot grow unbounded.
        promote_score: score assigned to a promoted box; must sit above the tracker's
            ``track_high_thresh`` so the first association stage can use it.
        vel_alpha: EMA weight on the newest displacement when updating a tracklet velocity.
    """

    conf_high: float = 0.25
    conf_lo: float = 0.01
    link_iou: float = 0.1
    link_dist: float = 0.75
    p_hit_true: float = 0.7
    p_hit_false: float = 0.15
    alpha: float = 0.05
    beta: float = 0.3
    max_age: int = 5
    max_tracklets: int = 256
    promote_score: float = 0.26
    vel_alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.p_hit_false < self.p_hit_true <= 1.0:
            raise ValueError(
                "need 0 < p_hit_false < p_hit_true <= 1, "
                f"got {self.p_hit_false} / {self.p_hit_true}")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.beta < 1.0:
            raise ValueError("alpha and beta must be in (0, 1)")
        if self.conf_lo >= self.conf_high:
            raise ValueError("conf_lo must be below conf_high")

    @property
    def llr_on(self) -> float:
        return math.log(self.p_hit_true / self.p_hit_false)

    @property
    def llr_off(self) -> float:
        return math.log((1.0 - self.p_hit_true) / (1.0 - self.p_hit_false))

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))


@dataclass
class _Tracklet:
    box: np.ndarray                      # (4,) xyxy
    score: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    hits: int = 0
    age: int = 0                         # frames since the last hit
    llr: float = 0.0
    promoted: bool = False
    frame_idx: int = 0


def _center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def _predict(box: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    out = box + velocity
    out[2] = max(out[2], out[0] + 1.0)
    out[3] = max(out[3], out[1] + 1.0)
    return out.astype(np.float32)


class TemporalPromoter:
    """Promote low-score detections that time vouches for.

    Usage (one call per frame, in order)::

        promoter = TemporalPromoter(PromoteCfg())
        boxes, scores = promoter.step(boxes_xyxy, scores)

    The input is the *raw* detector output for the frame, ideally run at a low ``conf``
    (``cfg.conf_lo``) so the low band is actually populated.  The returned arrays are the
    input's high-score detections plus any promoted low-score boxes, ready to hand to an
    association stage.
    """

    def __init__(self, cfg: PromoteCfg | None = None) -> None:
        self.cfg = cfg or PromoteCfg()
        self._tracklets: list[_Tracklet] = []
        self.frame_idx = -1
        self.n_promoted_total = 0
        self.n_spawned_total = 0
        self.last_promoted: np.ndarray = np.zeros((0, 4), np.float32)

    # -- internals ---------------------------------------------------------
    def _split(self, boxes: np.ndarray, scores: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        high = scores >= self.cfg.conf_high
        low = (scores >= self.cfg.conf_lo) & ~high
        return boxes[high], boxes[low], scores[low], high

    def _drop_anchored(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Keep only low boxes with no high-score counterpart (the rest add no information)."""
        if len(low) == 0 or len(high) == 0:
            return np.ones(len(low), dtype=bool)
        return iou_matrix(low, high).max(axis=1) < 0.5

    def _associate(self, low: np.ndarray) -> tuple[dict[int, int], list[int]]:
        """Greedy match low boxes to tracklets. Returns ({tracklet_idx: low_idx}, unmatched)."""
        if not self._tracklets or len(low) == 0:
            return {}, list(range(len(low)))
        preds = np.stack([_predict(t.box, t.velocity) for t in self._tracklets])
        iou = iou_matrix(low, preds)          # (n_low, n_tracklets)
        matches: dict[int, int] = {}
        used_low: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for li_raw, ti_raw in order:
            li, ti = int(li_raw), int(ti_raw)
            if li in used_low or ti in matches:
                continue
            ok = iou[li, ti] >= self.cfg.link_iou
            if not ok:
                t = self._tracklets[ti]
                h = max(float(t.box[3] - t.box[1]), 1.0)
                d = float(np.linalg.norm(_center(low[li]) - _center(preds[ti])) / h)
                ok = d <= self.cfg.link_dist
            if not ok:
                continue
            used_low.add(li)
            matches[ti] = li
        return matches, [i for i in range(len(low)) if i not in used_low]

    # -- public API --------------------------------------------------------
    def step(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame; return (boxes, scores) with promoted low-score boxes added."""
        cfg = self.cfg
        self.frame_idx += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        high, low, low_scores, _ = self._split(boxes, scores)
        if len(low):
            keep = self._drop_anchored(low, high)
            low, low_scores = low[keep], low_scores[keep]
        matches, unmatched = self._associate(low)

        # -- matched tracklets accumulate evidence
        for ti, li in matches.items():
            t = self._tracklets[ti]
            new_box = low[li]
            t.velocity = ((1.0 - cfg.vel_alpha) * t.velocity
                          + cfg.vel_alpha * (new_box - t.box)).astype(np.float32)
            t.box = new_box
            t.score = 0.7 * t.score + 0.3 * float(low_scores[li])
            t.hits += 1
            t.age = 0
            t.llr += cfg.llr_on
            t.frame_idx = self.frame_idx
            if not t.promoted and t.llr >= cfg.upper:
                t.promoted = True

        # -- unmatched tracklets decay and may die
        matched_idx = set(matches)
        drop: list[int] = []
        for ti, t in enumerate(self._tracklets):
            if ti in matched_idx:
                continue
            t.age += 1
            t.llr += cfg.llr_off
            if t.age > cfg.max_age or t.llr <= cfg.lower:
                drop.append(ti)
        for ti in sorted(drop, reverse=True):
            del self._tracklets[ti]

        # -- new tracklets from unmatched low boxes
        for li in unmatched:
            if len(self._tracklets) >= cfg.max_tracklets:
                break
            self._tracklets.append(_Tracklet(
                box=low[li].copy(), score=float(low_scores[li]),
                llr=cfg.llr_on, hits=1, age=0, frame_idx=self.frame_idx))
            self.n_spawned_total += 1

        # -- emit this frame's promoted boxes
        prom = [t.box.copy() for t in self._tracklets
                if t.promoted and t.age == 0 and t.frame_idx == self.frame_idx]
        self.n_promoted_total += len(prom)
        self.last_promoted = (np.stack(prom) if prom
                              else np.zeros((0, 4), np.float32)).astype(np.float32)
        if not prom:
            return high, scores[scores >= cfg.conf_high]
        out_b = np.concatenate([high, np.stack(prom)], axis=0)
        out_s = np.concatenate([scores[scores >= cfg.conf_high],
                                np.full(len(prom), cfg.promote_score, np.float32)])
        return out_b, out_s

    def reset(self) -> None:
        self._tracklets.clear()
        self.frame_idx = -1

    @property
    def n_tracklets(self) -> int:
        return len(self._tracklets)

    @property
    def n_promoted_live(self) -> int:
        return sum(1 for t in self._tracklets if t.promoted)
