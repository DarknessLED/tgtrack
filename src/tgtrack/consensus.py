"""Consensus-before-commit: the decision rule for E071.

The problem this solves
-----------------------
Every acceptance decision in `TargetTracker` is taken on **one frame**. The campaign's own
measurements say that is the wrong instrument for its dominant failure mode:

* E062/E063 — on `dancetrack0007` the 113 wrong frames are *all* in `tracking` state, the true
  target is detected in 108 of them at median IoU 0.93, so the error is not "we cannot see
  him", it is "we accepted the wrong box while the right one was available";
* E061 — the three identity transitions happen over one-or-two frames with smooth
  0.19/0.14/0.10 body-height jumps, i.e. they are **ties between two people at the same place**,
  not jumps to a distant stranger;
* E064 — the only second witness available (`stm` recent EMA) *is* the incumbent, so it can
  never outvote it; a two-reference test is mathematically unavailable.

One scalar gate cannot separate "the person I already track, briefly ambiguous" from "a new
person I should adopt", because both are plausible on a single frame. What separates them is
**persistence in time**: a real re-acquisition survives many frames, a tie does not. So the rule
becomes: *a non-incumbent candidate must win the association on several consecutive frames,
consistently, before it is allowed to take the identity.*

This module is the decision bookkeeping only — no appearance model, no thresholds baked in,
so it is unit-testable without a GPU and cannot silently change any existing behaviour until
the tracker is wired to it (E071 deliberately lands the mechanism and its tests first).

The trap that shapes the design
-------------------------------
`docs/E071_PLAN.md` warns: *"consensus must key on identity, not on box stability"*. Two
different people at the same screen position produce stable **boxes**; keying the vote on
box-IoU alone would therefore certify exactly the swaps we are trying to reject. So a chain
link requires both

1. ``iou(prev_box, box) >= iou_thresh``  — the proposal is spatially continuous, and
2. ``cosine_distance(prev_feat, feat) <= ident_thresh`` — the appearance agrees.

Requirement 2 is what makes this a *consensus of identity* rather than a consensus of
geometry, and it is the difference between this mechanism and the margin gate the campaign
already proved cannot separate the two cases (E029/E030/E032b).

Deliberately NOT in this module
-------------------------------
* where the gate sits (that is E070's axis, and it only slides along a trade-off);
* how the pooled descriptor is compared to `stm`/`ltm` (that needs the tracker's memory; the
  module only produces the pooled descriptor and leaves the test to the caller);
* any policy about `frames_since_update` (see the plan's trap about the blind window — a
  pending challenge must not be counted as a miss, and that decision belongs to the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import cosine_distance_matrix, iou_matrix

__all__ = ["ChallengeTracker", "ChallengeState", "ChallengeReport"]


class ChallengeState:
    """Return values of :meth:`ChallengeTracker.observe` (plain strings, cheap to log)."""

    IDLE = "idle"            # no challenge in progress
    PENDING = "pending"      # challenge open but not yet confirmed
    COMMITTED = "committed"  # consensus reached: the challenger may take the identity
    RESTARTED = "restarted"  # chain broke; a new challenge was opened with this frame


@dataclass
class ChallengeReport:
    """One frame's worth of challenge bookkeeping (what the tracker's diag records)."""

    state: str
    pending_n: int = 0
    agreed_n: int = 0
    span: int = 0
    pooled: np.ndarray | None = None
    reason: str = ""

    def as_diag(self) -> dict:
        return {
            "challenge_state": self.state,
            "pending_n": int(self.pending_n),
            "pending_agreed": int(self.agreed_n),
            "pending_span": int(self.span),
            "challenge_reason": self.reason,
        }


class ChallengeTracker:
    """Temporal vote over consecutive non-incumbent proposals.

    Parameters
    ----------
    frames_required:
        How many *consecutive, mutually consistent* frames a challenger must win before it may
        take the identity. ``0`` or ``1`` disables the mechanism (a single frame commits, which
        is the pre-E071 behaviour) — useful as an explicit control arm.
    iou_thresh:
        Minimum IoU between consecutive challenger boxes for the chain to continue. Low by
        design (0.15-0.30): this gates *agreement between proposals*, not *acceptance*.
    ident_thresh:
        Maximum cosine distance between consecutive challenger descriptors for the chain to
        continue. This is the leg that prevents "two people at the same place" from being read
        as agreement. ``+inf`` reproduces a box-only vote (the trap, available as a control).
    max_age:
        Frames a chain survives without a new proposal before it is abandoned. A one-frame
        lookalike blip therefore dies here, which is exactly the "we refused a tie" event the
        campaign wants to count.
    pool:
        When True, :attr:`pooled` exposes the L2-normalised mean of the chain's descriptors —
        the "pool before you decide" step borrowed from video face recognition. The caller
        decides what to compare it against; a chimeric chain (two different people) does not
        produce a coherent pooled descriptor.
    """

    def __init__(
        self,
        frames_required: int = 2,
        iou_thresh: float = 0.30,
        ident_thresh: float = 0.35,
        max_age: int = 8,
        pool: bool = True,
    ) -> None:
        if frames_required < 0:
            raise ValueError(f"frames_required must be >= 0, got {frames_required}")
        if not 0.0 <= iou_thresh <= 1.0:
            raise ValueError(f"iou_thresh must be in [0, 1], got {iou_thresh}")
        if max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {max_age}")
        self.frames_required = int(frames_required)
        self.iou_thresh = float(iou_thresh)
        self.ident_thresh = float(ident_thresh)
        self.max_age = int(max_age)
        self.pool = bool(pool)

        self._frames: list[int] = []
        self._boxes: list[np.ndarray] = []
        self._feats: list[np.ndarray] = []
        self._last_frame: int | None = None

        # mechanism evidence (a mechanism that never fires is inert, so these are mandatory)
        self.n_opened = 0
        self.n_committed = 0
        self.n_abandoned = 0
        self.n_restarted = 0
        self.n_pending_frames = 0

    # ------------------------------------------------------------------ properties
    @property
    def active(self) -> bool:
        return bool(self._frames)

    @property
    def agreed_n(self) -> int:
        return len(self._frames)

    @property
    def span(self) -> int:
        if not self._frames:
            return 0
        return int(self._frames[-1] - self._frames[0] + 1)

    @property
    def pooled(self) -> np.ndarray | None:
        """L2-normalised mean descriptor of the current chain, or None."""
        if not self.pool or not self._feats:
            return None
        stack = np.stack([np.asarray(f, dtype=np.float64).reshape(-1) for f in self._feats])
        mean = stack.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return (mean / norm).astype(np.float32)

    def stats(self) -> dict:
        return {
            "challenge_opened": self.n_opened,
            "challenge_committed": self.n_committed,
            "challenge_abandoned": self.n_abandoned,
            "challenge_restarted": self.n_restarted,
            "challenge_pending_frames": self.n_pending_frames,
        }

    # ------------------------------------------------------------------ main entry points
    def reset(self) -> None:
        """Drop the current chain without counting it as abandoned (call on target loss)."""
        self._frames.clear()
        self._boxes.clear()
        self._feats.clear()
        self._last_frame = None

    def tick(self, frame_idx: int) -> ChallengeReport:
        """Advance time without a proposal: abandon a chain that has gone stale."""
        if self._frames and self.max_age >= 0:
            age = int(frame_idx) - int(self._frames[-1])
            if age > self.max_age:
                self.n_abandoned += 1
                last = self._frames[-1]
                self.reset()
                return ChallengeReport(
                    state=ChallengeState.IDLE,
                    reason=f"abandoned_after_{age}f (last proposal frame {last})",
                )
        return self._report(ChallengeState.PENDING if self._frames else ChallengeState.IDLE)

    def observe(
        self,
        frame_idx: int,
        box_xyxy,
        feat=None,
        *,
        incumbent: bool = False,
    ) -> ChallengeReport:
        """Feed one frame's association outcome.

        ``incumbent=True`` means the association kept the person we already track: any open
        challenge is dead (so a tie that resolves back to the incumbent counts as abandoned),
        and the mechanism is a no-op.

        ``incumbent=False`` means a *non-incumbent* candidate won this frame. If that repeats
        consistently for ``frames_required`` frames, the state becomes
        :data:`ChallengeState.COMMITTED` and the caller may adopt it.
        """
        frame_idx = int(frame_idx)

        if incumbent:
            if self._frames:
                self.n_abandoned += 1
                span = self.span
                self.reset()
                return ChallengeReport(
                    state=ChallengeState.IDLE,
                    reason=f"incumbent_wins_after_{span}f",
                )
            self._last_frame = frame_idx
            return ChallengeReport(state=ChallengeState.IDLE, reason="incumbent")

        # a non-incumbent proposal arrives
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)[:4]
        if box.size < 4 or not np.all(np.isfinite(box)):
            return self.tick(frame_idx)

        vec = None
        if feat is not None:
            arr = np.asarray(feat, dtype=np.float64).reshape(-1)
            if arr.size and np.all(np.isfinite(arr)):
                n = float(np.linalg.norm(arr))
                if n > 1e-12:
                    # normalise now, but append only when the chain actually accepts this frame,
                    # so `len(_feats) == len(_frames)` holds at every point
                    vec = (arr / n).astype(np.float32)

        restart_reason = ""
        if self._frames:
            # too stale to chain onto?
            if self.max_age >= 0 and frame_idx - self._frames[-1] > self.max_age:
                self.n_abandoned += 1
                restart_reason = f"gap_{frame_idx - self._frames[-1]}f"
                self._drop_chain()
            else:
                link_iou = float(iou_matrix(box[None, :], self._boxes[-1][None, :])[0, 0])
                link_id = 0.0
                if vec is not None and self._feats:
                    link_id = float(cosine_distance_matrix(vec[None, :], self._feats[-1][None, :])[0, 0])
                if link_iou < self.iou_thresh or link_id > self.ident_thresh:
                    self.n_restarted += 1
                    restart_reason = (f"link_broke iou={link_iou:.3f}<{self.iou_thresh:.2f}"
                                      if link_iou < self.iou_thresh
                                      else f"identity_broke d={link_id:.3f}>{self.ident_thresh:.2f}")
                    self._drop_chain()

        if not self._frames:
            self.n_opened += 1
        self._frames.append(frame_idx)
        self._boxes.append(box)
        if vec is not None:
            self._feats.append(vec)
        self._last_frame = frame_idx

        if self.frames_required <= 1 or len(self._frames) >= self.frames_required:
            self.n_committed += 1
            rep = self._report(ChallengeState.COMMITTED, reason="consensus")
            self.reset()
            return rep

        self.n_pending_frames += 1
        state = ChallengeState.RESTARTED if restart_reason else ChallengeState.PENDING
        reason = restart_reason or f"pending_{len(self._frames)}/{self.frames_required}"
        return self._report(state, reason=reason)

    # ------------------------------------------------------------------ internals
    def _drop_chain(self) -> None:
        self._frames.clear()
        self._boxes.clear()
        self._feats.clear()

    def _report(self, state: str, reason: str = "") -> ChallengeReport:
        return ChallengeReport(
            state=state,
            pending_n=len(self._frames),
            agreed_n=len(self._frames),
            span=self.span,
            pooled=self.pooled,
            reason=reason,
        )
