"""E083 -- two structurally different hypotheses, decided on future frames.

The pre-registration is ``docs/E083_PLAN.md``; the amendment in
``docs/EXPERIMENTS.md`` (``### E083 -- AMENDMENT to the pre-registration``)
supersedes its clone-based design and is what this module implements.  E083 is
K = 2 *structurally different* hypotheses, not clones:

* ``H_crop``    the shipped tracker: one frozen reference, the click crop
  (``TargetTracker`` with ``cost.anchor_vote = "off"``).  Its reference set is
  R1 only.
* ``H_gallery`` the E079 k = 2 arm: the refreshed anchor R2 plus the long-term
  gallery R3 (``TargetTracker`` with ``cost.anchor_vote = "majority"``).  Its
  reference set is refreshed as it tracks.

The two hypotheses differ in ONE thing -- which reference set they trust -- and
the question E083 asks is which of the two reference sets predicts the target's
*next* frames, on the sequence the run is actually on.  E082 is why that question
has to be asked per sequence: the sign of the vote's effect is a property of the
sequence (dancetrack0010: vote_off TSR 0.2831 -> vote_majority 0.0478;
dancetrack0018: 1.000 -> 0.000), not of the candidate, so no per-candidate gate
can decide it in advance.

WHY THE LIKELIHOOD IS A TEMPORAL TEST, AND WHY THAT IS THE ONLY SYMMETRIC TEST
-----------------------------------------------------------------------------
Any score that compares a hypothesis' appearance to the frozen crop favours
H_crop; any that compares it to R2/R3 favours H_gallery.  A reference-based score
is therefore circular -- it measures which reference set the hypothesis *is*,
not whether it is right.  So the hypotheses are scored only on
future-frame evidence (temporal generalisation), through *disagreement-triggered
temporal validation*:

1. while the two reported boxes overlap (IoU >= 0.5) report ``H_crop``, which is
   bit-identical to the shipped tracker -- that is the falsifier for plumbing;
2. at the first disagreement (IoU < 0.5) at frame ``t0``, open a validation
   window of ``validate_frames`` (12) frames and freeze each hypothesis'
   reference set as it stands at ``t0``;
3. during the window, every frame, for each hypothesis ``h``, take the detection
   ``h`` accepted that frame and measure ``d_own(h)`` (its distance to ``h``'s own
   frozen reference set) and ``d_other(h)`` (the same crop's distance to the
   *other* hypothesis' set).  Accumulate the drift ratio
   ``r_h = mean(d_own(h)) / (mean(d_other(h)) + eps)``;
4. at window close ``logw_h = -lambda * r_h`` (``lambda = mh_drift_lambda = 2.0``),
   normalise with log-sum-exp, and switch the reported box only when
   ``w_win - w_lose >= mh_commit_margin`` (0.15) for ``mh_commit_frames`` (3)
   consecutive decisions -- the consecutive-frame logic of
   :class:`tgtrack.consensus.ChallengeTracker`, reused rather than rewritten.
   Otherwise ``H_crop`` keeps the identity (conservative default) and the window
   re-arms, so while the disagreement persists it re-opens every
   ``validate_frames`` frames;
5. as soon as the two boxes agree again (IoU >= 0.5) the reported hypothesis goes
   back to ``H_crop`` and the window closes: there is nothing to decide.

The rationale: a stale crop predicts the target's *future* appearance badly and
a fresh gallery predicts it well; when the gallery has absorbed a lookalike the
ordering reverses.  Scoring each hypothesis on how well its own reference set
predicts the crops that hypothesis goes on to accept -- while the same crops are
scored against the other set -- makes the test symmetric between the two
reference sets: neither set is privileged by construction, only by what it
actually predicts on frames that had not happened yet when it was frozen.

One consequence, stated up front because it bounds the claim: the window is only
opened by a *disagreement*, so on a sequence where the vote never changes the box
E083 can never fire, and its statistics are then exactly the shipped tracker's.

Where the symmetry is imperfect, stated rather than hidden
----------------------------------------------------------
Both hypotheses are scored on the SAME frames with the SAME function, and each
set is free to score the other's crops, so neither reference set is privileged by
the *test*.  The residual asymmetry is in the *estimator*: a reference set answers
with its nearest entry, and a minimum over more entries is stochastically smaller,
so H_gallery (R2 + every gallery row) reads slightly closer than a one-entry bed
would at equal quality.  That bias points at H_gallery, which is why a tie is
handed to H_crop and why ``mh_commit_margin`` exists at all: the margin has to
absorb it before anything switches.  Measuring each bed with its *mean* entry
would remove the bias but would leave the tracker's own units (its admission rule
is nearest-neighbour), so the nearest-entry distance is used deliberately and the
bias is recorded here.

MODES (``cost.mh_mode``, default ``"off"`` so nothing ships changed)
-------------------------------------------------------------------
``"off"``        delegate straight to one shipped ``TargetTracker``; the mode is
                 a no-op and ``initialize``/``step``/``status``/``events`` are
                 bit-identical to constructing that tracker directly (pinned by
                 the tests -- this is the falsifier).
``"dual"``       both hypotheses run; the likelihood above decides which box is
                 reported.
``"dual_frozen"`` both hypotheses run, but the reported hypothesis is pinned to
                 ``H_crop`` after frame 0.  The control that isolates "spend
                 compute on two boxes" from "let a posterior switch them".

This module is implementation and plumbing only: the measurement sweep is the
Lead's, and no shipped default changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .consensus import ChallengeState, ChallengeTracker
from .target_tracker import TargetTracker
from .types import (
    Detection,
    TargetEvent,
    TargetStatus,
    cosine_distance_matrix,
    iou_matrix,
)

__all__ = [
    "AGREE_IOU",
    "CROP",
    "GALLERY",
    "MH_MODES",
    "MultiHypothesisTracker",
    "ReferenceSet",
    "mode_from_cfg",
]

#: Boxes that overlap at least this much count as agreeing: the two hypotheses
#: report the same person and there is nothing to decide.
AGREE_IOU = 0.5

#: Hypothesis names.  ``CROP`` is index 0 of every posterior (the conservative
#: default), so ``weights()[0]`` is always ``H_crop``.
CROP = "crop"
GALLERY = "gallery"

#: Modes accepted by ``cost.mh_mode``.
MH_MODES: tuple[str, ...] = ("off", "dual", "dual_frozen")

#: Denominator guard of the drift ratio: without it a reference set that happens
#: to be *identical* to the other one (and to the accepted crops) would divide by
#: zero and hand out a spurious win.
_EPS = 1e-6

#: A reported box counts as "the detection the hypothesis accepted" when it
#: matches it this closely.  Anything looser would let a fused/rebuilt box pass
#: for a detection observation.
_BOX_MATCH_IOU = 0.99


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _norm_mode(value: Any) -> str:
    mode = str(value if value is not None else "off").strip().lower() or "off"
    if mode not in MH_MODES:
        raise ValueError(f"unknown mh_mode {value!r}; valid: {list(MH_MODES)}")
    return mode


def mode_from_cfg(cfg: Config) -> str:
    """The multi-hypothesis mode a config asks for (``cost.mh_mode``)."""
    cost = getattr(cfg, "cost", None)
    return _norm_mode(getattr(cost, "mh_mode", "off"))


def _iou_of(a: Any, b: Any) -> float:
    """IoU of two reported boxes; 0.0 when either is missing or unreadable."""
    if a is None or b is None:
        return 0.0
    try:
        left = np.asarray(a, np.float32).reshape(-1)[:4]
        right = np.asarray(b, np.float32).reshape(-1)[:4]
    except Exception:  # pragma: no cover - defensive
        return 0.0
    if left.size < 4 or right.size < 4:
        return 0.0
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return 0.0
    return float(iou_matrix(left[None, :], right[None, :])[0, 0])


def _unit_feat(value: Any) -> np.ndarray | None:
    """A finite 1-D descriptor, or ``None`` when the value is unusable."""
    if value is None:
        return None
    try:
        v = np.asarray(value, np.float32).reshape(-1)
    except Exception:  # pragma: no cover - defensive
        return None
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    return v


# --------------------------------------------------------------------------- #
# reference sets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReferenceSet:
    """One hypothesis' appearance reference set, frozen when a window opens.

    ``bank`` is an ``(M, D)`` stack of unit descriptors and :meth:`distance` is
    the nearest-entry cosine distance -- the same quantity the tracker's anchor
    veto and its gallery both answer with, so the two hypotheses are asked the
    identical question and only the *contents* of the bank differ:

    * ``H_crop``    M = 1: the frozen click crop (R1), never updated;
    * ``H_gallery`` M = 1 + |R3|: the refreshed anchor R2 plus every full-body
      gallery entry at the moment of the disagreement.
    """

    bank: np.ndarray
    kind: str = ""

    @property
    def size(self) -> int:
        return int(self.bank.shape[0]) if self.bank.ndim == 2 else 0

    @property
    def dim(self) -> int:
        return int(self.bank.shape[1]) if self.bank.ndim == 2 else 0

    def distance(self, feat: Any) -> float:
        """Nearest-entry cosine distance from ``feat`` to the frozen bank.

        ``inf`` means "cannot answer" (no bank, unreadable feature, dimension
        mismatch, non-finite distance).  Such a sample is dropped rather than
        scored, so a broken encoder or an empty gallery can never award a
        hypothesis a win by accident.
        """
        if self.size == 0:
            return float("inf")
        v = _unit_feat(feat)
        if v is None or v.size != self.dim:
            return float("inf")
        try:
            d = np.asarray(cosine_distance_matrix(v, self.bank), np.float64).reshape(-1)
        except Exception:  # pragma: no cover - defensive
            return float("inf")
        d = d[np.isfinite(d)]
        return float(d.min()) if d.size else float("inf")


def _crop_bank(tracker: TargetTracker) -> np.ndarray | None:
    """R1: the frozen click crop, a one-entry bank."""
    feat = _unit_feat(getattr(tracker, "_anchor_feat", None))
    return None if feat is None else feat.reshape(1, -1)


def _gallery_bank(tracker: TargetTracker) -> np.ndarray | None:
    """R2 + R3: the refreshed anchor and the full-body gallery, as they stand.

    No fallback to R1: the gallery hypothesis' structural claim is that these two
    refreshed references predict better than the frozen crop, so if neither is
    available there is no such claim to test and the bed is empty (which makes
    the window degenerate rather than quietly turning H_gallery into H_crop).
    """
    rows: list[np.ndarray] = []
    refreshed = _unit_feat(getattr(tracker, "_vote_r2", None))
    if refreshed is not None:
        rows.append(refreshed)
    try:
        entries = list(tracker.ltm.entries())
    except Exception:  # pragma: no cover - defensive
        entries = []
    for entry in entries:
        if str(getattr(entry, "part", "full")) != "full":
            continue
        vec = _unit_feat(getattr(entry, "feat", None))
        if vec is not None:
            rows.append(vec)
    if not rows:
        return None
    dim = int(rows[0].size)
    rows = [r for r in rows if int(r.size) == dim]
    return np.stack(rows, axis=0) if rows else None


def _capture_refs(hyp: "Hypothesis", frame_idx: int) -> ReferenceSet | None:
    """Freeze one hypothesis' reference set for the window opening at ``frame_idx``."""
    bank = _crop_bank(hyp.tracker) if hyp.kind == CROP else _gallery_bank(hyp.tracker)
    if bank is None:
        return None
    return ReferenceSet(bank=np.array(bank, np.float32, copy=True), kind=hyp.kind)


# --------------------------------------------------------------------------- #
# one hypothesis
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """One structural hypothesis: a name, a kind and its own tracker."""

    name: str
    kind: str
    tracker: TargetTracker
    #: Frozen reference set, non-None only while a validation window is open.
    refs: ReferenceSet | None = None
    #: The status this hypothesis reported on the current frame.
    status: TargetStatus | None = None
    #: Accumulators of the drift ratio over the open window.
    own_sum: float = 0.0
    own_n: int = 0
    other_sum: float = 0.0
    other_n: int = 0
    #: Frames on which this hypothesis' box was the one emitted.
    reported_frames: int = 0

    def begin_window(self, refs: ReferenceSet | None) -> None:
        """Open a window: freeze the reference set and zero the accumulators."""
        self.refs = refs
        self.own_sum = 0.0
        self.own_n = 0
        self.other_sum = 0.0
        self.other_n = 0

    def end_window(self) -> None:
        self.refs = None

    @property
    def box(self) -> np.ndarray | None:
        return None if self.status is None else self.status.xyxy

    @property
    def drift_ratio(self) -> float:
        """``mean(d_own) / (mean(d_other) + eps)``; ``inf`` without samples."""
        if self.own_n <= 0 or self.other_n <= 0:
            return float("inf")
        own = self.own_sum / float(self.own_n)
        other = self.other_sum / float(self.other_n)
        return float(own / (other + _EPS))

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "reported_frames": int(self.reported_frames),
            "ref_size": 0 if self.refs is None else self.refs.size,
            "own_n": int(self.own_n),
            "other_n": int(self.other_n),
            "drift_ratio": self.drift_ratio,
        }


# --------------------------------------------------------------------------- #
# the pair
# --------------------------------------------------------------------------- #
class MultiHypothesisTracker:
    """K = 2 structurally different target hypotheses, one reported box.

    ``H_crop`` is the shipped tracker (``anchor_vote = "off"``); ``H_gallery`` is
    the E079 k = 2 arm (``anchor_vote = "majority"``).  Both are real
    :class:`~tgtrack.target_tracker.TargetTracker` instances that see the same
    frames and the same detections, so a disagreement can only come from the one
    thing that differs: which reference set may admit a candidate.

    Args:
        cfg: the run configuration.  The hypothesis trackers are cloned from it,
            forcing ``cost.anchor_vote`` to ``"off"`` / ``"majority"`` -- the pair
            must differ structurally, so a vote mode in ``cfg`` is overridden
            here (it is not ignored: it is what the ``off`` mode still honours).
        embedder: forwarded to both hypothesis trackers.
        mode: ``None`` (default) takes ``cost.mh_mode``; an explicit value
            overrides the config for this object.
        commit_margin / commit_frames / validate_frames: ``None`` (default) takes
            ``cost.mh_commit_margin`` / ``mh_commit_frames`` /
            ``mh_validate_frames``, i.e. 0.15 / 3 / 12.

    The ``None`` sentinels are deliberate: a config knob that could not be set
    through the config would make ``mh_mode`` unable to say ``"off"``, which is
    the value that guarantees nothing ships changed.
    """

    def __init__(
        self,
        cfg: Config,
        embedder: Any = None,
        mode: str | None = None,
        commit_margin: float | None = None,
        commit_frames: int | None = None,
        validate_frames: int | None = None,
    ) -> None:
        cost = getattr(cfg, "cost", None)
        self.cfg = cfg
        self.mode = _norm_mode(mode if mode is not None else getattr(cost, "mh_mode", "off"))
        self.commit_margin = float(
            getattr(cost, "mh_commit_margin", 0.15) if commit_margin is None else commit_margin
        )
        self.commit_frames = int(
            getattr(cost, "mh_commit_frames", 3) if commit_frames is None else commit_frames
        )
        self.validate_frames = int(
            getattr(cost, "mh_validate_frames", 12) if validate_frames is None else validate_frames
        )
        self.drift_lambda = float(getattr(cost, "mh_drift_lambda", 2.0))

        if not (0.0 <= self.commit_margin <= 1.0):
            raise ValueError(f"mh_commit_margin must be in [0, 1], got {self.commit_margin!r}")
        if self.commit_frames < 1:
            raise ValueError(f"mh_commit_frames must be >= 1, got {self.commit_frames!r}")
        if self.validate_frames < 1:
            raise ValueError(f"mh_validate_frames must be >= 1, got {self.validate_frames!r}")
        if not np.isfinite(self.drift_lambda) or self.drift_lambda < 0.0:
            raise ValueError(f"mh_drift_lambda must be finite and >= 0, got {self.drift_lambda!r}")

        self.names: tuple[str, ...] = (CROP, GALLERY)

        # ---- engagement counters (E083_PLAN names first, extras after) ----
        self.n_spawned = 0
        self.n_dropped = 0
        self.n_commits = 0
        self.n_commit_blocked = 0
        self.n_swaps_reported = 0
        #: Windows opened / closed, and why a close did or did not switch.
        self.n_windows_opened = 0
        self.n_windows_closed = 0
        self.n_agree_resets = 0
        self.n_margin_missed = 0
        self.n_crop_holds = 0
        self.n_degenerate_windows = 0
        #: Frames on which the two reported boxes disagreed at all.
        self.n_disagree_frames = 0
        #: Evidence bookkeeping: admissible samples and dropped ones.
        self.n_evidence_samples = 0
        self.n_evidence_dropped = 0

        # ---- the hypotheses ----
        self._single: TargetTracker | None = None
        self.hypotheses: list[Hypothesis] = []
        if self.mode == "off":
            # Not a hypothesis pair at all: the shipped tracker, untouched, including
            # whatever ``cost.anchor_vote`` the config carries.
            self._single = TargetTracker(cfg, embedder)
            self.n_spawned = 1
        else:
            self.hypotheses = [
                Hypothesis(CROP, CROP, TargetTracker(cfg.clone(**{"cost.anchor_vote": "off"}), embedder)),
                Hypothesis(GALLERY, GALLERY, TargetTracker(cfg.clone(**{"cost.anchor_vote": "majority"}), embedder)),
            ]
            self.n_spawned = len(self.hypotheses)

        # ---- posterior / window state ----
        self._weights = np.full(len(self.hypotheses) or 2, 0.5, np.float64)
        self._reported = 0
        self._window_open = False
        self._window_start: int | None = None
        self._window_age = 0
        self._last_iou = 1.0
        self._last_win = 0
        self._last_margin = 0.0
        self._last_ratios: list[float] | None = None
        self._frame_idx = -1
        self._frame_dets: list[Detection] = []
        self._last_status: TargetStatus | None = None

        # The commit rule reuses the E071 challenge chain: `frames_required` is the
        # number of CONSECUTIVE decisions that must agree.  The link tests are made
        # inert on purpose (a decision box may legitimately have moved: iou_thresh
        # 0.0 always passes, and no feat is passed so link_id stays 0.0 < 2.0), and
        # max_age spans the gap between two decision points (a window opens one
        # frame after the previous close, so the gap is validate_frames + 1).
        self._challenge = ChallengeTracker(
            frames_required=self.commit_frames,
            iou_thresh=0.0,
            ident_thresh=2.0,
            max_age=self.validate_frames + 1,
            pool=False,
        )

        # ---- the emitted event stream (see `events`) ----
        self._events: list[TargetEvent] = []
        self._event_marks: list[int] = [0 for _ in self.hypotheses]

    # ---------------------------------------------------------------- properties
    @property
    def k(self) -> int:
        return 1 if self._single is not None else len(self.hypotheses)

    @property
    def events(self) -> list[TargetEvent]:
        """The events of the box stream that was actually emitted.

        ``H_crop``/``H_gallery`` each keep their own event log; only the log of
        the hypothesis that owned the reported box is forwarded here, so the
        evaluator's onset attribution sees the same stream the caller saw.
        """
        if self._single is not None:
            return self._single.events
        return self._events

    @property
    def reported_index(self) -> int:
        return self._reported

    def weights(self) -> np.ndarray:
        """Posterior over the hypotheses; index 0 is always ``H_crop``."""
        return np.array(self._weights, np.float64, copy=True)

    def reported(self) -> int:
        """Index of the hypothesis whose box ``step``/``status`` emit."""
        return self._reported

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, frame_bgr: np.ndarray, box_xyxy: Any, frame_idx: int = 0) -> TargetStatus:
        """Click-initialise every hypothesis on the same frame and box."""
        if self._single is not None:
            status = self._single.initialize(frame_bgr, box_xyxy, frame_idx=frame_idx)
            self._last_status = status
            return status

        for hyp in self.hypotheses:
            hyp.status = hyp.tracker.initialize(frame_bgr, box_xyxy, frame_idx=frame_idx)
            hyp.begin_window(None)
        self._frame_idx = int(frame_idx)
        self._reported = 0
        self._weights = np.full(len(self.hypotheses), 0.5, np.float64)
        self._window_open = False
        self._window_start = None
        self._window_age = 0
        self._last_iou = 1.0
        self._last_win = 0
        self._last_margin = 0.0
        self._last_ratios = None
        self._challenge.reset()
        self._event_marks = [0 for _ in self.hypotheses]
        self._events = []
        self._collect_events()
        self._last_status = self._emit()
        return self._last_status

    def step(
        self,
        frame_bgr: np.ndarray | None,
        dets: Any,
        frame_idx: int | None = None,
        frame_shape: Any = None,
    ) -> TargetStatus:
        """Advance both hypotheses one frame and report one box."""
        if self._single is not None:
            status = self._single.step(frame_bgr, dets, frame_idx=frame_idx, frame_shape=frame_shape)
            self._last_status = status
            return status

        dets = list(dets or [])
        self._frame_dets = dets
        for hyp in self.hypotheses:
            hyp.status = hyp.tracker.step(
                frame_bgr, dets, frame_idx=frame_idx, frame_shape=frame_shape
            )

        now = int(self.hypotheses[0].status.frame_idx) if self.hypotheses[0].status else self._frame_idx + 1
        self._frame_idx = now
        box_crop, box_gallery = self.hypotheses[0].box, self.hypotheses[1].box
        # A comparison needs two boxes.  A frame on which either hypothesis reports
        # nothing is not evidence: it neither opens a window nor ages one, because
        # neither "they agree" nor "they disagree" is observable on it.
        comparable = box_crop is not None and box_gallery is not None
        self._last_iou = _iou_of(box_crop, box_gallery) if comparable else 0.0
        agree = bool(comparable and self._last_iou >= AGREE_IOU)
        if comparable and not agree:
            self.n_disagree_frames += 1

        if self.mode == "dual_frozen":
            # Control: both hypotheses pay the compute, the reported one never moves.
            self._reported = 0
        elif not comparable:
            pass
        elif agree:
            # (5) they agree again: nothing to decide, back to the conservative default.
            # The chain is broken here too, even if the last window already closed,
            # so a margin that was confirmed once cannot mature across an agreement.
            if self._window_open:
                self._close_window()
                self.n_agree_resets += 1
            elif self._challenge.active:
                self.n_agree_resets += 1
            self._hold_crop()
            self._weights = np.full(len(self.hypotheses), 0.5, np.float64)
        elif self._window_open:
            # (3) one evidence frame inside the open window.
            self._accumulate()
            self._window_age += 1
            if self._window_age >= self.validate_frames:
                self._decide(now)
        elif not agree:
            # (2) first disagreement: open a window and freeze both reference sets.
            self._open_window(now)

        self._collect_events()
        self.hypotheses[self._reported].reported_frames += 1
        self._last_status = self._emit()
        return self._last_status

    def status(self, frame_idx: int | None = None) -> TargetStatus:
        """The status of the reported hypothesis (a copy of the last one)."""
        if self._single is not None:
            return self._single.status(frame_idx=frame_idx)
        if self._last_status is not None:
            return self._emit()
        status = self.hypotheses[self._reported].tracker.status(frame_idx=frame_idx)
        self._last_status = self._emit()
        return self._last_status

    # ---------------------------------------------------------------- window logic
    def _open_window(self, frame_idx: int) -> None:
        for hyp in self.hypotheses:
            hyp.begin_window(_capture_refs(hyp, frame_idx))
        self._window_open = True
        self._window_start = int(frame_idx)
        self._window_age = 0
        self._last_win = 0
        self._last_margin = 0.0
        self._last_ratios = None
        self._weights = np.full(len(self.hypotheses), 0.5, np.float64)
        self.n_windows_opened += 1

    def _close_window(self) -> None:
        for hyp in self.hypotheses:
            hyp.end_window()
        self._window_open = False
        self._window_age = 0

    def _accumulate(self) -> None:
        """One evidence frame: score each hypothesis on the crop it accepted."""
        n = len(self.hypotheses)
        for i, hyp in enumerate(self.hypotheses):
            other = self.hypotheses[(i + 1) % n]
            if hyp.refs is None or other.refs is None:
                continue
            det = self._accepted_det(hyp)
            if det is None:
                continue
            feat = getattr(det, "feat", None)
            if feat is None:
                self.n_evidence_dropped += 1
                continue
            d_own = hyp.refs.distance(feat)
            d_other = other.refs.distance(feat)
            if not (np.isfinite(d_own) and np.isfinite(d_other)):
                # A comparison needs BOTH distances: a missing reference (empty
                # gallery) or an unreadable crop is not evidence for either side.
                self.n_evidence_dropped += 1
                continue
            hyp.own_sum += float(d_own)
            hyp.own_n += 1
            hyp.other_sum += float(d_other)
            hyp.other_n += 1
            self.n_evidence_samples += 1

    def _window_weights(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Posterior from the drift ratios, or ``(None, None)`` without evidence."""
        if any(h.own_n <= 0 or h.other_n <= 0 for h in self.hypotheses):
            return None, None
        ratios = np.array([h.drift_ratio for h in self.hypotheses], np.float64)
        if not np.all(np.isfinite(ratios)):
            return None, None
        logw = -self.drift_lambda * ratios
        logw = logw - float(np.max(logw))
        weights = np.exp(logw)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return None, None
        weights = weights / total
        if not np.all(np.isfinite(weights)):
            return None, None
        return weights, ratios

    def _decide(self, frame_idx: int) -> None:
        """Window close: score, and switch only on a margin confirmed N times."""
        weights, ratios = self._window_weights()
        self._close_window()
        self.n_windows_closed += 1
        self._last_ratios = None if ratios is None else [float(r) for r in ratios]

        if weights is None:
            # Degenerate window (no admissible evidence at all): no comparison was
            # made, so the conservative default keeps the identity.
            self.n_degenerate_windows += 1
            self.n_dropped += sum(1 for h in self.hypotheses if h.own_n <= 0)
            self._hold_crop()
            return

        self._weights = weights
        win = int(np.argmax(weights))
        loser = 1 - win
        margin = float(weights[win] - weights[loser])
        self._last_win = win
        self._last_margin = margin

        if win == 0:
            # Evidence for the crop hypothesis (or a tie, since argmax picks index 0).
            self.n_crop_holds += 1
            self._hold_crop()
            return
        if margin < self.commit_margin:
            # The gallery won but not by enough: a margin miss, not a switch.
            self.n_margin_missed += 1
            self._hold_crop()
            return

        box = self.hypotheses[win].box
        if box is None:
            self.n_evidence_dropped += 1
            self._hold_crop()
            return
        report = self._challenge.observe(frame_idx, box)
        if report.state is ChallengeState.COMMITTED:
            if self._reported != win:
                self.n_swaps_reported += 1
            self.n_commits += 1
            self._set_reported(win)
        else:
            # Passed the margin, held by the consecutive-decision window.
            self.n_commit_blocked += 1

    def _hold_crop(self) -> None:
        """Keep (or go back to) ``H_crop`` and break any pending switch chain."""
        self._challenge.reset()
        self._set_reported(0)

    # ---------------------------------------------------------------- evidence
    def _accepted_det(self, hyp: Hypothesis) -> Detection | None:
        """The detection ``hyp`` accepted this frame, or ``None``.

        The tracker reports the accepted detection's box verbatim, so the
        detection is recovered by an exact-box match against the frame's
        candidates.  A fused or rebuilt box (no exact match) is *not* counted as
        an observation -- the likelihood only consumes real crop observations.
        """
        status = hyp.status
        if status is None or status.xyxy is None:
            return None
        diag = status.diag or {}
        if not bool(diag.get("accepted", False)):
            return None
        best: Detection | None = None
        best_iou = 0.0
        for det in self._frame_dets:
            value = _iou_of(status.xyxy, getattr(det, "xyxy", None))
            if value > best_iou:
                best, best_iou = det, value
        if best is None or best_iou < _BOX_MATCH_IOU:
            return None
        return best

    # ---------------------------------------------------------------- reporting
    def _set_reported(self, index: int) -> None:
        index = 0 if index <= 0 else min(int(index), len(self.hypotheses) - 1)
        if index != self._reported:
            # Skip the new hypothesis' backlog: its earlier events belong to frames
            # on which the other one owned the reported box.
            self._event_marks[index] = len(getattr(self.hypotheses[index].tracker, "events", []))
            self._reported = index

    def _collect_events(self) -> None:
        if not self.hypotheses:
            return
        source = self.hypotheses[self._reported].tracker
        log = getattr(source, "events", None) or []
        mark = self._event_marks[self._reported]
        if len(log) > mark:
            self._events.extend(list(log[mark:]))
            self._event_marks[self._reported] = len(log)

    def _emit(self) -> TargetStatus:
        """A copy of the reported hypothesis' status, tagged with the posterior."""
        if not self.hypotheses:
            return TargetStatus(frame_idx=self._frame_idx, state="init")
        hyp = self.hypotheses[self._reported]
        status = hyp.status if hyp.status is not None else hyp.tracker.status()
        diag = dict(status.diag or {})
        diag.update(
            {
                "mh_mode": self.mode,
                "mh_reported": hyp.name,
                "mh_reported_index": int(self._reported),
                "mh_iou": float(self._last_iou),
                "mh_weights": [float(w) for w in self._weights],
                "mh_window": int(self._window_open),
                "mh_window_age": int(self._window_age),
            }
        )
        return TargetStatus(
            frame_idx=int(status.frame_idx),
            state=status.state,
            xyxy=None if status.xyxy is None else np.array(status.xyxy, np.float32, copy=True),
            score=float(status.score),
            frames_since_update=int(status.frames_since_update),
            age=int(status.age),
            reacquired=bool(status.reacquired),
            diag=diag,
        )

    # ---------------------------------------------------------------- diagnostics
    def stats(self) -> dict[str, Any]:
        """Engagement counters plus the reported box's share of the run."""
        if self._single is not None:
            out: dict[str, Any] = dict(self._single.stats())
            out.update(
                {
                    "mh_mode": self.mode,
                    "mh_k": 1,
                    "mh_reported": CROP,
                    "mh_reported_index": 0,
                    "mh_weights": [1.0, 0.0],
                    "mh_switches": 0,
                }
            )
            return out

        challenge = self._challenge.stats()
        out = {
            "mh_mode": self.mode,
            "mh_k": int(len(self.hypotheses)),
            "mh_reported": self.hypotheses[self._reported].name,
            "mh_reported_index": int(self._reported),
            "mh_weights": [float(w) for w in self._weights],
            "mh_drift_lambda": float(self.drift_lambda),
            "mh_commit_margin": float(self.commit_margin),
            "mh_commit_frames": int(self.commit_frames),
            "mh_validate_frames": int(self.validate_frames),
            "mh_iou": float(self._last_iou),
            "mh_window_open": int(self._window_open),
            "mh_window_start": None if self._window_start is None else int(self._window_start),
            "mh_window_age": int(self._window_age),
            "mh_last_win": int(self._last_win),
            "mh_last_margin": float(self._last_margin),
            "mh_last_drift_ratios": None if self._last_ratios is None else list(self._last_ratios),
            "n_spawned": int(self.n_spawned),
            "n_dropped": int(self.n_dropped),
            "n_commits": int(self.n_commits),
            "n_commit_blocked": int(self.n_commit_blocked),
            "n_swaps_reported": int(self.n_swaps_reported),
            "n_windows_opened": int(self.n_windows_opened),
            "n_windows_closed": int(self.n_windows_closed),
            "n_agree_resets": int(self.n_agree_resets),
            "n_margin_missed": int(self.n_margin_missed),
            "n_crop_holds": int(self.n_crop_holds),
            "n_degenerate_windows": int(self.n_degenerate_windows),
            "n_disagree_frames": int(self.n_disagree_frames),
            "n_evidence_samples": int(self.n_evidence_samples),
            "n_evidence_dropped": int(self.n_evidence_dropped),
            "mh_commit_chain_agreed": int(self._challenge.agreed_n),
        }
        out.update({f"mh_{key}": int(value) for key, value in challenge.items()})
        out["hypotheses"] = [
            {**hyp.summary(), "stats": hyp.tracker.stats()} for hyp in self.hypotheses
        ]
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        if self._single is not None:
            return "MultiHypothesisTracker(mode='off')"
        return (
            "MultiHypothesisTracker(mode=%r, reported=%r, weights=[%.3f, %.3f], windows=%d, commits=%d)"
            % (
                self.mode,
                self.hypotheses[self._reported].name,
                float(self._weights[0]),
                float(self._weights[1]),
                self.n_windows_closed,
                self.n_commits,
            )
        )
