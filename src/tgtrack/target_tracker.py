"""The clicked-target tracker 鈥?the heart of this system.

Why this exists as a separate module
------------------------------------
A multi-object tracker answers "which box belongs to which person". It does
that well and cheaply, but its objective is *coverage*: it must keep producing
boxes for everyone, and when two people cross it is allowed to be briefly wrong
as long as it re-fixes the IDs. A drone-style click-to-track system has the
opposite objective. It must (a) never lose the ONE person the operator chose,
and (b) never, ever silently continue on somebody else. A single wrong latch is
a failed mission, and 鈥?unlike an ID switch in MOT 鈥?it is invisible in the
output until the operator notices the drone is following a stranger.

So this tracker is deliberately *not* a MOTTracker with one track. It is a
single-hypothesis, evidence-weighted filter with four ideas that a MOT tracker
lacks:

1. **A long-term, part-aware appearance memory.** MOT trackers keep a short
   exponential average because they only need to survive an occlusion of a few
   frames. Surviving a full disappearance needs the whole history of the person
   鈥?every viewpoint, every lighting, every partially visible crop. We keep a
   diversity-gated gallery of embeddings per body part, so a face-only or
   upper-body-only sighting can still be matched against the correspondingly
   partial records from before.

2. **Refusal as a first-class outcome.** When the best candidate does not beat
   the runner-up by a margin, the correct action is to keep coasting on motion,
   not to pick one. We refuse, and we *remember the distractor* so the same
   look-alike does not trap us again.

3. **A motion model that knows when it is wrong.** An IMM over constant
   velocity / acceleration / turn covers fast and nonlinear motion; when the
   target is occluded we roll the filter back to before the occluder polluted
   it (the FastTracker idea) and coast with a decaying, uncertainty-grown gate.

4. **Appearance memory that freezes while blind.** While the target is
   occluded or out of frame we see the *occluder*, not the target. Absorbing
   those crops is how trackers slowly become confident about the wrong person.
   The gallery is frozen in exactly those states.

The public surface is intentionally tiny: :meth:`TargetTracker.initialize`
once (the click) and :meth:`TargetTracker.step` per frame.
"""

from __future__ import annotations

from dataclasses import replace

from enum import Enum
from typing import Any, Sequence

import numpy as np

from .trail import (
    TrailLinker,
    box_trail,
    trail_descriptor,
    trail_distance,
    velocity_disagreement,
)
from .association import CostBreakdown, MatchResult, compute_costs, gate_radius, select_candidate
from .config import Config, CostCfg, EmbedderCfg, StateCfg
from .consensus import ChallengeState, ChallengeTracker
from .memory import LongTermGallery, ShortTermMemory
from .motion import MotionModel
from .parts import crop_quality, estimate_partial_state, part_boxes, visible_ratio
from .types import PART_KEYS, Detection, TargetEvent, TargetStatus, clip_xyxy, cosine_distance_matrix, iou_matrix


def _iou1(a: Any, b: Any) -> float:
    """Scalar IoU of two ``xyxy`` boxes -- E099's "different person" test."""
    ax1, ay1, ax2, ay2 = (float(v) for v in np.asarray(a, float).ravel()[:4])
    bx1, by1, bx2, by2 = (float(v) for v in np.asarray(b, float).ravel()[:4])
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    inter = ix * iy
    area = max((ax2 - ax1) * (ay2 - ay1), 0.0) + max((bx2 - bx1) * (by2 - by1), 0.0)
    return float(inter / max(area - inter, 1e-9))

#: A candidate whose appearance distance to the target is below this is
#: "confusable" and gets memorized as a hard negative after a successful match.
#: Kept deliberately tight 鈥?a loose threshold poisons the negative bank with
#: ordinary bystanders and starts rejecting the target itself.
CONFUSABLE_DIST = 0.35

#: Hard cap on how many detections we are willing to push through the ReID
#: encoder in a single frame during global re-detection.
MAX_ENCODE = 64

#: E071: refusal reason for a candidate that won the frame but has not yet been
#: confirmed by the temporal vote.  Matched by the state machine in `step`, so
#: the string lives here instead of being spelled out twice.
REASON_CONSENSUS_PENDING = "consensus_pending"

#: E082 -- contrast thresholds at which the engagement counters report how many
#: ELIGIBLE candidates the gate would retain.  These are REPORTING probes, not
#: knobs: they never change what the vote admits.  They exist so the delta grid can
#: be chosen from a measured retention curve instead of from the shape of a number,
#: and so a grid of identical arms can be diagnosed ("the contrast axis is empty")
#: rather than reported as a null result.
_CONTRAST_PROBES: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)


class TargetState(str, Enum):
    """States of the clicked target's life.

    The string values of the "blind" states match ``StateCfg.frozen_states``
    on purpose: the same names drive both the state machine and the decision to
    freeze the appearance gallery.
    """

    INIT = "init"
    TRACKING = "tracking"
    #: A match was accepted after being lost; hold here until it sticks.
    REACQUIRING = "reacquiring"
    SOFT_OCCLUSION = "occluded_soft"
    LOST_SHORT = "lost_short"
    LOST_LONG = "lost_long"
    OUT_OF_FRAME = "out_of_frame"
    GAVE_UP = "gave_up"


#: States in which the target's box is being extrapolated rather than measured.
BLIND_STATES = frozenset(
    {
        TargetState.SOFT_OCCLUSION,
        TargetState.LOST_SHORT,
        TargetState.LOST_LONG,
        TargetState.OUT_OF_FRAME,
    }
)


class TargetTracker:
    """Track one user-selected person through a video.

    Typical use::

        tt = TargetTracker.load(cfg)
        tt.initialize(frame0, clicked_box)
        for i, (frame, dets) in enumerate(stream):
            status = tt.step(frame, dets, frame_idx=i + 1)

    The tracker never raises on a missing target; it reports states instead.
    """

    def __init__(
        self,
        cfg: Config,
        embedder: Any | None = None,
        *,
        motion: MotionModel | None = None,
        stm: ShortTermMemory | None = None,
        ltm: LongTermGallery | None = None,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.motion = motion if motion is not None else MotionModel(cfg.motion, cfg.cost)
        self.stm = stm if stm is not None else ShortTermMemory(cfg.memory)
        self.ltm = ltm if ltm is not None else LongTermGallery(cfg.memory)

        self.state: TargetState = TargetState.INIT
        self.age = 0
        self.hits = 0
        self.frames_since_update = 0
        self.frame_idx = -1
        self.events: list[TargetEvent] = []

        self._box: np.ndarray | None = None
        self._score = 0.0
        self._reacquired = False
        #: Target speed (px/frame) at the last accepted observation. The gate
        #: growth term uses this instead of the coast-damped live speed, which
        #: decays 0.98**fsu and made the search radius saturate while blind.
        self._v0_speed = 0.0
        self._last_diag: dict[str, Any] = {}
        #: E067: how many times the encoder raised. Monotonic, independent of
        #: ``_last_diag`` (which `initialize` replaces), so an encoder that dies
        #: mid-run can never look like a healthy geometry-only tracker.
        self._encode_errors: int = 0
        #: E078: cumulative outcome of the absorb guard. Monotonic, like
        #: ``_encode_errors`` and for the same reason -- "blocked 40 crops" and
        #: "never ran" must stay distinguishable in the run record (``stats()``).
        self._absorb_anchor_gated: int = 0
        self._absorb_anchor_blocked: int = 0
        #: Per-part exponential averages of accepted crops. These are the query
        #: side of part matching: they answer "what does the target's upper body
        #: look like *right now*" the way `stm` answers it for the whole body.
        self._part_ema: dict[str, np.ndarray] = {}
        self._part_ema_quality: dict[str, float] = {}
        #: Recent distance-to-LTM of accepted crops (drift monitor, E021).
        self._drift_hist: list[float] = []
        #: Recent accepted-crop-to-gallery distance (adaptive forgetting, E025).
        self._app_drift: list[float] = []
        #: Set once per loss episode so rollback happens once, not per frame.
        self._rolled_back = False
        self._was_out_of_frame = False
        #: Consecutive accepted matches; drives the re-acquisition hold.
        self._hold = 0
        #: E092 -- provisional re-acquisition candidate: the box that is asking for the
        #: identity while the target is lost, and how many consecutive frames it has
        #: survived.  Inert at ``reacquire_confirm_frames = 1`` (the shipped path).
        self._pending_box: np.ndarray | None = None
        self._pending_hits: int = 0
        #: E092b -- (frame, box) of the candidates considered while lost, used by the
        #: gap-tolerant commit rule.  Empty on the shipped path.
        self._pending_hist: list[tuple[int, np.ndarray]] = []
        #: E095 -- per-frame candidate boxes, recorded only when the refutation
        #: channel is armed (``state.decision_lag > 0``); empty otherwise so the
        #: shipped path pays nothing.
        self._refute_on: bool = int(getattr(self.cfg.state, "decision_lag", 0) or 0) > 0
        self._refute_by_frame: dict[int, list[list[float]]] = {}
        #: E097 -- per-candidate appearance cost, logged beside the boxes so an
        #: offline selector can weigh look against motion.
        self._refute_app_by_frame: dict[int, list[float]] = {}
        #: E099 -- pre-veto re-attribution.  ``_preveto_on`` is false unless
        #: ``state.preveto_window > 0``, so the shipped path stores nothing.
        self._preveto_on: bool = int(getattr(self.cfg.state, "preveto_window", 0) or 0) > 0
        #: frame -> {"dets", "boxes", "app", "shipped"}: the pre-veto pool retained
        #: for the DP, pruned to ``preveto_window`` frames.
        self._preveto_hist: dict[int, dict[str, Any]] = {}
        #: (dets, appearance costs) of the frame currently being associated; paired
        #: with the shipped choice in ``step()`` once the decision is known.
        self._preveto_pending: tuple[list[Detection], np.ndarray] | None = None
        self._preveto_switches: int = 0
        self._deferred_frame: bool = False
        #: Per-frame scratch used by the hard-negative bookkeeping.
        self._frame_dets: list[Detection] = []
        self._encoded_dets: list[Detection] = []
        self._last_breakdown: CostBreakdown | None = None
        #: Frozen appearance of the clicked crop (E027).  Never updated.
        self._anchor_feat: np.ndarray | None = None
        #: Distances from the click crop to the crops we have been tracking
        #: (E029).  Feeds the adaptive band; never used for anything else.
        self._anchor_dist_hist: list[float] = []
        # E063 rival-audit bookkeeping: last audited frame, last rival box (for
        # spatial consistency) and how many consecutive audits it won.
        self._last_audit: int = -10**9
        self._audit_rival: np.ndarray | None = None
        self._audit_n: int = 0
        # E059: motion-signature state (created lazily when trail_weight > 0)
        self._trail: TrailLinker | None = None
        self._trail_desc: list[Any] | None = None
        #: Mahalanobis history of accepted crops (E032 adaptive motion band).
        self._maha_hist: list[float] = []
        #: E058 anchor bootstrap: the click crop plus the crops that have been confidently
        #: matched since, and how many more we are still willing to average in.
        self._anchor_seed: np.ndarray | None = None
        self._bootstrap_feats: list[np.ndarray] = []
        self._bootstrap_left: int = 0
        #: E071 -- temporal consensus on the handback (docs/E071_PLAN.md). Built
        #: only when ``CostCfg.commit_consensus`` is on; with the switch off the
        #: attribute stays None and every hook below is skipped, which is what
        #: keeps the shipped path byte-for-byte identical to the pre-E071 one.
        self._challenge: ChallengeTracker | None = None
        #: Appearance of the last ACCEPTED box: the descriptor of "the person we
        #: currently report".  It is the only reference the consensus gate has
        #: for the incumbent/challenger question, because this tracker is
        #: single-hypothesis -- at association time exactly one candidate won,
        #: and the only evidence about WHO it is, is its distance to this
        #: vector.  Updated by `_on_match`, so it always describes the person
        #: whose box we hand to the operator.
        self._consensus_feat: np.ndarray | None = None
        #: E077 -- how often the channel-conditional acceptance branch admitted a
        #: candidate the additive bound would have refused, split by which channel
        #: carried it.  Zero in every shipped run (the switch is off), which is
        #: itself the evidence that the branch is inert by default.
        self._channel_counts: dict[str, int] = {"geometry": 0, "appearance": 0}
        #: E079 -- the reference vote over the anchor veto (docs/E079_PLAN.md).
        #: ``_vote_k`` is 0 when the mechanism is off, and every hook below is then
        #: a return-on-zero: the veto block computes exactly the same ``total`` array
        #: it computed before E079, which is what keeps the shipped path identical.
        self._vote_mode: str = str(getattr(cfg.cost, "anchor_vote", "off") or "off").strip().lower()
        if self._vote_mode not in ("off", "majority", "unanimous", "adaptive", "contrast", "any"):
            raise ValueError(
                f"unknown cost.anchor_vote {self._vote_mode!r}; "
                "expected 'off', 'majority', 'unanimous', 'adaptive', 'contrast' or 'any'"
            )
        #: E084b -- ``any`` is k = 1, the ONLY point of the k-axis the campaign never
        #: measured (E079 ran k = 2, E081/E082 k = 2, ``unanimous`` k = 3).  The E084
        #: step-0 audit showed why it matters: of the ceiling refusals that were the
        #: TRUE target, R3 alone would have rescued 18 and R2 alone 0 -- so the two-bed
        #: majority is refusing targets for the sake of a reference that never helps.
        self._vote_k: int = {
            "majority": 2,
            "unanimous": 3,
            "adaptive": 2,
            "contrast": 2,
            "any": 1,
        }.get(self._vote_mode, 0)
        self._vote_alpha: float = float(getattr(cfg.cost, "anchor_vote_alpha", 0.1))
        #: E081 -- distress gate.  Sampled once per frame from the state the frame
        #: ENTERED in, i.e. strictly trailing and causal.  A plain list is used
        #: instead of a deque to keep this module's imports untouched.
        self._vote_dw: int = max(1, int(getattr(cfg.cost, "anchor_vote_distress_window", 50) or 50))
        self._vote_tau: float = float(getattr(cfg.cost, "anchor_vote_distress_tau", 0.35))
        self._vote_hist: list[int] = []
        #: E082 -- the contrast gate.  ``_vote_delta < 0`` disables the gate (and is
        #: bit-identical to 'majority'), so the shipped default costs nothing.  The
        #: counters below describe the ELIGIBLE population, i.e. exactly the
        #: candidates the blanket vote would have rescued: the retention curve over
        #: ``_CONTRAST_PROBES`` is what the deltas are chosen from, and it is reported
        #: whether or not the gate is on, so "the contrast axis is empty" is a
        #: measurement rather than an assumption.
        self._vote_delta: float = float(getattr(cfg.cost, "anchor_vote_contrast", -1.0))
        self._vote_contrast_n: int = 0
        self._vote_contrast_sum: float = 0.0
        self._vote_contrast_min: float = float("inf")
        self._vote_contrast_max: float = float("-inf")
        self._vote_contrast_ge: dict[str, int] = {f"{t:.2f}": 0 for t in _CONTRAST_PROBES}
        #: Eligible candidates the contrast gate REMOVED.  Zero here with a non-zero
        #: ``vote_rescued`` would mean the gate never bound, i.e. the E082 arm is the
        #: E079 arm wearing a different name.
        self._vote_contrast_blocked: int = 0
        #: Frames on which the distress gate suppressed the vote.  Distinguishes
        #: "the gate is doing work" from "the gate is never reached", which is the
        #: distinction E081's pre-registered kill criterion depends on.
        self._vote_gated: int = 0
        #: R2, the refreshed anchor. ``None`` while the vote is off; seeded from the
        #: click crop in ``initialize`` and otherwise moved only by
        #: ``_vote_update_r2``, which requires R1 and R3 to agree first.
        self._vote_r2: np.ndarray | None = None
        #: Engagement evidence, monotonic for the same reason as the E078 counters:
        #: "the vote never rescued anything" and "the vote never ran" must not look
        #: like each other in the run record.
        self._vote_frames: int = 0
        self._vote_n: int = 0
        self._vote_r1_pass: int = 0
        self._vote_r2_pass: int = 0
        self._vote_r3_pass: int = 0
        self._vote_rescued: int = 0
        self._vote_rescued_accepted: int = 0
        self._vote_r2_updates: int = 0
        #: Per-frame indices of the candidates the vote admitted (used to report
        #: whether the res admitted on this frame was one of them).
        self._vote_rescued_idx: set[int] = set()
        #: E084 step 0 -- opt-in JSONL log of ceiling-refused candidates.  ``None``
        #: means off, and then no code below does anything different.
        self._bed_log_path: str = str(getattr(cfg.cost, "anchor_bed_log", "") or "")
        self._bed_log = None
        self._bed_last: dict | None = None
        if self._bed_log_path:
            try:
                from pathlib import Path as _P
                _P(self._bed_log_path).parent.mkdir(parents=True, exist_ok=True)
                self._bed_log = open(self._bed_log_path, "a", encoding="utf-8")
            except OSError:
                self._bed_log = None
        #: E079 measurement of the E029 band: how often ``_anchor_limit`` is pinned
        #: at its ceiling, i.e. how often the "adaptive" band is really an absolute
        #: distance cut.  Independent of the vote, so both arms report it.
        self._band_frames: int = 0
        self._band_saturated: int = 0
        if bool(getattr(cfg.cost, "commit_consensus", False)):
            self._challenge = ChallengeTracker(
                frames_required=int(getattr(cfg.cost, "consensus_frames", 2)),
                iou_thresh=float(getattr(cfg.cost, "consensus_iou", 0.30)),
                ident_thresh=float(getattr(cfg.cost, "consensus_ident", 0.35)),
                pool=bool(getattr(cfg.cost, "consensus_pool", True)),
            )

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        cfg: Config | None = None,
        *,
        config_path: str | None = None,
        with_embedder: bool = True,
        device: str = "cuda",
        force_fallback: bool = False,
        **overrides: Any,
    ) -> "TargetTracker":
        """Build a tracker with a real ReID embedder.

        This is the convenience entry point used by the scripts and the
        evaluator. Importing the embedder lazily keeps `torch` out of the
        import path of the pure-logic modules, which matters because the unit
        tests must run without a GPU.
        """
        if cfg is None:
            from .config import load_config

            cfg = load_config(config_path, **overrides)
        emb = None
        if with_embedder:
            from .appearance import Embedder

            emb = Embedder(cfg.embedder, device=device, force_fallback=force_fallback)
        return cls(cfg, emb)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def started(self) -> bool:
        return self._box is not None and self.state is not TargetState.INIT

    @property
    def box(self) -> np.ndarray | None:
        return None if self._box is None else self._box.copy()

    @property
    def is_blind(self) -> bool:
        return self.state in BLIND_STATES

    @property
    def memory_frozen(self) -> bool:
        """True when appearance memory must not absorb anything this frame."""
        return self.state.value in tuple(self.cfg.state.frozen_states) or self.state in BLIND_STATES

    def refute_appearance(self) -> dict[int, list[float]]:
        """E097 -- per-candidate appearance cost, parallel to ``refute_candidates()``."""
        return self._refute_app_by_frame

    def refute_candidates(self) -> dict[int, list[list[float]]]:
        """E095 -- candidate boxes per frame, for the bounded-lag refutation pass.

        Only populated while ``state.decision_lag > 0``; returns an empty mapping
        otherwise, which is what makes the shipped path bit-identical (the pass is
        then a no-op because it has nothing to re-attribute).
        """
        return self._refute_by_frame

    def status(self, frame_idx: int | None = None) -> TargetStatus:
        return TargetStatus(
            frame_idx=int(self.frame_idx if frame_idx is None else frame_idx),
            state=self.state.value,
            xyxy=self.box,
            score=float(self._score),
            frames_since_update=int(self.frames_since_update),
            age=int(self.age),
            reacquired=bool(self._reacquired),
            diag=dict(self._last_diag),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "frames_since_update": self.frames_since_update,
            "motion_models": self.motion.model_names,
            "is_imm": self.motion.is_imm,
            "speed_px": round(self.motion.speed(), 2) if self.motion.started else 0.0,
            "stm_len": len(self.stm),
            "ltm_len": len(self.ltm),
            "ltm_parts": {p: sum(1 for e in self.ltm.entries() if e.part == p) for p in self.ltm.parts()},
            "hard_negatives": self.ltm.num_hard_negatives,
            "events": len(self.events),
            # E078: mechanism evidence, on the same principle as the E067/E071 counters
            # below -- a guard that never fires is a no-op, so "gated 0" and "blocked 0
            # because nothing was ever out of band" must not look alike.
            "absorb_anchor_only": bool(
                getattr(self.cfg.memory, "ltm_absorb_anchor_only", False)
            ),
            "absorb_anchor_ratio": float(
                getattr(self.cfg.memory, "ltm_absorb_anchor_ratio", 1.25)
            ),
            "absorb_anchor_gated": int(self._absorb_anchor_gated),
            "absorb_anchor_blocked": int(self._absorb_anchor_blocked),
            # E067: a broken encoder silently degraded the tracker to
            # geometry-only, which is how the first profile read "encoder = 2%".
            "encoder_errors": int(self._encode_errors),
            "encoder_error": self._last_diag.get("encoder_error"),
            "encoder_error_frame": self._last_diag.get("encoder_error_frame"),
            # E071: mechanism evidence. A consensus rule that never opens a
            # challenge is indistinguishable from a no-op, so the decision
            # counters are part of the run record even when the switch is off
            # (all zeros then, which is itself the proof that it was off).
            "challenge_enabled": self._challenge is not None,
            **self._challenge_stats(),
            # E077: mechanism evidence for the two arms.  ``channel_accept_*`` count
            # frames the OR-branch admitted that the additive bound would have
            # refused; ``aspect_clamp_*`` count coast steps that clipped an
            # extrapolated aspect.  All zero in the shipped configuration.
            "channel_accept_enabled": bool(getattr(self.cfg.cost, "channel_accept", False)),
            "channel_accept_geometry": int(self._channel_counts.get("geometry", 0)),
            "channel_accept_appearance": int(self._channel_counts.get("appearance", 0)),
            "aspect_clamp_enabled": bool(getattr(self.cfg.motion, "aspect_clamp_coast", False)),
            "aspect_clamp_steps": int(getattr(self.motion, "aspect_clamp_steps", 0)),
            "aspect_clamp_calls": int(getattr(self.motion, "aspect_clamp_calls", 0)),
            # E079: mechanism evidence for the reference vote, plus the E029 band
            # saturation share it is a vote about.  All of it is monotonic and
            # reported in both arms, so "the vote never fired" and "the vote fired
            # and rescued nothing" stay distinguishable, and the control arm can be
            # read against the reference's own band behaviour.
            "anchor_vote": self._vote_mode,
            "anchor_vote_alpha": float(self._vote_alpha),
            "vote_frames": int(self._vote_frames),
            "vote_n": int(self._vote_n),
            "vote_r1": int(self._vote_r1_pass),
            "vote_r2": int(self._vote_r2_pass),
            "vote_r3": int(self._vote_r3_pass),
            "vote_rescued": int(self._vote_rescued),
            "vote_rescued_accepted": int(self._vote_rescued_accepted),
            "vote_r2_updates": int(self._vote_r2_updates),
            # E081 -- the distress gate's own bookkeeping, so "the gate never ran"
            # cannot be mistaken for "the gate never helped".
            "vote_distress": float(self._vote_distress()),
            "vote_gated": int(self._vote_gated),
            # E082 -- the contrast axis, reported in every arm.  ``n`` / ``mean`` /
            # ``min`` / ``max`` describe the ELIGIBLE population (candidates the
            # blanket vote would rescue); ``ge_*`` is the same population's retention
            # curve, so the delta grid is chosen from data.  ``blocked`` counts the
            # eligible candidates the gate removed: if it stays 0 while ``rescued``
            # moves, the gate never bound and the arm is E079 under another name.
            "anchor_vote_contrast": float(self._vote_delta),
            "vote_contrast_n": int(self._vote_contrast_n),
            "vote_contrast_mean": (
                float(self._vote_contrast_sum) / float(self._vote_contrast_n)
                if self._vote_contrast_n else None
            ),
            "vote_contrast_min": (
                float(self._vote_contrast_min)
                if self._vote_contrast_n and np.isfinite(self._vote_contrast_min) else None
            ),
            "vote_contrast_max": (
                float(self._vote_contrast_max)
                if self._vote_contrast_n and np.isfinite(self._vote_contrast_max) else None
            ),
            "vote_contrast_blocked": int(self._vote_contrast_blocked),
            **{
                f"vote_contrast_ge_{key}": int(n)
                for key, n in self._vote_contrast_ge.items()
            },
            "anchor_band_frames": int(self._band_frames),
            "anchor_band_saturated": int(self._band_saturated),
            "anchor_band_saturation": (
                float(self._band_saturated) / float(self._band_frames)
                if self._band_frames else None
            ),
        }

    def _challenge_stats(self) -> dict[str, Any]:
        """E071 counters, zeroed when the mechanism is not built."""
        if self._challenge is not None:
            return dict(self._challenge.stats())
        return {
            "challenge_opened": 0,
            "challenge_committed": 0,
            "challenge_abandoned": 0,
            "challenge_restarted": 0,
            "challenge_pending_frames": 0,
        }


    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def _emit(self, kind: str, **detail: Any) -> None:
        self.events.append(TargetEvent(frame_idx=int(self.frame_idx), kind=kind, detail=dict(detail)))

    def snapshot(self) -> dict[str, Any]:
        """Serializable state, so a run can be replayed from any frame.

        ``ShortTermMemory`` exposes a ring buffer rather than a snapshot API, so
        we record its current mean plus its recent history and rebuild from
        that on restore. ``LongTermGallery`` snapshots itself.
        """
        cur = self.stm.current()
        return {
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "hold": self._hold,
            "frames_since_update": self.frames_since_update,
            "frame_idx": self.frame_idx,
            "box": None if self._box is None else self._box.tolist(),
            "score": self._score,
            "stm_cur": None if cur is None else cur.tolist(),
            "stm_recent": [np.asarray(r, np.float32).tolist() for r in self.stm.recent(8)],
            "part_ema": {k: v.tolist() for k, v in self._part_ema.items()},
            "ltm": self.ltm.snapshot(),
            # E071: the vote is *state* -- a challenger half-way to a commit must
            # not be forgotten by a snapshot round-trip, or replaying a run from
            # any frame would give a different commit decision than the run it
            # replays.
            "consensus_feat": (
                None if self._consensus_feat is None
                else np.asarray(self._consensus_feat, np.float32).reshape(-1).tolist()
            ),
            "challenge": self._challenge_snapshot(),
        }

    # ------------------------------------------------------------------ #
    # E071 consensus state (snapshot helpers)
    # ------------------------------------------------------------------ #
    def _challenge_snapshot(self) -> dict[str, Any] | None:
        """Serialise the consensus chain.

        ``ChallengeTracker`` exposes no snapshot API of its own, and its chain is
        three parallel lists plus five counters, so mirroring those is the whole
        of it.  Kept here rather than in ``consensus.py`` so the module stays a
        pure decision function with a tiny surface.
        """
        ch = self._challenge
        if ch is None:
            return None
        return {
            "config": {
                "frames_required": int(ch.frames_required),
                "iou_thresh": float(ch.iou_thresh),
                "ident_thresh": float(ch.ident_thresh),
                "max_age": int(ch.max_age),
                "pool": bool(ch.pool),
            },
            "chain_frames": [int(f) for f in ch._frames],
            "chain_boxes": [np.asarray(b, np.float64).reshape(-1)[:4].tolist() for b in ch._boxes],
            "chain_feats": [np.asarray(f, np.float32).reshape(-1).tolist() for f in ch._feats],
            "last_frame": None if ch._last_frame is None else int(ch._last_frame),
            "counters": {
                "n_opened": int(ch.n_opened),
                "n_committed": int(ch.n_committed),
                "n_abandoned": int(ch.n_abandoned),
                "n_restarted": int(ch.n_restarted),
                "n_pending_frames": int(ch.n_pending_frames),
            },
        }

    def _challenge_restore(self, snap: Any) -> None:
        """Inverse of :meth:`_challenge_snapshot`; tolerates partial snapshots."""
        ch = self._challenge
        if ch is None or not isinstance(snap, dict):
            return
        cfg = dict(snap.get("config") or {})
        for key in ("frames_required", "iou_thresh", "ident_thresh", "max_age", "pool"):
            if key in cfg:
                try:
                    setattr(ch, key, type(getattr(ch, key))(cfg[key]))
                except Exception:  # pragma: no cover - defensive
                    pass
        ch._frames = [int(f) for f in (snap.get("chain_frames") or [])]
        ch._boxes = [
            np.asarray(b, np.float64).reshape(-1)[:4] for b in (snap.get("chain_boxes") or [])
        ]
        ch._feats = [
            np.asarray(f, np.float32).reshape(-1) for f in (snap.get("chain_feats") or [])
        ]
        last = snap.get("last_frame")
        ch._last_frame = None if last is None else int(last)
        for key, val in dict(snap.get("counters") or {}).items():
            try:
                setattr(ch, key, int(val))
            except Exception:  # pragma: no cover - defensive
                pass

    def restore(self, snap: dict[str, Any]) -> None:
        """Inverse of :meth:`snapshot`. Tolerates older/partial snapshots."""
        self.state = TargetState(snap["state"])
        self.age = int(snap["age"])
        self.hits = int(snap["hits"])
        self._hold = int(snap.get("hold", 0))
        self.frames_since_update = int(snap["frames_since_update"])
        self.frame_idx = int(snap["frame_idx"])
        self._box = None if snap["box"] is None else np.asarray(snap["box"], np.float32)
        self._score = float(snap["score"])

        self.stm.reset()
        for f in snap.get("stm_recent") or []:
            self.stm.update(np.asarray(f, np.float32), quality=1.0, frame_idx=self.frame_idx)
        cur = snap.get("stm_cur")
        if cur is not None:
            self.stm.update(np.asarray(cur, np.float32), quality=1.0, frame_idx=self.frame_idx)

        self._part_ema = {
            k: np.asarray(v, np.float32) for k, v in (snap.get("part_ema") or {}).items()
        }
        self._part_ema_quality = {k: 1.0 for k in self._part_ema}
        if snap.get("ltm") is not None:
            self.ltm.restore(snap["ltm"])
        # E071
        cf = snap.get("consensus_feat")
        self._consensus_feat = None if cf is None else np.asarray(cf, np.float32).reshape(-1)
        self._challenge_restore(snap.get("challenge"))

    # ------------------------------------------------------------------ #
    # initialization (the click)
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        frame_bgr: np.ndarray,
        box_xyxy: np.ndarray,
        frame_idx: int = 0,
        score: float = 1.0,
    ) -> TargetStatus:
        """Seed the tracker from the operator's click.

        A click is a confirmation by definition, so the target starts in
        ``TRACKING`` rather than waiting out ``confirm_frames``.
        """
        h, w = self._frame_hw(frame_bgr)
        box = clip_xyxy(np.asarray(box_xyxy, np.float32).reshape(4), w, h)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"degenerate initialization box {box_xyxy!r}")

        self.frame_idx = int(frame_idx)
        self.age = 0
        self.hits = 1
        self.frames_since_update = 0
        self._box = box
        self._score = float(score)
        self._reacquired = False
        self._v0_speed = 0.0
        self._rolled_back = False
        self._was_out_of_frame = False
        self._hold = 0
        self._frame_dets = []
        self._encoded_dets = []
        self._last_breakdown = None
        self._part_ema.clear()
        self._part_ema_quality.clear()
        self.stm.reset()
        self.ltm.clear()
        self.motion.reset(box)

        self.state = TargetState.TRACKING
        self._emit("init", box=box.tolist())

        # Seed appearance memory from the clicked crop with maximal quality:
        # this is the only frame we know for certain is the right person.
        self._absorb(frame_bgr, box, score=1.0, quality=1.0, force=True)
        seed = self._part_ema.get("full")
        self._anchor_feat = None if seed is None else np.array(seed, np.float32, copy=True)
        # E058: the anchor starts as the click crop and can be re-estimated from the first
        # few confidently matched crops (`memory.anchor_bootstrap_frames`).
        self._anchor_seed = None if self._anchor_feat is None else np.array(self._anchor_feat, np.float32, copy=True)
        #: E079: R2 starts where R1 starts -- the click crop -- so the two references
        #: are not correlated by construction only later; R2 can only drift away from
        #: R1 through ``_vote_update_r2``, i.e. through frames on which R1 and R3
        #: already agreed.
        self._vote_r2 = (
            None
            if (self._anchor_feat is None or self._vote_k <= 0)
            else np.array(self._anchor_feat, np.float32, copy=True)
        )
        self._bootstrap_feats = []
        self._bootstrap_left = max(0, int(getattr(self.cfg.cost, "anchor_bootstrap_frames", 0) or 0))
        self._last_diag = {"init": True, "seed_parts": sorted(self._part_ema)}
        return self.status()

    def _bootstrap_anchor(self, feat: np.ndarray | None) -> None:
        """Average a confidently matched crop into the click anchor (E058).

        The anchor is what every E027-E030 veto measures against, so its noise floor is a
        first-order quantity: one crop at the click carries pose, blur and a possibly-partial
        view.  Here the anchor becomes the L2-normalised mean of the click crop and the first
        ``anchor_bootstrap_frames`` accepted crops, keeping the identity direction while the
        per-crop noise averages down.  Nothing happens when the knob is 0 or the crop has no
        descriptor, so the historical behaviour is bit-identical at the default.
        """
        if feat is None or self._anchor_seed is None or self._bootstrap_left <= 0:
            return
        self._bootstrap_feats.append(np.asarray(feat, np.float32).ravel())
        self._bootstrap_left -= 1
        stack = np.asarray([self._anchor_seed, *self._bootstrap_feats], np.float32)
        mean = stack.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm <= 1e-8:
            return
        self._anchor_feat = (mean / norm).astype(np.float32)
        self._last_diag["anchor_bootstrap"] = len(self._bootstrap_feats)

    # ------------------------------------------------------------------ #
    # E079 -- the reference vote over the anchor veto
    # ------------------------------------------------------------------ #
    def _vote_sample(self) -> None:
        """E081 -- record whether the frame the tracker is ENTERING was blind.

        Sampled in :meth:`step` before association, so the window never contains
        information from the frame being decided.  Sampling after the state update
        would let one ambiguous frame both raise distress and benefit from the vote
        it enables, i.e. the gate would see the future.
        """
        if self._vote_k <= 0:
            return
        self._vote_hist.append(1 if self.is_blind else 0)
        if len(self._vote_hist) > self._vote_dw:
            del self._vote_hist[0]

    def _vote_distress(self) -> float:
        """Share of the trailing window spent blind; 0.0 before any sample."""
        n = len(self._vote_hist)
        if n == 0:
            return 0.0
        return float(sum(self._vote_hist)) / float(n)

    def _vote_gallery_distances(self, feats: np.ndarray) -> np.ndarray:
        """R3: cosine distance from each candidate to the nearest gallery entry.

        One batched product per frame instead of one ``LongTermGallery.best`` call
        per candidate.  The single-query path re-stacks the whole bank on every
        call, which dominates the frame budget once the veto fires on every
        detection (measured 0.5-1 ms per candidate against ~200 entries, i.e. a
        ~5x slowdown of the whole run).  The result is the same quantity the
        gallery answers with ``part=None, touch=False``: same candidate entries
        (``entries()`` == ``_candidates(None)``), same cosine, same minimum.  A
        candidate the gallery cannot score stays ``+inf``, exactly like ``best``.
        """
        f = np.asarray(feats, dtype=np.float32)
        if f.ndim == 1:
            f = f[None, :]
        n = int(f.shape[0]) if f.ndim == 2 else 0
        out = np.full(n, np.inf, dtype=np.float32)
        entries = self.ltm.entries() if self.ltm is not None else []
        if n == 0 or not entries:
            return out
        dim = self.ltm.dim
        if dim is not None and int(f.shape[1]) != int(dim):
            return out
        usable = np.all(np.isfinite(f), axis=1) & (np.linalg.norm(f, axis=1) > 1e-12)
        if not bool(np.any(usable)):
            return out
        bank = np.stack([np.asarray(e.feat, dtype=np.float32).reshape(-1) for e in entries], axis=0)
        dists = cosine_distance_matrix(f[usable], bank).min(axis=1)
        out[usable] = np.asarray(dists, dtype=np.float32)
        return out

    @staticmethod
    def _vote_contrast(
        dist_r1: np.ndarray,
        dist_r2: np.ndarray,
        dist_r3: np.ndarray,
    ) -> np.ndarray:
        """E082 -- distance to the click crop minus distance to the best refreshed ref.

        ``contrast > 0`` means the candidate looks MORE like the refreshed references
        (the EMA anchor R2 and the long-term gallery R3) than like the frozen click
        crop R1: the crop, not the candidate, is the stale reference.  That is the
        quantity E082 gates on, and it is deliberately a bare difference of three
        distances the vote already computes, so no new scale or constant enters.

        Non-finite inputs propagate.  A candidate the gallery cannot score keeps
        ``+inf`` and gets ``-inf``/NaN contrast, and ``inf - inf`` becomes NaN; the
        gate rejects every non-finite contrast rather than reading it as a huge
        margin.
        """
        with np.errstate(invalid="ignore"):
            return np.asarray(dist_r1, np.float64) - np.minimum(
                np.asarray(dist_r2, np.float64), np.asarray(dist_r3, np.float64)
            )

    def _vote_note_contrast(self, contrast: np.ndarray, eligible: np.ndarray) -> None:
        """E082 -- accumulate the contrast retention curve over ELIGIBLE candidates.

        ``eligible`` is the PRE-gate admission mask (the blanket k-of-3 answer), so the
        curve answers exactly the question the delta grid asks: of the candidates E079
        would have rescued, how many still clear each contrast threshold?  Reported in
        every mode so the grid can be chosen from data and an arm where the gate never
        binds is visible as ``vote_contrast_blocked == 0``.
        """
        mask = np.asarray(eligible, dtype=bool)
        if not bool(np.any(mask)):
            return
        c = np.asarray(contrast, np.float64)[mask]
        finite = c[np.isfinite(c)]
        self._vote_contrast_n += int(c.size)
        if finite.size:
            self._vote_contrast_sum += float(np.sum(finite))
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            self._vote_contrast_min = min(self._vote_contrast_min, lo)
            self._vote_contrast_max = max(self._vote_contrast_max, hi)
            self._last_diag["vote_contrast_min"] = lo
            self._last_diag["vote_contrast_max"] = hi
        for key, thr in zip(self._vote_contrast_ge, _CONTRAST_PROBES):
            self._vote_contrast_ge[key] += int(np.count_nonzero(finite >= thr))

    def _vote_admit(
        self,
        feats: np.ndarray,
        dist_r1: np.ndarray,
        over: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        """Which anchor-vetoed candidates the k-of-3 vote lets through.

        ``feats`` / ``dist_r1`` / ``over`` are the (n, D) descriptor stack, the
        distance of each candidate to the frozen click crop (R1) and the boolean
        mask of candidates R1 refused.  Returns a boolean mask of the SAME length
        in which ``True`` means "the vote admits this candidate"; every entry
        outside ``over`` is ``False`` because those candidates were never vetoed
        in the first place.

        R1 is recomputed from ``dist_r1`` rather than assumed ``False`` so the
        arithmetic is testable on its own: for a candidate that reaches this
        method R1 has already rejected it, which is exactly why ``unanimous``
        (k = 3) admits nothing and is the mechanism's null control.
        """
        admitted = np.zeros(int(over.size), dtype=bool)
        if self._vote_k <= 0 or not bool(np.any(over)):
            return admitted
        # E081 -- the distress gate.  Only 'adaptive' consults it; tau < 0 disables
        # it entirely, which is what makes 'adaptive' with tau < 0 bit-identical to
        # 'majority' and tau = 1.0 (never distressed) identical to 'off'.  Those two
        # end points are the experiment's controls and are pinned by tests.
        if self._vote_mode == "adaptive" and self._vote_tau >= 0.0:
            distress = self._vote_distress()
            self._last_diag["vote_distress"] = float(distress)
            if distress <= float(self._vote_tau):
                self._vote_gated += 1
                self._last_diag["vote_gated"] = 1
                return admitted
        idx = np.flatnonzero(over)
        self._vote_frames += 1
        self._vote_n += int(idx.size)

        r1 = np.asarray(dist_r1, np.float64)[idx] <= float(limit)
        #: d2/d3 are kept as float64 so the E082 contrast is computed from exactly the
        #: numbers the vote compared against the band.  A missing R2 or an unscorable
        #: gallery entry stays +inf, which fails ``<= limit`` exactly like the all-False
        #: masks they produced before -- bit-identical for the E079/E081 arms.
        d2 = np.full(idx.size, np.inf, dtype=np.float64)
        if self._vote_r2 is not None:
            d2 = np.asarray(cosine_distance_matrix(feats[idx], self._vote_r2).ravel(), np.float64)
        r2 = d2 <= float(limit)
        d3 = np.asarray(self._vote_gallery_distances(feats[idx]), np.float64)
        r3 = np.isfinite(d3) & (d3 <= float(limit))

        passes = r1.astype(np.int8) + r2.astype(np.int8) + r3.astype(np.int8)
        ok = passes >= int(self._vote_k)
        self._vote_r1_pass += int(np.count_nonzero(r1))
        self._vote_r2_pass += int(np.count_nonzero(r2))
        self._vote_r3_pass += int(np.count_nonzero(r3))

        # E082 -- the contrast gate.  Instrumented in EVERY mode (the curve is a
        # measurement), applied only by 'contrast'.  At delta = 0 the gate cannot
        # remove a candidate the vote admitted: such a candidate has R1 in the
        # minority, i.e. d_R1 > limit >= min(d_R2, d_R3), so contrast > 0 and the
        # gate is a no-op.  That is what pins 'contrast' delta = 0 to 'majority'
        # bit-for-bit, and it is the experiment's plumbing falsifier.
        #
        # Non-finite contrast obeys the plan literally: NaN compares False and is
        # rejected (an unusable R1 distance is not evidence that the refs are right),
        # +inf passes every delta (an infinitely stale crop IS the strongest evidence
        # for the refreshed refs) and -inf never passes.  The only place delta = 0
        # could differ from 'majority' is therefore a NaN R1 distance, which needs a
        # NaN descriptor; the sweep's per-sequence identity check pins that it never
        # happens.
        contrast = self._vote_contrast(np.asarray(dist_r1, np.float64)[idx], d2, d3)
        self._vote_note_contrast(contrast, ok)
        if self._vote_mode == "contrast" and self._vote_delta >= 0.0:
            gate = contrast >= float(self._vote_delta)
            blocked = int(np.count_nonzero(ok & ~gate))
            self._vote_contrast_blocked += blocked
            self._last_diag["vote_contrast_blocked"] = blocked
            ok = ok & gate

        self._vote_rescued += int(np.count_nonzero(ok))
        admitted[idx] = ok
        self._vote_rescued_idx = {int(k) for k in idx[ok]}
        if self._bed_log is not None:
            # E084 step 0: keep the per-candidate bed numbers so the caller can log
            # them together with the candidate boxes that only it knows about.
            self._bed_last = {
                "idx": idx,
                "d1": np.asarray(dist_r1, np.float64)[idx],
                "d2": d2,
                "d3": d3,
                "limit": float(limit),
                "passes": passes,
                "ok": ok,
            }

        self._last_diag["vote_n"] = int(idx.size)
        self._last_diag["vote_r1"] = int(np.count_nonzero(r1))
        self._last_diag["vote_r2"] = int(np.count_nonzero(r2))
        self._last_diag["vote_r3"] = int(np.count_nonzero(r3))
        self._last_diag["vote_rescued"] = int(np.count_nonzero(ok))
        return admitted

    def _bed_log_frame(self, sub: list, veto) -> None:
        """E084 step 0: one JSONL line per ceiling-refused candidate (opt-in, off by default)."""
        last = self._bed_last
        if self._bed_log is None or last is None or not bool(np.any(veto)):
            return
        import json as _json

        def _num(x) -> float | None:
            v = float(x)
            if v != v or v == float("inf") or v == float("-inf"):
                return None
            return v

        for pos, cand in enumerate(last["idx"]):
            det = sub[int(cand)]
            box = [round(float(v), 2) for v in np.asarray(det.xyxy, dtype=float).ravel()[:4]]
            self._bed_log.write(_json.dumps({
                "frame": int(self.frame_idx),
                "cand": int(cand),
                "box": box,
                "d1": _num(last["d1"][pos]),
                "d2": _num(last["d2"][pos]),
                "d3": _num(last["d3"][pos]),
                "limit": last["limit"],
                "passes": int(last["passes"][pos]),
                "admitted": bool(last["ok"][pos]),
            }) + "\n")
        self._bed_log.flush()

    def _vote_update_r2(self, cfg: Any, feat: np.ndarray | None) -> None:
        """Move the refreshed anchor R2 towards an accepted crop -- if R1 and R3 agree.

        The conjunction is the anti-poisoning rule and it is the load-bearing part
        of the design: the only frames allowed to teach R2 are frames on which the
        unpoisonable click crop (R1) AND the gallery (R3) both already call this the
        target, so a drift the gallery absorbed cannot also train the reference
        that will later admit candidates.  A frame where either disagrees is simply
        not evidence about the target's current look.
        """
        if self._vote_k <= 0 or self._vote_r2 is None or feat is None:
            return
        limit = self._anchor_limit(cfg)
        if limit is None:
            return
        vec = np.asarray(feat, np.float32).ravel().reshape(1, -1)
        if not bool(np.all(np.isfinite(vec))):
            return
        d1 = float(cosine_distance_matrix(vec, self._anchor_feat)[0, 0])
        if not (np.isfinite(d1) and d1 <= float(limit)):
            return
        d3, _entry = self.ltm.best(vec[0])
        if not (np.isfinite(d3) and float(d3) <= float(limit)):
            return
        alpha = float(self._vote_alpha)
        new = (1.0 - alpha) * self._vote_r2 + alpha * vec[0]
        norm = float(np.linalg.norm(new))
        if not np.isfinite(norm) or norm <= 1e-8:
            return
        self._vote_r2 = (new / norm).astype(np.float32)
        self._vote_r2_updates += 1

    def _vote_note_band(self, cfg: Any, limit: float) -> None:
        """Count frames on which the E029 band is pinned at its ceiling.

        E078 measured a ~75 % saturation share; the same number decides what the
        vote is actually voting on, so it is part of the run record for BOTH arms
        (the vote does not change the band).
        """
        try:
            hi = float(getattr(cfg, "anchor_adaptive_ceil", 0.75))
            adaptive = bool(getattr(cfg, "anchor_adaptive", True))
        except (TypeError, ValueError):
            return
        sat = bool(adaptive and float(limit) >= hi - 1e-9)
        self._band_frames += 1
        if sat:
            self._band_saturated += 1
        self._last_diag["anchor_band"] = float(limit)
        self._last_diag["anchor_band_sat"] = sat

    # ------------------------------------------------------------------ #
    # the per-frame entry point
    # ------------------------------------------------------------------ #
    def step(
        self,
        frame_bgr: np.ndarray,
        dets: Sequence[Detection] | None,
        frame_idx: int | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> TargetStatus:
        """Advance one frame and return what the target did.

        Args:
            frame_bgr: the frame (used only for appearance encoding).
            dets: all person detections this frame. May be empty or None.
            frame_idx: frame number; defaults to a self-managed counter.
            frame_shape: ``(height, width)`` override when ``frame_bgr`` is
                unavailable (metadata-only replay).
        """
        if not self.started:
            raise RuntimeError("TargetTracker.initialize() must be called before step()")

        h, w = self._frame_hw(frame_bgr) if frame_bgr is not None else self._frame_hw(frame_shape)
        if frame_idx is None:
            self.frame_idx += 1
        else:
            self.frame_idx = int(frame_idx)
        self.age += 1
        # E071: remembered so that a refused challenger can leave the
        # incumbent's time-since-seen untouched -- a pending challenge must not
        # be counted as a miss (docs/E071_PLAN.md, trap 1).
        fsu_before = self.frames_since_update
        self.frames_since_update += 1
        self._reacquired = False

        dets = list(dets or [])
        self._frame_dets = dets
        # E059: keep a short per-detection tube so the search path has a motion
        # descriptor to compare against the target's own history.
        _need_tubes = (
            float(getattr(self.cfg.cost, "trail_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.cfg.cost, "tube_velocity_weight", 0.0) or 0.0) > 0.0
        )
        if dets and _need_tubes:
            if self._trail is None:
                self._trail = TrailLinker(
                    max_len=int(getattr(self.cfg.cost, "trail_len", 16) or 16),
                    min_len=int(getattr(self.cfg.cost, "trail_steps", 8) or 8),
                )
            self._trail_desc = self._trail.update(dets, self.frame_idx)
        else:
            self._trail_desc = None
        prev_state = self.state
        # E081: sample distress from the state this frame enters in (trailing).
        self._vote_sample()

        result, breakdown, chosen = self._associate(frame_bgr, dets, h, w)
        if self._preveto_on:
            # E099: retain the pre-veto pool with the shipped choice as its prior.
            self._preveto_record(chosen)

        if result.accepted and chosen is not None:
            if self._provisional_reacquire_ok(chosen, prev_state):
                self._on_match(frame_bgr, chosen, result, h, w)
                self._check_drift(chosen)
                self._audit_switch(frame_bgr, dets, chosen, h, w)
            else:
                # E092: the candidate is plausible but has not persisted yet, so the
                # frame is a miss for identity purposes -- keep the prediction, keep
                # counting time lost, and let the candidate keep asking.
                self._on_miss(prev_state, result)
        elif str(result.reason).startswith(REASON_CONSENSUS_PENDING):
            # E071: the frame's winner is a challenger the vote has not
            # confirmed yet, so the incumbent keeps the identity. This is a
            # refusal of the challenger, NOT a missed observation of the
            # incumbent: undo the miss bookkeeping and hold the last state.
            self.frames_since_update = fsu_before
            self._on_consensus_pending()
        else:
            self._on_miss(prev_state, result)
            # E099: the shipped tracker just refused every candidate while coasting.
            # That is exactly the regime E097 measured 65.84 % pool recall in, so let
            # the bounded window over the pre-veto pool speak -- but only for a
            # different person, never for a near variant of the committed box.
            if self._preveto_on and fsu_before >= int(
                getattr(self.cfg.state, "lost_short_after", 12) or 12
            ):
                self._preveto_try_reattribute(frame_bgr, result, h, w)
        # ---- out-of-frame bookkeeping (independent of detection) --------
        self._update_out_of_frame(w, h)
        if self._box is not None and h and w:
            try:
                self._last_diag["partial"] = estimate_partial_state(self._box, w, h)["kind"]
            except Exception:
                pass

        self._last_diag = dict(self._last_diag)
        self._last_diag.update(
            {
                "state": self.state.value,
                "accepted": bool(result.accepted),
                "reason": result.reason,
                "cost": float(result.cost) if np.isfinite(result.cost) else float("inf"),
                "margin": None if not np.isfinite(result.margin) else float(result.margin),
                "n_dets": len(dets),
                "fsu": self.frames_since_update,
            }
        )
        if self._deferred_frame:
            # E092: make the deferral explicit for the debug CSV and the refusal counts.
            self._last_diag["reason"] = "deferred:reacquire_pending"
            self._last_diag["accepted"] = False
            self._last_diag["reacquire_pending_hits"] = int(self._pending_hits)
            self._deferred_frame = False
        if breakdown is not None:
            self._last_diag["gate_pass"] = int(np.count_nonzero(breakdown.gate))
        return self.status()

    # ------------------------------------------------------------------ #
    # association
    # ------------------------------------------------------------------ #
    def _associate(
        self,
        frame_bgr: np.ndarray | None,
        dets: list[Detection],
        h: int,
        w: int,
    ) -> tuple[MatchResult, CostBreakdown | None, Detection | None]:
        """Two-stage association: cheap motion gate, then appearance.

        Encoding every detection every frame would dominate the runtime, and
        most detections are nowhere near the target. Stage one costs a few
        matrix operations and throws away everything outside the gate; only the
        survivors pay for a ReID forward pass.
        """
        cfg: CostCfg = self.cfg.cost

        # E079: per-frame diagnostics must not leak into the next frame (``_last_diag``
        # is copied forward in ``step``), and the rescued-candidate set is per frame.
        for _k in ("vote_n", "vote_r1", "vote_r2", "vote_r3", "vote_rescued",
                   "vote_rescued_accepted", "anchor_band", "anchor_band_sat"):
            self._last_diag.pop(_k, None)
        self._vote_rescued_idx = set()

        # E031: a truncated view gives a systematically biased box (half the height,
        # centre offset by h/4), so its geometry deserves less weight than its
        # appearance.  Apply the override only while the view is actually partial.
        pmw = getattr(cfg, "partial_motion_weight", None)
        if pmw is not None and self._in_degenerate_view(cfg):
            cfg = replace(
                cfg,
                w_motion=float(pmw),
                w_appearance=float(
                    getattr(cfg, "partial_appearance_weight", None) or cfg.w_appearance
                ),
            )

        # The motion model must take exactly ONE time step per frame, whether or
        # not anything was detected.  ``coast()`` is idempotent *within* a frame
        # by contract (motion.py:794-806 -- once ``_predicted`` is set it only
        # damps the velocity and returns the same box), and nothing clears that
        # flag except an observation.  So if we skipped ``predict()`` on frames
        # with no detections, the first ``coast()`` would advance the box and
        # every later one would be a no-op: the target would freeze in place
        # forever and could never walk out of frame.
        pred = self.motion.predict() if self.motion.started else self.box
        if not dets:
            b = compute_costs([], None, cfg)
            return select_candidate(b, cfg, lost=self.is_blind), b, None

        vel = self.motion.velocity() if self.motion.started else None
        last = self.motion.history()[-1] if self.motion.started and self.motion.history() else None

        # ---- stage 1: geometry only --------------------------------------
        searching = self._global_search()
        frame_diag = float(np.hypot(float(w), float(h)))
        cheap = compute_costs(
            dets,
            pred,
            cfg,
            velocity=vel,
            last_box=last,
            motion=self.motion if self.motion.started else None,
            frame_shape=(h, w),
            apply_gate=not searching,
            apply_motion=not searching,
            gate_radius_override=self._gate_override(frame_diag),
        )
        keep = self._encoding_set(dets, cheap)
        if not keep:
            self._last_breakdown = cheap
            self._encoded_dets = []
            return select_candidate(cheap, cfg, lost=self.is_blind), cheap, None

        # ---- stage 2: appearance on the survivors ------------------------
        self._ensure_features(frame_bgr, [dets[i] for i in keep], h, w)
        sub = [dets[i] for i in keep]
        if self._refute_on:
            self._refute_by_frame[int(self.frame_idx)] = [
                [float(v) for v in np.asarray(d.xyxy, float).ravel()[:4]] for d in sub
            ]
        self._encoded_dets = sub
        full = compute_costs(
            sub,
            pred,
            cfg,
            velocity=vel,
            last_box=last,
            motion=self.motion if self.motion.started else None,
            # E022: while searching, the short-term memory describes the person
            # we are walking away from; letting it vote is how a latch gets
            # confirmed.  Score the search on the long-term click anchor instead.
            stm_feat=(self.stm.current() if (not searching or cfg.search_uses_stm) else None),
            stm_part_feats=((self._part_ema or None) if (not searching or cfg.search_uses_stm) else None),
            ltm=self.ltm,
            frame_idx=self.frame_idx,
            frame_shape=(h, w),
            apply_gate=not searching,
            apply_motion=not searching,
            gate_radius_override=self._gate_override(frame_diag),
        )
        # E097 diagnostic: the per-candidate appearance cost of the *pre-veto* pool.
        # It must be read from ``full`` (stage 2, after the encoding gate) because the
        # cheap stage-1 pass has no appearance term at all -- reading it there raised
        # UnboundLocalError and killed every ``lag_refute_dump`` run.
        if self._refute_on:
            self._refute_app_by_frame[int(self.frame_idx)] = [
                float(v) for v in np.asarray(full.appearance, np.float64).ravel()
            ]
        # E099: hand the pre-veto pool to ``step()``, which is where the shipped
        # choice becomes known (the DP needs it as its switching prior).
        if self._preveto_on:
            self._preveto_pending = (
                list(sub),
                np.asarray(full.appearance, np.float64).ravel().copy(),
            )
        # E059: motion-signature evidence, applied as a soft cost and ONLY while
        # searching.  During normal tracking the motion model already owns this
        # information; the point is to give the search path a channel that does not
        # depend on appearance, which on DanceTrack does not separate dancers.
        tw = float(getattr(cfg, "trail_weight", 0.0) or 0.0)
        if searching and tw > 0.0 and len(sub) and self._trail_desc is not None:
            tdesc = self._target_trail_descriptor(cfg)
            if tdesc is not None:
                keep_t = [self._trail_desc[i] for i in keep]
                dists = np.asarray(
                    [trail_distance(tdesc, d) for d in keep_t], dtype=np.float64
                )
                known = np.isfinite(dists)
                if known.any():
                    add = np.where(known, tw * dists, 0.0)
                    full.total = np.asarray(full.total, np.float64) + add
                    self._last_diag["trail_n"] = int(known.sum())
                    self._last_diag["trail_min"] = float(np.min(dists[known]))

        # E061: tube-velocity consistency.  The DanceTrack swap (E060 trace, f54-56)
        # is a tie-break between two detections that overlap the prediction almost
        # equally; appearance and size are indistinguishable there, but the two
        # people are walking in different directions.  The target's direction is the
        # Kalman velocity; each candidate's is its own tube's recent delta.  Applied
        # during tracking as well as searching -- this is an in-tracking failure.
        tvw = float(getattr(cfg, "tube_velocity_weight", 0.0) or 0.0)
        if tvw > 0.0 and len(sub) and self.motion.started and self._trail is not None:
            vels = getattr(self._trail, "vels", None) or []
            tv = np.asarray(self.motion.velocity(), dtype=np.float64).ravel()[:2]
            t_speed = float(np.hypot(tv[0], tv[1]))
            min_speed = float(getattr(cfg, "tube_velocity_min_speed", 0.004) or 0.004)
            if t_speed > min_speed and len(vels) == len(dets):
                add = np.zeros(len(sub), dtype=np.float64)
                for k, di in enumerate(keep):
                    v = vels[di] if di < len(vels) else None
                    if v is None:
                        continue
                    add[k] = tvw * velocity_disagreement(v, tv, min_speed)
                if add.any():
                    full.total = np.asarray(full.total, np.float64) + add
                    self._last_diag["tube_vel_n"] = int(np.count_nonzero(add))
                    self._last_diag["tube_vel_max"] = float(add.max())

        # E027: an identity ceiling against the frozen click crop.  Applied only
        # as a VETO over the candidates that already have a measurement, so it can
        # never make an unreadable candidate look admissible.
        limit = self._anchor_limit(cfg)
        search_only = bool(getattr(cfg, "anchor_ceiling_search_only", False))
        if limit is not None:
            self._vote_note_band(cfg, float(limit))
        # E089: the band above is recorded at its while-tracking value; the veto below
        # uses the lost-widened one.  Identical to `limit` at the default scale 1.0.
        limit = self._lost_scaled_limit(cfg, limit)
        if limit is not None and (searching or not search_only) and self._anchor_feat is not None and len(sub):
            feats = [d.feat for d in sub]
            if all(f is not None for f in feats):
                # .ravel(): a single-feature bank comes back as (N, 1), and an
                # (N, 1) mask broadcast against the (N,) cost array silently
                # produces an (N, N) cost matrix -> "only length-1 arrays can be
                # converted to Python scalars" deep inside select_candidate.
                feat_arr = np.asarray(feats, np.float32)
                dist = cosine_distance_matrix(feat_arr, self._anchor_feat).ravel()
                # E091: the veto threshold may be per-candidate (widened only for
                # candidates already inside the motion gate); at the default knob it is
                # the plain scalar and this is exactly the shipped comparison.
                limit_eff = self._lost_geom_limit(cfg, limit, full)
                if isinstance(limit_eff, np.ndarray):
                    over = dist > np.asarray(limit_eff, np.float64)
                else:
                    over = dist > float(limit)
                # E079: a candidate the click crop rejects is admitted iff at
                # least k of the three references pass it.  ``admitted`` is all
                # False while the vote is off, so ``veto`` is then exactly the
                # pre-E079 ``over`` mask and the assignment below is unchanged.
                admitted = self._vote_admit(feat_arr, dist, over, float(limit))
                veto = over & ~admitted
                if veto.any():
                    full.total = np.where(veto, np.inf, full.total)
                    full.excluded = [
                        (full.excluded[i] if i < len(full.excluded) else "")
                        or ("anchor_ceiling" if veto[i] else "")
                        for i in range(len(veto))
                    ]
                    self._last_diag["anchor_rejected"] = int(veto.sum())
                    # E084 step 0: opt-in record of EVERY ceiling-refused candidate
                    # (the `over` mask, not the vote's residual veto) together with the
                    # three bed distances, so the ceiling itself can be audited against
                    # GT offline -- the question is "which bed refuses the true target",
                    # which is about `over`, and the vote only ever sees its remainder.
                    self._bed_log_frame(sub, over)
        # E032: adaptive motion band, applied as a veto exactly like the appearance
        # band.  ``maha`` is inf when no motion model exists, so it is naturally inert
        # before the filter has started.
        mlim = self._motion_limit(cfg)
        if mlim is not None and len(sub):
            over_m = np.asarray(full.maha, np.float64) > float(mlim)
            # E032b: a large motion residual alone is NOT evidence of a swap -- 0004's
            # target legitimately jumps (fast rotation/crossing), which is why the plain
            # band cost it 20 pp of TSR.  Veto only when the candidate ALSO fails the
            # appearance band: "arrived from the wrong place" *and* "does not look like
            # the target".  Either axis alone is ambiguous (E029/E032); the conjunction
            # is the discriminator.
            alim = self._anchor_limit(cfg)
            if alim is None:
                alim = float(getattr(cfg, "motion_band_app_floor", 0.30))
            over_m = over_m & (np.asarray(full.appearance, np.float64) > float(alim))
            if over_m.any():
                full.total = np.where(over_m, np.inf, full.total)
                full.excluded = [
                    (full.excluded[i] if i < len(full.excluded) else "")
                    or ("motion_band" if over_m[i] else "")
                    for i in range(len(over_m))
                ]
                self._last_diag["motion_band_rejected"] = int(over_m.sum())
        res = select_candidate(full, cfg, lost=self.is_blind)
        self._last_breakdown = full
        # E077: record which channel (if any) carried the frame.  The channel is
        # only counted when the acceptance actually stands, which is decided after
        # the search-margin / search-needs-appearance rules below -- so the count
        # is taken at the end of the method, from the survivor.
        if res.channel in self._channel_counts:
            self._last_diag["accept_channel"] = res.channel
        else:
            self._last_diag.pop("accept_channel", None)
        for _k in ("norm_centre", "log_h_ratio"):
            _v = (res.diag or {}).get(_k)
            if _v is None:
                self._last_diag.pop("ch_" + _k, None)
            else:
                self._last_diag["ch_" + _k] = float(_v)
        _app = (res.diag or {}).get("appearance")
        if _app is not None:
            self._last_diag["ch_app_cost"] = float(_app)
            self._last_diag["ch_app_valid"] = bool((res.diag or {}).get("appearance_valid", False))
        if res.accepted and not searching and self._anchor_feat is not None and len(sub):
            feat = sub[int(res.index)].feat
            if feat is not None:
                self._bootstrap_anchor(feat)
                self._anchor_dist_hist.append(
                    float(cosine_distance_matrix(
                        np.asarray(feat, np.float32)[None, :], self._anchor_feat
                    )[0, 0])
                )
                del self._anchor_dist_hist[:-int(getattr(cfg, "anchor_adaptive_window", 30))]
            if np.isfinite(float(full.maha[int(res.index)])):
                self._maha_hist.append(float(full.maha[int(res.index)]))
                del self._maha_hist[:-int(getattr(cfg, "motion_gate_window", 30))]
        if searching and res.accepted and cfg.search_margin is not None:
            # E022: a re-acquisition must WIN, not merely finish first.  In a
            # crowd the winner is often a hair ahead of a dozen lookalikes, which
            # is exactly the coin-flip the margin test exists to refuse.
            need = float(cfg.search_margin)
            if np.isfinite(res.margin) and float(res.margin) < need:
                res = MatchResult(
                    index=res.index,
                    cost=float(res.cost),
                    accepted=False,
                    reason=f"search_margin: {float(res.margin):.3f} < {need:.3f}",
                    margin=float(res.margin),
                    diag=dict(res.diag or {}),
                )
        if searching and res.index is not None:
            # During a global search, geometry is not evidence: any person
            # standing where we last saw the target would be accepted.  A rebind
            # must therefore be justified by appearance.
            #
            # If the best candidate has NO usable appearance, the honest answer
            # is to keep searching - NOT to fall back to the next candidate that
            # happens to have some.  Measured on MOT17-02 with the target's upper
            # body masked (real yolo26m@1280, 60 frames): the fallback accepted a
            # fully visible bystander at frame 14 and stayed on the wrong person
            # for the remaining 46 frames - TSR 0.23, precision 0.21, one false
            # acquisition - because a half-occluded target has a degraded
            # descriptor while an unoccluded bystander does not.  Preferring
            # "I do not know" over "it must be that other person" is the whole
            # point of this tracker, so an unverifiable best candidate is a
            # refusal, not a reason to look further down the list.
            valid = np.asarray(full.appearance_valid, dtype=bool)
            if not bool(valid[int(res.index)]):
                res = MatchResult(
                    index=None,
                    cost=float(res.cost),
                    accepted=False,
                    reason="search_needs_appearance",
                    margin=float(res.margin),
                    diag=dict(res.diag or {}),
                )
        # ---- E071: temporal consensus on the handback ----------------------
        # Deliberately LAST, after every existing veto.  Those answer "is this
        # candidate the target?", and a candidate they refuse never reaches this
        # gate; the vote only ever answers the different question "is this a NEW
        # person taking over from the one we already report?".  Wiring it before
        # the vetoes would let a challenger accumulate votes on frames where the
        # tracker was never going to accept anything, which would turn a
        # sustained refusal into an eventual handover -- the opposite of the
        # intent.
        if self._challenge is not None and not self.is_blind:
            if res.accepted and res.index is not None:
                res = self._consensus_gate(res, sub[int(res.index)])
            else:
                # No proposal this frame, so no vote: age any open chain instead
                # of letting it survive unbounded.  A one-frame lookalike blip
                # dies here, and that abandonment is the "we refused a tie"
                # event the campaign wants to count.
                self._last_diag.update(self._challenge.tick(self.frame_idx).as_diag())
        # E077: count the channel only now, when `accepted` is final -- a candidate
        # the OR-branch admitted and the search/consensus rules then refused is not
        # an engagement of the mechanism.
        if res.accepted and res.channel in self._channel_counts:
            self._channel_counts[res.channel] += 1
            self._last_diag["accept_channel"] = res.channel
        elif not res.accepted:
            self._last_diag.pop("accept_channel", None)
        if res.accepted and res.index is not None:
            # E079: did the vote actually carry THIS frame?  A rescued candidate that
            # some later rule (search margin, appearance requirement, consensus)
            # refuses is not an engagement of the mechanism, so the count is taken
            # from the survivor, exactly like the E077 channel count above.
            if int(res.index) in self._vote_rescued_idx:
                self._vote_rescued_accepted += 1
                self._last_diag["vote_rescued_accepted"] = 1
            # E079: R2 may move only on a frame whose final accepted box ALREADY
            # passes both the frozen click crop and the gallery.  Placed here, after
            # every veto, so a candidate that was admitted and then refused cannot
            # teach the reference.
            if not searching and self._vote_k > 0:
                self._vote_update_r2(cfg, sub[int(res.index)].feat)
        # E077 evidence: WHY the frame was refused.  The OR-branch only ever
        # bypasses the "cost > bound" test, so a refusal that arrives as
        # `all_gated:` (every candidate vetoed upstream, `total = inf`) is proof
        # that the arm could not have engaged there -- a distinction the metrics
        # alone cannot make.
        if not res.accepted:
            self._last_diag["refuse_reason"] = str(res.reason)[:160]
        else:
            self._last_diag.pop("refuse_reason", None)
        if res.index is not None:
            return res, full, sub[int(res.index)]
        return res, full, None

    # ------------------------------------------------------------------ #
    # E071 consensus
    # ------------------------------------------------------------------ #
    def _consensus_is_incumbent(self, feat: np.ndarray | None) -> bool:
        """Is the frame's winning candidate the person we already report?

        Three answers, and only one of them is "challenger":

        * no reference yet (the first accepted frame after the click, or a
          snapshot that carried no consensus state) -> incumbent: with nothing to
          compare against there is no incumbent to *keep*, so the vote must stay
          out of the way and the existing behaviour is preserved;
        * no usable descriptor for this candidate -> incumbent, for the same
          reason.  The identity leg is the whole point of the vote, and guessing
          "different person" from a missing descriptor would refuse candidates on
          frames where the encoder failed -- E067's failure mode wearing a new
          hat;
        * otherwise the candidate is a challenger when its cosine distance to the
          incumbent's descriptor exceeds ``consensus_ident``.
        """
        ch = self._challenge
        if ch is None or self._consensus_feat is None or feat is None:
            return True
        vec = np.asarray(feat, np.float32).reshape(-1)
        if vec.size == 0 or vec.size != self._consensus_feat.size:
            return True
        if not (np.all(np.isfinite(vec)) and np.all(np.isfinite(self._consensus_feat))):
            return True
        d = float(cosine_distance_matrix(vec[None, :], self._consensus_feat[None, :])[0, 0])
        return bool(d <= float(ch.ident_thresh))

    def _consensus_gate(self, res: MatchResult, det: Detection) -> MatchResult:
        """Feed one accepted proposal to the vote; refuse it until committed.

        The commit rule from the plan: a *non-incumbent* candidate may only take
        the identity once ``ChallengeTracker`` reports ``COMMITTED``.  Until then
        the incumbent keeps the identity, which from the association layer's
        point of view is a refusal of this candidate with reason
        ``consensus_pending`` -- not a miss (see :meth:`_on_consensus_pending`).
        """
        ch = self._challenge
        assert ch is not None  # guarded by the caller
        feat = getattr(det, "feat", None)
        if feat is not None:
            try:
                feat = np.asarray(feat, np.float32).reshape(-1)
            except Exception:  # pragma: no cover - defensive
                feat = None
        incumbent = self._consensus_is_incumbent(feat)
        report = ch.observe(
            self.frame_idx,
            np.asarray(det.xyxy, np.float64).reshape(-1)[:4],
            feat,
            incumbent=incumbent,
        )
        # An incumbent frame carries no proposal, so the chain clock has to be
        # advanced explicitly; on a proposal frame this is a no-op.
        if incumbent:
            ch.tick(self.frame_idx)
        self._last_diag.update(report.as_diag())
        if incumbent or report.state == ChallengeState.COMMITTED:
            if report.state == ChallengeState.COMMITTED:
                self._emit(
                    "consensus_commit",
                    span=int(report.span),
                    pending_n=int(report.pending_n),
                    committed=int(ch.n_committed),
                )
            return res
        self._last_diag["consensus_refused"] = 1
        self._emit(
            "consensus_pending",
            pending_n=int(report.pending_n),
            state=str(report.state),
            why=str(report.reason),
        )
        return MatchResult(
            index=res.index,
            cost=float(res.cost),
            accepted=False,
            reason=REASON_CONSENSUS_PENDING,
            margin=float(res.margin),
            diag=dict(res.diag or {}),
        )

    def _on_consensus_pending(self) -> None:
        """The chosen candidate was refused by the vote; keep the incumbent.

        Crucially this is *not* :meth:`_on_miss`: the incumbent was not observed,
        but it was not contradicted either, so ``frames_since_update`` was already
        restored by the caller and the state machine must not escalate.  Counting
        a pending challenge as a miss would push TRACKING -> SOFT_OCCLUSION after
        ``soft_occlusion_after`` frames, i.e. it would turn the vote into a blind
        window -- exactly the cascade ``docs/E071_PLAN.md`` warns about, and it
        would freeze appearance memory right when the challenger's crops are the
        evidence needed to decide.
        """
        self._last_diag["consensus_pending"] = 1
        if self.motion.started and self.state is not TargetState.GAVE_UP:
            # Report the incumbent's coasted position: refusing the challenger
            # must not freeze the operator's box on the spot where the
            # challenger happened to be.
            try:
                self._box = np.asarray(self.motion.coast(), np.float32)
            except Exception:  # pragma: no cover - defensive
                pass

    def _global_search(self) -> bool:
        """True when the target could be anywhere (so the motion gate is off).

        Which states count is configurable via ``StateCfg.global_states``.  The
        gate is only one of two guards: ``_associate`` additionally demands
        appearance evidence from any candidate accepted while searching, so
        widening the search does not by itself license a latch.
        """
        if self.state in (TargetState.GAVE_UP,):
            return True
        names = tuple(getattr(self.cfg.state, "global_states", ()) or ())
        return self._was_out_of_frame or self.state.value in names

    def _gate_override(self, frame_diag: float | None = None) -> float | None:
        """Widen the gate with time-since-seen so a fast target is not lost.

        ``CostCfg.gate_base`` and ``gate_speed_gain`` are *fractions of the
        frame*, not pixel counts - the tracked path applies them through
        ``association.gate_radius``, which multiplies the base by the frame
        diagonal.  This override used to call ``MotionModel.gate_radius()``,
        which returns ``gate_base + gate_speed_gain * speed`` in bare pixels;
        on 1080p that is ~15 px where the canonical radius is ~600 px, so a
        target that reappeared a few hundred pixels away was rejected as
        ``outside_radius`` forever.  Both paths now agree.
        """
        if self.state is TargetState.TRACKING:
            return None
        fsu = max(self.frames_since_update, 0)
        cfg: CostCfg = self.cfg.cost
        # The velocity the target had when we last actually saw it.  Using the
        # coasted velocity makes BOTH terms decay as 0.98**fsu, so the radius
        # stopped growing (measured peak 264 px at fsu=49, then 258 px at
        # fsu=61) exactly when it was needed most - hence the freeze, which is
        # what makes the radius strictly monotone in time-since-seen.
        freeze = bool(getattr(self.cfg.state, "gate_growth_freeze_speed", True))
        if freeze:
            speed = float(self._v0_speed)
            vel = np.array([speed, 0.0], dtype=np.float32) if speed > 0 else None
        else:
            speed = self.motion.speed() if self.motion.started else 0.0
            vel = self.motion.velocity() if self.motion.started else None
        base = gate_radius(vel, cfg.gate_base, cfg.gate_speed_gain, frame_diag)
        grown = base + speed * fsu
        ref = float(frame_diag) if frame_diag and frame_diag > 0 else 640.0
        # Never let the gate exceed the frame: past that point every candidate is
        # inside it and the gate is simply off, which the appearance guard (not
        # a radius) is there to police.
        return float(min(grown, ref))

    def _encoding_set(self, dets: list[Detection], cheap: CostBreakdown) -> list[int]:
        """Indices worth an appearance forward pass.

        While the target is tracked we only look near the prediction. While it
        is lost we must consider everything, but we still cap the batch so one
        crowded frame cannot stall the pipeline.
        """
        n = len(dets)
        cfg: CostCfg = self.cfg.cost
        if self.embedder is None:
            # No encoder: keep whatever features the caller supplied. Nothing
            # to encode, so just keep all gated candidates.
            gated = np.flatnonzero(np.asarray(cheap.gate, bool))
            return [int(i) for i in gated][:MAX_ENCODE]

        if self._global_search():
            order = np.argsort([-float(d.score) for d in dets])[:MAX_ENCODE]
            return sorted(int(i) for i in order)

        finite = np.isfinite(np.asarray(cheap.total, np.float32))
        idx = [int(i) for i in np.flatnonzero(finite)]
        idx.sort(key=lambda i: float(cheap.total[i]))
        top_k = max(int(getattr(cfg, "app_topk", 5)) * 3, 12)
        return idx[: min(top_k, MAX_ENCODE)]

    def _ensure_features(
        self,
        frame_bgr: np.ndarray | None,
        dets: list[Detection],
        h: int,
        w: int,
    ) -> None:
        """Encode target-relevant parts for detections that lack features."""
        if self.embedder is None or frame_bgr is None or not dets:
            return
        todo = [d for d in dets if not d.part_feats and d.feat is None]
        if not todo:
            return
        parts = self._query_parts()
        boxes = np.stack([d.xyxy for d in todo], axis=0).astype(np.float32)
        try:
            enc = self.embedder.encode(frame_bgr, boxes, parts)
        except Exception as exc:  # a broken encoder must not kill tracking
            # ... but it must never be silent either (E067). Keep the
            # geometry-only fallback and record the failure in two places: a
            # diag key for the frame, and a counter that no per-frame reset can
            # hide.
            self._note_encode_error("ensure_features", exc)
            return
        for d, e in zip(todo, enc):
            if not e:
                continue
            d.part_feats = {k: v for k, v in e.items() if v is not None}
            full = d.part_feats.get("full")
            if full is not None:
                d.feat = full
            try:
                d.visible_ratio = visible_ratio(d.xyxy, w, h)
                sharp, qual = crop_quality(frame_bgr, d.xyxy)
                d.sharpness = float(sharp)
                # Quality is deliberately the *product* of "is this crop sharp"
                # and "how much of the person is in frame": a crisp crop of a
                # half-visible person is still a partial view, and must not be
                # allowed to outweigh a full view in the gallery.
                d.quality = float(np.clip(qual * d.visible_ratio, 0.0, 1.0))
            except Exception:
                pass

    def _note_encode_error(self, where: str, exc: BaseException) -> None:
        """Record an encoder failure where it can never be silent (E067).

        `_ensure_features` and `_features_for` deliberately catch encoder errors
        so a dead encoder degrades the tracker to geometry-only instead of
        crashing a whole grid. The catch used to leave a single diag key that
        `initialize` wipes and nothing reads, so the profile that started E067
        reported "encoder = 2% of wall" for a run where the encoder was never
        called at all. Both signals are written now: the per-frame diag (with
        the frame it happened on) and a monotonic counter exposed by `stats()`.
        """
        msg = f"{type(exc).__name__}: {exc}"
        self._last_diag["encoder_error"] = msg
        self._last_diag["encoder_error_where"] = str(where)
        self._last_diag["encoder_error_frame"] = int(self.frame_idx)
        self._encode_errors += 1

    def _query_parts(self) -> tuple[str, ...]:
        """Which part crops to encode for candidates, given the target's state.

        Early in a track we ask for everything. Once the target's own memory
        tells us which parts we actually have, we stop paying for the rest.
        """
        cfg: EmbedderCfg = self.cfg.embedder
        wanted = tuple(getattr(cfg, "target_parts", ("full", "upper", "lower", "head")))
        if not self.ltm.entries():
            return wanted
        have = set(self.ltm.parts())
        # Always keep 'full': it is the fallback bank and the MOT layer's cue.
        keep = tuple(p for p in wanted if p == "full" or p in have or p in self._part_ema)
        return keep or ("full",)

    # ------------------------------------------------------------------ #
    # match / miss transitions
    # ------------------------------------------------------------------ #
    def _audit_switch(self, frame_bgr, dets, chosen, h, w) -> None:
        """E063: in-track rival audit against the frozen click anchor.

        E062 measured the failure this exists for: on the sequence we lose, the
        true target is DETECTED in 108 of the 113 wrong frames (median IoU 0.93,
        detected-anywhere rate 0.97) while the tracker sits in TRACKING on
        somebody else.  Appearance cannot catch that -- the long-term gallery
        absorbed the impostor (E036/E039) -- so the only descriptor that stays
        honest is the crop captured at the click.  Every ``audit_interval``
        frames we ask it whether some OTHER detection is ``audit_margin`` closer
        to the click than the box we are following, and switch only after
        ``audit_persist`` consecutive, spatially consistent answers.

        A switch adopts the rival's geometry, re-seeds the short-term memory with
        its appearance and keeps the click anchors, so it cannot be a way to
        wander: it is bounded by evidence measured against the click.

        E064 makes the evidence two-sided: with ``audit_use_recent`` on, the same
        rival must ALSO beat the followed box against ``stm.current()`` (the EMA
        of the recent confident crops) by ``audit_margin``.  Both margins land in
        ``_last_diag`` (``audit_margin_click`` / ``audit_margin_recent``) so a
        veto can be attributed to the reference that issued it.
        """
        cfg = self.cfg.cost
        interval = int(getattr(cfg, "audit_interval", 0) or 0)
        if interval <= 0 or self._anchor_feat is None or self._box is None:
            return
        if self.state is not TargetState.TRACKING or self.frames_since_update > 0:
            return
        if self.frame_idx - self._last_audit < interval:
            return
        cur_det = chosen
        if cur_det.feat is None:
            return
        self._last_audit = self.frame_idx
        cur_box = np.asarray(cur_det.xyxy, np.float64)
        others = [
            i for i, d in enumerate(dets)
            if float(iou_matrix(np.asarray(d.xyxy, np.float64)[None], cur_box[None])[0, 0]) < 0.5
        ]
        if not others:
            return
        need = [i for i in others if dets[i].feat is None]
        if need and self.embedder is not None and frame_bgr is not None:
            try:
                enc = self.embedder.encode(frame_bgr, [dets[i].xyxy for i in need], parts=("full",))
            except Exception:
                enc = []
            for i, e in zip(need, enc or []):
                f = None if not isinstance(e, dict) else e.get("full")
                if f is not None:
                    dets[i].feat = np.asarray(f, np.float32).ravel()
        keep = [(i, dets[i].feat) for i in others if dets[i].feat is not None]
        if not keep:
            return
        rival_d = cosine_distance_matrix(
            np.asarray([f for _, f in keep], np.float32), self._anchor_feat).ravel()
        j = int(np.argmin(rival_d))
        rival_i, rival_cost = keep[j][0], float(rival_d[j])
        cur_d = float(cosine_distance_matrix(
            np.asarray(cur_det.feat, np.float32)[None], self._anchor_feat).ravel()[0])
        margin = float(getattr(cfg, "audit_margin", 0.05) or 0.0)
        self._last_diag["audit_cur"] = round(cur_d, 4)
        self._last_diag["audit_best"] = round(rival_cost, 4)
        self._last_diag["audit_margin_click"] = round(float(cur_d - rival_cost), 4)
        # E064 -- a SECOND reference.  The click crop is the only witness the
        # impostor cannot edit (E036/E039), but it also cannot move: on a target
        # whose look legitimately changes (dancetrack0005, E024) it prefers a
        # stranger to the true target, which is how E063 vetoed a CORRECT track
        # there (0.4000 -> 0.2950 TSR).  Requiring the recent appearance to
        # agree turns that into a no-vote instead of a wrong vote.  Both tests
        # score the SAME rival (the click-best one): the question is "is this
        # switch supported by both references", not "re-rank every rival under
        # a second opinion".
        recent_ok = True
        recent_margin: float | None = None
        recent = (self.stm.current()
                  if bool(getattr(cfg, "audit_use_recent", True)) else None)
        if recent is not None and int(np.size(recent)) != int(np.size(cur_det.feat)):
            recent = None  # a feature-space mismatch is not evidence
        if recent is not None:
            try:
                rr = float(cosine_distance_matrix(
                    np.asarray([f for _, f in keep], np.float32),
                    np.asarray(recent, np.float32)).ravel()[j])
                rc = float(cosine_distance_matrix(
                    np.asarray(cur_det.feat, np.float32)[None],
                    np.asarray(recent, np.float32)).ravel()[0])
                recent_margin = float(rc - rr)
                recent_ok = bool(rr + margin < rc)
            except Exception:
                recent = None
        self._last_diag["audit_margin_recent"] = (
            None if recent_margin is None else round(recent_margin, 4))
        if not (rival_cost + margin < cur_d) or not recent_ok:
            self._audit_n = 0
            self._audit_rival = None
            return
        rbox = np.asarray(dets[rival_i].xyxy, np.float64)
        if self._audit_rival is not None and float(
                iou_matrix(np.asarray(self._audit_rival, np.float64)[None], rbox[None])[0, 0]) >= 0.3:
            self._audit_n += 1
        else:
            self._audit_n = 1
        self._audit_rival = rbox.copy()
        if self._audit_n < int(getattr(cfg, "audit_persist", 2) or 1):
            return
        self._box = rbox.copy()
        if self.motion.started:
            self.motion.reset(self._box)
            self.motion.update(self._box, score=float(dets[rival_i].score))
        self.stm.reset()
        self.stm.update(np.asarray(dets[rival_i].feat, np.float32), quality=1.0, frame_idx=self.frame_idx)
        self._audit_n = 0
        self._audit_rival = None
        self._last_diag["audit_switch"] = 1
        self._emit("audit_switch", margin=round(float(cur_d - rival_cost), 4),
                   margin_recent=None if recent_margin is None else round(recent_margin, 4))
    def _provisional_reacquire_ok(self, chosen: Detection, prev: TargetState) -> bool:
        """E092 -- may this candidate take the identity *this* frame?

        Returns True for ordinary tracking (the target was not lost), and for a
        re-acquisition only once the same candidate (IoU >= ``reacquire_confirm_iou``)
        has been offered on ``reacquire_confirm_frames`` consecutive frames.  At the
        default of 1 the function returns True immediately and touches nothing, so the
        shipped path is bit-identical.

        Why time and not geometry: E091 measured that widening the lost-regime ceiling
        for candidates *near* the stale prediction recovers a ceiling-bound sequence
        (+50.3 pp) and destroys a neighbour-switch sequence (-20.0 pp), with nothing
        spatial separating the two cases.  What does separate them is persistence -- a
        wrong latch is typically one frame, a true re-appearance stays put.
        """
        n = int(getattr(self.cfg.state, "reacquire_confirm_frames", 1) or 1)
        if n <= 1:
            return True
        if prev not in BLIND_STATES:
            # Ordinary tracking: no confirmation is owed, and any pending candidate is
            # stale by definition.
            self._pending_box = None
            self._pending_hits = 0
            self._pending_hist = []
            return True
        try:
            box = np.asarray(chosen.xyxy, np.float64).reshape(-1)[:4]
        except Exception:  # pragma: no cover - defensive
            return True
        thr = float(getattr(self.cfg.state, "reacquire_confirm_iou", 0.3) or 0.3)
        #: E092b -- with a window > 0 the rule tolerates gaps: the question becomes "have
        #: N of the last M candidates agreed with each other", not "N consecutive frames".
        #: E092's strict streak was reset by one jittered box or one alternating
        #: distractor, which is why it cost 23-28 pp of TSR on sequences whose
        #: re-acquisition was real.  0 keeps E092's exact rule (and the shipped path
        #: unchanged at n = 1).
        window = int(getattr(self.cfg.state, "reacquire_confirm_window", 0) or 0)
        if window <= 0:
            same = False
            if self._pending_box is not None:
                try:
                    same = float(
                        iou_matrix(self._pending_box[None], box[None])[0, 0]) >= thr
                except Exception:  # pragma: no cover - defensive
                    same = False
            if same:
                self._pending_hits += 1
            else:
                self._pending_box = box
                self._pending_hits = 1
            if self._pending_hits >= n:
                self._pending_box = None
                self._pending_hits = 0
                return True
            self._deferred_frame = True
            return False

        frame = int(getattr(self, "frame_idx", 0) or 0)
        hist = [(f, b) for (f, b) in self._pending_hist if frame - f < window]
        hist.append((frame, box))
        best = 0
        for _f_i, b_i in hist:
            count = 0
            for _f_j, b_j in hist:
                try:
                    if float(iou_matrix(b_i[None], b_j[None])[0, 0]) >= thr:
                        count += 1
                except Exception:  # pragma: no cover - defensive
                    continue
            best = max(best, count)
        self._pending_box = box
        self._pending_hits = best
        if best >= n:
            self._pending_hist = []
            self._pending_box = None
            self._pending_hits = 0
            return True
        self._pending_hist = hist
        self._deferred_frame = True
        return False

    # ------------------------------------------------------------------ #
    # E099 -- pre-veto re-attribution
    # ------------------------------------------------------------------ #
    def _preveto_record(self, chosen: Detection | None) -> None:
        """Retain this frame's pre-veto pool; a no-op unless ``preveto_window > 0``.

        The pool is the survivor set of the encoding stage -- the candidates the
        identity ceiling/band/vote have *not* scored away yet.  E097 measured that
        this set contains the true target far more often than the committed
        identity (65.84 % vs 22.02 % of visible frames on five informative
        sequences), which is why re-attribution looks here instead of at the
        post-veto set E095 used.
        """
        pending = self._preveto_pending
        self._preveto_pending = None
        window = int(getattr(self.cfg.state, "preveto_window", 0) or 0)
        if not self._preveto_on or window <= 0 or pending is None:
            return
        dets, app = pending
        n = len(dets)
        if n == 0 or app.size != n:
            return
        boxes = np.asarray([np.asarray(d.xyxy, float).ravel()[:4] for d in dets], float)
        shipped = -1
        if chosen is not None:
            cb = np.asarray(chosen.xyxy, float).ravel()[:4]
            ious = np.asarray([_iou1(cb, b) for b in boxes], float)
            if ious.size:
                j = int(np.argmax(ious))
                if float(ious[j]) >= 0.9:
                    shipped = j
        self._preveto_hist[int(self.frame_idx)] = {
            "dets": list(dets),
            "boxes": boxes,
            "app": np.asarray(app, float).copy(),
            "shipped": int(shipped),
        }
        cutoff = int(self.frame_idx) - window + 1
        for key in [k for k in self._preveto_hist if k < cutoff]:
            self._preveto_hist.pop(key, None)

    def _preveto_pick(self) -> Detection | None:
        """Run the E097 window DP over the retained pool; return the last-frame pick.

        Emission: ``w_app * appearance_cost + w_switch * [not the shipped choice]``;
        transition: ``w_mot * min(centre displacement / box scale, 5)``.  The DP is
        solved forwards only -- the answer we need is the best chain's *current*
        candidate, so no backtracking is required.
        """
        frames = sorted(self._preveto_hist)
        if len(frames) < 2 or frames[-1] != int(self.frame_idx):
            return None
        st = self.cfg.state
        w_app = float(getattr(st, "preveto_app_weight", 2.0))
        w_mot = float(getattr(st, "preveto_motion_weight", 1.0))
        w_sw = float(getattr(st, "preveto_switch_penalty", 0.25))
        first = frames[0]
        e0 = self._preveto_hist[first]
        idx0 = np.arange(e0["app"].size)
        prev_cost = w_app * e0["app"] + np.where(idx0 == e0["shipped"], 0.0, w_sw)
        for f_prev, f in zip(frames, frames[1:]):
            prev_e = self._preveto_hist[f_prev]
            cur = self._preveto_hist[f]
            pb, cb = prev_e["boxes"], cur["boxes"]
            if pb.shape[0] == 0 or cb.shape[0] == 0:
                return None
            pc = (pb[:, :2] + pb[:, 2:]) / 2.0
            cc = (cb[:, :2] + cb[:, 2:]) / 2.0
            scale = np.maximum(
                np.sqrt(np.maximum((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]), 1.0)), 8.0
            )
            d = np.linalg.norm(cc[None, :, :] - pc[:, None, :], axis=2) / scale[:, None]
            total = prev_cost[:, None] + w_mot * np.minimum(d, 5.0)
            arg = np.argmin(total, axis=0)
            n = cb.shape[0]
            cur_idx = np.arange(n)
            prev_cost = (
                total[arg, cur_idx]
                + w_app * cur["app"]
                + np.where(cur_idx == cur["shipped"], 0.0, w_sw)
            )
        best = int(np.argmin(prev_cost))
        dets = self._preveto_hist[frames[-1]]["dets"]
        return dets[best] if best < len(dets) else None

    def _preveto_try_reattribute(
        self, frame_bgr: np.ndarray | None, result: MatchResult, h: int, w: int
    ) -> None:
        """Accept the pool's current pick, but only as an outright person replacement."""
        det = self._preveto_pick()
        if det is None:
            return
        committed = self._box
        if committed is not None and _iou1(committed, det.xyxy) >= 0.3:
            # The pool agrees with what we already report (or a near variant of it):
            # identity stays put.  Only a *different person* may be adopted.
            return
        self._preveto_switches += 1
        shim = MatchResult(
            index=None,
            cost=float(result.cost),
            accepted=True,
            reason="preveto:reattributed",
        )
        self._on_match(frame_bgr, det, shim, h, w)
        self._check_drift(det)
        self._last_diag["preveto_switched"] = 1
        self._last_diag["preveto_window_used"] = len(self._preveto_hist)

    def _on_match(
        self,
        frame_bgr: np.ndarray | None,
        det: Detection,
        result: MatchResult,
        h: int,
        w: int,
    ) -> None:
        prev = self.state
        self.motion.update(det.xyxy, det.score)
        # Remember the speed at the last real observation: it is the only honest
        # velocity estimate we have once the target is occluded.
        try:
            self._v0_speed = float(self.motion.speed())
        except Exception:  # pragma: no cover - defensive
            pass
        self.frames_since_update = 0
        self.hits += 1
        self._hold += 1
        self._box = np.asarray(det.xyxy, np.float32)
        self._score = float(det.score)
        self._rolled_back = False
        # E071: this crop is now the person we report, so its descriptor becomes
        # the reference the consensus gate compares the next frame against. Only
        # a usable vector replaces it -- an unreadable crop (encoder failure,
        # E067) must not erase the only evidence we have about the incumbent.
        if self._challenge is not None:
            feat = getattr(det, "feat", None)
            if feat is not None:
                try:
                    vec = np.asarray(feat, np.float32).reshape(-1)
                    n = float(np.linalg.norm(vec))
                    if vec.size and np.all(np.isfinite(vec)) and n > 1e-12:
                        self._consensus_feat = vec / n
                except Exception:  # pragma: no cover - defensive
                    pass

        # Absorb appearance only when memory is not frozen. A match during a
        # blind state is a *re-acquisition*: the crop may be the right person
        # seen from a new angle (valuable) but it may also be the start of a
        # latch, so it enters with reduced confidence and the gallery only
        # unfreezes once the track sticks (REACQUIRING -> TRACKING).
        # E025: adapt the memory time-scale to how fast the target's own
        # appearance is changing.  E024 showed a fast half-life is worth +5.5 pp
        # on one sequence and -8 pp on another, so the value has to be inferred,
        # not chosen.  Distance to the gallery *before* absorbing this crop is the
        # estimate: high = the look is changing, so old crops are stale.
        try:
            self._update_half_life(det)
        except Exception as exc:  # pragma: no cover - defensive
            # Deliberately visible: E026 first measured "inert" only because a
            # NameError here was swallowed by a bare `except: pass`.  A diagnostic
            # key is cheap; a silently disabled experiment is not.
            self._last_diag["half_life_error"] = f"{type(exc).__name__}: {exc}"
        if not self.memory_frozen:
            # E067: hand over the accepted detection so its already-computed
            # features are reused instead of re-encoding the same box.
            self._absorb(
                frame_bgr, det.xyxy, det.score, self._crop_quality(det), force=False, det=det
            )

        if prev in BLIND_STATES:
            self._reacquired = True
            self.state = TargetState.REACQUIRING
            self._emit(
                "reacquire",
                after=int(self.frames_since_update),
                cost=float(result.cost),
                margin=None if not np.isfinite(result.margin) else float(result.margin),
            )
            if self._was_out_of_frame:
                self._emit("reentered")
                self._was_out_of_frame = False
        elif self.state is TargetState.REACQUIRING:
            if self._hold >= max(int(self.cfg.state.reacquire_hold), 1):
                self.state = TargetState.TRACKING
                self._emit("confirm")
        else:
            self.state = TargetState.TRACKING

        # A confirmed track is the only moment we can safely label a bystander
        # as a distractor: we know which box is the target, so any *other*
        # detection that looks nearly identical is a look-alike to avoid.
        if self.state is TargetState.TRACKING:
            self._memorize_lookalikes(det)

    def _motion_limit(self, cfg: Any) -> float | None:
        """Adaptive Mahalanobis ceiling from the track's own motion history (E032)."""
        if not bool(getattr(cfg, "motion_gate_adaptive", False)):
            return None
        hist = self._maha_hist
        if len(hist) < 5:
            return None
        arr = np.asarray(hist, np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        k = float(getattr(cfg, "motion_gate_k", 3.0))
        lo = float(getattr(cfg, "motion_gate_floor", 4.0))
        hi = getattr(cfg, "motion_gate_ceil", None)
        limit = max(lo, med + k * mad)
        if hi is not None:
            limit = min(limit, float(hi))
        return float(min(limit, float(getattr(cfg, "motion_gate", 9.4877))))

    def _in_degenerate_view(self, cfg: Any) -> bool:
        """Is the tracked box's SHAPE inconsistent with the target's own reference shape?

        E031 first gated the partial-view fusion on ``estimate_partial_state(...)``'s
        label, and that regressed DanceTrack badly (0005 0.675 -> 0.180) because the
        label fires on ordinary occlusion there.  The discriminating fact is not
        "something is occluded" but "the box we are measuring no longer has the shape
        this person had at the click": a half-visible body measures twice as wide as
        it is tall relative to its own reference aspect, whereas a person merely
        walking behind someone does not change shape.  So the gate is the log-ratio
        between the current box aspect and the reference aspect captured at
        initialization.
        """
        ref = getattr(self.motion, "_ref_state", None)
        if ref is None or self._box is None:
            return self._last_diag.get("partial") not in (None, "full")
        box = np.asarray(self._box, np.float64)
        h_box = max(1e-3, float(box[3] - box[1]))
        a_now = max(1e-3, float(box[2] - box[0]) / h_box)
        a_ref = max(1e-3, float(np.abs(ref[2])))
        return abs(np.log(a_now / a_ref)) > float(getattr(cfg, "partial_fusion_shape_dev", 0.45))

    def _target_trail_descriptor(self, cfg: Any) -> np.ndarray | None:
        """E059: descriptor of the target's own recent motion, from the filter history."""
        if not self.motion.started:
            return None
        hist = self.motion.history()
        if not hist:
            return None
        steps = int(getattr(cfg, "trail_steps", 8) or 8)
        boxes = [np.asarray(b, dtype=np.float64).ravel()[:4] for b in list(hist)[-(steps + 1):]]
        pts = box_trail(boxes)
        if pts is None:
            return None
        return trail_descriptor(pts, steps=steps)

    def _lost_scaled_limit(self, cfg: Any, limit: float | None) -> float | None:
        """E089: widen (or drop) the anchor ceiling while the target is lost.

        The ceiling is applied as a veto over candidates that already carry a
        measurement, and it exists to prevent an identity switch while the tracker is
        watching the target.  E088's audit showed that on the frames where the shipped
        tracker misses a *visible* target, 41 % are refused by this gate while the
        tracker is already in ``lost_short`` / ``lost_long`` -- there is no identity left
        to protect, so the veto buys nothing and costs a recoverable reacquisition.

        ``cost.anchor_band_lost_scale`` is the knob, shipped at 1.25 (E089): 1.0 is the
        pre-E089 behaviour and returns the limit untouched (bit-identical control), > 1
        multiplies the band, and a value <= 0 drops the ceiling entirely (returns
        ``None``) once the target is lost.
        """
        if limit is None:
            return None
        # Accept either the whole Config or just its cost section: `_associate` hands in
        # `cfg.cost`, while tests and future callers may hold the Config itself.
        cost_cfg = getattr(cfg, "cost", cfg)
        try:
            scale = float(getattr(cost_cfg, "anchor_band_lost_scale", 1.0))
        except (TypeError, ValueError):
            return limit
        if scale == 1.0:
            return limit
        # `lost_short_after` lives in the state section, not in cost (cfg here is CostCfg).
        state_cfg = getattr(getattr(self, "cfg", None), "state", None)
        after = int(getattr(state_cfg, "lost_short_after", 12))
        if int(getattr(self, "frames_since_update", 0)) <= after:
            return limit
        self._last_diag["anchor_band_lost_scale"] = scale
        if scale <= 0.0:
            return None
        return float(limit) * scale

    def _lost_geom_limit(self, cfg: Any, limit: float | None, b: Any) -> Any:
        """E091: widen the lost-regime ceiling further, but only near the prediction.

        Returns ``limit`` (a scalar) untouched unless ``cost.anchor_band_lost_geom_scale
        > 1`` and the target is lost; in that case returns a per-candidate array whose
        entries are ``limit * geom_scale`` for candidates with ``norm_centre <= tau``
        and ``limit`` for the rest.  ``norm_centre = |candidate centre - prediction| /
        gate_radius``, so ``tau = 1.0`` means "inside the motion gate" -- exactly the
        candidates the shipped veto refuses on appearance grounds although the motion
        model considers them plausible.

        Bit-identical to the shipped path when the knob is at its default 1.0 (the
        caller then compares against the scalar and nothing else changes).
        """
        if limit is None:
            return None
        cost_cfg = getattr(cfg, "cost", cfg)
        try:
            gs = float(getattr(cost_cfg, "anchor_band_lost_geom_scale", 1.0))
        except (TypeError, ValueError):
            return limit
        if not (gs > 1.0):
            return limit
        state_cfg = getattr(getattr(self, "cfg", None), "state", None)
        after = int(getattr(state_cfg, "lost_short_after", 12))
        if int(getattr(self, "frames_since_update", 0)) <= after:
            return limit
        nc = getattr(b, "norm_centre", None)
        if nc is None:
            return limit
        nc_arr = np.asarray(nc, np.float64).ravel()
        if nc_arr.size == 0:
            return limit
        try:
            tau = float(getattr(cost_cfg, "anchor_band_lost_geom_tau", 1.0))
        except (TypeError, ValueError):
            tau = 1.0
        near = np.isfinite(nc_arr) & (nc_arr <= tau)
        if not bool(near.any()):
            return limit
        self._last_diag["anchor_band_lost_geom_scale"] = gs
        self._last_diag["anchor_band_lost_geom_n"] = int(near.sum())
        return np.where(near, float(limit) * gs, float(limit))

    def _anchor_limit(self, cfg: Any) -> float | None:
        """Distance ceiling against the frozen click crop (E027 fixed, E029 adaptive).

        Returns ``None`` when no ceiling applies.  In adaptive mode the band is
        ``median + k * 1.4826 * MAD`` of the distances already observed on crops we
        were tracking, clipped to ``[floor, ceil]``, so it self-calibrates to how
        much this particular target's appearance moves.
        """
        if self._anchor_feat is None:
            return None
        if not bool(getattr(cfg, "anchor_adaptive", False)):
            limit = getattr(cfg, "anchor_max_dist", None)
            return None if limit is None else float(limit)
        hist = self._anchor_dist_hist
        if len(hist) < 5:
            # Bootstrap: too little evidence to calibrate; fall back to the fixed
            # bound if one is set, otherwise admit everything.
            limit = getattr(cfg, "anchor_max_dist", None)
            return None if limit is None else float(limit)
        arr = np.asarray(hist, np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        k = float(getattr(cfg, "anchor_adaptive_k", 3.0))
        lo = float(getattr(cfg, "anchor_adaptive_floor", 0.35))
        hi = float(getattr(cfg, "anchor_adaptive_ceil", 0.75))
        return float(np.clip(med + k * mad, lo, hi))

    def _update_half_life(self, det: Detection) -> None:
        """Set the gallery recency half-life from the target's own drift rate (E025).

        Inert unless ``memory.ltm_adaptive_half_life`` is on.  The estimate is the
        cosine distance from the crop we just accepted to the nearest entry that
        was already in the gallery, averaged over the last few accepted frames: a
        small distance means the target looks like it did before, so old crops
        still carry signal (slow decay); a large distance means the look is
        changing and the older crops are stale (fast decay).
        """
        mem = self.cfg.memory
        if not bool(getattr(mem, "ltm_adaptive_half_life", False)) or self.ltm is None:
            return
        feat = getattr(det, "feat", None)
        if feat is None or not self.ltm.entries():
            return
        if bool(getattr(mem, "ltm_adaptive_use_anchor", True)):
            # E026: measure against the frozen click crop, not the live gallery --
            # a gallery that absorbs every accepted crop cannot detect drift from
            # itself (E025 was inert for exactly this reason).
            ref = self.ltm.entries()[0].feat
            dist = float(cosine_distance_matrix(np.asarray(feat, np.float32), ref)[0])
        else:
            dist, _entry = self.ltm.best(feat, part="full")
        if not np.isfinite(dist):
            return
        self._app_drift.append(float(dist))
        del self._app_drift[:-5]
        mean = float(np.mean(self._app_drift))
        thr = float(getattr(mem, "ltm_adaptive_threshold", 0.30))
        fast = float(getattr(mem, "ltm_half_life_fast", 150.0))
        slow = float(getattr(mem, "ltm_half_life_slow", 600.0))
        self.ltm.set_half_life(fast if mean > thr else slow)
        self._last_diag["app_drift"] = round(mean, 4)
        self._last_diag["half_life"] = fast if mean > thr else slow

    def _check_drift(self, chosen: Detection) -> None:
        """Force a re-anchor when the followed person stops resembling the click.

        A single-target tracker rarely "loses" the target outright; it follows
        somebody else and every cue then agrees with itself, because the
        short-term memory absorbs the stranger and the motion model follows the
        absorbing body.  The long-term bank is the one memory that cannot be
        absorbed that way: it froze at the click and only grows on accepted,
        quality-filtered crops.  So the signature of a silent identity swap is
        that the accepted crop's distance to the *long-term* bank climbs while
        the tracker still reports TRACKING.

        When it crosses ``memory.drift_threshold`` for ``memory.drift_window``
        consecutive accepted frames, the short-term memory is dropped and the
        tracker is pushed to ``lost_short`` -- it will now search by the original
        click's appearance instead of by the person it drifted onto.  Inert by
        default (``drift_window == 0``).
        """
        mem = self.cfg.memory
        win = int(getattr(mem, "drift_window", 0) or 0)
        thr = getattr(mem, "drift_threshold", None)
        if win <= 0 or thr is None or self.ltm is None or chosen is None:
            return
        feat = getattr(chosen, "feat", None)
        if feat is None or not self.ltm.entries():
            return
        try:
            dist, _entry = self.ltm.best(feat, part="full")
        except Exception:
            return
        if not np.isfinite(dist):
            return
        self._drift_hist.append(float(dist))
        del self._drift_hist[:-win]
        if len(self._drift_hist) < win:
            return
        recent = float(np.mean(self._drift_hist))
        if recent <= float(thr):
            return
        # Confident-and-wrong: forget what we have been looking at and go back
        # to searching for the person that was actually clicked.
        self._emit("drift", ltm_dist=round(recent, 4), threshold=float(thr), window=win)
        self._drift_hist.clear()
        try:
            self.stm.reset()
        except Exception:
            pass
        self.state = TargetState.LOST_SHORT
        self.frames_since_update = max(self.frames_since_update, 1)

    def _on_miss(self, prev: TargetState, result: MatchResult) -> None:
        fsu = self.frames_since_update
        st: StateCfg = self.cfg.state
        self._hold = 0

        # Refusal because of ambiguity is the dangerous case: we saw the target
        # and a look-alike and could not tell them apart. Remember the
        # competitor so it is cheaper to reject next time, and say so.
        # `lone_candidate` belongs to the same class: the only person on the
        # frame did not look like the template, so we may be looking at the
        # moment the target went behind somebody.
        if result.reason.startswith(("ambiguous", "lone_candidate")) and result.index is not None:
            self._remember_competitor(result)
            self._emit("swap_risk", margin=float(result.margin), cost=float(result.cost))

        # FastTracker's occlusion rollback: the first frames of an occlusion are
        # usually the target passing behind something, and the detector may
        # still be firing on the occluder. Undo those updates *once*, before
        # they drag the velocity estimate off course.
        if (
            not self._rolled_back
            and fsu >= max(int(st.soft_occlusion_after), 1)
            and self.motion.started
        ):
            depth = int(getattr(self.cfg.motion, "rollback_depth", 5))
            try:
                self.motion.rollback(depth)
                self._emit("occlusion_rollback", depth=depth)
            except Exception as exc:  # pragma: no cover - defensive
                self._last_diag["rollback_error"] = repr(exc)
            self._rolled_back = True

        if fsu >= int(st.give_up_after):
            if self.state is not TargetState.GAVE_UP:
                self._emit("give_up", fsu=fsu)
            self.state = TargetState.GAVE_UP
        elif fsu > int(st.lost_long_after):
            if prev is not TargetState.LOST_LONG:
                self._emit("lost_long", fsu=fsu)
            self.state = TargetState.LOST_LONG
        elif fsu > int(st.lost_short_after):
            if prev is not TargetState.LOST_SHORT:
                self._emit("lost_short", fsu=fsu)
            self.state = TargetState.LOST_SHORT
        elif fsu >= max(int(st.soft_occlusion_after), 1):
            self.state = TargetState.SOFT_OCCLUSION
        else:
            self.state = TargetState.TRACKING

        # Coast: extrapolate the box so the reported position stays meaningful
        # for a few frames of occlusion, which is what an operator expects.
        if self.motion.started and self.state is not TargetState.GAVE_UP:
            try:
                self._box = np.asarray(self.motion.coast(), np.float32)
            except Exception:  # pragma: no cover - defensive
                pass

    def _update_out_of_frame(self, w: int, h: int) -> None:
        """Detect full departure from the frame and announce it once."""
        if self._box is None:
            return
        m = float(self.cfg.state.out_of_frame_margin)
        x0, y0, x1, y1 = (float(v) for v in self._box)
        gone = x1 < -m or y1 < -m or x0 > w + m or y0 > h + m
        if gone and not self._was_out_of_frame:
            self._was_out_of_frame = True
            self.state = TargetState.OUT_OF_FRAME
            self._emit("out_of_frame", box=self._box.tolist())
        elif not gone and self._was_out_of_frame:
            self._was_out_of_frame = False

    # ------------------------------------------------------------------ #
    # appearance memory
    # ------------------------------------------------------------------ #
    def _crop_quality(self, det: Detection) -> float:
        """Confidence that this crop is a clean, useful view of the target."""
        q = float(getattr(det, "quality", 1.0) or 0.0)
        q *= float(np.clip(getattr(det, "visible_ratio", 1.0) or 0.0, 0.0, 1.0))
        q *= float(np.clip(det.score, 0.0, 1.0))
        return float(np.clip(q, 0.0, 1.0))

    def _absorb(
        self,
        frame_bgr: np.ndarray | None,
        box_xyxy: np.ndarray,
        score: float = 1.0,
        quality: float = 1.0,
        force: bool = False,
        det: Detection | None = None,
    ) -> None:
        """Push one observation into short- and long-term memory.

        ``det`` is the accepted detection when the caller has one: its features
        were already computed by the geometry pass this frame, so passing it
        avoids a second forward of the identical box (E067).
        """
        feats = self._features_for(frame_bgr, box_xyxy, force=force, det=det)
        if not feats:
            return
        full = feats.get("full")
        # E078 -- absorb-ratio guard.  One decision per accepted crop, taken on the
        # full-body descriptor: that is the only space the click anchor lives in, and the
        # same descriptor the E027/E029 veto measures.  Short-term memory and the part EMA
        # are deliberately NOT gated -- they are how a legitimately changed look is still
        # re-acquirable.  Only the long-term gallery, the memory that is supposed to still
        # remember who was clicked, is protected from absorbing a drift.
        absorb_ok = self._absorb_allowed(full, force=force)
        if full is not None:
            self.stm.update(full, quality=quality, frame_idx=self.frame_idx)
            if absorb_ok and (force or quality >= float(self.cfg.memory.ltm_min_quality)):
                self.ltm.add(full, part="full", quality=quality, frame_idx=self.frame_idx)
        for part, f in feats.items():
            if f is None:
                continue
            if part != "full":
                if absorb_ok:
                    self.ltm.add(f, part=part, quality=quality, frame_idx=self.frame_idx)
            self._ema_part(part, f, quality)

    def _absorb_band(self) -> float | None:
        """The ceiling the E078 absorb guard compares an anchor distance against.

        Reuses the E029 self-calibrating band (``_anchor_limit``) so the guard is a ratio
        against something the tracker already calibrates on the target itself, instead of a
        second absolute constant nobody can defend.  ``_anchor_limit`` returns ``None``
        while the band is not observable (no anchor, adaptive band still in bootstrap, or
        no band configured); the guard then falls back to the configured ceiling
        (``anchor_adaptive_ceil`` / ``anchor_max_dist``) so it keeps the same meaning it
        has once the band exists, and admits everything when neither is set at all -- no
        band, no evidence, no veto.
        """
        limit = self._anchor_limit(self.cfg.cost)
        if limit is not None:
            return float(limit)
        cost = self.cfg.cost
        if bool(getattr(cost, "anchor_adaptive", False)):
            ceil = getattr(cost, "anchor_adaptive_ceil", None)
            return None if ceil is None else float(ceil)
        fixed = getattr(cost, "anchor_max_dist", None)
        return None if fixed is None else float(fixed)

    def _absorb_allowed(self, feat: np.ndarray | None, *, force: bool = False) -> bool:
        """E078: may this crop be appended to the long-term gallery?

        Always ``True`` unless ``memory.ltm_absorb_anchor_only`` is on.  With the guard on,
        the click crop itself (``force=True`` from ``initialize``) is always recorded -- it
        IS the anchor -- and any other crop is admitted only while

            cos_dist(crop, anchor) <= ``ltm_absorb_anchor_ratio`` * ``_absorb_band()``

        Hard negatives never come through here (they use ``ltm.add_hard_negative``), so an
        impostor is still remembered as a *negative*: the guard stops the gallery from
        calling him the target, it does not mute the half of the memory that makes a swap
        detectable.
        """
        mem = self.cfg.memory
        if force or not bool(getattr(mem, "ltm_absorb_anchor_only", False)):
            return True
        if feat is None or self._anchor_feat is None:
            return True
        band = self._absorb_band()
        if band is None:
            return True
        ratio = float(getattr(mem, "ltm_absorb_anchor_ratio", 1.25))
        limit = ratio * float(band)
        dist = float(
            cosine_distance_matrix(
                np.asarray(feat, np.float32).reshape(1, -1), self._anchor_feat
            )[0, 0]
        )
        allowed = bool(np.isfinite(dist) and dist <= limit)
        self._absorb_anchor_gated += 1
        if not allowed:
            self._absorb_anchor_blocked += 1
        self._last_diag["absorb_anchor_dist"] = dist
        self._last_diag["absorb_anchor_limit"] = limit
        self._last_diag["absorb_anchor_blocked"] = self._absorb_anchor_blocked
        return allowed

    def _features_for(
        self,
        frame_bgr: np.ndarray | None,
        box_xyxy: np.ndarray,
        force: bool = False,
        det: Detection | None = None,
    ) -> dict[str, np.ndarray]:
        if self.embedder is None or frame_bgr is None:
            return {}
        parts = ("full",) if force is False and not self.ltm.entries else self._query_parts()
        if force:
            parts = tuple(getattr(self.cfg.embedder, "target_parts", ("full",)))
        reused = self._reuse_features(det, box_xyxy, parts)
        if reused is not None:
            return reused
        try:
            out = self.embedder.encode(frame_bgr, np.asarray(box_xyxy, np.float32).reshape(1, 4), parts)
        except Exception as exc:
            self._last_diag["absorb_encode_error"] = repr(exc)
            self._note_encode_error("features_for", exc)
            return {}
        if not out:
            return {}
        return {k: v for k, v in (out[0] or {}).items() if v is not None}

    def _reuse_features(
        self,
        det: Detection | None,
        box_xyxy: np.ndarray,
        parts: tuple[str, ...],
    ) -> dict[str, np.ndarray] | None:
        """Features the geometry pass already computed for this detection (E067).

        Every surviving detection is encoded in `_ensure_features`, and the
        winning one used to be encoded *again* by the appearance pass: same
        pixels, same parts, same encoder, 94.0 ms/frame of pure duplication on
        dancetrack0007. The cache is the identity of the current frame's encoded
        detections (`_encoded_dets`, rewritten by `step()` every frame), so a
        vector from an earlier frame can never be reused by accident. Returns
        None unless *every* requested part is present, in which case the caller
        encodes exactly as before.
        """
        if det is None or not det.part_feats:
            return None
        if not any(d is det for d in self._encoded_dets):
            return None
        det_box = np.asarray(det.xyxy, np.float32).reshape(4)
        if not np.array_equal(det_box, np.asarray(box_xyxy, np.float32).reshape(4)):
            return None
        out: dict[str, np.ndarray] = {}
        for part in parts:
            vec = det.part_feats.get(part)
            if vec is None:
                return None
            out[part] = vec
        return out

    def _ema_part(self, part: str, feat: np.ndarray, quality: float) -> None:
        f = np.asarray(feat, np.float32).reshape(-1)
        n = float(np.linalg.norm(f))
        if not np.isfinite(n) or n <= 1e-8:
            return
        f = f / n
        alpha = float(np.clip(self.cfg.memory.stm_alpha, 0.0, 0.999))
        # Quality modulates how much we trust the new view.
        a = alpha * float(np.clip(quality, 0.0, 1.0))
        prev = self._part_ema.get(part)
        self._part_ema[part] = f if prev is None else a * prev + (1.0 - a) * f
        self._part_ema_quality[part] = float(np.clip(quality, 0.0, 1.0))

    def _hn_iou_ok(self, d: Detection, chosen: Detection) -> bool:
        """E038: keep only spatially distinct look-alikes out of the hard-negative bank."""
        max_iou = getattr(self.cfg.memory, "hard_negative_max_iou", None)
        if max_iou is None:
            return True
        a, b = d.xyxy, chosen.xyxy
        ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
        iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
        inter = ix * iy
        ua = ((float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
              + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter)
        return (inter / ua if ua > 0 else 1.0) <= float(max_iou)

    def _memorize_lookalikes(self, chosen: Detection) -> None:
        """Store near-identical *other* detections as hard negatives."""
        # E037: the bank learns from our own accepted box, so a wrong latch teaches it that the
        # true target is a look-alike.  Restrict admission to confident tracking frames.
        if bool(getattr(self.cfg.memory, "hard_negative_confident_only", False)):
            if self.state is not TargetState.TRACKING:
                return
        target_feat = chosen.feat if chosen.feat is not None else chosen.part_feats.get("full")
        if target_feat is None:
            return
        for d in getattr(self, "_frame_dets", ()):  # populated in step()
            if d is chosen:
                continue
            f = d.feat if d.feat is not None else d.part_feats.get("full")
            if f is None:
                continue
            if bool(getattr(d, "_is_target", False)):
                continue
            dist = 1.0 - float(np.clip(np.dot(target_feat, f), -1.0, 1.0))
            if dist <= CONFUSABLE_DIST and self._hn_iou_ok(d, chosen):
                self.ltm.add_hard_negative(f, frame_idx=self.frame_idx)

    def _remember_competitor(self, result: MatchResult) -> None:
        """Memorize the runner-up that made the decision ambiguous."""
        b = getattr(self, "_last_breakdown", None)
        dets = self._frame_dets
        if b is None or not dets:
            return
        total = np.asarray(b.total, np.float32)
        finite = np.isfinite(total)
        if np.count_nonzero(finite) < 2:
            return
        order = np.argsort(np.where(finite, total, np.inf))
        # order is over the encoded subset; map back through the caller.
        sub = getattr(self, "_encoded_dets", None) or dets
        try:
            runner = sub[int(order[1])]
        except Exception:  # pragma: no cover - defensive
            return
        f = runner.feat if runner.feat is not None else runner.part_feats.get("full")
        if f is not None:
            self.ltm.add_hard_negative(f, frame_idx=self.frame_idx)

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_hw(frame_bgr: Any) -> tuple[int, int]:
        """``(height, width)`` from an image or an explicit ``(h, w)`` tuple."""
        if frame_bgr is None:
            return 0, 0
        if isinstance(frame_bgr, (tuple, list)) and len(frame_bgr) == 2:
            return int(frame_bgr[0]), int(frame_bgr[1])
        shape = getattr(frame_bgr, "shape", None)
        if shape is None or len(shape) < 2:
            return 0, 0
        return int(shape[0]), int(shape[1])


def part_boxes_for(box_xyxy: np.ndarray, frame_size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Convenience wrapper so callers do not import :mod:`parts` directly."""
    return part_boxes(box_xyxy, PART_KEYS, frame_size)
