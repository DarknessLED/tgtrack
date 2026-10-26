"""Central configuration surface.

Every tunable knob lives here so experiments are declarative and reproducible.
Load with `load_config(path)` or override programmatically with `Config(**overrides)`.

The strategy (not the models) is what we optimize, so this file is the
experiment surface: association costs, memory policies, state-machine timings.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: Where model weights live. Kept in-repo on purpose: github.com is unreachable
#: from this machine, and an unresolved model name sends ultralytics into a
#: ~9-minute download-retry loop that ends in a CPU fallback.
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "data" / "weights"


def resolve_weight(name: str) -> str:
    """Return a local path for ``name`` when we already have it on disk.

    Order: an absolute path that exists -> ``data/weights/<basename>`` ->
    ``./<name>`` -> the caller's string unchanged (letting the library try to
    fetch it).
    """
    if not name or name == "auto":
        return name
    p = Path(name)
    if p.is_absolute():
        return str(p) if p.exists() else name
    cand = WEIGHTS_DIR / p.name
    if cand.exists():
        return str(cand)
    here = Path.cwd() / name
    if here.exists():
        return str(here)
    return name


@dataclass
class DetectorCfg:
    #: Default is honestly LARGE for a "nano" project, on measurement.  On this
    #: GPU the detector is not network-bound — preprocessing (letterbox+NMS)
    #: dominates, so n/s/m cost about the same per frame while recall differs
    #: enormously.  Pooled over 5 MOT17 sequences (2,658 visible GT boxes,
    #: visibility >= 0.25, IoU 0.5; `_x/probe_recall.py --confirm`):
    #:     yolo26n@640   recall 0.368   33.5 ms   29.9 FPS
    #:     yolo26m@1280  recall 0.733   42.9 ms   23.3 FPS
    #: i.e. 2.0x the recall for +28% frame time.  Requirement (4) is "detect ALL
    #: persons", and the all-person MOT layer's low DetA (0.25) was this recall
    #: gap, not the tracker.  yolo26n@640 remains available for fast sweeps.
    weights: str = field(default_factory=lambda: resolve_weight("yolo26m.pt"))
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.7
    #: COCO person class index. Set to None to keep all classes.
    person_class: int | None = 0
    device: str = "cuda"
    half: bool = True
    max_det: int = 300
    #: Optional ONNX/TensorRT variant for speed comparisons.
    export_to: str | None = None
    #: E054/E055 — box-level fusion of extra checkpoints. When non-empty, `detect()` runs these
    #: checkpoints on the same frame as well and merges every person box with NMS at
    #: `fusion_iou`. Motivation is measured, not assumed: the general detector anchors tight
    #: boxes but misses occluded people, a CrowdHuman specialist finds them with looser boxes,
    #: and the union recalls **0.830 vs 0.731 / 0.803** (results/e054_union.json, same E003
    #: protocol). Empty by default, so nothing changes unless a caller asks for it.
    extra_weights: tuple[str, ...] = ()
    fusion_iou: float = 0.55
    #: E056 — geometry snapping. For every box the primary detector emits, look for a box from
    #: these checkpoints that overlaps it and, if one does at `snap_iou` or better, ADOPT ITS
    #: GEOMETRY while keeping the primary detector's box count and scores. Motivation is E047 vs
    #: E055: the CrowdHuman specialist brings recall but loose boxes (AssA 0.5431 -> 0.5341), and
    #: adding the general model's boxes on top made things worse (E055 HOTA 0.4944). Snapping aims
    #: for the specialist's recall WITHOUT its loose boxes, and without any new boxes at all.
    snap_weights: tuple[str, ...] = ()
    snap_iou: float = 0.30


#: Bump this whenever a change makes embeddings from an older build
#: INCOMPATIBLE with a gallery built by a newer one.  The shipped ReID weights
#: are a single fixed network, so the embedding space is pinned by
#: (weights file, input resolution, version) — and embeddings of the same person
#: taken at a different `imgsz` are NOT comparable (arXiv 2403.10022).  Any
#: persisted or cached gallery must record `EmbedderCfg.spec` and refuse to be
#: loaded against a different one; mixing them silently corrupts identity matching.
EMBED_SPEC_VERSION = 2


def embed_spec(model: str, imgsz: int) -> str:
    """The identity of an embedding space, for stamping galleries and caches."""
    return f"{Path(model).name}@imgsz{imgsz}@v{EMBED_SPEC_VERSION}"


@dataclass
class EmbedderCfg:
    #: "yolo26n-reid.onnx" | "yolo26s-reid.onnx" | ... | "auto" (detector backbone feats)
    model: str = field(default_factory=lambda: resolve_weight("yolo26n-reid.onnx"))
    #: Input resolution.  The shipped ONNX declares `imgsz=[448,448]` in its
    #: metadata — it was TRAINED at 448 — so feeding it 224 throws away half its
    #: training resolution.  Measured with scripts/exp_partial_view.py on a real
    #: retrieval task (identity Rank-1, 3 identities, 44 queries):
    #:     imgsz 224 -> clean 0.886, occluded 0.655   (0.295 ms/crop batched)
    #:     imgsz 448 -> clean 1.000, occluded 0.780   (1.069 ms/crop batched)
    #: Part-based matching at 448 holds 0.989 occluded Rank-1.  448 costs ~3.6x
    #: batched but only ~1.27x at batch size 1, and the tracker encodes parts only
    #: for the target and a few candidates, so 448 is the right default here.
    imgsz: int = 448
    fp16: bool = True
    #: Which part crops to compute for the TARGET (expensive) vs all persons.
    target_parts: tuple[str, ...] = ("full", "upper", "lower", "head")
    #: Parts computed for every detection, to feed the all-person MOT layer.
    global_parts: tuple[str, ...] = ("full",)
    #: Batch size ceiling for one encoder forward pass.  At imgsz 448 a batch of
    #: 128 needs a single ~275 MB activation buffer, which is fine alone but OOMs
    #: the 8 GB GPU when several worker processes encode at once (observed as
    #: `BFCArena::AllocateRawInternal Failed to allocate ... 269746176` and a dead
    #: pool worker).  64 halves the peak and leaves room for parallel runs, which
    #: this project depends on; part crops are small, so the throughput cost is
    #: marginal.
    max_batch: int = 64

    @property
    def spec(self) -> str:
        """Stamp identifying the embedding space these features live in.

        Store this alongside any persisted gallery and refuse to load a gallery
        whose stamp differs — otherwise features from two different embedding
        spaces get compared as if they were commensurate.
        """
        return embed_spec(self.model, self.imgsz)


@dataclass
class MotCfg:
    """All-person multi-object tracking layer ('others only need an ID')."""

    #: ultralytics tracker yaml basename: bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack
    tracker: str = "botsort.yaml"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    with_reid: bool = True
    gmc_method: str = "sparseOptFlow"
    #: Which ultralytics backend to use for the all-person layer (see R1 report).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionCfg:
    """Motion model / prediction for the selected target."""

    #: "kf" (single Kalman) | "nsa_kf" (noise-scaled) | "imm" (multi-hypothesis)
    model: str = "imm"
    #: State = [cx, cy, a, h, vcx, vcy, va, vh]. Process-noise std weights.
    std_weight_position: float = 1.0 / 20
    std_weight_velocity: float = 1.0 / 160
    #: NSA: scale process noise by detection confidence (1-conf).
    nsa_adaptive: bool = True
    #: IMM Markov transition matrix (diagonal = stay probability).
    imm_stay_prob: float = 0.85
    #: IMM model set: 'cv' (constant velocity), 'ca' (accel), 'ct' (turn).
    imm_models: tuple[str, ...] = ("cv", "ca", "ct")
    #: Velocity decay applied while coasting (1.0 = no damping).
    coast_velocity_damp: float = 0.98
    #: Direction-consistency (OC-SORT OCM) window and weight.
    direction_window: int = 3
    direction_weight: float = 0.2
    #: Kalman rollback depth when occlusion onset is detected (FastTracker idea).
    #: Frames to roll the state back on occlusion onset.  0 = off, and 0 is the
    #: shipped default on measured evidence, not caution.  `scripts/ablate_motion.py`
    #: (26 scenarios x 10 configs = 260 runs) found rollback net NEGATIVE in every
    #: single scenario: imm 93.6% -> 91.1% (rb3) -> 90.3% (rb5) success, kf 92.2% ->
    #: 89.9% -> 89.3%; worst per-scenario cost fast_linear_step20 1.000 -> 0.883.
    #: Mechanism: rollback fires on occlusion ONSET, but during a detector blackout
    #: the target is still really there, so rolling back 3-5 frames discards the most
    #: recent — and best — velocity estimate.  It is a reactivation tool, and the
    #: reactivation benefit did not materialise on any tested scenario.  The mechanism
    #: is kept (motion.rollback) for cases where a track must be restarted, but it is
    #: not part of the default strategy.
    rollback_depth: int = 0

    #: Shape prior (E014).  A person's box aspect/height cannot legitimately
    #: change by an order of magnitude between frames, but the filter's
    #: aspect/height *velocity* terms can integrate a partial view into a
    #: runaway "narrowing" prediction: `results/e014_upper` shows the predicted
    #: width collapsing 159 -> 127 -> 94 -> ~0 px in four frames while the
    #: person's true box width stayed ~170 px; once IoU with the correct
    #: detection is 0 the tracker cannot re-accept until `lost_short`.  Nine
    #: different appearance/gallery variants all land on exactly the same 0.7527
    #: target-success rate under the upper-body mask, which is the tell that the
    #: failure is geometric, not appearance-side.  The prior bounds the PREDICTED
    #: box to a multiple of the last OBSERVED aspect/height (so a genuine
    #: approach/recede still passes, since the reference updates on every
    #: accepted observation) and clamps the aspect velocity.
    #: MEASURED NEGATIVE (E014): default off.  Turning it on moved the
    #: upper-mask target-success rate 0.7527 -> 0.7345 (MOT17-02, 275 visible
    #: frames) and changed nothing on `mask=none` (0.9236 both), i.e. inside the
    #: noise of this measurement.  The collapse it targets is real (see the
    #: trace below) but it is a *ratchet*: each accepted observation narrows the
    #: reference a little, so a relative band around the reference cannot catch
    #: it.  The knob and its tests are kept because they pin the DIVERGENCE
    #: mechanism; the fix that matters is `rescue_app_dist` in `CostCfg`.
    shape_prior: bool = False
    aspect_min_ratio: float = 0.60
    aspect_max_ratio: float = 1.70
    size_min_ratio: float = 0.60
    size_max_ratio: float = 1.90
    #: Max |aspect velocity| per frame (aspect units); 0 = unbounded.
    aspect_vel_max: float = 0.05

    #: --- E016: velocity bound (filter-divergence guard) ---
    #: The E014 trace shows the *centre* velocity running away as well as the
    #: aspect: after one poisoned update the filter predicted the box moving
    #: -40 px/frame for 12 frames while the person was actually walking +8 px
    #: right, i.e. the prediction diverged ~400 px and every correct detection
    #: then failed the motion gate.  `_MAX_STEP` bounds per-frame drift but not
    #: the velocity state itself.  Bound each component's velocity to
    #: `velocity_clamp_ratio` x the recent OBSERVED speed (px/frame), which a
    #: real pedestrian cannot exceed by that factor.  None = off.
    velocity_clamp_ratio: float | None = 3.0
    #: Floor for the observed-speed reference (px/frame) so a nearly static
    #: target does not get a zero-width velocity allowance.
    velocity_clamp_floor: float = 2.0


@dataclass
class MemoryCfg:
    """Short- and long-term appearance memory for the selected target."""

    #: EMA factor for the short-term 'current appearance' (higher = slower).
    stm_alpha: float = 0.9
    #: Ring buffer length for recent embeddings.
    stm_len: int = 64
    #: Frame index over which STM is considered stale (feature re-init trigger).
    stm_stale_frames: int = 90

    #: Long-term gallery capacity.
    ltm_capacity: int = 96
    #: Minimum cosine distance to any existing entry for a new entry to be 'diverse'.
    ltm_diversity_thresh: float = 0.15
    #: Minimum crop quality to allow a gallery insert.
    ltm_min_quality: float = 0.35
    #: Recency half-life in frames for the recency weight (None disables).
    ltm_half_life: float = 600.0
    #: If the gallery is full, replace the entry with the lowest quality*recency.
    ltm_replace_policy: str = "quality_recency"  # | 'fifo'
    #: Part-aware gallery: keep separate banks per part key so partial queries match parts.
    part_aware: bool = True
    #: Hard-negative mining: store embeddings of confirmed distractors.
    hard_negative_bank: bool = True
    #: E041 measured 8 as the shipped value: DanceTrack pooled TSR 0.475 -> 0.512 and FAR
    #: 0.312 -> 0.280 (Pareto), with dancetrack0007 gaining +11.1 pp TSR / -9.5 pp FAR while
    #: 0004 and 0005 stay bit-identical.  cap4 gave the same 0007 gain but cost 0004 12.5 pp,
    #: so the plateau is narrow and this value is measured, not interpolated.
    hard_negative_capacity: int = 8
    #: E037 -- only learn hard negatives from confident TRACKING matches.  The bank is
    #: populated from the tracker's own accepted box, so after a wrong latch the true
    #: target looks exactly like "someone who resembles the person I am following" and
    #: gets written in as a negative, making the latch self-reinforcing (E036).
    hard_negative_confident_only: bool = False
    #: E038 -- refuse to admit the assignment LOSER as a hard negative.  A crowded frame
    #: puts the competing candidate right next to the accepted box, so an IoU ceiling is
    #: the cheap way to keep only spatially distinct look-alikes; None disables the gate.
    #: E038 measured 0.30 as the shipped value: DanceTrack pooled TSR 0.450 -> 0.475 and
    #: FAR 0.328 -> 0.312 (a Pareto move), dancetrack0005 untouched at 0.695, MOT17 upper
    #: view untouched at 0.8545, at the cost of dancetrack0007 (-8.6 pp TSR).
    hard_negative_max_iou: float | None = 0.30
    #: E025 -- adaptive forgetting.  E024 measured that a FAST recency half-life
    #: (150) is worth +5.5 pp TSR and -0.14 FAR on dancetrack0004 while being
    #: catastrophic on dancetrack0007, so a fixed time-scale cannot be right.  With
    #: ``ltm_adaptive_half_life`` on, the tracker measures the accepted crop's
    #: distance to the gallery and shortens the half-life when that distance is
    #: high (the target's appearance is changing fast -> old crops are stale) and
    #: lengthens it when the distance is low.  Change-point detection in spirit,
    #: not another constant.  Off by default.
    ltm_adaptive_half_life: bool = False
    ltm_half_life_fast: float = 150.0
    ltm_half_life_slow: float = 600.0
    ltm_adaptive_threshold: float = 0.30
    #: E026 -- what the drift is measured AGAINST.  E025 measured it against the
    #: live gallery and the result was inert, for a structural reason: the gallery
    #: absorbs every accepted crop, so it chases the appearance and the distance
    #: stays small no matter how fast the look is changing -- the measurement is
    #: self-referential.  With ``ltm_adaptive_use_anchor`` the reference is the
    #: FIRST gallery entry (the crop taken at the click), which is never updated,
    #: so the distance actually reflects "how far have we come from the clicked
    #: appearance".  True = frozen anchor (E026); False = the E025 behaviour.
    ltm_adaptive_use_anchor: bool = True
    #: E023 -- how many of the FIRST gallery admissions are pinned against
    #: eviction.  The first crops are the ones taken at/after the operator's click,
    #: i.e. the only hard evidence of WHO was selected; with a decayed
    #: quality*recency policy a 200-frame run can evict them and the gallery then
    #: drifts entirely onto whoever the tracker has been following (E020: on
    #: dancetrack0007 the memory preferred the stranger).  Borrowed from
    #: continual-learning anchor sets and SLAM reference submaps.  0 = off.
    ltm_anchor_count: int = 0
    #: Drift monitor (E021). When the person we are following stops resembling the
    #: long-term bank that froze at the click, the tracker is not "losing" the
    #: target -- it is confidently following somebody else, the failure a MOT
    #: tracker tolerates and a click-to-track product cannot. With
    #: ``drift_window > 0`` and a finite ``drift_threshold``, the mean appearance
    #: distance to the LTM bank over the last ``drift_window`` accepted frames is
    #: compared against the threshold; exceeding it drops the short-term memory and
    #: forces a ``lost_short`` search anchored on the original click.
    #: Default 0/None = inert (off), so nothing changes until it is measured.
    drift_window: int = 0
    drift_threshold: float | None = None


@dataclass
class CostCfg:
    """Association cost weights and gates for the selected target."""

    #: Combined cost = w_m*motion + w_a*appearance + w_d*direction + w_s*size.
    w_motion: float = 1.0
    w_appearance: float = 1.6
    w_direction: float = 0.3
    w_size: float = 0.2
    #: Cost above this is rejected (target not matched this frame).
    accept_cost: float = 1.0
    #: Looser acceptance while in a lost state (re-acquisition).
    accept_cost_lost: float = 1.25
    #: Mahalanobis gate chi-square-ish threshold.
    motion_gate: float = 9.4877
    #: E032 -- ADAPTIVE MOTION BAND.  The appearance band (E029) cannot catch
    #: dancetrack0007's impostor because it starts out close to the click crop; what it
    #: cannot fake is *arriving from the wrong place*.  The filter already computes the
    #: Mahalanobis distance of every candidate (``CostBreakdown.maha``), but the gate
    #: compares it with a constant.  Keep the history of the Mahalanobis distance of the
    #: crops we actually accepted and refuse anything beyond ``median + k * 1.4826 * MAD``
    #: of that -- the same control-chart construction as E029, applied to the *motion*
    #: axis instead of the appearance axis.  A target that really moves/accelerates
    #: widens its own band; a swap arrives as a spike.
    motion_gate_adaptive: bool = False
    motion_gate_k: float = 3.0
    motion_gate_floor: float = 4.0
    motion_gate_ceil: float | None = None
    motion_gate_window: int = 30
    #: E032b -- the motion veto only fires when the candidate ALSO misses the
    #: appearance band; used as the fallback appearance limit when the adaptive
    #: appearance band is off.
    motion_band_app_floor: float = 0.30
    #: Adaptive spatial gate = base + speed_gain * |v| * dt.
    gate_base: float = 0.10
    gate_speed_gain: float = 2.5
    #: Require the best candidate to beat the runner-up by this margin
    #: (distractor suppression). In cost units.
    distractor_margin: float = 0.06
    #: E022 -- identity guard for the global-search path.  While the tracker is
    #: blind it drops the motion term and selects on appearance, and E020/E021
    #: showed that this is where a stranger gets latched: the short-term memory
    #: has already absorbed the wrong person, so scoring against it *confirms*
    #: the stranger.  With ``search_uses_stm = False`` the search is scored
    #: against the long-term bank only (the click anchor) plus the hard negatives.
    search_uses_stm: bool = True
    #: E022 -- minimum margin over the runner-up demanded while SEARCHING, on top
    #: of ``distractor_margin``.  A crowd offers many candidates, so a genuine
    #: re-acquisition wins by a lot while a stranger wins by a hair.  None = inert.
    #: SHIPPED AT 0.20 (default) on the strength of E022: DanceTrack pooled FAR
    #: 0.258 -> 0.197 (TSR 0.554 -> 0.544, and 0007's own TSR unchanged), while on
    #: the MOT17 observability grid it costs 0.36 pp of upper-body TSR and is free
    #: on the unmasked mask.  Set to None to get the pre-E022 behaviour back.
    search_margin: float | None = 0.20
    #: Stricter margin used while the motion prior is dead (``lost=True``).
    #: 0 disables the override and falls back to ``distractor_margin``.
    #:
    #: Rationale: in these states the gate is switched off, so appearance is the
    #: ONLY evidence left, while ``accept_cost_lost`` simultaneously LOOSENS the
    #: absolute cost bound. Tolerating a worse-looking candidate is right (the
    #: target has aged); tolerating an AMBIGUOUS one is not, because a rebind
    #: writes a new identity into memory. The two knobs therefore move in
    #: opposite directions on purpose. Measured live on MOT17-02 (target id 17,
    #: ``--mask upper``): a wrong identity was rebound after 13 blind frames at
    #: margin 0.0635 against the 0.06 tracking threshold -- clearing it by 0.0035.
    #: Left at 0.0 (inert) until the A/B on an observable target says otherwise.
    distractor_margin_lost: float = 0.0
    #: E071 -- TEMPORAL CONSENSUS ON THE HANDBACK (docs/E071_PLAN.md).
    #:
    #: Every refusal in the E029/E032/E060-E063 line is a *one-frame* test: a
    #: cost bound, a margin over the runner-up, an appearance band.  That is the
    #: wrong instrument for the failure that actually dominates here -- a
    #: one/two-frame tie between two people, decided by whichever descriptor
    #: happened to survive a partial crop on that single frame (E061: TSR 0.451
    #: against the click-anchored subset, and the loss is a *confident* wrong
    #: lock, not a lost target).  A vote over consecutive proposals separates
    #: "this person keeps winning" from "this person won once".
    #:
    #: ``commit_consensus`` is the master switch and defaults to False, so the
    #: shipped behaviour is byte-for-byte unchanged; the other four knobs are
    #: inert while it is off.  ``consensus_frames`` is how many consecutive,
    #: mutually consistent non-incumbent proposals are required before a
    #: challenger may take the identity (0/1 = commit on the first frame, i.e.
    #: the pre-E071 behaviour, available as an explicit control arm).
    #:
    #: ``consensus_iou`` gates agreement between *proposals*, not acceptance --
    #: it is deliberately loose (0.15-0.30) because a person under a partial
    #: crop moves in the box; ``consensus_ident`` is the leg that stops two
    #: people standing in the same place from being read as agreement (cosine
    #: distance between consecutive proposal descriptors).
    #:
    #: ``consensus_pool`` exposes the chain's pooled descriptor (L2-normalised
    #: mean) for the caller -- the "pool before you decide" step from video face
    #: recognition.  It is instrumentation: the shipped commit rule does not
    #: read it, so turning it on cannot change a decision.
    commit_consensus: bool = False
    consensus_frames: int = 2
    consensus_iou: float = 0.30
    consensus_ident: float = 0.35
    consensus_pool: bool = True
    #: Appearance cost charged to a candidate whose crop yielded NO usable
    #: descriptor. This is a HAZARD, not a neutral default.
    #:
    #: A feature-less candidate is charged 0.5 (x w_appearance 1.6 = 0.8 of the
    #: 1.0 accept budget), while a genuinely DIFFERENT person at cosine 0.0 is
    #: charged 1.0 (x 1.6 = 1.6). So "we could not read this person" scores
    #: BETTER than "this is a different person", and a clean-descriptor bystander
    #: can beat the true target the moment the target's own crop degenerates.
    #: Measured by motion-dev on the `unidentifiable_target` family: with a
    #: bystander at appearance similarity 0.40-0.85 the tracker hands the
    #: identity over on the FIRST frame the target is unreadable, at a median
    #: accepted margin of 0.88-1.11 -- 15x `distractor_margin`, so no margin
    #: threshold can refuse it. Levers are a larger value here (treat absent
    #: evidence as no evidence) or requiring consecutive confirmation before a
    #: handover. Left at 0.5 = the measured baseline until that A/B is done.
    missing_appearance_cost: float = 0.5
    #: Absolute cost bound for accepting a LONE candidate, i.e. one where no
    #: finite runner-up exists. ``None`` disables the rule (current behaviour).
    #:
    #: The margin test cannot protect a lone candidate: with a single finite
    #: entry the runner-up margin stays ``inf``, so ``margin < distractor_margin``
    #: is vacuously false and ANY threshold accepts it. That is not a corner case
    #: -- it is exactly how the identity is handed over. motion-dev's frame trace
    #: of `unidentifiable_missing_s060` (bystander at cosine 0.60 sitting on the
    #: stale prediction): at frame 18 the target stops being detected, `n_dets`
    #: drops 2 -> 1, and the lone bystander is accepted at best_cost 0.90 with no
    #: runner-up. Meanwhile legitimate lone-candidate accepts in the no-bystander
    #: control cost 0.02. A bound near 0.4 therefore separates "the only person
    #: here looks nothing like the template" from "the only person here is the
    #: target", with ~2x headroom on both sides.
    #:
    #: Applied whenever exactly one finite candidate exists — in BOTH directions
    #: of the state machine, and only when that candidate's appearance was
    #: actually readable (a placeholder cost for unreadable evidence must not be
    #: treated as evidence). The lost path used to be exempt on the theory that
    #: re-acquisition legitimately costs more; that exemption is what had reduced
    #: the guard to a 2-frame deferral (handover 18 -> 20, false acquisition
    #: 76.9% -> 75.6%), because after `soft_occlusion_after` frames the state goes
    #: blind and the lost path re-accepted the same lone bystander with no bound at
    #: all (`accept_cost_lost` 1.25 vs the bystander's 0.83).
    #:
    #: With the exemption removed the guard finally does what it was built for. On
    #: the unidentifiable family (`results/motion_ablation/lone_bound.md`, 100 runs,
    #: 7,800 frames, 0 errors) the lone-candidate hijack count goes 3/4 runs -> 1/4
    #: at 0.5 and -> 0/4 at <= 0.4, and every surviving handover is a frame-36
    #: PRICED comparison where two candidates are genuinely ranked against each other.
    #:
    #: It is nevertheless shipped OFF, on REAL-DATA evidence. `results/target_obs19_shipped`
    #: vs `results/target_obs19` (identical command, MOT17-02-FRCNN, target id 19,
    #: 300 frames, subsample 2, masks none/upper/lower): the bound costs **3.6 pp of
    #: target success on the upper-body-only mask (0.7527 -> 0.7164)** and 0.0 pp on
    #: the none and lower masks, while false acquisition does not move at all
    #: (0.00667 / 0.0 / 0.01333 both ways). The synthetic benefit is real but the
    #: real-data benefit is zero and the real-data cost is not, so it does not ship
    #: enabled. Set to a float to enable it.
    #: E027 -- identity ceiling against the FROZEN click crop.  The long-term
    #: gallery absorbs every accepted crop, so after a wrong latch the impostor
    #: matches "itself" better than the target does (E020: 129 of 141 wrong frames
    #: accepted a different person whose appearance cost was 0.07-0.17, better than
    #: the true target's 0.22-0.30).  A gallery cannot detect that drift from the
    #: inside; the one descriptor that cannot be poisoned is the crop captured at
    #: the click.  With ``anchor_max_dist`` set, a candidate whose cosine distance
    #: to that frozen crop exceeds it is not admissible at all.  None = off.
    #: E028 -- apply the anchor ceiling only while SEARCHING for the target.
    #: E027 measured that the ceiling is a large win where the failure is a wrong
    #: latch (dancetrack0005: TSR 0.18 -> 0.705, FAR 0.81 -> 0.095, id_consistency
    #: 0.18 -> 0.88) and a large loss where the target's own look legitimately
    #: drifts (dancetrack0004: 0.235 -> 0.105, which is exactly the sequence E024
    #: showed needs FAST forgetting, i.e. its appearance really does change).  Wrong
    #: latches are born in the search path (E022/E026), and a continuously tracked
    #: target is continuously corroborated, so the veto belongs on the search path
    #: only.  True = search only; False = E027 behaviour (the default, so the
    #: E027 grid stays comparable: a knob may not change the meaning of the
    #: variants already measured against it).
    anchor_ceiling_search_only: bool = False
    #: E029 -- ADAPTIVE ceiling.  E027/E028 showed a fixed anchor distance cannot be
    #: right for two sequences at once: dancetrack0005 needs it tight (the impostor
    #: is far from the click crop, the true target is near) while dancetrack0004
    #: needs it loose (its target's look genuinely drifts, E024).  The fix is not a
    #: better constant but a BAND calibrated on the target itself: keep the history
    #: of "distance from the click crop to the crops we have been confidently
    #: tracking" and veto anything beyond `median + k * 1.4826 * MAD` (a robust
    #: z-score, the same construction statistical process control uses for a
    #: control chart).  The band therefore widens by itself on a target whose look
    #: drifts, and stays tight on a stable one, with no per-sequence tuning.
    #: SHIPPED (E029/E030): strict Pareto gain -- DanceTrack pooled TSR 0.272 -> 0.437
    #: and FAR 0.467 -> 0.265 with every sequence non-regressing, MOT17 upper-view TSR
    #: 0.7782 -> 0.7818 and `none` unchanged.  Tables recorded before E029 were measured
    #: with this OFF.
    anchor_adaptive: bool = True
    anchor_adaptive_k: float = 4.0
    anchor_adaptive_floor: float = 0.35
    anchor_adaptive_ceil: float = 0.75
    anchor_adaptive_window: int = 30
    #: E058 — ANCHOR BOOTSTRAP.  Everything in E027-E030 hangs off one crop: the single frame
    #: at the click.  A single crop is a noisy estimate of a person's appearance (pose, motion
    #: blur, which half is visible), and E029's whole ceiling is calibrated against it — so the
    #: quality of that one crop propagates into every veto the tracker makes.  This knob re-
    #: estimates the anchor from the first N frames in which the target was CONFIDENTLY matched,
    #: as the L2-normalised mean of the click crop plus those crops (averaging zero-mean noise
    #: while keeping the identity direction).  It is a mechanism, not a parameter to be tuned:
    #: 0 = off (the historical single-crop behaviour) and the experiment asks whether averaging
    #: K accepted observations beats one click.
    anchor_bootstrap_frames: int = 0
    #: E031 -- reliability-weighted fusion for PARTIAL VIEWS.  E014's forensics showed
    #: the upper-mask loss is geometric: when only the top half is visible the
    #: observed box is half as tall and its centre sits h/4 above the true centre, so
    #: the constant-velocity filter integrates a biased measurement and the prediction
    #: diverges -- the same failure the E016 velocity clamp only partly fixed.  The
    #: ingredient the tracker already computes but never uses is that it KNOWS the view
    #: is partial (`_last_diag["partial"]`).  With ``partial_motion_weight`` set, a
    #: partial view down-weights the motion term and up-weights appearance: geometry
    #: measured from a truncated box deserves less trust than the identity evidence.
    #: None = off (pre-E031 behaviour).
    #: SHIPPED (E031b): MOT17 upper-body TSR 0.7818 -> 0.8545 and `none` 0.9200 ->
    #: 0.9273, i.e. the partial-view criterion is met (+10.2 pp over the 0.7527
    #: starting point).  DanceTrack pays a little: pooled TSR 0.437 -> 0.450 but
    #: FAR 0.265 -> 0.328, all of it on dancetrack0007.  Taken because the
    #: partial-view target is a hard criterion and the identity FAR criterion was
    #: unmet either way.
    partial_motion_weight: float | None = 0.5
    partial_appearance_weight: float | None = 2.0
    #: E031b -- how far the box aspect must deviate from the target's own reference
    #: aspect (log ratio) before the partial-view fusion is armed.  The first E031
    #: gate used `estimate_partial_state`'s label and regressed DanceTrack instead
    #: (ordinary occlusion triggered it); shape deviation from the reference does not.
    partial_fusion_shape_dev: float = 0.45
    anchor_max_dist: float | None = None
    lone_candidate_accept_cost: float | None = None
    #: Appearance cost is min over the top-k gallery entries' distances.
    app_topk: int = 5
    #: Blend STM EMA and LTM gallery appearance costs.
    stm_weight: float = 0.6
    ltm_weight: float = 0.4
    #: Penalty added to a candidate whose appearance matches a hard negative.
    hard_negative_penalty: float = 0.35
    #: Use part-aware matching when the query crop is partial.
    part_matching: bool = True
    #: Minimum visible fraction of the full-body box to trust a full-body embedding.
    partial_full_body_thresh: float = 0.55


    #: E059: weight of the motion-signature ("trail") term added to the total cost
    #: while the tracker is SEARCHING.  The descriptor is the candidate's own
    #: turning-angle / log-speed history over ``trail_steps`` frames, compared with
    #: the target's history at the moment it was lost.  It exists because on
    #: DanceTrack appearance cannot tell dancers apart (pairwise AUC 0.43-0.58,
    #: probe_embed_metric.py) and motion/gait is what the literature falls back on
    #: (GAE-Net 2025; gait+HAR multi-year ReID 2025).  ``0.0`` = off (shipped).
    trail_weight: float = 0.0
    #: Number of displacement steps in a trail descriptor.
    trail_steps: int = 8
    #: How many frames of history each local tube keeps.
    trail_len: int = 16
    #: E061: weight of the TUBE-VELOCITY consistency term.  Unlike ``trail_weight``
    #: (an 8-step turning/speed signature, E059, refuted) this uses only the
    #: candidate's own short-tube velocity direction and compares it with the
    #: target's Kalman velocity.  It exists because the DanceTrack swap happens in a
    #: one-or-two-frame tie-break where two detections overlap the prediction almost
    #: equally (iou_pred 0.92 vs 0.68) and appearance/size cannot separate them -- but
    #: the two people are moving in different directions.  ``0.0`` = off (shipped).
    tube_velocity_weight: float = 0.0
    #: Minimum tube speed (in body heights per frame) before the direction is trusted.
    tube_velocity_min_speed: float = 0.004
    #: E063 in-track rival audit.  E062 measured that on the sequence we lose the
    #: true target is DETECTED in 108 of 113 wrong frames (median IoU 0.93) while the
    #: tracker stays in TRACKING on somebody else, and the appearance gallery cannot
    #: catch it because it absorbed the impostor.  Every ``audit_interval`` frames
    #: (0 = off) the frozen click anchor is asked whether some OTHER detection is
    #: ``audit_margin`` closer to it than the box we follow; a switch needs
    #: ``audit_persist`` consecutive, spatially consistent answers.
    audit_interval: int = 0
    audit_margin: float = 0.05
    audit_persist: int = 2
    #: E064 -- require TWO references to agree before the audit switches.
    #: E063's single reference (the click crop) won dancetrack0007 (+20.3 pp TSR) but
    #: damaged dancetrack0005 (0.4000 -> 0.2950), which is exactly the sequence E024
    #: showed needs fast memory forgetting because its target's look genuinely
    #: changes: against a frozen crop "whoever resembles the click" stops being the
    #: target, so a CORRECT track gets vetoed repeatedly.  With this on, the rival
    #: must also beat the followed box against ``stm.current()`` (the EMA of the
    #: recent confident crops) by ``audit_margin``; the two references then have to
    #: agree.  Inert when the recent reference is absent (a fresh track).
    audit_use_recent: bool = True


@dataclass
class StateCfg:
    """State-machine timings, in frames."""

    #: Consecutive accepted frames required to confirm a new/re-acquired target.
    confirm_frames: int = 3
    #: Frames of marginal/no match before TRACKING -> OCCLUDED_SOFT.
    soft_occlusion_after: int = 2
    #: Frames before OCCLUDED_SOFT -> LOST_SHORT.
    lost_short_after: int = 12
    #: Frames before LOST_SHORT -> LOST_LONG (global re-detection kicks in).
    lost_long_after: int = 60
    #: Frames in LOST_LONG before reporting permanent loss.
    give_up_after: int = 3000
    #: Predicted-box-outside-frame tolerance (px) before declaring OUT_OF_FRAME.
    out_of_frame_margin: float = 8.0
    #: While lost, an unconfirmed re-acquisition candidate must persist this many
    #: frames before being confirmed (anti-distractor latch protection).
    reacquire_hold: int = 2
    #: Freeze the appearance memory while in these states.
    frozen_states: tuple[str, ...] = ("occluded_soft", "lost_short", "lost_long", "out_of_frame")
    #: States in which the prediction is no longer trustworthy, so the motion
    #: gate is switched off and a re-acquisition must be justified by appearance
    #: alone.  ``lost_short`` is included deliberately: by 12 frames without an
    #: update the motion prior has already been rolled back once and is known to
    #: be unreliable, and waiting for ``lost_long`` (60 frames, ~2 s at 30 fps)
    #: made occlusion recovery latency exactly ``lost_long_after - gap + 1``.
    #: Turning the gate off is not a loss of safety: ``_associate`` still
    #: refuses any candidate without appearance evidence.
    global_states: tuple[str, ...] = ("lost_short", "lost_long", "gave_up", "out_of_frame")
    #: When growing the gate during a blind state, use the speed measured at the
    #: moment the target was last seen instead of the coast-damped current
    #: speed.  The coast halves the velocity within ~35 frames, so a growth term
    #: driven by it saturates exactly when a distant reappearance must be
    #: reached.
    gate_growth_freeze_speed: bool = True


@dataclass
class EvalCfg:
    iou_thresh: float = 0.5
    #: TrackEval-style metrics for the all-person layer.
    mot_metrics: tuple[str, ...] = ("HOTA", "MOTA", "IDF1", "AssA", "DetA")
    #: Target-centric metrics.
    target_metrics: tuple[str, ...] = (
        "target_success_rate",
        "target_precision",
        "loss_rate",
        "reacquire_rate",
        "reacquire_latency",
        "false_acquisition_rate",
        "id_consistency",
    )
    #: Frames to allow for re-acquisition latency accounting.
    reacquire_window: int = 150


@dataclass
class Config:
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    embedder: EmbedderCfg = field(default_factory=EmbedderCfg)
    mot: MotCfg = field(default_factory=MotCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    state: StateCfg = field(default_factory=StateCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    #: Free-form experiment metadata.
    name: str = "default"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples -> lists for JSON friendliness
        return json.loads(json.dumps(d, default=list))

    def clone(self, **overrides: Any) -> Config:
        """Deep-copy with dotted-path overrides, e.g. `clone(**{"cost.w_motion": 2.0})`."""
        new = copy.deepcopy(self)
        for key, value in overrides.items():
            parts = key.split(".")
            obj: Any = new
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise AttributeError(f"unknown config key: {key}")
            setattr(obj, parts[-1], value)
        return new


def _apply_dict(obj: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"{type(obj).__name__} has no field {key!r}")
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML/JSON config, then apply dotted-path `overrides`."""
    cfg = Config()
    if path is not None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            # YAML is an optional dependency of this module (JSON works without it), so the
            # import has to happen here rather than at module scope -- it was missing entirely,
            # which made every YAML config raise NameError: name 'yaml' is not defined.
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if data:
            _apply_dict(cfg, data)
    if overrides:
        cfg = cfg.clone(**overrides)
    return cfg
