#!/usr/bin/env python
"""Clicked-target end-to-end harness + strategy ablation grid (requirement 5).

What it does
------------
For every MOT sequence under ``--data-root``:

  1. picks one GT identity programmatically (default strategy ``stressed``: the
     identity that is longest-lived *and* interrupted/clipped, so the run really
     exercises loss + re-acquisition instead of a clean walk),
  2. emulates the operator click with ``tgtrack.simulate_click`` on the first
     frame that identity is visible,
  3. builds a fresh ``TargetTracker`` (real ReID embedder unless ``--no-embed``)
     and steps it over every frame, feeding detections from the real YOLO26
     person detector (or ``--detector gt`` for a controlled oracle run),
  4. scores the run with ``tgtrack.evaluate.evaluate_target``.

Every strategy switch in the grid is a real ``Config`` field flipped through
``Config.clone(**dotted.overrides)``, so a run is reproducible from
(sequence, variant, mask) alone.  A fresh tracker and a fresh mask plan per task
means nothing can leak between runs.

Two synthetic partial-view stresses (implemented here, never in the loader):

  ``--mask lower|upper|left|right``
      paint a mask over half of the target's GT box, so only the other half of
      the body is visible (lower -> upper body only).
  ``--mask edge``
      translate the frame so the target hangs off the border of the image
      ("walking out of frame") instead of being painted over.

Both rewrite the ground truth *consistently*: the target's GT box becomes its
**visible** rectangle (the visible fraction is recorded in ``visibility``), other
people keep their boxes with a reduced ``visibility``, and anyone hidden by more
than ``--drop-below`` is removed from the frame's GT.  This is what keeps
``target_success_rate`` meaningful under partial view -- a tracker that keeps its
box on the visible half is correct, and does not get punished for the half it
cannot see.  ``target_success_rate_fullgt`` is reported alongside it, scored
against the untouched full-body box, so both readings are available.

Notes
-----
* Console encoding on this machine is GBK: every message printed is ASCII-only.
* Parallelism: ``--jobs N`` runs (sequence, variant, mask) triples in a
  spawn-based process pool.  Measure the scaling with ``time`` once the real
  datasets are extracted; the pool is correct by construction (pure functions,
  no shared state) rather than tuned.
* ``evaluate_target`` exposes ``reacquire_latency`` (mean),
  ``reacquire_latency_p90`` and ``reacquire_latency_max`` per run; the pooled
  table reports the frame-weighted mean plus the mean per-run p90 (a pooled p90
  is not a metric the evaluator defines).

Examples
--------
    python scripts/run_target.py --dry-run --out results/target_eval
    python scripts/run_target.py --data-root data/sequences --limit-sequences 2 \
        --max-frames 150 --mask none,lower --ablate part_matching,distractor_margin \
        --jobs 4 --debug-csv
    python scripts/run_target.py --list-variants
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence as TSequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tgtrack import (  # noqa: E402
    Config,
    Detection,
    GtBox,
    Sequence,
    TargetStatus,
    combine_target,
    discover_sequences,
    ensure_ort_cuda,
    evaluate_target,
    load_config,
    load_mot_sequence,
    make_target_gt,
    simulate_click,
)
from tgtrack.evaluate import BLIND_STATES, target_frame_table  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
MASK_MODES = ("none", "lower", "upper", "left", "right", "edge")

TARGET_STRATEGIES = ("stressed", "longest", "most-occluded", "first", "invisible")

#: Default observability guard: a masked target whose *post-mask visible* box is
#: smaller than this is not a partial-view test, it is an invisible-target test.
DEFAULT_MIN_VISIBLE_H = 80.0
DEFAULT_MIN_VISIBLE_AREA = 0.0

#: Deliberately low IoU bar for ``wrong_identity_locked``: the metric layer only
#: books a `false_acquisition` at IoU >= 0.5 with another identity, so a wrong
#: lock sitting at IoU 0.224 (measured on MOT17-02 id 17) would otherwise be
#: reported as "no false acquisition at all".
WRONG_IDENTITY_IOU = 0.1

#: Run-level control tag.  Runs whose chosen target is unobservable under the
#: requested mask are reported in their own table, never in the partial-view one.
CONTROL_PARTIAL_VIEW = "partial-view"
CONTROL_INVISIBLE = "invisible"

#: group -> variants.  ``baseline`` is always included by :func:`build_variants`.
GROUPS = (
    "part_matching",
    "part_aware",
    "distractor_margin",
    "hard_negative",
    "stm_ltm",
    "ltm_capacity",
    "freeze",
    "target_parts",
    "gate",
    "motion_guard",
    "drift",
    "search_guard",
    "ltm_anchor",
    "gallery_hygiene",
    "adaptive_forget",
    "anchor_ceiling",
    "anchor_adaptive",
    "partial_fusion",
    "motion_band",
    "anchor_bootstrap",
    "trail",
    "tube_velocity",
    "audit",
    "consensus",
    "channel",
    "absorb",
    "vote",
)

DEFAULT_MARGINS = (0.0, 0.06, 0.12)

#: Keys every successful run record must carry.  Pinned by the tests so the
#: table can never silently lose a headline number.
REQUIRED_RUN_KEYS = (
    "sequence",
    "variant",
    "group",
    "mask",
    "target_id",
    "frames_processed",
    "n_frames",
    "n_visible_frames",
    "target_success_rate",
    "target_success_rate_fullgt",
    "target_precision",
    "loss_rate",
    "loss_rate_locked",
    "reacquire_rate",
    "reacquire_latency",
    "reacquire_latency_p90",
    "reacquire_latency_max",
    "reacquire_events",
    "reacquire_ok",
    "false_acquisition_rate",
    "false_acquisition_rate_locked",
    "false_acquisition_events",
    "false_acquisition_events_locked",
    "false_acquisition_frames",
    "false_acquisition_frames_locked",
    "n_reported_locked",
    "hallucination_rate",
    "id_consistency",
    "target_switch_count",
    "fragmentation",
    "longest_correct_streak",
    "mean_correct_streak",
    "visibility_mean",
    "ms_per_frame",
    # observability of the *chosen* target: without these a reader cannot tell
    # whether the scenario was winnable at all (task-10)
    "control",
    "invisible_target",
    "target_reason",
    "visible_median_w",
    "visible_median_h",
    "visible_median_area",
    "visible_frames",
    "detected_frames",
    "detection_rate",
    "wrong_identity_locked",
    "coasting_correct_frames",
)

#: Columns of the pooled table (label, key, format, lower_is_better)
TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("group", "group", "s"),
    ("variant", "variant", "s"),
    ("mask", "mask", "s"),
    ("seqs", "n_sequences", "d"),
    ("frames", "n_frames", "d"),
    ("visible", "n_visible_frames", "d"),
    ("TSR", "target_success_rate", ".4f"),
    ("prec", "target_precision", ".4f"),
    ("loss", "loss_rate", ".4f"),
    ("loss_lock", "loss_rate_locked", ".4f"),
    ("recq_rate", "reacquire_rate", ".4f"),
    ("recq_lat", "reacquire_latency", ".2f"),
    ("recq_p90", "reacquire_latency_p90", ".2f"),
    ("recq_max", "reacquire_latency_max", ".2f"),
    ("FAR", "false_acquisition_rate", ".5f"),
    ("FAR_lock", "false_acquisition_rate_locked", ".5f"),
    ("FA_ev", "false_acquisition_events", "d"),
    ("FA_ev_lock", "false_acquisition_events_locked", "d"),
    ("id_cons", "id_consistency", ".4f"),
    ("switches", "target_switch_count", "d"),
    ("frag", "fragmentation", "d"),
    ("longest", "longest_correct_streak", "d"),
    ("coast_ok", "coasting_correct_frames", "d"),
    ("worst_id", "wrong_identity_locked", "d"),
    ("detect", "detection_rate", ".3f"),
    ("ms/frame", "ms_per_frame", ".2f"),
)

# Headline claims: (group, variant) pairs the report must be able to quote.
CLAIM_GROUPS = GROUPS


# --------------------------------------------------------------------------- #
# strategy variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One named point in the strategy grid."""

    name: str
    group: str
    overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def is_baseline(self) -> bool:
        return self.name == "baseline"


def build_variants(
    groups: str | TSequence[str] | None = None,
    *,
    margins: TSequence[float] = DEFAULT_MARGINS,
) -> list[Variant]:
    """The ablation grid: ``baseline`` plus every selected group's variants.

    ``groups`` accepts ``None``/``"all"`` or a comma-separated string / sequence
    of group names.  A token may also narrow its group to specific variants with
    ``group:variant`` (several via ``group:v1/v2``) -- which is what a single-arm
    run needs, and what a habit of typing ``--ablate consensus:consensus_3f``
    expects.  Unknown group names raise ``ValueError`` (a typo must not silently
    drop a claim from the table), and so do unknown variant names.
    """
    if groups is None:
        selected = list(GROUPS)
    elif isinstance(groups, str):
        selected = list(GROUPS) if groups.strip().lower() in ("", "all") else [
            g.strip() for g in groups.split(",") if g.strip()
        ]
    else:
        selected = [str(g) for g in groups]
    specs: list[tuple[str, list[str] | None]] = []
    for token in selected:
        name, _, tail = token.partition(":")
        keep = [v.strip() for v in tail.split("/") if v.strip()] if tail else []
        specs.append((name.strip(), keep or None))
    selected = [name for name, _ in specs]
    unknown = [g for g in selected if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown ablation group(s) {unknown}; known: {list(GROUPS)}")

    out: list[Variant] = [Variant("baseline", "baseline", {}, "shipped default config")]
    for g, only in specs:
        first = len(out)
        if g == "part_matching":
            out.append(Variant("part_matching_off", g, {"cost.part_matching": False}, "no part-aware matching"))
        elif g == "part_aware":
            out.append(Variant("part_aware_off", g, {"memory.part_aware": False}, "single-bank gallery"))
        elif g == "distractor_margin":
            for m in margins:
                out.append(
                    Variant(
                        f"margin_{float(m):.2f}",
                        g,
                        {"cost.distractor_margin": float(m)},
                        f"anti-latch margin {float(m):.2f}",
                    )
                )
        elif g == "hard_negative":
            out.append(Variant("hard_negative_off", g, {"memory.hard_negative_bank": False}, "no distractor bank"))
            out.append(Variant("hard_negative_confident", g,
                               {"memory.hard_negative_confident_only": True},
                               "learn hard negatives only from confident TRACKING matches (E037)"))
            for cap in (6, 12):
                out.append(Variant(f"hard_negative_cap{cap}", g,
                                   {"memory.hard_negative_capacity": int(cap)},
                                   f"smaller hard-negative bank, cap {cap} (E040)"))
        elif g == "stm_ltm":
            out.append(Variant("stm_only", g, {"cost.stm_weight": 1.0, "cost.ltm_weight": 0.0}, "short-term only"))
            out.append(Variant("balanced", g, {"cost.stm_weight": 0.6, "cost.ltm_weight": 0.4}, "shipped mix"))
            out.append(Variant("ltm_only", g, {"cost.stm_weight": 0.0, "cost.ltm_weight": 1.0}, "long-term only"))
        elif g == "ltm_capacity":
            for cap in (1, 96, 384):
                out.append(Variant(f"ltm_cap_{cap}", g, {"memory.ltm_capacity": int(cap)}, f"gallery capacity {cap}"))
        elif g == "freeze":
            out.append(Variant("freeze_on", g, {}, "freeze gallery while blind (default)"))
            out.append(Variant("freeze_off", g, {"state.frozen_states": ()}, "absorb while blind"))
        elif g == "target_parts":
            out.append(Variant("parts_full_only", g, {"embedder.target_parts": ("full",)}, "cheap: full-body only"))
            out.append(
                Variant(
                    "parts_all",
                    g,
                    {"embedder.target_parts": ("full", "upper", "lower", "head")},
                    "full part set (default)",
                )
            )
        elif g == "motion_guard":
            # E016: bound the Kalman velocity state to a multiple of the recent
            # OBSERVED speed.  Motivation: after one poisoned update the filter
            # predicted -40 px/frame for 12 frames while the person walked +8,
            # diverged ~400 px, and every correct detection then failed the motion
            # gate (results/e014_shapeprio debug CSV, MOT17-02 upper-body mask).
            out.append(Variant("vclamp_off", g, {"motion.velocity_clamp_ratio": None}, "no velocity bound"))
            for ratio in (2.0, 4.0):
                out.append(
                    Variant(
                        f"vclamp_{ratio:g}",
                        g,
                        {"motion.velocity_clamp_ratio": float(ratio)},
                        f"velocity bounded to {ratio:g}x observed speed",
                    )
                )
        elif g == "drift":
            # E021: a click-to-track tracker's failure mode is not losing the
            # target, it is confidently following somebody else.  The long-term
            # bank froze at the click and only grows on verified matches, so the
            # accepted crop's distance to it climbing is the swap signature.
            for w in (4, 8):
                for thr in (0.35, 0.50):
                    out.append(
                        Variant(
                            f"drift_w{w}_{thr:.2f}",
                            g,
                            {"memory.drift_window": int(w), "memory.drift_threshold": float(thr)},
                            f"re-anchor when mean LTM distance > {thr:.2f} over {w} frames",
                        )
                    )
        elif g == "search_guard":
            # E022: the global-search path is where FAR is born (E020/E021).  Two
            # independent guards -- stop letting the stale short-term memory vote,
            # and require the winner to actually beat the runner-up.
            out.append(
                Variant(
                    "search_stm_off",
                    g,
                    {"cost.search_uses_stm": False},
                    "score the search on the long-term anchor only",
                )
            )
            for m in (0.10, 0.20):
                out.append(
                    Variant(
                        f"search_margin_{m:.2f}",
                        g,
                        {"cost.search_margin": float(m)},
                        f"search winner must lead by {m:.2f}",
                    )
                )
            out.append(
                Variant(
                    "search_stm_off_margin_0.10",
                    g,
                    {"cost.search_uses_stm": False, "cost.search_margin": 0.10},
                    "both guards",
                )
            )
        elif g == "ltm_anchor":
            # E023: pin the click-time crops so a long run cannot evict the only
            # evidence of who was selected (E020: the gallery preferred the stranger).
            for n_anchor in (4, 16, 32):
                out.append(
                    Variant(
                        f"anchor_{n_anchor}",
                        g,
                        {"memory.ltm_anchor_count": int(n_anchor)},
                        f"first {n_anchor} gallery crops are never evicted",
                    )
                )
        elif g == "gallery_hygiene":
            # E024: after E023 proved eviction is not the poisoning vector
            # (pinning click crops changed nothing at 200 frames), test the two
            # remaining ways a wrong person can outvote the click evidence:
            # recency decay, and low-quality crop admission.
            for tag, hl in (("hl_none", None), ("hl_150", 150.0), ("hl_5000", 5000.0)):
                out.append(
                    Variant(
                        tag,
                        g,
                        {"memory.ltm_half_life": hl},
                        f"recency half-life {hl}",
                    )
                )
            for q in (0.50, 0.70):
                out.append(
                    Variant(
                        f"minq_{q:.2f}",
                        g,
                        {"memory.ltm_min_quality": float(q)},
                        f"only admit crops with quality >= {q:.2f}",
                    )
                )
        elif g == "adaptive_forget":
            # E025: E024 bracketed the right recency time-scale between 150 (great on
            # dancetrack0004) and 600 (great on dancetrack0007), so infer it from the
            # target's own appearance drift instead of picking one.
            out.append(
                Variant(
                    "adaptive_thr_0.30",
                    g,
                    {"memory.ltm_adaptive_half_life": True},
                    "half-life 150 when the look drifts, 600 when it does not",
                )
            )
            for thr in (0.25, 0.35):
                out.append(
                    Variant(
                        f"adaptive_thr_{thr:.2f}",
                        g,
                        {"memory.ltm_adaptive_half_life": True, "memory.ltm_adaptive_threshold": float(thr)},
                        f"adaptive forgetting, switch at drift {thr:.2f}",
                    )
                )
            # E026: the reference matters.  Against the live gallery (which absorbs
            # the appearance) the signal is structurally blind; against the frozen
            # click crop it is the real "how far from the clicked look are we".
            for thr in (0.25, 0.35, 0.50):
                out.append(
                    Variant(
                        f"anchor_drift_{thr:.2f}",
                        g,
                        {
                            "memory.ltm_adaptive_half_life": True,
                            "memory.ltm_adaptive_use_anchor": True,
                            "memory.ltm_adaptive_threshold": float(thr),
                        },
                        f"adaptive forgetting vs the frozen click anchor, switch at {thr:.2f}",
                    )
                )
        elif g == "anchor_ceiling":
            # E027: veto candidates that are far from the FROZEN click crop.  The
            # gallery absorbs the impostor after a wrong latch; the click crop cannot.
            for lim in (0.45, 0.55, 0.65):
                out.append(
                    Variant(
                        f"anchor_ceiling_{lim:.2f}",
                        g,
                        {"cost.anchor_max_dist": float(lim)},
                        f"refuse candidates more than {lim:.2f} from the clicked crop",
                    )
                )
            # E028: the same ceiling, armed only on the search path.
            for lim in (0.55, 0.65):
                out.append(
                    Variant(
                        f"anchor_search_{lim:.2f}",
                        g,
                        {"cost.anchor_max_dist": float(lim), "cost.anchor_ceiling_search_only": True},
                        f"anchor ceiling {lim:.2f}, search path only",
                    )
                )
        elif g == "anchor_adaptive":
            # E029: a self-calibrating band instead of a constant -- median + k*MAD
            # of the distances observed on confidently tracked crops.
            for k in (2.0, 3.0, 4.0):
                out.append(
                    Variant(
                        f"anchor_adapt_k{k:.0f}",
                        g,
                        {"cost.anchor_adaptive": True, "cost.anchor_adaptive_k": float(k)},
                        f"adaptive identity band, median + {k:.0f} robust sigma",
                    )
                )
            # E030: k4 is the best band so far; try capping it, because the one
            # sequence it does not fix (0007, FAR 0.435) is the one whose impostor
            # stays *close* to the click crop.
            for hi in (0.55, 0.60, 0.65):
                out.append(
                    Variant(
                        f"anchor_adapt_k4_c{hi:.2f}",
                        g,
                        {"cost.anchor_adaptive": True, "cost.anchor_adaptive_k": 4.0,
                         "cost.anchor_adaptive_ceil": float(hi)},
                        f"adaptive band k=4 capped at {hi:.2f}",
                    )
                )
            out.append(
                Variant(
                    "anchor_adapt_k3_hi0.95",
                    g,
                    {"cost.anchor_adaptive": True, "cost.anchor_adaptive_k": 3.0,
                     "cost.anchor_adaptive_ceil": 0.95},
                    "adaptive band with a much looser upper clip",
                )
            )
        elif g == "partial_fusion":
            # E031: trust geometry less and appearance more when the view is partial.
            # (1.0, 1.6) is the pre-E031 legacy weighting and is kept as the OFF control, so the
            # criterion-3 headline claim can always be re-measured on a sequence it was not tuned on.
            for wm, wa in ((1.0, 1.6), (0.4, 2.2), (0.5, 2.0), (0.2, 2.4)):
                out.append(
                    Variant(
                        f"partial_m{wm:.1f}_a{wa:.1f}",
                        g,
                        {"cost.partial_motion_weight": float(wm),
                         "cost.partial_appearance_weight": float(wa)},
                        f"partial view: motion {wm:.1f}, appearance {wa:.1f}",
                    )
                )
        elif g == "anchor_bootstrap":
            # E058: whether the click anchor should be averaged over the first K confident
            # matches instead of being the single click crop.  8 and 16 are two orders of
            # magnitude apart on the only axis that matters (how fast noise averages down),
            # not a tuned grid.
            for k in (0, 4, 8, 16):
                out.append(
                    Variant(
                        f"bootstrap_{int(k)}",
                        g,
                        {"cost.anchor_bootstrap_frames": int(k)},
                        "click anchor averaged over the first N accepted crops" if k else "single click crop",
                    )
                )
        elif g == "tube_velocity":
            # E061: each candidate's own tube velocity vs the target's Kalman
            # velocity.  Targets the in-tracking swap that E060 located at f54-56 of
            # dancetrack0007, where two detections overlap the prediction almost
            # equally and only their headings differ.
            for w in (0.3, 0.6, 1.2):
                out.append(
                    Variant(
                        f"tube_vel_w{w:g}",
                        g,
                        {"cost.tube_velocity_weight": float(w)},
                        f"tube-velocity consistency, weight {w:g}",
                    )
                )
        elif g == "audit":
            # E063: in-track rival audit against the frozen click anchor.  E062
            # showed the target is detected (median IoU 0.93) in 108 of the 113
            # frames where we follow somebody else while in TRACKING, so a lighter
            # check than the per-frame association can win these frames back.
            out.append(Variant("audit_off", g, {"cost.audit_interval": 0},
                               "E063 audit disabled (shipped baseline)"))
            for iv, mg in ((3, 0.05), (5, 0.05), (5, 0.10)):
                out.append(
                    Variant(
                        f"audit_{iv}_m{int(mg * 100):02d}",
                        g,
                        {"cost.audit_interval": int(iv), "cost.audit_margin": float(mg),
                         "cost.audit_persist": 2,
                         # E064 pinned these three to the SINGLE-reference
                         # behaviour they were measured with, so the E063 tables
                         # stay reproducible; the 2-reference variants below are
                         # new names.
                         "cost.audit_use_recent": False},
                        f"rival audit every {iv} frames, margin {mg:g}, 1 reference",
                    )
                )
            # E064: two references must agree (click crop AND recent appearance).
            # E063's single reference won dancetrack0007 and damaged dancetrack0005,
            # the sequence whose look legitimately changes (E024).
            for iv in (3, 5):
                out.append(
                    Variant(
                        f"audit_2ref_{iv}_m05",
                        g,
                        {"cost.audit_interval": int(iv), "cost.audit_margin": 0.05,
                         "cost.audit_persist": 2, "cost.audit_use_recent": True},
                        f"rival audit every {iv} frames, margin 0.05, 2 references",
                    )
                )
            out.append(
                Variant(
                    "audit_1ref_5_m05",
                    g,
                    {"cost.audit_interval": 5, "cost.audit_margin": 0.05,
                     "cost.audit_persist": 2, "cost.audit_use_recent": False},
                    "rival audit every 5 frames, margin 0.05, 1 reference (matched control)",
                )
            )
        elif g == "trail":
            # E059: motion-signature evidence (per-step turning angles + log speeds
            # from the candidate's own short tube) as a SECOND channel for the search
            # path.  Exists because on DanceTrack appearance cannot separate dancers
            # (pairwise AUC 0.43-0.58) and motion/gait is the fallback the literature
            # recommends (GAE-Net 2025).  Weight 0 = off = the shipped baseline.
            for w in (0.5, 1.0, 2.0):
                out.append(
                    Variant(
                        f"trail_w{w:g}",
                        g,
                        {"cost.trail_weight": float(w)},
                        f"motion-signature cost, weight {w:g}",
                    )
                )
        elif g == "motion_band":
            # E032: the E029 construction on the motion axis instead of appearance.
            for k in (2.0, 3.0, 4.0):
                out.append(
                    Variant(
                        f"motion_band_k{k:.0f}",
                        g,
                        {"cost.motion_gate_adaptive": True, "cost.motion_gate_k": float(k)},
                        f"adaptive motion band, median + {k:.0f} robust sigma",
                    )
                )
        elif g == "gate":
            # The gate/search fix: the radius used to be in bare pixels (not scaled by
            # the frame diagonal) and grew from the coast-damped speed, so it peaked and
            # then DECAYED; and global search armed only past `lost_long_after`.  A
            # target that reappeared a few hundred px away was rejected as
            # `outside_radius` until fsu=61.  These four variants read out the fix.
            out.append(
                Variant(
                    "gate_legacy",
                    g,
                    {"state.global_states": (), "state.gate_growth_freeze_speed": False},
                    "pre-fix: pixel radius, search only after lost_long",
                )
            )
            out.append(Variant("gate_search_early", g, {}, "search armed at lost_short (default)"))
            out.append(
                Variant(
                    "gate_search_late",
                    g,
                    {"state.global_states": ("lost_long", "gave_up", "out_of_frame")},
                    "search armed only at lost_long",
                )
            )
            out.append(
                Variant(
                    "gate_growth_unfrozen",
                    g,
                    {"state.gate_growth_freeze_speed": False},
                    "grow the radius from the decaying coasted speed",
                )
            )
        elif g == "consensus":
            # E071: temporal consensus on the handback (docs/E071_PLAN.md).
            # `consensus_off` is the reference arm and repeats the shipped value
            # explicitly, so a reader of the table can see what the other arms
            # are compared against without going to config.py.
            out.append(
                Variant(
                    "consensus_off",
                    g,
                    {"cost.commit_consensus": False},
                    "reference: no vote, pre-E071 handback rule",
                )
            )
            out.append(
                Variant(
                    "consensus_2f",
                    g,
                    {"cost.commit_consensus": True, "cost.consensus_frames": 2},
                    "two consecutive consistent proposals before a handover",
                )
            )
            out.append(
                Variant(
                    "consensus_3f",
                    g,
                    {"cost.commit_consensus": True, "cost.consensus_frames": 3},
                    "three consecutive consistent proposals before a handover",
                )
            )
            out.append(
                Variant(
                    "consensus_2f_pool",
                    g,
                    {
                        "cost.commit_consensus": True,
                        "cost.consensus_frames": 2,
                        "cost.consensus_pool": True,
                    },
                    "2 frames + pooled chain descriptor exposed in the diag",
                )
            )
        elif g == "channel":
            # E077 (docs/E077_PLAN.md): two independent fixes for the two defects
            # E076 isolated, plus the conjunction.  Both switches default to False,
            # so `channel_off` is the shipped behaviour spelled out explicitly and
            # the three candidates differ from it in EXACTLY one or two booleans --
            # the two thresholds (`cost.channel_geom_tau`, `cost.channel_app_tau`)
            # are NOT swept here, they were derived once from the dev split
            # (results/e077_calib/thresholds.json) and are identical in every arm.
            out.append(
                Variant(
                    "channel_off",
                    g,
                    {"cost.channel_accept": False, "motion.aspect_clamp_coast": False},
                    "reference: both E077 arms off (the shipped behaviour)",
                )
            )
            out.append(
                Variant(
                    "channel_fusion",
                    g,
                    {"cost.channel_accept": True},
                    "B: channel-conditional acceptance -- geometry OR appearance may carry the frame",
                )
            )
            out.append(
                Variant(
                    "channel_aspect",
                    g,
                    {"motion.aspect_clamp_coast": True},
                    "C: bounded-aspect coasting -- clamp the coasted aspect to the last observed band",
                )
            )
            out.append(
                Variant(
                    "channel_both",
                    g,
                    {"cost.channel_accept": True, "motion.aspect_clamp_coast": True},
                    "D: B + C",
                )
            )
        elif g == "absorb":
            # E078 (docs/EXPERIMENTS.md): gate long-term gallery admission on the
            # distance to the frozen click anchor.  E077-PREP measured that 52.5 % of
            # `tracking`-reported frames are wrong and that 88.6 % of those sit on a
            # DIFFERENT real person at IoU >= 0.5 (median area ratio 1.03): the gallery
            # absorbs the impostor and every later frame confirms him.  The ratios are
            # derived from data (results/absorb_sweep/ratio_derivation.json), not tuned:
            # 0.90 is the Youden-optimal cut of the measured correct-vs-wrong ratio
            # distribution, 1.09 its correct-frame p99.  `absorb_off` spells out the
            # shipped behaviour, which is what the default (False) does anyway.
            out.append(
                Variant(
                    "absorb_off",
                    g,
                    {
                        "memory.ltm_absorb_anchor_only": False,
                        "memory.ltm_absorb_anchor_ratio": 1.25,
                    },
                    "reference: gallery absorbs every accepted crop (shipped behaviour)",
                )
            )
            out.append(
                Variant(
                    "absorb_tight",
                    g,
                    {
                        "memory.ltm_absorb_anchor_only": True,
                        "memory.ltm_absorb_anchor_ratio": 0.90,
                    },
                    "absorb only inside 0.90 x the click-anchor band (Youden-optimal cut)",
                )
            )
            out.append(
                Variant(
                    "absorb_loose",
                    g,
                    {
                        "memory.ltm_absorb_anchor_only": True,
                        "memory.ltm_absorb_anchor_ratio": 1.09,
                    },
                    "absorb only inside 1.09 x the click-anchor band (correct-frame p99)",
                )
            )
        elif g == "vote":
            # E079 (docs/E079_PLAN.md): the anchor veto is a one-reference test against
            # the frozen click crop, and E077 showed it is where the blindness lives
            # (`all_gated:anchor_ceiling`).  This group adds a refreshed anchor (R2) and
            # the long-term gallery (R3) and lets a vetoed candidate in when it passes
            # k of the 3 references, each measured against the SAME E029 band -- no new
            # constant.  `vote_off` spells out the shipped behaviour; `vote_unanimous`
            # (k = 3) is the null control, because a candidate only reaches the vote
            # after R1 already rejected it, so k = 3 can never admit anything.
            out.append(
                Variant(
                    "vote_off",
                    g,
                    {"cost.anchor_vote": "off", "cost.anchor_vote_alpha": 0.1},
                    "reference: the click-crop veto, one reference (shipped behaviour)",
                )
            )
            out.append(
                Variant(
                    "vote_majority",
                    g,
                    {"cost.anchor_vote": "majority", "cost.anchor_vote_alpha": 0.1},
                    "admit an anchor-vetoed candidate when >= 2 of R1/R2/R3 pass it",
                )
            )
            out.append(
                Variant(
                    "vote_unanimous",
                    g,
                    {"cost.anchor_vote": "unanimous", "cost.anchor_vote_alpha": 0.1},
                    "k = 3 control: unreachable for a vetoed candidate -> identical to vote_off",
                )
            )
            # E081 (docs/E081_PLAN.md): E079 cleared the FAR bar by 13.4 pp and missed
            # the TSR bar by 0.12 pp.  The pre-registered follow-up is to spend the
            # remaining freedom on WHEN the vote may run, not on another reference:
            # arm it only while the tracker is in a distressed regime (share of the
            # trailing 50 frames spent blind).  tau = 1.0 is the null control (never
            # distressed -> never votes -> identical to vote_off); tau < 0 disables
            # the gate and reduces to vote_majority.
            for _tau in (0.20, 0.35, 0.50, 0.65):
                out.append(
                    Variant(
                        f"vote_adaptive_{_tau:.2f}",
                        g,
                        {
                            "cost.anchor_vote": "adaptive",
                            "cost.anchor_vote_alpha": 0.1,
                            "cost.anchor_vote_distress_window": 50,
                            "cost.anchor_vote_distress_tau": float(_tau),
                        },
                        f"k = 2 vote armed only while distress > {_tau:.2f}",
                    )
                )
            out.append(
                Variant(
                    "vote_adaptive_1.00",
                    g,
                    {
                        "cost.anchor_vote": "adaptive",
                        "cost.anchor_vote_alpha": 0.1,
                        "cost.anchor_vote_distress_window": 50,
                        "cost.anchor_vote_distress_tau": 1.0,
                    },
                    "null control: distress never exceeds 1.0 -> the gate never opens",
                )
            )
        if only is not None:
            def _hit(name: str) -> bool:
                return any(name == o or name.startswith(o) for o in only)

            made = out[first:]
            missing = [o for o in only if not any(_hit(v.name) for v in made)]
            if missing:
                raise ValueError(
                    f"unknown variant(s) {missing} in group {g!r}; known: {[v.name for v in made]}"
                )
            out[first:] = [v for v in made if _hit(v.name)]
    return out


def variant_table(variants: TSequence[Variant]) -> str:
    lines = ["%-20s %-18s %s" % ("variant", "group", "overrides")]
    for v in variants:
        ov = ", ".join(f"{k}={val!r}" for k, val in sorted(v.overrides.items())) or "(defaults)"
        lines.append("%-20s %-18s %s" % (v.name, v.group, ov))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #
def _gt_stats(seq: Sequence) -> dict[int, dict[str, float]]:
    """Per-identity presence / occlusion / edge statistics of a sequence."""
    stats: dict[int, dict[str, float]] = {}
    n = len(seq)
    w = float(seq.width or 0)
    h = float(seq.height or 0)
    if not w or not h:
        w, h = (float(v) for v in _frame_size(seq))
    for idx in range(n):
        seen: set[int] = set()
        for box in seq.gt_at(idx):
            tid = int(box.track_id)
            seen.add(tid)
            st = stats.setdefault(
                tid,
                {"frames": 0, "first": float(idx), "last": float(idx), "occluded": 0.0, "edge": 0.0, "vis": 0.0, "area": 0.0},
            )
            st["frames"] += 1.0
            st["last"] = float(idx)
            x0, y0, x1, y1 = (float(v) for v in box.xyxy)
            if float(getattr(box, "visibility", 1.0)) < 0.9:
                st["occluded"] += 1.0
            if x0 <= 1.0 or y0 <= 1.0 or x1 >= w - 1.0 or y1 >= h - 1.0:
                st["edge"] += 1.0
            st["vis"] += float(getattr(box, "visibility", 1.0))
            st["area"] += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    for tid, st in stats.items():
        span = max(1.0, st["last"] - st["first"] + 1.0)
        st["gap_ratio"] = max(0.0, span - st["frames"]) / span
        st["occlusion_ratio"] = st["occluded"] / max(1.0, st["frames"])
        st["edge_ratio"] = st["edge"] / max(1.0, st["frames"])
        st["mean_visibility"] = st["vis"] / max(1.0, st["frames"])
        st["span"] = span
    return stats


def choose_target_id(
    seq: Sequence,
    strategy: str = "stressed",
    *,
    candidates: Iterable[int] | None = None,
) -> int | None:
    """Pick which GT identity the simulated operator clicks.

    ``stressed`` (default) deliberately prefers an identity that is long-lived
    *and* hard: interrupted presence, low visibility and edge contact all raise
    the score, so the run exercises loss/re-acquisition and partial views instead
    of an unbroken walk across the frame::

        score = frames * (1 + 2*gap_ratio + 2*occlusion_ratio + edge_ratio)

    ``longest`` uses raw presence, ``most-occluded`` the lowest mean visibility,
    ``first`` the earliest appearance.  ``invisible`` is the control strategy: it
    picks the *smallest* person, i.e. the identity most likely to vanish once a
    mask rewrites its box (the true post-mask ranking lives in :func:`pick_target`,
    which passes it real observability data; this raw ranking is the documented
    fallback when no mask information is available).  Ties break on the smallest
    id, so the choice is deterministic.

    ``candidates`` restricts the pool (used by :func:`pick_target` to inject the
    observability guard).  An empty pool returns ``None`` rather than guessing.
    """
    stats = _gt_stats(seq)
    if candidates is not None:
        allowed = {int(t) for t in candidates}
        stats = {t: s for t, s in stats.items() if t in allowed}
    if not stats:
        return None
    if strategy not in TARGET_STRATEGIES:
        raise ValueError(f"unknown target strategy {strategy!r}; known: {list(TARGET_STRATEGIES)}")
    if strategy == "longest":
        key = lambda t: (-stats[t]["frames"], t)  # noqa: E731
    elif strategy == "first":
        key = lambda t: (stats[t]["first"], t)  # noqa: E731
    elif strategy == "most-occluded":
        key = lambda t: (stats[t]["mean_visibility"], -stats[t]["frames"], t)  # noqa: E731
    elif strategy == "invisible":
        key = lambda t: (stats[t]["area"] / max(1.0, stats[t]["frames"]), t)  # noqa: E731
    else:
        key = lambda t: (  # noqa: E731
            -(
                stats[t]["frames"]
                * (
                    1.0
                    + 2.0 * stats[t]["gap_ratio"]
                    + 2.0 * stats[t]["occlusion_ratio"]
                    + stats[t]["edge_ratio"]
                )
            ),
            -stats[t]["frames"],
            t,
        )
    return int(min(stats, key=key))


# --------------------------------------------------------------------------- #
# observability guard (task-10)
# --------------------------------------------------------------------------- #
def visible_box_stats(
    seq: Sequence,
    target_id: int,
    mode: str = "none",
    *,
    limit: int | None = None,
    margin: float = 0.02,
    edge_frac: float = 0.5,
    drop_below: float = 0.2,
    fill: Any = 0,
) -> dict[str, Any]:
    """Median size of ``target_id``'s *post-mask visible* box under ``mode``.

    This reuses :func:`build_mask_plan` -- the very transform the run applies --
    instead of re-deriving the visible rectangle, so the guard cannot drift away
    from the scenario it guards.  ``frames`` counts the frames in which the
    identity survives the mask at all; ``frames == 0`` means the target does not
    exist in the run.
    """
    plan = build_mask_plan(
        seq, target_id, mode, limit=limit, margin=margin, edge_frac=edge_frac, drop_below=drop_below, fill=fill
    )
    idx = [i for i in plan.frames() if plan.gt_target.get(i) is not None]
    out: dict[str, Any] = {
        "mode": str(mode),
        "frames": len(idx),
        "first": (int(idx[0]) if idx else None),
        "median_w": 0.0,
        "median_h": 0.0,
        "median_area": 0.0,
        "min_h": 0.0,
        "min_area": 0.0,
        "visible_fraction": 0.0,
    }
    if not idx:
        return out
    ws: list[float] = []
    hs: list[float] = []
    areas: list[float] = []
    for i in idx:
        x0, y0, x1, y1 = (float(v) for v in plan.gt_target[i])
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        ws.append(w)
        hs.append(h)
        areas.append(w * h)
    out.update(
        {
            "median_w": float(np.median(ws)),
            "median_h": float(np.median(hs)),
            "median_area": float(np.median(areas)),
            "min_h": float(np.min(hs)),
            "min_area": float(np.min(areas)),
            "visible_fraction": float(plan.visible_fraction()),
        }
    )
    return out


def observability_table(
    seq: Sequence,
    masks: TSequence[str] = ("none",),
    *,
    limit: int | None = None,
    margin: float = 0.02,
    edge_frac: float = 0.5,
    drop_below: float = 0.2,
    fill: Any = 0,
) -> dict[int, dict[str, Any]]:
    """Post-mask visible-box statistics for every GT identity, per mask.

    Each record carries the per-mask stats plus the *binding* (worst) median
    width / height / area / frame count across the requested masks: a candidate
    only counts as observable when it is observable under **every** mask the run
    will apply, which is exactly what the guard has to enforce.
    """
    out: dict[int, dict[str, Any]] = {}
    for tid in sorted(int(t) for t in _gt_stats(seq)):
        per = {
            str(m): visible_box_stats(
                seq, tid, m, limit=limit, margin=margin, edge_frac=edge_frac, drop_below=drop_below, fill=fill
            )
            for m in masks
        }
        present = [m for m, st in per.items() if int(st["frames"]) > 0]
        if present:
            worst = min(present, key=lambda m: (per[m]["median_h"], per[m]["median_area"], m))
            median_w = min(per[m]["median_w"] for m in present)
            median_h = min(per[m]["median_h"] for m in present)
            median_area = min(per[m]["median_area"] for m in present)
            frames = min(int(per[m]["frames"]) for m in present)
        else:  # pragma: no cover - every identity is present in some frame
            worst, median_w, median_h, median_area, frames = (str(masks[0]), 0.0, 0.0, 0.0, 0)
        out[tid] = {
            "target_id": tid,
            "masks": per,
            "worst_mask": worst,
            "median_w": float(median_w),
            "median_h": float(median_h),
            "median_area": float(median_area),
            "frames": int(frames),
        }
    return out


def _observable(rec: dict[str, Any], min_h: float, min_area: float) -> bool:
    if int(rec.get("frames", 0)) <= 0:
        return False
    if float(rec.get("median_h", 0.0)) < float(min_h):
        return False
    if float(min_area) > 0.0 and float(rec.get("median_area", 0.0)) < float(min_area):
        return False
    return True


def _invisible_message(
    seq: Sequence,
    tid: int,
    rec: dict[str, Any],
    masks: TSequence[str],
    guard: dict[str, Any],
    *,
    forced: bool = False,
) -> str:
    how = "forced by --target-id" if forced else "chosen by the strategy"
    if float(guard.get("min_visible_area", 0.0)) > 0.0:
        need = "%.0f px of height and %.0f px^2" % (
            float(guard["min_visible_h"]),
            float(guard["min_visible_area"]),
        )
    else:
        need = "%.0f px of height" % float(guard["min_visible_h"])
    return (
        "%s: target id %d (%s) has a median post-mask visible box of %.0fx%.0f px (%.0f px^2, %d frames) "
        "under mask=%s -- below the observability guard (%s). This run measures an UNOBSERVABLE target: "
        "it is reported in the invisible-target control table, never as a partial-view result."
        % (
            seq.name,
            int(tid),
            how,
            float(rec.get("median_w", 0.0)),
            float(rec.get("median_h", 0.0)),
            float(rec.get("median_area", 0.0)),
            int(rec.get("frames", 0)),
            ",".join(str(m) for m in masks),
            need,
        )
    )


def pick_target(
    seq: Sequence,
    strategy: str = "stressed",
    *,
    masks: TSequence[str] = ("none",),
    min_visible_h: float = DEFAULT_MIN_VISIBLE_H,
    min_visible_area: float = DEFAULT_MIN_VISIBLE_AREA,
    allow_invisible: bool = False,
    forced_id: int | None = None,
    limit: int | None = None,
    margin: float = 0.02,
    edge_frac: float = 0.5,
    drop_below: float = 0.2,
    fill: Any = 0,
) -> dict[str, Any]:
    """Choose the clicked identity **and decide whether the scenario is meaningful**.

    Selection used to look at track statistics only, so on MOT17-02 a 38x54 px
    post-mask box was chosen and no detection overlapped it on any frame: the run
    produced a table that looked like a partial-view failure while measuring an
    invisible target.  This function adds the missing constraint.

    The decision is a plain dict so the CLI and the tests can read every part of
    it (and so it can be written to ``summary.json``):

    ``target_id`` / ``reason``
        ``forced`` (explicit ``--target-id``, observable), ``forced_invisible``
        (explicit but unobservable -- kept, loudly flagged),
        ``observable`` (guard satisfied), ``invisible_strategy`` (the ``invisible``
        control strategy), ``invisible_allowed`` (guard failed but
        ``--allow-invisible-target`` was given), ``no_candidate`` (nothing is
        observable and the caller must fail), ``no_identities``,
        ``forced_missing``.
    ``invisible`` / ``control``
        whether the chosen target is unobservable, and therefore whether the run
        belongs in the invisible-target control table instead of the partial-view
        table.
    ``observability``
        per-identity records from :func:`observability_table`, so the report can
        show what was available and not only what was chosen.
    """
    mask_list = [str(m) for m in (masks or ("none",))]
    obs = observability_table(
        seq, mask_list, limit=limit, margin=margin, edge_frac=edge_frac, drop_below=drop_below, fill=fill
    )
    guard: dict[str, Any] = {
        "min_visible_h": float(min_visible_h),
        "min_visible_area": float(min_visible_area),
        "masks": mask_list,
    }
    decision: dict[str, Any] = {
        "target_id": None,
        "reason": "no_identities",
        "invisible": False,
        "control": CONTROL_PARTIAL_VIEW,
        "guard_passed": False,
        "guard": guard,
        "observability": obs,
        "selected": None,
        "message": "",
    }
    if not obs:
        decision["message"] = f"{seq.name}: no ground-truth identity in the sequence"
        return decision
    observable = [t for t in sorted(obs) if _observable(obs[t], min_visible_h, min_visible_area)]

    def _mark(tid: int, reason: str, *, invisible: bool, forced: bool = False) -> dict[str, Any]:
        rec = obs[int(tid)]
        decision["target_id"] = int(tid)
        decision["reason"] = reason
        decision["selected"] = rec
        decision["invisible"] = bool(invisible)
        decision["guard_passed"] = not invisible
        decision["control"] = CONTROL_INVISIBLE if invisible else CONTROL_PARTIAL_VIEW
        if invisible:
            decision["message"] = _invisible_message(seq, int(tid), rec, mask_list, guard, forced=forced)
        return decision

    if forced_id is not None:
        tid = int(forced_id)
        if tid not in obs:
            decision["reason"] = "forced_missing"
            decision["message"] = (
                f"{seq.name}: --target-id {tid} is not a ground-truth identity (known: {sorted(obs)})"
            )
            return decision
        if _observable(obs[tid], min_visible_h, min_visible_area):
            return _mark(tid, "forced", invisible=False)
        return _mark(tid, "forced_invisible", invisible=True, forced=True)

    if strategy == "invisible":
        tid = int(min(obs, key=lambda t: (obs[t]["median_area"], obs[t]["median_h"], t)))
        if _observable(obs[tid], min_visible_h, min_visible_area):
            # the least observable identity is still usable: keep it as an
            # ordinary run rather than polluting the control table
            return _mark(tid, "invisible_strategy", invisible=False)
        return _mark(tid, "invisible_strategy", invisible=True)

    if observable:
        tid = choose_target_id(seq, strategy, candidates=observable)
        if tid is None:  # pragma: no cover - observable is non-empty
            decision["reason"] = "no_candidate"
            return decision
        return _mark(int(tid), "observable", invisible=False)

    if allow_invisible:
        tid = choose_target_id(seq, strategy)
        if tid is None:  # pragma: no cover - obs is non-empty
            decision["reason"] = "no_candidate"
            return decision
        return _mark(int(tid), "invisible_allowed", invisible=True)

    ranked = sorted(obs, key=lambda t: (-obs[t]["median_h"], -obs[t]["median_area"], t))[:3]
    detail = "; ".join(
        "id %d median %.0fx%.0f px (%.0f px^2, %d frames)"
        % (t, obs[t]["median_w"], obs[t]["median_h"], obs[t]["median_area"], obs[t]["frames"])
        for t in ranked
    )
    decision["reason"] = "no_candidate"
    if float(min_visible_area) > 0.0:
        need = ">= %.0f px high and %.0f px^2" % (float(min_visible_h), float(min_visible_area))
    else:
        need = ">= %.0f px high" % float(min_visible_h)
    decision["message"] = (
        "%s: no ground-truth identity survives mask=%s with a median post-mask visible box %s "
        "(largest candidates: %s). Lower --min-visible-h/--min-visible-area, or pass "
        "--allow-invisible-target to run the unobservable-target control on purpose."
        % (seq.name, ",".join(mask_list), need, detail)
    )
    return decision


# --------------------------------------------------------------------------- #
# synthetic partial-view masking
# --------------------------------------------------------------------------- #
def _half_rects(box: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """``(visible_rect, hidden_rect)`` for a half-plane mask of ``box``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if mode == "lower":
        return np.array([x0, y0, x1, ym]), np.array([x0, ym, x1, y1])
    if mode == "upper":
        return np.array([x0, ym, x1, y1]), np.array([x0, y0, x1, ym])
    if mode == "left":
        return np.array([xm, y0, x1, y1]), np.array([x0, y0, xm, y1])
    if mode == "right":
        return np.array([x0, y0, xm, y1]), np.array([xm, y0, x1, y1])
    raise ValueError(f"not a half-plane mask: {mode!r}")


def _area(rect: np.ndarray) -> float:
    return max(0.0, float(rect[2] - rect[0])) * max(0.0, float(rect[3] - rect[1]))


def _intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])], dtype=np.float64
    )


def _pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes (0.0 when they do not overlap)."""
    ra = np.asarray(a, dtype=np.float64)
    rb = np.asarray(b, dtype=np.float64)
    inter = _area(_intersect(ra, rb))
    union = _area(ra) + _area(rb) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def clip_rect(rect: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            max(0.0, min(float(rect[0]), width)),
            max(0.0, min(float(rect[1]), height)),
            max(0.0, min(float(rect[2]), width)),
            max(0.0, min(float(rect[3]), height)),
        ],
        dtype=np.float64,
    )


def _resolve_fill(frame: np.ndarray, fill: Any) -> Any:
    """Resolve the occluder colour (``"border"`` = median of the top strip)."""
    if isinstance(fill, str):
        if fill != "border":
            raise ValueError(f"unknown fill {fill!r}")
        strip = frame[: max(1, frame.shape[0] // 40), :, :].reshape(-1, frame.shape[2])
        return tuple(int(v) for v in np.median(strip, axis=0))
    return fill


@dataclass
class MaskPlan:
    """A consistent (frame, ground truth) transform for one synthetic stress."""

    mode: str
    target_id: int
    frame_w: int
    frame_h: int
    hidden: dict[int, np.ndarray] = field(default_factory=dict)
    delta: tuple[int, int] = (0, 0)
    gt_target: dict[int, np.ndarray | None] = field(default_factory=dict)
    full_gt_target: dict[int, np.ndarray | None] = field(default_factory=dict)
    all_gt: dict[int, list[GtBox]] = field(default_factory=dict)
    visibility: dict[int, float | None] = field(default_factory=dict)
    fill: Any = 0
    margin: float = 0.02
    drop_below: float = 0.2

    @property
    def is_identity(self) -> bool:
        return self.mode == "none"

    def frames(self) -> list[int]:
        return sorted(self.all_gt)

    def visible_fraction(self) -> float:
        vals = [v for v in self.visibility.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    def apply(self, frame: np.ndarray | None, frame_idx: int) -> np.ndarray | None:
        """Return the masked frame (never mutates the loader's cached array)."""
        if frame is None or self.is_identity:
            return frame
        import cv2

        idx = int(frame_idx)
        out = frame
        dx, dy = self.delta
        if dx or dy:
            mat = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
            out = cv2.warpAffine(
                out,
                mat,
                (out.shape[1], out.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=_resolve_fill(out, self.fill),
            )
        rect = self.hidden.get(idx)
        if rect is not None:
            out = out.copy()
            x0, y0, x1, y1 = (int(round(float(v))) for v in rect)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(out.shape[1], x1), min(out.shape[0], y1)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = _resolve_fill(out, self.fill)
        return out


def build_mask_plan(
    seq: Sequence,
    target_id: int,
    mode: str = "none",
    *,
    limit: int | None = None,
    margin: float = 0.02,
    edge_frac: float = 0.5,
    drop_below: float = 0.2,
    fill: Any = 0,
) -> MaskPlan:
    """Build the partial-view stress plan for one sequence.

    Only the target's own box is reduced to its visible rectangle; every other
    person keeps their box with a reduced ``visibility`` (and is dropped when
    more than ``1 - drop_below`` of them is hidden).  ``edge`` mode applies one
    constant translation to the whole sequence instead of painting, so the scene
    stays geometrically consistent for the motion model.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask mode {mode!r}; known: {list(MASK_MODES)}")
    n = len(seq) if limit is None else max(0, min(int(limit), len(seq)))
    w, h = _frame_size(seq)
    plan = MaskPlan(mode=mode, target_id=int(target_id), frame_w=w, frame_h=h, fill=fill, margin=margin,
                    drop_below=drop_below)
    full = make_target_gt(seq, target_id)
    plan.full_gt_target = {i: (None if full.get(i) is None else np.asarray(full[i], np.float64).copy()) for i in range(n)}

    if mode == "none":
        # provably identity transform: the ground truth is copied verbatim (no
        # clipping, no visibility edits), so a `none` run measures the harness
        # itself instead of the stress.
        for i in range(n):
            plan.all_gt[i] = list(seq.gt_at(i))
            box = plan.full_gt_target[i]
            plan.gt_target[i] = None if box is None else np.asarray(box, np.float64).copy()
            plan.visibility[i] = None if box is None else 1.0
        return plan

    if mode == "edge":
        # Push the whole scene toward the border the target is nearest to, by a
        # constant number of pixels, so the target straddles (or leaves) the
        # frame while the motion inside the frame stays a pure translation --
        # this is the out-of-frame / coasting stress.
        first = next((i for i in range(n) if plan.full_gt_target[i] is not None), None)
        if first is not None:
            box = np.asarray(plan.full_gt_target[first], np.float64)
            width = max(1.0, float(box[2] - box[0]))
            shift = int(math.ceil(float(edge_frac) * width))
            near_left = (box[0] - 0.0) <= (float(w) - box[2])
            plan.delta = (-shift if near_left else shift, 0)

    for i in range(n):
        boxes = seq.gt_at(i)
        target_box = plan.full_gt_target[i]
        hidden: np.ndarray | None = None
        visible: np.ndarray | None = None
        if target_box is not None and mode not in ("none", "edge"):
            visible, hidden = _half_rects(target_box, mode)
            mx = margin * max(1.0, float(target_box[2] - target_box[0]))
            my = margin * max(1.0, float(target_box[3] - target_box[1]))
            pad = max(1.0, 0.5 * (mx + my))
            # Grow the occluder *outward*, away from the visible half, so the
            # painted rectangle never intrudes on the visible rectangle: the
            # two are exactly complementary (up to the frame clip) and the
            # harness can therefore claim an unambiguous visible fraction.
            hx0, hy0, hx1, hy1 = (float(v) for v in hidden)
            if mode == "lower":
                hx0, hy0, hx1, hy1 = hx0 - pad, hy0, hx1 + pad, hy1 + my
            elif mode == "upper":
                hx0, hy0, hx1, hy1 = hx0 - pad, hy0 - my, hx1 + pad, hy1
            elif mode == "left":
                hx0, hy0, hx1, hy1 = hx0 - pad, hy0 - pad, hx1, hy1 + pad
            else:  # right
                hx0, hy0, hx1, hy1 = hx0, hy0 - pad, hx1 + pad, hy1 + pad
            hidden = clip_rect(np.array([hx0, hy0, hx1, hy1], np.float64), w, h)
            plan.hidden[i] = hidden

        dx, dy = plan.delta
        out_boxes: list[GtBox] = []
        vis_frac = 1.0
        for box in boxes:
            rect = np.asarray(box.xyxy, np.float64).copy()
            if dx or dy:
                rect = np.array([rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy], np.float64)
            clipped = clip_rect(rect, w, h)
            orig_area = _area(rect)
            frac = 1.0 if orig_area <= 0 else _area(clipped) / orig_area
            if hidden is not None:
                frac *= 1.0 - (_area(_intersect(rect, hidden)) / max(1e-9, orig_area))
            frac = float(max(0.0, min(1.0, frac)))
            if int(box.track_id) == int(target_id):
                vis_frac = frac
                if mode not in ("none", "edge"):
                    rect = np.asarray(visible, np.float64)
                else:
                    rect = clipped
                if mode == "edge":
                    vis_frac = 1.0 if orig_area <= 0 else _area(clipped) / orig_area
                if _area(rect) <= 1.0:
                    continue
            elif frac < drop_below:
                continue
            out_boxes.append(
                GtBox(
                    frame=int(i),
                    track_id=int(box.track_id),
                    xyxy=rect,
                    conf=float(getattr(box, "conf", 1.0)),
                    visibility=frac,
                )
            )
            if int(box.track_id) == int(target_id):
                plan.gt_target[i] = np.asarray(rect, np.float64).copy()
        plan.all_gt[i] = out_boxes
        target_present = int(target_id) in {int(b.track_id) for b in out_boxes}
        plan.visibility[i] = vis_frac if target_present else None
        if not target_present:
            plan.gt_target[i] = None

    for i in range(n):
        plan.gt_target.setdefault(i, None)
    return plan


def _frame_size(seq: Sequence, fallback: tuple[int, int] = (320, 240)) -> tuple[int, int]:
    w, h = int(seq.width or 0), int(seq.height or 0)
    if w > 0 and h > 0:
        return w, h
    for i in range(min(len(seq), 8)):
        img = seq.frame(i)
        if img is not None:
            return int(img.shape[1]), int(img.shape[0])
    return int(fallback[0]), int(fallback[1])


# --------------------------------------------------------------------------- #
# detection / embedding sources
# --------------------------------------------------------------------------- #
class GtDetector:
    """Oracle detector: the masked-world GT boxes as person detections.

    Identity is deliberately **not** leaked (``track_id`` stays ``None``), so the
    target layer still has to associate.  This is the controlled mode used by
    ``--detector gt`` and by the tests; a masked half-body box arrives here as a
    genuine partial detection.
    """

    def __init__(self, plan: MaskPlan, *, score: float = 1.0) -> None:
        self.plan = plan
        self.score = float(score)

    def detect(self, frame: np.ndarray | None, frame_idx: int | None = None) -> list[Detection]:
        idx = 0 if frame_idx is None else int(frame_idx)
        out = []
        for box in self.plan.all_gt.get(idx, ()):
            out.append(
                Detection(
                    xyxy=np.asarray(box.xyxy, np.float64).copy(),
                    score=self.score,
                    cls=0,
                    visible_ratio=float(getattr(box, "visibility", 1.0)),
                )
            )
        return out

    def info(self) -> dict[str, Any]:
        return {"kind": "gt-oracle", "n_frames": len(self.plan.all_gt)}


class YoloDetector:
    """Thin adapter around ``tgtrack.detector.PersonDetector``."""

    def __init__(self, cfg: Config) -> None:
        from tgtrack.detector import PersonDetector

        self.model = PersonDetector(cfg.detector)

    def detect(self, frame: np.ndarray | None, frame_idx: int | None = None) -> list[Detection]:
        if frame is None:
            return []
        return list(self.model.detect(frame))

    def info(self) -> dict[str, Any]:
        try:
            return dict(self.model.model_info())
        except Exception as exc:  # pragma: no cover - diagnostics only
            return {"kind": "yolo", "error": repr(exc)}


#: one detector / embedder per process: both are stateless, and rebuilding an
#: ONNX session per run would dominate the runtime of the grid.
_DETECTOR_CACHE: dict[Any, Any] = {}
_EMBEDDER_CACHE: dict[Any, Any] = {}
_ORT_READY = False


def _ort_once(device: str | None) -> None:
    """Call ``ensure_ort_cuda()`` once per process (ReID is CPU-only without it)."""
    global _ORT_READY
    if _ORT_READY:
        return
    _ORT_READY = True
    if device == "cpu" or os.environ.get("TGTRACK_SKIP_ORT"):
        return
    try:
        ensure_ort_cuda()
    except Exception as exc:  # pragma: no cover - depends on the machine
        print(f"[ort] WARN ensure_ort_cuda failed ({type(exc).__name__}: {exc})", flush=True)


def get_detector(cfg: Config, kind: str = "yolo") -> Any:
    if kind != "yolo":
        raise ValueError(f"get_detector only builds the real detector, not {kind!r}")
    key = (
        cfg.detector.weights,
        int(cfg.detector.imgsz),
        float(cfg.detector.conf),
        float(cfg.detector.iou),
        str(cfg.detector.device),
        bool(cfg.detector.half),
        int(cfg.detector.max_det),
        cfg.detector.person_class,
    )
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = YoloDetector(cfg)
    return _DETECTOR_CACHE[key]


def get_embedder(cfg: Config, *, device: str | None = None, force_fallback: bool = False) -> Any:
    key = (cfg.embedder.model, int(cfg.embedder.imgsz), bool(cfg.embedder.fp16), device, bool(force_fallback))
    if key not in _EMBEDDER_CACHE:
        from tgtrack.appearance import Embedder

        _ort_once(device)
        _EMBEDDER_CACHE[key] = Embedder(cfg.embedder, device=device or "cuda", force_fallback=force_fallback)
    return _EMBEDDER_CACHE[key]


# --------------------------------------------------------------------------- #
# synthetic sequence (for --dry-run and the tests)
# --------------------------------------------------------------------------- #
def synthetic_sequence(
    n_frames: int = 24,
    *,
    width: int = 320,
    height: int = 240,
    name: str = "synth-stress",
    gap: tuple[int, int] | None = None,
) -> Sequence:
    """A deterministic 3-person clip that stresses exactly what we claim.

    * id 1 (the stressed target) walks right, is absent for ``gap`` frames and
      walks into the right border;
    * id 2 is a **similar-looking** person walking the other way and crossing the
      target (the false-acquisition decoy);
    * id 3 is an unrelated bystander standing still near the top-left.

    ``gap`` defaults to a 3-frame hole in the middle of the clip, so any
    ``n_frames >= 8`` still contains both a loss and a re-acquisition.

    Frames are in-memory ``uint8`` BGR arrays, so this needs no dataset, no
    decoder and no GPU.
    """
    # Tall enough that a half-masked target still clears the default
    # observability guard (>= 80 px of visible height), so the synthetic clip
    # exercises the *tracking* question rather than the "is the target even
    # visible?" question.
    bw, bh = 26, 170
    frames: list[np.ndarray] = []
    gt: dict[int, list[GtBox]] = {}
    if gap is None:
        g0 = max(1, int(n_frames) // 2)
        g1 = min(int(n_frames), g0 + 3)
    else:
        g0, g1 = int(gap[0]), int(gap[1])
    for i in range(int(n_frames)):
        img = np.full((height, width, 3), 38, np.uint8)
        img[:, :] = (38, 40, 44)
        boxes: dict[int, tuple[int, int]] = {}
        ty = (height - bh) // 2 + int(3 * math.sin(i / 4.0))
        if not (g0 <= i < g1):
            boxes[1] = (8 + 9 * i, ty)
        boxes[2] = (width - 40 - 9 * i, ty + 4)
        boxes[3] = (24, 12)
        for tid, (x, y) in boxes.items():
            x2, y2 = x + bw, min(height - 1, y + bh)
            if x2 <= 0 or x >= width - 1:
                continue
            x0c, x2c = max(0, x), min(width - 1, x2)
            colour = (196, 196, 200) if tid != 3 else (90, 190, 90)
            img[y:y2, x0c:x2c] = colour
            stripe = y + (int(0.30 * bh) if tid == 1 else int(0.36 * bh))
            img[stripe : stripe + 5, x0c:x2c] = (255, 255, 255) if tid != 3 else (30, 30, 30)
            rect = np.array([x0c, y, x2c, y2], np.float64)
            if _area(rect) <= 1.0:
                continue
            gt.setdefault(i, []).append(GtBox(frame=i, track_id=int(tid), xyxy=rect, conf=1.0, visibility=1.0))
        frames.append(img)
    return Sequence(name=name, frames=frames, gt=gt, fps=25.0, width=width, height=height)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def _status_idle(frame_idx: int) -> TargetStatus:
    """Placeholder status for frames processed before the click frame."""
    return TargetStatus(frame_idx=int(frame_idx), state="idle", xyxy=None, score=0.0)


def _embedder_info(embedder: Any) -> Any:
    """JSON-safe description of the embedder actually used by a run."""
    if embedder is None:
        return "disabled"
    info = getattr(embedder, "info", None)
    if callable(info):
        try:
            return info()
        except Exception as exc:  # pragma: no cover - diagnostics only
            return {"kind": type(embedder).__name__, "error": repr(exc)}
    return {"kind": type(embedder).__name__}


def _annotate(frame: np.ndarray, status: TargetStatus, gt_box: np.ndarray | None, label: str) -> np.ndarray:
    import cv2

    out = frame.copy()
    if gt_box is not None:
        x0, y0, x1, y1 = (int(round(float(v))) for v in gt_box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (60, 220, 60), 1)
    if status.xyxy is not None:
        x0, y0, x1, y1 = (int(round(float(v))) for v in status.xyxy)
        colour = (60, 60, 240) if status.is_reporting else (200, 160, 40)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(out, label[:60], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def classify_wrong_lock(
    rows: TSequence[dict[str, Any]],
    *,
    iou_thresh: float = 0.5,
    wrong_iou: float = WRONG_IDENTITY_IOU,
) -> dict[str, Any]:
    """Split the frames of a run into confident-wrong / coasting-right / coasting-wrong.

    A **confident** wrong lock is a frame in which

    * the tracker is *not* in a blind state -- it believes it is locked on, and
    * the box it reports overlaps a real ground-truth identity that is not the
      clicked target (IoU >= ``wrong_iou``).

    The bar is deliberately looser than the metric layer's ``false_acquisition``
    (IoU >= 0.5), because a wrong lock sitting at IoU 0.22 is a wrong lock even
    though the metric layer does not book it.  ``metric_locked_false_acquisition_frames``
    is the metric layer's own count on the same rows, so the report can state that
    the two attributions agree instead of asserting it.

    Frames in a blind state are split into ``coasting_correct_frames`` (still on
    the target -- the intended behaviour of short-term memory) and
    ``coasting_wrong_frames`` (drifted onto somebody else).  Conflating those two
    with a confident wrong lock is what made the earlier failure unattributable.
    """
    locked_wrong: list[int] = []
    locked_fa: list[int] = []
    locked_halluc: list[int] = []
    coast_ok: list[int] = []
    coast_wrong: list[int] = []
    for r in rows:
        if not bool(r.get("reported")):
            continue
        frame = int(r.get("frame", 0))
        locked = bool(r.get("locked"))
        best_id = r.get("best_gt_id")
        try:
            best_iou = float(r.get("best_gt_iou"))
        except (TypeError, ValueError):
            best_iou = 0.0
        tgt_id = r.get("target_person_id")
        other = best_id is not None and (tgt_id is None or int(best_id) != int(tgt_id))
        if locked:
            if bool(r.get("false_acquisition")):
                locked_fa.append(frame)
            if other and best_iou >= float(wrong_iou):
                locked_wrong.append(frame)
            if best_id is None:
                locked_halluc.append(frame)
        elif bool(r.get("correct")):
            coast_ok.append(frame)
        elif bool(r.get("false_acquisition")) or (other and best_iou >= float(wrong_iou)):
            coast_wrong.append(frame)
    return {
        "wrong_identity_locked": len(locked_wrong),
        "wrong_identity_locked_frames": locked_wrong,
        "wrong_identity_locked_first_frame": (locked_wrong[0] if locked_wrong else None),
        "wrong_identity_iou_bar": float(wrong_iou),
        "coasting_correct_frames": len(coast_ok),
        "coasting_wrong_frames": len(coast_wrong),
        "hallucination_locked_frames": len(locked_halluc),
        # the metric layer's attribution of the *same* rows: the run may not claim
        # a failure the metrics do not see, nor hide one they do
        "metric_locked_false_acquisition_frames": len(locked_fa),
    }


def run_sequence(
    seq: Sequence,
    target_id: int,
    cfg: Config,
    *,
    variant: str = "baseline",
    group: str = "baseline",
    mask: str = "none",
    max_frames: int | None = None,
    detector: Any | None = None,
    detector_kind: str = "gt",
    embedder: Any | None = None,
    use_embedder: bool = False,
    device: str | None = None,
    force_fallback: bool = False,
    debug_csv: str | Path | None = None,
    video_path: str | Path | None = None,
    margin: float = 0.02,
    edge_frac: float = 0.5,
    drop_below: float = 0.2,
    fill: Any = 0,
    quiet: bool = True,
) -> dict[str, Any]:
    """Track one clicked target through one sequence and score it.

    ``detector`` accepts any object with ``detect(frame_bgr, frame_idx)``); when
    it is ``None`` the oracle :class:`GtDetector` built from the mask plan is
    used.  A fresh :class:`TargetTracker` is constructed here, so runs cannot
    contaminate each other.
    """
    n = len(seq) if max_frames is None else max(0, min(int(max_frames), len(seq)))
    if n <= 0:
        raise ValueError(f"sequence {seq.name!r} has no frames to process")
    plan = build_mask_plan(
        seq, target_id, mask, limit=n, margin=margin, edge_frac=edge_frac, drop_below=drop_below, fill=fill
    )
    first = next((i for i in range(n) if plan.gt_target.get(i) is not None), None)
    if first is None:
        raise ValueError(f"target {target_id} is not visible in the first {n} frames of {seq.name!r}")

    if detector is None:
        detector = GtDetector(plan) if detector_kind != "yolo" else get_detector(cfg, "yolo")
    if embedder is None and use_embedder:
        try:
            embedder = get_embedder(cfg, device=device, force_fallback=force_fallback)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[embed] WARN unavailable ({type(exc).__name__}: {exc}); continuing without ReID", flush=True)
            embedder = None

    tracker = TargetTracker(cfg, embedder)

    click_box = plan.full_gt_target[first]
    if click_box is None:  # pragma: no cover - guarded by `first`
        raise ValueError("internal: no click box")
    click_xy = (int(round(0.5 * (click_box[0] + click_box[2]))), int(round(0.5 * (click_box[1] + click_box[3]))))
    clicked_id = simulate_click(seq, click_xy, first)

    statuses: dict[int, TargetStatus] = {i: _status_idle(i) for i in range(first)}
    times: list[float] = []
    # Detection coverage of the target's own (post-mask) GT box: the honest way to
    # tell "the tracker lost the target" apart from "the detector never saw it".
    target_frames = 0
    detected_frames = 0
    any_detection_frames = 0
    writer = None
    video_out: str | None = None
    t_start = time.perf_counter()

    for i in range(first, n):
        t0 = time.perf_counter()
        raw = seq.frame(i)
        frame = plan.apply(raw, i)
        if frame is None:
            frame = np.zeros((plan.frame_h, plan.frame_w, 3), np.uint8)
        dets = list(detector.detect(frame, i))
        if dets:
            any_detection_frames += 1
        gt_now = plan.gt_target.get(i)
        if gt_now is not None:
            target_frames += 1
            if any(_pair_iou(d.xyxy, gt_now) >= WRONG_IDENTITY_IOU for d in dets):
                detected_frames += 1
        if i == first:
            # A masked plan can put the target's visible rect entirely outside the
            # frame on the chosen init frame (`--mask upper` on MOT17-04 produced
            # y 1084..1188 on a 1080-px frame, which `clip_xyxy` collapses to a
            # zero-area box).  That is not a tracking failure, so do not crash on
            # it: fall forward to the first frame whose post-mask target box still
            # has area, and only give up if there is none.
            init_box = plan.gt_target[i]
            if init_box is None or float(init_box[2] - init_box[0]) <= 1.0 or float(init_box[3] - init_box[1]) <= 1.0:
                for j in range(i + 1, n):
                    cand = plan.gt_target.get(j)
                    if cand is not None and float(cand[2] - cand[0]) > 1.0 and float(cand[3] - cand[1]) > 1.0:
                        init_box = cand
                        i = first = j
                        break
                else:
                    raise RuntimeError(
                        f"no initializable target box in {seq.name} "
                        f"(mask={plan.mode}, target={plan.target_id})"
                    )
            status = tracker.initialize(frame, init_box, frame_idx=i)
        else:
            status = tracker.step(frame, dets, frame_idx=i)
        statuses[i] = status
        times.append((time.perf_counter() - t0) * 1000.0)

        if video_path is not None:
            import cv2

            canvas = _annotate(frame, status, plan.gt_target.get(i), f"{seq.name} f{i} {status.state}")
            if writer is None:
                h, w = canvas.shape[:2]
                vp = Path(video_path)
                # the outdir may not exist yet (outputs are written after the
                # grid), and cv2.VideoWriter silently fails on a missing parent
                vp.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(vp), cv2.VideoWriter_fourcc(*"mp4v"), float(seq.fps or 25.0), (w, h)
                )
                if not writer.isOpened():
                    print("[video] WARN cv2.VideoWriter could not open the output; skipping", flush=True)
                    writer = None
                    video_path = None
            if writer is not None:
                writer.write(canvas)
    if writer is not None:
        writer.release()
        video_out = str(video_path)

    events = list(tracker.events)
    metrics = evaluate_target(
        plan.gt_target,
        statuses,
        events,
        plan.all_gt,
        cfg=cfg.eval,
        debug_csv=debug_csv,
    )
    rows = target_frame_table(plan.gt_target, statuses, plan.all_gt, cfg=cfg.eval)
    wrong = classify_wrong_lock(rows, iou_thresh=float(cfg.eval.iou_thresh))
    vis_boxes = [
        (float(b[2] - b[0]), float(b[3] - b[1]))
        for i in range(n)
        if (b := plan.gt_target.get(i)) is not None
    ]
    run: dict[str, Any] = {
        "sequence": seq.name,
        "variant": variant,
        "group": group,
        "mask": mask,
        "target_id": int(target_id),
        "clicked_id": None if clicked_id is None else int(clicked_id),
        "click_xy": [int(click_xy[0]), int(click_xy[1])],
        "click_frame": int(first),
        "frames_processed": int(n - first),
        "visibility_mean": float(plan.visible_fraction()),
        "mask_delta": [int(v) for v in plan.delta],
        "ms_per_frame": float(np.mean(times)) if times else 0.0,
        "ms_p95": float(np.percentile(times, 95)) if times else 0.0,
        "wall_seconds": float(time.perf_counter() - t_start),
        "detector": detector.info() if hasattr(detector, "info") else type(detector).__name__,
        "embedder": _embedder_info(embedder),
        "tracker_stats": tracker.stats(),
        "n_events": len(events),
    }
    run.update(
        {
            # scenario class: overwritten by run_spec() with the observability
            # decision taken by pick_target(); a direct run_sequence() call has no
            # guard to report, hence guard_passed=None
            "control": CONTROL_PARTIAL_VIEW,
            "invisible_target": False,
            "target_reason": "direct",
            "guard_passed": None,
            "visible_median_w": float(np.median([w for w, _ in vis_boxes])) if vis_boxes else 0.0,
            "visible_median_h": float(np.median([h for _, h in vis_boxes])) if vis_boxes else 0.0,
            "visible_median_area": float(np.median([w * h for w, h in vis_boxes])) if vis_boxes else 0.0,
            "visible_frames": len(vis_boxes),
            "detected_frames": int(detected_frames),
            "detection_rate": (
                float(detected_frames) / float(target_frames) if target_frames else float("nan")
            ),
            "detector_frames": int(any_detection_frames),
            "frame_size": [int(plan.frame_w), int(plan.frame_h)],
        }
    )
    run.update(wrong)
    # the run's own attribution of the wrong lock must equal the metric layer's:
    # if it does not, one of the two is lying about the same frames
    run["onset_metric_agrees"] = bool(
        int(wrong["metric_locked_false_acquisition_frames"])
        == int(metrics.get("false_acquisition_frames_locked", -1))
    )
    if mask != "none" and plan.full_gt_target:
        # secondary reading: same run scored against the untouched full-body box
        try:
            alt = evaluate_target(plan.full_gt_target, statuses, events, seq.gt, cfg=cfg.eval)
            run["target_success_rate_fullgt"] = float(alt["target_success_rate"])
            run["target_precision_fullgt"] = float(alt["target_precision"])
        except Exception as exc:  # pragma: no cover - defensive
            run["target_success_rate_fullgt"] = float("nan")
            run["fullgt_error"] = repr(exc)
    else:
        run["target_success_rate_fullgt"] = float(run_metrics_tsr(metrics))
    run.update({k: v for k, v in metrics.items() if k not in run})
    run["metrics"] = metrics
    if video_out:
        run["video"] = video_out
    if debug_csv:
        run["debug_csv"] = str(debug_csv)
    if not quiet:
        print(
            "[run] %s %s mask=%s TSR=%.3f FAR=%.4f FAev=%d ms=%.1f"
            % (
                seq.name,
                variant,
                mask,
                float(run.get("target_success_rate", float("nan"))),
                float(run.get("false_acquisition_rate", float("nan"))),
                int(run.get("false_acquisition_events", 0)),
                float(run["ms_per_frame"]),
            ),
            flush=True,
        )
    return run


def run_metrics_tsr(metrics: dict[str, Any]) -> float:
    return float(metrics.get("target_success_rate", float("nan")))


# --------------------------------------------------------------------------- #
# grid execution (parallel)
# --------------------------------------------------------------------------- #
def run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one (sequence, variant, mask) spec.  Top-level so it pickles."""
    tag = {
        "sequence": spec.get("seq_name"),
        "variant": spec.get("variant"),
        "group": spec.get("group"),
        "mask": spec.get("mask"),
    }
    try:
        cfg = load_config(spec.get("config_path") or None)
        overrides = dict(spec.get("overrides") or {})
        if overrides:
            cfg = cfg.clone(**overrides)
        opts = dict(spec.get("opts") or {})
        device = opts.get("device")
        if device:
            cfg.detector.device = str(device)
        if opts.get("dry_run"):
            seq = synthetic_sequence(int(opts.get("synth_frames") or 24))
        else:
            seq = load_mot_sequence(
                spec["seq_dir"],
                subsample=int(opts.get("subsample") or 1),
                max_frames=opts.get("max_frames"),
                min_visible=float(opts.get("min_visible") or 0.0),
                min_conf=float(opts.get("min_conf") or 1.0),
            )
        run = run_sequence(
            seq,
            int(spec["target_id"]),
            cfg,
            variant=str(spec["variant"]),
            group=str(spec["group"]),
            mask=str(spec["mask"]),
            max_frames=opts.get("max_frames"),
            detector_kind=str(opts.get("detector") or "yolo"),
            use_embedder=bool(opts.get("use_embedder")),
            device=None if not device else str(device),
            force_fallback=bool(opts.get("embed_fallback")),
            debug_csv=opts.get("debug_csv"),
            video_path=opts.get("video_path"),
            margin=float(opts.get("margin", 0.02)),
            edge_frac=float(opts.get("edge_frac", 0.5)),
            drop_below=float(opts.get("drop_below", 0.2)),
            fill=opts.get("fill", 0),
            quiet=bool(opts.get("quiet", True)),
        )
        run["seq_dir"] = spec.get("seq_dir")
        dec = dict(spec.get("target") or {})
        if dec:
            sel = dict(dec.get("selected") or {})
            run["target_reason"] = str(dec.get("reason") or run.get("target_reason", ""))
            run["control"] = str(dec.get("control") or CONTROL_PARTIAL_VIEW)
            run["invisible_target"] = bool(dec.get("invisible"))
            run["guard_passed"] = bool(dec.get("guard_passed"))
            run["guard"] = dict(dec.get("guard") or {})
            # the guard's own numbers, kept beside the measured ones so a
            # disagreement between the two is visible instead of silent
            run["guard_visible_median_h"] = float(sel.get("median_h", 0.0))
            run["guard_visible_median_area"] = float(sel.get("median_area", 0.0))
            run["guard_visible_frames"] = int(sel.get("frames", 0))
            run["guard_stats_agree"] = bool(
                abs(float(run.get("visible_median_h", 0.0)) - float(sel.get("median_h", 0.0))) < 1e-6
                and int(run.get("visible_frames", 0)) == int(sel.get("frames", 0))
            )
            run["target_message"] = str(dec.get("message") or "")
        return run
    except Exception as exc:
        tag["error"] = f"{type(exc).__name__}: {exc}"
        return tag


def run_grid(specs: TSequence[dict[str, Any]], *, jobs: int = 1, quiet: bool = False) -> list[dict[str, Any]]:
    """Run every spec, serially or in a spawn-based process pool."""
    if jobs is None or int(jobs) <= 1 or len(specs) <= 1:
        return [run_spec(s) for s in specs]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=ctx) as pool:
        for res in pool.map(run_spec, list(specs)):
            out.append(res)
            if not quiet:
                print(
                    "[done] %s %s mask=%s %s"
                    % (
                        res.get("sequence"),
                        res.get("variant"),
                        res.get("mask"),
                        res.get("error") or ("TSR=%.3f" % float(res.get("target_success_rate", float("nan")))),
                    ),
                    flush=True,
                )
    return out


# --------------------------------------------------------------------------- #
# pooling + output
# --------------------------------------------------------------------------- #
def pool_runs(
    runs: TSequence[dict[str, Any]],
    *,
    control: str | None = CONTROL_PARTIAL_VIEW,
) -> list[dict[str, Any]]:
    """Frame-weighted pooling of successful runs per (variant, mask).

    ``control`` selects *which scenario class* is pooled.  The default is
    ``CONTROL_PARTIAL_VIEW``: a run whose target was not observable under the
    requested mask measures the absence of a target, not a partial-view strategy,
    and must never enter the headline.  Pass ``control=CONTROL_INVISIBLE`` for the
    unobservable-target control table, or ``control=None`` to pool everything
    (used by tests and by callers that already partitioned their runs).
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        if r.get("error") or "metrics" not in r:
            continue
        if control is not None and str(r.get("control", CONTROL_PARTIAL_VIEW)) != str(control):
            continue
        buckets.setdefault((str(r["variant"]), str(r["mask"])), []).append(r)
    pooled: list[dict[str, Any]] = []
    for (variant, mask), group in buckets.items():
        metrics = [r["metrics"] for r in group]
        row = combine_target(metrics) if metrics else {}
        row = dict(row)
        row["variant"] = variant
        row["group"] = str(group[0].get("group", ""))
        row["mask"] = mask
        row["n_sequences"] = len(group)
        row["control"] = str(control) if control is not None else "all"
        if str(control) == CONTROL_INVISIBLE:
            # the control pool must be self-describing: these are runs whose target
            # could not be observed, not partial-view results
            row["invisible_target"] = True
        row["ms_per_frame"] = float(np.mean([float(r.get("ms_per_frame", 0.0)) for r in group]))
        row["visibility_mean"] = float(np.mean([float(r.get("visibility_mean", 1.0)) for r in group]))
        row["longest_correct_streak"] = int(max((int(r.get("longest_correct_streak", 0)) for r in group), default=0))
        row["mean_correct_streak"] = float(
            np.mean([float(r.get("mean_correct_streak", 0.0)) for r in group])
        )
        row["target_success_rate_fullgt"] = float(
            np.mean([float(r.get("target_success_rate_fullgt", float("nan"))) for r in group])
        )
        p90s = [float(r["reacquire_latency_p90"]) for r in group if _finite(r.get("reacquire_latency_p90"))]
        row["reacquire_latency_p90"] = float(np.mean(p90s)) if p90s else float("nan")
        maxima = [float(r["reacquire_latency_max"]) for r in group if _finite(r.get("reacquire_latency_max"))]
        row["reacquire_latency_max"] = max(maxima) if maxima else float("nan")
        row["sequences"] = sorted(str(r.get("sequence")) for r in group)
        pooled.append(row)
    pooled.sort(key=lambda r: (str(r["group"]), str(r["variant"]), str(r["mask"])))
    return pooled


def _finite(value: Any) -> bool:
    """True when ``value`` coerces to a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    if spec == "s":
        return str(value)
    if spec == "d":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "-"


def _delta(row: dict[str, Any], base: dict[str, Any] | None, key: str) -> float:
    if base is None:
        return float("nan")
    a = row.get(key)
    b = base.get(key)
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return float("nan")


def claims_lines(pooled: TSequence[dict[str, Any]], baseline: str = "baseline") -> list[str]:
    """Human-readable deltas vs the baseline, per mask (the report's headline)."""
    index = {(str(r["variant"]), str(r["mask"])): r for r in pooled}
    lines: list[str] = []
    for r in pooled:
        if str(r["variant"]) == baseline:
            continue
        base = index.get((baseline, str(r["mask"])))
        if base is None:
            continue
        lines.append(
            "- [%s] %s (mask=%s): TSR %.4f vs %.4f (%+.4f) | FAR %.5f vs %.5f (%+.5f) | "
            "FA events %d vs %d (%+d) | precision %.4f vs %.4f | ms/frame %.2f vs %.2f (%+.2f)"
            % (
                r["group"],
                r["variant"],
                r["mask"],
                float(r.get("target_success_rate", float("nan"))),
                float(base.get("target_success_rate", float("nan"))),
                _delta(r, base, "target_success_rate"),
                float(r.get("false_acquisition_rate", float("nan"))),
                float(base.get("false_acquisition_rate", float("nan"))),
                _delta(r, base, "false_acquisition_rate"),
                int(r.get("false_acquisition_events", 0)),
                int(base.get("false_acquisition_events", 0)),
                int(r.get("false_acquisition_events", 0)) - int(base.get("false_acquisition_events", 0)),
                float(r.get("target_precision", float("nan"))),
                float(base.get("target_precision", float("nan"))),
                float(r.get("ms_per_frame", 0.0)),
                float(base.get("ms_per_frame", 0.0)),
                _delta(r, base, "ms_per_frame"),
            )
        )
    return lines


def partial_view_table(pooled: TSequence[dict[str, Any]]) -> list[str]:
    """TSR/FAR cross-tab of the part mechanism per mask (task-7's core claim)."""
    index = {(str(r["variant"]), str(r["mask"])): r for r in pooled}
    masks = sorted({str(r["mask"]) for r in pooled})
    rows = ["| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in masks:
        on = index.get(("baseline", m))
        off = index.get(("part_matching_off", m))
        if on is None or off is None:
            continue
        rows.append(
            "| %s | %.4f | %.4f | %+.4f | %.5f | %.5f | %+.5f | %.2f | %.2f |"
            % (
                m,
                float(on.get("target_success_rate", float("nan"))),
                float(off.get("target_success_rate", float("nan"))),
                _delta(off, on, "target_success_rate"),
                float(on.get("false_acquisition_rate", float("nan"))),
                float(off.get("false_acquisition_rate", float("nan"))),
                _delta(off, on, "false_acquisition_rate"),
                float(on.get("ms_per_frame", 0.0)),
                float(off.get("ms_per_frame", 0.0)),
            )
        )
    return rows


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_outputs(
    runs: TSequence[dict[str, Any]],
    pooled: TSequence[dict[str, Any]],
    outdir: str | Path,
    *,
    meta: dict[str, Any] | None = None,
    invisible: TSequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Write ``per_run.json`` + ``summary.json`` + ``table.csv`` + ``table.md``.

    ``pooled`` is the headline pool (observable targets only); ``invisible`` is the
    unobservable-target control pool, kept out of the headline table and printed in
    its own section.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    errors = [r for r in runs if r.get("error")]
    payload = {
        "meta": dict(meta or {}),
        "n_runs": len(runs),
        "n_errors": len(errors),
        "runs": [_jsonable(r) for r in runs],
        "pooled": [_jsonable(r) for r in pooled],
    }
    per_run = out / "per_run.json"
    per_run.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = [c[0] for c in TABLE_COLUMNS]
    csv_lines = [",".join(["group", "variant", "mask"] + header[3:])]
    for r in pooled:
        cells = []
        for _, key, spec in TABLE_COLUMNS[3:]:
            v = r.get(key)
            cells.append("" if v is None or (isinstance(v, float) and math.isnan(v)) else _fmt(v, spec))
        csv_lines.append(",".join([str(r.get("group", "")), str(r.get("variant", "")), str(r.get("mask", ""))] + cells))
    table_csv = out / "table.csv"
    table_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md: list[str] = ["# Clicked-target strategy ablation", ""]
    if meta:
        md.append("Config: " + ", ".join(f"{k}={v}" for k, v in sorted(meta.items()) if k != "variants") + "")
        md.append("")
    md.append("## Pooled results (frame-weighted over sequences)")
    md.append("")
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for r in pooled:
        md.append("| " + " | ".join(_fmt(r.get(k), s) for _, k, s in TABLE_COLUMNS) + " |")
    md.append("")
    guard = dict((meta or {}).get("guard") or {})
    if guard:
        md.append(
            "Observability guard: the target must keep a median post-mask visible box of at least %s px of "
            "height and %s px^2 under **every** requested mask; runs that fail it are excluded from every "
            "table above and listed in the control section below."
            % (_fmt(guard.get("min_visible_h"), ".0f"), _fmt(guard.get("min_visible_area"), ".0f"))
        )
        md.append("")
    md.append(
        "Attribution note: coast_ok counts frames in a blind state that are still on the target (success by "
        "inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box "
        "overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at "
        "least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by "
        "worst_id and not by FAR -- read the two columns together."
    )
    md.append("")
    md.append("## Invisible-target control (excluded from the headline)")
    md.append("")
    if invisible:
        cols = (
            ("group", "group", "s"),
            ("variant", "variant", "s"),
            ("mask", "mask", "s"),
            ("seqs", "n_sequences", "d"),
            ("frames", "n_frames", "d"),
            ("visible_h", "visible_median_h", ".1f"),
            ("visible_area", "visible_median_area", ".0f"),
            ("TSR", "target_success_rate", ".4f"),
            ("FAR", "false_acquisition_rate", ".5f"),
        )
        md.append("| " + " | ".join(c[0] for c in cols) + " |")
        md.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in invisible:
            md.append("| " + " | ".join(_fmt(r.get(k), s) for _, k, s in cols) + " |")
        md.append("")
        md.append(
            "These runs are not evidence about partial-view tracking: the chosen target was too small (or "
            "absent) to be observed, so the numbers measure the absence of a target, not a strategy."
        )
    else:
        md.append("(no unobservable-target run)")
    md.append("")
    decisions = dict((meta or {}).get("target_decisions") or {})
    if decisions:
        md.append("## Target selection (per sequence)")
        md.append("")
        md.append("| sequence | target | reason | control | worst mask | visible median | frames |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name in sorted(decisions):
            d = dict(decisions[name] or {})
            sel = dict(d.get("selected") or {})
            md.append(
                "| %s | %s | %s | %s | %s | %sx%s | %s |"
                % (
                    name,
                    _fmt(d.get("target_id"), "d"),
                    _fmt(d.get("reason"), "s"),
                    _fmt(d.get("control"), "s"),
                    _fmt(sel.get("worst_mask"), "s"),
                    _fmt(sel.get("median_w"), ".0f"),
                    _fmt(sel.get("median_h"), ".0f"),
                    _fmt(sel.get("frames"), "d"),
                )
            )
        md.append("")
        for name in sorted(decisions):
            msg = str((decisions[name] or {}).get("message") or "")
            if msg:
                md.append("- %s: %s" % (name, msg))
        md.append("")
    md.append("## Partial-view cross-tab (part matching ON vs OFF)")
    md.append("")
    md.extend(partial_view_table(pooled) or ["(part_matching group not selected)"])
    md.append("")
    md.append("## Claims (delta vs the same-mask baseline)")
    md.append("")
    md.extend(claims_lines(pooled) or ["(only the baseline was run)"])
    md.append("")
    md.append("## Per-run detail")
    md.append("")
    detail_cols = (
        ("sequence", "s"),
        ("variant", "s"),
        ("mask", "s"),
        ("target_id", "d"),
        ("frames_processed", "d"),
        ("target_success_rate", ".4f"),
        ("target_precision", ".4f"),
        ("false_acquisition_rate", ".5f"),
        ("false_acquisition_events", "d"),
        ("id_consistency", ".4f"),
        ("target_switch_count", "d"),
        ("ms_per_frame", ".2f"),
    )
    md.append("| " + " | ".join(k for k, _ in detail_cols) + " |")
    md.append("| " + " | ".join("---" for _ in detail_cols) + " |")
    for r in runs:
        if r.get("error"):
            md.append("| %s | %s | %s | error: %s |" % (r.get("sequence"), r.get("variant"), r.get("mask"), r["error"]))
            continue
        md.append("| " + " | ".join(_fmt(r.get(k), s) for k, s in detail_cols) + " |")
    if errors:
        md.append("")
        md.append("## Errors")
        md.append("")
        for r in errors:
            md.append("- %s / %s / %s: %s" % (r.get("sequence"), r.get("variant"), r.get("mask"), r["error"]))
    summary = {
        "meta": dict(meta or {}),
        "n_runs": len(runs),
        "n_errors": len(errors),
        "n_headline_runs": len(
            [
                r
                for r in runs
                if not r.get("error")
                and str(r.get("control", CONTROL_PARTIAL_VIEW)) == CONTROL_PARTIAL_VIEW
            ]
        ),
        "n_invisible_runs": len(
            [
                r
                for r in runs
                if not r.get("error") and str(r.get("control")) == CONTROL_INVISIBLE
            ]
        ),
        "headline": [_jsonable(r) for r in pooled],
        "control": [_jsonable(r) for r in invisible],
        "targets": {
            str(r.get("sequence")): {
                "target_id": r.get("target_id"),
                "reason": r.get("target_reason"),
                "control": r.get("control", CONTROL_PARTIAL_VIEW),
                "guard_passed": r.get("guard_passed"),
                "visible_median_w": r.get("visible_median_w"),
                "visible_median_h": r.get("visible_median_h"),
                "visible_median_area": r.get("visible_median_area"),
                "detection_rate": r.get("detection_rate"),
                "wrong_identity_locked": r.get("wrong_identity_locked"),
                "coasting_correct_frames": r.get("coasting_correct_frames"),
                "onsets_agree": r.get("onset_metric_agrees"),
                "message": r.get("target_message"),
            }
            for r in runs
            if not r.get("error")
        },
    }
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    table_md = out / "table.md"
    table_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "per_run.json": per_run,
        "summary.json": summary_json,
        "table.csv": table_csv,
        "table.md": table_md,
        "n_runs": len(runs),
        "n_errors": len(errors),
        "n_headline_runs": summary["n_headline_runs"],
        "n_invisible_runs": summary["n_invisible_runs"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_target.py",
        description="Clicked-target end-to-end harness + strategy ablation grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", default="data/sequences", help="root holding MOT sequences (discover_sequences)")
    p.add_argument("--out", default="results/target_eval", help="output directory")
    p.add_argument("--max-frames", type=int, default=None, help="process at most N frames per sequence")
    p.add_argument("--subsample", type=int, default=1, help="keep every K-th frame (loader renumbers to 0..n-1)")
    p.add_argument("--jobs", type=int, default=1, help="process pool over (sequence, variant, mask)")
    p.add_argument("--ablate", default="all", help=f"comma list of groups, or 'all'. One of: {','.join(GROUPS)}")
    p.add_argument("--limit-sequences", type=int, default=None, help="only the first N sequences")
    p.add_argument("--mask", default="none", help=f"comma list of {','.join(MASK_MODES)}")
    p.add_argument("--margin", type=float, default=0.02, help="mask pad as a fraction of the target box")
    p.add_argument("--edge-frac", type=float, default=0.5, help="edge mask: fraction of the target width pushed off-screen")
    p.add_argument("--drop-below", type=float, default=0.2, help="drop other people hidden below this visibility")
    p.add_argument("--mask-fill", default="0", help="occluder colour: '0' (black) or 'border'")
    p.add_argument("--target-id", type=int, default=None, help="force the clicked identity (default: auto-pick)")
    p.add_argument("--target-strategy", default="stressed", choices=list(TARGET_STRATEGIES))
    p.add_argument(
        "--min-visible-h",
        type=float,
        default=DEFAULT_MIN_VISIBLE_H,
        help="observability guard: minimum median post-mask visible height (px) of the chosen target",
    )
    p.add_argument(
        "--min-visible-area",
        type=float,
        default=DEFAULT_MIN_VISIBLE_AREA,
        help="observability guard: minimum median post-mask visible area (px^2); 0 disables the area test",
    )
    p.add_argument(
        "--allow-invisible-target",
        action="store_true",
        help="keep an unobservable target as a labelled control run instead of failing",
    )
    p.add_argument(
        "--observability",
        action="store_true",
        help="print the per-identity post-mask observability table for the requested masks and exit",
    )
    p.add_argument("--detector", default="yolo", choices=("yolo", "gt"), help="real YOLO26 detector or the GT oracle")
    p.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (also gates ensure_ort_cuda)")
    p.add_argument("--config", default=None, help="optional YAML/JSON config (Config schema)")
    p.add_argument("--min-visible", type=float, default=0.0, help="loader: drop GT boxes below this visibility")
    p.add_argument("--min-conf", type=float, default=1.0, help="loader: drop GT boxes below this confidence")
    p.add_argument("--no-embed", action="store_true", help="skip the ReID embedder (motion-only association)")
    p.add_argument("--embed-fallback", action="store_true", help="force the dependency-free descriptor")
    p.add_argument("--debug-csv", action="store_true", help="write per-frame dumps via evaluate_target(debug_csv=...)")
    p.add_argument("--save-video", action="store_true", help="write an annotated mp4 per run (mp4v, no ffmpeg)")
    p.add_argument("--dry-run", action="store_true", help="use the built-in synthetic sequence instead of a dataset")
    p.add_argument("--synth-frames", type=int, default=24, help="--dry-run: frames of the synthetic sequence")
    p.add_argument("--list-variants", action="store_true", help="print the ablation grid and exit")
    p.add_argument("--quiet", action="store_true", help="only the final summary line")
    return p


def _parse_masks(text: str) -> list[str]:
    modes = [m.strip() for m in str(text).split(",") if m.strip()]
    unknown = [m for m in modes if m not in MASK_MODES]
    if unknown:
        raise ValueError(f"unknown mask mode(s) {unknown}; known: {list(MASK_MODES)}")
    return modes or ["none"]


def _load_sequences(args: argparse.Namespace) -> list[Sequence]:
    if args.dry_run:
        return [synthetic_sequence(int(args.synth_frames))]
    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"--data-root not found: {root} (use --dry-run for the synthetic sequence)")
    dirs = discover_sequences(root)
    if args.limit_sequences:
        dirs = dirs[: int(args.limit_sequences)]
    seqs: list[Sequence] = []
    for d in dirs:
        try:
            seqs.append(
                load_mot_sequence(
                    d,
                    subsample=int(args.subsample),
                    max_frames=args.max_frames,
                    min_visible=float(args.min_visible),
                    min_conf=float(args.min_conf),
                )
            )
        except Exception as exc:
            print(f"[load] WARN skipping {d}: {type(exc).__name__}: {exc}", flush=True)
    return seqs


def main(argv: TSequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    variants = build_variants(args.ablate)
    if args.list_variants:
        print(variant_table(variants))
        return 0
    masks = _parse_masks(args.mask)
    fill: Any = 0 if str(args.mask_fill) in ("0", "black") else args.mask_fill
    outdir = Path(args.out)
    if args.no_embed:
        print(
            "[warn] --no-embed: without appearance evidence association is rejected after the seed frame "
            "(cost > accept_cost), so the run only exercises detection-level plumbing. Use --embed-fallback "
            "for a GPU-free run that still associates.",
            file=sys.stderr,
            flush=True,
        )

    dir_map: dict[str, str] = {}
    if not args.dry_run:
        try:
            dir_map = {Path(d).name: str(d) for d in discover_sequences(Path(args.data_root))}
        except Exception as exc:
            print(f"[discover] WARN {type(exc).__name__}: {exc}", flush=True)

    try:
        seqs = _load_sequences(args)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    if not seqs:
        print(f"[error] no sequences found under {args.data_root}", file=sys.stderr, flush=True)
        return 2
    if args.observability:
        for seq in seqs:
            obs = observability_table(
                seq,
                masks,
                limit=args.max_frames,
                margin=float(args.margin),
                edge_frac=float(args.edge_frac),
                drop_below=float(args.drop_below),
                fill=fill,
            )
            print("[obs] %s: %d frames, %d identities, mask(s)=%s" % (seq.name, len(seq), len(obs), ",".join(masks)), flush=True)
            for tid in sorted(obs):
                rec = obs[tid]
                print(
                    "  [obs] id=%-4d frames=%-5d median=%.0fx%.0f area=%.0f worst_mask=%s"
                    % (
                        tid,
                        rec["frames"],
                        rec["median_w"],
                        rec["median_h"],
                        rec["median_area"],
                        rec["worst_mask"],
                    ),
                    flush=True,
                )
                for m in sorted(rec["masks"]):
                    st = rec["masks"][m]
                    print(
                        "        mask=%-6s frames=%-5d median=%.0fx%.0f area=%-9.0f visible=%.2f"
                        % (m, st["frames"], st["median_w"], st["median_h"], st["median_area"], st["visible_fraction"]),
                        flush=True,
                    )
        return 0

    specs: list[dict[str, Any]] = []
    picks: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    blocked: list[str] = []
    for seq in seqs:
        dec = pick_target(
            seq,
            args.target_strategy,
            masks=masks,
            min_visible_h=float(args.min_visible_h),
            min_visible_area=float(args.min_visible_area),
            allow_invisible=bool(args.allow_invisible_target),
            forced_id=args.target_id,
            limit=args.max_frames,
            margin=float(args.margin),
            edge_frac=float(args.edge_frac),
            drop_below=float(args.drop_below),
            fill=fill,
        )
        decisions[seq.name] = dict(dec)
        tid = dec.get("target_id")
        if tid is None:
            if dec.get("reason") == "no_candidate":
                blocked.append(str(dec.get("message") or seq.name))
                print("[guard] %s" % dec.get("message"), file=sys.stderr, flush=True)
            else:
                print("[pick] WARN %s: %s" % (seq.name, dec.get("message") or dec.get("reason")), flush=True)
            continue
        picks[seq.name] = int(tid)
        if dec.get("invisible"):
            print("[guard] control run: %s" % dec.get("message"), file=sys.stderr, flush=True)
        for v in variants:
            for m in masks:
                tag = f"{seq.name}__{v.name}__{m}"
                debug_csv = str(outdir / "runs" / f"{tag}.csv") if args.debug_csv else None
                video_path = str(outdir / "runs" / f"{tag}.mp4") if args.save_video else None
                specs.append(
                    {
                        "seq_dir": None if args.dry_run else dir_map.get(seq.name, seq.name),
                        "seq_name": seq.name,
                        "target_id": int(tid),
                        "target": dec,
                        "variant": v.name,
                        "group": v.group,
                        "overrides": dict(v.overrides),
                        "mask": m,
                        "config_path": args.config,
                        "opts": {
                            "dry_run": bool(args.dry_run),
                            "synth_frames": int(args.synth_frames),
                            "max_frames": args.max_frames,
                            "subsample": int(args.subsample),
                            "min_visible": float(args.min_visible),
                            "min_conf": float(args.min_conf),
                            "detector": args.detector,
                            "device": args.device,
                            "use_embedder": not args.no_embed,
                            "embed_fallback": bool(args.embed_fallback),
                            "debug_csv": debug_csv,
                            "video_path": video_path,
                            "quiet": True,
                            "margin": float(args.margin),
                            "edge_frac": float(args.edge_frac),
                            "drop_below": float(args.drop_below),
                            "fill": fill,
                        },
                    }
                )
    if blocked:
        print(
            "[error] %d sequence(s) had no target observable under mask=%s. Nothing was run: a table computed "
            "on an invisible target looks like a strategy failure but is not one. Adjust "
            "--min-visible-h/--min-visible-area, name another identity with --target-id, or pass "
            "--allow-invisible-target to run the labelled control instead."
            % (len(blocked), ",".join(masks)),
            file=sys.stderr,
            flush=True,
        )
        return 3
    if not specs:
        print("[error] nothing to run (no target identity selected)", file=sys.stderr, flush=True)
        return 2

    t0 = time.perf_counter()
    if not args.quiet:
        print(f"[grid] {len(specs)} runs over {len(seqs)} sequence(s), jobs={args.jobs}", flush=True)
    runs = run_grid(specs, jobs=int(args.jobs), quiet=bool(args.quiet))
    wall = time.perf_counter() - t0
    # headline = observable-target runs only; the unobservable-target control is
    # pooled separately so it can never inflate or depress the headline
    pooled = pool_runs(runs)
    control_pooled = pool_runs(runs, control=CONTROL_INVISIBLE)

    # Label the descriptor the runs *actually* used.  A fallback (non-neural)
    # descriptor must never be reported as a real ReID model: the headline
    # table would otherwise claim appearance quality it does not have.
    embedder_label = "disabled" if args.no_embed else ("fallback" if args.embed_fallback else "real")
    for r in runs:
        info = r.get("embedder")
        if isinstance(info, dict) and "is_real" in info:
            embedder_label = "real" if info.get("is_real") else "fallback"
            break

    meta = {
        "data_root": args.data_root,
        "dry_run": bool(args.dry_run),
        "detector": args.detector,
        "embedder": embedder_label,
        "masks": masks,
        "ablate": args.ablate,
        "jobs": int(args.jobs),
        "subsample": int(args.subsample),
        "max_frames": args.max_frames,
        "target_strategy": args.target_strategy,
        "target_ids": picks,
        "variants": [v.name for v in variants],
        # task-10 audit trail: what the guard demanded, what each sequence offered,
        # and which identity was finally clicked
        "guard": {
            "min_visible_h": float(args.min_visible_h),
            "min_visible_area": float(args.min_visible_area),
            "masks": list(masks),
        },
        "allow_invisible_target": bool(args.allow_invisible_target),
        "target_decisions": decisions,
    }
    written = write_outputs(
        runs, pooled, outdir, meta=meta | {"wall_seconds": round(wall, 2)}, invisible=control_pooled
    )

    ok = [r for r in runs if not r.get("error")]
    print(
        "[result] runs=%d ok=%d errors=%d wall=%.1fs jobs=%d out=%s"
        % (len(runs), len(ok), len(runs) - len(ok), wall, int(args.jobs), outdir),
        flush=True,
    )
    for row in pooled:
        print(
            "[pool] %-20s mask=%-6s TSR=%.4f prec=%.4f FAR=%.5f FAev=%-3d idcons=%.4f ms=%.1f"
            % (
                row["variant"],
                row["mask"],
                float(row.get("target_success_rate", float("nan"))),
                float(row.get("target_precision", float("nan"))),
                float(row.get("false_acquisition_rate", float("nan"))),
                int(row.get("false_acquisition_events", 0)),
                float(row.get("id_consistency", float("nan"))),
                float(row.get("ms_per_frame", 0.0)),
            ),
            flush=True,
        )
    n_control = int(written.get("n_invisible_runs", 0))
    if n_control:
        print(
            "[control] %d run(s) measured an UNOBSERVABLE target and are excluded from the headline above "
            "(control table in table.md / summary.json)" % n_control,
            flush=True,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
