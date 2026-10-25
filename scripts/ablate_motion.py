#!/usr/bin/env python
"""Motion-strategy ablation harness for the clicked-target tracker.

WHAT THIS ANSWERS
-----------------
"Is IMM worth it over a plain Kalman filter for a click-to-track system, and
under exactly which motion conditions does Kalman rollback help?"

``MotionModel`` has already been measured *in isolation* on synthetic
trajectories (IMM beats KF by ~25% on circles and ~61% on constant-acceleration
paths).  That is a statement about a filter, not about a system.  This harness
closes that gap: it drives the real
:class:`tgtrack.target_tracker.TargetTracker` as a black box over deterministic
synthetic scenarios and scores every run with
:func:`tgtrack.evaluate.evaluate_target`, so the reported numbers are
target-centric *system* outcomes (success rate, re-acquisition latency, false
acquisitions), not filter residuals.

APPEARANCE IS HELD FIXED
------------------------
Motion must be the only thing that varies, so the ReID encoder is replaced by a
deterministic stub.  Every person gets a fixed L2-normalised vector, and the
stub returns the vector of the person whose ground-truth box best matches the
query crop.  No ONNX session, no GPU, no randomness: two runs of this script
agree exactly.  Two stub modes matter here:

* one shared vector for everybody -> the *identical lookalike* case.  The
  appearance channel then carries ZERO identity information, which is the
  hardest possible input for the tracker's ``distractor_margin`` refusal rule.
* ``IDENT_A`` for the target / ``IDENT_B`` for the bystander -> a genuinely
  different person, which appearance alone can reject.  This is the harness' own
  sensitivity control: same geometry, opposite appearance, and the outcome must
  differ.

The stub is deliberately *not* the same helper as ``tests/test_target_tracker.py``'s
``StubEmbedder``.  That one keys identity on ``x < split``, which cannot express
either "two different people with one appearance" or "two different appearances
at one location" -- and scenario 4 needs both.

All scenarios move inside a 6000x720 synthetic frame so that no run is disturbed
by the tracker's out-of-frame state machine.  The frame is never rendered
(``np.zeros``); only its shape is read.  Detections and ground truth are
metadata.

USAGE
-----
    python scripts/ablate_motion.py                 # full sweep + results/
    python scripts/ablate_motion.py --quick         # smoke run
    python scripts/ablate_motion.py --no-write      # print findings only

Outputs ``results/motion_ablation/{raw.json,table.md,table.csv}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tgtrack.config import Config  # noqa: E402
from tgtrack.evaluate import GtBox, combine_target, evaluate_target  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402
from tgtrack.types import Detection  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
#: Synthetic frame size.  Wide on purpose: an 80 px/frame target must never be
#: pushed out of frame, because OUT_OF_FRAME is a different experiment.
FRAME_H = 720
FRAME_W = 6000
FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

BOX_W, BOX_H = 120.0, 240.0
CY = 300.0
TARGET_ID, DISTRACTOR_ID = 1, 2

IDENT_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
IDENT_B = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

DEFAULT_OUTDIR = REPO_ROOT / "results" / "motion_ablation"

FAMILY_NOTES: dict[str, str] = {
    "fast_linear": "Constant-velocity walk at `step` px/frame with a brief detector drop "
                   "(frames 20-27): the target stays visible in GT but is not detected.",
    "nonlinear": "Circles and a figure-eight at several radii/speeds -- the trajectories on "
                 "which a constant-velocity filter is provably mis-specified.",
    "occlusion": "The target leaves the scene for `gap` frames and comes back, either on its "
                 "own trajectory (`inplace`) or at a distant location (`distant`).",
    "distractor": "An equal-size person crosses the target's path.  `lookalike` gives both the "
                  "same embedding (appearance cannot help); `different` is the control.",
    "nonlinear_blackout": "**The decisive experiment.**  The target follows a curving path while the "
                          "detector is blind for `gap` frames.  The target is still in GT (so those "
                          "frames count as visible), but there is no observation: every reported box "
                          "during the gap is *pure motion prediction*, and success needs IoU>=0.5 with "
                          "the true box.  A constant-velocity model extrapolates the tangent and leaves "
                          "the arc; a turn/acceleration model must stay on it.  This is the only family "
                          "where prediction quality can change the target-centric outcome.",
    "linear_blackout": "Control for `nonlinear_blackout`: the same blind window on a straight path, "
                       "where extrapolation is exact and every motion model should be equal.",
    "distractor_overlap": "**Hard-overlap distractor.**  A bystander rides *on top of* the target "
                          "(box offset (6,4) px, IoU 0.876) for `n_overlap` frames while the target "
                          "keeps a clean template, so the geometry/size/motion costs are near-identical "
                          "and the whole decision falls on appearance.  The bystander's descriptor is "
                          "correlated with the target's by a swept cosine, while the target's own crop "
                          "carries a fixed pose offset (`pose`, default 0.97): once the bystander is "
                          "*more* similar to the template than the target's current crop, the best "
                          "candidate is the wrong person and only `CostCfg.distractor_margin` can refuse.  "
                          "After the window the bystander peels away, so a tracker that latched onto it "
                          "loses the target and racks up false acquisitions.",
    "overlap_sameid": "Control for `distractor_overlap`: one person, detected **twice** on the same box "
                      "with the same descriptor.  Nothing can separate the two candidates, so any "
                      "refusal here is pure cost.",
    "overlap_solo": "Second control for `distractor_overlap`: the same trajectory and the same window "
                    "with no second person at all, i.e. the geometry by itself is harmless.",
    "unidentifiable_target": "**The requirement-(4) failure mode.**  The target walks, then HALTS, then "
                             "stops being *identifiable*: for `n_blind` frames it is still detected at its true "
                             "box but its descriptor is unusable (`mode='missing'`: no features at all, i.e. "
                             "`appearance_valid=False`; `mode='orthogonal'`: a valid but orthogonal vector).  "
                             "Meanwhile an unoccluded bystander with a clean, distinct descriptor rides the "
                             "target's **stale prediction** -- the constant-velocity coast the halted target "
                             "has fallen behind -- so geometry points at the bystander and appearance is the "
                             "only cue that can refuse.  The bystander's similarity to the target's template is "
                             "swept, and the target becomes identifiable again in the last `n_post` frames: the "
                             "question is whether the tracker hands the track to the bystander (wrong-identity "
                             "frames, no recovery) or refuses and re-acquires the true target.",
    "unidentifiable_solo": "Control for `unidentifiable_target`: the same halt, gap and blind window with no "
                           "bystander at all, i.e. the cost of the target's own unidentifiability, with no "
                           "wrong identity available to latch onto.",
}


def _blackout_scenario(
    name: str,
    family: str,
    gap: int,
    drop_at: int,
    boxes: Sequence[np.ndarray],
    params: dict[str, Any],
    note: str,
) -> Scenario:
    """Target follows ``boxes``, but the detector reports nothing for ``gap`` frames."""
    b = _Builder(name, family, len(boxes), {**params, "gap": int(gap), "drop_at": int(drop_at)})
    for f, box in enumerate(boxes):
        if drop_at <= f < drop_at + gap:
            b.hide_target(f, box)
        else:
            b.see_target(f, box)
    b.note = note
    return b.build()


def _box(cx: float, cy: float, w: float = BOX_W, h: float = BOX_H) -> np.ndarray:
    """Axis-aligned box from its centre and size."""
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def centre(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64).reshape(4)
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two ``xyxy`` boxes (the scenario overlap measure)."""
    a = np.asarray(a, dtype=np.float64).reshape(4)
    b = np.asarray(b, dtype=np.float64).reshape(4)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / total) if total > 0 else 0.0


def correlated_pair(cos: float, dim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors whose cosine similarity is exactly ``cos``.

    ``a`` is ``e0``; ``b`` is ``a`` rotated by ``acos(cos)`` inside the
    ``(e0, e1)`` plane, so the geometry of a whole similarity ladder can be
    reasoned about as rotation angles: two vectors built with ``cos=p`` and
    ``cos=q`` have similarity ``cos(acos(p) - acos(q))``.
    """
    c = float(min(1.0, max(-1.0, cos)))
    a = np.zeros(int(dim), dtype=np.float32)
    b = np.zeros(int(dim), dtype=np.float32)
    a[0] = 1.0
    b[0] = c
    b[1] = math.sqrt(max(0.0, 1.0 - c * c))
    return a, b


# --------------------------------------------------------------------------- #
# Scenario container
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """One deterministic frame sequence plus everything the evaluator needs.

    Attributes:
        name: unique id, used as the JSON/CSV key.
        family: one of :data:`FAMILY_NOTES`.
        params: the swept parameter values (``{"step": 20}``) so findings can be
            computed without parsing names.
        n_frames: number of frames; frame indices are ``0..n_frames-1``.
        click: the operator's click box (the target's box on frame 0).
        gt_target: per-frame target GT box, or ``None`` while the target is absent.
        all_gt: per-frame all-person GT, used to decide *which* identity a report
            latched onto (identity consistency / false acquisition).
        dets: per-frame detections handed to the tracker.
        person_boxes: per-frame ``(person_id, box)`` for the stub embedder.
        vectors: per-person embedding (``IDENT_A``/``IDENT_B``).
    """

    name: str
    family: str
    n_frames: int
    click: np.ndarray
    gt_target: dict[int, np.ndarray | None]
    all_gt: dict[int, list[GtBox]]
    dets: dict[int, list[Detection]]
    person_boxes: dict[int, list[tuple[int, np.ndarray]]]
    vectors: dict[int, np.ndarray]
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""


class _Builder:
    """Accumulates a scenario frame by frame."""

    def __init__(self, name: str, family: str, n_frames: int, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.family = family
        self.n = int(n_frames)
        self.params = dict(params or {})
        self.dets: dict[int, list[Detection]] = {f: [] for f in range(self.n)}
        self.gt: dict[int, np.ndarray | None] = {f: None for f in range(self.n)}
        self.all: dict[int, list[GtBox]] = {f: [] for f in range(self.n)}
        self.persons: dict[int, list[tuple[int, np.ndarray]]] = {f: [] for f in range(self.n)}
        self.click: np.ndarray | None = None
        self.note = ""

    # -- target ---------------------------------------------------------- #
    def see_target(self, f: int, box: np.ndarray, score: float = 0.9) -> None:
        """Target is present, in GT, and the detector reports it."""
        if not 0 <= f < self.n:
            return
        b = np.asarray(box, dtype=np.float64).reshape(4)
        self.dets[f].append(Detection(xyxy=b.astype(np.float32), score=float(score), track_id=TARGET_ID))
        self.gt[f] = b
        self.all[f].append(GtBox(f, TARGET_ID, b))
        self.persons[f].append((TARGET_ID, b))
        if self.click is None:
            self.click = b.copy()

    def hide_target(self, f: int, box: np.ndarray) -> None:
        """Target is visible in GT but the detector misses it (a detector drop)."""
        if not 0 <= f < self.n:
            return
        b = np.asarray(box, dtype=np.float64).reshape(4)
        self.gt[f] = b
        self.all[f].append(GtBox(f, TARGET_ID, b))
        self.persons[f].append((TARGET_ID, b))
        if self.click is None:
            self.click = b.copy()

    def see_other(self, f: int, box: np.ndarray, pid: int = DISTRACTOR_ID, score: float = 0.9) -> None:
        """Another person is present and detected (never the target)."""
        if not 0 <= f < self.n:
            return
        b = np.asarray(box, dtype=np.float64).reshape(4)
        self.dets[f].append(Detection(xyxy=b.astype(np.float32), score=float(score), track_id=int(pid)))
        self.all[f].append(GtBox(f, int(pid), b))
        self.persons[f].append((int(pid), b))

    def see_detection(self, f: int, box: np.ndarray, pid: int = DISTRACTOR_ID, score: float = 0.9) -> None:
        """A detection with no separate ground-truth person (a duplicate box).

        Used by the same-identity control: the detector reports the *one* person
        twice, so the scenario offers two indistinguishable candidates but only
        one real identity -- a false acquisition is then impossible by
        construction and any refusal is pure cost.
        """
        if not 0 <= f < self.n:
            return
        b = np.asarray(box, dtype=np.float64).reshape(4)
        self.dets[f].append(Detection(xyxy=b.astype(np.float32), score=float(score), track_id=int(pid)))
        self.persons[f].append((int(pid), b))

    # -- finish ---------------------------------------------------------- #
    def build(self, vectors: dict[int, np.ndarray] | None = None) -> Scenario:
        if self.click is None:
            raise ValueError(f"scenario {self.name!r} never showed the target on frame 0")
        return Scenario(
            name=self.name,
            family=self.family,
            n_frames=self.n,
            click=self.click,
            gt_target=dict(self.gt),
            all_gt={f: list(v) for f, v in self.all.items()},
            dets={f: list(v) for f, v in self.dets.items()},
            person_boxes={f: list(v) for f, v in self.persons.items()},
            vectors=dict(vectors or {TARGET_ID: IDENT_A}),
            params=dict(self.params),
            note=self.note,
        )


# --------------------------------------------------------------------------- #
# Scenario generators
# --------------------------------------------------------------------------- #
def fast_linear(
    step: float,
    n_frames: int = 60,
    drop: tuple[int, int] = (20, 28),
    x0: float = 260.0,
) -> Scenario:
    """Constant-velocity walk at ``step`` px/frame with a brief detector drop."""
    b = _Builder(f"fast_linear_step{step:g}", "fast_linear", n_frames, {"step": float(step)})
    for f in range(n_frames):
        box = _box(x0 + step * f, CY)
        if drop[0] <= f < drop[1]:
            b.hide_target(f, box)
        else:
            b.see_target(f, box)
    b.note = f"{step:g} px/frame, detector drop on frames {drop[0]}-{drop[1] - 1}"
    return b.build()


def circle(
    radius: float,
    speed: float,
    n_frames: int = 120,
    cx0: float = 1200.0,
    cy0: float = 360.0,
) -> Scenario:
    """Uniform circular motion: constant speed, constantly changing direction."""
    b = _Builder(
        f"circle_r{radius:g}_v{speed:g}",
        "nonlinear",
        n_frames,
        {"radius": float(radius), "speed": float(speed)},
    )
    omega = speed / radius
    for f in range(n_frames):
        t = omega * f
        b.see_target(f, _box(cx0 + radius * math.sin(t), cy0 - radius * math.cos(t)))
    b.note = f"radius {radius:g} px, {speed:g} px/frame ({2 * math.pi / omega:.0f} frames/rev)"
    return b.build()


def figure_eight(
    radius: float,
    speed: float,
    n_frames: int = 120,
    cx0: float = 1200.0,
    cy0: float = 360.0,
) -> Scenario:
    """Lemniscate of Gerono -- two turning directions per lap, plus a cusp."""
    b = _Builder(
        f"figure8_r{radius:g}_v{speed:g}",
        "nonlinear",
        n_frames,
        {"radius": float(radius), "speed": float(speed)},
    )
    omega = speed / radius
    for f in range(n_frames):
        t = omega * f
        b.see_target(f, _box(cx0 + radius * math.sin(t), cy0 + 0.5 * radius * math.sin(2.0 * t)))
    b.note = f"radius {radius:g} px, {speed:g} px/frame, turning both ways"
    return b.build()


def occlusion(
    gap: int,
    distant: bool = False,
    speed: float = 6.0,
    n_pre: int = 21,
    n_post: int = 40,
    x0: float = 260.0,
    jump: float = 600.0,
) -> Scenario:
    """The target vanishes for ``gap`` frames, then reappears.

    ``distant=True`` puts the reappearance ``jump`` px away on a different row,
    so only appearance (and the tracker's global search) can rebind it.
    """
    n_frames = n_pre + gap + n_post
    kind = "distant" if distant else "inplace"
    b = _Builder(
        f"occlusion_{kind}_gap{gap}",
        "occlusion",
        n_frames,
        {"gap": int(gap), "distant": bool(distant)},
    )
    for f in range(n_frames):
        if f < n_pre:
            b.see_target(f, _box(x0 + speed * f, CY))
        elif f < n_pre + gap:
            continue  # absent: not in GT, not detected, no person box
        elif distant:
            b.see_target(f, _box(x0 + speed * f + jump, 150.0))
        else:
            b.see_target(f, _box(x0 + speed * f, CY))
    reappear = n_pre + gap
    b.note = (
        f"{gap} frames absent; reappears {'%.0f px away on row y=150' % jump if distant else 'on its own path'}"
        f" at frame {reappear}"
    )
    return b.build()


def linear_blackout(step: float, gap: int, drop_at: int = 20, n_frames: int = 70, x0: float = 260.0) -> Scenario:
    """Straight path with a blind window -- the control for :func:`circle_blackout`."""
    boxes = [_box(x0 + step * f, CY) for f in range(n_frames)]
    return _blackout_scenario(
        f"linear_blackout_step{step:g}_gap{gap}",
        "linear_blackout",
        gap,
        drop_at,
        boxes,
        {"step": float(step), "shape": "linear"},
        f"{step:g} px/frame straight; detector blind for {gap} frames from frame {drop_at}",
    )


def circle_blackout(
    radius: float,
    speed: float,
    gap: int,
    drop_at: int = 20,
    n_frames: int = 70,
    cx0: float = 1200.0,
    cy0: float = 360.0,
) -> Scenario:
    """Circular path with a blind window: prediction quality decides the reported box."""
    omega = speed / radius
    boxes = [
        _box(cx0 + radius * math.sin(omega * f), cy0 - radius * math.cos(omega * f)) for f in range(n_frames)
    ]
    return _blackout_scenario(
        f"circle_blackout_r{radius:g}_v{speed:g}_gap{gap}",
        "nonlinear_blackout",
        gap,
        drop_at,
        boxes,
        {"radius": float(radius), "speed": float(speed), "shape": "circle"},
        f"radius {radius:g} px, {speed:g} px/frame; detector blind for {gap} frames "
        f"({1000.0 * speed * gap / radius:.1f} mrad of turn unobserved)",
    )


def figure8_blackout(
    radius: float,
    speed: float,
    gap: int,
    drop_at: int = 20,
    n_frames: int = 70,
    cx0: float = 1200.0,
    cy0: float = 360.0,
) -> Scenario:
    """Figure-eight with a blind window (curvature reverses inside the window)."""
    omega = speed / radius
    boxes = [
        _box(cx0 + radius * math.sin(omega * f), cy0 + 0.5 * radius * math.sin(2.0 * omega * f))
        for f in range(n_frames)
    ]
    return _blackout_scenario(
        f"figure8_blackout_r{radius:g}_v{speed:g}_gap{gap}",
        "nonlinear_blackout",
        gap,
        drop_at,
        boxes,
        {"radius": float(radius), "speed": float(speed), "shape": "figure8"},
        f"radius {radius:g} px, {speed:g} px/frame, both turn directions; blind for {gap} frames",
    )


def distractor(
    lookalike: bool = True,
    n_frames: int = 100,
    speed: float = 6.0,
    x0: float = 300.0,
    d_start: int = 45,
    d_end: int = 95,
    d_dx0: float = 360.0,
    d_dx_rate: float = -12.0,
    d_dy0: float = -200.0,
    d_dy_rate: float = 8.0,
) -> Scenario:
    """An equal-size person crosses the target's path.

    The crosser starts ahead-and-above and closes on the target, becoming
    nearly coincident around frame ~72 and then passing behind it.  With
    ``lookalike=True`` both people share one embedding, so appearance is
    uninformative and only the motion/margin machinery can separate them.
    """
    kind = "lookalike" if lookalike else "different"
    name = f"distractor_{kind}"
    b = _Builder(name, "distractor", n_frames, {"lookalike": bool(lookalike)})
    for f in range(n_frames):
        b.see_target(f, _box(x0 + speed * f, CY))
        if d_start <= f < d_end:
            dt = f - d_start
            b.see_other(f, _box(x0 + speed * f + d_dx0 + d_dx_rate * dt, CY + d_dy0 + d_dy_rate * dt))
    vectors = {TARGET_ID: IDENT_A, DISTRACTOR_ID: IDENT_A if lookalike else IDENT_B}
    b.note = (
        "identical embedding for both people (appearance gives no identity)"
        if lookalike
        else "orthogonal embedding for the bystander (appearance alone separates them)"
    )
    return b.build(vectors=vectors)


#: The similarity ladder for the hard-overlap family.  ``0.0`` is an orthogonal
#: bystander; from ~0.965 upwards the bystander becomes *at least as similar* to
#: the target's template as the target's own pose-varied crop, which is where
#: only the margin gate can help.
DISTRACTOR_COSINES: tuple[float, ...] = (0.0, 0.60, 0.85, 0.92, 0.97, 0.99, 1.00)

#: ``CostCfg.distractor_margin`` values to sweep.
DISTRACTOR_MARGINS: tuple[float, ...] = (0.0, 0.03, 0.06, 0.12)

#: Models used in the distractor sweep.  ``rollback_depth`` is left at the
#: shipped default (0) so the margin is the only thing that changes.
DISTRACTOR_MODELS: tuple[str, ...] = ("kf", "imm")


def overlap_distractor(
    similarity: float = 0.85,
    *,
    pose: float = 0.97,
    n_pre: int = 16,
    n_overlap: int = 28,
    n_post: int = 22,
    offset: Sequence[float] = (6.0, 4.0),
    speed: float = 6.0,
    x0: float = 300.0,
    dim: int = 4,
    peel: Sequence[float] = (-14.0, -9.0),
) -> Scenario:
    """A bystander rides on top of the target; only appearance can separate them.

    Frames ``0..n_pre-1``:  the target walks right and the bystander closes in
    from up-and-left, so the target's lock is established while the bystander is
    still far away.  Frames ``n_pre..n_pre+n_overlap-1``: the two boxes differ by
    ``offset`` (default ``(6, 4)`` px -> IoU 0.876 for 120x240 boxes), so the
    motion, size and geometry costs are near-identical.  Frames after that: the
    bystander peels away along ``peel`` (default ``(-14, -9)`` px/frame relative
    to the target, i.e. backwards/up-left) while the target keeps walking, so a
    tracker that latched onto the bystander is separated from the target and its
    report stops overlapping the target's box.  ``peel`` also sets the direction
    of the relative motion, which matters: the default sends the bystander
    *behind* the target, the same direction a velocity-damped blind coast drifts,
    so the two can coincide after a long refusal window.
    ``overlap_distractor(1.0, peel=(14.0, 9.0))`` is the forward-peel control.

    The target's own crop carries a fixed pose offset (``pose``, default 0.97
    cosine to its frame-0 template), while the bystander's crop sits at the swept
    ``similarity``.  ``similarity > pose`` makes the bystander the better
    *appearance* match, but appearance is only one term of the association cost:
    the ``offset`` box also pays a geometry/motion penalty, and at the default
    ``CostCfg`` weights the measured margin (``runner_up - best_cost``) stays
    slightly positive, so the target keeps winning outright unless
    ``CostCfg.distractor_margin`` refuses it.  Measured at ``similarity=1.00``:
    margin 0.00/0.03 -> 0 refusals, success 1.0000, no false acquisition; margin
    0.06/0.12 -> 50 refusals, success 0.6667, and (with the default backwards
    peel) 5 false-acquisition frames.  The refusal, not the appearance tie, is
    what costs success.
    """
    cos = float(min(1.0, max(0.0, similarity)))
    ox, oy = float(offset[0]), float(offset[1])
    px, py = float(peel[0]), float(peel[1])
    n_frames = int(n_pre + n_overlap + n_post)
    name = f"overlap_distractor_s{int(round(cos * 100)):03d}" + ("" if px < 0 else "_fwd")
    params = {
        "similarity": cos,
        "pose": float(pose),
        "n_pre": int(n_pre),
        "n_overlap": int(n_overlap),
        "n_post": int(n_post),
        "offset": [ox, oy],
        "peel": [px, py],
        "same_identity": False,
    }
    b = _Builder(name, "distractor_overlap", n_frames, params)

    a_vec, _ = correlated_pair(1.0, dim)          # e0: the frame-0 crop -> the seeded template
    _, t_vec = correlated_pair(float(pose), dim)  # the target's own per-frame crop
    _, d_vec = correlated_pair(cos, dim)          # the bystander's crop

    t_frames = np.tile(t_vec, (n_frames, 1))
    t_frames[0] = a_vec
    vectors = {
        TARGET_ID: t_frames.astype(np.float32),
        DISTRACTOR_ID: np.tile(d_vec, (n_frames, 1)).astype(np.float32),
    }

    d_start = max(0, int(n_pre) - 10)
    span = max(1, int(n_pre) - d_start)
    for f in range(n_frames):
        tx = x0 + speed * f
        b.see_target(f, _box(tx, CY))
        if f < d_start:
            continue
        if f < n_pre:
            k = (f - d_start) / span
            dx = -240.0 + (ox + 240.0) * k
            dy = -150.0 + (oy + 150.0) * k
        elif f < n_pre + n_overlap:
            dx, dy = ox, oy
        else:
            k = f - (n_pre + n_overlap) + 1
            dx, dy = ox + px * k, oy + py * k
        b.see_other(f, _box(tx + dx, CY + dy))

    b.note = (
        f"bystander box IoU {box_iou(_box(0.0, 0.0), _box(ox, oy)):.3f} over {n_overlap} frames; "
        f"cos(target crop, bystander crop) = {float(np.dot(t_vec, d_vec)):.3f} "
        f"(target's own crop vs template = {pose:.3f})"
    )
    return b.build(vectors=vectors)


def overlap_sameid(
    *,
    n_pre: int = 16,
    n_overlap: int = 28,
    n_post: int = 22,
    speed: float = 6.0,
    x0: float = 300.0,
    dim: int = 4,
) -> Scenario:
    """Control: one person reported twice on the same box with the same descriptor.

    There is no second identity in ``all_gt``, so a false acquisition is
    impossible by construction: every refusal this scenario produces is pure
    cost.  A margin gate whose purpose is anti-latch refusal should fire here and
    *lose lock time*, which is the honest price of the mechanism.
    """
    n_frames = int(n_pre + n_overlap + n_post)
    params = {
        "similarity": 1.0,
        "pose": 1.0,
        "n_pre": int(n_pre),
        "n_overlap": int(n_overlap),
        "n_post": int(n_post),
        "offset": [0.0, 0.0],
        "same_identity": True,
    }
    b = _Builder("overlap_sameid", "overlap_sameid", n_frames, params)
    a_vec, _ = correlated_pair(1.0, dim)
    for f in range(n_frames):
        box = _box(x0 + speed * f, CY)
        b.see_target(f, box)
        if n_pre <= f < n_pre + n_overlap:
            b.see_detection(f, box)
    b.note = (
        f"one person reported twice on the identical box for {n_overlap} frames: the two "
        "candidates are indistinguishable, so the margin gate can only refuse"
    )
    return b.build(vectors={TARGET_ID: a_vec, DISTRACTOR_ID: a_vec})


def overlap_solo(
    *,
    n_pre: int = 16,
    n_overlap: int = 28,
    n_post: int = 22,
    speed: float = 6.0,
    x0: float = 300.0,
    dim: int = 4,
) -> Scenario:
    """Control: the overlap trajectory and window with no second person at all."""
    n_frames = int(n_pre + n_overlap + n_post)
    params = {
        "similarity": 1.0,
        "pose": 0.97,
        "n_pre": int(n_pre),
        "n_overlap": int(n_overlap),
        "n_post": int(n_post),
        "offset": [0.0, 0.0],
        "same_identity": False,
    }
    b = _Builder("overlap_solo", "overlap_solo", n_frames, params)
    a_vec, _ = correlated_pair(1.0, dim)
    _, t_vec = correlated_pair(0.97, dim)
    t_frames = np.tile(t_vec, (n_frames, 1))
    t_frames[0] = a_vec
    for f in range(n_frames):
        b.see_target(f, _box(x0 + speed * f, CY))
    b.note = "same trajectory/window as distractor_overlap with no bystander: the geometry alone is harmless"
    return b.build(vectors={TARGET_ID: t_frames.astype(np.float32)})


# --------------------------------------------------------------------------- #
# Unidentifiable target + identifiable bystander at the stale prediction
# --------------------------------------------------------------------------- #
#: Bystander/template similarities.  The rungs between 0.0 and 0.6 bracket the
#: point where ``w_appearance * (1 - cos)`` alone drops a bystander below the
#: tracker's accept threshold, so they separate "refused on appearance" from
#: "accepted on geometry".
UNID_SIMILARITIES: tuple[float, ...] = (0.0, 0.4, 0.6, 0.85)
#: How the target's appearance stops being usable while it is still detected.
UNID_MODES: tuple[str, ...] = ("missing", "orthogonal")
UNID_MODELS: tuple[str, ...] = ("kf", "imm")
#: (label, cfg overrides): the shipped tracker vs the pre-fix arming rule
#: (``state.global_states = ()`` restores the old "search only after
#: ``lost_long_after`` frames" behaviour).
UNID_SWEEPS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("shipped", None),
    ("no_early_search", {"state.global_states": ()}),
)
#: `CostCfg.missing_appearance_cost` values for the pricing sweep.  0.5 is the
#: shipped value (what `association.compute_costs` used to hardcode): an
#: unreadable candidate is priced at ``missing_appearance_cost * w_appearance``
#: = 0.8 against the 1.0 accept budget, i.e. *cheaper* than a genuine mismatch
#: (cosine 1.0 -> 1.6), so "we could not read this person" can outrank "this is
#: someone else".
#: The first sweep tested only 0.5/1.0/1.2 and found no effect at all -- but
#: raising the price makes the unreadable TRUE target *more* expensive, so it
#: cannot possibly help, and the measured handover was unchanged by design.
#: The direction that should work is the LOW side: price a missing appearance
#: near zero so the unreadable true target is not rejected in favour of a
#: readable stranger.  0.0/0.1/0.25 test that; 0.5 stays as the shipped value
#: and 1.0 is kept as the failed upper control.
MISSING_COSTS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)


def unidentifiable_target(
    similarity: float = 0.6,
    *,
    mode: str = "missing",
    n_pre: int = 18,
    n_gap: int = 18,
    n_blind: int = 24,
    n_post: int = 18,
    speed: float = 6.0,
    offset: tuple[float, float] = (6.0, 4.0),
    peel: tuple[float, float] = (-14.0, -9.0),
    x0: float = 320.0,
    dim: int = 4,
) -> Scenario:
    """Target halts and goes unidentifiable; a clean bystander holds its stale prediction.

    Phases (``n_frames = n_pre + n_gap + n_blind + n_post``):

    * **A** ``f < n_pre``: the target walks at ``speed`` px/frame and is
      detected with a clean descriptor (its own crop is 0.97 similar to the
      frame-0 crop that seeded the template).  A bystander closes in from
      ``(self.offset[0], self.offset[1] - 260)``.
    * **gap** ``n_pre <= f < n_pre + n_gap``: the target *halts*, so the
      tracker's constant-velocity coast keeps moving -- that coast is the
      "stale prediction".  The target is still in GT (visible) but the detector
      misses it; the bystander is detected at the coast position.
    * **blind** ``n_gap`` frames: the target is detected again at its true
      (halted) box, but its descriptor is unusable -- ``mode='missing'`` gives
      an all-zero row, which the stub embedder turns into ``{}`` (no features,
      i.e. the tracker's ``appearance_valid=False`` guard case);
      ``mode='orthogonal'`` gives a valid but orthogonal vector.  The bystander
      still sits on the coast.
    * **post**: the target's descriptor is clean again (nothing stops the
      tracker from recovering) and the bystander peels away at ``peel``.

    A wrong-identity report is only *measurable* once the halted target and the
    coast (with the bystander on it) have separated by more than the identity
    IoU of 0.9, which takes a few frames at ``speed``; the two boxes coincide at
    the halt by construction, because "the coast position" is where the target
    was when it stopped.
    """
    if mode not in UNID_MODES:
        raise ValueError(f"mode must be one of {UNID_MODES}, got {mode!r}")
    cos = float(min(1.0, max(0.0, similarity)))
    ox, oy = float(offset[0]), float(offset[1])
    px, py = float(peel[0]), float(peel[1])
    n_pre, n_gap, n_blind, n_post = int(n_pre), int(n_gap), int(n_blind), int(n_post)
    n_frames = n_pre + n_gap + n_blind + n_post
    name = f"unidentifiable_{mode}_s{int(round(cos * 100)):03d}"
    params = {
        "similarity": cos,
        "mode": mode,
        "pose": 0.97,
        "n_pre": n_pre,
        "n_gap": n_gap,
        "n_blind": n_blind,
        "n_post": n_post,
        "speed": float(speed),
        "offset": [ox, oy],
        "peel": [px, py],
        "same_identity": False,
    }
    b = _Builder(name, "unidentifiable_target", n_frames, params)

    a_vec, _ = correlated_pair(1.0, dim)   # frame-0 crop -> the seeded template
    _, t_vec = correlated_pair(0.97, dim)  # the target's clean per-frame crop
    _, d_vec = correlated_pair(cos, dim)   # the bystander's crop
    _, o_vec = correlated_pair(0.0, dim)   # orthogonal to the template

    rows = np.tile(t_vec, (n_frames, 1))
    rows[0] = a_vec
    blind_lo = n_pre + n_gap
    blind_hi = blind_lo + n_blind
    if mode == "missing":
        rows[blind_lo:blind_hi] = 0.0      # -> encode() returns {} -> no appearance
    else:
        rows[blind_lo:blind_hi] = o_vec
    vectors = {
        TARGET_ID: rows.astype(np.float32),
        DISTRACTOR_ID: np.tile(d_vec, (n_frames, 1)).astype(np.float32),
    }

    x_halt = x0 + speed * (n_pre - 1)      # the target stops here
    d_start = max(0, n_pre - 8)
    span = max(1, n_pre - d_start)
    for f in range(n_frames):
        tx = x0 + speed * f if f < n_pre else x_halt
        if f < n_pre:
            b.see_target(f, _box(tx, CY))
        elif f < blind_lo:
            b.hide_target(f, _box(tx, CY))          # detector misses the halted target
        else:
            b.see_target(f, _box(tx, CY))           # detected, but appearance unusable
        # The bystander rides the STALE PREDICTION: stale(f) is the coast box,
        # i.e. the target's last-velocity extrapolation, which keeps advancing
        # after the target halts.
        stale = x0 + speed * f
        if f < d_start:
            continue
        if f < n_pre:
            k = (f - d_start) / span
            bx, by = ox, (oy - 260.0) + 260.0 * k
        elif f < blind_hi:
            bx, by = ox, oy
        else:
            k = f - blind_hi + 1
            bx, by = ox + px * k, oy + py * k
        b.see_other(f, _box(stale + bx, CY + by))
    b.note = (
        f"target halts at frame {n_pre} and its descriptor is unusable for {n_blind} frames "
        f"({mode}); bystander with cos={cos:.2f} rides the stale prediction"
    )
    return b.build(vectors=vectors)


def unidentifiable_solo(
    *,
    mode: str = "missing",
    n_pre: int = 18,
    n_gap: int = 18,
    n_blind: int = 24,
    n_post: int = 18,
    speed: float = 6.0,
    x0: float = 320.0,
    dim: int = 4,
) -> Scenario:
    """Control: the same halt/gap/blind window with no bystander to latch onto."""
    n_pre, n_gap, n_blind, n_post = int(n_pre), int(n_gap), int(n_blind), int(n_post)
    n_frames = n_pre + n_gap + n_blind + n_post
    name = f"unidentifiable_solo_{mode}"
    params = {
        "similarity": 0.0,
        "mode": mode,
        "pose": 0.97,
        "n_pre": n_pre,
        "n_gap": n_gap,
        "n_blind": n_blind,
        "n_post": n_post,
        "speed": float(speed),
        "offset": [0.0, 0.0],
        "peel": [0.0, 0.0],
        "same_identity": False,
    }
    b = _Builder(name, "unidentifiable_solo", n_frames, params)
    a_vec, _ = correlated_pair(1.0, dim)
    _, t_vec = correlated_pair(0.97, dim)
    _, o_vec = correlated_pair(0.0, dim)
    rows = np.tile(t_vec, (n_frames, 1))
    rows[0] = a_vec
    blind_lo = n_pre + n_gap
    blind_hi = blind_lo + n_blind
    rows[blind_lo:blind_hi] = 0.0 if mode == "missing" else o_vec
    x_halt = x0 + speed * (n_pre - 1)
    for f in range(n_frames):
        tx = x0 + speed * f if f < n_pre else x_halt
        if f < n_pre or f >= blind_lo:
            b.see_target(f, _box(tx, CY))
        else:
            b.hide_target(f, _box(tx, CY))
    b.note = f"no bystander: cost of the target's own unidentifiability ({mode})"
    return b.build(vectors={TARGET_ID: rows.astype(np.float32)})


def default_scenarios() -> list[Scenario]:
    """The full sweep: 6 fast-linear + 4 nonlinear + 8 occlusion + 2 distractor + 6 blackout + 5 overlap + 4 unidentifiable."""
    out: list[Scenario] = []
    for step in (2, 5, 10, 20, 40, 80):
        out.append(fast_linear(step))
    out.append(circle(60, 4))
    out.append(circle(100, 8))
    out.append(circle(150, 12))
    out.append(figure_eight(80, 6))
    # The decisive experiment: pure motion prediction while the detector is blind.
    out.append(linear_blackout(20, gap=15))
    out.append(linear_blackout(40, gap=15))
    out.append(circle_blackout(60, 4, gap=15))
    out.append(circle_blackout(150, 12, gap=20, n_frames=75))
    out.append(circle_blackout(60, 4, gap=30, n_frames=90))
    out.append(figure8_blackout(80, 6, gap=15))
    for gap in (5, 15, 40, 90):
        out.append(occlusion(gap, distant=False))
        out.append(occlusion(gap, distant=True))
    out.append(distractor(lookalike=True))
    out.append(distractor(lookalike=False))
    # The hard-overlap family (only three rungs here; the full ladder is the
    # dedicated --distractor sweep, which also sweeps `distractor_margin`).
    out.append(overlap_solo())
    out.append(overlap_sameid())
    out.append(overlap_distractor(0.85))
    out.append(overlap_distractor(0.92))
    out.append(overlap_distractor(0.99))
    # The requirement-(4) family: the target goes unidentifiable while an
    # unoccluded bystander holds the stale prediction.  One clean rung and one
    # clear-mismatch rung in the main sweep; the full ladder (plus the
    # global_states ablation) is the dedicated --unidentifiable sweep.
    out.append(unidentifiable_solo())
    out.append(unidentifiable_target(0.60))
    out.append(unidentifiable_target(0.00))
    out.append(unidentifiable_target(0.60, mode="orthogonal"))
    return out


def quick_scenarios() -> list[Scenario]:
    """A reduced set for smoke runs and tests (same families, short frames)."""
    return [
        fast_linear(10, n_frames=24, drop=(10, 13)),
        fast_linear(40, n_frames=24, drop=(10, 13)),
        circle(60, 4, n_frames=40),
        figure_eight(80, 6, n_frames=40),
        linear_blackout(20, gap=8, drop_at=8, n_frames=28),
        circle_blackout(60, 4, gap=8, drop_at=8, n_frames=28),
        occlusion(15, distant=False, n_pre=6, n_post=10),
        distractor(lookalike=True, n_frames=60),
        overlap_distractor(0.99, n_pre=8, n_overlap=12, n_post=10),
        unidentifiable_target(0.60, n_pre=8, n_gap=8, n_blind=10, n_post=8),
    ]


# --------------------------------------------------------------------------- #
# Deterministic stub embedder
# --------------------------------------------------------------------------- #
class ScenarioEmbedder:
    """Deterministic stand-in for the ReID encoder.

    ``encode`` is called synchronously inside ``TargetTracker.step``, so the
    harness sets :attr:`frame` before each step and the stub can map a query box
    to the person whose box is nearest its centre.  The tracker never sees the
    person id -- only the fixed vector that comes back.
    """

    def __init__(self, vectors: dict[int, np.ndarray], person_boxes: dict[int, list[tuple[int, np.ndarray]]]) -> None:
        # A value may be 1-D (fixed descriptor) or 2-D (n_frames, dim) to model a
        # descriptor that drifts with pose; see `feat_for`.  An all-zero vector
        # (or an all-zero row of a 2-D value) means **no usable appearance**:
        # `encode` then returns ``{}`` for that box, which is exactly the
        # `appearance_valid=False` case the tracker's search guard keys on.
        self.vectors = {int(k): np.asarray(v, dtype=np.float32) for k, v in vectors.items()}
        self.person_boxes = person_boxes
        self.frame = 0
        self.calls = 0
        self.unknown = 0

    def feat_for(self, box: np.ndarray) -> np.ndarray:
        cands = self.person_boxes.get(self.frame) or []
        if not cands:
            self.unknown += 1
            return IDENT_A.copy()
        c = centre(box)
        pid = min(cands, key=lambda pb: float(np.linalg.norm(centre(pb[1]) - c)))[0]
        vec = self.vectors.get(int(pid))
        if vec is None:
            return IDENT_A.copy()
        if vec.ndim > 1:
            idx = min(max(int(self.frame), 0), vec.shape[0] - 1)
            vec = vec[idx]
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size == 0 or not float(np.linalg.norm(vec)):
            return np.zeros(0, dtype=np.float32)
        return vec.copy()

    def encode(self, img_bgr: Any, boxes_xyxy: Any, parts: Sequence[str] = ("full",)) -> list[dict[str, np.ndarray]]:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        self.calls += 1
        out: list[dict[str, np.ndarray]] = []
        for b in boxes:
            vec = self.feat_for(b)
            out.append({p: vec for p in parts} if vec.size else {})
        return out


# --------------------------------------------------------------------------- #
# Configuration sweep
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MotionSpec:
    """One point in the motion-strategy sweep."""

    label: str
    model: str = "kf"
    nsa_adaptive: bool = True
    rollback_depth: int = 5
    imm_models: tuple[str, ...] = ("cv", "ca", "ct")
    distractor_margin: float | None = None
    #: Non-motion dotted-path overrides applied with ``Config.clone`` (e.g.
    #: ``(("state.global_states", ()),)``).  Stored as an item tuple so the
    #: frozen dataclass stays hashable; see `unid_specs`.
    cfg_overrides: tuple[tuple[str, Any], ...] = ()

    def apply(self, cfg: Config) -> Config:
        cfg.motion.model = self.model
        cfg.motion.nsa_adaptive = self.nsa_adaptive
        cfg.motion.rollback_depth = int(self.rollback_depth)
        cfg.motion.imm_models = tuple(self.imm_models)
        if self.distractor_margin is not None:
            cfg.cost.distractor_margin = float(self.distractor_margin)
        if self.cfg_overrides:
            cfg = cfg.clone(**dict(self.cfg_overrides))
        return cfg

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.label,
            "model": self.model,
            "nsa_adaptive": bool(self.nsa_adaptive),
            "rollback_depth": int(self.rollback_depth),
            "imm_models": list(self.imm_models),
            "distractor_margin": None if self.distractor_margin is None else float(self.distractor_margin),
            "cfg_overrides": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in self.cfg_overrides},
        }


DEFAULT_SPECS: tuple[MotionSpec, ...] = (
    MotionSpec("kf", model="kf", rollback_depth=5),
    MotionSpec("kf-rb0", model="kf", rollback_depth=0),
    MotionSpec("kf-rb3", model="kf", rollback_depth=3),
    MotionSpec("nsa_kf", model="nsa_kf", rollback_depth=5),
    MotionSpec("nsa_kf-static", model="nsa_kf", nsa_adaptive=False, rollback_depth=5),
    MotionSpec("imm", model="imm", rollback_depth=5),
    MotionSpec("imm-rb0", model="imm", rollback_depth=0),
    MotionSpec("imm-rb3", model="imm", rollback_depth=3),
    MotionSpec("imm-cv", model="imm", imm_models=("cv",), rollback_depth=5),
    MotionSpec("imm-cvca", model="imm", imm_models=("cv", "ca"), rollback_depth=5),
)

#: Metrics copied into the CSV / leaderboard (all produced by evaluate_target).
REPORT_METRICS: tuple[str, ...] = (
    "target_success_rate",
    "target_precision",
    "loss_rate",
    "reacquire_rate",
    "reacquire_latency",
    "reacquire_latency_p90",
    "false_acquisition_rate",
    "false_acquisition_frames",
    "id_consistency",
    "hallucination_rate",
    "fragmentation",
    "longest_lost_streak",
)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_scenario(
    scenario: Scenario,
    spec: MotionSpec,
    *,
    trace: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive the real tracker through one scenario and score it.

    With ``trace=True`` the result additionally carries a ``trace`` list holding
    one diagnostics row per frame (:data:`TRACE_COLUMNS`).  The driving loop is
    identical either way: tracing observes the run, it must not change it.

    ``cfg_overrides`` is a dotted-path patch applied on top of ``spec.apply()``
    (``Config.clone``), for non-motion controls such as
    ``{"state.global_states": ()}``.
    """
    cfg = spec.apply(Config())
    # E029 turned the adaptive identity band ON by default.  This synthetic family
    # isolates the MOTION and decision subsystems in worlds whose appearance model
    # is a stub, so an anchor-distance veto here would be measuring noise; the band
    # is switched off explicitly rather than left to the ambient default.  The real
    # datasets keep it on (that is where E029/E030 measured it).
    cfg.cost.anchor_adaptive = False
    if cfg_overrides:
        cfg = cfg.clone(**cfg_overrides)
    embedder = ScenarioEmbedder(scenario.vectors, scenario.person_boxes)
    tracker = TargetTracker(cfg, embedder)

    rows: list[dict[str, Any]] = []
    statuses: dict[int, Any] = {}
    embedder.frame = 0
    statuses[0] = tracker.initialize(FRAME, scenario.click, frame_idx=0)
    if trace:
        rows.append(_trace_row(tracker, scenario, 0, statuses[0], cfg))
    for f in range(1, scenario.n_frames):
        embedder.frame = f
        statuses[f] = tracker.step(FRAME, scenario.dets.get(f, []), frame_idx=f)
        if trace:
            rows.append(_trace_row(tracker, scenario, f, statuses[f], cfg))

    metrics = evaluate_target(
        scenario.gt_target,
        statuses,
        list(tracker.events),
        scenario.all_gt,
        cfg.eval,
    )
    return {
        "scenario": scenario.name,
        "family": scenario.family,
        "params": dict(scenario.params),
        "n_frames": int(scenario.n_frames),
        "states": _state_histogram(statuses),
        "embedder_calls": int(embedder.calls),
        "unmapped_crops": int(embedder.unknown),
        **({"trace": rows} if trace else {}),
        **spec.as_dict(),
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# per-frame diagnostics
# --------------------------------------------------------------------------- #
TRACE_COLUMNS: tuple[str, ...] = (
    "frame",
    "state",
    "fsu",
    "n_dets",
    "gated",
    "best_cost",
    "thr",
    "runner_up",
    "accepted",
    "reason",
    "gate_r",
    "pred_to_gt",
    "iou_target",
    "iou_bystander",
    "gt_present",
    "blind",
)

_TRACE_WIDTH: dict[str, int] = {
    "frame": 5,
    "state": 14,
    "fsu": 4,
    "n_dets": 6,
    "gated": 5,
    "best_cost": 9,
    "thr": 6,
    "runner_up": 9,
    "accepted": 8,
    "reason": 26,
    "gate_r": 8,
    "pred_to_gt": 9,
    "iou_target": 10,
    "iou_bystander": 12,
    "gt_present": 10,
    "blind": 5,
}


def _trace_row(
    tracker: Any,
    scenario: Scenario,
    frame: int,
    status: Any,
    cfg: Config,
) -> dict[str, Any]:
    """One frame of association diagnostics, read straight off the tracker.

    ``state``/``fsu``/``n_dets``/``gated``/``best_cost``/``reason``/``accepted``
    are the tracker's own ``TargetStatus.diag`` values for the association that
    just happened.  ``thr`` is the accept threshold that applied (the lost
    variant while blind), ``runner_up`` is reconstructed from the tracker's
    margin (``association.select_candidate`` defines ``margin`` as runner-up
    minus best), and ``pred_to_gt`` is the distance from the tracker's reported
    (while blind: predicted) box to the ground-truth box -- the quantity the
    gate radius has to cover for a distant reappearance to be seen at all.

    ``gate_r`` and ``blind`` are read after the step, so they lead ``fsu`` by
    one frame; the radius is still the one the tracker would apply next frame.
    """
    d = dict(getattr(status, "diag", None) or {})
    box = getattr(status, "xyxy", None)
    gt = scenario.gt_target.get(frame)
    pred_to_gt = None
    if box is not None and gt is not None:
        pred_to_gt = float(
            np.linalg.norm(
                centre(np.asarray(box, dtype=np.float32)) - centre(np.asarray(gt, dtype=np.float32))
            )
        )
    # Who is the report actually ON?  IoU, exactly as `evaluate_target` decides
    # which ground-truth person a report belongs to, so a "handover" detected
    # here is the same event the false-acquisition metric charges for.
    iou_target = None
    iou_bystander = None
    bystander = dict(scenario.person_boxes.get(frame, [])).get(DISTRACTOR_ID)
    if box is not None:
        box = np.asarray(box, dtype=np.float32)
        if gt is not None:
            iou_target = box_iou(box, np.asarray(gt, dtype=np.float32))
        if bystander is not None:
            iou_bystander = box_iou(box, np.asarray(bystander, dtype=np.float32))
    cost = d.get("cost")
    margin = d.get("margin")
    runner_up = None
    if cost is not None and margin is not None and np.isfinite(cost) and np.isfinite(margin):
        runner_up = float(cost) + float(margin)
    blind = bool(getattr(tracker, "is_blind", False))
    try:
        gate_r = tracker._gate_override()
    except Exception:  # pragma: no cover - diagnostics must never break a run
        gate_r = None
    return {
        "frame": int(frame),
        "state": d.get("state"),
        "fsu": d.get("fsu"),
        "n_dets": d.get("n_dets"),
        "gated": d.get("gate_pass"),
        "best_cost": None if cost is None or not np.isfinite(cost) else float(cost),
        "thr": float(cfg.cost.accept_cost_lost if blind else cfg.cost.accept_cost),
        "runner_up": runner_up,
        "accepted": bool(d.get("accepted", False)),
        "reason": d.get("reason"),
        "gate_r": None if gate_r is None else float(gate_r),
        "pred_to_gt": pred_to_gt,
        "iou_target": iou_target,
        "iou_bystander": iou_bystander,
        "gt_present": gt is not None,
        "blind": blind,
    }


def handover_frame(
    rows: Sequence[dict[str, Any]],
    *,
    start: int = 0,
) -> dict[str, Any] | None:
    """First frame at/after ``start`` whose report sits on the bystander, not the target.

    A report is attributed the way ``evaluate_target`` attributes it: by IoU.
    Returns ``None`` while the report is still closest to the target (or is on
    neither person).
    """
    for row in rows:
        frame = int(row.get("frame", 0))
        if frame < start or not row.get("accepted"):
            continue
        on_target = row.get("iou_target")
        on_bystander = row.get("iou_bystander")
        if on_bystander is None:
            continue
        if on_target is None or float(on_bystander) > float(on_target):
            return {
                "frame": frame,
                "n_dets": row.get("n_dets"),
                "best_cost": row.get("best_cost"),
                "runner_up": row.get("runner_up"),
                "iou_target": on_target,
                "iou_bystander": on_bystander,
                "phase": "gap" if not row.get("blind") else "blind",
            }
    return None


def handover_split(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """How many runs hand the track over with 1, 2 or more detections present.

    ``n_dets == 1`` means the bystander was the *only* finite candidate, so
    ``association.select_candidate`` left the margin at ``inf`` and the
    ambiguity test could not fire whatever ``distractor_margin`` says.
    """
    from collections import Counter

    counts: dict[tuple[str, str], Counter] = {}
    for result in bundle["results"]:
        if result.get("family") != "unidentifiable_target":
            continue
        key = (str(result.get("params", {}).get("mode")), str(result.get("config")))
        bucket = counts.setdefault(key, Counter())
        hand = result.get("handover")
        if not hand:
            bucket["never"] += 1
            continue
        n_dets = hand.get("n_dets")
        bucket["n_dets=1" if n_dets == 1 else f"n_dets={n_dets}"] += 1
        bucket[str(hand.get("phase"))] += 1
    return [
        {"mode": mode, "config": config, **{k: int(v) for k, v in sorted(bucket.items())}}
        for (mode, config), bucket in sorted(counts.items())
    ]


def _trace_cell(column: str, value: Any) -> str:
    width = _TRACE_WIDTH.get(column, 10)
    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        text = f"{float(value):.2f}"
    if len(text) > width:
        text = text[: width - 1] + "~"
    return f"{text:>{width}s}"


def render_trace(rows: Sequence[dict[str, Any]], *, scenario: str = "", spec: str = "") -> str:
    """Human-readable per-frame table (one line per frame)."""
    out = [f"trace: {scenario} | spec {spec} | {len(rows)} frames"]
    out.append("  ".join(f"{c:>{_TRACE_WIDTH[c]}s}" for c in TRACE_COLUMNS))
    out.append("-" * len(out[-1]))
    for row in rows:
        out.append("  ".join(_trace_cell(c, row.get(c)) for c in TRACE_COLUMNS))
    return "\n".join(out)


def write_trace_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Write trace rows to ``path`` (same sanitiser as the sweep CSV)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRACE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in TRACE_COLUMNS})
    return p


def _state_histogram(statuses: dict[int, Any]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for st in statuses.values():
        key = str(getattr(st, "state", "?"))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items()))


def run_sweep(
    scenarios: Sequence[Scenario],
    specs: Sequence[MotionSpec] = DEFAULT_SPECS,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every (scenario, spec) pair.  Failures are recorded, not raised."""
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scenario in scenarios:
        for spec in specs:
            try:
                results.append(run_scenario(scenario, spec))
            except Exception as exc:  # noqa: BLE001 - a harness must survive one bad run
                errors.append({"scenario": scenario.name, "config": spec.label, "error": f"{type(exc).__name__}: {exc}"})
        if progress:
            print(f"  ran {scenario.name:<28} ({len(specs)} configs)", flush=True)
    return _bundle(scenarios, specs, results, errors)


def _bundle(
    scenarios: Sequence[Scenario],
    specs: Sequence[MotionSpec],
    results: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Turn raw per-run metric dicts into leaderboards + findings."""
    labels = [s.label for s in specs]
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in results}):
        rows = [r for r in results if r["family"] == family]
        by_family[family] = {
            label: combine_target([r["metrics"] for r in rows if r["config"] == label])
            for label in labels
            if any(r["config"] == label for r in rows)
        }
    leaderboard = {
        label: combine_target([r["metrics"] for r in results if r["config"] == label])
        for label in labels
        if any(r["config"] == label for r in results)
    }
    bundle = {
        "meta": {
            "harness": "scripts/ablate_motion.py",
            "n_scenarios": len(scenarios),
            "n_configs": len(specs),
            "n_runs": len(results),
            "n_errors": len(errors),
            "frames_simulated": int(sum(r["n_frames"] for r in results)),
            "appearance": "deterministic stub embedder (motion is the only variable)",
            "metric_source": "tgtrack.evaluate.evaluate_target",
            "deterministic": True,
        },
        "specs": [s.as_dict() for s in specs],
        "scenarios": [
            {
                "name": s.name,
                "family": s.family,
                "n_frames": s.n_frames,
                "params": dict(s.params),
                "note": s.note,
            }
            for s in scenarios
        ],
        "results": results,
        "leaderboard": leaderboard,
        "by_family": by_family,
        "errors": errors,
    }
    bundle["findings"] = findings(bundle)
    return bundle


# --------------------------------------------------------------------------- #
# Hard-overlap distractor sweep (similarity ladder x distractor_margin)
# --------------------------------------------------------------------------- #
def distractor_specs(
    models: Sequence[str] = DISTRACTOR_MODELS,
    margins: Sequence[float] = DISTRACTOR_MARGINS,
) -> list[MotionSpec]:
    """One spec per (model, ``distractor_margin``) cell.

    ``rollback_depth`` stays at the shipped default (0) so that the margin is the
    only thing that changes between cells.
    """
    return [
        MotionSpec(
            f"{model}-m{int(round(float(m) * 100)):02d}",
            model=model,
            rollback_depth=0,
            distractor_margin=float(m),
        )
        for model in models
        for m in margins
    ]


def _refusal_counts(rows: Sequence[dict[str, Any]], *, window: tuple[int, int] | None = None) -> dict[str, Any]:
    """Aggregate a trace into the numbers the margin question is asked with."""
    sel = list(rows)
    if window is not None:
        lo, hi = int(window[0]), int(window[1])
        sel = [r for r in sel if lo <= int(r["frame"]) < hi]
    reasons: dict[str, int] = {}
    for r in sel:
        key = str(r.get("reason") or "").split(":")[0].strip() or "none"
        reasons[key] = reasons.get(key, 0) + 1
    margins: list[float] = []
    for r in sel:
        best, rival = r.get("best_cost"), r.get("runner_up")
        if best is None or rival is None:
            continue
        try:
            margins.append(float(rival) - float(best))
        except (TypeError, ValueError):
            continue
    # The margin on the frames the tracker actually ACCEPTED -- this is the
    # number to compare against a live rebind trace: how much better the chosen
    # candidate was than its runner-up when the gate let it through.
    accepted_margins: list[float] = []
    for r in sel:
        if not r.get("accepted"):
            continue
        best, rival = r.get("best_cost"), r.get("runner_up")
        if best is None or rival is None:
            continue
        try:
            accepted_margins.append(float(rival) - float(best))
        except (TypeError, ValueError):
            continue
    return {
        "frames": len(sel),
        "ambiguous": int(reasons.get("ambiguous", 0)),
        "all_gated": int(reasons.get("all_gated", 0)),
        "accepted": int(sum(1 for r in sel if r.get("accepted"))),
        "margin_median": float(np.median(margins)) if margins else float("nan"),
        "margin_min": float(np.min(margins)) if margins else float("nan"),
        "accepted_margin_median": float(np.median(accepted_margins)) if accepted_margins else float("nan"),
        "accepted_margin_min": float(np.min(accepted_margins)) if accepted_margins else float("nan"),
        "accepted_margins": [float(m) for m in accepted_margins],
        "reasons": reasons,
    }


def run_distractor_sweep(
    scenarios: Sequence[Scenario] | None = None,
    *,
    cosines: Sequence[float] = DISTRACTOR_COSINES,
    margins: Sequence[float] = DISTRACTOR_MARGINS,
    models: Sequence[str] = DISTRACTOR_MODELS,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the similarity ladder against the margin grid, tracing every run.

    Every run is traced so the sweep can report *why* a cell behaved as it did:
    the refusal reasons and the measured best-vs-runner-up margin inside the
    overlap window.  Traces are aggregated and dropped -- they never reach
    ``raw.json``.
    """
    if scenarios is None:
        scenarios = [overlap_distractor(c) for c in cosines]
    specs = distractor_specs(models, margins)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scenario in scenarios:
        lo = int(scenario.params.get("n_pre", 0))
        hi = lo + int(scenario.params.get("n_overlap", 0))
        for spec in specs:
            try:
                result = run_scenario(scenario, spec, trace=True)
                rows = result.pop("trace")
                result["refusals"] = _refusal_counts(rows)
                result["refusals_overlap"] = _refusal_counts(rows, window=(lo, hi))
            except Exception as exc:  # noqa: BLE001 - a harness must survive one bad run
                errors.append({"scenario": scenario.name, "config": spec.label, "error": f"{type(exc).__name__}: {exc}"})
                continue
            results.append(result)
        if progress:
            print(f"  ran {scenario.name:<28} ({len(specs)} margin cells)", flush=True)

    bundle = _bundle(scenarios, specs, results, errors)
    bundle["meta"]["sweep"] = "hard-overlap distractor: similarity ladder x distractor_margin"
    bundle["meta"]["trace"] = "every run traced; traces aggregated into `refusals`/`refusals_overlap`"
    bundle["findings"] = distractor_findings(bundle)
    return bundle


def distractor_grid(
    bundle: dict[str, Any],
    *,
    column: str,
    model: str,
    source: str = "metrics",
) -> dict[str, Any]:
    """Rows = similarity, columns = ``distractor_margin``, cells = one number."""
    runs = [r for r in bundle["results"] if r.get("model") == model]
    margins = sorted({float(r["distractor_margin"]) for r in runs if r.get("distractor_margin") is not None})
    cosines = sorted({float(r["params"].get("similarity", float("nan"))) for r in runs})
    grid: list[list[float]] = []
    for cos in cosines:
        line: list[float] = []
        for margin in margins:
            cell = [
                r
                for r in runs
                if float(r["params"].get("similarity", float("nan"))) == cos
                and float(r["distractor_margin"]) == margin
            ]
            values = [(_f(r.get(source, {}).get(column)) if source != "metrics" else _f(r["metrics"].get(column))) for r in cell]
            values = [v for v in values if math.isfinite(v)]
            line.append(float(np.mean(values)) if values else float("nan"))
        grid.append(line)
    return {"model": model, "column": column, "source": source, "margins": margins, "cosines": cosines, "grid": grid}


def _cell_text(value: float, *, percent: bool, digits: int = 2) -> str:
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100.0:.1f}" if percent else f"{float(value):.{digits}f}"


#: (column, source, is-a-percentage, title)
DISTRACTOR_VIEWS: tuple[tuple[str, str, bool, str], ...] = (
    ("false_acquisition_rate", "metrics", True, "false_acquisition_rate (%)  [primary endpoint]"),
    ("target_success_rate", "metrics", True, "target_success_rate (%)"),
    ("id_consistency", "metrics", False, "id_consistency"),
    ("ambiguous", "refusals_overlap", False, "refused frames inside the overlap window (reason=ambiguous)"),
    ("margin_median", "refusals_overlap", False, "median measured margin  runner_up - best_cost  (overlap window)"),
)


def render_distractor_markdown(bundle: dict[str, Any]) -> str:
    """The similarity x margin sweep as markdown tables, one block per model."""
    meta = bundle["meta"]
    lines: list[str] = [
        "# Hard-overlap distractor sweep",
        "",
        "Two people ride on the same box for the whole overlap window (IoU 0.876), so motion, "
        "size and geometry are near-identical and only appearance can separate them.  The "
        "bystander's descriptor is correlated with the target's by the row's cosine while the "
        "target's own crop carries a fixed pose offset (0.97), so for high rows the bystander is "
        "*more* similar to the template than the target itself: those cells are where the "
        "`CostCfg.distractor_margin` refusal is the only thing standing between the tracker and a "
        "latch.  The bystander peels away after the window, which is what turns a latch into "
        "false acquisitions and lost success.",
        "",
        f"- runs: {meta.get('n_runs')} ({meta.get('n_scenarios')} similarity rows x "
        f"{meta.get('n_configs')} (model, margin) cells), errors: {meta.get('n_errors')}",
        f"- frames simulated: {meta.get('frames_simulated')}",
        f"- metrics: `{meta.get('metric_source')}` (never hand-rolled)",
        "- `target_success_rate` counts the whole scenario, so a latch is penalised twice: the "
        "frames during the latch and the frames after the peel.",
        "",
    ]
    for model in sorted({str(r.get("model")) for r in bundle["results"]}):
        lines.append(f"## model `{model}`  (rollback_depth = 0)")
        lines.append("")
        for column, source, percent, title in DISTRACTOR_VIEWS:
            data = distractor_grid(bundle, column=column, model=model, source=source)
            header = ["similarity \\ margin"] + [f"{m:.2f}" for m in data["margins"]]
            rows = [
                [f"{cos:.2f}"] + [_cell_text(v, percent=percent) for v in line]
                for cos, line in zip(data["cosines"], data["grid"])
            ]
            lines.append(f"### {title}")
            lines.append("")
            lines.extend(_md_table(header, rows))
            lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.extend(f"- {line}" for line in bundle.get("findings", []))
    lines.append("")
    return "\n".join(lines)


def distractor_findings(bundle: dict[str, Any]) -> list[str]:
    """Number-bearing answers to the margin question, per model."""
    out: list[str] = []
    for model in sorted({str(r.get("model")) for r in bundle["results"]}):
        fa = distractor_grid(bundle, column="false_acquisition_rate", model=model)
        ok = distractor_grid(bundle, column="target_success_rate", model=model)
        ref = distractor_grid(bundle, column="ambiguous", model=model, source="refusals_overlap")
        ref_margin = distractor_grid(bundle, column="margin_median", model=model, source="refusals_overlap")
        if not fa["cosines"]:
            continue
        worst = max(
            ((fa["grid"][i][j], fa["cosines"][i], fa["margins"][j]) for i in range(len(fa["cosines"])) for j in range(len(fa["margins"])) if math.isfinite(fa["grid"][i][j])),
            default=(float("nan"), float("nan"), float("nan")),
        )
        out.append(
            f"[{model}] worst cell: false_acquisition_rate {_cell_text(worst[0], percent=True)}% at "
            f"similarity {worst[1]:.2f}, distractor_margin {worst[2]:.2f}"
        )
        # Does the margin gate buy refusal?  Compare margin 0 against the largest margin.
        if len(fa["margins"]) >= 2:
            i = int(np.argmax(fa["cosines"]))
            base, gated = fa["grid"][i][0], fa["grid"][i][-1]
            cost = ok["grid"][i][0] - ok["grid"][i][-1]
            out.append(
                f"[{model}] most similar bystander (similarity {fa['cosines'][i]:.2f}): margin 0.00 -> "
                f"FA {_cell_text(base, percent=True)}%, success {_cell_text(ok['grid'][i][0], percent=True)}%; "
                f"margin {fa['margins'][-1]:.2f} -> FA {_cell_text(gated, percent=True)}%, "
                f"success {_cell_text(ok['grid'][i][-1], percent=True)}% "
                f"(success cost {_cell_text(cost, percent=True)} pp)"
            )
            refused = ref["grid"][i][-1]
            out.append(
                f"[{model}] refusals at similarity {fa['cosines'][i]:.2f}: "
                f"{_cell_text(ref['grid'][i][0], percent=False, digits=1)} frames at margin 0.00 vs "
                f"{_cell_text(refused, percent=False, digits=1)} at margin {fa['margins'][-1]:.2f}"
            )
            measured = ref_margin["grid"][i][0]
            out.append(
                f"[{model}] measured margin (runner_up - best_cost) at similarity {fa['cosines'][i]:.2f}: "
                f"median {_cell_text(measured, percent=False, digits=3)}, so a nominal distractor_margin of "
                f"{fa['margins'][-1]:.2f} refuses {_cell_text(refused, percent=False, digits=1)} overlap frames"
            )
        # Where does the ladder start to hurt?
        first_bad = None
        for i, cos in enumerate(fa["cosines"]):
            line = [v for v in fa["grid"][i] if math.isfinite(v)]
            if line and max(line) > 0.0 and first_bad is None:
                first_bad = cos
        all_cells = [v for row in fa["grid"] for v in row if math.isfinite(v)]
        peak = max(all_cells) if all_cells else float("nan")
        out.append(
            f"[{model}] first similarity row with any false acquisition: "
            + (
                f"{first_bad:.2f}"
                if first_bad is not None
                else f"none (max false_acquisition_rate over the ladder {_cell_text(peak, percent=True)}%)"
            )
        )
    return out


def render_readme(
    bundle: dict[str, Any],
    distractor: dict[str, Any] | None = None,
    unid: dict[str, Any] | None = None,
) -> str:
    """Human-readable index for this results directory."""
    meta = bundle["meta"]
    lines: list[str] = [
        "# Motion-strategy ablation (`scripts/ablate_motion.py`)",
        "",
        "Drives the real `tgtrack.target_tracker.TargetTracker` as a black box over deterministic "
        "synthetic sequences and scores every run with `tgtrack.evaluate.evaluate_target` "
        "(no hand-rolled metrics).  Appearance is a deterministic stub embedder, so **motion is the "
        "only variable**; the metric numbers still come from the same evaluator the rest of the "
        "project uses.",
        "",
        "## Run it",
        "",
        "```",
        "python scripts/ablate_motion.py                  # full sweep + distractor + unid sweeps + this readme",
        "python scripts/ablate_motion.py --quick          # smoke run",
        "python scripts/ablate_motion.py --no-distractor --no-unid   # main sweep only",
        "python scripts/ablate_motion.py --distractor     # hard-overlap similarity x margin grid only",
        "python scripts/ablate_motion.py --unid           # unidentifiable-target sweep only",
        "python scripts/ablate_motion.py --trace occlusion_distant_gap15 --no-write",
        "python -m pytest tests/test_ablate_motion.py",
        "```",
        "",
        "## Files",
        "",
        "| file | contents |",
        "| --- | --- |",
        "| `raw.json` | every run: config, scenario, full metric dict, state histogram |",
        "| `table.md` | leaderboard + per-family tables + findings |",
        "| `table.csv` | one row per (scenario, config) run |",
        "| `distractor_sweep.md` / `.csv` | the hard-overlap similarity x margin sweep |",
        "| `unid_sweep.md` / `.csv` | the unidentifiable-target x arming-rule sweep |",
        "| `readme.md` | this file |",
        "",
        f"## This run",
        "",
        f"- scenarios: {meta.get('n_scenarios')} | configs: {meta.get('n_configs')} | runs: {meta.get('n_runs')} | "
        f"errors: {meta.get('n_errors')} | frames: {meta.get('frames_simulated')}",
        f"- metrics: `{meta.get('metric_source')}`; appearance: {meta.get('appearance')}",
        "",
        "## Scenario families",
        "",
        "| family | what it is |",
        "| --- | --- |",
    ]
    for family in sorted(FAMILY_NOTES):
        lines.append(f"| `{family}` | {FAMILY_NOTES[family]} |")
    lines += [
        "",
        "## Config axes swept",
        "",
        "| config | model | nsa_adaptive | rollback_depth | imm_models |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in DEFAULT_SPECS:
        lines.append(
            f"| `{spec.label}` | {spec.model} | {spec.nsa_adaptive} | {spec.rollback_depth} | "
            f"{'+'.join(spec.imm_models)} |"
        )
    lines += [
        "",
        "## Headline (pooled over the families)",
        "",
    ]
    spec_lookup = _spec_lookup(bundle)
    lines.extend(_md_table(LEADERBOARD_HEADER, [
        _leaderboard_row(label, {**dict(bundle["leaderboard"][label]), **spec_lookup.get(label, {})})
        for label, _ in _rank(bundle["leaderboard"])
    ]))
    lines += ["", "### Findings", ""]
    lines.extend(f"- {line}" for line in bundle.get("findings", []))
    lines += [
        "",
        "## Where prediction quality is actually measurable",
        "",
        "With a single person and no detector drop, association always finds the target, so motion "
        "quality cannot change the outcome (that was the first sweep's null result).  The "
        "`*_blackout` families fix that: the detector drops the target while it stays in ground "
        "truth, so every reported box during the gap is pure motion prediction and success needs "
        "IoU>=0.5 with the true box.  `linear_blackout` is the control where extrapolation is exact.",
        "",
        "## Occlusion: how the tracker behaves when the target comes back far away",
        "",
        "`python scripts/ablate_motion.py --trace occlusion_distant_gap15,occlusion_distant_gap40 --no-write`",
        "",
        "A distant re-appearance is refused while the gate radius is smaller than the distance to the "
        "re-appeared target, so re-acquisition depends on the search arming.  After the fix that "
        "`StateCfg.global_states` (default `('lost_short','lost_long','gave_up','out_of_frame')`) makes "
        "`_global_search()` true from `lost_short` (fsu > 12) and `_gate_override()` uses the canonical "
        "`association.gate_radius` (whose `gate_base` is a FRACTION of the frame diagonal, ~604 px here), "
        "**all four gaps measured (5/15/40/90) re-acquire with latency 0.0**, and the previously "
        "never-re-acquiring gap 5/15 cells now succeed.  A re-appearance past the ~604 px gate (jump "
        "900) still shows the difference: shipped 0.8689 success / latency 8.0 at gap 5 and 1.0000 / 0.0 "
        "at gap 15/40, versus 0.3443 / never-re-acquire for the `state.global_states = ()` control.  "
        "`--trace` prints "
        "`frame, state, fsu, n_dets, gated, best_cost, thr, runner_up, accepted, reason, gate_r, "
        "pred_to_gt, gt_present, blind` per frame.",
        "",
    ]
    if distractor is not None:
        lines += [
            "## Hard-overlap distractor (the anti-latch claim)",
            "",
            "See `distractor_sweep.md` for the full similarity x `distractor_margin` grid and "
            "`distractor_sweep.csv` for the numbers.  Headline:",
            "",
        ]
        lines.extend(f"- {line}" for line in distractor.get("findings", []))
        lines.append("")
    if unid is not None:
        lines += [
            "## Unidentifiable target vs a bystander on the stale prediction (requirement 4)",
            "",
            "See `unid_sweep.md` for the similarity x arming-rule grids (success, false acquisition, "
            "accepted frames and the median ACCEPTED margin inside the blind window) and "
            "`unid_sweep.csv` for the numbers.  Headline:",
            "",
        ]
        lines.extend(f"- {line}" for line in unid.get("findings", []))
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- The stub embedder isolates motion/appearance interaction *by construction*; it is not a "
        "ReID benchmark.  It exists so that a motion change is the only difference between two runs.",
        "- `id_consistency` is geometric: `TargetTracker` does not put a track id in `_last_diag`, so "
        "the evaluator decides which identity a report latched onto by IoU against `all_gt`.",
        "- Frames are all-zero arrays: only box geometry is simulated, never pixels or crops.",
        "- `MotionModel.predict()` advances one frame per call and is not idempotent (`coast()` is); "
        "the harness calls it exactly once per frame through `TargetTracker.step`.",
        "",
    ]
    return "\n".join(lines)


def write_distractor_outputs(bundle: dict[str, Any], outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "distractor_sweep.md"
    csv_path = outdir / "distractor_sweep.csv"
    md_path.write_text(render_distractor_markdown(bundle), encoding="utf-8")
    rows = csv_rows(bundle)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"distractor_md": md_path, "distractor_csv": csv_path}


# --------------------------------------------------------------------------- #
# Unidentifiable-target sweep: unusable appearance + a bystander on the coast
# --------------------------------------------------------------------------- #
#: (column, source, is-a-percentage, title)
UNID_VIEWS: tuple[tuple[str, str, bool, str], ...] = (
    ("target_success_rate", "metrics", True, "target_success_rate (%)  [does the track come back?]"),
    ("false_acquisition_rate", "metrics", True, "false_acquisition_rate (%)  [handed the track to the bystander?]"),
    ("false_acquisition_frames", "metrics", False, "false_acquisition_frames (of n_blind + n_post)"),
    ("id_consistency", "metrics", False, "id_consistency"),
    ("target_switch_count", "metrics", False, "target_switch_count"),
    ("accepted", "refusals_blind", False, "accepted frames inside the blind window"),
    ("accepted_margin_median", "refusals_blind", False,
     "median ACCEPTED margin (runner_up - best_cost) inside the blind window"),
    ("margin_median", "refusals_blind", False,
     "median margin over all blind-window frames (accepted or not)"),
)


def unid_specs(
    models: Sequence[str] = UNID_MODELS,
    sweeps: Sequence[tuple[str, dict[str, Any] | None]] = UNID_SWEEPS,
) -> list[MotionSpec]:
    """One spec per (model, arming rule); label ``<model>-<rule>``.

    ``rollback_depth`` stays at the shipped 0 so the arming rule is the only
    non-motion knob that moves between cells.
    """
    return [
        MotionSpec(
            f"{model}-{label}",
            model=model,
            rollback_depth=0,
            cfg_overrides=tuple(sorted(dict(overrides or {}).items())),
        )
        for model in models
        for label, overrides in sweeps
    ]


def unid_cost_specs(
    models: Sequence[str] = UNID_MODELS,
    costs: Sequence[float] = MISSING_COSTS,
) -> list[MotionSpec]:
    """One spec per (model, ``missing_appearance_cost``); label ``<model>-c<nnn>``.

    ``rollback_depth`` stays at the shipped 0 and nothing else moves, so the
    price of an unreadable candidate is the only knob that changes.
    """
    return [
        MotionSpec(
            f"{model}-c{int(round(cost * 100)):03d}",
            model=model,
            rollback_depth=0,
            cfg_overrides=(("cost.missing_appearance_cost", float(cost)),),
        )
        for model in models
        for cost in costs
    ]


def _unid_scenarios(
    modes: Sequence[str],
    similarities: Sequence[float],
) -> list[Scenario]:
    """The unid family: one target run per (mode, similarity) plus its solo control."""
    scenarios: list[Scenario] = []
    for mode in modes:
        for cos in similarities:
            scenarios.append(unidentifiable_target(cos, mode=mode))
        scenarios.append(unidentifiable_solo(mode=mode))
    return scenarios


def _unid_window(scenario: Scenario) -> tuple[int, int]:
    p = scenario.params
    lo = int(p.get("n_pre", 0)) + int(p.get("n_gap", 0))
    return lo, lo + int(p.get("n_blind", 0))


def run_unid_sweep(
    scenarios: Sequence[Scenario] | None = None,
    *,
    modes: Sequence[str] = ("missing",),
    similarities: Sequence[float] = UNID_SIMILARITIES,
    models: Sequence[str] = UNID_MODELS,
    sweeps: Sequence[tuple[str, dict[str, Any] | None]] = UNID_SWEEPS,
    progress: bool = True,
) -> dict[str, Any]:
    """Sweep similarity x (model, arming rule), tracing every run.

    Every run is traced so the sweep can report how many frames inside the blind
    window were actually accepted (i.e. whether the tracker took the bystander's
    box) rather than inferring it from the end-of-scenario metrics.
    """
    if scenarios is None:
        scenarios = _unid_scenarios(modes, similarities)
    specs = unid_specs(models, sweeps)
    bundle = _run_unid(
        scenarios,
        specs,
        sweep="unidentifiable target: similarity x (model, arming rule)",
        progress=progress,
    )
    bundle["findings"] = unid_findings(bundle)
    return bundle


def _run_unid(
    scenarios: Sequence[Scenario],
    specs: Sequence[MotionSpec],
    *,
    sweep: str,
    progress: bool = True,
) -> dict[str, Any]:
    """Run one unid grid, tracing every run and recording the handover frame.

    ``handover`` is the first frame at/after the target stops being observed
    whose *accepted* report sits on the bystander by IoU -- the same attribution
    ``evaluate_target`` uses.  Its ``n_dets`` separates the two suspected holes:
    ``n_dets == 1`` means the bystander was the only finite candidate, so
    ``association.select_candidate`` left ``margin = inf`` and the ambiguity test
    could not fire whatever ``distractor_margin`` says; ``n_dets >= 2`` means the
    runner-up was present and priced.
    """
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for scenario in scenarios:
        lo, hi = _unid_window(scenario)
        n_blind = max(1, hi - lo)
        start = int(scenario.params.get("n_pre", 0))
        for spec in specs:
            try:
                result = run_scenario(scenario, spec, trace=True)
                rows = result.pop("trace")
                result["refusals"] = _refusal_counts(rows)
                result["refusals_blind"] = _refusal_counts(rows, window=(lo, hi))
                result["blind_ratio"] = float(result["refusals_blind"]["accepted"]) / n_blind
                result["handover"] = handover_frame(rows, start=start)
            except Exception as exc:  # noqa: BLE001 - a harness must survive one bad run
                errors.append({"scenario": scenario.name, "config": spec.label, "error": f"{type(exc).__name__}: {exc}"})
                continue
            results.append(result)
        if progress:
            print(f"  ran {scenario.name:<34} ({len(specs)} cells)", flush=True)

    bundle = _bundle(scenarios, specs, results, errors)
    bundle["meta"]["sweep"] = sweep
    bundle["meta"]["trace"] = (
        "every run traced; traces aggregated into `refusals`/`refusals_blind` "
        "and reduced to a `handover` record"
    )
    return bundle


def run_unid_cost_sweep(
    scenarios: Sequence[Scenario] | None = None,
    *,
    modes: Sequence[str] = ("missing",),
    similarities: Sequence[float] = UNID_SIMILARITIES,
    models: Sequence[str] = UNID_MODELS,
    costs: Sequence[float] = MISSING_COSTS,
    progress: bool = True,
) -> dict[str, Any]:
    """Sweep similarity x (model, ``CostCfg.missing_appearance_cost``).

    The shipped price for an unreadable candidate is 0.5, i.e. cheaper than a
    genuine mismatch; this grid asks whether pricing it at or above the accept
    budget stops the tracker handing the track to the bystander.
    """
    if scenarios is None:
        scenarios = _unid_scenarios(modes, similarities)
    specs = unid_cost_specs(models, costs)
    bundle = _run_unid(
        scenarios,
        specs,
        sweep="unidentifiable target: similarity x (model, missing_appearance_cost)",
        progress=progress,
    )
    bundle["findings"] = unid_cost_findings(bundle)
    return bundle


def unid_grid(
    bundle: dict[str, Any],
    *,
    column: str,
    mode: str,
    model: str,
    source: str = "metrics",
) -> dict[str, Any]:
    """Rows = similarity rung, columns = arming rule, cells = one number."""
    runs = [
        r
        for r in bundle["results"]
        if r.get("family") == "unidentifiable_target"
        and str(r.get("model")) == model
        and str(r.get("params", {}).get("mode")) == mode
    ]
    cols = sorted({str(r.get("config", "")).split("-", 1)[-1] for r in runs})
    sims = sorted({_f(r.get("params", {}).get("similarity")) for r in runs})
    grid: list[list[float]] = []
    for cos in sims:
        line: list[float] = []
        for col in cols:
            cell = [
                r
                for r in runs
                if _f(r.get("params", {}).get("similarity")) == cos
                and str(r.get("config", "")).split("-", 1)[-1] == col
            ]
            values = [
                (_f(r.get(source, {}).get(column)) if source != "metrics" else _f(r["metrics"].get(column)))
                for r in cell
            ]
            values = [v for v in values if math.isfinite(v)]
            line.append(float(np.mean(values)) if values else float("nan"))
        grid.append(line)
    return {"mode": mode, "model": model, "column": column, "source": source, "cols": cols, "cosines": sims, "grid": grid}


def unid_solo_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """The no-bystander control, one row per (mode, model, rule)."""
    kept = [r for r in bundle["results"] if r.get("family") == "unidentifiable_solo"]
    return [
        {
            "mode": str(r.get("params", {}).get("mode")),
            "model": str(r.get("model")),
            "config": str(r.get("config")),
            "target_success_rate": _f(r["metrics"].get("target_success_rate")),
            "false_acquisition_rate": _f(r["metrics"].get("false_acquisition_rate")),
            "id_consistency": _f(r["metrics"].get("id_consistency")),
            "accepted_blind": _f(r.get("refusals_blind", {}).get("accepted")),
        }
        for r in kept
    ]


def render_unid_markdown(bundle: dict[str, Any]) -> str:
    """The unidentifiable-target sweep as markdown tables."""
    meta = bundle["meta"]
    modes = sorted({str(r.get("params", {}).get("mode")) for r in bundle["results"]})
    models = sorted({str(r.get("model")) for r in bundle["results"]})
    lines: list[str] = [
        "# Unidentifiable target + identifiable bystander on the stale prediction",
        "",
        "The target walks, HALTS, and then stops being identifiable (its descriptor is missing -- "
        "`encode` returns no features, which is the tracker's `appearance_valid=False` case -- or is "
        "valid but orthogonal).  The tracker's constant-velocity coast keeps advancing after the halt, "
        "and an **unoccluded bystander with a clean descriptor rides that stale prediction**, so "
        "geometry points at the bystander and appearance is the only cue that can refuse.  The target "
        "becomes identifiable again in the last frames, so `target_success_rate` measures whether the "
        "track came back at all.",
        "",
        f"- runs: {meta.get('n_runs')} ({meta.get('n_scenarios')} scenarios x {meta.get('n_configs')} cells), "
        f"errors: {meta.get('n_errors')}",
        f"- frames simulated: {meta.get('frames_simulated')}",
        f"- metrics: `{meta.get('metric_source')}` (never hand-rolled)",
        "- `shipped` = the current arming rule (`state.global_states` defaults to "
        "`('lost_short','lost_long','gave_up','out_of_frame')`); `no_early_search` = "
        "`state.global_states = ()`, the pre-fix rule that only searches after `lost_long_after` frames.",
        "- `id_consistency` is `nan` (rendered `-`) when no report is correct enough to name an identity.",
        "",
    ]
    for mode in modes:
        lines.append(f"## mode `{mode}`")
        lines.append("")
        for model in models:
            lines.append(f"### model `{model}`")
            lines.append("")
            for column, source, percent, title in UNID_VIEWS:
                data = unid_grid(bundle, column=column, mode=mode, model=model, source=source)
                if not data["cosines"]:
                    continue
                header = ["similarity \\ arming rule"] + list(data["cols"])
                rows = [
                    [f"{cos:.2f}"] + [_cell_text(v, percent=percent) for v in line]
                    for cos, line in zip(data["cosines"], data["grid"])
                ]
                lines.append(f"#### {title}")
                lines.append("")
                lines.extend(_md_table(header, rows))
                lines.append("")
    solo = unid_solo_rows(bundle)
    if solo:
        lines.append("## control: no bystander (`unidentifiable_solo`)")
        lines.append("")
        lines.extend(
            _md_table(
                ["mode", "model", "config", "success", "false_acq", "id_consistency", "accepted in blind window"],
                [
                    [
                        r["mode"],
                        r["model"],
                        r["config"],
                        _cell_text(r["target_success_rate"], percent=True),
                        _cell_text(r["false_acquisition_rate"], percent=True),
                        _cell_text(r["id_consistency"], percent=False),
                        _cell_text(r["accepted_blind"], percent=False, digits=0),
                    ]
                    for r in solo
                ],
            )
        )
        lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.extend(f"- {line}" for line in bundle.get("findings", []))
    lines.append("")
    return "\n".join(lines)


def unid_findings(bundle: dict[str, Any]) -> list[str]:
    """Number-bearing answers to the requirement-(4) question."""
    out: list[str] = []
    modes = sorted({str(r.get("params", {}).get("mode")) for r in bundle["results"]})
    models = sorted({str(r.get("model")) for r in bundle["results"]})
    worst_far = -1.0
    worst_where = "n/a"
    for mode in modes:
        for model in models:
            success = unid_grid(bundle, column="target_success_rate", mode=mode, model=model)
            if not success["cosines"]:
                continue
            far = unid_grid(bundle, column="false_acquisition_rate", mode=mode, model=model)
            accepted = unid_grid(bundle, column="accepted", mode=mode, model=model, source="refusals_blind")
            acc_margin = unid_grid(
                bundle, column="accepted_margin_median", mode=mode, model=model, source="refusals_blind"
            )
            for cos, s_line, f_line, a_line, m_line in zip(
                success["cosines"], success["grid"], far["grid"], accepted["grid"], acc_margin["grid"]
            ):
                for col, s_val, f_val, a_val, m_val in zip(success["cols"], s_line, f_line, a_line, m_line):
                    if math.isfinite(f_val) and f_val > worst_far:
                        worst_far, worst_where = f_val, f"mode={mode} model={model} rung={col} cos={cos:.2f}"
                    out.append(
                        f"UNIDENTIFIABLE mode={mode} model={model} rule={col} similarity={cos:.2f}: "
                        f"success {_pct(s_val)}, false acquisition {_pct(f_val)}, "
                        f"accepted {_num(a_val, 0)} frames inside the blind window, "
                        f"median accepted margin {_num(m_val, 3)}"
                    )
    out.append(
        f"UNIDENTIFIABLE worst false-acquisition cell (all modes/models/rules): "
        f"{_pct(worst_far) if worst_far >= 0 else 'n/a'} at {worst_where}"
    )
    solo = unid_solo_rows(bundle)
    by_key = {(r["mode"], r["model"], r["config"].split("-", 1)[-1]): r for r in solo}
    for mode in modes:
        for model in models:
            shipped = by_key.get((mode, model, "shipped"))
            control = by_key.get((mode, model, "no_early_search"))
            if shipped is None:
                continue
            line = (
                f"UNIDENTIFIABLE control (no bystander) mode={mode} model={model}: "
                f"success {_pct(shipped['target_success_rate'])}, "
                f"false acquisition {_pct(shipped['false_acquisition_rate'])}"
            )
            if control is not None and math.isfinite(control["target_success_rate"]):
                line += (
                    f"; with the pre-fix rule {_pct(control['target_success_rate'])} "
                    f"({_pp(shipped['target_success_rate'], control['target_success_rate'])})"
                )
            out.append(line)
    for mode in modes:
        for model in models:
            for column, label in (
                ("target_success_rate", "success"),
                ("false_acquisition_rate", "false acquisition"),
            ):
                shipped = unid_grid(bundle, column=column, mode=mode, model=model)
                if not shipped["cosines"] or len(shipped["cols"]) < 2:
                    continue
                i = shipped["cols"].index("shipped") if "shipped" in shipped["cols"] else 0
                j = 1 - i if len(shipped["cols"]) == 2 else i
                deltas = [
                    (abs(line[i] - line[j]), cos)
                    for cos, line in zip(shipped["cosines"], shipped["grid"])
                    if math.isfinite(line[i]) and math.isfinite(line[j])
                ]
                if deltas:
                    worst, cos = max(deltas)
                    out.append(
                        f"UNIDENTIFIABLE arming-rule effect mode={mode} model={model} metric={label}: "
                        f"largest |shipped - no_early_search| = {worst * 100.0:.1f} pp at similarity {cos:.2f} "
                        f"(shipped {_pct(shipped['grid'][shipped['cosines'].index(cos)][i])} vs "
                        f"{_pct(shipped['grid'][shipped['cosines'].index(cos)][j])})"
                    )
    return out


def write_unid_outputs(bundle: dict[str, Any], outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "unid_sweep.md"
    csv_path = outdir / "unid_sweep.csv"
    md_path.write_text(render_unid_markdown(bundle), encoding="utf-8")
    rows = csv_rows(bundle)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"unid_md": md_path, "unid_csv": csv_path}


def unid_cost_grid(
    bundle: dict[str, Any],
    *,
    column: str,
    mode: str,
    model: str,
    source: str = "metrics",
) -> dict[str, Any]:
    """Rows = similarity rung, columns = ``missing_appearance_cost``."""
    runs = [
        r
        for r in bundle["results"]
        if r.get("family") == "unidentifiable_target"
        and str(r.get("model")) == model
        and str(r.get("params", {}).get("mode")) == mode
    ]
    cols = sorted({str(r.get("config", "")).split("-", 1)[-1] for r in runs})
    sims = sorted({_f(r.get("params", {}).get("similarity")) for r in runs})
    grid: list[list[float]] = []
    for cos in sims:
        line: list[float] = []
        for col in cols:
            cell = [
                r
                for r in runs
                if _f(r.get("params", {}).get("similarity")) == cos
                and str(r.get("config", "")).split("-", 1)[-1] == col
            ]
            values: list[float] = []
            for r in cell:
                container = r.get(source) or {}
                if not isinstance(container, dict):
                    continue
                values.append(_f(container.get(column)))
            values = [v for v in values if math.isfinite(v)]
            line.append(float(np.mean(values)) if values else float("nan"))
        grid.append(line)
    return {
        "mode": mode,
        "model": model,
        "column": column,
        "source": source,
        "cols": cols,
        "cosines": sims,
        "grid": grid,
    }


#: (column, source, is-percent, title) for the ``missing_appearance_cost`` tables.
UNID_COST_VIEWS: tuple[tuple[str, str, bool, str], ...] = (
    ("target_success_rate", "metrics", True, "target success rate (%)"),
    ("false_acquisition_rate", "metrics", True, "false acquisition rate (%)"),
    ("false_acquisition_frames", "metrics", False, "false-acquisition frames"),
    ("id_consistency", "metrics", False, "id consistency"),
    ("frame", "handover", False, "handover frame (first frame on the bystander)"),
    ("n_dets", "handover", False, "n_dets at the handover"),
    (
        "accepted_margin_median",
        "refusals_blind",
        False,
        "median accepted margin inside the blind window",
    ),
)


def _unid_modes_models(bundle: dict[str, Any]) -> tuple[list[str], list[str]]:
    runs = [r for r in bundle["results"] if r.get("family") == "unidentifiable_target"]
    modes = sorted({str(r.get("params", {}).get("mode")) for r in runs})
    models = sorted({str(r.get("model")) for r in runs})
    return modes, models


def render_unid_cost_markdown(bundle: dict[str, Any]) -> str:
    meta = bundle.get("meta", {})
    lines = [
        "# Unidentifiable target: `missing_appearance_cost` x similarity",
        "",
        "The `unidentifiable_target` family makes the target's descriptor unusable",
        "for a stretch of frames while an unoccluded bystander rides the stale",
        "prediction.  `CostCfg.missing_appearance_cost` is the price charged to a",
        "candidate whose appearance could not be read at all; the shipped 0.5 makes",
        "it *cheaper* than a genuine mismatch (cosine 1.0 -> 1.6), so \"we could not",
        "read this person\" can win the selection.  This grid sweeps that price.",
        "",
        f"- runs: {meta.get('n_runs')} "
        f"({meta.get('n_scenarios')} scenarios x {meta.get('n_configs')} configs), "
        f"errors: {meta.get('n_errors')}",
        "- handover = first frame whose *accepted* report sits closer (by IoU) to the",
        "  bystander than to the target, starting from the frame the target stops",
        "  being observed",
        "",
    ]
    if not bundle.get("results"):
        lines.append("_no results_")
        return "\n".join(lines)

    split = handover_split(bundle)
    if split:
        lines += [
            "## Where the handover happens",
            "",
            "`n_dets=1` means the bystander was the only finite candidate, so",
            "`association.select_candidate` left `margin = inf` and the ambiguity test",
            "could not fire at any `distractor_margin`; `n_dets=2` means the runner-up",
            "was present and priced.",
            "",
            "| mode | config | n_dets=1 | n_dets=2 | never | in the gap | in the blind window |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in split:
            lines.append(
                f"| {row.get('mode')} | {row.get('config')} | {row.get('n_dets=1', 0)} | "
                f"{row.get('n_dets=2', 0)} | {row.get('never', 0)} | {row.get('gap', 0)} | "
                f"{row.get('blind', 0)} |"
            )
        lines.append("")

    modes, models = _unid_modes_models(bundle)
    for model in models:
        lines += [f"## model `{model}`", ""]
        for mode in modes:
            for column, source, percent, title in UNID_COST_VIEWS:
                grid = unid_cost_grid(
                    bundle, column=column, mode=mode, model=model, source=source
                )
                if not grid["cols"]:
                    continue
                lines += [f"### {mode}: {title}", ""]
                lines.append("| similarity | " + " | ".join(grid["cols"]) + " |")
                lines.append("|" + "---|" * (len(grid["cols"]) + 1))
                for cos, row in zip(grid["cosines"], grid["grid"]):
                    cells = [_cell_text(v, percent=percent, digits=2) for v in row]
                    lines.append(f"| {cos:.2f} | " + " | ".join(cells) + " |")
                lines.append("")

    lines += ["## Findings", ""]
    for line in bundle.get("findings", []):
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def unid_cost_findings(bundle: dict[str, Any]) -> list[str]:
    out: list[str] = []
    split = handover_split(bundle)
    n1 = sum(int(r.get("n_dets=1", 0)) for r in split)
    n2 = sum(int(r.get("n_dets=2", 0)) for r in split)
    never = sum(int(r.get("never", 0)) for r in split)
    out.append(
        f"handovers split by n_dets: n=1 {n1} run(s) (lone finite candidate, "
        f"margin=inf so the ambiguity test is vacuous), n=2 {n2} run(s) "
        f"(runner-up present and priced), never {never} run(s)"
    )
    modes, models = _unid_modes_models(bundle)
    for model in models:
        for mode in modes:
            tsr = unid_cost_grid(bundle, column="target_success_rate", mode=mode, model=model)
            far = unid_cost_grid(bundle, column="false_acquisition_rate", mode=mode, model=model)
            if not tsr["cosines"]:
                continue
            idx = len(tsr["cosines"]) - 1
            cells = [
                f"{col}: TSR {_pct(tsr['grid'][idx][j])} / FAR {_pct(far['grid'][idx][j])}"
                for j, col in enumerate(tsr["cols"])
            ]
            out.append(
                f"{mode}/{model} at similarity {tsr['cosines'][idx]:.2f} - " + "; ".join(cells)
            )
            finite = [
                far["grid"][idx][j]
                for j in range(len(far["cols"]))
                if math.isfinite(far["grid"][idx][j])
            ]
            if finite:
                best = min(finite)
                out.append(
                    f"{mode}/{model}: best false-acquisition rate over the whole price axis "
                    f"= {_pct(best)} -> "
                    + (
                        "the handover closes"
                        if best <= 0.0
                        else "pricing alone does NOT close the handover"
                    )
                )
    return out


def write_unid_cost_outputs(
    bundle: dict[str, Any], outdir: str | Path
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "unid_cost.md"
    csv_path = outdir / "unid_cost.csv"
    md_path.write_text(render_unid_cost_markdown(bundle), encoding="utf-8")
    rows = csv_rows(bundle)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"unid_cost_md": md_path, "unid_cost_csv": csv_path}


# --------------------------------------------------------------------------- #
# Lone-candidate guard (``CostCfg.lone_candidate_accept_cost``)
# --------------------------------------------------------------------------- #
LONE_BOUNDS: tuple[float | None, ...] = (0.2, 0.3, 0.4, 0.5, None)


def _lone_label(bound: float | None) -> str:
    """Config-label suffix for one bound; ``None`` is the inert control."""
    return "none" if bound is None else f"{int(round(float(bound) * 100)):03d}"


def lone_specs(
    models: Sequence[str] = ("kf", "imm"),
    bounds: Sequence[float | None] = LONE_BOUNDS,
) -> list[MotionSpec]:
    """One spec per (model, ``lone_candidate_accept_cost``); label ``<model>-l<bound>``.

    ``None`` sets an EXPLICIT override to ``None`` (bound off) rather than leaving
    the shipped default alone. That distinction matters: once the shipped default
    became a live bound, "no override" silently turned the off control into a
    second copy of the treatment, which is exactly how a sweep stops measuring
    what it claims to. Every cell moves exactly one knob, with ``rollback_depth``
    pinned at the shipped 0.
    """
    return [
        MotionSpec(
            f"{model}-l{_lone_label(bound)}",
            model=model,
            rollback_depth=0,
            cfg_overrides=(
                ("cost.lone_candidate_accept_cost", None if bound is None else float(bound)),
            ),
        )
        for model in models
        for bound in bounds
    ]


def clean_scenarios() -> list[Scenario]:
    """The default set minus the unid families (the guard's cost elsewhere)."""
    return [s for s in default_scenarios() if not s.family.startswith("unidentifiable")]


def run_lone_sweep(
    scenarios: Sequence[Scenario] | None = None,
    *,
    modes: Sequence[str] = ("missing", "orthogonal"),
    similarities: Sequence[float] = UNID_SIMILARITIES,
    models: Sequence[str] = ("kf", "imm"),
    bounds: Sequence[float | None] = LONE_BOUNDS,
    progress: bool = True,
) -> dict[str, Any]:
    """The unid family x lone-candidate bound, traced like every unid sweep."""
    if scenarios is None:
        scenarios = _unid_scenarios(modes, similarities)
    specs = lone_specs(models, bounds)
    bundle = _run_unid(
        scenarios,
        specs,
        sweep="unid family x CostCfg.lone_candidate_accept_cost",
        progress=progress,
    )
    bundle["findings"] = lone_findings(bundle)
    return bundle


def run_lone_clean_sweep(
    scenarios: Sequence[Scenario] | None = None,
    *,
    bounds: Sequence[float | None] = LONE_BOUNDS,
    model: str = "imm",
    progress: bool = True,
) -> dict[str, Any]:
    """The same bounds on the clean families, to price the guard there."""
    if scenarios is None:
        scenarios = clean_scenarios()
    bundle = run_sweep(scenarios, lone_specs((model,), bounds), progress=progress)
    bundle["findings"] = lone_clean_findings(bundle)
    return bundle


def lone_handover_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per target-detectable run: the handover frame and its ``n_dets``."""
    out: list[dict[str, Any]] = []
    for r in bundle["results"]:
        if r.get("family") != "unidentifiable_target":
            continue
        hand = r.get("handover") or {}
        m = r.get("metrics", {})
        out.append(
            {
                "mode": str(r.get("params", {}).get("mode")),
                "model": str(r.get("model")),
                "bound": str(r.get("config", "")).split("-", 1)[-1],
                "similarity": _f(r.get("params", {}).get("similarity")),
                "frame": hand.get("frame"),
                "n_dets": hand.get("n_dets"),
                "phase": hand.get("phase"),
                "tsr": _f(m.get("target_success_rate")),
                "far": _f(m.get("false_acquisition_rate")),
                "idc": _f(m.get("id_consistency")),
            }
        )
    return out


def lone_clean_table(clean: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-bound pooled numbers on the clean families."""
    rows: list[dict[str, Any]] = []
    for label, metrics in clean["leaderboard"].items():
        rows.append(
            {
                "config": str(label),
                "bound": str(label).split("-", 1)[-1],
                "success": _f(metrics.get("target_success_rate")),
                "iou": _f(metrics.get("target_precision")),
                "loss": _f(metrics.get("loss_rate")),
                "reacq": _f(metrics.get("reacquire_rate")),
                "far": _f(metrics.get("false_acquisition_rate")),
                "idc": _f(metrics.get("id_consistency")),
            }
        )
    return sorted(rows, key=lambda r: r["config"])


def lone_findings(bundle: dict[str, Any]) -> list[str]:
    """Number-bearing answers: does a bound remove the handover, and at what cost?"""
    rows = lone_handover_rows(bundle)
    solo = unid_solo_rows(bundle)
    modes = sorted({r["mode"] for r in rows})
    models = sorted({r["model"] for r in rows})
    bounds = sorted({r["bound"] for r in rows})
    out: list[str] = []
    for mode in modes:
        for model in models:
            best: tuple[float, str] = (float("inf"), "n/a")
            for bound in bounds:
                cell = [
                    r
                    for r in rows
                    if r["mode"] == mode and r["model"] == model and r["bound"] == bound
                ]
                if not cell:
                    continue
                hands = [r for r in cell if r["frame"] is not None]
                far = [r["far"] for r in cell if math.isfinite(r["far"])]
                peak = max(far) if far else float("nan")
                if math.isfinite(peak) and peak < best[0]:
                    best = (peak, bound)
                n1 = sum(1 for r in hands if r["n_dets"] == 1)
                n2 = sum(1 for r in hands if (r["n_dets"] or 0) >= 2)
                frames = sorted({int(r["frame"]) for r in hands})
                out.append(
                    f"LONE mode={mode} model={model} bound={bound}: "
                    f"{len(hands)}/{len(cell)} runs hand over (n_dets=1: {n1}, n_dets>=2: {n2}), "
                    f"handover frames {frames if frames else 'none'}, "
                    f"peak false acquisition {_pct(peak)}"
                )
            if math.isfinite(best[0]):
                out.append(
                    f"LONE mode={mode} model={model}: lowest peak false acquisition {_pct(best[0])} "
                    f"at bound {best[1]}"
                )
    by_key = {(r["mode"], r["model"], r["config"].split("-", 1)[-1]): r for r in solo}
    for mode in modes:
        for model in models:
            for bound in bounds:
                r = by_key.get((mode, model, bound))
                if r is None:
                    continue
                out.append(
                    f"LONE control (no bystander) mode={mode} model={model} bound={bound}: "
                    f"success {_pct(r['target_success_rate'])}, "
                    f"false acquisition {_pct(r['false_acquisition_rate'])}"
                )
    return out


def lone_clean_findings(clean: dict[str, Any]) -> list[str]:
    """What the guard costs on the families where it has nothing to refuse."""
    rows = lone_clean_table(clean)
    base = next((r for r in rows if r["bound"] == "lnone"), None)
    out: list[str] = []
    for r in rows:
        line = (
            f"LONE clean families bound={r['bound']}: success {_pct(r['success'])}, "
            f"IoU {_num(r['iou'], 3)}, loss {_pct(r['loss'])}, "
            f"false acquisition {_pct(r['far'])}, id_consistency {_num(r['idc'], 3)}"
        )
        if base is not None and r is not base:
            line += (
                f" (success {_pp(r['success'], base['success'])} vs the inert control, "
                f"false acquisition {_pp(r['far'], base['far'])})"
            )
        out.append(line)
    return out


def render_lone_markdown(bundle: dict[str, Any], clean: dict[str, Any] | None = None) -> str:
    """The lone-candidate bound sweep as markdown tables."""
    meta = bundle["meta"]
    modes = sorted({str(r.get("params", {}).get("mode")) for r in bundle["results"]})
    models = sorted({str(r.get("model")) for r in bundle["results"]})
    lines: list[str] = [
        "# Lone-candidate guard: `CostCfg.lone_candidate_accept_cost`",
        "",
        "On the unid family the bystander is usually the ONLY finite candidate, so "
        "`association.select_candidate` leaves `margin = inf` and the ambiguity test cannot fire "
        "whatever `distractor_margin` says.  `lone_candidate_accept_cost` refuses an association "
        "when exactly one candidate is finite and its cost exceeds the bound.  This grid asks "
        "whether a bound REMOVES the handover (false acquisition -> 0) or merely moves it, and what "
        "it costs on the clean families.",
        "",
        f"- runs: {meta.get('n_runs')} ({meta.get('n_scenarios')} scenarios x {meta.get('n_configs')} cells), "
        f"errors: {meta.get('n_errors')}",
        f"- frames simulated: {meta.get('frames_simulated')}",
        "- the `lnone` column is the inert shipped default (`lone_candidate_accept_cost = None`).",
        "",
    ]
    for mode in modes:
        lines.append(f"## mode `{mode}`")
        lines.append("")
        for model in models:
            lines.append(f"### model `{model}`")
            lines.append("")
            for column, source, percent, title in UNID_VIEWS:
                data = unid_grid(bundle, column=column, mode=mode, model=model, source=source)
                if not data["cosines"]:
                    continue
                header = ["similarity \\ bound"] + list(data["cols"])
                rows = [
                    [f"{cos:.2f}"] + [_cell_text(v, percent=percent) for v in line]
                    for cos, line in zip(data["cosines"], data["grid"])
                ]
                lines.append(f"#### {title}")
                lines.append("")
                lines.extend(_md_table(header, rows))
                lines.append("")
    rows = lone_handover_rows(bundle)
    if rows:
        lines.append("## handover frame and `n_dets` at handover")
        lines.append("")
        lines.extend(
            _md_table(
                ["mode", "model", "bound", "similarity", "handover frame", "n_dets", "phase", "success", "false_acq"],
                [
                    [
                        r["mode"],
                        r["model"],
                        r["bound"],
                        f"{r['similarity']:.2f}",
                        "" if r["frame"] is None else str(r["frame"]),
                        "" if r["n_dets"] is None else str(r["n_dets"]),
                        "" if r["phase"] is None else str(r["phase"]),
                        _cell_text(r["tsr"], percent=True),
                        _cell_text(r["far"], percent=True),
                    ]
                    for r in sorted(rows, key=lambda r: (r["mode"], r["model"], r["bound"], r["similarity"]))
                ],
            )
        )
        lines.append("")
    solo = unid_solo_rows(bundle)
    if solo:
        lines.append("## control: no bystander (`unidentifiable_solo`)")
        lines.append("")
        lines.extend(
            _md_table(
                ["mode", "model", "bound", "success", "false_acq", "id_consistency", "accepted in blind window"],
                [
                    [
                        r["mode"],
                        r["model"],
                        r["config"].split("-", 1)[-1],
                        _cell_text(r["target_success_rate"], percent=True),
                        _cell_text(r["false_acquisition_rate"], percent=True),
                        _cell_text(r["id_consistency"], percent=False),
                        _cell_text(r["accepted_blind"], percent=False, digits=0),
                    ]
                    for r in solo
                ],
            )
        )
        lines.append("")
    if clean is not None:
        lines.append("## cost on the clean families (no unid scenarios)")
        lines.append("")
        lines.extend(
            _md_table(
                ["bound", "success", "IoU", "loss", "re-acquire rate", "false_acq", "id_consistency"],
                [
                    [
                        r["bound"],
                        _cell_text(r["success"], percent=True),
                        _cell_text(r["iou"], percent=False, digits=3),
                        _cell_text(r["loss"], percent=True),
                        _cell_text(r["reacq"], percent=True),
                        _cell_text(r["far"], percent=True),
                        _cell_text(r["idc"], percent=False, digits=3),
                    ]
                    for r in lone_clean_table(clean)
                ],
            )
        )
        lines.append("")
        lines.extend(f"- {line}" for line in lone_clean_findings(clean))
        lines.append("")
    lines.extend(f"- {line}" for line in bundle.get("findings", []))
    return "\n".join(lines) + "\n"


def write_lone_outputs(
    bundle: dict[str, Any],
    outdir: str | Path,
    clean: dict[str, Any] | None = None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "lone_bound.md"
    csv_path = outdir / "lone_bound.csv"
    md_path.write_text(render_lone_markdown(bundle, clean), encoding="utf-8")
    rows = csv_rows(bundle)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"lone_md": md_path, "lone_csv": csv_path}


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _pct(value: Any) -> str:
    v = _f(value)
    return "n/a" if not math.isfinite(v) else f"{100.0 * v:.1f}%"


def _num(value: Any, digits: int = 2) -> str:
    v = _f(value)
    return "n/a" if not math.isfinite(v) else f"{v:.{digits}f}"


def _pp(a: Any, b: Any) -> str:
    """Point delta of two rates, in percentage points."""
    va, vb = _f(a), _f(b)
    if not (math.isfinite(va) and math.isfinite(vb)):
        return "n/a"
    return f"{100.0 * (va - vb):+.1f}pp"


def _pick(bundle: dict[str, Any], label: str, family: str | None = None) -> dict[str, Any]:
    table = bundle["leaderboard"] if family is None else bundle["by_family"].get(family, {})
    return table.get(label, {})


def _rank(table: dict[str, Any], metric: str = "target_success_rate") -> list[tuple[str, float]]:
    ranked = [(_f(m.get(metric)), label) for label, m in table.items()]
    ranked = [(v, label) for v, label in ranked if math.isfinite(v)]
    return sorted(((label, v) for v, label in ranked), key=lambda kv: (-kv[1], kv[0]))


def findings(bundle: dict[str, Any]) -> list[str]:
    """Human-readable, number-bearing conclusions for the report."""
    out: list[str] = []
    board = bundle["leaderboard"]
    if not board:
        return ["no results"]

    # 1. overall leaderboard
    overall = _rank(board)
    if overall:
        best, best_v = overall[0]
        worst, worst_v = overall[-1]
        out.append(
            f"OVERALL (pooled over {bundle['meta']['n_scenarios']} scenarios): best "
            f"target_success_rate {_pct(best_v)} ({best}), worst {_pct(worst_v)} ({worst}) "
            f"-- spread {100.0 * (best_v - worst_v):.1f}pp."
        )

    # 2. fast linear: where does speed break the lock?
    lin = bundle["by_family"].get("fast_linear", {})
    if lin:
        ranked = _rank(lin)
        label = ranked[0][0] if ranked else next(iter(lin))
        rows = [r for r in bundle["results"] if r["family"] == "fast_linear" and r["config"] == label]
        per_step = sorted((_f(r["params"].get("step")), r) for r in rows)
        parts = [f"{step:g}px:{_pct(r['metrics']['target_success_rate'])}" for step, r in per_step]
        broken = [step for step, r in per_step if _f(r["metrics"]["target_success_rate"]) < 0.9]
        out.append(
            f"FAST LINEAR (best config {label}): success by speed step = {', '.join(parts)}; "
            f"first step below 90% success: {f'{broken[0]:g} px/frame' if broken else 'none up to 80 px/frame'}."
        )
        lat = [(step, _f(r["metrics"]["reacquire_latency"])) for step, r in per_step]
        lat = [(s, v) for s, v in lat if math.isfinite(v)]
        if lat:
            out.append(
                "FAST LINEAR re-acquisition after the detector drop: latency "
                + ", ".join(f"{s:g}px:{v:.1f}f" for s, v in lat)
                + f" (config {label})."
            )

    # 3. nonlinear: does IMM pay off on the whole system?
    nl = bundle["by_family"].get("nonlinear", {})
    if nl and "imm" in nl and "kf" in nl:
        mi, mk = nl["imm"], nl["kf"]
        out.append(
            f"NONLINEAR (circles + figure-eight): imm success {_pct(mi['target_success_rate'])} vs "
            f"kf {_pct(mk['target_success_rate'])} ({_pp(mi['target_success_rate'], mk['target_success_rate'])}); "
            f"mean IoU imm {_num(mi.get('target_precision'), 3)} vs kf {_num(mk.get('target_precision'), 3)}."
        )
        if "nsa_kf" in nl:
            mn = nl["nsa_kf"]
            out.append(
                f"NONLINEAR nsa_kf {_pct(mn['target_success_rate'])}; "
                f"imm - nsa_kf = {_pp(mi['target_success_rate'], mn['target_success_rate'])}."
            )

    # 3b. THE DECISIVE EXPERIMENT: pure prediction while the detector is blind.
    nb = bundle["by_family"].get("nonlinear_blackout", {})
    lb = bundle["by_family"].get("linear_blackout", {})
    if nb:
        parts = []
        for label in ("kf", "nsa_kf", "imm"):
            if label not in nb:
                continue
            m = nb[label]
            ctl = lb.get(label)
            piece = f"{label} success {_pct(m['target_success_rate'])} / IoU {_num(m.get('target_precision'), 3)}"
            if ctl:
                piece += f" (linear control {_pct(ctl['target_success_rate'])} / {_num(ctl.get('target_precision'), 3)})"
            parts.append(piece)
        out.append(
            "BLACKOUT (pure motion prediction while the detector is blind, target still in GT): "
            + "; ".join(parts)
            + "."
        )
        if "imm" in nb and "kf" in nb:
            out.append(
                f"BLACKOUT nonlinear: imm - kf success "
                f"{_pp(nb['imm']['target_success_rate'], nb['kf']['target_success_rate'])}, IoU "
                f"{_f(nb['imm'].get('target_precision')) - _f(nb['kf'].get('target_precision')):+.3f}; "
                f"loss rate kf {_pct(nb['kf']['loss_rate'])} vs imm {_pct(nb['imm']['loss_rate'])}."
            )
        agg: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for r in bundle["results"]:
            if r["family"] == "nonlinear_blackout":
                agg.setdefault((int(r["params"]["gap"]), r["config"]), []).append(r["metrics"])
        if agg:
            per_gap = []
            for gap in sorted({g for g, _ in agg}):
                cells = [
                    f"{label} {_pct(combine_target(agg[(gap, label)])['target_success_rate'])}"
                    for label in ("kf", "nsa_kf", "imm")
                    if (gap, label) in agg
                ]
                per_gap.append(f"{gap}f " + "/".join(cells))
            out.append("BLACKOUT success by blind-window length: " + ", ".join(per_gap) + ".")

    # 4. rollback: does depth matter when the target disappears?
    occ = bundle["by_family"].get("occlusion", {})
    if occ:
        for base, rb0, rb3 in (("kf", "kf-rb0", "kf-rb3"), ("imm", "imm-rb0", "imm-rb3")):
            if all(k in occ for k in (base, rb0, rb3)):
                out.append(
                    f"ROLLBACK on occlusion ({base}): success rb0 {_pct(occ[rb0]['target_success_rate'])} / "
                    f"rb3 {_pct(occ[rb3]['target_success_rate'])} / rb5 {_pct(occ[base]['target_success_rate'])}; "
                    f"re-acquire latency rb0 {_num(occ[rb0].get('reacquire_latency'), 1)}f / "
                    f"rb3 {_num(occ[rb3].get('reacquire_latency'), 1)}f / "
                    f"rb5 {_num(occ[base].get('reacquire_latency'), 1)}f; "
                    f"re-acquire rate rb0 {_pct(occ[rb0]['reacquire_rate'])} / "
                    f"rb5 {_pct(occ[base]['reacquire_rate'])}."
                )
        rows = [r for r in bundle["results"] if r["family"] == "occlusion"]
        for kind in ("inplace", "distant"):
            sub = [r for r in rows if bool(r["params"].get("distant")) == (kind == "distant")]
            if not sub:
                continue
            per_gap: dict[int, list[dict[str, Any]]] = {}
            for r in sub:
                per_gap.setdefault(int(r["params"]["gap"]), []).append(r["metrics"])
            out.append(
                f"OCCLUSION {kind}: re-acquire rate by gap = "
                + ", ".join(
                    f"{gap}f:{_pct(combine_target(v)['reacquire_rate'])}"
                    for gap, v in sorted(per_gap.items())
                )
                + "."
            )

    # 5. the distractor: refusal vs false acquisition
    dist = [r for r in bundle["results"] if r["family"] == "distractor"]
    for lookalike in (True, False):
        sub = [r for r in dist if bool(r["params"].get("lookalike")) == lookalike]
        if not sub:
            continue
        tag = "IDENTICAL LOOKALIKE" if lookalike else "DIFFERENT-APPEARANCE CONTROL"
        parts = []
        for label in ("kf", "nsa_kf", "imm"):
            if not any(r["config"] == label for r in sub):
                continue
            m = combine_target([r["metrics"] for r in sub if r["config"] == label])
            parts.append(
                f"{label}: success {_pct(m['target_success_rate'])}, FA {_pct(m['false_acquisition_rate'])}"
                f" ({int(m['false_acquisition_frames'])}f), reacq {_pct(m['reacquire_rate'])}"
            )
        out.append(f"DISTRACTOR {tag}: " + "; ".join(parts) + ".")
    if dist:
        like = [r["metrics"] for r in dist if bool(r["params"].get("lookalike"))]
        diff = [r["metrics"] for r in dist if not bool(r["params"].get("lookalike"))]
        if like and diff:
            ml, md = combine_target(like), combine_target(diff)
            delta = 100.0 * (_f(md["target_success_rate"]) - _f(ml["target_success_rate"]))
            out.append(
                f"DISTRACTOR sensitivity check: success {_pct(ml['target_success_rate'])} (identical embedding) vs "
                f"{_pct(md['target_success_rate'])} (orthogonal embedding), {delta:+.1f}pp; "
                "the appearance channel is the only difference between those two runs."
            )

    # 6. NSA adaptivity: does confidence-scaling change anything?
    if "nsa_kf" in board and "nsa_kf-static" in board:
        a, b = board["nsa_kf"], board["nsa_kf-static"]
        out.append(
            f"NSA ADAPTIVITY: nsa_kf adaptive success {_pct(a['target_success_rate'])} vs static "
            f"{_pct(b['target_success_rate'])} ({_pp(a['target_success_rate'], b['target_success_rate'])}); "
            f"precision {_num(a.get('target_precision'), 3)} vs {_num(b.get('target_precision'), 3)}."
        )

    # 7. The requirement-(4) family: does the track come back when the target is
    #    unidentifiable and a bystander sits on the stale prediction?
    unid_runs = [r for r in bundle["results"] if r.get("family") in ("unidentifiable_target", "unidentifiable_solo")]
    if unid_runs:
        for tag, family in (("with bystander", "unidentifiable_target"), ("no bystander", "unidentifiable_solo")):
            sel = [r for r in unid_runs if r.get("family") == family]
            if not sel:
                continue
            m = combine_target([r["metrics"] for r in sel])
            out.append(
                f"UNIDENTIFIABLE ({tag}): success {_pct(m['target_success_rate'])}, "
                f"false acquisition {_pct(m['false_acquisition_rate'])} over {len(sel)} run(s)."
            )
        tgt = [r for r in unid_runs if r.get("family") == "unidentifiable_target"]
        if tgt:
            worst = max(tgt, key=lambda r: _f(r["metrics"].get("false_acquisition_rate")))
            wfa = _f(worst["metrics"].get("false_acquisition_rate"))
            if wfa > 0:
                acc = worst.get("refusals_blind", {})
                out.append(
                    f"UNIDENTIFIABLE worst run {worst['scenario']} / {worst['config']}: "
                    f"false acquisition {_pct(wfa)} ({_num(worst['metrics'].get('false_acquisition_frames'), 0)} frames), "
                    f"success {_pct(worst['metrics'].get('target_success_rate'))}, "
                    f"median accepted margin in the blind window {_num(acc.get('accepted_margin_median'), 3)}."
                )
            else:
                out.append(
                    "UNIDENTIFIABLE: no run produced a false acquisition -- every configuration refused the "
                    "bystander and recovered the target (a NULL for the anti-latch rule on this geometry)."
                )

    if bundle.get("errors"):
        out.append(f"ERRORS: {len(bundle['errors'])} run(s) failed -- see raw.json.")
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Recursively make a result JSON-safe: NaN/inf -> null."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _csv_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The union of every row's keys, in first-seen order.

    Rows are heterogeneous on purpose (the handover and blind-window columns only
    exist for the sweeps that trace), so the header cannot come from row 0 alone:
    a later row's extra key would make ``csv.DictWriter`` raise
    ``ValueError: dict contains fields not in fieldnames``.
    """
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields or ["scenario", "config"]


def csv_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in bundle["results"]:
        row = {
            "scenario": r["scenario"],
            "family": r["family"],
            "params": json.dumps(r["params"], sort_keys=True),
            "config": r["config"],
            "model": r["model"],
            "nsa_adaptive": r["nsa_adaptive"],
            "rollback_depth": r["rollback_depth"],
            "imm_models": "+".join(r["imm_models"]),
            "distractor_margin": r.get("distractor_margin"),
            "n_frames": r["n_frames"],
            "n_visible_frames": r["metrics"].get("n_visible_frames"),
        }
        for key in REPORT_METRICS:
            row[key] = r["metrics"].get(key)
        ref = r.get("refusals_overlap")
        if isinstance(ref, dict):
            row["overlap_frames"] = ref.get("frames")
            row["refused_frames"] = ref.get("ambiguous")
            row["accepted_in_overlap"] = ref.get("accepted")
            row["margin_median"] = ref.get("margin_median")
        blind = r.get("refusals_blind")
        if isinstance(blind, dict):
            row["blind_frames"] = blind.get("frames")
            row["accepted_in_blind"] = blind.get("accepted")
            row["refused_in_blind"] = blind.get("all_gated")
            row["blind_accept_ratio"] = r.get("blind_ratio")
        over = r.get("cfg_overrides")
        if isinstance(over, dict) and "cost.missing_appearance_cost" in over:
            row["missing_appearance_cost"] = over["cost.missing_appearance_cost"]
        if isinstance(over, dict) and "cost.lone_candidate_accept_cost" in over:
            row["lone_candidate_accept_cost"] = over["cost.lone_candidate_accept_cost"]
        hand = r.get("handover")
        if isinstance(hand, dict):
            row["handover_frame"] = hand.get("frame")
            row["handover_n_dets"] = hand.get("n_dets")
            row["handover_phase"] = hand.get("phase")
        if isinstance(r.get("sortie"), str):
            row["sortie"] = r["sortie"]
        rows.append(row)
    return rows


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return lines


LEADERBOARD_HEADER = (
    "config", "model", "nsa", "rb", "imm_models", "success", "IoU", "loss", "reacq rate", "reacq lat", "FA rate", "id const",
)


def _leaderboard_row(label: str, m: dict[str, Any]) -> list[str]:
    return [
        label,
        str(m.get("model", "")),
        "on" if m.get("nsa_adaptive") else "off",
        str(m.get("rollback_depth", "")),
        "+".join(m.get("imm_models") or []),
        _pct(m.get("target_success_rate")),
        _num(m.get("target_precision"), 3),
        _pct(m.get("loss_rate")),
        _pct(m.get("reacquire_rate")),
        _num(m.get("reacquire_latency"), 1),
        _pct(m.get("false_acquisition_rate")),
        _pct(m.get("id_consistency")),
    ]


def _spec_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["config"]: s for s in bundle["specs"]}


def render_markdown(bundle: dict[str, Any]) -> str:
    specs = _spec_lookup(bundle)
    meta = bundle["meta"]
    lines: list[str] = [
        "# Motion-strategy ablation: KF vs NSA-KF vs IMM (clicked-target tracker)",
        "",
        "Generated by `scripts/ablate_motion.py`.  Every number below comes from "
        "`tgtrack.evaluate.evaluate_target` on a full run of the real "
        "`TargetTracker`; appearance is a deterministic stub, so motion is the only "
        "variable.",
        "",
        f"- scenarios: **{meta['n_scenarios']}**, configs: **{meta['n_configs']}**, runs: **{meta['n_runs']}**"
        f" (frames simulated: {meta['frames_simulated']:,})",
        f"- errors: {meta['n_errors']}, deterministic: {meta['deterministic']}",
        "",
        "## Leaderboard (pooled over all scenarios)",
        "",
    ]
    def merged(label: str) -> dict[str, Any]:
        m = dict(bundle["leaderboard"].get(label, {}))
        m.update(specs.get(label, {}))
        return m

    ranked = _rank(bundle["leaderboard"])
    lines += _md_table(LEADERBOARD_HEADER, [_leaderboard_row(label, merged(label)) for label, _ in ranked])

    for family in sorted(bundle["by_family"]):
        table = bundle["by_family"][family]
        lines += ["", f"## Family: `{family}`", "", FAMILY_NOTES.get(family, ""), ""]
        sub = _rank(table)
        lines += _md_table(LEADERBOARD_HEADER, [_leaderboard_row(label, merged(label)) for label, _ in sub])

    lines += ["", "## Scenarios", ""]
    lines += _md_table(
        ("scenario", "family", "frames", "note"),
        [(s["name"], s["family"], s["n_frames"], s["note"]) for s in bundle["scenarios"]],
    )

    lines += ["", "## Findings", ""]
    lines += [f"- {f}" for f in bundle.get("findings", [])]
    if bundle.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"- `{e['scenario']}` / `{e['config']}`: {e['error']}" for e in bundle["errors"]]
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    bundle: dict[str, Any],
    outdir: str | Path,
    *,
    distractor: dict[str, Any] | None = None,
    unid: dict[str, Any] | None = None,
    unid_cost: dict[str, Any] | None = None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw.json"
    md_path = outdir / "table.md"
    csv_path = outdir / "table.csv"
    readme_path = outdir / "readme.md"

    raw_path.write_text(json.dumps(_jsonable(bundle), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(bundle), encoding="utf-8")

    rows = csv_rows(bundle)
    fields = _csv_fields(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paths = {"raw": raw_path, "md": md_path, "csv": csv_path}
    if distractor is not None:
        paths.update(write_distractor_outputs(distractor, outdir))
    if unid is not None:
        paths.update(write_unid_outputs(unid, outdir))
    if unid_cost is not None:
        paths.update(write_unid_cost_outputs(unid_cost, outdir))
    readme_path.write_text(render_readme(bundle, distractor, unid), encoding="utf-8")
    paths["readme"] = readme_path
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _trace_main(args: argparse.Namespace) -> int:
    """``--trace``: per-frame diagnostics for named scenarios instead of a sweep."""
    by_name = {s.name: s for s in default_scenarios()}
    for cos in DISTRACTOR_COSINES:
        for scenario in (overlap_distractor(cos), overlap_distractor(cos, peel=(14.0, 9.0))):
            by_name.setdefault(scenario.name, scenario)
    for mode in UNID_MODES:
        by_name.setdefault(unidentifiable_solo(mode=mode).name, unidentifiable_solo(mode=mode))
        for cos in UNID_SIMILARITIES:
            scenario = unidentifiable_target(cos, mode=mode)
            by_name.setdefault(scenario.name, scenario)
    specs = {s.label: s for s in DEFAULT_SPECS}
    for spec in distractor_specs():
        specs.setdefault(spec.label, spec)
    for spec in unid_specs():
        specs.setdefault(spec.label, spec)
    for spec in lone_specs():
        specs.setdefault(spec.label, spec)
    if args.trace_spec not in specs:
        print(f"unknown --trace-spec {args.trace_spec!r}; known: {', '.join(sorted(specs))}")
        return 2
    names = [n.strip() for n in str(args.trace).split(",") if n.strip()]
    written: list[Path] = []
    rc = 0
    for name in names:
        scenario = by_name.get(name)
        if scenario is None:
            print(f"unknown scenario {name!r}; known: {', '.join(sorted(by_name))}")
            rc = 2
            continue
        result = run_scenario(scenario, specs[args.trace_spec], trace=True)
        rows = result["trace"]
        print()
        print(render_trace(rows, scenario=scenario.name, spec=args.trace_spec))
        print(f"  note: {scenario.note}")
        print(
            "  final metrics: "
            f"success={result['metrics'].get('target_success_rate')} "
            f"reacq_rate={result['metrics'].get('reacquire_rate')} "
            f"reacq_latency={result['metrics'].get('reacquire_latency')}"
        )
        if args.trace_csv:
            base = Path(args.trace_csv)
            path = base if len(names) == 1 else base.with_name(f"{base.stem}_{name}{base.suffix}")
            written.append(write_trace_csv(path, rows))
    for path in written:
        print(f"\nwrote trace csv: {path}")
    return rc


def quick_distractor_scenarios(cosines: Sequence[float] = (0.85, 0.99)) -> list[Scenario]:
    """Short overlap scenarios for smoke runs."""
    return [overlap_distractor(c, n_pre=8, n_overlap=12, n_post=10) for c in cosines]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Motion-strategy ablation on the clicked-target tracker.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="where to write raw.json/table.md/table.csv")
    parser.add_argument("--quick", action="store_true", help="reduced scenario set (smoke run)")
    parser.add_argument("--no-write", action="store_true", help="print findings only")
    parser.add_argument(
        "--trace",
        default="",
        help="comma-separated scenario names: print per-frame diagnostics instead of sweeping",
    )
    parser.add_argument("--trace-spec", default="kf", help="config label used by --trace (default: kf)")
    parser.add_argument("--trace-csv", default="", help="also write the trace rows to this CSV (one file per scenario)")
    parser.add_argument(
        "--no-distractor",
        action="store_true",
        help="skip the hard-overlap similarity x distractor_margin sweep",
    )
    parser.add_argument(
        "--distractor",
        action="store_true",
        help="run only the hard-overlap distractor sweep (skip the scenario family sweep)",
    )
    parser.add_argument(
        "--no-unid",
        action="store_true",
        help="skip the unidentifiable-target x arming-rule sweep",
    )
    parser.add_argument(
        "--unid",
        action="store_true",
        help="run only the unidentifiable-target sweeps (arming rule + missing_appearance_cost)",
    )
    parser.add_argument(
        "--unid-cost",
        action="store_true",
        help="run only the missing_appearance_cost x similarity sweep",
    )
    parser.add_argument(
        "--lone",
        action="store_true",
        help="run only the lone_candidate_accept_cost sweeps (unid family + clean families)",
    )
    parser.add_argument(
        "--no-lone",
        action="store_true",
        help="skip the lone_candidate_accept_cost sweeps",
    )
    args = parser.parse_args(argv)

    if args.trace:
        return _trace_main(args)

    bundle: dict[str, Any] | None = None
    if not (args.distractor or args.unid or args.unid_cost or args.lone):
        scenarios = quick_scenarios() if args.quick else default_scenarios()
        print(
            f"motion ablation: {len(scenarios)} scenarios x {len(DEFAULT_SPECS)} configs "
            f"= {len(scenarios) * len(DEFAULT_SPECS)} tracker runs",
            flush=True,
        )
        bundle = run_sweep(scenarios, DEFAULT_SPECS, progress=True)

        print("\n=== LEADERBOARD (pooled, sorted by target_success_rate) ===")
        specs = _spec_lookup(bundle)
        for label, value in _rank(bundle["leaderboard"]):
            m = dict(bundle["leaderboard"][label])
            m.update(specs.get(label, {}))
            print("  " + " | ".join(_leaderboard_row(label, m)))

        print("\n=== FINDINGS ===")
        for line in bundle["findings"]:
            print("  - " + line)

    distractor: dict[str, Any] | None = None
    if args.distractor or not (args.no_distractor or args.unid or args.unid_cost or args.lone):
        if args.quick:
            d_scenarios = quick_distractor_scenarios()
            d_cosines: Sequence[float] = (0.85, 1.00)
            d_margins: Sequence[float] = (0.0, 0.12)
            d_models: Sequence[str] = ("imm",)
        else:
            d_scenarios = None
            d_cosines, d_margins, d_models = DISTRACTOR_COSINES, DISTRACTOR_MARGINS, DISTRACTOR_MODELS
        print(
            f"\n=== HARD-OVERLAP DISTRACTOR SWEEP "
            f"({len(d_cosines)} similarities x {len(d_margins)} margins x {len(d_models)} models) ===",
            flush=True,
        )
        distractor = run_distractor_sweep(
            d_scenarios,
            cosines=d_cosines,
            margins=d_margins,
            models=d_models,
            progress=True,
        )
        print()
        print(render_distractor_markdown(distractor))

    unid: dict[str, Any] | None = None
    if args.unid or not (args.no_unid or args.distractor or args.unid_cost or args.lone):
        if args.quick:
            u_scenarios = [
                unidentifiable_target(0.60, n_pre=8, n_gap=8, n_blind=10, n_post=8),
                unidentifiable_solo(n_pre=8, n_gap=8, n_blind=10, n_post=8),
            ]
            u_modes: Sequence[str] = ("missing",)
            u_models: Sequence[str] = ("kf",)
        else:
            u_scenarios = None
            u_modes, u_models = ("missing", "orthogonal"), UNID_MODELS
        print(
            f"\n=== UNIDENTIFIABLE-TARGET SWEEP "
            f"(modes {list(u_modes)} x {len(UNID_SIMILARITIES)} similarities x {len(u_models)} models "
            f"x {len(UNID_SWEEPS)} arming rules) ===",
            flush=True,
        )
        unid = run_unid_sweep(u_scenarios, modes=u_modes, models=u_models, progress=True)
        print()
        print(render_unid_markdown(unid))

    unid_cost: dict[str, Any] | None = None
    if (args.unid or args.unid_cost) or not (args.no_unid or args.distractor or args.lone):
        if args.quick:
            c_scenarios = [
                unidentifiable_target(0.60, n_pre=8, n_gap=8, n_blind=10, n_post=8),
                unidentifiable_solo(n_pre=8, n_gap=8, n_blind=10, n_post=8),
            ]
            c_modes: Sequence[str] = ("missing",)
            c_models: Sequence[str] = ("kf",)
            c_costs: Sequence[float] = (0.5, 1.2)
        else:
            c_scenarios = None
            c_modes, c_models, c_costs = ("missing", "orthogonal"), UNID_MODELS, MISSING_COSTS
        print(
            f"\n=== UNIDENTIFIABLE-TARGET PRICING SWEEP "
            f"(modes {list(c_modes)} x {len(UNID_SIMILARITIES)} similarities x {len(c_models)} models "
            f"x {len(c_costs)} missing_appearance_cost values) ===",
            flush=True,
        )
        unid_cost = run_unid_cost_sweep(
            c_scenarios, modes=c_modes, models=c_models, costs=c_costs, progress=True
        )
        print()
        print(render_unid_cost_markdown(unid_cost))

    lone: dict[str, Any] | None = None
    lone_clean: dict[str, Any] | None = None
    if args.lone or not (
        args.no_lone or args.distractor or args.unid or args.unid_cost
    ):
        if args.quick:
            l_scenarios = [
                unidentifiable_target(0.60, n_pre=8, n_gap=8, n_blind=10, n_post=8),
                unidentifiable_target(0.00, n_pre=8, n_gap=8, n_blind=10, n_post=8),
                unidentifiable_solo(n_pre=8, n_gap=8, n_blind=10, n_post=8),
            ]
            l_modes: Sequence[str] = ("missing",)
            l_models: Sequence[str] = ("kf",)
            l_bounds: Sequence[float | None] = (0.4, None)
            l_clean = clean_scenarios()[:6]
        else:
            l_scenarios = None
            l_modes, l_models, l_bounds = ("missing", "orthogonal"), ("kf", "imm"), LONE_BOUNDS
            l_clean = None
        print(
            f"\n=== LONE-CANDIDATE BOUND SWEEP "
            f"(modes {list(l_modes)} x {len(UNID_SIMILARITIES)} similarities x {len(l_models)} models "
            f"x {len(l_bounds)} bounds) ===",
            flush=True,
        )
        lone = run_lone_sweep(
            l_scenarios, modes=l_modes, models=l_models, bounds=l_bounds, progress=True
        )
        print()
        print(render_lone_markdown(lone))
        print(
            f"\n=== LONE-CANDIDATE BOUND SWEEP on the clean families "
            f"({len(l_bounds)} bounds x {len(clean_scenarios() if l_clean is None else l_clean)} scenarios) ===",
            flush=True,
        )
        lone_clean = run_lone_clean_sweep(l_clean, bounds=l_bounds, progress=True)
        print()
        for line in lone_clean["findings"]:
            print("  - " + line)

    if not args.no_write:
        paths: dict[str, Path] = {}
        if bundle is not None:
            paths.update(
                write_outputs(
                    bundle,
                    args.outdir,
                    distractor=distractor,
                    unid=unid,
                    unid_cost=unid_cost,
                )
            )
        else:
            if distractor is not None:
                paths.update(write_distractor_outputs(distractor, args.outdir))
            if unid is not None:
                paths.update(write_unid_outputs(unid, args.outdir))
            if unid_cost is not None:
                paths.update(write_unid_cost_outputs(unid_cost, args.outdir))
        if lone is not None:
            paths.update(write_lone_outputs(lone, args.outdir, lone_clean))
        print("\nwrote:")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
