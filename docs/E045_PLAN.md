# E045 — a second identity signal (the only remaining route to criterion ②)

Written at the end of round 21, after E039–E044 established that **every knob on every axis of the
single-target layer is at a measured local optimum**. This document exists so the next round can
execute without re-deriving the reasoning.

## Why no further tuning can work

Criterion ② needs TSR 0.512 → ≥0.55 and FAR 0.280 → <0.20. Sixteen consecutive experiments on the
same family — *a distance between a candidate crop and the target's own past* — produced:

| family | experiments | outcome |
|---|---|---|
| appearance distance (STM, LTM, anchors, band, adaptive forgetting) | E017–E018, E021, E023–E030 | shipped E028–E030 (+~0.06) then exhausted; E032b showed distance and margin are not independent |
| motion / geometry gates | E032, E016 | motion band rejected, velocity clamp shipped |
| negative memory | E036–E042 | real mechanism, shipped E038/E041, then saturated |
| fusion weights, margins, memory mix | E031, E043, E044 | all at measured local optima |

E032b is the load-bearing result: adding the motion axis to the appearance axis moved the
DanceTrack joint operating point by **0.000**. Both axes measure "how far is this crop from the
target's history", so they cannot separate *"the look changed"* from *"this is someone else"* —
the exact distinction criterion ② requires.

## The hypothesis

> A **structural** descriptor computed from the candidate itself, compared against a structural
> descriptor of the click, is independent of the appearance-distance family and can adjudicate the
> cases where appearance is ambiguous (crowded dancers, partial views, look changes).

If this fails, the criterion needs a different *sensor*, not a different feature on the same crop.

## Ranked candidates (cost rises fast; take them in order, stop at the first that pays)

1. **Pose skeleton geometry — cheapest real structure.** `yolo26n-pose.pt` is already downloaded
   (`data/weights/yolo26n-pose.pt`, 7,878,574 B, obtained in the E001–E003 round). Per candidate,
   take the 17 keypoints confined to the *visible* box region, normalise by the box, and compare:
   - limb-length ratios (scale-invariant, survives partial views),
   - torso/hip/shoulder orientation angles (rotation-invariant),
   - head-shoulder-shape ratio.
   Cost: one extra forward pass per candidate (~3 ms CPU at 1280). Integration point:
   `TargetTracker._associate` stage 2, as a third cost term with its own gate — not a weight on
   the existing appearance axis.
   *Expected ceiling:* DanceTrack is a dance benchmark where bodies are clearly posed, so pose
   should be more distinctive than a 512-d global embedding. This is also the signal that survives
   partial views structurally rather than statistically.

2. **Local patch geometry (LightGlue-style sparse matching).** Extract top-k keypoints in the
   visible box, match to the click crop, score by inlier ratio after a similarity transform.
   Cross-domain import (3D vision / image matching). Cost: high on CPU; only viable if candidate
   1 is ambiguous-but-promising.

3. **Inter-person structure (the cross-domain idea worth one shot).** Everyone's tracklet identity
   is already solved by the MOT layer, so the *neighbourhood* of a candidate is known: who is 1 m
   left, who is behind. A target's neighbour set is strongly identity-bearing in a formation/dance
   context and is **orthogonal to both appearance and motion**. Cheap: reuse the MOT layer's boxes
   and IDs, one extra cost term comparing the candidate's neighbour set to the click frame's.

## Protocol (non-negotiable, learned from E016/E037)

- Fresh runs only; `scripts/check_baseline_freshness.py` must report OK before and after.
- Same epoch for both arms of every comparison; no default changed mid-grid.
- Three sequences, `mask=none`, 200 frames, subsample 2, `--target-id 3/2/1` for 0004/0005/0007.
- Acceptance bar to ship: pooled TSR ≥ 0.55 **and** pooled FAR < 0.20, with no sequence losing
  more than 3 pp TSR versus the shipped baseline (0.512 / 0.280).
- Record the negative result with the same care as a win — E039's lesson was that the incomplete
  version of a mechanism taught the opposite of the truth.

## Falsification

If pose structure moves the DanceTrack joint point by < 1 pp, then the failure is not "wrong
descriptor" but "not enough information in one person's pixels", and criterion ② should be
re-scoped in the report to what is achievable at that resolution — with the measurements
(E032b's 0.000, E044's bimodality, E020's 74/111-rank trace) as the evidence.
