# Research plan — from "works" to "publishable"

The first phase built and honestly measured a working system. This document is the
*second* phase: a standing loop of **find the binding constraint → survey → hypothesise →
measure → keep or kill**, with the result ledger in [`EXPERIMENTS.md`](EXPERIMENTS.md).

North-star targets (from the goal, not from ambition):

| metric | now | target |
|---|---|---|
| MOT17 all-person HOTA (`tracktrack`, `yolo26m@1280`) | **0.4827** | **≥ 0.55** |
| DanceTrack single-target FAR (`mask=none`) | **0.493** | **< 0.20** (TSR ≥ 0.55) |
| MOT17 single-target TSR, upper-body-only mask | **0.7527** | **≥ 0.85** |

---

## 1. Where the loss actually is (measured, not assumed)

Every number below is reproducible with the script named next to it.

### The all-person layer is detection-limited, not association-limited

`results/mot_eval_m1280/` — six backends on real MOT17, 42 pairs, `yolo26m@1280`:

| backend | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| tracktrack | **0.4827** | 0.4415 | 0.5322 | 0.4329 | 0.5860 | 346 |
| botsort | 0.4728 | 0.4402 | 0.5129 | 0.4334 | 0.5649 | 431 |
| deepocsort | 0.4706 | 0.4391 | 0.5093 | 0.4304 | 0.5613 | 500 |
| ocsort | 0.4425 | 0.4218 | 0.4697 | 0.3949 | 0.5129 | 1148 |
| bytetrack | 0.4421 | 0.4201 | 0.4706 | 0.3965 | 0.5092 | 1114 |
| fasttrack | 0.4325 | 0.4206 | 0.4499 | 0.3938 | 0.4953 | 1210 |

`DetA` is flat at **0.42–0.44** while `AssA` spans **0.45–0.53**. Since
`HOTA ≈ sqrt(DetA·AssA)`, the ceiling on HOTA is set by detection: with `AssA` pinned,
going from `DetA 0.4415` to `0.60` would give `HOTA ≈ 0.565` — i.e. **the target is
reachable on detection alone**, before any association work.

### …but "detection-limited" was the wrong diagnosis of *why*

`scripts/probe_recall.py` measured recall 0.721 at the shipped `conf=0.25` on MOT17-02.
Two follow-ups killed the two obvious fixes:

- **More resolution does not help — it hurts.** `scripts/probe_tiling.py`: slicing the
  frame SAHI-style leaves recall unchanged (2×2: 0.721 → 0.721) and then degrades it
  (3×2: 0.680, 4×3: 0.521), at 3–11× the cost. Native-resolution `imgsz=1920` is also
  worse than 1280 (0.687 vs 0.721). **Missed people are not small**: at `conf=0.25` the
  missed boxes have median height **84 px** and **0 %** are under 40 px
  (`scripts/probe_miss.py`). The object scale is already fine; the *score* is low.
- **It is the confidence threshold, and the ceiling is far away.** Same frames, same
  weights, only `conf` changes:

| conf | recall | det/frame |
|---|---|---|
| 0.25 (shipped) | 0.721 | 19.1 |
| 0.15 | 0.775 | 25.3 |
| 0.10 | 0.797 | 30.2 |
| 0.05 | **0.840** | 43.4 |
| 0.02 | **0.883** | 69.3 |
| 0.01 | **0.911** | 95.8 |

So **19 pp of recall is sitting below the shipped threshold**, and the missed boxes are
*not* invisible: at `conf=0.25`, 42 % of the misses have visibility < 0.5 but their median
visibility is 0.59, and 52 % are never touched even at IoU 0.1 — they are in the low-score
band, not absent.

### The structural consequence nobody sets correctly

The ultralytics tracker configs split detections into a high band and a low band
(`track_high_thresh 0.25 / track_low_thresh 0.1` for bytetrack/botsort) precisely so that
a second association pass can rescue occluded people. **If the detector's `conf` is 0.25,
the low band is empty by construction** and the second pass has nothing to do — the
tracker silently degrades to single-stage association. That is the state of the shipped
pipeline, and it is the cheapest possible experiment (E002).

### The single-target layer loses on identity, not on position

- DanceTrack `mask=none`: FAR **0.493** — half the time the clicked identity is somebody
  else, on a dataset designed to make appearance useless. `DetA` there is 0.70 while
  `AssA` is only 0.29, the mirror image of MOT17.
- MOT17 upper-body mask: TSR 0.7527 while part-aware *retrieval* reaches Rank-1 0.855
  (`results/partial_view_real/`). **A 10 pp gap between what the descriptor can do offline
  and what the tracker converts into tracking.** The mechanism exists but is not being
  turned into decisions.
- The handover hole: with a similar bystander and an unreadable target, the swap happens
  in a single frame and no margin threshold can refuse it (`results/motion_ablation/`).

---

## 2. Hypothesis board (ranked by expected gain ÷ cost)

Each entry states the claim, the experiment that would **refute** it, and what it changes
in the north-star numbers.

### H1 · The low-score branch is dead at `conf=0.25` — *testing now (E002)*
**Claim.** Setting detector `conf` below the tracker's `track_low_thresh` re-enables the
second association stage and recovers a large part of the 19 pp of missing recall, so
`DetA` rises without any new model.
**Refutes it.** `HOTA` flat across `conf ∈ {0.25, 0.10, 0.05}` on matched pairs.
**Payoff.** DetA +5–10 pp ⇒ HOTA +2–5 pp. Cost: one CLI flag.

### H2 · The target layer should see the low-score band too
**Claim.** For a *single* person, low-confidence boxes near the motion prediction are
useful evidence — but only once appearance confirms them, because our unit of failure is a
wrong identity, not a missed frame. A two-stage target association (high-score first, then
low-score against the *frozen* gallery) raises re-acquisition and lowers the handover rate.
**Refutes it.** TSR and FAR both unchanged on `run_target.py --ablate` with a new
`low_score` variant; or FAR rises (which would be a *finding*, not a failure).
**Payoff.** TSR on the upper-body mask and after occlusion; FAR on DanceTrack.

### H3 · Pose is the orthogonal identity cue we are missing — *highest novelty*
**Claim.** Everything in the system keys on *appearance*, which DanceTrack deliberately
makes useless (people in identical clothes). Pretrained pose gives two things appearance
cannot: (a) **keypoint-anchored part boxes** instead of fixed-Y bands — so "upper body"
means shoulders-to-hips, not 0–60 % of a bounding box that also contains the background;
and (b) **skeleton/anthropometry as a second modality** — limb proportions and gait phase
are clothing-invariant. Zero training: `yolo26-pose` ships in the same ultralytics wheel.
**Refutes it.** Skeleton-shape distances do not separate identities better than chance on
DanceTrack crops (measure the same- vs different-identity gap, exactly as
`exp_partial_view.py::part_discriminability` does for appearance).
**Payoff.** FAR 0.49 → < 0.35 on DanceTrack; upper-body TSR +5–10 pp via better parts.
**Why it is underexplored.** Occluded-ReID papers use pose *to align crops*; almost nobody
uses the skeleton *as a descriptor* in a tracking cost, because benchmarks are
appearance-dominated. DanceTrack + VisDrone make it testable.

### H4 · Track-before-detect tubelet completion
**Claim.** When a track with strong history predicts a person who is not detected, the
track itself is evidence of presence (radar TBD). Used *only* to extend a tubelet when
appearance also confirms, it buys recall that no threshold can.
**Refutes it.** Hallucination rate rises faster than TSR — a real risk, so the ablation
must report `wrong_box_rate` / `hallucination_rate` alongside TSR.
**Payoff.** Recall in crowd/occlusion; the hardest VisDrone sequences.

### H5 · Calibrated refusal instead of `distractor_margin = 0.06`
**Claim.** The margin is a hand-tuned scalar with no guarantee. Conformal prediction gives
a **distribution-free bound on false acquisition** at a chosen risk level, learned from a
calibration split, and adapts per-scene instead of globally.
**Refutes it.** The conformal threshold is not better calibrated than 0.06 on held-out
sequences (compare empirical FAR vs nominal risk).
**Payoff.** The product promise ("never silently follows a stranger") becomes a *guarantee*
with a number, which is the paper-grade claim this repository is actually built around.

### H6 · Crowd-flow reappearance prior (borrowed from pedestrian dynamics)
**Claim.** Where an occluded target reappears is not predicted well by its own velocity
alone; the *flow of the people around it* carries the information. Optical flow of
neighbouring tracks gives a reappearance prior that beats constant-velocity coasting.
**Refutes it.** Reappearance error of flow prior ≥ coasting baseline on MOTE/occlusion
windows.

### H7 · Spatially-indexed long-term gallery (borrowed from visual place recognition)
**Claim.** Long-term re-acquisition after the target leaves and the camera moves is a
*retrieval over a map*, not over a bag of features. NetVLAD-style indexing on
`(position, scale, appearance)` turns a linear 96-entry gallery into a scene-aware index.

### H8 · Active perception on ambiguity
**Claim.** A following drone can *choose* what to look at. When the margin is tight, zoom
or re-tile on the ambiguous pair rather than guessing — a perception action, not a tracker
tweak. Underexplored because benchmarks give you a fixed video, but our product does not.
**Payoff.** Directly attacks the handover hole with a new degree of freedom.

---

## 3. Protocol (how a hypothesis becomes a result)

1. **State the refutation before running.** If the experiment cannot fail, it is not an
   experiment — the earlier `run_target` grid that returned TSR 1.0 for all 21 variants
   cost real time and taught nothing.
2. **Matched A/B, same frames, same seeds, same GPU state.** Differences must come from
   the one knob under test. Compare against the *shipped* default, not against a number
   remembered from another run.
3. **Report cost with the gain** (ms/frame or params), because a gain that doubles latency
   is not a gain here — this is an 8 GB laptop GPU.
4. **Report the metrics that can go down** (`wrong_box_rate`, `hallucination_rate`,
   `false_acquisition_rate`), not only the one that should go up.
5. **Write the row in `EXPERIMENTS.md` even when it fails** — especially when it fails.
   Three of the first phase's four claims died under their own controls; the ledger is what
   stopped us re-running them.
6. **A knob ships enabled only if it wins on real data.** Synthetic-only wins have already
   been rejected once (the lone-candidate guard).
