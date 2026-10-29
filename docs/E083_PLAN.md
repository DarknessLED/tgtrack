# E083 — Multi-hypothesis tracking with calibrated commitment (pre-registration)

Status: **PRE-REGISTERED, not implemented.** Written 2026-10-28 while E082's sweep ran.
Nothing in this document may be edited after the first measurement starts; deviations go in the
ledger entry, not here.

## Why this, now

Every single-frame identity mechanism this campaign tried has failed (12 of them: E021, E023,
E024, E025, E026, E027-as-shipped-only-with-E029, E032, E038-partial, E059, E061, E063, E064).
The ones that worked all act on **how the tracker decides**, not on **what it measures**:

| mechanism | what it changed | result |
|---|---|---|
| OC-SORT-style observation-centric motion | the prediction it trusts | shipped, +2.91 pp partial-view |
| the refusal gate / distractor margin | when it refuses to accept | shipped, sharply optimal |
| self-calibrating band (E029) | how the ceiling is set | shipped |
| frozen-crop vote (E079) | whether a veto can be overruled | best measured signal, killed by its CI |

E076's loss decomposition says **24 % of the loss is "wrong while tracking"** — the tracker is
holding a confident box on the wrong person, and *no downstream test can see it*, because the box
is stable and the appearance distance is small (E075: wrong detections cost 0.019 vs 0.006 for
correct ones).

The root cause is architectural, not statistical: **the tracker commits**. Once a candidate is
accepted it becomes the motion filter's only state, so the identity that was correct one frame
earlier is *discarded* rather than kept alive as a competitor. Every later frame is then
conditioned on that one decision. In a crowded, occluding scene the correct explanation is often
a set of two or three possibilities for a few frames, and a point estimate throws that away.

This is precisely the multi-hypothesis data-association (MHT/JPDA) idea from radar tracking and
SLAM, which the MOT literature largely abandoned for cost reasons and which nobody in the
2021-26 DanceTrack top-venue work uses (TrackTrack, Hybrid-SORT, Deep HM-SORT are all single
hypothesis). It is the one item on the campaign's own "unexplored" list that directly attacks the
largest *recoverable* error class.

## Hypothesis

Keeping K = 3 hypotheses alive for a bounded window (each with its own motion state, sharing the
appearance memory) and **committing only when the posterior separates** will:

1. reduce the "wrong while tracking" share of the loss, and
2. raise TSR without raising FAR,

because the wrong latch is repaired *inside* the tracker instead of being detected (and mostly
missed) by downstream refusal logic.

## Mechanism

`src/tgtrack/hypotheses.py` — new module, nothing else changes.

```
class Hypothesis:            # one candidate identity
    tracker: TargetTracker   # its own motion filter + association, sharing the gallery
    logw: float              # log posterior weight
    age: int                 # frames since spawned
    born_frame: int
    last_accepted: int

class MultiHypothesisTracker:
    def __init__(cfg, k=1, embedder=None, ...)
    def initialize(frame_bgr, box_xyxy, frame_idx=0) -> TargetStatus
    def step(frame_bgr, dets, frame_idx=None) -> TargetStatus
    .reported()  -> Hypothesis        # the one whose box is emitted
    .weights()   -> np.ndarray
    .stats()     -> dict              # engagement counters, see below
```

Per frame:

1. Every live hypothesis steps on the *same* detection list, with the boxes currently claimed by
   the other hypotheses **excluded** (IoU > 0.5 with a claimed box is masked out) so the
   hypotheses are forced to explain *different* people.
2. Appearance log-likelihood for hypothesis *i*: `ell_i = -(d_i - d_min) / tau` where `d_i` is
   that hypothesis' current appearance distance to the reference set and `tau = 0.10`.
3. Blind penalty: while a hypothesis is blind, `ell_i -= beta` with `beta = 0.05` per frame, so a
   hypothesis that stopped seeing anything decays but does not die instantly.
4. `logw` normalised each frame (log-sum-exp); a hypothesis with `w < 0.02` for 5 consecutive
   frames is dropped; if fewer than 2 live hypotheses remain and the reported one is blind, spawn
   one from the best rejected candidate above the distractor margin.
5. **Commitment rule**: the emitted box is `argmax w`. A change of the emitted hypothesis is only
   reported after `w_new - w_old >= commit_margin (0.15)` on `commit_frames (3)` consecutive
   frames. Between frames the emitted box follows the currently committed hypothesis (this is
   what makes the mechanism different from just re-running association).
6. `k = 1` must be **bit-identical** to the current shipped `TargetTracker` (plumbing falsifier).

New `CostCfg`/`StateCfg` knobs (all defaulted to inert): `mh_k = 1`, `mh_tau = 0.10`,
`mh_blind_penalty = 0.05`, `mh_drop_weight = 0.02`, `mh_drop_frames = 5`,
`mh_commit_margin = 0.15`, `mh_commit_frames = 3`.

## Measurement protocol (identical to E079/E081/E082 so results are comparable)

- 22 held-out DanceTrack-val sequences, dev `0004/0005/0007` excluded and asserted absent,
  300 frames, subsample 2, mask none, `--jobs 1`, explicit `--target-id` from the frozen map.
- New sweep `scripts/mh_sweep.py` with arms: `mh_k1` (falsifier), `mh_k2`, `mh_k3`,
  `mh_k3_random` (the same K with a random tie-break instead of the posterior — the control that
  separates "more hypotheses" from "calibrated hypotheses").
- Cluster bootstrap 2000 reps seed 0 over sequences; EXPLORE half = odd sequence positions,
  TEST half = even, as in E081/E082.

## Ship rule (pre-registered, not movable)

Ship `mh_k = 3` only if **all** hold:

1. paired ΔTSR vs `mh_k1` > 0 with the 95 % CI excluding 0;
2. paired ΔFAR CI does not include a worsening > 3 pp;
3. the `mh_k3_random` control does **not** also pass (otherwise the gain is from spending more
   compute on more boxes, not from the posterior);
4. wall time ≤ 2.5× the `mh_k1` arm per frame.

Otherwise the verdict is **KILL** and the default stays `mh_k = 1`.

## Kill criteria (checked before any tuning)

- **Degeneracy**: if the mean distinct-hypothesis share (fraction of frames where the K
  hypotheses hold boxes with pairwise IoU < 0.5) is < 0.30, the hypotheses are not exploring
  different people and the mechanism is inert by construction — report as inert, do not tune.
- **Falsifier**: if `mh_k1` is not bit-identical to the shipped tracker on 22/22 sequences, the
  plumbing is wrong and no other number is quotable.
- **Engagement**: count `n_spawned`, `n_dropped`, `n_commits`, `n_commit_blocked` (a switch that
  passed the margin but was held by the consensus window), and `n_swaps_reported`. A mechanism
  that never switches is a no-op.

## What would make me trust a positive

A gain that (a) survives the random-tie-break control, (b) is concentrated in sequences where
E076 measured a high wrong-while-tracking share, and (c) shows `n_commits > 0` with the committed
box matching the GT better than the pre-commit box on the majority of switches. If (b) or (c)
fails, the mechanism is doing something else and the ledger must say so.
