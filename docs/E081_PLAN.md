# E081 — arm the E079 vote only while the tracker is in distress

**Status**: PRE-REGISTERED before the sweep was launched (2026-10-28 02:50 local).
Written after E079 closed as KILL-BY-RULE (round 77) and before any E081 number existed.

## The observation E079 left open

E079's `vote_majority` was the first mechanism in the whole campaign to move both
criterion-2 metrics in the right direction on held-out data:

| arm | TSR | FAR | id_consistency |
|---|---|---|---|
| `vote_off` (shipped) | 0.2104 | 0.2472 | 0.4358 |
| `vote_majority` | **0.2668** | **0.1912** | **0.5012** |
| paired delta (21 seqs) | **+5.88 pp** [−1.85, +15.04] | **−5.65 pp** | +6.87 pp |

It cleared the FAR bar (needed ≤ +3 pp, got −5.65 pp) by 13.4 pp and missed the TSR
bar (needed ≥ +3 pp, got +5.88 pp) by a confidence interval only: the lower bound is
−1.85 pp. The TSR gain is real in the point estimate and unpriced in uncertainty.

## Hypothesis

The vote overrules the frozen click crop on EVERY frame. But the click crop is a
*good* reference while the tracker is healthy and a *stale* reference after it has
been blind for a while (that is exactly what E014/E020/E075 measured: the failures
are blindness first, then a wrong latch during the coast). So part of the vote's
gain comes from frames where it should not have fired, and those frames carry the
TSR cost that keeps the interval wide.

**If** the vote is armed only while the tracker is currently distressed, **then** it
recovers the FAR gain where the click crop is genuinely stale while giving back less
TSR than the unconditional vote.

## Mechanism

`cost.anchor_vote = "adaptive"`, k = 2 (same references as `majority`), plus a gate:

    distress = share of the trailing W frames spent blind
               (OCCLUDED_SOFT / LOST_SHORT / LOST_LONG / OUT_OF_FRAME)
    vote runs  iff  distress > tau

- W = 50 frames (`anchor_vote_distress_window`).
- distress is sampled in `step()` from the state the frame ENTERS in, so the window
  is strictly trailing and the gate cannot see the frame it is deciding.
- tau < 0 disables the gate → `adaptive` becomes bit-identical to `majority`.
- tau = 1.0 can never be exceeded → the gate never opens → identical to `off`.
  Both end points are pinned by tests (`tests/test_anchor_vote.py`, 23 tests).

## Design (frozen before measurement)

- 22 held-out DanceTrack-val sequences, 300 frames, subsample 2, `--mask none`,
  dev 0004/0005/0007 excluded and asserted absent (E070's protocol, reused verbatim).
- Arms: `vote_adaptive_{0.20, 0.35, 0.50, 0.65}`, plus the `1.00` null control.
  Reference and `majority` come from E079's completed 22-sequence records.
- tau is chosen on the EXPLORE half only (the 11 sequences ordered by name, odd
  positions), by TSR subject to explore FAR < 0.20.
- The chosen tau is then evaluated on the TEST half (even positions), paired against
  `vote_off`, cluster bootstrap 2000 reps seed 0.

## Pre-registered decision rule

SHIP only if BOTH hold on the TEST half, paired against `vote_off`:
1. dTSR > 0 with a 95 % CI excluding 0; and
2. dFAR < 0 with a 95 % CI whose upper bound is below +3 pp.

KILL if: the gate never opens (`vote_frames == 0` on the explore half while the veto
fired, i.e. the mechanism is inert), or no arm on the explore half reaches
explore TSR > the reference while keeping explore FAR < 0.20.

Either way the ledger entry states the verdict and the defaults stay
`anchor_vote = "off"` unless SHIP.

## Secondary checks (not gating)

- `vote_gated` / `vote_frames` per sequence: the gate must visibly act, otherwise the
  arm is measuring noise.
- MOT17-02 upper-body partial-view grid (target 19, 300 frames): criterion 3 must not
  regress below 0.8545 − 0.5 pp.
