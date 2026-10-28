# E082 — require a CONTRAST, not just a majority, before overriding the click crop

**Status**: PRE-REGISTERED (2026-10-28 21:05 local), written while the E081 sweep was still
running and before any E082 number exists. No `src/` file may be edited until the E081 sweep
releases `results/vote_sweep/.sweep.lock` — a mid-sweep source edit invalidates comparability
(E070's trap, recorded in `docs/BRIEF.md`).

## What E081 established (and why E082 is not a threshold tweak)

E081 gated the E079 vote on the tracker's own distress (trailing blindness share). On the full
22 held-out sequences:

| arm | TSR | FAR | id |
|---|---|---|---|
| `vote_off` (shipped) | 0.2104 | 0.2472 | 0.4358 |
| `vote_majority` (blanket) | 0.2668 | 0.1912 | 0.5012 |
| `vote_adaptive_0.20` | 0.2362 | **0.1896** | 0.4875 |
| paired `majority` vs off | +5.88 pp | -5.65 pp | +6.64 pp |
| paired `adaptive_0.20` vs off | **+2.63 pp** [-0.37, +6.34] | **-5.75 pp** [-14.92, +2.42] | +5.27 pp |

The gate kept essentially ALL of the FAR benefit (-5.75 vs -5.65 pp) and gave back more than
half of the TSR benefit (+2.63 vs +5.88 pp). In other words **gating on distress removes the
good rescues first**: the rescues that recover the target happen in frames where the tracker is
NOT distressed, which is exactly the regime E076 labelled "wrong while tracking" (24 % of the
loss) and which E075 showed is a confident-but-wrong lock. So the useful question is not *when
the tracker is uncertain* but *when the frozen click crop is the wrong reference*.

## Hypothesis

A candidate reaches the vote only because R1 (frozen click crop) refused it. The vote currently
admits it whenever 2 of 3 references pass an ABSOLUTE distance bound — that is, R2 and R3 only
have to be "close enough", not closer than R1 by any margin. A candidate that is marginally
below the bound in R2/R3 but wildly far in R1 is being admitted on weak evidence.

**If** the vote additionally requires that the candidate looks substantially MORE like the
refreshed references than like the click crop, **then** the rescues it admits are the ones where
R1 is genuinely stale, and the TSR point estimate keeps its sign while its interval tightens.

## Mechanism (one extra array operation in `_vote_admit`)

    contrast(c) = dist_R1(c) - min(dist_R2(c), dist_R3(c))
    admit when  passes >= 2  AND  contrast >= delta

- `delta = 0` is bit-identical to E079's `vote_majority` (pinned by a test), so the E079
  numbers stay the reference arm rather than being re-measured.
- New knob: `CostCfg.anchor_vote_contrast: float = -1.0` (negative = disabled = shipped).
- No new descriptor, no new constant in the cost function, no new frame of latency: all three
  distances are already computed by the existing vote.

## Design (frozen before measurement)

- Protocol identical to E070/E073/E078/E079/E081: 22 held-out DanceTrack-val sequences, dev
  0004/0005/0007 excluded and asserted absent, 300 frames, subsample 2, `--mask none`, `--jobs 1`.
- Arms: `vote_contrast_{0.10, 0.20, 0.30}` plus `vote_contrast_0.00` as the bit-identical
  control against E079's `vote_majority`.
- Selection on the EXPLORE half only (odd positions of the sorted 22), scored once on TEST half.
- Ship rule (same shape as E079/E081, evaluated on TEST, paired vs `vote_off`):
  1. dTSR > 0 with the paired 95 % CI excluding 0; and
  2. dFAR < 0 with the paired 95 % CI upper bound below +3 pp.
- KILL if no arm achieves both, or if every arm is bit-identical to `vote_majority`
  (i.e. the contrast never binds — the falsifier for the mechanism, reported as "the contrast
  axis is empty" rather than as a failed threshold).

## What this experiment cannot settle

If the contrast gate tightens the TSR interval but the point estimate stays near +3-6 pp, the
remaining limitation is the size of the surviving sample (E079: only 391 of 6,714 rescues were
finally accepted, i.e. ~6 % of frames), not the choice of gate. That limit would then be a
property of the downstream refusal stack, and the next target would be the *acceptance* path
rather than the *admission* path.
