# E079 — Multi-reference identity vote (replacing the hard anchor veto)

Pre-registered before implementation. Author: Lead. Date: 2026-10-28.

## Why (the measurement that forces this)

E076 (24 traces / 6806 frames, shipped config) decomposes the held-out single-target loss as

    0.768 = 0.511 blindness + 0.238 wrong-while-tracking + 0.019 other

so **two thirds of the loss is the tracker refusing correct evidence**. E077 then localised the
refusal to one line: on dancetrack0097, **61/70 frames refuse with `all_gated:anchor_ceiling`** — the
E027/E029 hard appearance ceiling sets `total = +inf` for *every* candidate, because the single
frozen click crop is a strict reference and the target's look drifts (pose, blur, viewpoint, scale).
When that happens no other cost can matter: the veto is upstream of the acceptance bound.

The anchor is the only *unpoisonable* reference we have (E021/E025/E078 all failed trying to police
poisonable ones), but it is also the only **rigid** one. E078's own data add the second half of the
argument: the adaptive band **saturates at its 0.75 ceiling on ~75 % of frames**, so the "adaptive"
anchor is in practice a constant threshold most of the time.

A single hard reference cannot be both safe and tolerant. A **vote between references with different
poisonability profiles** can.

## Mechanism

Three independent references, each with its own band:

| ref | what it is | poisonable? | band |
|---|---|---|---|
| **R1** | frozen click crop (existing) | no | existing `anchor_adaptive_*` band / `anchor_adaptive_ceil` |
| **R2** | refreshed anchor: EMA of accepted descriptors | **only by agreement** | same band as R1 |
| **R3** | LTM gallery best distance (existing) | yes | existing gallery acceptance |

Decision: a candidate whose current refusal is the anchor veto is admitted if it passes **at least
k of the three** references (k = 2 majority arm, k = 3 unanimous control arm). Everything else —
distractor margin, lone-candidate bound, search margin, missing-appearance rule while searching, and
the whole blind-state transition logic — is untouched and still applies afterwards.

**R2's update rule is the anti-poisoning device, and it is the load-bearing detail:** R2 may only be
updated on frames where **R1 and R3 both already accept** the same box. A swap that fools R3 (the
poisonable gallery) cannot move R2, because R1 — unpoisonable — is still refusing. So R2 can follow
genuine appearance drift (slow, gradual, both references agreeing) but cannot be dragged by a single
hard decision.

`alpha = 0.1` (about a 10-frame time constant). Not swept, to keep the multiplicity at 3 arms.

## Knobs (all OFF by default; shipped behaviour must be bit-identical with the flag off)

```
CostCfg.anchor_vote: str = "off"          # "off" | "majority" | "unanimous"
CostCfg.anchor_vote_alpha: float = 0.1
```

## Arms (`--ablate vote`)

| arm | meaning |
|---|---|
| `vote_off` | reference: today's hard veto (must reproduce the E070/E073/E078 reference numbers exactly) |
| `vote_majority` | k = 2 of 3 |
| `vote_unanimous` | k = 3 of 3 (control — must be stricter than the reference or the implementation is wrong) |

## Protocol (identical to E070/E073/E078 so numbers are comparable)

22 held-out DanceTrack-val sequences, dev split `dancetrack0004/0005/0007` excluded **and asserted
absent**, `--max-frames 300 --subsample 2 --mask none --jobs 1 --debug-csv`, target identity picked by
the `vote_off` pass and pinned for every other arm (each run's `meta.target_ids` asserted). Output
`results/vote_sweep/` with `runs/`, `summary.json`, `REPORT.md`. Smoke one sequence before the full
run. If the box is still slow, report partial coverage with the exact sequence count, never silently.

## Pre-registered ship rule (do not move it after seeing numbers)

**SHIP** `vote_majority` only if all three hold:
1. pooled TSR rises by **>= +3.0 pp** with the paired 95 % CI (2000-rep cluster bootstrap over
   sequences, seed 0) excluding zero;
2. the **upper** bound of the paired dFAR CI is **< +3.0 pp**;
3. the mechanism is engaged: the `blind AND present` share falls versus `vote_off`.

Otherwise **KILL** and record why. `vote_unanimous` is a control, never a candidate.

## Engagement instrumentation (mandatory, else a null result is uninterpretable)

Per-run diagnostics: `vote_n`, `vote_r1`, `vote_r2`, `vote_r3` (how many references passed for the
chosen candidate), `vote_rescued` (candidates that would have been `+inf` and were admitted by the
vote), `vote_r2_updates` (frames where R2 was allowed to move), and the reference-band saturation
share. Aggregate counts go in `REPORT.md`.

## Known ways this can fail (state them in the report rather than discovering them late)

- **Majority may just be looser.** If R3 alone is already permissive, k=2 is nearly "R3 decides",
  which is the poisonable path — then FAR should rise and the ship rule rejects it. That is a real
  finding, not a bug.
- **R2 may never move** if R1 and R3 rarely agree on real data; then the vote is effectively 2-way
  (R1 vs R3) and the arm is a re-weighting of the veto. Report the R2 update rate.
- `vote_unanimous` being *identical* to the reference means the veto was already unanimous (i.e. R3
  and R2 never independently pass when R1 fails), which would make the whole mechanism inert.
