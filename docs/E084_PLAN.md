# E084 plan — switch the identity reference: a click-time MULTI-PART bank

Pre-registered before implementation. Written after E083 was killed (dominated by E079) and
E082b was launched, because both produced the same lesson from different directions: **every
statistic computed from the tracker's own observation history is self-referential**, and adding
another vote over the same evidence cannot separate "the look changed" from "this is someone else".

## Hypothesis

The failure is not the *threshold* on the reference but the *reference itself*. Today the identity
ceiling is a single distance to **R1 = the full-body crop at click time**
(`cost.anchor_ceiling_*`, E027) plus two statistically derived beds (R2 = EMA of accepted crops,
R3 = an LTM gallery entry). All three are full-body and all three are contaminated by the same
failure mode: during a partial view, or once a wrong latch is absorbed, every bed drifts the same way.

A person's **parts are not equally informative and do not drift together**. The click frame already
contains the upper body, the lower body and the head as separate crops. If the ceiling is enforced
per part ("the candidate must look like the target in *at least one* part that is *actually visible*")
then a partially visible target is still verifiable, while a bystander must coincidentally match the
*same* part. This is the one idea in the campaign that has a measured mechanism behind it rather than
a knob: `scripts/exp_partial_view.py` showed the part descriptors separating 0.92 (true, lower body
occluded) vs 0.48 (impostor) where the whole-body descriptor managed only 0.757 vs ~0.5.

## What must be measured FIRST (the prerequisite, not the hypothesis)

Before implementing a multi-part ceiling, measure **which bed refuses an admissible candidate, and
whether that candidate was the true target**. Today that is invisible: the debug CSV records the
*accepted* box, never the rejected candidates. So step 0 is instrumentation:

- new knob `cost.anchor_bed_log` (path, default "" = off): when set, one JSONL line per
  anchor-vetoed candidate per frame with `frame`, `candidate_index`, `box_xyxy`, `d_R1`, `d_R2`,
  `d_R3`, `limit`, `passes`, `admitted`. Gated by the knob so the shipped path is untouched
  (the sweep's bit-identity controls must keep holding);
- join those lines against each sequence's GT: label each rejected candidate as
  **target / distractor / nobody-in-particular (IoU < 0.3)**;
- report, per sequence: how often the ceiling refuses the *true* target, and which bed is
  responsible (R1 vs R2 vs R3).

Kill criterion for step 0: if, on the 6 informative held-out sequences, the ceiling refuses the true
target in fewer than ~5 % of the frames where it refuses anyone, then there is nothing for a per-part
ceiling to recover and E084 stops there (recorded as a negative).

## Mechanism (only if step 0 passes)

- `memory.part_bank_at_click: bool` — capture `full/upper/lower/head` crops at `initialize()`,
  embed each, store as the click bank (`Embedder.encode(..., parts=("full","upper","lower","head"))`,
  one batched call, already supported).
- `cost.part_ceiling: str` in {`"off"`, `"any"`} — with `"any"`: a candidate passes the identity
  ceiling iff for at least one part whose *visible ratio* on the candidate box is >= 0.5, its distance
  to that part's click crop is <= `limit`. Visibility comes from `parts.estimate_partial_state`
  (already measured and tested); a part that is not visible is not evidence and neither passes nor
  fails.
- `"off"` must be **bit-identical** to today's behaviour (structural falsifier, same pattern as
  E081's tau=1.0 and E082's delta=0 controls).
- One knob more, and only one: `cost.part_ceiling_margin` added *in part units*, default 0.

## Protocol

- held-out split only: the 6 informative sequences (`0010`, `0018`, `0034`, `0073`, `0077`, `0079`)
  for the screening run, then all 22 if the screen is positive;
- 300 frames, subsample 2, mask none, `--jobs 1`, `scripts/vote_sweep.py --arms
  vote_off,part_any,part_off` (the third arm is the bit-identity control);
- read TSR, FAR, id_consistency and the **per-sequence** deltas, and report win/tie/loss counts with
  magnitudes next to the pooled numbers — E082 established that a pooled CI cannot certify a sparse
  effect, so the pooled number alone is not the decision.

## Ship rule (fixed now, before any measurement)

Ship `part_ceiling="any"` only if, on the FULL 22 held-out sequences:
1. pooled TSR rises by >= +3 pp **and** the paired (sequence-clustered) 95 % CI excludes 0;
2. pooled FAR does not rise by more than 3 pp, and its CI upper bound does not exceed the shipped
   FAR's upper bound by more than 3 pp;
3. **and** the win/tie/loss counts are not dominated by a single sequence (no one sequence may
   contribute more than half of the total gain) — the E079/E083 failure mode is a mechanism carried by
   2-3 sequences, which is exactly what a pooled mean hides.

If any part fails, the arm is recorded as a negative and the default is left alone.
