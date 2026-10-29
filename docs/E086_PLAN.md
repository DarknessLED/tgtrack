# E086 — pricing "unreadable" above "different person": does it close the single-frame handover?

**Pre-registration.** Written before any measurement of this knob exists. E086 is a *measurement*
experiment on a mechanism that is already implemented (`CostCfg.missing_appearance_cost`), not a new
mechanism, so the only thing that can be gained is the A/B the README says has never been done.

## The hole, restated from the measurement that found it

motion-dev's `unidentifiable_target` family isolates it: the target walks, halts, and stops being
*readable* by the encoder while still detected and still present in the ground truth. The handover to a
bystander happens on the **first** frame the target is unreadable, while the state is still `tracking`
with `frames_since_update = 0` — so no blind state is entered, global search never arms, and
`search_needs_appearance` is never consulted. The median *accepted* margin is **0.88–1.11**, i.e. ~15×
`distractor_margin`, so no margin threshold can refuse it. Pooled over 30 runs: **39.2 % target success
/ 54.6 % false acquisition** with the bystander, against **58.7 % / 0.0 %** in the solo control.

The reason is the price of absent evidence. A genuinely different person at cosine distance 0.0 is
charged 1.0 (× 1.6 = 1.6), while a feature-less candidate is charged
`missing_appearance_cost` = **0.5** — so "we could not read this person" scores *better* than "this is a
different person", and a clean-descriptor bystander beats the true target exactly when the target's own
crop degenerates.

## Hypothesis

Raising `missing_appearance_cost` so that absent evidence is priced **at or above** a maximally
different person makes the tracker refuse *both* the unreadable true target and the bystander, which is
the correct behaviour for a click-to-track system: no report is better than a confident report of the
wrong person. The lever exists (`cost.missing_appearance_cost`, default **0.5**) and the README states
that with the lone rule off this price "only moves the *recovery*, never the handover itself" — that is
the claim this experiment tests.

## Arms

`cost.missing_appearance_cost` ∈ **{0.5 (control = shipped), 0.75, 1.0, 1.5}**.

Rationale for the grid: 1.0 is the break-even point where "unreadable" ties "different person"; 1.5 and
above make it strictly worse than any possible descriptor; 0.75 is the half step, so the response curve
is not read from two points. `0.5` must reproduce the shipped numbers bit for bit — that is the
structural control (same role that `contrast 0.00` played in E082 and `mh_off`/`mh_dual_frozen` in E083).

## Measurement

1. **Synthetic mechanism check** — `scripts/ablate_motion.py`, family `unidentifiable_target`, 30 runs
   per arm with the bystander present plus the solo control, same seeds as the reference measurement in
   `results/motion_ablation/`. Metrics: target success, false-acquisition rate, **handover count** and
   the frame index of the accepted takeover (the reference observation is "every surviving handover is a
   frame-36 *priced* comparison instead of a single-frame takeover" — that frame index is the mechanism
   signature, so it must be reported per arm).
2. **Real-data price** — the observability-guarded MOT17 grid, exactly the protocol used for the lone
   rule: `MOT17-02`, target id **19**, 300 frames, subsample 2, GT detector, masks **{none, upper}**
   (`results/target_obs19` is the reference epoch; `results/target_obs19_shipped` is a *different*
   baseline and must not be mixed in — the README retracts a claim that did exactly that). Metrics:
   target success, false-acquisition rate, `target_switch_count`, `wrong_box_rate`. If the rig supports a
   second real sequence cheaply, add `MOT17-04` with the same discipline; it is optional, not required.

## Predictions, declared in advance

- **P1 (mechanism).** At `missing_appearance_cost >= 1.0` the frame-1 handover disappears on the
  synthetic family: false acquisition falls toward the solo control's 0.0 and the surviving handovers
  (if any) are the frame-36 *priced* comparisons, not frame-1 takeovers.
- **P2 (synthetic price).** Target success on the synthetic family falls by **at most 3 pp** at 1.0
  (the reference lone-rule price was 2.6 pp at 0.5; a price above ~5 pp means the knob is too blunt).
- **P3 (real-data price).** On the real MOT17 grid the arms stay within **±2 pp** of control, because the
  observability mask makes the target readable for most frames and the knob only fires when appearance is
  missing. **If P3 is violated, that bound is the result** and it decides the ship.

## Ship rule (fixed before measuring)

Ship a non-default value only if **both** hold:

1. synthetic: handover count → **0** with target-success loss **≤ 3 pp** vs the 0.5 control; and
2. real: target success falls **≤ 1 pp** with **no increase** in false-acquisition rate.

Anything else is recorded as a negative and the default stays **0.5**. This rule may not be relaxed after
the numbers exist; the same discipline as E082 (which was killed for missing its own rule by 0.46 pp).

## Deliverables

- `results/handover_mac/report.md` — per-arm tables for both grids, the handover frame indices, the
  control-identity check (`0.5` vs shipped, must be bit-identical), and a one-paragraph verdict.
- A ledger entry **E086** in `docs/EXPERIMENTS.md` (append; do not edit earlier entries).
- If shipped: `CostCfg.missing_appearance_cost` default change + tests + the README limitation bullet
  updated to say the A/B was run and what it decided. If not shipped: the README bullet gains the
  measured price and the knob, the sweep and the negative are kept (as with the lone-rule negative).

---

## REVISION (2026-10-29 03:5x) -- the original premise is already falsified by existing data

Before running anything I checked the rig, and the measurement this plan asked for
already exists: `results/motion_ablation/unid_cost.md` swept
`CostCfg.missing_appearance_cost` over `{0.0, 0.1, 0.25, 0.5, 1.0}` across the
`unidentifiable_target` family (similarity x mode x model, 40 configurations) and
**every metric is bit-identical at every price** -- TSR 0.2950, FAR 0.7692,
handover frame 18.00, `n_dets at the handover = 1.00`. The reason is structural, not
statistical: the handover happens when the bystander is the ONLY finite candidate,
so `association.select_candidate` leaves `margin = inf` and the price of the
candidate never enters the comparison. Extending the sweep to 0.75/1.0/1.5 cannot
change that, and the README already says so ("price only moves the recovery, never
the handover itself"). **E086 as pre-registered is therefore dead on arrival, and
the honest response is to publish the reason rather than to run it.** The synthetic
half of the original plan is not deleted -- it is answered above and stays in the
ledger as a pre-registration that was falsified by a later re-read of our own data.

### Revised question (E086-r)

The one thing that is genuinely unmeasured is the real-data A/B of the OTHER knob
in the same hole, `CostCfg.lone_candidate_accept_cost` (shipped default `None`).
The synthetic rig measured it (`results/motion_ablation/lone_bound.md`: handovers
3/4 -> 1/4, success 94.7 % -> 97.4 % at price 2.6 pp), but the real-data claim that
circulated in the README was RETRACTED because the comparison used
`results/target_obs19_shipped`, which contains no armed variant -- i.e. it compared
against a different baseline, not a controlled pair. So:

- **Arms**: shipped control (`lone_candidate_accept_cost = None`) plus armed
  `{0.30, 0.40, 0.50}` (cost units: a lone candidate is refused when its combined
  cost exceeds the bound). Larger bound = more permissive.
- **Protocol**: `scripts/vote_sweep.py` (DanceTrack val, 300 frames, subsample 2,
  mask none, GT detector, the same per-sequence target ids E079-E084 used).
- **Sequences**: the six informative ones, chosen by a rule fixed BEFORE the new
  numbers exist -- four where the shipped tracker actually reports strangers
  (`dancetrack0010` FAR 0.680, `0018` 0.587, `0073` 0.583, `0077` 0.410) and two
  near-zero-FAR controls (`0034` 0.155, `0079` 0.023). Without the high-FAR four a
  guard against false acquisition has nothing to measure; the controls say whether
  the guard binds outside the regime it targets.
- **Predictions**: P1 pooled FAR falls by >= 2 pp at 0.40 and 0.50; P2 pooled TSR
  loss <= 1 pp; P3 the two controls are bit-identical.
- **Ship rule**: enable only if P1 and P2 hold at the same price AND no single
  sequence loses more than 3 pp TSR. Otherwise record the negative and keep
  `None` shipped.
- **Falsifier**: if P3 fails (the controls move), the guard binds outside the
  handover regime -- that is a kill regardless of the pooled numbers, because a
  global refusal changes search behaviour on sequences that never had the disease.

This revision is written before the armed summaries exist; no E086-r number has been
read at the time of writing.
