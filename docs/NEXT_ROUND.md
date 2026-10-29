# Next round: gate calibration on the HELD-OUT split

Written at the end of the review round. Everything below is derived from measurements already on disk;
no new model, no new dataset, no fine-tuning is needed.

## The diagnosis this round produced

1. **Criterion 2's real value is TSR 0.2104 / FAR 0.2472 over the 22 held-out sequences**
   (E070's shipped-configuration re-run on today's code; E068's 0.2102 / 0.2364 is superseded — it predates `target_tracker.py` 18:40:11) 
   (`results/heldout_dancetrack/report.json`, E068), not the dev-split 0.512/0.280. The dev TSR lies
   outside the held-out CI, so the dev split certifies nothing (its intervals were +-0.394 TSR and
   +-0.708 FAR on n=3).
2. On held-out data the failure mode is **losing the target**, not locking onto the wrong person
   (FAR is statistically unchanged between dev and held-out).
3. The target is **detected in ~84 %** of the frames where we lose it
   (`results/tracklet_evidence/analysis_reid.json`, E069): a MOT box with IoU >= 0.5 to the target's GT
   exists on 84.4 % of 969 measurable loss frames.
4. The all-person MOT layer's **tracklet id is not usable as an external anchor**: only 9.7 % of frames
   inside >=5-frame loss runs are single-id anchored and all five >=50-frame runs are unanchored (E069).
5. Two independent families are therefore closed: self-referential appearance evidence (E025, E032,
   E032b, E064 -> METHOD.md section 8) and external MOT tracklet identity (E069).

**Conclusion: the remaining gap is gate calibration, not missing evidence.** The same acceptance gate
that prevents a wrong-identity lock (E022 `cost.search_margin = 0.20`, E029 adaptive band) is the one
refusing a correct re-acquisition.

## The one experiment to run

Sweep the search-side gate **on the 22 held-out sequences only** (never on dancetrack0004/0005/0007):

- primary axis: `cost.search_margin` in {0.0, 0.10, 0.15, 0.20 (shipped), 0.30};
- secondary: `cost.anchor_adaptive_k` in {3.0, 4.0 (shipped)}, only if the primary shows a monotone trade;
- protocol: `scripts/heldout_report.py` (it refuses sweep flags by design — add a narrow, explicit
  exception for these two knobs, or drive it from a child process that pins `build_variants`), 300
  frames, subsample 2, `--jobs 1`, `--target-id` from each run's own `meta.target_ids`;
- report per axis point: pooled TSR and FAR with sequence-level cluster-bootstrap 95 % CI (2000 reps),
  plus the count of sequences whose TSR moves by > 5 pp, so a bimodal shift cannot hide inside a mean;
- acceptance rule, fixed BEFORE the sweep: ship a value only if pooled TSR rises by >= +5 pp **and** the
  upper bound of its FAR CI does not exceed the shipped FAR's upper bound. Anything else is recorded as
  a negative and the default is left alone.

## Why not the alternatives

- More detector recall: six negatives (E001/E002/E003/E011/E055/E056) and an active detection problem
  would show low coverage, but coverage is 84.4 %.
- New identity evidence: both admissible families are closed by measurement (points 4-5).
- More knobs in the same family: E032b showed the family is bit-identical at its own end state.

## Update 2026-10-28 (round 27) — E081 closed KILL; the gate is measured, the hypothesis is dead

- `anchor_vote` stays **"off"**. E081's distress gate fails its own ship rule (best paired
  dTSR +2.63 pp, CI [-0.37, +6.34]) and, more importantly, falsifies the hypothesis behind it:
  the gate keeps 85-102 % of the FAR benefit but only 7-45 % of the TSR benefit, so the useful
  rescues happen in frames where the tracker is confident, not distressed.
- The gate's plumbing is *proven* correct on real data: the `tau = 1.00` null arm is
  bit-identical to `vote_off` on all 22 held-out sequences (0.00 pp, CI [0.0, 0.0]).
- `vote_adaptive_0.20` pooled FAR **0.1896** is the first sub-0.20 FAR in the campaign with TSR
  above the reference. Criterion 2's FAR half is reachable; its TSR half (>= 0.55) is not.
- Next experiment is pre-registered in `docs/E082_PLAN.md`: replace "when is the tracker
  uncertain" with "when is the click crop the wrong reference", via a contrast requirement
  `d_R1 - min(d_R2, d_R3) >= delta` (delta = 0 must reproduce E079's blanket arm bit-for-bit).

## Update 2026-10-28 (round 28) — E082 closed KILL by rule, but it produced the best arm so far

- `anchor_vote` stays **"off"**. E082's contrast gate at `delta = 0.10`
  (`vote_contrast_0.10`) is the best arm measured on **both** held-out axes and the first
  configuration ever below the FAR 0.20 barrier: **TSR 0.2770 / FAR 0.1830** vs shipped
  **0.2104 / 0.2472** (pooled dTSR +6.83 pp, paired CI [-0.46, +15.93]; dFAR -6.42 pp).
  It misses the pre-registered rule by 0.46 pp and is therefore **not shipped**.
- The rule did not fail for statistical noise reasons but for **structural** ones: the effect is
  sparse. Per-sequence: 5 gains > 2 pp (+177.0 pp total), 1-2 losses > 2 pp (-44.5 pp), and
  **15 of 22 sequences bit-identical** (median dTSR 0.00). A 22-sequence cluster bootstrap has
  almost no power against that shape; see METHOD.md's new "Lesson (E082)" section.
  The falsifier held: `contrast 0.00` is bit-identical to `vote_majority` on all 22 sequences.
- **Queue, in order:**
  1. **E082b** (pre-registered): `contrast_0.10` vs `vote_off` only, same 22 sequences,
     `--max-frames 600` to lift the sequences pinned at the TSR floor. ~44 pairs, ~50 min.
     Its purpose is to *buy power*, not to move the bar.
  2. **E083 readout** (implemented, sweep launching): dual structured hypotheses
     (`mh_dual` = shipped tracker + a gallery-vote clone, disagreement-triggered 12-frame
     likelihood validation, logw = -2 r_h, commit at margin 0.15 for 3 frames) versus
     `mh_dual_frozen` (both run, reported box pinned to the shipped arm — the control) and
     `mh_off` (structural falsifier). Per-sequence expectations were declared in advance:
     0010/0079 must **not** adopt the vote's side; 0018/0034/0077/0073 should.
  3. **E084** (pre-registered): ultimately, switch (rather than repair) the identity reference —
     a click-time multi-part *bank* instead of the single full-body crop R1. Requires logging
     which admissible candidate each anchor bed rejects and why, which is applied only after
     the E082b/E083 sweeps release the GPU.
- **Harness trap (cost ~1 h of wall clock):** `scripts/vote_sweep.py`'s single-instance lock
  races. The E082 parent released it, but a surviving child process immediately re-acquired it
  (pid 69636 at 23:23:25, child 63888), so every "wait for the lock file to disappear" launcher
  starts 1 s runs that are all refused. The correct wait condition is **"no live
  `vote_sweep.py` process"**, not "no lock file" — `_x/e083_sweep2.py` implements that.

## Update 2026-10-29 (round 29) — E083 KILL (dominated by E079); E082b running

- **E083 (dual structured hypotheses) is dead.** On the 9 informative held-out sequences `mh_dual`
  means **0.3170** vs `vote_off` **0.2597** (+5.73 pp) but vs `vote_majority` **0.3984** — i.e. it
  recovers roughly *half* of the vote's gain on each winner (+41.3 of +74.2 on 0018, +7.7 of +42.7 on
  0034, +15.7 of +34.0 on 0077, +6.0 of +18.4 on 0073) while still paying roughly half of its loss on
  each loser (-10.7 of -23.5 on 0010, -8.3 of -21.0 on 0079). A coin-flip discriminator, exactly as
  the `r_h = mean(d_own)/mean(d_other)` formulation predicts.
- **The valuable residue is the harness, not the mechanism:** `mh_dual_frozen` is bit-identical to
  `vote_off` on **9/9** sequences and `mh_off` on **2/2**. The dual-tracker plumbing demonstrably does
  not leak into the reported box, so any future two-hypothesis experiment has a clean rig.
- **E082b is running now** (`results/vote_sweep_600/`, `vote_off` vs `vote_contrast_0.10`, 600 frames,
  22 held-out sequences, `E079_OUT` redirected so the two epochs can never be pooled; `--max-frames`
  now exists and *refuses* a non-300 frame count without `E079_OUT`, precisely to prevent cross-epoch
  arithmetic). Purpose: buy statistical power on the one arm that is best on both Pareto axes
  (0.2770 / 0.1830), not to move the ship rule.
- **Next after E082b:** E084, re-scoped by E083's lesson from "add a vote" to "switch the reference":
  a click-time **multi-part bank** (full/upper/lower/head crops each scored separately) instead of a
  single full-body crop, so the beds are structurally different rather than statistically derived from
  the same poisoned history. Requires the bed-rejection logging patch (which admissible candidate each
  bed refuses and why) — applied only once the GPU is free.
## Update 2026-10-29 (round 30) — E082b's premise falsified, E084 measured and killed, k = 1 killed

- **E082b is dead, and it falsified its own premise.** 600 frames do not buy power, they buy drift:
  `vote_off` pooled TSR **0.2045 -> 0.1512** (-5.3 pp) and FAR 0.2388 -> 0.2725, with the sequences
  pinned at the floor **unchanged at 7/22** -- so the extra horizon adds failure modes without lifting
  anything off the floor (`dancetrack0079` alone: 0.9133 -> 0.0683). The design premise ("more frames
  lift the floor sequences") was false, so the contrast arm was never spent. **Cross-epoch numbers are
  never pooled**: the 300-frame protocol flatters the tracker and sequence rankings are
  horizon-dependent.
- **E082b also produced a reusable harness trap**, now in METHOD: `PROTOCOL["max_frames"]` mutated after
  `parse_args` does not reach per-pair CHILD processes, which re-import the module -- every pair failed
  `PROTOCOL GUARD failed: max_frames=300` while the parent believed 600. Any protocol constant a child
  stamps must travel by **environment** (`E079_MAX_FRAMES`).
- **E084 step 0 answered its pre-registered question and killed its own mechanism** (3.96 % of ceiling
  refusals are the true target, bar 5 %): 67 % of refusals are genuine distractors, 7 % background.
  The distribution is the finding -- target refusals are **0 %** on the two sequences where the vote
  loses (`0010`, `0079`) and **18-33 %** on three where it wins -- i.e. the ceiling binds exactly where
  the metric is decided. And the rescue is always the same bed: **R3 18 times, R2 zero times**.
- **E084b (k = 1) is KILL, and it explains why the contrast gate is the only arm that ever won**:
  `vote_any` mean TSR 0.5406 / FAR 0.2287 vs shipped 0.3495 / 0.4064 on the six informative sequences,
  but dominated by `contrast_0.10` (0.5926 / 0.1282) and it breaks `0079` (0.9133 -> 0.6800, FAR
  0.023 -> 0.217) where the contrast gate is bit-identical. Absolute closeness to one bed admits
  similar-looking strangers; the *relative* requirement (R1 worse than both refreshed beds by a margin)
  is the discriminator.
- **E085 (fresh-cohort replication) is blocked on data, not on protocol**: `data/raw/dancetrack/` has
  val only and `huggingface.co` is unreachable from this host. `scripts/vote_sweep.py` now takes a
  cohort via `VOTE_DATA_ROOT`/`VOTE_DEV_CHECK=0` and skips the E070 reference comparison on a custom
  root (val and train share sequence NUMBERS, so that comparison would be a name collision).
- **What this leaves open on criterion 2:** `anchor_vote` stays `"off"`; the best measured arm remains
  `vote_contrast_0.10` (0.2770 / 0.1830), unshipped because a 22-sequence cohort cannot certify a
  sparse effect and a fresh cohort is unavailable. The TSR half of the criterion (>= 0.55) is not
  reachable by identity mechanisms at all -- every arm in the family lands within +7 pp of 0.2104 --
  so the next work should target product-level defects the campaign has *documented* rather than more
  knobs in a closed family.
- **Next: the handover hole.** motion-dev's grid found that identity handover between two people
  (target occluded, distractor visible at the same place) is a real defect with two knobs already
  exposed (`handover_*`), and the README documents it as a limitation. Turning a documented limitation
  into a fixed behaviour is worth more than another criterion-2 arm that the ship rule cannot certify.


## Update 2026-10-29 (round 31) — the lone guard measured on real data: real effect, dominated, and the identity line is now exhausted

**E086, originally pre-registered, was falsified before it ran.** It proposed extending
`missing_appearance_cost` from {0.5..1.5}. `results/motion_ablation/unid_cost.md` already
sweeps it over {0.0, 0.1, 0.25, 0.5, 1.0} and every metric is **bit-identical at every
price**, because the handover happens with `n_dets = 1` where `select_candidate` leaves
`margin = inf`: with no runner-up, price never enters the decision. The post-mortem is
appended to `docs/E086_PLAN.md`; the mechanism was never going to work.

**E086-r measured the other knob in the same hole, with the control the retraction said
was missing.** Six held-out DanceTrack sequences (four shipped FAR >= 0.40, two low-FAR),
shipped `None` vs armed `{0.30, 0.40, 0.50}` (`results/lone_guard/report.md`):

| pool | off | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|
| TSR | 0.3495 | 0.3092 | 0.3005 | 0.2991 |
| FAR | 0.4064 | 0.2442 | 0.2562 | 0.2162 |
| id_cons | 0.4397 | **0.5612** | 0.5281 | 0.5463 |

The guard is **not inert on real data** (unlike its sibling knob): pooled FAR -16 to
-19 pp at a -4 to -5 pp TSR price, `id_consistency` +12.2 pp. Ship rule fails (pooled-TSR
and worst-case clauses; `0077` -15.7 pp), and on the same six sequences
`anchor_vote="contrast"` δ=0.10 measures **0.5926 / 0.1282**, dominating every lone arm
on both axes. Default stays `None`.

**The falsifier lesson (P3).** P3 said the two low-shipped-FAR sequences must not move.
`dancetrack0034` moved *favourably* (+11.3 pp TSR, -7.6 pp FAR): its shipped run was
silently latched onto strangers (FAR 0.155 is not a control, and my sequence labelling
called it one), and refusing lone candidates also kept the gallery clean, raising later
correct matches. "Controls must not move" is the wrong shape — the right statement is
"movement must be explicable". The pre-registered rule still killed the arm, and it was
not relaxed afterwards.

**Where this leaves the campaign.** The identity line is now exhausted rather than
merely slow: on the k axis every point is measured (k=1 `vote_any` dominated, k=2 the
shipped default, k=3 `unanimous` KILL), the contrast axis is swept with its best point
0.46 pp short of the ship rule, the adaptive axis KILL, the dual-hypothesis mechanism KILL
(worse than plain vote), and the lone guard dominated by contrast. Four independent
mechanisms have now landed in the same 4-7 pp band with the same trade-off shape, which is
the signature of a **metric-level ceiling**, not a missing knob.

The open question is no longer identity. It is the distribution: 15 of 22 sequences are
bit-unchanged by every arm, and **7 / 22 sit pinned at TSR < 0.06**. A pooled mean over a
population with a 7-sequence floor is a poor target to optimise. **Next (E087): diagnose
the floor, not the mean** -- for each floor sequence, split the visible frames into
`detected & tracked`, `detected & not tracked`, `not detected`, and report which stage owns
the loss. If the floor is detection or observability, every remaining identity knob is
buying points in a band the benchmark cannot resolve, and the honest next move is a
protocol decision (report FAR/TSR on the informative subset, state the floor), not another
arm. Preregister before running: expectation `not detected` dominates the floor sequences,
because E001-E011 already showed DetA flat at 0.42-0.44 across six detector backends.

### Addendum (same round, no GPU): E087 was run immediately -- and its answer redirects the plan

The "next: diagnose the floor" item above was executed with data already on disk (the 22
shipped `vote_off` summaries; script `_x/e087_loss_split.py`), so the prediction can be
scored rather than left open.

- **Measured:** of the total target loss, **30 % happens while locked on the wrong identity**
  (summed `loss_rate_locked` 5.232) and **70 % happens while not locked** (residual 12.268).
  18 of 22 sequences are unlocked-dominant; 6 of the 7 floor sequences (TSR < 0.06) are, and
  `dancetrack0097` is the cleanest case — `id_consistency` **1.000**, FAR 0.000, but it does not
  report on 84 % of visible frames.
- **My preregistered expectation was "not detected dominates"** (inferred from DetA being flat
  at 0.42-0.44 across six backends). That specific claim was **not testable** with the stored
  fields — vote_sweep summaries carry no per-frame detection attribution — and what I measured
  instead is a different split (locked vs unlocked), which points at the same conclusion by a
  shorter route: the loss lives in seeking/reporting, not in the identity decision.
- **Consequence:** the identity family's ceiling is ~30 % of the loss, which retro-explains
  (a) 15/22 sequences bit-unchanged under every identity arm, (b) criterion ② being
  unreachable by identity work even with perfect decisions, and (c) the violently sparse
  per-sequence distribution of the vote (it gains where locked-loss is large).
- **Revised next step (E088), replacing "another arm":** on the six unlocked-dominant floor
  sequences, run one arm with `scripts/run_target.py --debug-csv` (the vote_sweep runs store
  only the pooled table, so per-frame attribution needs a real run, ~35 min) and split the
  unlocked loss into `no report` vs `reported but wrong/unlocked`, crossed with
  `reacquire_latency` and `frames_since_update`. Preregister the expectation: `no report` >
  80 % of unlocked loss, which would move the target from identity knobs to the search gate
  and the report policy.

### Round 29 (2026-10-29) -- E088: the loss is not "not reporting", it is a refused candidate

- Ran the per-frame attribution E087 could not do: `--debug-csv` on the shipped arm for
  `dancetrack0097` / `0081` / `0035`, reproducing their reference summaries exactly
  (0.1599/0.0000/1.0000, 0.0274/0.2000, 0.0219/0.0572).
- **Preregistered expectation refuted, cleanly:** `no report` is **0 of 1482** visible frames --
  the tracker always emits a box. The unlocked loss is 91 % a *reported, unlocked, wrong* box
  (dead reckoning), 55 % of it with `frames_since_update >= 60`.
- **New fact that changes the plan:** on those missed frames `ch_app_valid` is True 100 % of the
  time (a candidate existed) and the refusal is **41 % `all_gated:anchor_ceiling`** plus a tail
  of `ambiguous: margin < 0.060`. The tracker knows it is lost (`lost_short` 31 %,
  `lost_long` 55 %) and still lets the while-tracking identity ceiling veto the candidate.
- GT-oracle contrast (7 seqs): 70 % unlocked / 30 % locked, i.e. E087's split reproduced from a
  second source; the shipped detector's fewer detections shift it to 91/9.
- Doc corrections: the README claim that `dancetrack0097` "does not report on 84 % of visible
  frames" was wrong and is replaced by the measured decomposition.

**Next: E089 -- scale the identity ceiling while the tracker is lost (preregistered).**

Hypothesis: the anchor ceiling is load-bearing only while tracking; during `lost_short` /
`lost_long` it converts recoverable reacquisitions into coast-misses. Mechanism: new inert knob
`cost.anchor_band_lost_scale: float = 1.0` multiplying the anchor band limit when
`frames_since_update > cost.lost_short_after` (`0.0` = ceiling disabled while lost, documented).
Arms to measure on the full 22-sequence held-out protocol (300 frames, subsample 2, mask none,
default detector): `scale in {1.0 (control), 1.25, 1.5, 2.0, 0.0}`.

- P1: pooled TSR **+>= 3 pp** vs `vote_off` (0.2104) at the best scale.
- P2: `id_consistency` drops by <= 0.05 on sequences where it was >= 0.90 (the ceiling protects
  those; a large drop means the knob is a switch, not a relaxer).
- P3: FAR rise <= 0.03.
- P4: the gain concentrates on sequences whose unlocked loss share >= 0.6 (E087/E088), and is
  near zero elsewhere.
- **Wiring check before interpreting anything:** on six sequences the `all_gated:anchor_ceiling`
  count must drop by >= 50 % at `scale = 0.0`. If it does not, the knob is not plumbed into the
  ceiling; stop and debug rather than read the TSR.
- Ship rule: the campaign rule (paired per-sequence dTSR bootstrap CI excluding 0) with P2/P3 as
  hard constraints. Scale 1.0 must be bit-identical to `vote_off`.

### Round 31 - E089 measured and shipped; the ceiling-while-lost family is closed

- Shipped `cost.anchor_band_lost_scale = 1.25`: 22-sequence TSR 0.2104 -> **0.2395**, FAR
  0.2472 -> **0.2089**, paired dTSR **+2.91 pp [+0.13, +6.64]**, worst sequence -2.0 pp, 13 of 22
  sequences bit-identical, partial-view guard bit-identical at 0.8545.
- The aggressive arm (ceiling off while lost) is measured and **killed**: +5.57 pp pooled, but P2 fails
  on `dancetrack0097` (idcons 1.000 -> 0.929) and `dancetrack0077` loses 21.0 pp.
- Predictions were scored honestly: P1 missed by 0.09 pp, P4 refuted. The residue that matters is P4's
  refutation - the gain does not land on the E087/E088 floor set, so the identity ceiling is not only a
  re-acquisition problem; it also fires on sequences that look healthy.
- Reference epoch: `vote_sweep.py`'s `vote_off` arm pins scale 1.0, a new `shipped` arm follows the
  config default. Every future 22-sequence run must state which epoch it used.
- Criterion 2 status: shipped **0.2395 / 0.2089**. FAR is 0.9 pp above the barrier and TSR is 31 pp
  short. The identity layer has been mined to the point where nearly every remaining arm trades one
  axis for the other.
- Next: the *reporting* half. E088 measured that 91 % of the loss is a box **coasted** onto the wrong
  person (0 % is an unreported frame), so quantify what a coasted box costs relative to no report at
  all - E090: suppress the report while `frames_since_update` is large and identity is unverified.
  Preregister before implementing.
