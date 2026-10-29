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

## Round 33 -- E090 killed; the lever family is exhausted, the deliverable is the map

- `margin_lost_0.03` (E090): pooled TSR 0.2395 -> 0.2267 (-1.28 pp, CI [-4.38, +1.14]), FAR 0.2089 ->
  0.2136, idcons 0.4790 -> 0.4698. Three hard-constraint violations: `0018` -29.0 pp, `0034` -7.7,
  `0081` -11.4; `0030` idcons 0.957 -> 0.531. The `0.01` arm was deliberately not run (the family was dead
  by rule) and the ledger says so.
- Wiring check evidence for the lesson: the same knob moved `0073` by +27.6 pp at 0.01 and +5.6 pp at 0.03,
  bit-identical on `0097`. Sequential identity decisions fork trajectories, so pooled means over a forked
  trajectory are not controlled comparisons.
- Ship state is unchanged: `anchor_band_lost_scale = 1.25`, `anchor_vote = "off"`, `distractor_margin_lost =
  0.06`. DanceTrack TSR 0.2395 / FAR 0.2089; MOT17-02 partial-view guard 0.8545.
- Next candidate (E091, pre-registered in round 34): the FAR cost of aggressive ceiling widening came
  disproportionately from three sequences (`0034` +7.6, `0065` +17.7, `0081` +27.0 pp FAR). E091 makes the
  widening *geometry-conditional*: multiply the appearance ceiling only for candidates already inside a
  strict spatial band, instead of widening the ceiling for the whole frame. Screen on 8 sequences first
  (E084b protocol) and promote to 22 only if it clears a screen bar.

## Round 34 -- E091 pre-registered: make the widening geometry-conditional

E089/E090 leave one clean pattern: every lever that widens the lost-regime identity decision buys TSR
on ceiling-dominated sequences and pays it back as false acquisitions where the motion model had already
ruled the candidate out. The 0.0 arm cost `0034` +7.6, `0065` +17.7 and `0081` +27.0 pp of FAR; E090's
margin relaxation cost `0030` +36.6 pp of FAR. While the tracker is `searching`, `cost_breakdown` is
called with `apply_gate = not searching`, so the motion gate is NOT applied -- a widened appearance
ceiling can therefore admit a box far from any plausible target position.

**Mechanism (implemented, opt-in).** `cost.anchor_band_lost_geom_scale` (default 1.0 = bit-identical) and
`cost.anchor_band_lost_geom_tau` (default 1.0 = "inside the motion gate"). While lost, the ceiling becomes
per-candidate: `limit * geom_scale` for candidates with `norm_centre <= tau`, plain shipped `limit`
otherwise. `TargetTracker._lost_geom_limit(cfg, limit, breakdown)`; the veto in `_associate` compares
against the array when it is one and against the scalar otherwise. Ablation group `ceiling_geom`
(`geom_widen_1.50`, `geom_widen_2.50`, control `geom_widen_1.00`); sweep arms added.

**Preregistered measurement.** Screen on six sequences chosen for their role, not their outcome
(`0097` ceiling 285/294, `0073` widen-works, `0077` widen-costs, `0034` + `0058` FAR casualties,
`0018` the E089 gainer that E090 broke). Promote to the 22-sequence protocol only if over the six:
(a) mean dTSR >= +2.0 pp, (b) no sequence worse than -3.0 pp, (c) pooled dFAR <= +2.0 pp. Then, at full
scale, the campaign rule with P2/P3 and "no sequence worse than 2.0 pp" as before.

**Predictions.** P1: the FAR casualties of E089 disappear (0034/0058 dFAR <= +1 pp), because the gate
removes exactly the far candidates that caused them. P2: `0097` gains, since its ceiling refusals are
candidates near a stale prediction. P3: `0077` does not lose more than 2 pp (the E089 casualty). P4: the
mechanism is worth less than +2 pp on `0018` (no ceiling refusals in the shipped run there), so the
screen's mean gain must come from 0097/0073 -- if it comes from 0018 the wiring is wrong.

### Round 34 result -- E091 SCREEN KILL, and what it rules out

| seq | shipped | geom 1.50 | dTSR | shipped FAR | geom FAR |
|---|---|---|---|---|---|
| 0097 | 0.1599 | 0.6633 | +50.34 | 0.0000 | 0.0400 |
| 0077 | 0.2700 | 0.0700 | **-20.00** | 0.2367 | 0.1000 |
| 0073 / 0034 / 0058 / 0018 | -- | identical | 0.00 | -- | identical |

Verdict: KILL by the preregistered bar (worst -20.0 pp). The geometry gate is a no-op on 0034 and 0058,
the very sequences it was designed to protect, so **P1 is refuted: the E089 false acquisitions are not
far-candidate admissions, they are near-but-wrong candidates.** On 0077 the gate actively creates a
20 pp loss (admitting the switched neighbour, who is near the stale prediction) that the shipped 1.25 arm
does not have. What survives is the 0097 result: a ceiling-bound sequence recovers +50.3 pp, which says
the ceiling lever is real but that spatial gating is the wrong qualifier for it.

Consequence for the campaign: the "widen more, but only here" family is closed. The remaining ways to buy
TSR under the same FAR are (a) a *temporal* qualifier (a re-acquisition must persist for several frames
before it can commit -- geometry cannot order candidates but time can), or (b) accepting that 1.25 is the
frontier of this mechanism and moving the FAR head-room question elsewhere.

## Round 35 -- E092 pre-registered: time, not space, is the lost-regime qualifier

E091's kill produced the sharpest statement in the campaign: while the target is lost, the dangerous
candidate is **not the distant one, it is the near one**. Widening for near candidates recovered +50.3 pp
on a ceiling-bound sequence and cost -20.0 pp on a neighbour-switch sequence, and nothing spatial
separated the two. What does separate them is persistence: a wrong latch is typically a one-frame event,
while a true re-appearance stays put from frame to frame.

**Mechanism (implemented, opt-in).** `state.reacquire_confirm_frames` (default 1 = shipped) plus
`state.reacquire_confirm_iou` (0.3). While the previous state is blind, an accepted candidate is recorded
as a *provisional* candidate instead of taking the identity: the frame is booked as a miss, the prediction
keeps being reported, and only the same candidate (IoU >= 0.3 with the previous provisional box) offered on
N consecutive frames may commit. `TargetTracker._provisional_reacquire_ok`, hooked in `step()` before
`_on_match`; the deferral is visible in the debug CSV as `deferred:reacquire_pending`. Ablation group
`reacquire` (`reacq_confirm_2`, `reacq_confirm_3`, control `reacq_confirm_1`); sweep arms added (35 arms,
31 ordered).

**Preregistered measurement.** Screen on five sequences chosen for role: `0034` (E089 FAR casualty),
`0065` (worst FAR casualty +17.7 pp and idcons -46.1 pp -- the clearest wrong-person latch), `0077`
(neighbour switch), `0097` (ceiling-bound, must keep its +50 pp), `0018` (stability control). Promote to
22 sequences only if over the five (a) pooled FAR falls by >= 1.0 pp, (b) no sequence loses more than
3.0 pp of TSR, (c) pooled TSR is not worse than -2.0 pp. Then the campaign rule with P2/P3 as before.

**Predictions.** P1: pooled FAR falls, and most on 0065 (the latch case). P2: TSR on 0097 does not fall
more than 2 pp (a persistent target satisfies the hold cheaply). P3: 0018 unchanged within 1 pp. P4: the
cost appears as *re-acquisition latency* on sequences where the target really does reappear briefly --
if no sequence shows a TSR loss at all, the counter is not actually gating anything and the wiring is wrong.

## Round 36 -- E092 is killed; the identity line is a Pareto surface

E092 (temporal re-acquisition hold) was screened and killed: pooled FAR -8.11 pp (the best FAR result the
campaign has produced) at pooled TSR -8.87 pp, with 0034 -23.39 and 0018 -28.17 pp. Bar clause (b) and (c)
both fired. The interesting part is the *shape* of the result: E089 sold -3.83 pp of FAR for +2.91 pp of
TSR, E092 sells +8.11 pp of TSR for -8.11 pp of FAR. Two independently motivated mechanisms, opposite
directions, near-identical exchange rate. Together with E082 (contrast gate: +6.83 TSR / -6.42 FAR),
E090 (+... / FAR +36.6 on one sequence) and E091 (a spatial gate that recovered +50.3 pp on a ceiling-bound
sequence and cost -20.0 pp on the neighbour-switching one), the record is consistent: the identity layer
sits on a Pareto surface and no single dial moves the operating point toward the corner.

Consequence for the remaining work: stop looking for identity mechanisms that improve both axes. The
honest options left are (i) a *sequence-conditional* selector -- decide per sequence which operating point
to use, which needs a signal available at click time; (ii) the temporal commit rule fixed properly (N hits
in M frames, IoU against the prediction) and re-screened, which is a mechanism repair rather than a new
idea; (iii) documentation, since criteria 1 and 2 as written are not reachable from this architecture (see
the criterion-2 analysis: TSR >= 0.55 is a Detection-dominated quantity on DanceTrack held-out, where the
shipped detector's ceiling on these sequences is what E055/E047 measured).

Next action: record E092 in the ledger (done), then (ii) as a cheap repair screen -- it is the only arm
with a measured FAR win on the table.

## Round 37 -- E092b pre-registered: the same temporal qualifier, made gap-tolerant

E092's screen produced the campaign's best FAR number (-8.11 pp pooled) and killed itself on TSR
(-8.87 pp, 0034 -23.39, 0018 -28.17). The ledger entry names the mechanism defect: the commit rule was a
strict consecutive-frame streak with IoU >= 0.3 against the *previous provisional box*, so one jittered
detection or one alternating distractor resets the counter and a legitimate re-acquisition of a fast or
partially occluded target can be deferred indefinitely.

**Mechanism (implemented, opt-in).** `state.reacquire_confirm_window` (default 0). With `M > 0` the rule
becomes "N of the last M candidates agree with each other" (agreement = IoU >= `reacquire_confirm_iou`
against another candidate in the window). N is still `reacquire_confirm_frames`. Window 0 keeps E092's
exact rule, so the shipped path and the E092 record remain bit-comparable. Ablation group `reacquire`
gains `reacq_window_2of4`, `reacq_window_3of6`, control `reacq_window_2of2` (window 0 -- must equal
`reacq_confirm_2` exactly).

**Preregistered measurement.** Same five sequences as E092 (0034, 0065, 0077, 0097, 0018) so the numbers
are directly comparable, and the same screen bar: (a) pooled FAR falls >= 1.0 pp, (b) no sequence loses
more than 3.0 pp of TSR, (c) pooled TSR >= -2.0 pp. Promote to 22 sequences only if all three hold.

**Predictions.** P1: 0034 and 0018 recover their TSR (E092's two casualties) while keeping >= half of
E092's FAR win -- if neither recovers, the strictness of the streak was not the defect and the temporal
qualifier is dead. P2: 0065 keeps its FAR win (0.2033 -> 0.0467 in E092), since that win came from
refusing a persisting distractor, which gap tolerance does not weaken much. P3: 0097 stays bit-identical.
P4: `reacq_window_2of2` reproduces `reacq_confirm_2` exactly on 0018 (0.2857 / 0.0476).

## Round 38 -- E092b measured: the temporal qualifier is dead, and the lesson is structural

E092b (docs/NEXT_ROUND.md round 37) replaced E092's strict consecutive-frame streak with "N of the
last M candidates agree". Result: **KILL on the same three clauses, with all four predictions
refuted**: 0034 0.3911 -> 0.1653 and 0018 0.5675 -> 0.2698, i.e. *worse* than the strict version on
the two sequences the repair was built for, while keeping the FAR win (-8.73 pp pooled). The defect
was therefore never the strictness of the streak.

The campaign now has three measured points on one line: E082 (`contrast_0.10`: TSR +6.83 / FAR
-6.42), E089 (`anchor_band_lost_scale=1.25`: TSR +2.91 / FAR -3.83) and E092/E092b (FAR -5.8..-8.7 /
TSR -8.2..-8.9). Every identity-layer lever moves TSR and FAR in opposite directions with an
exchange rate near 1. The reason is architectural: one candidate set, one commit per frame, so a
miss and a false report are the same decision. **Consequence: stop mining the identity layer.** The
open question is whether two independently-selected Pareto points *stack* (E093) or saturate; and
the longer-term direction is a second decision channel (E095), not another knob.

## Round 39 -- E093 pre-registered: do two independently-selected Pareto points stack?

The three surviving Pareto points were selected against *different* references and at different times
in the campaign, so one thing is not known: whether they stack. E082's `contrast_0.10` was measured
before E089 existed, on the pre-E089 epoch (`anchor_band_lost_scale = 1.0`), and its paired CI missed
the ship bar by 0.46 pp; E089's widening then shipped and moved the reference by +2.91 pp TSR. The
combination -- contrast vote **and** widened lost-regime ceiling -- has never been measured.

**Arm (implemented).** `stack_contrast_1.25` = `cost.anchor_vote="contrast"`,
`cost.anchor_vote_contrast=0.10`, `cost.anchor_band_lost_scale=1.25`. Reference = the shipped default
(`ceil_lost_1.25`, 22 records already on disk). Everything else is the campaign protocol: 22 held-out
sequences, 300 frames, subsample 2, mask none.

**Why it is worth one run.** The two mechanisms act at different points of the same decision: the
contrast vote rescues candidates the majority vetoed (raising recall), the widened ceiling stops the
lost-regime gate from vetoing them in the first place (also raising recall). They should be
*substitutes* rather than complements if both just relax the same veto -- in which case the stack
saturates near the better of the two and the exchange rate stays 1:1. But they are also gated by
different evidence (bed agreement versus ceiling distance), and E082's gains concentrated on
0018/0034/0077/0073/0094 while E089's concentrated on 0018/0034/0081/0077 -- overlapping but not
identical sets, which is exactly the shape that can add.

**Ship rule (unchanged, pre-committed).** Paired per-sequence dTSR bootstrap CI must exclude 0;
P2 (idcons drop <= 0.05 where reference >= 0.90) and P3 (FAR rise <= 0.03 pooled) are hard
constraints; no sequence may lose more than 2.0 pp of TSR. Predictions: P1 pooled dTSR in
[+2, +8] pp versus shipped; P2 FAR no worse than shipped; **P3 if the stack lands within 1 pp of
E082's own +6.83 the mechanisms are substitutes and the campaign's identity budget is spent (record
and stop); if it lands near +9.7 (+6.83 + 2.91) they add and the campaign has a real combination to
defend.** P4 no sequence worse than -2 pp.

## Round 40 -- E093 measured: substitutes, not complements. The identity layer is closed.

`stack_contrast_1.25` (E082 contrast vote + shipped E089 ceiling) on the 22-sequence protocol:

| point | TSR | FAR |
|---|---|---|
| `vote_off` (pre-E089) | 0.2104 | 0.2472 |
| shipped (`ceil_lost_1.25`) | 0.2395 | 0.2089 |
| **stack** | **0.2688** | **0.2037** |

stack vs shipped: dTSR **+2.93 pp [-2.24, +8.94]**, dFAR -0.52 pp, idcons +0.15 pp, blind&present
-6.62 pp. **KILL by rule** (CI includes 0; P4 fails on four sequences: 0010 -23.53, 0081 -11.42,
0097 -3.40, 0058 -2.36).

**P3 resolved: substitutes, negatively.** Against the same reference E082's contrast arm alone scored
+6.83 pp; the stack scores +5.83 pp -- one point lower. Adding the widened ceiling to the contrast vote
subtracts. P1 MET, P2 MET pooled (but 0058 FAR 0.0933 -> 0.9000 is a per-sequence explosion the pooled
number hides), P4 REFUTED.

**Decision: the identity layer is closed.** Four measured levers (E082, E089, E092/E092b, E093) lie on
one Pareto surface with a ~1:1 TSR/FAR exchange rate, and the two recall-side levers are substitutes.
The frontier point to quote is 0.2688 / 0.2037 (not shippable under the campaign rule; shipping it
would contradict docs/METHOD.md, which is why the rule exists). Criterion 2's TSR half is 0.28 away --
an order of magnitude more than any identity knob moves -- so the remaining headroom is elsewhere:

1. **E095 (second decision channel)**: a per-frame verification pass that can *refute* a committed box
   with evidence the identity layer does not use (follow-trajectory consistency, background-relative
   motion, or a second detector scale). The 1:1 exchange rate is an artefact of one decision per
   frame; a refutation channel breaks that coupling. Highest expected value.
2. **Criterion 1 (MOT17 HOTA 0.55, best 0.5050)**: detector-side work; the all-person layer is
   detection-limited (DetA flat 0.42-0.44 across six backends while AssA spans 0.45-0.53).
3. **Criterion 3 audit**: the README audit records the endpoint 0.8545 as real but the +10.18 pp
   attribution as not yet defensible (+7.27 pp verified). Re-measure with the shipped default before
   claiming the criterion.

Handoff state: shipped defaults unchanged (`anchor_band_lost_scale = 1.25`, `anchor_vote = "off"`);
22-sequence Sweep-report backups exist for E082/E089/E092b/E093; the current `results/vote_sweep/`
report *is* the E093 run.

## Round 41 -- E094 release verification in flight; E095 pre-registered as the next phase

**E094 (running).** The ship decision for `anchor_band_lost_scale = 1.25` was made on the
`ceil_lost_1.25` arm, which sets the value as an explicit override. What a user actually gets is the
*config default*, reached through the `shipped` arm (overrides: only `anchor_vote = "off"`). If those
two disagree anywhere, either the default did not land or another config field drifted -- both release
blockers. E094 therefore runs `shipped` across all 22 held-out sequences and compares it, per sequence
and per metric, against the `ceil_lost_1.25` records that justified the ship, and re-derives the pooled
numbers against the frozen pre-E089 `vote_off` epoch (`_x/e094_verify.py`). Expected: bit-identical on
22/22 (the constraints are the same single integer), pooled 0.2395 / 0.2089.

**E095 (pre-registered, `docs/E095_PLAN.md`).** The identity layer is closed (E093): four levers on one
Pareto surface with a ~1:1 TSR/FAR exchange rate, because one candidate set and one commit per frame
make a miss and a false report the same event. E095 attacks that coupling directly with a **bounded-lag
refutation channel**: the per-frame report goes into a ring buffer of depth L, and an entry is revised
just before delivery by a short-window global assignment over the buffered candidates (trajectory
smoothness + cross-frame identity competition). Only re-attribution is permitted -- never deletion,
which is the metric gaming the `blind & present` share exists to expose, and which E095's P3 guard
forbids explicitly.

Pre-registered predictions for E095: **P1 both metrics improve or the coupling is deeper than assumed**
(pooled TSR +>= 1 pp *and* FAR ->= 1 pp); P2 id_consistency rises; P3 b&p does not rise by more than
1.0 pp (anti-gaming guard); P4 gains concentrate on the oscillating sequences (0010, 0077, 0081). Kill
criteria: TSR falls, or b&p rises while FAR falls, or the `lag_refute_L0` control is not bit-identical.
Cost: no new model; O(L^2) on already-computed candidates, ~one E093 arm (~100 min) for the full
22-sequence measurement.

## Round 42 -- E096 pre-registered: the dose axis of the shipped lever was never measured

E089 shipped `anchor_band_lost_scale = 1.25` after a three-arm sweep (`vote_off` / `1.25` / `off`); the
arms `1.50` and `2.00` were defined in the harness but **never run** because the sweep was scope-cut to
the arms that decide the hypothesis. That leaves the shipped lever's dose only bracketed: 1.00 = 0 pp,
1.25 = +2.91 pp, off = +5.57 pp (killed for `dancetrack0077` -21.0 pp and a P2 violation on `0097`).
The response is monotone in TSR up to the point where it breaks a sequence, and 1.50 is exactly the
unmeasured point between "shipped" and "broken".

**Measurement.** `ceil_lost_1.50` (already in `scripts/vote_sweep.py`) against `vote_off` and `shipped`
on the 22-sequence protocol. E094 has just produced a full `shipped` record set, so the delivered
default is the direct reference; `vote_off` stays in the arm list only because the harness pins target
ids from it.

**Predictions.** P1: pooled dTSR versus shipped in [+1.0, +4.0] pp (the dose curve from 1.25 to off
gains 2.66 pp of TSR over the interval; 1.50 sits one third of the way along the scale axis, so
+1 to +4 pp is the honest bracket). P2: pooled FAR not worse than shipped by more than 0.5 pp.
P3: **no sequence loses more than 2.0 pp** -- this is the clause that E089's 1.25 arm only just met
(`dancetrack0063` -2.0 pp), and it is the clause the off arm failed catastrophically (`0077` -21.0 pp).
P4: if the P3 failure appears at 1.50, then the whole dose axis is unsafe beyond 1.25 and the shipped
value stands; if 1.50 passes cleanly, the campaign has a defensible step.

**Ship rule (unchanged).** To replace the shipped default, the paired per-sequence dTSR bootstrap CI
versus `shipped` must exclude 0, P2/P3 above must hold, and criterion 3's partial-view guard must be
re-run and stay at 0.8545. Otherwise 1.25 stays and the dose axis closes with 1.25 as the measured
optimum.


**E094 measured (closing this round): PASS.** `shipped` across 22 sequences is **bit-identical on
22/22** to the `ceil_lost_1.25` records (TSR 0.2395 / FAR 0.2089 / idcons 0.4790; pooled delta vs
`vote_off` +2.91 pp, CI [+0.22, +6.35], FAR -3.83 pp; reference no-op 22/22). The default is what was
measured. `docs/METHOD.md` now carries this as a release requirement: *every future default change must
be verified through the `shipped` arm, not only through the arm that motivated it.*


## Round 43 -- E095 and criterion 1 are the two remaining lines

With E094 closing release hygiene, the campaign's open work is exactly two things:

1. **Criterion 2** (`DanceTrack TSR >= 0.55, FAR < 0.20`; shipped 0.2395 / 0.2089). The identity layer
   is measurably closed (E093: four levers, one Pareto surface, ~1:1 exchange, substitutes not
   complements). The only mechanism left with a *reason* to be one-sided is E095's bounded-lag
   refutation channel (`docs/E095_PLAN.md`, pre-registered: P1 both metrics improve, P3 b&p must not
   rise -- the anti-gaming guard). This is the recommended next experiment: no new model, one arm's
   worth of GPU time.
2. **Criterion 1** (`MOT17 HOTA >= 0.55`, best 0.5050). This is a detector-side problem: the
   all-person layer is DetA-limited (0.44) and E001/E002/E003/E004/E005/E007/E010/E011/E012 closed the
   tracker-side routes. The remaining lever allowed by m04393 is a *different third-party pretrained
   detector of comparable size*, not fine-tuning. Any future attempt should start by re-reading
   `docs/research/R2_detectors.md` and `R4_all_person.md`, and must be measured on the same
   `tracktrack` parser + HOTA harness that produced 0.4980/0.5050.


**E096 measured (closing this round): KILL, 1.25 stands.** `ceil_lost_1.50` = TSR 0.2481 / FAR **0.1867**
(the first pooled FAR below 0.20), versus shipped 0.2395 / 0.2089: dTSR +0.86 pp [-1.21, +4.25], dFAR
-2.22 pp [-6.55, +0.35], **18/22 sequences bit-identical**, movers `0073` +29.60, `0034` -0.81,
`0097` -4.42 (idcons 1.0000 -> 0.8293), `0077` -6.67. P1 missed by 0.14 pp, P3 fails twice -> P4's
clause fires: the dose axis is unsafe beyond 1.25. Structural finding: the ceiling is a **bifurcation
switch**, not a dial -- 18 of 22 sequences do not notice a 20 % parameter change and the four that do
re-fork their trajectory, which is the same non-monotone signature E090 produced on one knob.


## Round 44 -- state of the campaign, and the only two doors left

**Shipped, verified**: `anchor_band_lost_scale = 1.25`, `anchor_vote = "off"`, DanceTrack held-out
TSR **0.2395** / FAR **0.2089** / idcons 0.4790, bit-identical to the arm that justified it (E094,
22/22) and unchanged by four later killed experiments (E090, E091, E092/E092b, E096). Partial-view
criterion met: 0.8545, guard re-verified bit-identical under the shipped levers (E089 guard).

**Closed lines (do not re-open without new evidence).** Identity-layer knobs: four mechanisms measured,
one Pareto surface, ~1:1 TSR/FAR exchange, recall-side levers are substitutes (E082/E089/E092/E092b/
E093/E096). All-person tracker-side MOT17 routes: E001-E012. Detector-side: E002 established the
bottleneck is occlusion, not small objects.

**Two doors left.**

1. **E095 -- the second decision channel** (`docs/E095_PLAN.md`, pre-registered): bounded-lag
   refutation of committed identities. It is the only remaining mechanism with a *reason* to be
   one-sided, because it uses evidence that arrives after the decision instead of re-weighting the
   evidence at it. P1 demands both metrics improve; P3 forbids buying FAR by deleting reports.
2. **Criterion 1 -- a different third-party detector** at comparable size (m04393 allows swapping,
   forbids fine-tuning). Best measured HOTA 0.5050 with `yolo26m@1280`; the target 0.55 needs a
   detector whose recall on *occluded* persons is materially better, not a tracker change.

**Practical next step if data is ever available**: E085 (fresh cohort) remains blocked -- the host
cannot reach HuggingFace (`status=000`), and `data/raw/dancetrack/` contains only the val split, so
every number in this campaign is measured on the same 22 held-out sequences. That is a real external
validity limit and should be stated as such in any write-up.


## Round 43 -- E095 measured: the refutation pass is inert (KILL)

Mechanism implemented (`src/tgtrack/refute.py`, `StateCfg.decision_lag`, tracker candidate log, 8 tests,
sweep arms). Control `lag_refute_L0` = TSR 0.5675 / FAR 0.0873 on 0018, exactly the shipped record.
Screen over 0010/0073/0077/0081/0097: **5/5 TSR bit-identical**, FAR +0.33 pp on one sequence (one
frame), b&p unchanged. P1/P2/P4 refuted. Reason: the buffer holds no rival candidate because the ceiling
vetoed the true target (E088: 41 % `all_gated:anchor_ceiling`), and the identity failures are sustained
wrong trajectories (55 % at `frames_since_update >= 60`), not flicker.


## Round 44 addendum -- E095b, the version of the second channel that can work

E095's post-pass was inert for a specific, measured reason: it re-attributes over the *post-veto*
candidate set, and while coasting that set does not contain the true target. E095b keeps the same window
machinery but makes the buffer the **pre-veto** set, using the per-bed distances `anchor_bed_log` already
writes (E084): a shadow candidate enters the buffer with its `d1/d2/d3` and is admitted only when the
window-level assignment prefers it to the reported person *outright* (not by a margin -- a margin is the
E082/E096 mistake).

Pre-registered constraints for E095b (do not start it without these):

1. Control `lag_refute_L0` must stay bit-identical, and the shadow-only control (`shadow_log`,
   no re-attribution) must equal the shipped default on 22/22 -- the buffer must not perturb the online
   path.
2. P1 needs **both** metrics to improve; a 1:1 trade is E093's result, not news.
3. P3 (b&p must not rise) stays a hard constraint: admitting shadows may never delete or blind a report.
4. The window evidence must be the *pre-veto* set only; reading GT or the vote's own rescue flag would
   make the pass circular.
5. Cost: the buffer is per-frame candidate lists, so a full 22-sequence pass is one sweep arm plus
   no extra tracker work -- budget one screen (5 sequences, ~25 min) before committing.

