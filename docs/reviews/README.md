# Multi-direction review protocol (round gate)

The user asked that every research round be followed by a multi-direction evaluation, repeated
until the evaluations are satisfactory. This file defines that gate and records each round.

## The four review directions

| direction | reviewer artefact | question it must answer |
|---|---|---|
| metrics integrity | `R_metrics.md` | Is every quoted number backed by an artifact, from the same epoch/config, and free of retracted claims? |
| exploration direction | `R_direction.md` | Which evidence axes are exhausted, which are open, what are the 5 highest-EV next experiments, what should be stopped? |
| methodology | `R_method.md` | Is the protocol statistically defensible (tuning/test separation, epoch drift, multiple comparisons, instrumentation guards, reproducibility)? |
| performance | `R_perf.md` | Where does wall time actually go, which cheap speedups are decision-preserving, which shipped knob is least cost-effective, what is the safe concurrency rule? |

Reviewers are run as independent subagents: they read `docs/EXPERIMENTS.md`, `docs/METHOD.md`,
`README.md`, `src/tgtrack/` and `results/`, and they never modify the implementation. Each review
must end in numbered, concrete actions (file to change, test to add, re-run to do), ordered by
severity — an evaluation that produces no actionable item is treated as a failed review.

## Round gate

A round is only "satisfactory" when all four hold:

1. **Metrics** — zero WRONG/UNSUPPORTED claims; every STALE claim either re-measured or explicitly
   labelled with its epoch.
2. **Direction** — the next experiment is chosen from the reviewer's ranked list, and the
   "stop doing this" item is either dropped or defended with a measurement.
3. **Method** — no new result is decided on data that tuned a shipped knob; any cross-epoch delta is
   marked, and instrumentation invariants (the `gt_at()` class of bug) are covered by tests.
4. **Performance** — the shipped configuration's cost is accounted for, and any decision-preserving
   speedup larger than 10 % is either adopted with parity evidence or explicitly deferred.

## Rounds

### Round A (after E063) — reviews dispatched
Four reviewers + one implementer were dispatched in parallel on 2026-10-25:

- metrics integrity reviewer
- exploration-direction reviewer
- methodology reviewer
- performance reviewer
- E064 implementer (two-reference audit: switch only when the click crop *and* the recent
  confident crop both prefer the rival)

Status: awaiting findings. Actions will be appended below with their disposition
(adopted / deferred / rejected-with-reason) so the gate is auditable rather than asserted.

#### Findings received so far (verified by the Lead before acceptance)

**F-01 (adopted, severity: high) — the criterion-2 "regression" was a silent target re-pick.**
The direction reviewer noticed that `results/e041_dancetrack0005/summary.json` used
`meta.target_ids = {dancetrack0005: 2}` while `results/e063b_dt0005/summary.json` used
`{dancetrack0005: 0}`; every other sequence keeps its id (0004 -> 3, 0007 -> 1). The Lead verified
this directly from both `summary.json` files — the reviewer is right. Consequences:

- The "0.512 / 0.280 -> 0.4138 / 0.3833 regression" is **not a regression**: the two numbers are
  measured on different people. Recomputing the pooled baseline on the *consistent* ids
  (0004 0.415/0.040, 0005 0.695/0.205, 0007 0.4264/0.595) returns TSR **0.512**, FAR **0.280**,
  i.e. exactly the older figure. The older number was correct; the later runs changed the subject.
- **E063's guard verdict is therefore VALID ONLY FOR 0004 AND 0007.** `results/e063b_dt0005`
  (interval 5 -> 0.295/0.705, interval 3 -> 0.200/0.790) was measured on target id 0, not id 2, so
  the "trades one sequence for another, keep it off" conclusion is **provisional and must be
  re-run with `--target-id 2`** before it can stand. Recorded as E063-REV (pending).
- Rule adopted for the rest of the campaign: every target-layer run passes `--target-id`
  explicitly (0005 -> 2, 0004 -> 3, 0007 -> 1) and every table quotes `meta.target_ids` and
  `target_decisions.reason`, so a re-pick can never masquerade as a mechanism effect.

**F-02 (adopted, severity: high) — a conjunctive two-reference audit would have destroyed E063's win.**
At the 0007 swap the rival is closer to the CLICK anchor (`audit_best 0.3405 < audit_cur 0.4638`)
while the recent confident crop still prefers the true target, so "both references must prefer the
rival" blocks exactly the good switch. E064 is therefore built as a 2x2 over evidence modes
(`click` | `recent` | `and` | `recent_not_click`), not as one conjunctive arm. The already-running
E064 subagent was given the original spec and could not be steered (subagents are not messageable),
so the 2x2 is a follow-up round item.

**F-03 (accepted as a direction, not yet implemented) — the free evidence is the all-person layer's
tracklet identity.** From `results/e059_diag7/runs/dancetrack0007__baseline__none.csv` the four swap
onsets are spatially smooth (centre jumps 0.19 / 0.143 / 0.096 / 0.104 body heights) and the wrong
stretches are long, stable runs (GT id 3: 41 of 44 frames wrong; id 6: 54 of 54). So no motion
plausibility test can see the failure — "is this the same person" is the only question that
separates those runs, and the all-person MOT layer already answers it with a tracklet id that
`scripts/run_target.py` never consumes. A tracklet-id audit is the cheapest untried discriminator
(no encoder cost) and is queued as E065.

#### Dispositions

_(to be filled from the four reviews)_

### Round A disposition (R_metrics — 24 claims audited: OK 12 | STALE 4 | WRONG 6 | UNSUPPORTED 2)

All six WRONG and four STALE items were corrected in `README.md` / `docs/EXPERIMENTS.md` in the same
turn the audit landed; the corrections are listed in a new **"Corrections after the first independent
audit (round A)"** section of `README.md` so the record shows what was believed and why it failed.

**The one that changes a verdict: criterion 3 is OPEN, not met.** The controlled same-run pair
(`results/e031b_mot17`, baseline vs shipped weights, identical command) is 0.7818 -> 0.8545 =
**+7.27 pp**, not +10.18 pp; the +10.18 compared runs from different epochs. Criterion 3 therefore
stands at +7.27 pp of a required +10 pp.

Also corrected: criterion 2's baseline is target-dependent (0.512/0.280 is `--target-id 2`; the
auto-pick yields 0.4138/0.3833) -> a task was dispatched to freeze `--target-id 3,2,1`; the lone-bound
"3.6 pp" cost is retracted (no armed variant exists in `results/target_obs19_shipped`); the sweep is
1/4 not 0/4; the 448 pair is 0.655->0.780 / 0.921->0.985; the VisDrone lower cell is 0.9236; E020's
"74/111" ceiling evidence carries a RETRACTION banner in the ledger (E062: wrong GT identity; the true
target was actually detected in 108 of the 113 wrong frames).

Ledger scoreboard after round A: criteria MET 2 of 4 (criterion 4 ledger+method), criterion 3 at
+7.27/+10 pp, criterion 2 at 0.512/0.280 on target-2 basis, criterion 1 at 0.5050 vs 0.55.

### Round A disposition (R_perf)

Stage budget (dancetrack0007, 40 frames, subsample 2, jobs 1): encoder **79.0 % of 286.5 ms/frame**
(ORT forward 182.9 + CPU 448^2 crop prep 39.6), detector 11.9 %, JPEG decode 4.4 %, association
python 3.4 %, motion/IMM 1.4 %, memory writes 0.3 %; 2.0 forward calls/frame, 19.3 crops/frame.

Accepted for implementation (decision-preserving, parity-gated): (1) **drop the duplicate encode** in
`_features_for` (-94.0 ms/frame, -33 % wall, bit-identical `part_feats`, 40/40 frames); (2) **call
`ort.InferenceSession.run` directly** instead of `AutoBackend` (-14 % wall, `max_abs_diff = 0.0`).
Combined: **286.5 -> ~150 ms/frame (-48 %)**. Delegated as E067 with a parity table as the acceptance
gate.

Recorded as dead complexity (not shipped): `cost.part_matching` shows dTSR = 0.0000 in **43/43**
paired rows across 7 masks while costing ~0 ms; the real cost is the 4-part `target_parts` encode
(43.8 ms/frame for mean dTSR +0.0414, 33/48 rows zero).

Concurrency rule (measured, not assumed): `--jobs 2` cuts grid wall only 7 % (startup overlap) while
per-run latency goes 443.7 -> 641.9 ms/frame and VRAM 4.2 -> 6.5 GB. **One GPU worker per card**; the
269,746,176 B `BFCArena` OOM is a second worker's arena on an 8 GB card.

Two hazards fixed under E067: `_ensure_features` swallows encoder exceptions, so a broken encoder
silently degrades to geometry-only tracking (it made the profiler read "encoder = 2 %"); and
`fp16=true` is a no-op against the shipped fp32 ONNX.

### Round A disposition (R_heldout) — S1 MANDATE EXECUTED

22 held-out DanceTrack sequences (dev split frozen at 0004/0005/0007), shipped defaults, no ablation,
all observable, 0 excluded, 6181 frames, wall 32.4 min:

| quantity | dev (n=3) | held-out (n=22) | 95% cluster CI |
|---|---|---|---|
| TSR | 0.512 | **0.2102** | [0.1154, 0.3253] |
| FAR | 0.280 | 0.2364 | [0.1419, 0.3391] |

Dev TSR outside the held-out CI -> **the dev headline is not reproduced**; FAR indistinguishable ->
the held-out failure mode is target LOSS, not wrong-identity lock. Recorded as E068; README now carries
a dedicated section and quotes criterion 2 as **0.21 [0.12, 0.33] over 22 sequences**. Enforcement
tool: `scripts/heldout_report.py` (new file, refuses any sweep flag, asserts
`meta.variants == ["baseline"]`, prints the frozen dev list).
