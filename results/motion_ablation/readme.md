# Motion-strategy ablation (`scripts/ablate_motion.py`)

Drives the real `tgtrack.target_tracker.TargetTracker` as a black box over deterministic synthetic sequences and scores every run with `tgtrack.evaluate.evaluate_target` (no hand-rolled metrics).  Appearance is a deterministic stub embedder, so **motion is the only variable**; the metric numbers still come from the same evaluator the rest of the project uses.

## Run it

```
python scripts/ablate_motion.py                  # full sweep + distractor + unid sweeps + this readme
python scripts/ablate_motion.py --quick          # smoke run
python scripts/ablate_motion.py --no-distractor --no-unid   # main sweep only
python scripts/ablate_motion.py --distractor     # hard-overlap similarity x margin grid only
python scripts/ablate_motion.py --unid           # unidentifiable-target sweep only
python scripts/ablate_motion.py --trace occlusion_distant_gap15 --no-write
python -m pytest tests/test_ablate_motion.py
```

## Files

| file | contents |
| --- | --- |
| `raw.json` | every run: config, scenario, full metric dict, state histogram |
| `table.md` | leaderboard + per-family tables + findings |
| `table.csv` | one row per (scenario, config) run |
| `distractor_sweep.md` / `.csv` | the hard-overlap similarity x margin sweep |
| `unid_sweep.md` / `.csv` | the unidentifiable-target x arming-rule sweep |
| `readme.md` | this file |

## This run

- scenarios: 35 | configs: 10 | runs: 350 | errors: 0 | frames: 29150
- metrics: `tgtrack.evaluate.evaluate_target`; appearance: deterministic stub embedder (motion is the only variable)

## Scenario families

| family | what it is |
| --- | --- |
| `distractor` | An equal-size person crosses the target's path.  `lookalike` gives both the same embedding (appearance cannot help); `different` is the control. |
| `distractor_overlap` | **Hard-overlap distractor.**  A bystander rides *on top of* the target (box offset (6,4) px, IoU 0.876) for `n_overlap` frames while the target keeps a clean template, so the geometry/size/motion costs are near-identical and the whole decision falls on appearance.  The bystander's descriptor is correlated with the target's by a swept cosine, while the target's own crop carries a fixed pose offset (`pose`, default 0.97): once the bystander is *more* similar to the template than the target's current crop, the best candidate is the wrong person and only `CostCfg.distractor_margin` can refuse.  After the window the bystander peels away, so a tracker that latched onto it loses the target and racks up false acquisitions. |
| `fast_linear` | Constant-velocity walk at `step` px/frame with a brief detector drop (frames 20-27): the target stays visible in GT but is not detected. |
| `linear_blackout` | Control for `nonlinear_blackout`: the same blind window on a straight path, where extrapolation is exact and every motion model should be equal. |
| `nonlinear` | Circles and a figure-eight at several radii/speeds -- the trajectories on which a constant-velocity filter is provably mis-specified. |
| `nonlinear_blackout` | **The decisive experiment.**  The target follows a curving path while the detector is blind for `gap` frames.  The target is still in GT (so those frames count as visible), but there is no observation: every reported box during the gap is *pure motion prediction*, and success needs IoU>=0.5 with the true box.  A constant-velocity model extrapolates the tangent and leaves the arc; a turn/acceleration model must stay on it.  This is the only family where prediction quality can change the target-centric outcome. |
| `occlusion` | The target leaves the scene for `gap` frames and comes back, either on its own trajectory (`inplace`) or at a distant location (`distant`). |
| `overlap_sameid` | Control for `distractor_overlap`: one person, detected **twice** on the same box with the same descriptor.  Nothing can separate the two candidates, so any refusal here is pure cost. |
| `overlap_solo` | Second control for `distractor_overlap`: the same trajectory and the same window with no second person at all, i.e. the geometry by itself is harmless. |
| `unidentifiable_solo` | Control for `unidentifiable_target`: the same halt, gap and blind window with no bystander at all, i.e. the cost of the target's own unidentifiability, with no wrong identity available to latch onto. |
| `unidentifiable_target` | **The requirement-(4) failure mode.**  The target walks, then HALTS, then stops being *identifiable*: for `n_blind` frames it is still detected at its true box but its descriptor is unusable (`mode='missing'`: no features at all, i.e. `appearance_valid=False`; `mode='orthogonal'`: a valid but orthogonal vector).  Meanwhile an unoccluded bystander with a clean, distinct descriptor rides the target's **stale prediction** -- the constant-velocity coast the halted target has fallen behind -- so geometry points at the bystander and appearance is the only cue that can refuse.  The bystander's similarity to the target's template is swept, and the target becomes identifiable again in the last `n_post` frames: the question is whether the tracker hands the track to the bystander (wrong-identity frames, no recovery) or refuses and re-acquires the true target. |

## Config axes swept

| config | model | nsa_adaptive | rollback_depth | imm_models |
| --- | --- | --- | --- | --- |
| `kf` | kf | True | 5 | cv+ca+ct |
| `kf-rb0` | kf | True | 0 | cv+ca+ct |
| `kf-rb3` | kf | True | 3 | cv+ca+ct |
| `nsa_kf` | nsa_kf | True | 5 | cv+ca+ct |
| `nsa_kf-static` | nsa_kf | False | 5 | cv+ca+ct |
| `imm` | imm | True | 5 | cv+ca+ct |
| `imm-rb0` | imm | True | 0 | cv+ca+ct |
| `imm-rb3` | imm | True | 3 | cv+ca+ct |
| `imm-cv` | imm | True | 5 | cv |
| `imm-cvca` | imm | True | 5 | cv+ca |

## Headline (pooled over the families)

| config | model | nsa | rb | imm_models | success | IoU | loss | reacq rate | reacq lat | FA rate | id const |
|---|---|---|---|---|---|---|---|---|---|---|---|
| imm-rb0 | imm | on | 0 | cv+ca+ct | 92.2% | 0.822 | 7.8% | 100.0% | 0.0 | 4.9% | 94.3% |
| kf-rb0 | kf | on | 0 | cv+ca+ct | 91.1% | 0.815 | 8.9% | 100.0% | 0.0 | 4.9% | 94.3% |
| imm-rb3 | imm | on | 3 | cv+ca+ct | 90.3% | 0.810 | 9.7% | 100.0% | 0.0 | 4.7% | 94.5% |
| imm | imm | on | 5 | cv+ca+ct | 89.6% | 0.805 | 10.4% | 100.0% | 0.0 | 4.3% | 95.0% |
| kf-rb3 | kf | on | 3 | cv+ca+ct | 89.4% | 0.804 | 10.6% | 100.0% | 0.0 | 4.7% | 94.4% |
| imm-cvca | imm | on | 5 | cv+ca | 89.0% | 0.801 | 11.0% | 100.0% | 0.0 | 4.3% | 94.9% |
| imm-cv | imm | on | 5 | cv | 88.9% | 0.800 | 11.1% | 100.0% | 0.0 | 4.3% | 94.9% |
| kf | kf | on | 5 | cv+ca+ct | 88.9% | 0.800 | 11.1% | 100.0% | 0.0 | 4.3% | 94.9% |
| nsa_kf | nsa_kf | on | 5 | cv+ca+ct | 88.9% | 0.800 | 11.1% | 100.0% | 0.0 | 4.3% | 94.9% |
| nsa_kf-static | nsa_kf | off | 5 | cv+ca+ct | 88.7% | 0.799 | 11.3% | 100.0% | 0.0 | 3.3% | 96.0% |

### Findings

- OVERALL (pooled over 35 scenarios): best target_success_rate 92.2% (imm-rb0), worst 88.7% (nsa_kf-static) -- spread 3.4pp.
- FAST LINEAR (best config imm-rb0): success by speed step = 2px:100.0%, 5px:100.0%, 10px:100.0%, 20px:100.0%, 40px:100.0%, 80px:98.3%; first step below 90% success: none up to 80 px/frame.
- NONLINEAR (circles + figure-eight): imm success 100.0% vs kf 100.0% (+0.0pp); mean IoU imm 1.000 vs kf 1.000.
- NONLINEAR nsa_kf 100.0%; imm - nsa_kf = +0.0pp.
- BLACKOUT (pure motion prediction while the detector is blind, target still in GT): kf success 82.3% / IoU 0.832 (linear control 80.0% / 0.814); nsa_kf success 82.3% / IoU 0.832 (linear control 80.0% / 0.814); imm success 88.5% / IoU 0.875 (linear control 80.0% / 0.814).
- BLACKOUT nonlinear: imm - kf success +6.2pp, IoU +0.042; loss rate kf 17.7% vs imm 11.5%.
- BLACKOUT success by blind-window length: 15f kf 88.6%/nsa_kf 88.6%/imm 92.9%, 20f kf 76.0%/nsa_kf 76.0%/imm 77.3%, 30f kf 77.8%/nsa_kf 77.8%/imm 91.1%.
- ROLLBACK on occlusion (kf): success rb0 100.0% / rb3 100.0% / rb5 100.0%; re-acquire latency rb0 0.0f / rb3 0.0f / rb5 0.0f; re-acquire rate rb0 100.0% / rb5 100.0%.
- ROLLBACK on occlusion (imm): success rb0 100.0% / rb3 100.0% / rb5 100.0%; re-acquire latency rb0 0.0f / rb3 0.0f / rb5 0.0f; re-acquire rate rb0 100.0% / rb5 100.0%.
- OCCLUSION inplace: re-acquire rate by gap = 5f:100.0%, 15f:100.0%, 40f:100.0%, 90f:100.0%.
- OCCLUSION distant: re-acquire rate by gap = 5f:100.0%, 15f:100.0%, 40f:100.0%, 90f:100.0%.
- DISTRACTOR IDENTICAL LOOKALIKE: kf: success 100.0%, FA 0.0% (0f), reacq n/a; nsa_kf: success 100.0%, FA 0.0% (0f), reacq n/a; imm: success 100.0%, FA 0.0% (0f), reacq n/a.
- DISTRACTOR DIFFERENT-APPEARANCE CONTROL: kf: success 100.0%, FA 0.0% (0f), reacq n/a; nsa_kf: success 100.0%, FA 0.0% (0f), reacq n/a; imm: success 100.0%, FA 0.0% (0f), reacq n/a.
- DISTRACTOR sensitivity check: success 100.0% (identical embedding) vs 100.0% (orthogonal embedding), +0.0pp; the appearance channel is the only difference between those two runs.
- NSA ADAPTIVITY: nsa_kf adaptive success 88.9% vs static 88.7% (+0.2pp); precision 0.800 vs 0.799.
- UNIDENTIFIABLE (with bystander): success 39.2%, false acquisition 54.6% over 30 run(s).
- UNIDENTIFIABLE (no bystander): success 58.7%, false acquisition 0.0% over 10 run(s).
- UNIDENTIFIABLE worst run unidentifiable_missing_s060 / kf: false acquisition 76.9% (60 frames), success 29.5%, median accepted margin in the blind window n/a.

## Where prediction quality is actually measurable

With a single person and no detector drop, association always finds the target, so motion quality cannot change the outcome (that was the first sweep's null result).  The `*_blackout` families fix that: the detector drops the target while it stays in ground truth, so every reported box during the gap is pure motion prediction and success needs IoU>=0.5 with the true box.  `linear_blackout` is the control where extrapolation is exact.

## Occlusion: how the tracker behaves when the target comes back far away

`python scripts/ablate_motion.py --trace occlusion_distant_gap15,occlusion_distant_gap40 --no-write`

A distant re-appearance is refused while the gate radius is smaller than the distance to the re-appeared target, so re-acquisition depends on the search arming.  After the fix that `StateCfg.global_states` (default `('lost_short','lost_long','gave_up','out_of_frame')`) makes `_global_search()` true from `lost_short` (fsu > 12) and `_gate_override()` uses the canonical `association.gate_radius` (whose `gate_base` is a FRACTION of the frame diagonal, ~604 px here), **all four gaps measured (5/15/40/90) re-acquire with latency 0.0**, and the previously never-re-acquiring gap 5/15 cells now succeed.  A re-appearance past the ~604 px gate (jump 900) still shows the difference: shipped 0.8689 success / latency 8.0 at gap 5 and 1.0000 / 0.0 at gap 15/40, versus 0.3443 / never-re-acquire for the `state.global_states = ()` control.  `--trace` prints `frame, state, fsu, n_dets, gated, best_cost, thr, runner_up, accepted, reason, gate_r, pred_to_gt, gt_present, blind` per frame.

## Hard-overlap distractor (the anti-latch claim)

See `distractor_sweep.md` for the full similarity x `distractor_margin` grid and `distractor_sweep.csv` for the numbers.  Headline:

- [imm] worst cell: false_acquisition_rate 7.6% at similarity 1.00, distractor_margin 0.12
- [imm] most similar bystander (similarity 1.00): margin 0.00 -> FA 0.0%, success 100.0%; margin 0.12 -> FA 7.6%, success 66.7% (success cost 33.3 pp)
- [imm] refusals at similarity 1.00: 0.0 frames at margin 0.00 vs 28.0 at margin 0.12
- [imm] measured margin (runner_up - best_cost) at similarity 1.00: median 0.057, so a nominal distractor_margin of 0.12 refuses 28.0 overlap frames
- [imm] first similarity row with any false acquisition: 0.97
- [kf] worst cell: false_acquisition_rate 7.6% at similarity 1.00, distractor_margin 0.12
- [kf] most similar bystander (similarity 1.00): margin 0.00 -> FA 0.0%, success 100.0%; margin 0.12 -> FA 7.6%, success 66.7% (success cost 33.3 pp)
- [kf] refusals at similarity 1.00: 0.0 frames at margin 0.00 vs 28.0 at margin 0.12
- [kf] measured margin (runner_up - best_cost) at similarity 1.00: median 0.057, so a nominal distractor_margin of 0.12 refuses 28.0 overlap frames
- [kf] first similarity row with any false acquisition: 0.97

## Unidentifiable target vs a bystander on the stale prediction (requirement 4)

See `unid_sweep.md` for the similarity x arming-rule grids (success, false acquisition, accepted frames and the median ACCEPTED margin inside the blind window) and `unid_sweep.csv` for the numbers.  Headline:

- UNIDENTIFIABLE mode=missing model=imm rule=no_early_search similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=missing model=imm rule=shipped similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=missing model=imm rule=no_early_search similarity=0.40: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.106
- UNIDENTIFIABLE mode=missing model=imm rule=shipped similarity=0.40: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.106
- UNIDENTIFIABLE mode=missing model=imm rule=no_early_search similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 0.884
- UNIDENTIFIABLE mode=missing model=imm rule=shipped similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 0.884
- UNIDENTIFIABLE mode=missing model=imm rule=no_early_search similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.044
- UNIDENTIFIABLE mode=missing model=imm rule=shipped similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.044
- UNIDENTIFIABLE mode=missing model=kf rule=no_early_search similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=missing model=kf rule=shipped similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=missing model=kf rule=no_early_search similarity=0.40: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.106
- UNIDENTIFIABLE mode=missing model=kf rule=shipped similarity=0.40: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.106
- UNIDENTIFIABLE mode=missing model=kf rule=no_early_search similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 0.884
- UNIDENTIFIABLE mode=missing model=kf rule=shipped similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 0.884
- UNIDENTIFIABLE mode=missing model=kf rule=no_early_search similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.044
- UNIDENTIFIABLE mode=missing model=kf rule=shipped similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.044
- UNIDENTIFIABLE mode=orthogonal model=imm rule=no_early_search similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=orthogonal model=imm rule=shipped similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=orthogonal model=imm rule=no_early_search similarity=0.40: success 52.6%, false acquisition 53.8%, accepted 24 frames inside the blind window, median accepted margin 1.026
- UNIDENTIFIABLE mode=orthogonal model=imm rule=shipped similarity=0.40: success 52.6%, false acquisition 53.8%, accepted 24 frames inside the blind window, median accepted margin 1.026
- UNIDENTIFIABLE mode=orthogonal model=imm rule=no_early_search similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.266
- UNIDENTIFIABLE mode=orthogonal model=imm rule=shipped similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.266
- UNIDENTIFIABLE mode=orthogonal model=imm rule=no_early_search similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.343
- UNIDENTIFIABLE mode=orthogonal model=imm rule=shipped similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.343
- UNIDENTIFIABLE mode=orthogonal model=kf rule=no_early_search similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=orthogonal model=kf rule=shipped similarity=0.00: success 55.1%, false acquisition 30.8%, accepted 0 frames inside the blind window, median accepted margin n/a
- UNIDENTIFIABLE mode=orthogonal model=kf rule=no_early_search similarity=0.40: success 52.6%, false acquisition 53.8%, accepted 24 frames inside the blind window, median accepted margin 1.026
- UNIDENTIFIABLE mode=orthogonal model=kf rule=shipped similarity=0.40: success 52.6%, false acquisition 53.8%, accepted 24 frames inside the blind window, median accepted margin 1.026
- UNIDENTIFIABLE mode=orthogonal model=kf rule=no_early_search similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.266
- UNIDENTIFIABLE mode=orthogonal model=kf rule=shipped similarity=0.60: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.266
- UNIDENTIFIABLE mode=orthogonal model=kf rule=no_early_search similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.342
- UNIDENTIFIABLE mode=orthogonal model=kf rule=shipped similarity=0.85: success 29.5%, false acquisition 76.9%, accepted 24 frames inside the blind window, median accepted margin 1.342
- UNIDENTIFIABLE worst false-acquisition cell (all modes/models/rules): 76.9% at mode=missing model=imm rung=no_early_search cos=0.40
- UNIDENTIFIABLE control (no bystander) mode=missing model=imm: success 55.1%, false acquisition 0.0%; with the pre-fix rule 55.1% (+0.0pp)
- UNIDENTIFIABLE control (no bystander) mode=missing model=kf: success 55.1%, false acquisition 0.0%; with the pre-fix rule 55.1% (+0.0pp)
- UNIDENTIFIABLE control (no bystander) mode=orthogonal model=imm: success 55.1%, false acquisition 0.0%; with the pre-fix rule 55.1% (+0.0pp)
- UNIDENTIFIABLE control (no bystander) mode=orthogonal model=kf: success 55.1%, false acquisition 0.0%; with the pre-fix rule 55.1% (+0.0pp)
- UNIDENTIFIABLE arming-rule effect mode=missing model=imm metric=success: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 29.5% vs 29.5%)
- UNIDENTIFIABLE arming-rule effect mode=missing model=imm metric=false acquisition: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 76.9% vs 76.9%)
- UNIDENTIFIABLE arming-rule effect mode=missing model=kf metric=success: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 29.5% vs 29.5%)
- UNIDENTIFIABLE arming-rule effect mode=missing model=kf metric=false acquisition: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 76.9% vs 76.9%)
- UNIDENTIFIABLE arming-rule effect mode=orthogonal model=imm metric=success: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 29.5% vs 29.5%)
- UNIDENTIFIABLE arming-rule effect mode=orthogonal model=imm metric=false acquisition: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 76.9% vs 76.9%)
- UNIDENTIFIABLE arming-rule effect mode=orthogonal model=kf metric=success: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 29.5% vs 29.5%)
- UNIDENTIFIABLE arming-rule effect mode=orthogonal model=kf metric=false acquisition: largest |shipped - no_early_search| = 0.0 pp at similarity 0.85 (shipped 76.9% vs 76.9%)

## Caveats

- The stub embedder isolates motion/appearance interaction *by construction*; it is not a ReID benchmark.  It exists so that a motion change is the only difference between two runs.
- `id_consistency` is geometric: `TargetTracker` does not put a track id in `_last_diag`, so the evaluator decides which identity a report latched onto by IoU against `all_gt`.
- Frames are all-zero arrays: only box geometry is simulated, never pixels or crops.
- `MotionModel.predict()` advances one frame per call and is not idempotent (`coast()` is); the harness calls it exactly once per frame through `TargetTracker.step`.
