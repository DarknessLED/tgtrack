# Unidentifiable target + identifiable bystander on the stale prediction

The target walks, HALTS, and then stops being identifiable (its descriptor is missing -- `encode` returns no features, which is the tracker's `appearance_valid=False` case -- or is valid but orthogonal).  The tracker's constant-velocity coast keeps advancing after the halt, and an **unoccluded bystander with a clean descriptor rides that stale prediction**, so geometry points at the bystander and appearance is the only cue that can refuse.  The target becomes identifiable again in the last frames, so `target_success_rate` measures whether the track came back at all.

- runs: 40 (10 scenarios x 4 cells), errors: 0
- frames simulated: 3120
- metrics: `tgtrack.evaluate.evaluate_target` (never hand-rolled)
- `shipped` = the current arming rule (`state.global_states` defaults to `('lost_short','lost_long','gave_up','out_of_frame')`); `no_early_search` = `state.global_states = ()`, the pre-fix rule that only searches after `lost_long_after` frames.
- `id_consistency` is `nan` (rendered `-`) when no report is correct enough to name an identity.

## mode `missing`

### model `imm`

#### target_success_rate (%)  [does the track come back?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 55.1 | 55.1 |
| 0.40 | 29.5 | 29.5 |
| 0.60 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 30.8 | 30.8 |
| 0.40 | 76.9 | 76.9 |
| 0.60 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 24.00 | 24.00 |
| 0.40 | 60.00 | 60.00 |
| 0.60 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 |

#### id_consistency

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.61 | 0.61 |
| 0.40 | 0.23 | 0.23 |
| 0.60 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 |

#### target_switch_count

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 2.00 | 2.00 |
| 0.40 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | - | - |
| 0.40 | 1.11 | 1.11 |
| 0.60 | 0.88 | 0.88 |
| 0.85 | 1.04 | 1.04 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.57 |
| 0.40 | 1.11 | 1.11 |
| 0.60 | 0.88 | 0.88 |
| 0.85 | 1.04 | 1.04 |

### model `kf`

#### target_success_rate (%)  [does the track come back?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 55.1 | 55.1 |
| 0.40 | 29.5 | 29.5 |
| 0.60 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 30.8 | 30.8 |
| 0.40 | 76.9 | 76.9 |
| 0.60 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 24.00 | 24.00 |
| 0.40 | 60.00 | 60.00 |
| 0.60 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 |

#### id_consistency

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.61 | 0.61 |
| 0.40 | 0.23 | 0.23 |
| 0.60 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 |

#### target_switch_count

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 2.00 | 2.00 |
| 0.40 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | - | - |
| 0.40 | 1.11 | 1.11 |
| 0.60 | 0.88 | 0.88 |
| 0.85 | 1.04 | 1.04 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.57 |
| 0.40 | 1.11 | 1.11 |
| 0.60 | 0.88 | 0.88 |
| 0.85 | 1.04 | 1.04 |

## mode `orthogonal`

### model `imm`

#### target_success_rate (%)  [does the track come back?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 55.1 | 55.1 |
| 0.40 | 52.6 | 52.6 |
| 0.60 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 30.8 | 30.8 |
| 0.40 | 53.8 | 53.8 |
| 0.60 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 24.00 | 24.00 |
| 0.40 | 42.00 | 42.00 |
| 0.60 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 |

#### id_consistency

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.61 | 0.61 |
| 0.40 | 0.46 | 0.46 |
| 0.60 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 |

#### target_switch_count

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 |
| 0.60 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | - | - |
| 0.40 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 |

### model `kf`

#### target_success_rate (%)  [does the track come back?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 55.1 | 55.1 |
| 0.40 | 52.6 | 52.6 |
| 0.60 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 30.8 | 30.8 |
| 0.40 | 53.8 | 53.8 |
| 0.60 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 24.00 | 24.00 |
| 0.40 | 42.00 | 42.00 |
| 0.60 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 |

#### id_consistency

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.61 | 0.61 |
| 0.40 | 0.46 | 0.46 |
| 0.60 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 |

#### target_switch_count

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 |
| 0.60 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | - | - |
| 0.40 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ arming rule | no_early_search | shipped |
|---|---|---|
| 0.00 | 0.00 | 0.00 |
| 0.40 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 |

## control: no bystander (`unidentifiable_solo`)

| mode | model | config | success | false_acq | id_consistency | accepted in blind window |
|---|---|---|---|---|---|---|
| missing | kf | kf-shipped | 55.1 | 0.0 | 1.00 | 0 |
| missing | kf | kf-no_early_search | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | imm-shipped | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | imm-no_early_search | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | kf-shipped | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | kf-no_early_search | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | imm-shipped | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | imm-no_early_search | 55.1 | 0.0 | 1.00 | 0 |

## Findings

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
