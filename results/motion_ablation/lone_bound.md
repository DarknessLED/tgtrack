# Lone-candidate guard: `CostCfg.lone_candidate_accept_cost`

On the unid family the bystander is usually the ONLY finite candidate, so `association.select_candidate` leaves `margin = inf` and the ambiguity test cannot fire whatever `distractor_margin` says.  `lone_candidate_accept_cost` refuses an association when exactly one candidate is finite and its cost exceeds the bound.  This grid asks whether a bound REMOVES the handover (false acquisition -> 0) or merely moves it, and what it costs on the clean families.

- runs: 100 (10 scenarios x 10 cells), errors: 0
- frames simulated: 7800
- the `lnone` column is the inert shipped default (`lone_candidate_accept_cost = None`).

## mode `missing`

### model `imm`

#### target_success_rate (%)  [does the track come back?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 55.1 | 55.1 | 55.1 | 55.1 | 29.5 |
| 0.60 | 55.1 | 55.1 | 55.1 | 55.1 | 29.5 |
| 0.85 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 30.8 | 30.8 | 30.8 | 30.8 | 76.9 |
| 0.60 | 30.8 | 30.8 | 30.8 | 30.8 | 76.9 |
| 0.85 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 24.00 | 24.00 | 24.00 | 24.00 | 60.00 |
| 0.60 | 24.00 | 24.00 | 24.00 | 24.00 | 60.00 |
| 0.85 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |

#### id_consistency

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.61 | 0.61 | 0.61 | 0.61 | 0.23 |
| 0.60 | 0.61 | 0.61 | 0.61 | 0.61 | 0.23 |
| 0.85 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |

#### target_switch_count

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 |
| 0.60 | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 24.00 |
| 0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 24.00 |
| 0.85 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | - | - | - | - | 1.11 |
| 0.60 | - | - | - | - | 0.88 |
| 0.85 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| 0.40 | 0.04 | 0.04 | 0.04 | 0.04 | 1.11 |
| 0.60 | 0.02 | 0.02 | 0.02 | 0.02 | 0.88 |
| 0.85 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 |

### model `kf`

#### target_success_rate (%)  [does the track come back?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 55.1 | 55.1 | 55.1 | 55.1 | 29.5 |
| 0.60 | 55.1 | 55.1 | 55.1 | 55.1 | 29.5 |
| 0.85 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 30.8 | 30.8 | 30.8 | 30.8 | 76.9 |
| 0.60 | 30.8 | 30.8 | 30.8 | 30.8 | 76.9 |
| 0.85 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 24.00 | 24.00 | 24.00 | 24.00 | 60.00 |
| 0.60 | 24.00 | 24.00 | 24.00 | 24.00 | 60.00 |
| 0.85 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |

#### id_consistency

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.61 | 0.61 | 0.61 | 0.61 | 0.23 |
| 0.60 | 0.61 | 0.61 | 0.61 | 0.61 | 0.23 |
| 0.85 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |

#### target_switch_count

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 |
| 0.60 | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 24.00 |
| 0.60 | 0.00 | 0.00 | 0.00 | 0.00 | 24.00 |
| 0.85 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | - | - | - | - | 1.11 |
| 0.60 | - | - | - | - | 0.88 |
| 0.85 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| 0.40 | 0.04 | 0.04 | 0.04 | 0.04 | 1.11 |
| 0.60 | 0.02 | 0.02 | 0.02 | 0.02 | 0.88 |
| 0.85 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 |

## mode `orthogonal`

### model `imm`

#### target_success_rate (%)  [does the track come back?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 55.1 | 55.1 | 55.1 | 55.1 | 52.6 |
| 0.60 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |
| 0.85 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 52.6 | 52.6 | 52.6 | 52.6 | 53.8 |
| 0.60 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |
| 0.85 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 41.00 | 41.00 | 41.00 | 41.00 | 42.00 |
| 0.60 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |
| 0.85 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |

#### id_consistency

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.47 | 0.47 | 0.47 | 0.47 | 0.46 |
| 0.60 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |
| 0.85 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |

#### target_switch_count

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

### model `kf`

#### target_success_rate (%)  [does the track come back?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 55.1 | 55.1 | 55.1 | 55.1 | 52.6 |
| 0.60 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |
| 0.85 | 32.1 | 32.1 | 32.1 | 32.1 | 29.5 |

#### false_acquisition_rate (%)  [handed the track to the bystander?]

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 52.6 | 52.6 | 52.6 | 52.6 | 53.8 |
| 0.60 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |
| 0.85 | 75.6 | 75.6 | 75.6 | 75.6 | 76.9 |

#### false_acquisition_frames (of n_blind + n_post)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 41.00 | 41.00 | 41.00 | 41.00 | 42.00 |
| 0.60 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |
| 0.85 | 59.00 | 59.00 | 59.00 | 59.00 | 60.00 |

#### id_consistency

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.47 | 0.47 | 0.47 | 0.47 | 0.46 |
| 0.60 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |
| 0.85 | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 |

#### target_switch_count

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.40 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

#### accepted frames inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.60 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.85 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |

#### median ACCEPTED margin (runner_up - best_cost) inside the blind window

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

#### median margin over all blind-window frames (accepted or not)

| similarity \ bound | l020 | l030 | l040 | l050 | lnone |
|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

## handover frame and `n_dets` at handover

| mode | model | bound | similarity | handover frame | n_dets | phase | success | false_acq |
|---|---|---|---|---|---|---|---|---|
| missing | imm | l020 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | imm | l020 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | imm | l020 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | imm | l020 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | imm | l030 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | imm | l030 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | imm | l030 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | imm | l030 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | imm | l040 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | imm | l040 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | imm | l040 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | imm | l040 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | imm | l050 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | imm | l050 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | imm | l050 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | imm | l050 | 0.85 | 31 | 1 | gap | 32.1 | 75.6 |
| missing | imm | lnone | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | imm | lnone | 0.40 | 18 | 1 | gap | 29.5 | 76.9 |
| missing | imm | lnone | 0.60 | 18 | 1 | gap | 29.5 | 76.9 |
| missing | imm | lnone | 0.85 | 18 | 1 | gap | 29.5 | 76.9 |
| missing | kf | l020 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | kf | l020 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | kf | l020 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | kf | l020 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | kf | l030 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | kf | l030 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | kf | l030 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | kf | l030 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | kf | l040 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | kf | l040 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | kf | l040 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | kf | l040 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| missing | kf | l050 | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | kf | l050 | 0.40 |  |  |  | 55.1 | 30.8 |
| missing | kf | l050 | 0.60 |  |  |  | 55.1 | 30.8 |
| missing | kf | l050 | 0.85 | 31 | 1 | gap | 32.1 | 75.6 |
| missing | kf | lnone | 0.00 |  |  |  | 55.1 | 30.8 |
| missing | kf | lnone | 0.40 | 18 | 1 | gap | 29.5 | 76.9 |
| missing | kf | lnone | 0.60 | 18 | 1 | gap | 29.5 | 76.9 |
| missing | kf | lnone | 0.85 | 18 | 1 | gap | 29.5 | 76.9 |
| orthogonal | imm | l020 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | imm | l020 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | imm | l020 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l020 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l030 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | imm | l030 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | imm | l030 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l030 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l040 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | imm | l040 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | imm | l040 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l040 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l050 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | imm | l050 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | imm | l050 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | imm | l050 | 0.85 | 31 | 1 | gap | 32.1 | 75.6 |
| orthogonal | imm | lnone | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | imm | lnone | 0.40 | 18 | 1 | gap | 52.6 | 53.8 |
| orthogonal | imm | lnone | 0.60 | 18 | 1 | gap | 29.5 | 76.9 |
| orthogonal | imm | lnone | 0.85 | 18 | 1 | gap | 29.5 | 76.9 |
| orthogonal | kf | l020 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | kf | l020 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | kf | l020 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l020 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l030 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | kf | l030 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | kf | l030 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l030 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l040 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | kf | l040 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | kf | l040 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l040 | 0.85 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l050 | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | kf | l050 | 0.40 | 36 | 2 | gap | 55.1 | 52.6 |
| orthogonal | kf | l050 | 0.60 | 36 | 2 | gap | 32.1 | 75.6 |
| orthogonal | kf | l050 | 0.85 | 31 | 1 | gap | 32.1 | 75.6 |
| orthogonal | kf | lnone | 0.00 |  |  |  | 55.1 | 30.8 |
| orthogonal | kf | lnone | 0.40 | 18 | 1 | gap | 52.6 | 53.8 |
| orthogonal | kf | lnone | 0.60 | 18 | 1 | gap | 29.5 | 76.9 |
| orthogonal | kf | lnone | 0.85 | 18 | 1 | gap | 29.5 | 76.9 |

## control: no bystander (`unidentifiable_solo`)

| mode | model | bound | success | false_acq | id_consistency | accepted in blind window |
|---|---|---|---|---|---|---|
| missing | kf | l020 | 55.1 | 0.0 | 1.00 | 0 |
| missing | kf | l030 | 55.1 | 0.0 | 1.00 | 0 |
| missing | kf | l040 | 55.1 | 0.0 | 1.00 | 0 |
| missing | kf | l050 | 55.1 | 0.0 | 1.00 | 0 |
| missing | kf | lnone | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | l020 | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | l030 | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | l040 | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | l050 | 55.1 | 0.0 | 1.00 | 0 |
| missing | imm | lnone | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | l020 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | l030 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | l040 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | l050 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | kf | lnone | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | l020 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | l030 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | l040 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | l050 | 55.1 | 0.0 | 1.00 | 0 |
| orthogonal | imm | lnone | 55.1 | 0.0 | 1.00 | 0 |

## cost on the clean families (no unid scenarios)

| bound | success | IoU | loss | re-acquire rate | false_acq | id_consistency |
|---|---|---|---|---|---|---|
| l020 | 85.7 | 0.755 | 14.3 | 100.0 | 0.0 | 1.000 |
| l030 | 88.1 | 0.776 | 11.9 | 100.0 | 0.0 | 1.000 |
| l040 | 93.5 | 0.821 | 6.5 | 100.0 | 0.0 | 1.000 |
| l050 | 96.3 | 0.845 | 3.7 | 100.0 | 0.0 | 1.000 |
| lnone | 98.9 | 0.868 | 1.1 | 100.0 | 0.0 | 1.000 |

- LONE clean families bound=l020: success 85.7%, IoU 0.755, loss 14.3%, false acquisition 0.0%, id_consistency 1.000 (success -13.2pp vs the inert control, false acquisition +0.0pp)
- LONE clean families bound=l030: success 88.1%, IoU 0.776, loss 11.9%, false acquisition 0.0%, id_consistency 1.000 (success -10.8pp vs the inert control, false acquisition +0.0pp)
- LONE clean families bound=l040: success 93.5%, IoU 0.821, loss 6.5%, false acquisition 0.0%, id_consistency 1.000 (success -5.4pp vs the inert control, false acquisition +0.0pp)
- LONE clean families bound=l050: success 96.3%, IoU 0.845, loss 3.7%, false acquisition 0.0%, id_consistency 1.000 (success -2.6pp vs the inert control, false acquisition +0.0pp)
- LONE clean families bound=lnone: success 98.9%, IoU 0.868, loss 1.1%, false acquisition 0.0%, id_consistency 1.000

- LONE mode=missing model=imm bound=l020: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=imm bound=l030: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=imm bound=l040: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=imm bound=l050: 1/4 runs hand over (n_dets=1: 1, n_dets>=2: 0), handover frames [31], peak false acquisition 75.6%
- LONE mode=missing model=imm bound=lnone: 3/4 runs hand over (n_dets=1: 3, n_dets>=2: 0), handover frames [18], peak false acquisition 76.9%
- LONE mode=missing model=imm: lowest peak false acquisition 75.6% at bound l020
- LONE mode=missing model=kf bound=l020: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=kf bound=l030: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=kf bound=l040: 1/4 runs hand over (n_dets=1: 0, n_dets>=2: 1), handover frames [36], peak false acquisition 75.6%
- LONE mode=missing model=kf bound=l050: 1/4 runs hand over (n_dets=1: 1, n_dets>=2: 0), handover frames [31], peak false acquisition 75.6%
- LONE mode=missing model=kf bound=lnone: 3/4 runs hand over (n_dets=1: 3, n_dets>=2: 0), handover frames [18], peak false acquisition 76.9%
- LONE mode=missing model=kf: lowest peak false acquisition 75.6% at bound l020
- LONE mode=orthogonal model=imm bound=l020: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=imm bound=l030: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=imm bound=l040: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=imm bound=l050: 3/4 runs hand over (n_dets=1: 1, n_dets>=2: 2), handover frames [31, 36], peak false acquisition 75.6%
- LONE mode=orthogonal model=imm bound=lnone: 3/4 runs hand over (n_dets=1: 3, n_dets>=2: 0), handover frames [18], peak false acquisition 76.9%
- LONE mode=orthogonal model=imm: lowest peak false acquisition 75.6% at bound l020
- LONE mode=orthogonal model=kf bound=l020: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=kf bound=l030: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=kf bound=l040: 3/4 runs hand over (n_dets=1: 0, n_dets>=2: 3), handover frames [36], peak false acquisition 75.6%
- LONE mode=orthogonal model=kf bound=l050: 3/4 runs hand over (n_dets=1: 1, n_dets>=2: 2), handover frames [31, 36], peak false acquisition 75.6%
- LONE mode=orthogonal model=kf bound=lnone: 3/4 runs hand over (n_dets=1: 3, n_dets>=2: 0), handover frames [18], peak false acquisition 76.9%
- LONE mode=orthogonal model=kf: lowest peak false acquisition 75.6% at bound l020
- LONE control (no bystander) mode=missing model=imm bound=l020: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=imm bound=l030: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=imm bound=l040: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=imm bound=l050: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=imm bound=lnone: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=kf bound=l020: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=kf bound=l030: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=kf bound=l040: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=kf bound=l050: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=missing model=kf bound=lnone: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=imm bound=l020: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=imm bound=l030: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=imm bound=l040: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=imm bound=l050: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=imm bound=lnone: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=kf bound=l020: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=kf bound=l030: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=kf bound=l040: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=kf bound=l050: success 55.1%, false acquisition 0.0%
- LONE control (no bystander) mode=orthogonal model=kf bound=lnone: success 55.1%, false acquisition 0.0%
