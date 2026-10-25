# Hard-overlap distractor sweep

Two people ride on the same box for the whole overlap window (IoU 0.876), so motion, size and geometry are near-identical and only appearance can separate them.  The bystander's descriptor is correlated with the target's by the row's cosine while the target's own crop carries a fixed pose offset (0.97), so for high rows the bystander is *more* similar to the template than the target itself: those cells are where the `CostCfg.distractor_margin` refusal is the only thing standing between the tracker and a latch.  The bystander peels away after the window, which is what turns a latch into false acquisitions and lost success.

- runs: 56 (7 similarity rows x 8 (model, margin) cells), errors: 0
- frames simulated: 3696
- metrics: `tgtrack.evaluate.evaluate_target` (never hand-rolled)
- `target_success_rate` counts the whole scenario, so a latch is penalised twice: the frames during the latch and the frames after the peel.

## model `imm`  (rollback_depth = 0)

### false_acquisition_rate (%)  [primary endpoint]

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.60 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.85 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.92 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.97 | 0.0 | 0.0 | 0.0 | 7.6 |
| 0.99 | 0.0 | 0.0 | 0.0 | 7.6 |
| 1.00 | 0.0 | 0.0 | 7.6 | 7.6 |

### target_success_rate (%)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.60 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.85 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.92 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.97 | 100.0 | 100.0 | 100.0 | 66.7 |
| 0.99 | 100.0 | 100.0 | 100.0 | 66.7 |
| 1.00 | 100.0 | 100.0 | 66.7 | 66.7 |

### id_consistency

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.97 | 1.00 | 1.00 | 1.00 | 0.90 |
| 0.99 | 1.00 | 1.00 | 1.00 | 0.90 |
| 1.00 | 1.00 | 1.00 | 0.90 | 0.90 |

### refused frames inside the overlap window (reason=ambiguous)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.85 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.92 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.97 | 0.00 | 0.00 | 0.00 | 28.00 |
| 0.99 | 0.00 | 0.00 | 0.00 | 28.00 |
| 1.00 | 0.00 | 0.00 | 28.00 | 28.00 |

### median measured margin  runner_up - best_cost  (overlap window)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 1.42 | 1.42 | 1.42 | 1.42 |
| 0.60 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.85 | 0.20 | 0.20 | 0.20 | 0.20 |
| 0.92 | 0.12 | 0.12 | 0.12 | 0.12 |
| 0.97 | 0.08 | 0.08 | 0.08 | 0.00 |
| 0.99 | 0.07 | 0.07 | 0.07 | 0.01 |
| 1.00 | 0.06 | 0.06 | 0.02 | 0.02 |

## model `kf`  (rollback_depth = 0)

### false_acquisition_rate (%)  [primary endpoint]

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.60 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.85 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.92 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.97 | 0.0 | 0.0 | 0.0 | 7.6 |
| 0.99 | 0.0 | 0.0 | 0.0 | 7.6 |
| 1.00 | 0.0 | 0.0 | 7.6 | 7.6 |

### target_success_rate (%)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.60 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.85 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.92 | 100.0 | 100.0 | 100.0 | 100.0 |
| 0.97 | 100.0 | 100.0 | 100.0 | 66.7 |
| 0.99 | 100.0 | 100.0 | 100.0 | 66.7 |
| 1.00 | 100.0 | 100.0 | 66.7 | 66.7 |

### id_consistency

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.97 | 1.00 | 1.00 | 1.00 | 0.90 |
| 0.99 | 1.00 | 1.00 | 1.00 | 0.90 |
| 1.00 | 1.00 | 1.00 | 0.90 | 0.90 |

### refused frames inside the overlap window (reason=ambiguous)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.85 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.92 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.97 | 0.00 | 0.00 | 0.00 | 28.00 |
| 0.99 | 0.00 | 0.00 | 0.00 | 28.00 |
| 1.00 | 0.00 | 0.00 | 28.00 | 28.00 |

### median measured margin  runner_up - best_cost  (overlap window)

| similarity \ margin | 0.00 | 0.03 | 0.06 | 0.12 |
|---|---|---|---|---|
| 0.00 | 1.42 | 1.42 | 1.42 | 1.42 |
| 0.60 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.85 | 0.20 | 0.20 | 0.20 | 0.20 |
| 0.92 | 0.12 | 0.12 | 0.12 | 0.12 |
| 0.97 | 0.08 | 0.08 | 0.08 | 0.00 |
| 0.99 | 0.07 | 0.07 | 0.07 | 0.01 |
| 1.00 | 0.06 | 0.06 | 0.02 | 0.02 |

## Findings

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
