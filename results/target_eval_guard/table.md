# Clicked-target strategy ablation

Config: ablate=all, allow_invisible_target=False, data_root=data/sequences, detector=gt, dry_run=True, embedder=fallback, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper', 'left', 'right', 'edge']}, jobs=4, masks=['none', 'lower', 'upper', 'left', 'right', 'edge'], max_frames=24, subsample=1, target_decisions={'synth-stress': {'target_id': 1, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper', 'left', 'right', 'edge']}, 'observability': {1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'left': {'mode': 'left', 'frames': 21, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'right': {'mode': 'right', 'frames': 21, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'edge': {'mode': 'edge', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 3570.0, 'visible_fraction': 0.9908424908424908}}, 'worst_mask': 'lower', 'median_w': 13.0, 'median_h': 85.0, 'median_area': 2210.0, 'frames': 21}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'left': {'mode': 'left', 'frames': 24, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'right': {'mode': 'right', 'frames': 24, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'edge': {'mode': 'edge', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}}, 'worst_mask': 'lower', 'median_w': 13.0, 'median_h': 85.0, 'median_area': 2210.0, 'frames': 24}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'left': {'mode': 'left', 'frames': 24, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'right': {'mode': 'right', 'frames': 24, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'edge': {'mode': 'edge', 'frames': 24, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}}, 'worst_mask': 'lower', 'median_w': 13.0, 'median_h': 85.0, 'median_area': 2210.0, 'frames': 24}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 4420.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 85.0, 'median_area': 2210.0, 'min_h': 85.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'left': {'mode': 'left', 'frames': 21, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'right': {'mode': 'right', 'frames': 21, 'first': 0, 'median_w': 13.0, 'median_h': 170.0, 'median_area': 2210.0, 'min_h': 170.0, 'min_area': 2210.0, 'visible_fraction': 0.5}, 'edge': {'mode': 'edge', 'frames': 21, 'first': 0, 'median_w': 26.0, 'median_h': 170.0, 'median_area': 4420.0, 'min_h': 170.0, 'min_area': 3570.0, 'visible_fraction': 0.9908424908424908}}, 'worst_mask': 'lower', 'median_w': 13.0, 'median_h': 85.0, 'median_area': 2210.0, 'frames': 21}, 'message': ''}}, target_ids={'synth-stress': 1}, target_strategy=stressed, wall_seconds=9.7

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.92 |
| baseline | baseline | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 6.62 |
| baseline | baseline | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 8.47 |
| baseline | baseline | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.53 |
| baseline | baseline | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 7.14 |
| baseline | baseline | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 8.03 |
| distractor_margin | margin_0.00 | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.69 |
| distractor_margin | margin_0.00 | left | 1 | 24 | 21 | 0.0476 | 0.0738 | 0.9524 | 0.9524 | 0.0000 | - | - | - | 0.95833 | 0.95833 | 1 | 1 | 0.0417 | 1 | 0 | 1 | - | - | - | 8.10 |
| distractor_margin | margin_0.00 | lower | 1 | 24 | 21 | 0.5238 | 0.4852 | 0.4762 | 0.4762 | 0.0000 | - | - | - | 0.50000 | 0.50000 | 2 | 2 | 0.4783 | 3 | 1 | 10 | - | - | - | 9.33 |
| distractor_margin | margin_0.00 | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 10.11 |
| distractor_margin | margin_0.00 | right | 1 | 24 | 21 | 0.0952 | 0.1048 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.37500 | 0.75000 | 2 | 2 | 0.1818 | 3 | 1 | 1 | - | - | - | 7.98 |
| distractor_margin | margin_0.00 | upper | 1 | 24 | 21 | 0.9524 | 0.8536 | 0.0476 | 0.0476 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9091 | 2 | 2 | 10 | - | - | - | 8.02 |
| distractor_margin | margin_0.06 | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.72 |
| distractor_margin | margin_0.06 | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 6.48 |
| distractor_margin | margin_0.06 | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 8.71 |
| distractor_margin | margin_0.06 | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.09 |
| distractor_margin | margin_0.06 | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 6.69 |
| distractor_margin | margin_0.06 | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 8.20 |
| distractor_margin | margin_0.12 | edge | 1 | 24 | 21 | 0.9524 | 0.8469 | 0.0476 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8696 | 2 | 1 | 12 | - | - | - | 8.93 |
| distractor_margin | margin_0.12 | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 6.41 |
| distractor_margin | margin_0.12 | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 8.27 |
| distractor_margin | margin_0.12 | none | 1 | 24 | 21 | 0.9524 | 0.8469 | 0.0476 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8696 | 2 | 1 | 12 | - | - | - | 8.55 |
| distractor_margin | margin_0.12 | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.29167 | 0.63636 | 2 | 2 | 0.2222 | 3 | 1 | 1 | - | - | - | 7.11 |
| distractor_margin | margin_0.12 | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 8.09 |
| freeze | freeze_off | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.84 |
| freeze | freeze_off | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.89 |
| freeze | freeze_off | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 16.11 |
| freeze | freeze_off | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.97 |
| freeze | freeze_off | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.78 |
| freeze | freeze_off | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 12.95 |
| freeze | freeze_on | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 16.62 |
| freeze | freeze_on | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.10 |
| freeze | freeze_on | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 13.74 |
| freeze | freeze_on | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 15.33 |
| freeze | freeze_on | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 11.08 |
| freeze | freeze_on | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 12.27 |
| gate | gate_growth_unfrozen | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 15.56 |
| gate | gate_growth_unfrozen | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.37 |
| gate | gate_growth_unfrozen | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 12.81 |
| gate | gate_growth_unfrozen | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.85 |
| gate | gate_growth_unfrozen | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.99 |
| gate | gate_growth_unfrozen | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.70 |
| gate | gate_legacy | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.29 |
| gate | gate_legacy | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.0476 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 1 | - | - | - | 8.25 |
| gate | gate_legacy | lower | 1 | 24 | 21 | 0.1429 | 0.1917 | 0.8571 | 0.1905 | 1.0000 | 4.00 | 4.00 | 4.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 12.44 |
| gate | gate_legacy | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 12.95 |
| gate | gate_legacy | right | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.0476 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 1 | - | - | - | 7.68 |
| gate | gate_legacy | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.80 |
| gate | gate_search_early | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.55 |
| gate | gate_search_early | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 9.18 |
| gate | gate_search_early | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 14.39 |
| gate | gate_search_early | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.35 |
| gate | gate_search_early | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 11.99 |
| gate | gate_search_early | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.80 |
| gate | gate_search_late | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 15.17 |
| gate | gate_search_late | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.0476 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 1 | - | - | - | 7.64 |
| gate | gate_search_late | lower | 1 | 24 | 21 | 0.1429 | 0.1917 | 0.8571 | 0.1905 | 1.0000 | 4.00 | 4.00 | 4.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 13.18 |
| gate | gate_search_late | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.59 |
| gate | gate_search_late | right | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.0476 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 1 | - | - | - | 7.64 |
| gate | gate_search_late | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.46 |
| hard_negative | hard_negative_off | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.75 |
| hard_negative | hard_negative_off | left | 1 | 24 | 21 | 0.0476 | 0.0511 | 0.9524 | 0.9048 | 0.0000 | - | - | - | 0.87500 | 0.91304 | 1 | 1 | 0.0455 | 1 | 0 | 1 | - | - | - | 7.67 |
| hard_negative | hard_negative_off | lower | 1 | 24 | 21 | 0.0476 | 0.0870 | 0.9524 | 0.9524 | 0.0000 | - | - | - | 0.95833 | 0.95833 | 1 | 1 | 0.0417 | 1 | 0 | 1 | - | - | - | 8.10 |
| hard_negative | hard_negative_off | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.05 |
| hard_negative | hard_negative_off | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 6.92 |
| hard_negative | hard_negative_off | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 7.37 |
| ltm_capacity | ltm_cap_1 | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.77 |
| ltm_capacity | ltm_cap_1 | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.18 |
| ltm_capacity | ltm_cap_1 | lower | 1 | 24 | 21 | 0.4286 | 0.4306 | 0.5714 | 0.0952 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 7 | - | - | - | 11.38 |
| ltm_capacity | ltm_cap_1 | none | 1 | 24 | 21 | 0.0952 | 0.1389 | 0.9048 | 0.0952 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 1 | 1 | - | - | - | 11.80 |
| ltm_capacity | ltm_cap_1 | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.60 |
| ltm_capacity | ltm_cap_1 | upper | 1 | 24 | 21 | 0.4286 | 0.4306 | 0.5714 | 0.0952 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 7 | - | - | - | 8.18 |
| ltm_capacity | ltm_cap_384 | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 15.60 |
| ltm_capacity | ltm_cap_384 | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.34 |
| ltm_capacity | ltm_cap_384 | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 13.34 |
| ltm_capacity | ltm_cap_384 | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.00 |
| ltm_capacity | ltm_cap_384 | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.96 |
| ltm_capacity | ltm_cap_384 | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.48 |
| ltm_capacity | ltm_cap_96 | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.70 |
| ltm_capacity | ltm_cap_96 | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.05 |
| ltm_capacity | ltm_cap_96 | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 14.73 |
| ltm_capacity | ltm_cap_96 | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.94 |
| ltm_capacity | ltm_cap_96 | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.58 |
| ltm_capacity | ltm_cap_96 | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.89 |
| part_aware | part_aware_off | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 10.36 |
| part_aware | part_aware_off | left | 1 | 24 | 21 | 0.0476 | 0.0738 | 0.9524 | 0.9524 | 0.0000 | - | - | - | 0.95833 | 0.95833 | 1 | 1 | 0.0417 | 1 | 0 | 1 | - | - | - | 8.38 |
| part_aware | part_aware_off | lower | 1 | 24 | 21 | 0.0952 | 0.1423 | 0.9048 | 0.2381 | 0.0000 | - | - | - | 0.12500 | 0.42857 | 2 | 2 | 0.4000 | 3 | 1 | 1 | - | - | - | 9.15 |
| part_aware | part_aware_off | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.18 |
| part_aware | part_aware_off | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 7.38 |
| part_aware | part_aware_off | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 8.19 |
| part_matching | part_matching_off | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.67 |
| part_matching | part_matching_off | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 7.02 |
| part_matching | part_matching_off | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 9.21 |
| part_matching | part_matching_off | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.28 |
| part_matching | part_matching_off | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 7.11 |
| part_matching | part_matching_off | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 7.74 |
| stm_ltm | balanced | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 14.01 |
| stm_ltm | balanced | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 9.36 |
| stm_ltm | balanced | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 12.63 |
| stm_ltm | balanced | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 12.60 |
| stm_ltm | balanced | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 9.98 |
| stm_ltm | balanced | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 11.64 |
| stm_ltm | ltm_only | edge | 1 | 24 | 21 | 0.9524 | 0.8469 | 0.0476 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8696 | 2 | 1 | 12 | - | - | - | 14.57 |
| stm_ltm | ltm_only | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8750 | 2 | 1 | 12 | - | - | - | 13.55 |
| stm_ltm | ltm_only | lower | 1 | 24 | 21 | 0.0476 | 0.0870 | 0.9524 | 0.9524 | 0.0000 | - | - | - | 0.95833 | 0.95833 | 1 | 1 | 0.0417 | 1 | 0 | 1 | - | - | - | 11.27 |
| stm_ltm | ltm_only | none | 1 | 24 | 21 | 0.9524 | 0.8469 | 0.0476 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8696 | 2 | 1 | 12 | - | - | - | 12.65 |
| stm_ltm | ltm_only | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 11.28 |
| stm_ltm | ltm_only | upper | 1 | 24 | 21 | 0.5238 | 0.4786 | 0.4762 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.41667 | 2 | 2 | 0.5238 | 2 | 1 | 10 | - | - | - | 12.52 |
| stm_ltm | stm_only | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 13.22 |
| stm_ltm | stm_only | left | 1 | 24 | 21 | 0.0476 | 0.0738 | 0.9524 | 0.9524 | 0.0000 | - | - | - | 0.95833 | 0.95833 | 1 | 1 | 0.0417 | 1 | 0 | 1 | - | - | - | 10.18 |
| stm_ltm | stm_only | lower | 1 | 24 | 21 | 0.5238 | 0.4679 | 0.4762 | 0.4762 | 0.0000 | - | - | - | 0.45833 | 0.45833 | 3 | 3 | 0.5000 | 3 | 1 | 10 | - | - | - | 8.84 |
| stm_ltm | stm_only | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 9.11 |
| stm_ltm | stm_only | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 8.08 |
| stm_ltm | stm_only | upper | 1 | 24 | 21 | 0.9524 | 0.8536 | 0.0476 | 0.0476 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9091 | 2 | 2 | 10 | - | - | - | 8.03 |
| target_parts | parts_all | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 15.24 |
| target_parts | parts_all | left | 1 | 24 | 21 | 0.0476 | 0.0492 | 0.9524 | 0.4762 | 0.0000 | - | - | - | 0.41667 | 0.83333 | 1 | 1 | 0.0909 | 1 | 0 | 1 | - | - | - | 10.32 |
| target_parts | parts_all | lower | 1 | 24 | 21 | 0.1429 | 0.1944 | 0.8571 | 0.1905 | 1.0000 | 2.00 | 2.00 | 2.00 | 0.08333 | 0.28571 | 1 | 1 | 0.6000 | 2 | 2 | 1 | - | - | - | 11.42 |
| target_parts | parts_all | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 12.85 |
| target_parts | parts_all | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 10.66 |
| target_parts | parts_all | upper | 1 | 24 | 21 | 0.9048 | 0.8159 | 0.0952 | 0.0952 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.08333 | 0.08333 | 1 | 1 | 0.9048 | 2 | 2 | 10 | - | - | - | 10.68 |
| target_parts | parts_full_only | edge | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 6.97 |
| target_parts | parts_full_only | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 0.12500 | 1 | 1 | 0.8750 | 2 | 1 | 12 | - | - | - | 6.29 |
| target_parts | parts_full_only | lower | 1 | 24 | 21 | 0.9524 | 0.8536 | 0.0476 | 0.0476 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 10 | - | - | - | 6.65 |
| target_parts | parts_full_only | none | 1 | 24 | 21 | 0.6190 | 0.5438 | 0.3810 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 0.50000 | 1 | 1 | 0.5000 | 1 | 1 | 12 | - | - | - | 7.31 |
| target_parts | parts_full_only | right | 1 | 24 | 21 | 0.0952 | 0.0909 | 0.9048 | 0.4286 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.33333 | 0.66667 | 2 | 2 | 0.2000 | 3 | 1 | 1 | - | - | - | 5.03 |
| target_parts | parts_full_only | upper | 1 | 24 | 21 | 0.9524 | 0.8536 | 0.0476 | 0.0476 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 10 | - | - | - | 5.89 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| synth-stress | 1 | observable | partial-view | lower | 13x85 | 21 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 0.6190 | 0.6190 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 9.92 | 9.67 |
| left | 0.0476 | 0.0476 | +0.0000 | 0.41667 | 0.41667 | +0.00000 | 6.62 | 7.02 |
| lower | 0.1429 | 0.1429 | +0.0000 | 0.08333 | 0.08333 | +0.00000 | 8.47 | 9.21 |
| none | 0.6190 | 0.6190 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 9.53 | 9.28 |
| right | 0.0952 | 0.0952 | +0.0000 | 0.33333 | 0.33333 | +0.00000 | 7.14 | 7.11 |
| upper | 0.9048 | 0.9048 | +0.0000 | 0.08333 | 0.08333 | +0.00000 | 8.03 | 7.74 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.69 vs 9.92 (-0.23)
- [distractor_margin] margin_0.00 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.95833 vs 0.41667 (+0.54167) | FA events 1 vs 1 (+0) | precision 0.0738 vs 0.0492 | ms/frame 8.10 vs 6.62 (+1.48)
- [distractor_margin] margin_0.00 (mask=lower): TSR 0.5238 vs 0.1429 (+0.3810) | FAR 0.50000 vs 0.08333 (+0.41667) | FA events 2 vs 1 (+1) | precision 0.4852 vs 0.1944 | ms/frame 9.33 vs 8.47 (+0.86)
- [distractor_margin] margin_0.00 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 10.11 vs 9.53 (+0.58)
- [distractor_margin] margin_0.00 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.37500 vs 0.33333 (+0.04167) | FA events 2 vs 2 (+0) | precision 0.1048 vs 0.0909 | ms/frame 7.98 vs 7.14 (+0.84)
- [distractor_margin] margin_0.00 (mask=upper): TSR 0.9524 vs 0.9048 (+0.0476) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8536 vs 0.8159 | ms/frame 8.02 vs 8.03 (-0.00)
- [distractor_margin] margin_0.06 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.72 vs 9.92 (-0.19)
- [distractor_margin] margin_0.06 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 6.48 vs 6.62 (-0.14)
- [distractor_margin] margin_0.06 (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 8.71 vs 8.47 (+0.25)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.09 vs 9.53 (-0.44)
- [distractor_margin] margin_0.06 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 6.69 vs 7.14 (-0.45)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 8.20 vs 8.03 (+0.17)
- [distractor_margin] margin_0.12 (mask=edge): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8469 vs 0.5438 | ms/frame 8.93 vs 9.92 (-0.99)
- [distractor_margin] margin_0.12 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 6.41 vs 6.62 (-0.21)
- [distractor_margin] margin_0.12 (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 8.27 vs 8.47 (-0.20)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8469 vs 0.5438 | ms/frame 8.55 vs 9.53 (-0.97)
- [distractor_margin] margin_0.12 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.29167 vs 0.33333 (-0.04167) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 7.11 vs 7.14 (-0.02)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 8.09 vs 8.03 (+0.06)
- [freeze] freeze_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.84 vs 9.92 (+4.93)
- [freeze] freeze_off (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.89 vs 6.62 (+4.27)
- [freeze] freeze_off (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 16.11 vs 8.47 (+7.65)
- [freeze] freeze_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.97 vs 9.53 (+4.45)
- [freeze] freeze_off (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.78 vs 7.14 (+3.64)
- [freeze] freeze_off (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 12.95 vs 8.03 (+4.92)
- [freeze] freeze_on (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 16.62 vs 9.92 (+6.70)
- [freeze] freeze_on (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.10 vs 6.62 (+3.48)
- [freeze] freeze_on (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 13.74 vs 8.47 (+5.27)
- [freeze] freeze_on (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 15.33 vs 9.53 (+5.81)
- [freeze] freeze_on (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 11.08 vs 7.14 (+3.94)
- [freeze] freeze_on (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 12.27 vs 8.03 (+4.25)
- [gate] gate_growth_unfrozen (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 15.56 vs 9.92 (+5.64)
- [gate] gate_growth_unfrozen (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.37 vs 6.62 (+3.75)
- [gate] gate_growth_unfrozen (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 12.81 vs 8.47 (+4.34)
- [gate] gate_growth_unfrozen (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.85 vs 9.53 (+4.32)
- [gate] gate_growth_unfrozen (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.99 vs 7.14 (+3.85)
- [gate] gate_growth_unfrozen (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.70 vs 8.03 (+3.67)
- [gate] gate_legacy (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.29 vs 9.92 (+4.38)
- [gate] gate_legacy (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.00000 vs 0.41667 (-0.41667) | FA events 0 vs 1 (-1) | precision 0.0492 vs 0.0492 | ms/frame 8.25 vs 6.62 (+1.63)
- [gate] gate_legacy (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1917 vs 0.1944 | ms/frame 12.44 vs 8.47 (+3.98)
- [gate] gate_legacy (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 12.95 vs 9.53 (+3.42)
- [gate] gate_legacy (mask=right): TSR 0.0476 vs 0.0952 (-0.0476) | FAR 0.00000 vs 0.33333 (-0.33333) | FA events 0 vs 2 (-2) | precision 0.0492 vs 0.0909 | ms/frame 7.68 vs 7.14 (+0.55)
- [gate] gate_legacy (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.80 vs 8.03 (+3.78)
- [gate] gate_search_early (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.55 vs 9.92 (+4.64)
- [gate] gate_search_early (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 9.18 vs 6.62 (+2.57)
- [gate] gate_search_early (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 14.39 vs 8.47 (+5.92)
- [gate] gate_search_early (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.35 vs 9.53 (+3.83)
- [gate] gate_search_early (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 11.99 vs 7.14 (+4.85)
- [gate] gate_search_early (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.80 vs 8.03 (+3.78)
- [gate] gate_search_late (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 15.17 vs 9.92 (+5.25)
- [gate] gate_search_late (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.00000 vs 0.41667 (-0.41667) | FA events 0 vs 1 (-1) | precision 0.0492 vs 0.0492 | ms/frame 7.64 vs 6.62 (+1.02)
- [gate] gate_search_late (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1917 vs 0.1944 | ms/frame 13.18 vs 8.47 (+4.71)
- [gate] gate_search_late (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.59 vs 9.53 (+4.07)
- [gate] gate_search_late (mask=right): TSR 0.0476 vs 0.0952 (-0.0476) | FAR 0.00000 vs 0.33333 (-0.33333) | FA events 0 vs 2 (-2) | precision 0.0492 vs 0.0909 | ms/frame 7.64 vs 7.14 (+0.50)
- [gate] gate_search_late (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.46 vs 8.03 (+3.44)
- [hard_negative] hard_negative_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.75 vs 9.92 (-0.17)
- [hard_negative] hard_negative_off (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.87500 vs 0.41667 (+0.45833) | FA events 1 vs 1 (+0) | precision 0.0511 vs 0.0492 | ms/frame 7.67 vs 6.62 (+1.05)
- [hard_negative] hard_negative_off (mask=lower): TSR 0.0476 vs 0.1429 (-0.0952) | FAR 0.95833 vs 0.08333 (+0.87500) | FA events 1 vs 1 (+0) | precision 0.0870 vs 0.1944 | ms/frame 8.10 vs 8.47 (-0.36)
- [hard_negative] hard_negative_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.05 vs 9.53 (-0.48)
- [hard_negative] hard_negative_off (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 6.92 vs 7.14 (-0.21)
- [hard_negative] hard_negative_off (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 7.37 vs 8.03 (-0.66)
- [ltm_capacity] ltm_cap_1 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.77 vs 9.92 (+4.85)
- [ltm_capacity] ltm_cap_1 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.18 vs 6.62 (+3.56)
- [ltm_capacity] ltm_cap_1 (mask=lower): TSR 0.4286 vs 0.1429 (+0.2857) | FAR 0.00000 vs 0.08333 (-0.08333) | FA events 0 vs 1 (-1) | precision 0.4306 vs 0.1944 | ms/frame 11.38 vs 8.47 (+2.91)
- [ltm_capacity] ltm_cap_1 (mask=none): TSR 0.0952 vs 0.6190 (-0.5238) | FAR 0.00000 vs 0.50000 (-0.50000) | FA events 0 vs 1 (-1) | precision 0.1389 vs 0.5438 | ms/frame 11.80 vs 9.53 (+2.28)
- [ltm_capacity] ltm_cap_1 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.60 vs 7.14 (+3.47)
- [ltm_capacity] ltm_cap_1 (mask=upper): TSR 0.4286 vs 0.9048 (-0.4762) | FAR 0.00000 vs 0.08333 (-0.08333) | FA events 0 vs 1 (-1) | precision 0.4306 vs 0.8159 | ms/frame 8.18 vs 8.03 (+0.15)
- [ltm_capacity] ltm_cap_384 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 15.60 vs 9.92 (+5.69)
- [ltm_capacity] ltm_cap_384 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.34 vs 6.62 (+3.73)
- [ltm_capacity] ltm_cap_384 (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 13.34 vs 8.47 (+4.87)
- [ltm_capacity] ltm_cap_384 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.00 vs 9.53 (+4.47)
- [ltm_capacity] ltm_cap_384 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.96 vs 7.14 (+3.83)
- [ltm_capacity] ltm_cap_384 (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.48 vs 8.03 (+3.45)
- [ltm_capacity] ltm_cap_96 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.70 vs 9.92 (+4.78)
- [ltm_capacity] ltm_cap_96 (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.05 vs 6.62 (+3.43)
- [ltm_capacity] ltm_cap_96 (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 14.73 vs 8.47 (+6.26)
- [ltm_capacity] ltm_cap_96 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.94 vs 9.53 (+4.42)
- [ltm_capacity] ltm_cap_96 (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.58 vs 7.14 (+3.45)
- [ltm_capacity] ltm_cap_96 (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.89 vs 8.03 (+3.87)
- [part_aware] part_aware_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 10.36 vs 9.92 (+0.44)
- [part_aware] part_aware_off (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.95833 vs 0.41667 (+0.54167) | FA events 1 vs 1 (+0) | precision 0.0738 vs 0.0492 | ms/frame 8.38 vs 6.62 (+1.76)
- [part_aware] part_aware_off (mask=lower): TSR 0.0952 vs 0.1429 (-0.0476) | FAR 0.12500 vs 0.08333 (+0.04167) | FA events 2 vs 1 (+1) | precision 0.1423 vs 0.1944 | ms/frame 9.15 vs 8.47 (+0.68)
- [part_aware] part_aware_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.18 vs 9.53 (-0.34)
- [part_aware] part_aware_off (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 7.38 vs 7.14 (+0.25)
- [part_aware] part_aware_off (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 8.19 vs 8.03 (+0.17)
- [part_matching] part_matching_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.67 vs 9.92 (-0.25)
- [part_matching] part_matching_off (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 7.02 vs 6.62 (+0.40)
- [part_matching] part_matching_off (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 9.21 vs 8.47 (+0.75)
- [part_matching] part_matching_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.28 vs 9.53 (-0.24)
- [part_matching] part_matching_off (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 7.11 vs 7.14 (-0.02)
- [part_matching] part_matching_off (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 7.74 vs 8.03 (-0.29)
- [stm_ltm] balanced (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 14.01 vs 9.92 (+4.09)
- [stm_ltm] balanced (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 9.36 vs 6.62 (+2.74)
- [stm_ltm] balanced (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 12.63 vs 8.47 (+4.16)
- [stm_ltm] balanced (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 12.60 vs 9.53 (+3.08)
- [stm_ltm] balanced (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 9.98 vs 7.14 (+2.84)
- [stm_ltm] balanced (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 11.64 vs 8.03 (+3.61)
- [stm_ltm] ltm_only (mask=edge): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8469 vs 0.5438 | ms/frame 14.57 vs 9.92 (+4.66)
- [stm_ltm] ltm_only (mask=left): TSR 1.0000 vs 0.0476 (+0.9524) | FAR 0.12500 vs 0.41667 (-0.29167) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.0492 | ms/frame 13.55 vs 6.62 (+6.93)
- [stm_ltm] ltm_only (mask=lower): TSR 0.0476 vs 0.1429 (-0.0952) | FAR 0.95833 vs 0.08333 (+0.87500) | FA events 1 vs 1 (+0) | precision 0.0870 vs 0.1944 | ms/frame 11.27 vs 8.47 (+2.81)
- [stm_ltm] ltm_only (mask=none): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8469 vs 0.5438 | ms/frame 12.65 vs 9.53 (+3.12)
- [stm_ltm] ltm_only (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 11.28 vs 7.14 (+4.14)
- [stm_ltm] ltm_only (mask=upper): TSR 0.5238 vs 0.9048 (-0.3810) | FAR 0.41667 vs 0.08333 (+0.33333) | FA events 2 vs 1 (+1) | precision 0.4786 vs 0.8159 | ms/frame 12.52 vs 8.03 (+4.49)
- [stm_ltm] stm_only (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 13.22 vs 9.92 (+3.31)
- [stm_ltm] stm_only (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.95833 vs 0.41667 (+0.54167) | FA events 1 vs 1 (+0) | precision 0.0738 vs 0.0492 | ms/frame 10.18 vs 6.62 (+3.56)
- [stm_ltm] stm_only (mask=lower): TSR 0.5238 vs 0.1429 (+0.3810) | FAR 0.45833 vs 0.08333 (+0.37500) | FA events 3 vs 1 (+2) | precision 0.4679 vs 0.1944 | ms/frame 8.84 vs 8.47 (+0.37)
- [stm_ltm] stm_only (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 9.11 vs 9.53 (-0.42)
- [stm_ltm] stm_only (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 8.08 vs 7.14 (+0.94)
- [stm_ltm] stm_only (mask=upper): TSR 0.9524 vs 0.9048 (+0.0476) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8536 vs 0.8159 | ms/frame 8.03 vs 8.03 (+0.00)
- [target_parts] parts_all (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 15.24 vs 9.92 (+5.33)
- [target_parts] parts_all (mask=left): TSR 0.0476 vs 0.0476 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.0492 vs 0.0492 | ms/frame 10.32 vs 6.62 (+3.70)
- [target_parts] parts_all (mask=lower): TSR 0.1429 vs 0.1429 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.1944 vs 0.1944 | ms/frame 11.42 vs 8.47 (+2.95)
- [target_parts] parts_all (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 12.85 vs 9.53 (+3.32)
- [target_parts] parts_all (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 10.66 vs 7.14 (+3.52)
- [target_parts] parts_all (mask=upper): TSR 0.9048 vs 0.9048 (+0.0000) | FAR 0.08333 vs 0.08333 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8159 vs 0.8159 | ms/frame 10.68 vs 8.03 (+2.65)
- [target_parts] parts_full_only (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 6.97 vs 9.92 (-2.95)
- [target_parts] parts_full_only (mask=left): TSR 1.0000 vs 0.0476 (+0.9524) | FAR 0.12500 vs 0.41667 (-0.29167) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.0492 | ms/frame 6.29 vs 6.62 (-0.33)
- [target_parts] parts_full_only (mask=lower): TSR 0.9524 vs 0.1429 (+0.8095) | FAR 0.00000 vs 0.08333 (-0.08333) | FA events 0 vs 1 (-1) | precision 0.8536 vs 0.1944 | ms/frame 6.65 vs 8.47 (-1.82)
- [target_parts] parts_full_only (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5438 vs 0.5438 | ms/frame 7.31 vs 9.53 (-2.22)
- [target_parts] parts_full_only (mask=right): TSR 0.0952 vs 0.0952 (+0.0000) | FAR 0.33333 vs 0.33333 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.0909 vs 0.0909 | ms/frame 5.03 vs 7.14 (-2.11)
- [target_parts] parts_full_only (mask=upper): TSR 0.9524 vs 0.9048 (+0.0476) | FAR 0.00000 vs 0.08333 (-0.08333) | FA events 0 vs 1 (-1) | precision 0.8536 vs 0.8159 | ms/frame 5.89 vs 8.03 (-2.13)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth-stress | baseline | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.53 |
| synth-stress | baseline | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 8.47 |
| synth-stress | baseline | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 8.03 |
| synth-stress | baseline | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 6.62 |
| synth-stress | baseline | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 7.14 |
| synth-stress | baseline | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.92 |
| synth-stress | part_matching_off | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.28 |
| synth-stress | part_matching_off | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 9.21 |
| synth-stress | part_matching_off | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 7.74 |
| synth-stress | part_matching_off | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 7.02 |
| synth-stress | part_matching_off | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 7.11 |
| synth-stress | part_matching_off | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.67 |
| synth-stress | part_aware_off | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.18 |
| synth-stress | part_aware_off | lower | 1 | 24 | 0.0952 | 0.1423 | 0.12500 | 2 | 0.4000 | 3 | 9.15 |
| synth-stress | part_aware_off | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 8.19 |
| synth-stress | part_aware_off | left | 1 | 24 | 0.0476 | 0.0738 | 0.95833 | 1 | 0.0417 | 1 | 8.38 |
| synth-stress | part_aware_off | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 7.38 |
| synth-stress | part_aware_off | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 10.36 |
| synth-stress | margin_0.00 | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 10.11 |
| synth-stress | margin_0.00 | lower | 1 | 24 | 0.5238 | 0.4852 | 0.50000 | 2 | 0.4783 | 3 | 9.33 |
| synth-stress | margin_0.00 | upper | 1 | 24 | 0.9524 | 0.8536 | 0.08333 | 1 | 0.9091 | 2 | 8.02 |
| synth-stress | margin_0.00 | left | 1 | 24 | 0.0476 | 0.0738 | 0.95833 | 1 | 0.0417 | 1 | 8.10 |
| synth-stress | margin_0.00 | right | 1 | 24 | 0.0952 | 0.1048 | 0.37500 | 2 | 0.1818 | 3 | 7.98 |
| synth-stress | margin_0.00 | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.69 |
| synth-stress | margin_0.06 | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.09 |
| synth-stress | margin_0.06 | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 8.71 |
| synth-stress | margin_0.06 | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 8.20 |
| synth-stress | margin_0.06 | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 6.48 |
| synth-stress | margin_0.06 | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 6.69 |
| synth-stress | margin_0.06 | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.72 |
| synth-stress | margin_0.12 | none | 1 | 24 | 0.9524 | 0.8469 | 0.12500 | 1 | 0.8696 | 2 | 8.55 |
| synth-stress | margin_0.12 | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 8.27 |
| synth-stress | margin_0.12 | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 8.09 |
| synth-stress | margin_0.12 | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 6.41 |
| synth-stress | margin_0.12 | right | 1 | 24 | 0.0952 | 0.0909 | 0.29167 | 2 | 0.2222 | 3 | 7.11 |
| synth-stress | margin_0.12 | edge | 1 | 24 | 0.9524 | 0.8469 | 0.12500 | 1 | 0.8696 | 2 | 8.93 |
| synth-stress | hard_negative_off | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.05 |
| synth-stress | hard_negative_off | lower | 1 | 24 | 0.0476 | 0.0870 | 0.95833 | 1 | 0.0417 | 1 | 8.10 |
| synth-stress | hard_negative_off | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 7.37 |
| synth-stress | hard_negative_off | left | 1 | 24 | 0.0476 | 0.0511 | 0.87500 | 1 | 0.0455 | 1 | 7.67 |
| synth-stress | hard_negative_off | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 6.92 |
| synth-stress | hard_negative_off | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.75 |
| synth-stress | stm_only | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 9.11 |
| synth-stress | stm_only | lower | 1 | 24 | 0.5238 | 0.4679 | 0.45833 | 3 | 0.5000 | 3 | 8.84 |
| synth-stress | stm_only | upper | 1 | 24 | 0.9524 | 0.8536 | 0.08333 | 1 | 0.9091 | 2 | 8.03 |
| synth-stress | stm_only | left | 1 | 24 | 0.0476 | 0.0738 | 0.95833 | 1 | 0.0417 | 1 | 10.18 |
| synth-stress | stm_only | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 8.08 |
| synth-stress | stm_only | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.22 |
| synth-stress | balanced | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 12.60 |
| synth-stress | balanced | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 12.63 |
| synth-stress | balanced | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.64 |
| synth-stress | balanced | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 9.36 |
| synth-stress | balanced | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 9.98 |
| synth-stress | balanced | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.01 |
| synth-stress | ltm_only | none | 1 | 24 | 0.9524 | 0.8469 | 0.12500 | 1 | 0.8696 | 2 | 12.65 |
| synth-stress | ltm_only | lower | 1 | 24 | 0.0476 | 0.0870 | 0.95833 | 1 | 0.0417 | 1 | 11.27 |
| synth-stress | ltm_only | upper | 1 | 24 | 0.5238 | 0.4786 | 0.41667 | 2 | 0.5238 | 2 | 12.52 |
| synth-stress | ltm_only | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 13.55 |
| synth-stress | ltm_only | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 11.28 |
| synth-stress | ltm_only | edge | 1 | 24 | 0.9524 | 0.8469 | 0.12500 | 1 | 0.8696 | 2 | 14.57 |
| synth-stress | ltm_cap_1 | none | 1 | 24 | 0.0952 | 0.1389 | 0.00000 | 0 | 1.0000 | 0 | 11.80 |
| synth-stress | ltm_cap_1 | lower | 1 | 24 | 0.4286 | 0.4306 | 0.00000 | 0 | 1.0000 | 0 | 11.38 |
| synth-stress | ltm_cap_1 | upper | 1 | 24 | 0.4286 | 0.4306 | 0.00000 | 0 | 1.0000 | 0 | 8.18 |
| synth-stress | ltm_cap_1 | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.18 |
| synth-stress | ltm_cap_1 | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.60 |
| synth-stress | ltm_cap_1 | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.77 |
| synth-stress | ltm_cap_96 | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.94 |
| synth-stress | ltm_cap_96 | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 14.73 |
| synth-stress | ltm_cap_96 | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.89 |
| synth-stress | ltm_cap_96 | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.05 |
| synth-stress | ltm_cap_96 | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.58 |
| synth-stress | ltm_cap_96 | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.70 |
| synth-stress | ltm_cap_384 | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.00 |
| synth-stress | ltm_cap_384 | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 13.34 |
| synth-stress | ltm_cap_384 | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.48 |
| synth-stress | ltm_cap_384 | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.34 |
| synth-stress | ltm_cap_384 | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.96 |
| synth-stress | ltm_cap_384 | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 15.60 |
| synth-stress | freeze_on | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 15.33 |
| synth-stress | freeze_on | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 13.74 |
| synth-stress | freeze_on | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 12.27 |
| synth-stress | freeze_on | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.10 |
| synth-stress | freeze_on | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 11.08 |
| synth-stress | freeze_on | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 16.62 |
| synth-stress | freeze_off | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.97 |
| synth-stress | freeze_off | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 16.11 |
| synth-stress | freeze_off | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 12.95 |
| synth-stress | freeze_off | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.89 |
| synth-stress | freeze_off | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.78 |
| synth-stress | freeze_off | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.84 |
| synth-stress | parts_full_only | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 7.31 |
| synth-stress | parts_full_only | lower | 1 | 24 | 0.9524 | 0.8536 | 0.00000 | 0 | 1.0000 | 0 | 6.65 |
| synth-stress | parts_full_only | upper | 1 | 24 | 0.9524 | 0.8536 | 0.00000 | 0 | 1.0000 | 0 | 5.89 |
| synth-stress | parts_full_only | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 6.29 |
| synth-stress | parts_full_only | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 5.03 |
| synth-stress | parts_full_only | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 6.97 |
| synth-stress | parts_all | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 12.85 |
| synth-stress | parts_all | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 11.42 |
| synth-stress | parts_all | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 10.68 |
| synth-stress | parts_all | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.32 |
| synth-stress | parts_all | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.66 |
| synth-stress | parts_all | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 15.24 |
| synth-stress | gate_legacy | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 12.95 |
| synth-stress | gate_legacy | lower | 1 | 24 | 0.1429 | 0.1917 | 0.08333 | 1 | 0.6000 | 2 | 12.44 |
| synth-stress | gate_legacy | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.80 |
| synth-stress | gate_legacy | left | 1 | 24 | 0.0476 | 0.0492 | 0.00000 | 0 | 1.0000 | 0 | 8.25 |
| synth-stress | gate_legacy | right | 1 | 24 | 0.0476 | 0.0492 | 0.00000 | 0 | 1.0000 | 0 | 7.68 |
| synth-stress | gate_legacy | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.29 |
| synth-stress | gate_search_early | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.35 |
| synth-stress | gate_search_early | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 14.39 |
| synth-stress | gate_search_early | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.80 |
| synth-stress | gate_search_early | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 9.18 |
| synth-stress | gate_search_early | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 11.99 |
| synth-stress | gate_search_early | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 14.55 |
| synth-stress | gate_search_late | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.59 |
| synth-stress | gate_search_late | lower | 1 | 24 | 0.1429 | 0.1917 | 0.08333 | 1 | 0.6000 | 2 | 13.18 |
| synth-stress | gate_search_late | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.46 |
| synth-stress | gate_search_late | left | 1 | 24 | 0.0476 | 0.0492 | 0.00000 | 0 | 1.0000 | 0 | 7.64 |
| synth-stress | gate_search_late | right | 1 | 24 | 0.0476 | 0.0492 | 0.00000 | 0 | 1.0000 | 0 | 7.64 |
| synth-stress | gate_search_late | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 15.17 |
| synth-stress | gate_growth_unfrozen | none | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 13.85 |
| synth-stress | gate_growth_unfrozen | lower | 1 | 24 | 0.1429 | 0.1944 | 0.08333 | 1 | 0.6000 | 2 | 12.81 |
| synth-stress | gate_growth_unfrozen | upper | 1 | 24 | 0.9048 | 0.8159 | 0.08333 | 1 | 0.9048 | 2 | 11.70 |
| synth-stress | gate_growth_unfrozen | left | 1 | 24 | 0.0476 | 0.0492 | 0.41667 | 1 | 0.0909 | 1 | 10.37 |
| synth-stress | gate_growth_unfrozen | right | 1 | 24 | 0.0952 | 0.0909 | 0.33333 | 2 | 0.2000 | 3 | 10.99 |
| synth-stress | gate_growth_unfrozen | edge | 1 | 24 | 0.6190 | 0.5438 | 0.50000 | 1 | 0.5000 | 1 | 15.56 |
