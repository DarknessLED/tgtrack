# Clicked-target strategy ablation

Config: ablate=all, data_root=data/sequences, detector=gt, dry_run=True, embedder=fallback, jobs=4, masks=['none', 'lower', 'upper', 'left', 'right', 'edge'], max_frames=24, subsample=1, target_ids={'synth-stress': 1}, target_strategy=stressed, wall_seconds=11.34

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FA_ev | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.76 |
| baseline | baseline | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.95 |
| baseline | baseline | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.86 |
| baseline | baseline | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 13.28 |
| baseline | baseline | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 14.55 |
| baseline | baseline | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.23 |
| distractor_margin | margin_0.00 | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.68 |
| distractor_margin | margin_0.00 | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 19.53 |
| distractor_margin | margin_0.00 | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.50 |
| distractor_margin | margin_0.00 | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.86 |
| distractor_margin | margin_0.00 | right | 1 | 24 | 21 | 0.6190 | 0.5551 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.45833 | 2 | 0.5417 | 3 | 1 | 12 | 14.53 |
| distractor_margin | margin_0.00 | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.23 |
| distractor_margin | margin_0.06 | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.68 |
| distractor_margin | margin_0.06 | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 14.52 |
| distractor_margin | margin_0.06 | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.23 |
| distractor_margin | margin_0.06 | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 17.33 |
| distractor_margin | margin_0.06 | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.35 |
| distractor_margin | margin_0.06 | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.95 |
| distractor_margin | margin_0.12 | edge | 1 | 24 | 21 | 0.9524 | 0.8443 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 1 | 0.8696 | 2 | 1 | 12 | 15.49 |
| distractor_margin | margin_0.12 | left | 1 | 24 | 21 | 0.9048 | 0.7937 | 0.0952 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8636 | 2 | 2 | 12 | 13.62 |
| distractor_margin | margin_0.12 | lower | 1 | 24 | 21 | 0.5714 | 0.5114 | 0.4286 | 0.0000 | - | - | - | 0.45833 | 2 | 0.5217 | 1 | 0 | 12 | 15.12 |
| distractor_margin | margin_0.12 | none | 1 | 24 | 21 | 0.9524 | 0.8443 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 1 | 0.8696 | 2 | 1 | 12 | 14.32 |
| distractor_margin | margin_0.12 | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.37500 | 2 | 0.5909 | 3 | 1 | 12 | 14.68 |
| distractor_margin | margin_0.12 | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.49 |
| freeze | freeze_off | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.46 |
| freeze | freeze_off | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 13.89 |
| freeze | freeze_off | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 16.98 |
| freeze | freeze_off | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.83 |
| freeze | freeze_off | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.91 |
| freeze | freeze_off | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.55 |
| freeze | freeze_on | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 17.95 |
| freeze | freeze_on | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 18.09 |
| freeze | freeze_on | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.56 |
| freeze | freeze_on | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 14.80 |
| freeze | freeze_on | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 16.17 |
| freeze | freeze_on | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.75 |
| hard_negative | hard_negative_off | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 14.77 |
| hard_negative | hard_negative_off | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 14.31 |
| hard_negative | hard_negative_off | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.41 |
| hard_negative | hard_negative_off | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 14.02 |
| hard_negative | hard_negative_off | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 13.76 |
| hard_negative | hard_negative_off | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.25 |
| ltm_capacity | ltm_cap_1 | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.09 |
| ltm_capacity | ltm_cap_1 | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.61 |
| ltm_capacity | ltm_cap_1 | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 16.14 |
| ltm_capacity | ltm_cap_1 | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 13.91 |
| ltm_capacity | ltm_cap_1 | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.38 |
| ltm_capacity | ltm_cap_1 | upper | 1 | 24 | 21 | 0.5714 | 0.5351 | 0.4286 | 0.0000 | - | - | - | 0.16667 | 1 | 0.7500 | 1 | 0 | 12 | 15.19 |
| ltm_capacity | ltm_cap_384 | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.29 |
| ltm_capacity | ltm_cap_384 | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 14.67 |
| ltm_capacity | ltm_cap_384 | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 16.05 |
| ltm_capacity | ltm_cap_384 | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 17.11 |
| ltm_capacity | ltm_cap_384 | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.40 |
| ltm_capacity | ltm_cap_384 | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 16.03 |
| ltm_capacity | ltm_cap_96 | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 18.01 |
| ltm_capacity | ltm_cap_96 | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 16.38 |
| ltm_capacity | ltm_cap_96 | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.72 |
| ltm_capacity | ltm_cap_96 | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.45 |
| ltm_capacity | ltm_cap_96 | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 17.67 |
| ltm_capacity | ltm_cap_96 | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.42 |
| part_aware | part_aware_off | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.43 |
| part_aware | part_aware_off | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.69 |
| part_aware | part_aware_off | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.30 |
| part_aware | part_aware_off | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.97 |
| part_aware | part_aware_off | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 14.50 |
| part_aware | part_aware_off | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.03 |
| part_matching | part_matching_off | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 19.20 |
| part_matching | part_matching_off | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.76 |
| part_matching | part_matching_off | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.96 |
| part_matching | part_matching_off | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.65 |
| part_matching | part_matching_off | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.95 |
| part_matching | part_matching_off | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.23 |
| stm_ltm | balanced | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.43 |
| stm_ltm | balanced | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.21 |
| stm_ltm | balanced | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.99 |
| stm_ltm | balanced | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 14.81 |
| stm_ltm | balanced | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.45 |
| stm_ltm | balanced | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.36 |
| stm_ltm | ltm_only | edge | 1 | 24 | 21 | 0.9524 | 0.8443 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 1 | 0.8696 | 2 | 1 | 12 | 15.28 |
| stm_ltm | ltm_only | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 14.91 |
| stm_ltm | ltm_only | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.94 |
| stm_ltm | ltm_only | none | 1 | 24 | 21 | 0.9524 | 0.8443 | 0.0476 | 1.0000 | 1.00 | 1.00 | 1.00 | 0.12500 | 1 | 0.8696 | 2 | 1 | 12 | 15.14 |
| stm_ltm | ltm_only | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 14.53 |
| stm_ltm | ltm_only | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.73 |
| stm_ltm | stm_only | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 15.31 |
| stm_ltm | stm_only | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 14.21 |
| stm_ltm | stm_only | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 15.59 |
| stm_ltm | stm_only | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 14.35 |
| stm_ltm | stm_only | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 15.28 |
| stm_ltm | stm_only | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.11 |
| target_parts | parts_all | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 13.40 |
| target_parts | parts_all | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 15.06 |
| target_parts | parts_all | lower | 1 | 24 | 21 | 0.5714 | 0.5213 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 17.11 |
| target_parts | parts_all | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 16.07 |
| target_parts | parts_all | right | 1 | 24 | 21 | 0.6190 | 0.5417 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.41667 | 2 | 0.5652 | 3 | 1 | 12 | 12.83 |
| target_parts | parts_all | upper | 1 | 24 | 21 | 0.5714 | 0.5246 | 0.4286 | 0.0000 | - | - | - | 0.50000 | 1 | 0.5000 | 1 | 0 | 12 | 14.46 |
| target_parts | parts_full_only | edge | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 8.62 |
| target_parts | parts_full_only | left | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.12500 | 1 | 0.8750 | 2 | 1 | 12 | 7.76 |
| target_parts | parts_full_only | lower | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.00000 | 0 | 1.0000 | 0 | 1 | 12 | 7.50 |
| target_parts | parts_full_only | none | 1 | 24 | 21 | 0.6190 | 0.5418 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.50000 | 1 | 0.5000 | 1 | 1 | 12 | 8.08 |
| target_parts | parts_full_only | right | 1 | 24 | 21 | 0.6190 | 0.5551 | 0.3810 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.45833 | 2 | 0.5417 | 3 | 1 | 12 | 7.72 |
| target_parts | parts_full_only | upper | 1 | 24 | 21 | 1.0000 | 0.8750 | 0.0000 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.04167 | 1 | 0.9545 | 2 | 1 | 12 | 7.82 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 0.6190 | 0.6190 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 15.76 | 19.20 |
| left | 1.0000 | 1.0000 | +0.0000 | 0.12500 | 0.12500 | +0.00000 | 15.95 | 15.76 |
| lower | 0.5714 | 0.5714 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 14.86 | 14.96 |
| none | 0.6190 | 0.6190 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 13.28 | 15.65 |
| right | 0.6190 | 0.6190 | +0.0000 | 0.41667 | 0.41667 | +0.00000 | 14.55 | 15.95 |
| upper | 0.5714 | 0.5714 | +0.0000 | 0.50000 | 0.50000 | +0.00000 | 14.23 | 14.23 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.68 vs 15.76 (-0.07)
- [distractor_margin] margin_0.00 (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 19.53 vs 15.95 (+3.57)
- [distractor_margin] margin_0.00 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.50 vs 14.86 (+2.64)
- [distractor_margin] margin_0.00 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.86 vs 13.28 (+2.58)
- [distractor_margin] margin_0.00 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.45833 vs 0.41667 (+0.04167) | FA events 2 vs 2 (+0) | precision 0.5551 vs 0.5417 | ms/frame 14.53 vs 14.55 (-0.02)
- [distractor_margin] margin_0.00 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 15.23 vs 14.23 (+1.00)
- [distractor_margin] margin_0.06 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.68 vs 15.76 (+0.93)
- [distractor_margin] margin_0.06 (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 14.52 vs 15.95 (-1.43)
- [distractor_margin] margin_0.06 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.23 vs 14.86 (+2.38)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 17.33 vs 13.28 (+4.05)
- [distractor_margin] margin_0.06 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.35 vs 14.55 (+0.80)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.95 vs 14.23 (+0.72)
- [distractor_margin] margin_0.12 (mask=edge): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8443 vs 0.5418 | ms/frame 15.49 vs 15.76 (-0.26)
- [distractor_margin] margin_0.12 (mask=left): TSR 0.9048 vs 1.0000 (-0.0952) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.7937 vs 0.8750 | ms/frame 13.62 vs 15.95 (-2.34)
- [distractor_margin] margin_0.12 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.45833 vs 0.50000 (-0.04167) | FA events 2 vs 1 (+1) | precision 0.5114 vs 0.5213 | ms/frame 15.12 vs 14.86 (+0.27)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8443 vs 0.5418 | ms/frame 14.32 vs 13.28 (+1.04)
- [distractor_margin] margin_0.12 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.37500 vs 0.41667 (-0.04167) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 14.68 vs 14.55 (+0.13)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.49 vs 14.23 (+0.26)
- [freeze] freeze_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.46 vs 15.76 (-0.30)
- [freeze] freeze_off (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 13.89 vs 15.95 (-2.06)
- [freeze] freeze_off (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 16.98 vs 14.86 (+2.12)
- [freeze] freeze_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.83 vs 13.28 (+3.55)
- [freeze] freeze_off (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.91 vs 14.55 (+1.36)
- [freeze] freeze_off (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.55 vs 14.23 (+0.32)
- [freeze] freeze_on (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 17.95 vs 15.76 (+2.19)
- [freeze] freeze_on (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 18.09 vs 15.95 (+2.13)
- [freeze] freeze_on (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.56 vs 14.86 (+2.70)
- [freeze] freeze_on (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 14.80 vs 13.28 (+1.52)
- [freeze] freeze_on (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 16.17 vs 14.55 (+1.62)
- [freeze] freeze_on (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 15.75 vs 14.23 (+1.52)
- [hard_negative] hard_negative_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 14.77 vs 15.76 (-0.98)
- [hard_negative] hard_negative_off (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 14.31 vs 15.95 (-1.64)
- [hard_negative] hard_negative_off (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 14.41 vs 14.86 (-0.44)
- [hard_negative] hard_negative_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 14.02 vs 13.28 (+0.74)
- [hard_negative] hard_negative_off (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 13.76 vs 14.55 (-0.79)
- [hard_negative] hard_negative_off (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.25 vs 14.23 (+0.02)
- [ltm_capacity] ltm_cap_1 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.09 vs 15.76 (+0.33)
- [ltm_capacity] ltm_cap_1 (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 15.61 vs 15.95 (-0.34)
- [ltm_capacity] ltm_cap_1 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 16.14 vs 14.86 (+1.29)
- [ltm_capacity] ltm_cap_1 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 13.91 vs 13.28 (+0.63)
- [ltm_capacity] ltm_cap_1 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.38 vs 14.55 (+0.83)
- [ltm_capacity] ltm_cap_1 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.16667 vs 0.50000 (-0.33333) | FA events 1 vs 1 (+0) | precision 0.5351 vs 0.5246 | ms/frame 15.19 vs 14.23 (+0.96)
- [ltm_capacity] ltm_cap_384 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.29 vs 15.76 (-0.46)
- [ltm_capacity] ltm_cap_384 (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 14.67 vs 15.95 (-1.29)
- [ltm_capacity] ltm_cap_384 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 16.05 vs 14.86 (+1.19)
- [ltm_capacity] ltm_cap_384 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 17.11 vs 13.28 (+3.83)
- [ltm_capacity] ltm_cap_384 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.40 vs 14.55 (+0.85)
- [ltm_capacity] ltm_cap_384 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 16.03 vs 14.23 (+1.80)
- [ltm_capacity] ltm_cap_96 (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 18.01 vs 15.76 (+2.25)
- [ltm_capacity] ltm_cap_96 (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 16.38 vs 15.95 (+0.43)
- [ltm_capacity] ltm_cap_96 (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.72 vs 14.86 (+2.86)
- [ltm_capacity] ltm_cap_96 (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.45 vs 13.28 (+3.17)
- [ltm_capacity] ltm_cap_96 (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 17.67 vs 14.55 (+3.12)
- [ltm_capacity] ltm_cap_96 (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 15.42 vs 14.23 (+1.19)
- [part_aware] part_aware_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.43 vs 15.76 (+0.67)
- [part_aware] part_aware_off (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 15.69 vs 15.95 (-0.26)
- [part_aware] part_aware_off (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.30 vs 14.86 (+2.45)
- [part_aware] part_aware_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.97 vs 13.28 (+3.69)
- [part_aware] part_aware_off (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 14.50 vs 14.55 (-0.05)
- [part_aware] part_aware_off (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 15.03 vs 14.23 (+0.80)
- [part_matching] part_matching_off (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 19.20 vs 15.76 (+3.45)
- [part_matching] part_matching_off (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 15.76 vs 15.95 (-0.19)
- [part_matching] part_matching_off (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 14.96 vs 14.86 (+0.11)
- [part_matching] part_matching_off (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.65 vs 13.28 (+2.37)
- [part_matching] part_matching_off (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.95 vs 14.55 (+1.40)
- [part_matching] part_matching_off (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.23 vs 14.23 (-0.00)
- [stm_ltm] balanced (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.43 vs 15.76 (+0.67)
- [stm_ltm] balanced (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 15.21 vs 15.95 (-0.74)
- [stm_ltm] balanced (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 15.99 vs 14.86 (+1.13)
- [stm_ltm] balanced (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 14.81 vs 13.28 (+1.53)
- [stm_ltm] balanced (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.45 vs 14.55 (+0.90)
- [stm_ltm] balanced (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.36 vs 14.23 (+0.13)
- [stm_ltm] ltm_only (mask=edge): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8443 vs 0.5418 | ms/frame 15.28 vs 15.76 (-0.48)
- [stm_ltm] ltm_only (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 14.91 vs 15.95 (-1.04)
- [stm_ltm] ltm_only (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 15.94 vs 14.86 (+1.08)
- [stm_ltm] ltm_only (mask=none): TSR 0.9524 vs 0.6190 (+0.3333) | FAR 0.12500 vs 0.50000 (-0.37500) | FA events 1 vs 1 (+0) | precision 0.8443 vs 0.5418 | ms/frame 15.14 vs 13.28 (+1.86)
- [stm_ltm] ltm_only (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 14.53 vs 14.55 (-0.02)
- [stm_ltm] ltm_only (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 15.73 vs 14.23 (+1.50)
- [stm_ltm] stm_only (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 15.31 vs 15.76 (-0.44)
- [stm_ltm] stm_only (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 14.21 vs 15.95 (-1.74)
- [stm_ltm] stm_only (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 15.59 vs 14.86 (+0.74)
- [stm_ltm] stm_only (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 14.35 vs 13.28 (+1.07)
- [stm_ltm] stm_only (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 15.28 vs 14.55 (+0.73)
- [stm_ltm] stm_only (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.11 vs 14.23 (-0.12)
- [target_parts] parts_all (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 13.40 vs 15.76 (-2.35)
- [target_parts] parts_all (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 15.06 vs 15.95 (-0.89)
- [target_parts] parts_all (mask=lower): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5213 vs 0.5213 | ms/frame 17.11 vs 14.86 (+2.25)
- [target_parts] parts_all (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 16.07 vs 13.28 (+2.79)
- [target_parts] parts_all (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.41667 vs 0.41667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.5417 vs 0.5417 | ms/frame 12.83 vs 14.55 (-1.72)
- [target_parts] parts_all (mask=upper): TSR 0.5714 vs 0.5714 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5246 vs 0.5246 | ms/frame 14.46 vs 14.23 (+0.23)
- [target_parts] parts_full_only (mask=edge): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 8.62 vs 15.76 (-7.14)
- [target_parts] parts_full_only (mask=left): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.12500 vs 0.12500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.8750 | ms/frame 7.76 vs 15.95 (-8.19)
- [target_parts] parts_full_only (mask=lower): TSR 1.0000 vs 0.5714 (+0.4286) | FAR 0.00000 vs 0.50000 (-0.50000) | FA events 0 vs 1 (-1) | precision 0.8750 vs 0.5213 | ms/frame 7.50 vs 14.86 (-7.36)
- [target_parts] parts_full_only (mask=none): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.50000 vs 0.50000 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.5418 vs 0.5418 | ms/frame 8.08 vs 13.28 (-5.20)
- [target_parts] parts_full_only (mask=right): TSR 0.6190 vs 0.6190 (+0.0000) | FAR 0.45833 vs 0.41667 (+0.04167) | FA events 2 vs 2 (+0) | precision 0.5551 vs 0.5417 | ms/frame 7.72 vs 14.55 (-6.83)
- [target_parts] parts_full_only (mask=upper): TSR 1.0000 vs 0.5714 (+0.4286) | FAR 0.04167 vs 0.50000 (-0.45833) | FA events 1 vs 1 (+0) | precision 0.8750 vs 0.5246 | ms/frame 7.82 vs 14.23 (-6.41)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth-stress | baseline | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 13.28 |
| synth-stress | baseline | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 14.86 |
| synth-stress | baseline | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.23 |
| synth-stress | baseline | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.95 |
| synth-stress | baseline | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 14.55 |
| synth-stress | baseline | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.76 |
| synth-stress | part_matching_off | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.65 |
| synth-stress | part_matching_off | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 14.96 |
| synth-stress | part_matching_off | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.23 |
| synth-stress | part_matching_off | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.76 |
| synth-stress | part_matching_off | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.95 |
| synth-stress | part_matching_off | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 19.20 |
| synth-stress | part_aware_off | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.97 |
| synth-stress | part_aware_off | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.30 |
| synth-stress | part_aware_off | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 15.03 |
| synth-stress | part_aware_off | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.69 |
| synth-stress | part_aware_off | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 14.50 |
| synth-stress | part_aware_off | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.43 |
| synth-stress | margin_0.00 | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.86 |
| synth-stress | margin_0.00 | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.50 |
| synth-stress | margin_0.00 | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 15.23 |
| synth-stress | margin_0.00 | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 19.53 |
| synth-stress | margin_0.00 | right | 1 | 24 | 0.6190 | 0.5551 | 0.45833 | 2 | 0.5417 | 3 | 14.53 |
| synth-stress | margin_0.00 | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.68 |
| synth-stress | margin_0.06 | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 17.33 |
| synth-stress | margin_0.06 | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.23 |
| synth-stress | margin_0.06 | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.95 |
| synth-stress | margin_0.06 | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 14.52 |
| synth-stress | margin_0.06 | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.35 |
| synth-stress | margin_0.06 | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.68 |
| synth-stress | margin_0.12 | none | 1 | 24 | 0.9524 | 0.8443 | 0.12500 | 1 | 0.8696 | 2 | 14.32 |
| synth-stress | margin_0.12 | lower | 1 | 24 | 0.5714 | 0.5114 | 0.45833 | 2 | 0.5217 | 1 | 15.12 |
| synth-stress | margin_0.12 | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.49 |
| synth-stress | margin_0.12 | left | 1 | 24 | 0.9048 | 0.7937 | 0.12500 | 1 | 0.8636 | 2 | 13.62 |
| synth-stress | margin_0.12 | right | 1 | 24 | 0.6190 | 0.5417 | 0.37500 | 2 | 0.5909 | 3 | 14.68 |
| synth-stress | margin_0.12 | edge | 1 | 24 | 0.9524 | 0.8443 | 0.12500 | 1 | 0.8696 | 2 | 15.49 |
| synth-stress | hard_negative_off | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 14.02 |
| synth-stress | hard_negative_off | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 14.41 |
| synth-stress | hard_negative_off | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.25 |
| synth-stress | hard_negative_off | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 14.31 |
| synth-stress | hard_negative_off | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 13.76 |
| synth-stress | hard_negative_off | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 14.77 |
| synth-stress | stm_only | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 14.35 |
| synth-stress | stm_only | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 15.59 |
| synth-stress | stm_only | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.11 |
| synth-stress | stm_only | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 14.21 |
| synth-stress | stm_only | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.28 |
| synth-stress | stm_only | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.31 |
| synth-stress | balanced | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 14.81 |
| synth-stress | balanced | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 15.99 |
| synth-stress | balanced | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.36 |
| synth-stress | balanced | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.21 |
| synth-stress | balanced | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.45 |
| synth-stress | balanced | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.43 |
| synth-stress | ltm_only | none | 1 | 24 | 0.9524 | 0.8443 | 0.12500 | 1 | 0.8696 | 2 | 15.14 |
| synth-stress | ltm_only | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 15.94 |
| synth-stress | ltm_only | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 15.73 |
| synth-stress | ltm_only | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 14.91 |
| synth-stress | ltm_only | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 14.53 |
| synth-stress | ltm_only | edge | 1 | 24 | 0.9524 | 0.8443 | 0.12500 | 1 | 0.8696 | 2 | 15.28 |
| synth-stress | ltm_cap_1 | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 13.91 |
| synth-stress | ltm_cap_1 | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 16.14 |
| synth-stress | ltm_cap_1 | upper | 1 | 24 | 0.5714 | 0.5351 | 0.16667 | 1 | 0.7500 | 1 | 15.19 |
| synth-stress | ltm_cap_1 | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.61 |
| synth-stress | ltm_cap_1 | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.38 |
| synth-stress | ltm_cap_1 | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.09 |
| synth-stress | ltm_cap_96 | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.45 |
| synth-stress | ltm_cap_96 | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.72 |
| synth-stress | ltm_cap_96 | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 15.42 |
| synth-stress | ltm_cap_96 | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 16.38 |
| synth-stress | ltm_cap_96 | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 17.67 |
| synth-stress | ltm_cap_96 | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 18.01 |
| synth-stress | ltm_cap_384 | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 17.11 |
| synth-stress | ltm_cap_384 | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 16.05 |
| synth-stress | ltm_cap_384 | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 16.03 |
| synth-stress | ltm_cap_384 | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 14.67 |
| synth-stress | ltm_cap_384 | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.40 |
| synth-stress | ltm_cap_384 | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.29 |
| synth-stress | freeze_on | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 14.80 |
| synth-stress | freeze_on | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.56 |
| synth-stress | freeze_on | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 15.75 |
| synth-stress | freeze_on | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 18.09 |
| synth-stress | freeze_on | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 16.17 |
| synth-stress | freeze_on | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 17.95 |
| synth-stress | freeze_off | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.83 |
| synth-stress | freeze_off | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 16.98 |
| synth-stress | freeze_off | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.55 |
| synth-stress | freeze_off | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 13.89 |
| synth-stress | freeze_off | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 15.91 |
| synth-stress | freeze_off | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 15.46 |
| synth-stress | parts_full_only | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 8.08 |
| synth-stress | parts_full_only | lower | 1 | 24 | 1.0000 | 0.8750 | 0.00000 | 0 | 1.0000 | 0 | 7.50 |
| synth-stress | parts_full_only | upper | 1 | 24 | 1.0000 | 0.8750 | 0.04167 | 1 | 0.9545 | 2 | 7.82 |
| synth-stress | parts_full_only | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 7.76 |
| synth-stress | parts_full_only | right | 1 | 24 | 0.6190 | 0.5551 | 0.45833 | 2 | 0.5417 | 3 | 7.72 |
| synth-stress | parts_full_only | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 8.62 |
| synth-stress | parts_all | none | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 16.07 |
| synth-stress | parts_all | lower | 1 | 24 | 0.5714 | 0.5213 | 0.50000 | 1 | 0.5000 | 1 | 17.11 |
| synth-stress | parts_all | upper | 1 | 24 | 0.5714 | 0.5246 | 0.50000 | 1 | 0.5000 | 1 | 14.46 |
| synth-stress | parts_all | left | 1 | 24 | 1.0000 | 0.8750 | 0.12500 | 1 | 0.8750 | 2 | 15.06 |
| synth-stress | parts_all | right | 1 | 24 | 0.6190 | 0.5417 | 0.41667 | 2 | 0.5652 | 3 | 12.83 |
| synth-stress | parts_all | edge | 1 | 24 | 0.6190 | 0.5418 | 0.50000 | 1 | 0.5000 | 1 | 13.40 |
