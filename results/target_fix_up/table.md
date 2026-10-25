# Clicked-target strategy ablation

Config: ablate=all, data_root=data\raw\mot17\MOT17\train, detector=yolo, dry_run=False, embedder=real, jobs=2, masks=['upper'], max_frames=60, subsample=2, target_ids={'MOT17-02-FRCNN': 17}, target_strategy=stressed, wall_seconds=157.57

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FA_ev | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 285.56 |
| distractor_margin | margin_0.00 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 328.65 |
| distractor_margin | margin_0.06 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 215.16 |
| distractor_margin | margin_0.12 | upper | 1 | 60 | 60 | 0.5500 | 0.4674 | 0.4500 | - | - | - | - | 0.03333 | 1 | 0.9429 | 1 | 0 | 33 | 255.61 |
| freeze | freeze_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 227.75 |
| freeze | freeze_on | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 231.47 |
| gate | gate_growth_unfrozen | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 176.67 |
| gate | gate_legacy | upper | 1 | 60 | 60 | 1.0000 | 0.7742 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 169.53 |
| gate | gate_search_early | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 214.85 |
| gate | gate_search_late | upper | 1 | 60 | 60 | 1.0000 | 0.7742 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 165.88 |
| hard_negative | hard_negative_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 14 | 195.89 |
| ltm_capacity | ltm_cap_1 | upper | 1 | 60 | 60 | 0.3167 | 0.2890 | 0.6833 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 19 | 196.26 |
| ltm_capacity | ltm_cap_384 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 228.62 |
| ltm_capacity | ltm_cap_96 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 224.59 |
| part_aware | part_aware_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.03333 | 2 | 0.8750 | 1 | 0 | 14 | 347.00 |
| part_matching | part_matching_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 288.48 |
| stm_ltm | balanced | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 222.57 |
| stm_ltm | ltm_only | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 221.16 |
| stm_ltm | stm_only | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 222.92 |
| target_parts | parts_all | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 218.38 |
| target_parts | parts_full_only | upper | 1 | 60 | 60 | 0.2833 | 0.2552 | 0.7167 | - | - | - | - | 0.01667 | 1 | 0.9444 | 1 | 0 | 17 | 129.43 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upper | 0.2333 | 0.2333 | +0.0000 | 0.01667 | 0.01667 | +0.00000 | 285.56 | 288.48 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 328.65 vs 285.56 (+43.09)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 215.16 vs 285.56 (-70.41)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.5500 vs 0.2333 (+0.3167) | FAR 0.03333 vs 0.01667 (+0.01667) | FA events 1 vs 1 (+0) | precision 0.4674 vs 0.2111 | ms/frame 255.61 vs 285.56 (-29.96)
- [freeze] freeze_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 227.75 vs 285.56 (-57.81)
- [freeze] freeze_on (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 231.47 vs 285.56 (-54.09)
- [gate] gate_growth_unfrozen (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 176.67 vs 285.56 (-108.90)
- [gate] gate_legacy (mask=upper): TSR 1.0000 vs 0.2333 (+0.7667) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.7742 vs 0.2111 | ms/frame 169.53 vs 285.56 (-116.03)
- [gate] gate_search_early (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 214.85 vs 285.56 (-70.71)
- [gate] gate_search_late (mask=upper): TSR 1.0000 vs 0.2333 (+0.7667) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.7742 vs 0.2111 | ms/frame 165.88 vs 285.56 (-119.68)
- [hard_negative] hard_negative_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.2111 vs 0.2111 | ms/frame 195.89 vs 285.56 (-89.68)
- [ltm_capacity] ltm_cap_1 (mask=upper): TSR 0.3167 vs 0.2333 (+0.0833) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.2890 vs 0.2111 | ms/frame 196.26 vs 285.56 (-89.31)
- [ltm_capacity] ltm_cap_384 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 228.62 vs 285.56 (-56.94)
- [ltm_capacity] ltm_cap_96 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 224.59 vs 285.56 (-60.97)
- [part_aware] part_aware_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.03333 vs 0.01667 (+0.01667) | FA events 2 vs 1 (+1) | precision 0.2111 vs 0.2111 | ms/frame 347.00 vs 285.56 (+61.43)
- [part_matching] part_matching_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 288.48 vs 285.56 (+2.92)
- [stm_ltm] balanced (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 222.57 vs 285.56 (-63.00)
- [stm_ltm] ltm_only (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 221.16 vs 285.56 (-64.40)
- [stm_ltm] stm_only (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 222.92 vs 285.56 (-62.65)
- [target_parts] parts_all (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 218.38 vs 285.56 (-67.18)
- [target_parts] parts_full_only (mask=upper): TSR 0.2833 vs 0.2333 (+0.0500) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2552 vs 0.2111 | ms/frame 129.43 vs 285.56 (-156.14)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-FRCNN | baseline | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 285.56 |
| MOT17-02-FRCNN | part_matching_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 288.48 |
| MOT17-02-FRCNN | part_aware_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.03333 | 2 | 0.8750 | 1 | 347.00 |
| MOT17-02-FRCNN | margin_0.00 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 328.65 |
| MOT17-02-FRCNN | margin_0.06 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 215.16 |
| MOT17-02-FRCNN | margin_0.12 | upper | 17 | 60 | 0.5500 | 0.4674 | 0.03333 | 1 | 0.9429 | 1 | 255.61 |
| MOT17-02-FRCNN | hard_negative_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.00000 | 0 | 1.0000 | 0 | 195.89 |
| MOT17-02-FRCNN | stm_only | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 222.92 |
| MOT17-02-FRCNN | balanced | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 222.57 |
| MOT17-02-FRCNN | ltm_only | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 221.16 |
| MOT17-02-FRCNN | ltm_cap_1 | upper | 17 | 60 | 0.3167 | 0.2890 | 0.00000 | 0 | 1.0000 | 0 | 196.26 |
| MOT17-02-FRCNN | ltm_cap_96 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 224.59 |
| MOT17-02-FRCNN | ltm_cap_384 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 228.62 |
| MOT17-02-FRCNN | freeze_on | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 231.47 |
| MOT17-02-FRCNN | freeze_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 227.75 |
| MOT17-02-FRCNN | parts_full_only | upper | 17 | 60 | 0.2833 | 0.2552 | 0.01667 | 1 | 0.9444 | 1 | 129.43 |
| MOT17-02-FRCNN | parts_all | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 218.38 |
| MOT17-02-FRCNN | gate_legacy | upper | 17 | 60 | 1.0000 | 0.7742 | 0.00000 | 0 | 1.0000 | 0 | 169.53 |
| MOT17-02-FRCNN | gate_search_early | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 214.85 |
| MOT17-02-FRCNN | gate_search_late | upper | 17 | 60 | 1.0000 | 0.7742 | 0.00000 | 0 | 1.0000 | 0 | 165.88 |
| MOT17-02-FRCNN | gate_growth_unfrozen | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 176.67 |
