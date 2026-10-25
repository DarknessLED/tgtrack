# Clicked-target strategy ablation

Config: ablate=all, data_root=data\raw\mot17\MOT17\train, detector=yolo, dry_run=False, embedder=real, jobs=2, masks=['upper'], max_frames=60, subsample=2, target_ids={'MOT17-02-FRCNN': 17}, target_strategy=stressed, wall_seconds=144.15

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FA_ev | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 278.81 |
| distractor_margin | margin_0.00 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 181.09 |
| distractor_margin | margin_0.06 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 212.35 |
| distractor_margin | margin_0.12 | upper | 1 | 60 | 60 | 0.5500 | 0.4674 | 0.4500 | - | - | - | - | 0.03333 | 1 | 0.9429 | 1 | 0 | 33 | 249.59 |
| freeze | freeze_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 221.00 |
| freeze | freeze_on | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 221.16 |
| gate | gate_growth_unfrozen | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 162.68 |
| gate | gate_legacy | upper | 1 | 60 | 60 | 1.0000 | 0.7742 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 174.21 |
| gate | gate_search_early | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 214.16 |
| gate | gate_search_late | upper | 1 | 60 | 60 | 1.0000 | 0.7742 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 167.60 |
| hard_negative | hard_negative_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 14 | 191.67 |
| ltm_capacity | ltm_cap_1 | upper | 1 | 60 | 60 | 0.3167 | 0.2890 | 0.6833 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 19 | 194.89 |
| ltm_capacity | ltm_cap_384 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 222.95 |
| ltm_capacity | ltm_cap_96 | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 215.26 |
| part_aware | part_aware_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.03333 | 2 | 0.8750 | 1 | 0 | 14 | 203.74 |
| part_matching | part_matching_off | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 277.14 |
| stm_ltm | balanced | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 218.07 |
| stm_ltm | ltm_only | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 223.07 |
| stm_ltm | stm_only | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 215.25 |
| target_parts | parts_all | upper | 1 | 60 | 60 | 0.2333 | 0.2111 | 0.7667 | - | - | - | - | 0.01667 | 1 | 0.9333 | 1 | 0 | 14 | 225.74 |
| target_parts | parts_full_only | upper | 1 | 60 | 60 | 0.2833 | 0.2552 | 0.7167 | - | - | - | - | 0.01667 | 1 | 0.9444 | 1 | 0 | 17 | 126.70 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upper | 0.2333 | 0.2333 | +0.0000 | 0.01667 | 0.01667 | +0.00000 | 278.81 | 277.14 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 181.09 vs 278.81 (-97.72)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 212.35 vs 278.81 (-66.46)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.5500 vs 0.2333 (+0.3167) | FAR 0.03333 vs 0.01667 (+0.01667) | FA events 1 vs 1 (+0) | precision 0.4674 vs 0.2111 | ms/frame 249.59 vs 278.81 (-29.22)
- [freeze] freeze_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 221.00 vs 278.81 (-57.81)
- [freeze] freeze_on (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 221.16 vs 278.81 (-57.66)
- [gate] gate_growth_unfrozen (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 162.68 vs 278.81 (-116.13)
- [gate] gate_legacy (mask=upper): TSR 1.0000 vs 0.2333 (+0.7667) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.7742 vs 0.2111 | ms/frame 174.21 vs 278.81 (-104.60)
- [gate] gate_search_early (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 214.16 vs 278.81 (-64.65)
- [gate] gate_search_late (mask=upper): TSR 1.0000 vs 0.2333 (+0.7667) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.7742 vs 0.2111 | ms/frame 167.60 vs 278.81 (-111.21)
- [hard_negative] hard_negative_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.2111 vs 0.2111 | ms/frame 191.67 vs 278.81 (-87.14)
- [ltm_capacity] ltm_cap_1 (mask=upper): TSR 0.3167 vs 0.2333 (+0.0833) | FAR 0.00000 vs 0.01667 (-0.01667) | FA events 0 vs 1 (-1) | precision 0.2890 vs 0.2111 | ms/frame 194.89 vs 278.81 (-83.92)
- [ltm_capacity] ltm_cap_384 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 222.95 vs 278.81 (-55.86)
- [ltm_capacity] ltm_cap_96 (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 215.26 vs 278.81 (-63.55)
- [part_aware] part_aware_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.03333 vs 0.01667 (+0.01667) | FA events 2 vs 1 (+1) | precision 0.2111 vs 0.2111 | ms/frame 203.74 vs 278.81 (-75.07)
- [part_matching] part_matching_off (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 277.14 vs 278.81 (-1.68)
- [stm_ltm] balanced (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 218.07 vs 278.81 (-60.74)
- [stm_ltm] ltm_only (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 223.07 vs 278.81 (-55.74)
- [stm_ltm] stm_only (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 215.25 vs 278.81 (-63.56)
- [target_parts] parts_all (mask=upper): TSR 0.2333 vs 0.2333 (+0.0000) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2111 vs 0.2111 | ms/frame 225.74 vs 278.81 (-53.07)
- [target_parts] parts_full_only (mask=upper): TSR 0.2833 vs 0.2333 (+0.0500) | FAR 0.01667 vs 0.01667 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2552 vs 0.2111 | ms/frame 126.70 vs 278.81 (-152.11)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-FRCNN | baseline | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 278.81 |
| MOT17-02-FRCNN | part_matching_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 277.14 |
| MOT17-02-FRCNN | part_aware_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.03333 | 2 | 0.8750 | 1 | 203.74 |
| MOT17-02-FRCNN | margin_0.00 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 181.09 |
| MOT17-02-FRCNN | margin_0.06 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 212.35 |
| MOT17-02-FRCNN | margin_0.12 | upper | 17 | 60 | 0.5500 | 0.4674 | 0.03333 | 1 | 0.9429 | 1 | 249.59 |
| MOT17-02-FRCNN | hard_negative_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.00000 | 0 | 1.0000 | 0 | 191.67 |
| MOT17-02-FRCNN | stm_only | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 215.25 |
| MOT17-02-FRCNN | balanced | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 218.07 |
| MOT17-02-FRCNN | ltm_only | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 223.07 |
| MOT17-02-FRCNN | ltm_cap_1 | upper | 17 | 60 | 0.3167 | 0.2890 | 0.00000 | 0 | 1.0000 | 0 | 194.89 |
| MOT17-02-FRCNN | ltm_cap_96 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 215.26 |
| MOT17-02-FRCNN | ltm_cap_384 | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 222.95 |
| MOT17-02-FRCNN | freeze_on | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 221.16 |
| MOT17-02-FRCNN | freeze_off | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 221.00 |
| MOT17-02-FRCNN | parts_full_only | upper | 17 | 60 | 0.2833 | 0.2552 | 0.01667 | 1 | 0.9444 | 1 | 126.70 |
| MOT17-02-FRCNN | parts_all | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 225.74 |
| MOT17-02-FRCNN | gate_legacy | upper | 17 | 60 | 1.0000 | 0.7742 | 0.00000 | 0 | 1.0000 | 0 | 174.21 |
| MOT17-02-FRCNN | gate_search_early | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 214.16 |
| MOT17-02-FRCNN | gate_search_late | upper | 17 | 60 | 1.0000 | 0.7742 | 0.00000 | 0 | 1.0000 | 0 | 167.60 |
| MOT17-02-FRCNN | gate_growth_unfrozen | upper | 17 | 60 | 0.2333 | 0.2111 | 0.01667 | 1 | 0.9333 | 1 | 162.68 |
