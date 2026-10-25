# Clicked-target strategy ablation

Config: ablate=all, data_root=data\raw\mot17\MOT17\train, detector=yolo, dry_run=False, embedder=real, jobs=1, masks=['none'], max_frames=60, subsample=1, target_ids={'MOT17-02-FRCNN': 17}, target_strategy=stressed, wall_seconds=39.99

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FA_ev | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 73.67 |
| distractor_margin | margin_0.00 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.62 |
| distractor_margin | margin_0.06 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.29 |
| distractor_margin | margin_0.12 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 29.62 |
| freeze | freeze_off | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 28.45 |
| freeze | freeze_on | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 27.91 |
| hard_negative | hard_negative_off | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 29.80 |
| ltm_capacity | ltm_cap_1 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 28.60 |
| ltm_capacity | ltm_cap_384 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.46 |
| ltm_capacity | ltm_cap_96 | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 28.87 |
| part_aware | part_aware_off | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.03 |
| part_matching | part_matching_off | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.69 |
| stm_ltm | balanced | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 28.92 |
| stm_ltm | ltm_only | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 28.63 |
| stm_ltm | stm_only | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 29.74 |
| target_parts | parts_all | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 30.86 |
| target_parts | parts_full_only | none | 1 | 60 | 60 | 1.0000 | 0.8729 | 0.0000 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 0 | 60 | 29.87 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 1.0000 | 1.0000 | +0.0000 | 0.00000 | 0.00000 | +0.00000 | 73.67 | 30.69 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.62 vs 73.67 (-43.05)
- [distractor_margin] margin_0.06 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.29 vs 73.67 (-43.38)
- [distractor_margin] margin_0.12 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 29.62 vs 73.67 (-44.05)
- [freeze] freeze_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 28.45 vs 73.67 (-45.22)
- [freeze] freeze_on (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 27.91 vs 73.67 (-45.77)
- [hard_negative] hard_negative_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 29.80 vs 73.67 (-43.87)
- [ltm_capacity] ltm_cap_1 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 28.60 vs 73.67 (-45.08)
- [ltm_capacity] ltm_cap_384 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.46 vs 73.67 (-43.21)
- [ltm_capacity] ltm_cap_96 (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 28.87 vs 73.67 (-44.80)
- [part_aware] part_aware_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.03 vs 73.67 (-43.64)
- [part_matching] part_matching_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.69 vs 73.67 (-42.99)
- [stm_ltm] balanced (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 28.92 vs 73.67 (-44.76)
- [stm_ltm] ltm_only (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 28.63 vs 73.67 (-45.05)
- [stm_ltm] stm_only (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 29.74 vs 73.67 (-43.94)
- [target_parts] parts_all (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 30.86 vs 73.67 (-42.81)
- [target_parts] parts_full_only (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8729 vs 0.8729 | ms/frame 29.87 vs 73.67 (-43.81)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-FRCNN | baseline | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 73.67 |
| MOT17-02-FRCNN | part_matching_off | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.69 |
| MOT17-02-FRCNN | part_aware_off | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.03 |
| MOT17-02-FRCNN | margin_0.00 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.62 |
| MOT17-02-FRCNN | margin_0.06 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.29 |
| MOT17-02-FRCNN | margin_0.12 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 29.62 |
| MOT17-02-FRCNN | hard_negative_off | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 29.80 |
| MOT17-02-FRCNN | stm_only | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 29.74 |
| MOT17-02-FRCNN | balanced | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 28.92 |
| MOT17-02-FRCNN | ltm_only | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 28.63 |
| MOT17-02-FRCNN | ltm_cap_1 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 28.60 |
| MOT17-02-FRCNN | ltm_cap_96 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 28.87 |
| MOT17-02-FRCNN | ltm_cap_384 | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.46 |
| MOT17-02-FRCNN | freeze_on | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 27.91 |
| MOT17-02-FRCNN | freeze_off | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 28.45 |
| MOT17-02-FRCNN | parts_full_only | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 29.87 |
| MOT17-02-FRCNN | parts_all | none | 17 | 60 | 1.0000 | 0.8729 | 0.00000 | 0 | 1.0000 | 0 | 30.86 |
