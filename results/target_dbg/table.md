# Clicked-target strategy ablation

Config: ablate=all, data_root=data\raw\mot17\MOT17\train, detector=yolo, dry_run=False, embedder=real, jobs=1, masks=['none'], max_frames=60, subsample=2, target_ids={'MOT17-02-FRCNN': 17}, target_strategy=stressed, wall_seconds=265.17

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FA_ev | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 239.31 |
| distractor_margin | margin_0.00 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 200.53 |
| distractor_margin | margin_0.06 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 192.52 |
| distractor_margin | margin_0.12 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 201.06 |
| freeze | freeze_off | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 202.13 |
| freeze | freeze_on | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 199.04 |
| gate | gate_growth_unfrozen | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 201.56 |
| gate | gate_legacy | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 210.41 |
| gate | gate_search_early | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 203.68 |
| gate | gate_search_late | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 198.08 |
| hard_negative | hard_negative_off | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 193.97 |
| ltm_capacity | ltm_cap_1 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 196.39 |
| ltm_capacity | ltm_cap_384 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 211.86 |
| ltm_capacity | ltm_cap_96 | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 204.95 |
| part_aware | part_aware_off | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 213.11 |
| part_matching | part_matching_off | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 198.58 |
| stm_ltm | balanced | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 209.73 |
| stm_ltm | ltm_only | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 204.13 |
| stm_ltm | stm_only | none | 1 | 60 | 60 | 0.9833 | 0.8293 | 0.0167 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 216.85 |
| target_parts | parts_all | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 204.39 |
| target_parts | parts_full_only | none | 1 | 60 | 60 | 0.9667 | 0.8145 | 0.0333 | - | - | - | - | 0.00000 | 0 | 1.0000 | 0 | 1 | 30 | 156.51 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.9667 | 0.9667 | +0.0000 | 0.00000 | 0.00000 | +0.00000 | 239.31 | 198.58 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 200.53 vs 239.31 (-38.79)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 192.52 vs 239.31 (-46.79)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 201.06 vs 239.31 (-38.25)
- [freeze] freeze_off (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 202.13 vs 239.31 (-37.18)
- [freeze] freeze_on (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 199.04 vs 239.31 (-40.27)
- [gate] gate_growth_unfrozen (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 201.56 vs 239.31 (-37.75)
- [gate] gate_legacy (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 210.41 vs 239.31 (-28.90)
- [gate] gate_search_early (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 203.68 vs 239.31 (-35.63)
- [gate] gate_search_late (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 198.08 vs 239.31 (-41.23)
- [hard_negative] hard_negative_off (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 193.97 vs 239.31 (-45.35)
- [ltm_capacity] ltm_cap_1 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 196.39 vs 239.31 (-42.92)
- [ltm_capacity] ltm_cap_384 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 211.86 vs 239.31 (-27.45)
- [ltm_capacity] ltm_cap_96 (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 204.95 vs 239.31 (-34.36)
- [part_aware] part_aware_off (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 213.11 vs 239.31 (-26.20)
- [part_matching] part_matching_off (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 198.58 vs 239.31 (-40.73)
- [stm_ltm] balanced (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 209.73 vs 239.31 (-29.58)
- [stm_ltm] ltm_only (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 204.13 vs 239.31 (-35.18)
- [stm_ltm] stm_only (mask=none): TSR 0.9833 vs 0.9667 (+0.0167) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8293 vs 0.8145 | ms/frame 216.85 vs 239.31 (-22.47)
- [target_parts] parts_all (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 204.39 vs 239.31 (-34.92)
- [target_parts] parts_full_only (mask=none): TSR 0.9667 vs 0.9667 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8145 vs 0.8145 | ms/frame 156.51 vs 239.31 (-82.80)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-FRCNN | baseline | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 239.31 |
| MOT17-02-FRCNN | part_matching_off | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 198.58 |
| MOT17-02-FRCNN | part_aware_off | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 213.11 |
| MOT17-02-FRCNN | margin_0.00 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 200.53 |
| MOT17-02-FRCNN | margin_0.06 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 192.52 |
| MOT17-02-FRCNN | margin_0.12 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 201.06 |
| MOT17-02-FRCNN | hard_negative_off | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 193.97 |
| MOT17-02-FRCNN | stm_only | none | 17 | 60 | 0.9833 | 0.8293 | 0.00000 | 0 | 1.0000 | 0 | 216.85 |
| MOT17-02-FRCNN | balanced | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 209.73 |
| MOT17-02-FRCNN | ltm_only | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 204.13 |
| MOT17-02-FRCNN | ltm_cap_1 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 196.39 |
| MOT17-02-FRCNN | ltm_cap_96 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 204.95 |
| MOT17-02-FRCNN | ltm_cap_384 | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 211.86 |
| MOT17-02-FRCNN | freeze_on | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 199.04 |
| MOT17-02-FRCNN | freeze_off | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 202.13 |
| MOT17-02-FRCNN | parts_full_only | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 156.51 |
| MOT17-02-FRCNN | parts_all | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 204.39 |
| MOT17-02-FRCNN | gate_legacy | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 210.41 |
| MOT17-02-FRCNN | gate_search_early | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 203.68 |
| MOT17-02-FRCNN | gate_search_late | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 198.08 |
| MOT17-02-FRCNN | gate_growth_unfrozen | none | 17 | 60 | 0.9667 | 0.8145 | 0.00000 | 0 | 1.0000 | 0 | 201.56 |
