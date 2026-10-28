# Clicked-target strategy ablation

Config: ablate=channel, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0097, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=20, subsample=2, target_decisions={'dancetrack0097': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 20, 'first': 0, 'median_w': 204.5, 'median_h': 294.0, 'median_area': 61247.0, 'min_h': 285.0, 'min_area': 50150.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 204.5, 'median_h': 294.0, 'median_area': 61247.0, 'frames': 20}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 20, 'first': 0, 'median_w': 178.0, 'median_h': 397.5, 'median_area': 70844.0, 'min_h': 393.0, 'min_area': 62646.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 397.5, 'median_area': 70844.0, 'frames': 20}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 20, 'first': 0, 'median_w': 284.0, 'median_h': 413.0, 'median_area': 116036.5, 'min_h': 369.0, 'min_area': 61623.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 284.0, 'median_h': 413.0, 'median_area': 116036.5, 'frames': 20}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 20, 'first': 0, 'median_w': 163.5, 'median_h': 396.0, 'median_area': 63718.0, 'min_h': 383.0, 'min_area': 49288.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 163.5, 'median_h': 396.0, 'median_area': 63718.0, 'frames': 20}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 20, 'first': 0, 'median_w': 163.5, 'median_h': 396.0, 'median_area': 63718.0, 'min_h': 383.0, 'min_area': 49288.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 163.5, 'median_h': 396.0, 'median_area': 63718.0, 'frames': 20}, 'message': ''}}, target_ids={'dancetrack0097': 3}, target_strategy=stressed, wall_seconds=142.46

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 20 | 20 | 1.0000 | 0.8054 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 20 | - | - | - | 2237.65 |
| channel | channel_aspect | none | 1 | 20 | 20 | 1.0000 | 0.8054 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 20 | - | - | - | 786.85 |
| channel | channel_both | none | 1 | 20 | 20 | 1.0000 | 0.8054 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 20 | - | - | - | 828.42 |
| channel | channel_fusion | none | 1 | 20 | 20 | 1.0000 | 0.8054 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 20 | - | - | - | 778.41 |
| channel | channel_off | none | 1 | 20 | 20 | 1.0000 | 0.8054 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 20 | - | - | - | 785.62 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | 3 | forced | partial-view | none | 164x396 | 20 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [channel] channel_aspect (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8054 vs 0.8054 | ms/frame 786.85 vs 2237.65 (-1450.80)
- [channel] channel_both (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8054 vs 0.8054 | ms/frame 828.42 vs 2237.65 (-1409.23)
- [channel] channel_fusion (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8054 vs 0.8054 | ms/frame 778.41 vs 2237.65 (-1459.24)
- [channel] channel_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.8054 vs 0.8054 | ms/frame 785.62 vs 2237.65 (-1452.03)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | baseline | none | 3 | 20 | 1.0000 | 0.8054 | 0.00000 | 0 | 1.0000 | 0 | 2237.65 |
| dancetrack0097 | channel_off | none | 3 | 20 | 1.0000 | 0.8054 | 0.00000 | 0 | 1.0000 | 0 | 785.62 |
| dancetrack0097 | channel_fusion | none | 3 | 20 | 1.0000 | 0.8054 | 0.00000 | 0 | 1.0000 | 0 | 778.41 |
| dancetrack0097 | channel_aspect | none | 3 | 20 | 1.0000 | 0.8054 | 0.00000 | 0 | 1.0000 | 0 | 786.85 |
| dancetrack0097 | channel_both | none | 3 | 20 | 1.0000 | 0.8054 | 0.00000 | 0 | 1.0000 | 0 | 828.42 |
