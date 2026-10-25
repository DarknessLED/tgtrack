# Clicked-target strategy ablation

Config: ablate=part_matching, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, jobs=2, masks=['none', 'upper'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 126.5, 'median_area': 23219.75, 'min_h': 38.0, 'min_area': 7854.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 205.5, 'median_h': 126.5, 'median_area': 23219.75, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 181.5, 'median_area': 35209.5, 'min_h': 40.0, 'min_area': 11449.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 269.0, 'median_h': 181.5, 'median_area': 35209.5, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 213.5, 'median_area': 54351.5, 'min_h': 54.5, 'min_area': 17698.5, 'visible_fraction': 0.49999624545700294}}, 'worst_mask': 'upper', 'median_w': 317.0, 'median_h': 213.5, 'median_area': 54351.5, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=65.57

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 200 | 0.2650 | 0.2714 | 0.7350 | 0.2550 | - | - | - | - | 0.22500 | 0.42718 | 7 | 7 | 0.5361 | 10 | 4 | 45 | - | - | - | 170.92 |
| baseline | baseline | upper | 1 | 200 | 200 | 0.2300 | 0.2353 | 0.7700 | 0.1050 | - | - | - | - | 0.00500 | 0.01515 | 1 | 1 | 0.9787 | 2 | 9 | 15 | - | - | - | 138.49 |
| part_matching | part_matching_off | none | 1 | 200 | 200 | 0.2650 | 0.2714 | 0.7350 | 0.2550 | - | - | - | - | 0.22500 | 0.42718 | 7 | 7 | 0.5361 | 10 | 4 | 45 | - | - | - | 163.07 |
| part_matching | part_matching_off | upper | 1 | 200 | 200 | 0.2300 | 0.2353 | 0.7700 | 0.1050 | - | - | - | - | 0.00500 | 0.01515 | 1 | 1 | 0.9787 | 2 | 9 | 15 | - | - | - | 131.03 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 3 | forced | partial-view | upper | 387x223 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.2650 | 0.2650 | +0.0000 | 0.22500 | 0.22500 | +0.00000 | 170.92 | 163.07 |
| upper | 0.2300 | 0.2300 | +0.0000 | 0.00500 | 0.00500 | +0.00000 | 138.49 | 131.03 |

## Claims (delta vs the same-mask baseline)

- [part_matching] part_matching_off (mask=none): TSR 0.2650 vs 0.2650 (+0.0000) | FAR 0.22500 vs 0.22500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.2714 vs 0.2714 | ms/frame 163.07 vs 170.92 (-7.85)
- [part_matching] part_matching_off (mask=upper): TSR 0.2300 vs 0.2300 (+0.0000) | FAR 0.00500 vs 0.00500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.2353 vs 0.2353 | ms/frame 131.03 vs 138.49 (-7.46)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.2650 | 0.2714 | 0.22500 | 7 | 0.5361 | 10 | 170.92 |
| dancetrack0004 | baseline | upper | 3 | 200 | 0.2300 | 0.2353 | 0.00500 | 1 | 0.9787 | 2 | 138.49 |
| dancetrack0004 | part_matching_off | none | 3 | 200 | 0.2650 | 0.2714 | 0.22500 | 7 | 0.5361 | 10 | 163.07 |
| dancetrack0004 | part_matching_off | upper | 3 | 200 | 0.2300 | 0.2353 | 0.00500 | 1 | 0.9787 | 2 | 131.03 |
