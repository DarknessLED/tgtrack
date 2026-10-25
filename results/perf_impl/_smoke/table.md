# Clicked-target strategy ablation

Config: ablate=baseline, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\_root_dancetrack0007, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=3, subsample=2, target_decisions={'dancetrack0007': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 154.0, 'median_h': 320.0, 'median_area': 49280.0, 'min_h': 319.0, 'min_area': 48471.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 154.0, 'median_h': 320.0, 'median_area': 49280.0, 'frames': 3}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 157.0, 'median_h': 334.0, 'median_area': 52438.0, 'min_h': 333.0, 'min_area': 50616.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 334.0, 'median_area': 52438.0, 'frames': 3}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 118.0, 'median_h': 336.0, 'median_area': 39766.0, 'min_h': 335.0, 'min_area': 39648.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 118.0, 'median_h': 336.0, 'median_area': 39766.0, 'frames': 3}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 157.0, 'median_h': 339.0, 'median_area': 53066.0, 'min_h': 338.0, 'min_area': 51000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 339.0, 'median_area': 53066.0, 'frames': 3}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 116.0, 'median_h': 347.0, 'median_area': 40252.0, 'min_h': 346.0, 'min_area': 39324.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 116.0, 'median_h': 347.0, 'median_area': 40252.0, 'frames': 3}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 73.0, 'median_h': 320.0, 'median_area': 23360.0, 'min_h': 319.0, 'min_area': 22649.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 73.0, 'median_h': 320.0, 'median_area': 23360.0, 'frames': 3}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 237.0, 'median_h': 348.0, 'median_area': 82476.0, 'min_h': 348.0, 'min_area': 81084.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 237.0, 'median_h': 348.0, 'median_area': 82476.0, 'frames': 3}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 143.0, 'median_h': 283.0, 'median_area': 40469.0, 'min_h': 283.0, 'min_area': 40469.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.0, 'median_h': 283.0, 'median_area': 40469.0, 'frames': 3}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 3, 'first': 0, 'median_w': 157.0, 'median_h': 334.0, 'median_area': 52438.0, 'min_h': 333.0, 'min_area': 50616.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 334.0, 'median_area': 52438.0, 'frames': 3}, 'message': ''}}, target_ids={'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=12.45

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 3 | 3 | 1.0000 | 0.9502 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 3 | - | - | - | 1709.24 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | 1 | forced | partial-view | none | 157x334 | 3 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | baseline | none | 1 | 3 | 1.0000 | 0.9502 | 0.00000 | 0 | 1.0000 | 0 | 1709.24 |
