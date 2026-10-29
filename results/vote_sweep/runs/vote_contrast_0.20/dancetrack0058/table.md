# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0058, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0058': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 207.5, 'median_h': 459.0, 'median_area': 91445.5, 'min_h': 349.0, 'min_area': 38454.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 207.5, 'median_h': 459.0, 'median_area': 91445.5, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 192.0, 'median_h': 414.0, 'median_area': 77904.5, 'min_h': 321.0, 'min_area': 45326.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 414.0, 'median_area': 77904.5, 'frames': 296}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 201.5, 'median_h': 413.0, 'median_area': 82470.5, 'min_h': 319.0, 'min_area': 46002.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 201.5, 'median_h': 413.0, 'median_area': 82470.5, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 204.0, 'median_h': 441.0, 'median_area': 83230.0, 'min_h': 362.0, 'min_area': 38640.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 204.0, 'median_h': 441.0, 'median_area': 83230.0, 'frames': 299}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 203.5, 'median_h': 433.0, 'median_area': 83192.5, 'min_h': 321.0, 'min_area': 39204.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 203.5, 'median_h': 433.0, 'median_area': 83192.5, 'frames': 300}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 210.0, 'median_h': 398.5, 'median_area': 81609.0, 'min_h': 315.0, 'min_area': 45045.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 210.0, 'median_h': 398.5, 'median_area': 81609.0, 'frames': 300}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 227.5, 'median_h': 451.0, 'median_area': 98527.0, 'min_h': 328.0, 'min_area': 50700.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 227.5, 'median_h': 451.0, 'median_area': 98527.0, 'frames': 300}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 192.0, 'median_h': 414.0, 'median_area': 77904.5, 'min_h': 321.0, 'min_area': 45326.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 414.0, 'median_area': 77904.5, 'frames': 296}, 'message': ''}}, target_ids={'dancetrack0058': 1}, target_strategy=stressed, wall_seconds=53.4

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.20 | none | 1 | 300 | 296 | 0.0338 | 0.0945 | 0.9662 | 0.9155 | 0.0000 | - | - | - | 0.90000 | 0.95070 | 5 | 5 | 0.0357 | 5 | 0 | 10 | - | - | - | 164.68 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0058 | 1 | forced | partial-view | none | 192x414 | 296 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0058 | vote_contrast_0.20 | none | 1 | 300 | 0.0338 | 0.0945 | 0.90000 | 5 | 0.0357 | 5 | 164.68 |
