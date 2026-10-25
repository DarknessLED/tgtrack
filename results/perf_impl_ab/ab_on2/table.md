# Clicked-target strategy ablation

Config: ablate=baseline, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\_root_dancetrack0007, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=40, subsample=2, target_decisions={'dancetrack0007': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 157.0, 'median_h': 314.5, 'median_area': 49531.5, 'min_h': 312.0, 'min_area': 48471.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 314.5, 'median_area': 49531.5, 'frames': 40}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 151.0, 'median_h': 332.0, 'median_area': 49905.5, 'min_h': 330.0, 'min_area': 45678.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 151.0, 'median_h': 332.0, 'median_area': 49905.5, 'frames': 40}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 161.0, 'median_h': 328.0, 'median_area': 52888.5, 'min_h': 326.0, 'min_area': 39648.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 161.0, 'median_h': 328.0, 'median_area': 52888.5, 'frames': 40}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 167.0, 'median_h': 349.0, 'median_area': 57759.5, 'min_h': 338.0, 'min_area': 29064.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 167.0, 'median_h': 349.0, 'median_area': 57759.5, 'frames': 40}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 146.5, 'median_h': 326.5, 'median_area': 45621.0, 'min_h': 302.0, 'min_area': 39324.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 146.5, 'median_h': 326.5, 'median_area': 45621.0, 'frames': 40}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 69.0, 'median_h': 298.0, 'median_area': 20562.0, 'min_h': 297.0, 'min_area': 18711.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 69.0, 'median_h': 298.0, 'median_area': 20562.0, 'frames': 40}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 185.5, 'median_h': 342.0, 'median_area': 63440.5, 'min_h': 337.0, 'min_area': 62220.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 185.5, 'median_h': 342.0, 'median_area': 63440.5, 'frames': 40}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 205.0, 'median_h': 286.0, 'median_area': 59040.0, 'min_h': 279.0, 'min_area': 37944.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.0, 'median_h': 286.0, 'median_area': 59040.0, 'frames': 40}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 151.0, 'median_h': 332.0, 'median_area': 49905.5, 'min_h': 330.0, 'min_area': 45678.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 151.0, 'median_h': 332.0, 'median_area': 49905.5, 'frames': 40}, 'message': ''}}, target_ids={'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=17.63

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 40 | 40 | 1.0000 | 0.9606 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 277.91 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | 1 | forced | partial-view | none | 151x332 | 40 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | baseline | none | 1 | 40 | 1.0000 | 0.9606 | 0.00000 | 0 | 1.0000 | 0 | 277.91 |
