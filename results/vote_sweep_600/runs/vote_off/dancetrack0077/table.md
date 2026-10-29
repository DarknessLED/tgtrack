# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0077, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0077': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 210.5, 'median_h': 256.0, 'median_area': 44313.5, 'min_h': 70.0, 'min_area': 23749.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 210.5, 'median_h': 256.0, 'median_area': 44313.5, 'frames': 600}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 242.5, 'median_h': 264.0, 'median_area': 52251.0, 'min_h': 92.0, 'min_area': 24344.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 242.5, 'median_h': 264.0, 'median_area': 52251.0, 'frames': 600}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 195.5, 'median_h': 229.0, 'median_area': 39796.0, 'min_h': 76.0, 'min_area': 17871.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 195.5, 'median_h': 229.0, 'median_area': 39796.0, 'frames': 600}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 170.0, 'median_h': 335.0, 'median_area': 58640.0, 'min_h': 101.0, 'min_area': 25875.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 170.0, 'median_h': 335.0, 'median_area': 58640.0, 'frames': 600}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 245.0, 'median_h': 280.0, 'median_area': 61091.0, 'min_h': 86.0, 'min_area': 34228.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 245.0, 'median_h': 280.0, 'median_area': 61091.0, 'frames': 600}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 194.0, 'median_h': 269.5, 'median_area': 47738.0, 'min_h': 86.0, 'min_area': 28208.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 194.0, 'median_h': 269.5, 'median_area': 47738.0, 'frames': 600}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 205.0, 'median_h': 250.5, 'median_area': 51828.5, 'min_h': 101.0, 'min_area': 27258.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.0, 'median_h': 250.5, 'median_area': 51828.5, 'frames': 600}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 185.0, 'median_h': 241.0, 'median_area': 38826.0, 'min_h': 63.0, 'min_area': 13348.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 185.0, 'median_h': 241.0, 'median_area': 38826.0, 'frames': 600}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 173.5, 'median_h': 247.0, 'median_area': 40672.0, 'min_h': 77.0, 'min_area': 18876.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.5, 'median_h': 247.0, 'median_area': 40672.0, 'frames': 600}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 210.5, 'median_h': 256.0, 'median_area': 44313.5, 'min_h': 70.0, 'min_area': 23749.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 210.5, 'median_h': 256.0, 'median_area': 44313.5, 'frames': 600}, 'message': ''}}, target_ids={'dancetrack0077': 0}, target_strategy=stressed, wall_seconds=72.16

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 600 | 600 | 0.1200 | 0.1391 | 0.8800 | 0.3583 | - | - | - | - | 0.37167 | 0.73077 | 7 | 9 | 0.2441 | 8 | 3 | 63 | - | - | - | 115.88 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0077 | 0 | observable | partial-view | none | 210x256 | 600 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0077 | vote_off | none | 0 | 600 | 0.1200 | 0.1391 | 0.37167 | 7 | 0.2441 | 8 | 115.88 |
