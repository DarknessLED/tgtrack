# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0065, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0065': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 160.0, 'median_h': 340.0, 'median_area': 51810.0, 'min_h': 110.0, 'min_area': 13824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.0, 'median_h': 340.0, 'median_area': 51810.0, 'frames': 351}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 165.0, 'median_h': 353.0, 'median_area': 57340.0, 'min_h': 166.0, 'min_area': 21000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 165.0, 'median_h': 353.0, 'median_area': 57340.0, 'frames': 351}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 350, 'first': 0, 'median_w': 152.0, 'median_h': 318.0, 'median_area': 49946.0, 'min_h': 159.0, 'min_area': 19971.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 152.0, 'median_h': 318.0, 'median_area': 49946.0, 'frames': 350}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 170.0, 'median_h': 345.0, 'median_area': 54560.0, 'min_h': 85.0, 'min_area': 18360.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 170.0, 'median_h': 345.0, 'median_area': 54560.0, 'frames': 351}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 350, 'first': 0, 'median_w': 175.0, 'median_h': 337.0, 'median_area': 58350.0, 'min_h': 123.0, 'min_area': 19845.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 175.0, 'median_h': 337.0, 'median_area': 58350.0, 'frames': 350}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 160.0, 'median_h': 340.0, 'median_area': 51810.0, 'min_h': 110.0, 'min_area': 13824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.0, 'median_h': 340.0, 'median_area': 51810.0, 'frames': 351}, 'message': ''}}, target_ids={'dancetrack0065': 0}, target_strategy=stressed, wall_seconds=43.29

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 351 | 351 | 0.0427 | 0.0426 | 0.9573 | 0.0541 | - | - | - | - | 0.02279 | 0.23529 | 6 | 6 | 0.6522 | 9 | 6 | 6 | - | - | - | 115.66 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0065 | 0 | observable | partial-view | none | 160x340 | 351 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0065 | vote_off | none | 0 | 351 | 0.0427 | 0.0426 | 0.02279 | 6 | 0.6522 | 9 | 115.66 |
