# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0035, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0035': {'target_id': 5, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 331, 'first': 0, 'median_w': 183.0, 'median_h': 341.0, 'median_area': 61600.0, 'min_h': 220.0, 'min_area': 31518.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 183.0, 'median_h': 341.0, 'median_area': 61600.0, 'frames': 331}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 322, 'first': 0, 'median_w': 185.5, 'median_h': 357.0, 'median_area': 65033.0, 'min_h': 185.0, 'min_area': 26598.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 185.5, 'median_h': 357.0, 'median_area': 65033.0, 'frames': 322}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 348, 'first': 0, 'median_w': 200.0, 'median_h': 355.0, 'median_area': 69650.5, 'min_h': 198.0, 'min_area': 28140.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 200.0, 'median_h': 355.0, 'median_area': 69650.5, 'frames': 348}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 344, 'first': 0, 'median_w': 182.0, 'median_h': 352.0, 'median_area': 63599.0, 'min_h': 216.0, 'min_area': 29808.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.0, 'median_h': 352.0, 'median_area': 63599.0, 'frames': 344}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 347, 'first': 0, 'median_w': 210.0, 'median_h': 365.0, 'median_area': 77615.0, 'min_h': 220.0, 'min_area': 29260.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 210.0, 'median_h': 365.0, 'median_area': 77615.0, 'frames': 347}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 250, 'first': 3, 'median_w': 182.0, 'median_h': 345.0, 'median_area': 63435.5, 'min_h': 230.0, 'min_area': 32890.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.0, 'median_h': 345.0, 'median_area': 63435.5, 'frames': 250}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 338, 'first': 0, 'median_w': 183.0, 'median_h': 358.0, 'median_area': 64461.5, 'min_h': 210.0, 'min_area': 31034.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 183.0, 'median_h': 358.0, 'median_area': 64461.5, 'frames': 338}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 352, 'first': 0, 'median_w': 190.0, 'median_h': 356.0, 'median_area': 67135.0, 'min_h': 175.0, 'min_area': 32900.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 190.0, 'median_h': 356.0, 'median_area': 67135.0, 'frames': 352}}, 'selected': {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 250, 'first': 3, 'median_w': 182.0, 'median_h': 345.0, 'median_area': 63435.5, 'min_h': 230.0, 'min_area': 32890.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.0, 'median_h': 345.0, 'median_area': 63435.5, 'frames': 250}, 'message': ''}}, target_ids={'dancetrack0035': 5}, target_strategy=stressed, wall_seconds=72.64

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 352 | 250 | 0.0200 | 0.0135 | 0.9800 | 0.0240 | 0.6875 | 22.09 | 39.00 | 51.00 | 0.05444 | 0.60714 | 14 | 13 | 0.1364 | 16 | 4 | 1 | - | - | - | 199.17 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0035 | 5 | observable | partial-view | none | 182x345 | 250 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0035 | vote_off | none | 5 | 349 | 0.0200 | 0.0135 | 0.05444 | 14 | 0.1364 | 16 | 199.17 |
