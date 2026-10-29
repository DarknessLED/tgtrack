# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0010, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0010': {'target_id': 3, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 599, 'first': 0, 'median_w': 208.0, 'median_h': 559.0, 'median_area': 112095.0, 'min_h': 226.0, 'min_area': 28440.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 208.0, 'median_h': 559.0, 'median_area': 112095.0, 'frames': 599}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 203.0, 'median_h': 520.0, 'median_area': 105583.5, 'min_h': 336.0, 'min_area': 28560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 203.0, 'median_h': 520.0, 'median_area': 105583.5, 'frames': 600}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 195.0, 'median_h': 468.5, 'median_area': 93738.5, 'min_h': 244.0, 'min_area': 30702.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 195.0, 'median_h': 468.5, 'median_area': 93738.5, 'frames': 600}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 543, 'first': 0, 'median_w': 193.0, 'median_h': 432.0, 'median_area': 85113.0, 'min_h': 270.0, 'min_area': 29667.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 193.0, 'median_h': 432.0, 'median_area': 85113.0, 'frames': 543}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 594, 'first': 0, 'median_w': 198.5, 'median_h': 476.0, 'median_area': 94809.0, 'min_h': 220.0, 'min_area': 28600.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 198.5, 'median_h': 476.0, 'median_area': 94809.0, 'frames': 594}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 571, 'first': 0, 'median_w': 178.0, 'median_h': 477.0, 'median_area': 85500.0, 'min_h': 243.0, 'min_area': 30618.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 477.0, 'median_area': 85500.0, 'frames': 571}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 543, 'first': 0, 'median_w': 193.0, 'median_h': 432.0, 'median_area': 85113.0, 'min_h': 270.0, 'min_area': 29667.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 193.0, 'median_h': 432.0, 'median_area': 85113.0, 'frames': 543}, 'message': ''}}, target_ids={'dancetrack0010': 3}, target_strategy=stressed, wall_seconds=63.26

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 600 | 543 | 0.0994 | 0.0870 | 0.9006 | 0.6483 | 1.0000 | 82.00 | 121.60 | 124.00 | 0.67500 | 0.86564 | 13 | 21 | 0.1157 | 24 | 6 | 37 | - | - | - | 100.76 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | 3 | observable | partial-view | none | 193x432 | 543 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | vote_off | none | 3 | 600 | 0.0994 | 0.0870 | 0.67500 | 13 | 0.1157 | 24 | 100.76 |
