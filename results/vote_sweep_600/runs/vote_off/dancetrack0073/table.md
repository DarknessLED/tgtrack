# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0073, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0073': {'target_id': 7, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 352, 'first': 0, 'median_w': 232.0, 'median_h': 305.5, 'median_area': 71062.5, 'min_h': 242.0, 'min_area': 35642.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 232.0, 'median_h': 305.5, 'median_area': 71062.5, 'frames': 352}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 243.0, 'median_h': 382.0, 'median_area': 95122.0, 'min_h': 172.0, 'min_area': 20313.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 243.0, 'median_h': 382.0, 'median_area': 95122.0, 'frames': 351}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 352, 'first': 0, 'median_w': 314.0, 'median_h': 463.0, 'median_area': 150591.5, 'min_h': 366.0, 'min_area': 68808.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 314.0, 'median_h': 463.0, 'median_area': 150591.5, 'frames': 352}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 351, 'first': 0, 'median_w': 421.0, 'median_h': 567.0, 'median_area': 237629.0, 'min_h': 449.0, 'min_area': 149574.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 421.0, 'median_h': 567.0, 'median_area': 237629.0, 'frames': 351}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 231, 'first': 121, 'median_w': 145.0, 'median_h': 321.0, 'median_area': 44649.0, 'min_h': 206.0, 'min_area': 17181.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 145.0, 'median_h': 321.0, 'median_area': 44649.0, 'frames': 231}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 352, 'first': 0, 'median_w': 324.0, 'median_h': 490.0, 'median_area': 163475.0, 'min_h': 371.0, 'min_area': 76266.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 324.0, 'median_h': 490.0, 'median_area': 163475.0, 'frames': 352}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 352, 'first': 0, 'median_w': 273.5, 'median_h': 426.0, 'median_area': 113558.0, 'min_h': 304.0, 'min_area': 60450.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 273.5, 'median_h': 426.0, 'median_area': 113558.0, 'frames': 352}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 302, 'first': 0, 'median_w': 233.5, 'median_h': 334.5, 'median_area': 78961.5, 'min_h': 253.0, 'min_area': 40296.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 233.5, 'median_h': 334.5, 'median_area': 78961.5, 'frames': 302}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 265, 'first': 80, 'median_w': 155.0, 'median_h': 395.0, 'median_area': 60771.0, 'min_h': 205.0, 'min_area': 22785.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 155.0, 'median_h': 395.0, 'median_area': 60771.0, 'frames': 265}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 280, 'first': 72, 'median_w': 188.0, 'median_h': 462.0, 'median_area': 78376.0, 'min_h': 215.0, 'min_area': 31416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 188.0, 'median_h': 462.0, 'median_area': 78376.0, 'frames': 280}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 331, 'first': 21, 'median_w': 201.0, 'median_h': 509.0, 'median_area': 98847.0, 'min_h': 213.0, 'min_area': 27492.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 201.0, 'median_h': 509.0, 'median_area': 98847.0, 'frames': 331}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 309, 'first': 0, 'median_w': 195.0, 'median_h': 468.0, 'median_area': 88736.0, 'min_h': 204.0, 'min_area': 26703.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 195.0, 'median_h': 468.0, 'median_area': 88736.0, 'frames': 309}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 226, 'first': 126, 'median_w': 185.0, 'median_h': 442.0, 'median_area': 78797.0, 'min_h': 350.0, 'min_area': 37471.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 185.0, 'median_h': 442.0, 'median_area': 78797.0, 'frames': 226}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 27, 'median_w': 153.5, 'median_h': 357.0, 'median_area': 56649.0, 'min_h': 188.0, 'min_area': 25273.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 153.5, 'median_h': 357.0, 'median_area': 56649.0, 'frames': 186}}, 'selected': {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 302, 'first': 0, 'median_w': 233.5, 'median_h': 334.5, 'median_area': 78961.5, 'min_h': 253.0, 'min_area': 40296.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 233.5, 'median_h': 334.5, 'median_area': 78961.5, 'frames': 302}, 'message': ''}}, target_ids={'dancetrack0073': 7}, target_strategy=stressed, wall_seconds=44.92

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 352 | 302 | 0.1887 | 0.1710 | 0.8113 | 0.6093 | 0.6667 | 114.00 | 124.40 | 127.00 | 0.58807 | 0.75281 | 11 | 12 | 0.2129 | 17 | 3 | 47 | - | - | - | 120.73 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0073 | 7 | observable | partial-view | none | 234x334 | 302 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0073 | vote_off | none | 7 | 352 | 0.1887 | 0.1710 | 0.58807 | 11 | 0.2129 | 17 | 120.73 |
