# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0058, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0058': {'target_id': 1, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 195.5, 'median_h': 467.5, 'median_area': 94752.0, 'min_h': 281.0, 'min_area': 38454.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 195.5, 'median_h': 467.5, 'median_area': 94752.0, 'frames': 600}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 592, 'first': 0, 'median_w': 178.0, 'median_h': 408.0, 'median_area': 72307.0, 'min_h': 306.0, 'min_area': 32130.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 408.0, 'median_area': 72307.0, 'frames': 592}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 175.5, 'median_h': 392.0, 'median_area': 71823.0, 'min_h': 228.0, 'min_area': 25560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 175.5, 'median_h': 392.0, 'median_area': 71823.0, 'frames': 600}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 599, 'first': 0, 'median_w': 199.0, 'median_h': 428.0, 'median_area': 78848.0, 'min_h': 226.0, 'min_area': 36411.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 199.0, 'median_h': 428.0, 'median_area': 78848.0, 'frames': 599}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 180.0, 'median_h': 428.0, 'median_area': 76902.0, 'min_h': 321.0, 'min_area': 39204.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 180.0, 'median_h': 428.0, 'median_area': 76902.0, 'frames': 600}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 599, 'first': 0, 'median_w': 196.0, 'median_h': 415.0, 'median_area': 80410.0, 'min_h': 315.0, 'min_area': 42000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 196.0, 'median_h': 415.0, 'median_area': 80410.0, 'frames': 599}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 554, 'first': 0, 'median_w': 184.0, 'median_h': 414.0, 'median_area': 74359.5, 'min_h': 239.0, 'min_area': 32535.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 184.0, 'median_h': 414.0, 'median_area': 74359.5, 'frames': 554}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 592, 'first': 0, 'median_w': 178.0, 'median_h': 408.0, 'median_area': 72307.0, 'min_h': 306.0, 'min_area': 32130.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 408.0, 'median_area': 72307.0, 'frames': 592}, 'message': ''}}, target_ids={'dancetrack0058': 1}, target_strategy=stressed, wall_seconds=86.48

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 600 | 592 | 0.0456 | 0.0588 | 0.9544 | 0.5794 | 0.3333 | 11.00 | 11.00 | 11.00 | 0.62000 | 0.94038 | 19 | 19 | 0.0363 | 26 | 4 | 12 | - | - | - | 138.49 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0058 | 1 | observable | partial-view | none | 178x408 | 592 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0058 | vote_off | none | 1 | 600 | 0.0456 | 0.0588 | 0.62000 | 19 | 0.0363 | 26 | 138.49 |
