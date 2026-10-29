# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0077, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0077': {'target_id': 0, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 305.5, 'median_h': 172.5, 'median_area': 45127.0, 'min_h': 70.0, 'min_area': 23749.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 305.5, 'median_h': 172.5, 'median_area': 45127.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 316.0, 'median_h': 205.0, 'median_area': 59532.5, 'min_h': 92.0, 'min_area': 24344.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 316.0, 'median_h': 205.0, 'median_area': 59532.5, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 254.0, 'median_h': 165.0, 'median_area': 39915.0, 'min_h': 76.0, 'min_area': 17871.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 254.0, 'median_h': 165.0, 'median_area': 39915.0, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 239.0, 'median_h': 344.5, 'median_area': 73944.0, 'min_h': 101.0, 'min_area': 29120.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 239.0, 'median_h': 344.5, 'median_area': 73944.0, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 384.0, 'median_h': 213.5, 'median_area': 70434.0, 'min_h': 86.0, 'min_area': 34228.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 384.0, 'median_h': 213.5, 'median_area': 70434.0, 'frames': 300}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 307.0, 'median_h': 206.0, 'median_area': 52080.0, 'min_h': 86.0, 'min_area': 28208.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 307.0, 'median_h': 206.0, 'median_area': 52080.0, 'frames': 300}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 320.0, 'median_h': 210.0, 'median_area': 64233.5, 'min_h': 101.0, 'min_area': 28910.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 320.0, 'median_h': 210.0, 'median_area': 64233.5, 'frames': 300}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 268.5, 'median_h': 163.0, 'median_area': 38608.5, 'min_h': 63.0, 'min_area': 13348.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 268.5, 'median_h': 163.0, 'median_area': 38608.5, 'frames': 300}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 253.5, 'median_h': 173.5, 'median_area': 38960.0, 'min_h': 77.0, 'min_area': 18876.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 253.5, 'median_h': 173.5, 'median_area': 38960.0, 'frames': 300}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 305.5, 'median_h': 172.5, 'median_area': 45127.0, 'min_h': 70.0, 'min_area': 23749.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 305.5, 'median_h': 172.5, 'median_area': 45127.0, 'frames': 300}, 'message': ''}}, target_ids={'dancetrack0077': 0}, target_strategy=stressed, wall_seconds=210.67

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | ceil_lost_off | none | 1 | 300 | 300 | 0.0233 | 0.0279 | 0.9767 | 0.0400 | - | - | - | - | 0.07667 | 0.57895 | 6 | 6 | 0.2333 | 2 | 0 | 7 | - | - | - | 685.14 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0077 | 0 | forced | partial-view | none | 306x172 | 300 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0077 | ceil_lost_off | none | 0 | 300 | 0.0233 | 0.0279 | 0.07667 | 6 | 0.2333 | 2 | 685.14 |
