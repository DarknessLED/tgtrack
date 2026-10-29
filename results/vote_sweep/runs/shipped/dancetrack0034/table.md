# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0034, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0034': {'target_id': 11, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 281, 'first': 0, 'median_w': 201.0, 'median_h': 395.0, 'median_area': 76136.0, 'min_h': 277.0, 'min_area': 20720.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 201.0, 'median_h': 395.0, 'median_area': 76136.0, 'frames': 281}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 256, 'first': 0, 'median_w': 197.5, 'median_h': 433.5, 'median_area': 80597.5, 'min_h': 146.0, 'min_area': 15895.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 197.5, 'median_h': 433.5, 'median_area': 80597.5, 'frames': 256}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 276, 'first': 6, 'median_w': 210.5, 'median_h': 412.0, 'median_area': 82423.0, 'min_h': 200.0, 'min_area': 39624.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 210.5, 'median_h': 412.0, 'median_area': 82423.0, 'frames': 276}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 237.5, 'median_h': 485.0, 'median_area': 114190.0, 'min_h': 253.0, 'min_area': 39325.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 237.5, 'median_h': 485.0, 'median_area': 114190.0, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 291, 'first': 0, 'median_w': 217.0, 'median_h': 446.0, 'median_area': 93744.0, 'min_h': 248.0, 'min_area': 38564.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 217.0, 'median_h': 446.0, 'median_area': 93744.0, 'frames': 291}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 289, 'first': 8, 'median_w': 201.0, 'median_h': 419.0, 'median_area': 80640.0, 'min_h': 197.0, 'min_area': 34160.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 201.0, 'median_h': 419.0, 'median_area': 80640.0, 'frames': 289}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 122, 'first': 8, 'median_w': 161.5, 'median_h': 321.5, 'median_area': 50642.0, 'min_h': 148.0, 'min_area': 12876.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 161.5, 'median_h': 321.5, 'median_area': 50642.0, 'frames': 122}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 294, 'first': 0, 'median_w': 167.5, 'median_h': 353.5, 'median_area': 57866.0, 'min_h': 166.0, 'min_area': 21303.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 167.5, 'median_h': 353.5, 'median_area': 57866.0, 'frames': 294}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 280, 'first': 0, 'median_w': 192.0, 'median_h': 404.0, 'median_area': 78457.0, 'min_h': 178.0, 'min_area': 31150.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 404.0, 'median_area': 78457.0, 'frames': 280}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 290, 'first': 0, 'median_w': 207.0, 'median_h': 448.5, 'median_area': 90516.0, 'min_h': 243.0, 'min_area': 43166.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 207.0, 'median_h': 448.5, 'median_area': 90516.0, 'frames': 290}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 174, 'first': 0, 'median_w': 281.0, 'median_h': 627.0, 'median_area': 174338.5, 'min_h': 242.0, 'min_area': 14520.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 281.0, 'median_h': 627.0, 'median_area': 174338.5, 'frames': 174}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 248, 'first': 9, 'median_w': 134.5, 'median_h': 303.0, 'median_area': 41671.0, 'min_h': 156.0, 'min_area': 10605.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 134.5, 'median_h': 303.0, 'median_area': 41671.0, 'frames': 248}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 194, 'first': 0, 'median_w': 144.5, 'median_h': 305.0, 'median_area': 44960.0, 'min_h': 173.0, 'min_area': 10726.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 144.5, 'median_h': 305.0, 'median_area': 44960.0, 'frames': 194}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 282, 'first': 0, 'median_w': 163.5, 'median_h': 305.0, 'median_area': 50477.0, 'min_h': 138.0, 'min_area': 18300.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 163.5, 'median_h': 305.0, 'median_area': 50477.0, 'frames': 282}}, 'selected': {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 248, 'first': 9, 'median_w': 134.5, 'median_h': 303.0, 'median_area': 41671.0, 'min_h': 156.0, 'min_area': 10605.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 134.5, 'median_h': 303.0, 'median_area': 41671.0, 'frames': 248}, 'message': ''}}, target_ids={'dancetrack0034': 11}, target_strategy=stressed, wall_seconds=250.71

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | shipped_e089 | none | 1 | 300 | 248 | 0.3911 | 0.3359 | 0.6089 | 0.1089 | 1.0000 | 37.20 | 79.20 | 96.00 | 0.23024 | 0.32174 | 18 | 19 | 0.5839 | 27 | 10 | 33 | - | - | - | 795.01 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0034 | 11 | forced | partial-view | none | 134x303 | 248 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0034 | shipped_e089 | none | 11 | 291 | 0.3911 | 0.3359 | 0.23024 | 18 | 0.5839 | 27 | 795.01 |
