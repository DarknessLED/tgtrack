# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0090, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0090': {'target_id': 12, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 461, 'first': 0, 'median_w': 208.0, 'median_h': 348.0, 'median_area': 72912.0, 'min_h': 171.0, 'min_area': 20976.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 208.0, 'median_h': 348.0, 'median_area': 72912.0, 'frames': 461}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 499, 'first': 0, 'median_w': 236.0, 'median_h': 352.0, 'median_area': 84485.0, 'min_h': 175.0, 'min_area': 20125.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 236.0, 'median_h': 352.0, 'median_area': 84485.0, 'frames': 499}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 497, 'first': 0, 'median_w': 227.0, 'median_h': 361.0, 'median_area': 74655.0, 'min_h': 198.0, 'min_area': 24820.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 227.0, 'median_h': 361.0, 'median_area': 74655.0, 'frames': 497}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 463, 'first': 0, 'median_w': 176.0, 'median_h': 264.0, 'median_area': 45540.0, 'min_h': 153.0, 'min_area': 17672.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 176.0, 'median_h': 264.0, 'median_area': 45540.0, 'frames': 463}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 498, 'first': 0, 'median_w': 226.0, 'median_h': 356.5, 'median_area': 87656.5, 'min_h': 227.0, 'min_area': 22355.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 226.0, 'median_h': 356.5, 'median_area': 87656.5, 'frames': 498}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 487, 'first': 0, 'median_w': 180.0, 'median_h': 255.0, 'median_area': 42840.0, 'min_h': 172.0, 'min_area': 19400.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 180.0, 'median_h': 255.0, 'median_area': 42840.0, 'frames': 487}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 499, 'first': 0, 'median_w': 217.0, 'median_h': 338.0, 'median_area': 75570.0, 'min_h': 171.0, 'min_area': 19323.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 217.0, 'median_h': 338.0, 'median_area': 75570.0, 'frames': 499}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 444, 'first': 0, 'median_w': 152.0, 'median_h': 271.0, 'median_area': 39416.0, 'min_h': 152.0, 'min_area': 20008.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 152.0, 'median_h': 271.0, 'median_area': 39416.0, 'frames': 444}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 481, 'first': 0, 'median_w': 161.0, 'median_h': 282.0, 'median_area': 48670.0, 'min_h': 169.0, 'min_area': 17710.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 161.0, 'median_h': 282.0, 'median_area': 48670.0, 'frames': 481}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 482, 'first': 0, 'median_w': 213.5, 'median_h': 338.0, 'median_area': 67529.0, 'min_h': 159.0, 'min_area': 27370.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 213.5, 'median_h': 338.0, 'median_area': 67529.0, 'frames': 482}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 452, 'first': 0, 'median_w': 160.0, 'median_h': 237.5, 'median_area': 32608.5, 'min_h': 136.0, 'min_area': 16116.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.0, 'median_h': 237.5, 'median_area': 32608.5, 'frames': 452}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 453, 'first': 0, 'median_w': 192.0, 'median_h': 331.0, 'median_area': 62292.0, 'min_h': 170.0, 'min_area': 20862.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 331.0, 'median_area': 62292.0, 'frames': 453}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 412, 'first': 2, 'median_w': 147.0, 'median_h': 258.5, 'median_area': 34488.0, 'min_h': 150.0, 'min_area': 17982.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 147.0, 'median_h': 258.5, 'median_area': 34488.0, 'frames': 412}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 447, 'first': 0, 'median_w': 157.0, 'median_h': 269.0, 'median_area': 37100.0, 'min_h': 152.0, 'min_area': 19864.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 269.0, 'median_area': 37100.0, 'frames': 447}}, 'selected': {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 412, 'first': 2, 'median_w': 147.0, 'median_h': 258.5, 'median_area': 34488.0, 'min_h': 150.0, 'min_area': 17982.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 147.0, 'median_h': 258.5, 'median_area': 34488.0, 'frames': 412}, 'message': ''}}, target_ids={'dancetrack0090': 12}, target_strategy=stressed, wall_seconds=76.72

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 502 | 412 | 0.0752 | 0.0716 | 0.9248 | 0.5631 | 0.3529 | 76.83 | 126.50 | 138.00 | 0.64200 | 0.97170 | 10 | 14 | 0.0062 | 28 | 8 | 7 | - | - | - | 148.32 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0090 | 12 | observable | partial-view | none | 147x258 | 412 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0090 | vote_off | none | 12 | 500 | 0.0752 | 0.0716 | 0.64200 | 10 | 0.0062 | 28 | 148.32 |
