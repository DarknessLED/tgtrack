# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0034, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0034': {'target_id': 11, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 426, 'first': 0, 'median_w': 224.5, 'median_h': 372.0, 'median_area': 84502.5, 'min_h': 175.0, 'min_area': 20720.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.5, 'median_h': 372.0, 'median_area': 84502.5, 'frames': 426}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 396, 'first': 0, 'median_w': 221.0, 'median_h': 333.0, 'median_area': 79218.0, 'min_h': 146.0, 'min_area': 15895.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 221.0, 'median_h': 333.0, 'median_area': 79218.0, 'frames': 396}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 392, 'first': 6, 'median_w': 173.5, 'median_h': 390.0, 'median_area': 68551.0, 'min_h': 142.0, 'min_area': 17214.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.5, 'median_h': 390.0, 'median_area': 68551.0, 'frames': 392}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 456, 'first': 0, 'median_w': 224.0, 'median_h': 424.0, 'median_area': 88139.0, 'min_h': 164.0, 'min_area': 34932.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.0, 'median_h': 424.0, 'median_area': 88139.0, 'frames': 456}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 444, 'first': 0, 'median_w': 241.5, 'median_h': 446.0, 'median_area': 112220.0, 'min_h': 216.0, 'min_area': 38564.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 241.5, 'median_h': 446.0, 'median_area': 112220.0, 'frames': 444}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 451, 'first': 8, 'median_w': 236.0, 'median_h': 448.0, 'median_area': 100485.0, 'min_h': 197.0, 'min_area': 34160.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 236.0, 'median_h': 448.0, 'median_area': 100485.0, 'frames': 451}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 247, 'first': 8, 'median_w': 230.0, 'median_h': 349.0, 'median_area': 79870.0, 'min_h': 148.0, 'min_area': 12876.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 230.0, 'median_h': 349.0, 'median_area': 79870.0, 'frames': 247}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 439, 'first': 0, 'median_w': 181.0, 'median_h': 366.0, 'median_area': 63350.0, 'min_h': 166.0, 'min_area': 21303.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 181.0, 'median_h': 366.0, 'median_area': 63350.0, 'frames': 439}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 438, 'first': 0, 'median_w': 223.5, 'median_h': 444.0, 'median_area': 98279.0, 'min_h': 178.0, 'min_area': 31150.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 223.5, 'median_h': 444.0, 'median_area': 98279.0, 'frames': 438}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 451, 'first': 0, 'median_w': 237.0, 'median_h': 483.0, 'median_area': 105222.0, 'min_h': 183.0, 'min_area': 35319.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 237.0, 'median_h': 483.0, 'median_area': 105222.0, 'frames': 451}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 274, 'first': 0, 'median_w': 243.5, 'median_h': 468.0, 'median_area': 100871.5, 'min_h': 242.0, 'min_area': 14520.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 243.5, 'median_h': 468.0, 'median_area': 100871.5, 'frames': 274}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 399, 'first': 9, 'median_w': 166.0, 'median_h': 303.0, 'median_area': 49300.0, 'min_h': 149.0, 'min_area': 10605.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 166.0, 'median_h': 303.0, 'median_area': 49300.0, 'frames': 399}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 322, 'first': 0, 'median_w': 168.5, 'median_h': 302.0, 'median_area': 51572.5, 'min_h': 150.0, 'min_area': 10726.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 168.5, 'median_h': 302.0, 'median_area': 51572.5, 'frames': 322}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 440, 'first': 0, 'median_w': 182.0, 'median_h': 310.0, 'median_area': 54954.0, 'min_h': 138.0, 'min_area': 18224.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.0, 'median_h': 310.0, 'median_area': 54954.0, 'frames': 440}}, 'selected': {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 399, 'first': 9, 'median_w': 166.0, 'median_h': 303.0, 'median_area': 49300.0, 'min_h': 149.0, 'min_area': 10605.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 166.0, 'median_h': 303.0, 'median_area': 49300.0, 'frames': 399}, 'message': ''}}, target_ids={'dancetrack0034': 11}, target_strategy=stressed, wall_seconds=86.2

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 462 | 399 | 0.1128 | 0.0958 | 0.8872 | 0.1805 | 0.2222 | 14.50 | 19.70 | 21.00 | 0.16777 | 0.44348 | 32 | 31 | 0.3719 | 27 | 6 | 27 | - | - | - | 179.75 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0034 | 11 | observable | partial-view | none | 166x303 | 399 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0034 | vote_off | none | 11 | 453 | 0.1128 | 0.0958 | 0.16777 | 32 | 0.3719 | 27 | 179.75 |
