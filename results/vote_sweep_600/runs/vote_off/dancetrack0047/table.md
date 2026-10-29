# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0047, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0047': {'target_id': 3, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 585, 'first': 0, 'median_w': 167.0, 'median_h': 175.0, 'median_area': 31200.0, 'min_h': 69.0, 'min_area': 6708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 167.0, 'median_h': 175.0, 'median_area': 31200.0, 'frames': 585}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 596, 'first': 0, 'median_w': 178.0, 'median_h': 171.0, 'median_area': 29464.0, 'min_h': 63.0, 'min_area': 7040.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 171.0, 'median_area': 29464.0, 'frames': 596}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 568, 'first': 0, 'median_w': 166.5, 'median_h': 161.0, 'median_area': 24451.5, 'min_h': 52.0, 'min_area': 5159.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 166.5, 'median_h': 161.0, 'median_area': 24451.5, 'frames': 568}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 560, 'first': 0, 'median_w': 145.0, 'median_h': 217.5, 'median_area': 27506.0, 'min_h': 57.0, 'min_area': 8007.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 145.0, 'median_h': 217.5, 'median_area': 27506.0, 'frames': 560}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 173.5, 'median_h': 219.0, 'median_area': 35735.5, 'min_h': 69.0, 'min_area': 9180.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.5, 'median_h': 219.0, 'median_area': 35735.5, 'frames': 600}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 589, 'first': 0, 'median_w': 178.0, 'median_h': 167.0, 'median_area': 28304.0, 'min_h': 59.0, 'min_area': 7462.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 167.0, 'median_area': 28304.0, 'frames': 589}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 600, 'first': 0, 'median_w': 156.0, 'median_h': 196.5, 'median_area': 29583.0, 'min_h': 65.0, 'min_area': 6201.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 156.0, 'median_h': 196.5, 'median_area': 29583.0, 'frames': 600}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 595, 'first': 0, 'median_w': 142.0, 'median_h': 288.0, 'median_area': 38880.0, 'min_h': 137.0, 'min_area': 7605.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 142.0, 'median_h': 288.0, 'median_area': 38880.0, 'frames': 595}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 560, 'first': 0, 'median_w': 145.0, 'median_h': 217.5, 'median_area': 27506.0, 'min_h': 57.0, 'min_area': 8007.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 145.0, 'median_h': 217.5, 'median_area': 27506.0, 'frames': 560}, 'message': ''}}, target_ids={'dancetrack0047': 3}, target_strategy=stressed, wall_seconds=90.68

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 600 | 560 | 0.0679 | 0.0565 | 0.9321 | 0.2089 | 0.0000 | - | - | - | 0.24333 | 0.74479 | 15 | 15 | 0.1889 | 29 | 1 | 36 | - | - | - | 146.58 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0047 | 3 | observable | partial-view | none | 145x218 | 560 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0047 | vote_off | none | 3 | 600 | 0.0679 | 0.0565 | 0.24333 | 15 | 0.1889 | 29 | 146.58 |
