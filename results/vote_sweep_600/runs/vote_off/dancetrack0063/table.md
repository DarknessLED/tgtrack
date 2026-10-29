# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0063, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0063': {'target_id': 4, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 500, 'first': 0, 'median_w': 182.0, 'median_h': 422.0, 'median_area': 76565.0, 'min_h': 277.0, 'min_area': 36828.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.0, 'median_h': 422.0, 'median_area': 76565.0, 'frames': 500}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 487, 'first': 0, 'median_w': 175.0, 'median_h': 426.0, 'median_area': 74529.0, 'min_h': 180.0, 'min_area': 26912.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 175.0, 'median_h': 426.0, 'median_area': 74529.0, 'frames': 487}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 470, 'first': 0, 'median_w': 175.5, 'median_h': 402.0, 'median_area': 68361.0, 'min_h': 147.0, 'min_area': 22500.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 175.5, 'median_h': 402.0, 'median_area': 68361.0, 'frames': 470}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 489, 'first': 0, 'median_w': 189.0, 'median_h': 429.0, 'median_area': 80224.0, 'min_h': 279.0, 'min_area': 37204.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.0, 'median_h': 429.0, 'median_area': 80224.0, 'frames': 489}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 428, 'first': 0, 'median_w': 164.5, 'median_h': 393.0, 'median_area': 65316.0, 'min_h': 165.0, 'min_area': 21615.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 164.5, 'median_h': 393.0, 'median_area': 65316.0, 'frames': 428}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 479, 'first': 0, 'median_w': 188.0, 'median_h': 385.0, 'median_area': 73367.0, 'min_h': 289.0, 'min_area': 43186.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 188.0, 'median_h': 385.0, 'median_area': 73367.0, 'frames': 479}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 459, 'first': 0, 'median_w': 173.0, 'median_h': 377.0, 'median_area': 66785.0, 'min_h': 164.0, 'min_area': 26733.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.0, 'median_h': 377.0, 'median_area': 66785.0, 'frames': 459}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 493, 'first': 0, 'median_w': 190.0, 'median_h': 408.0, 'median_area': 80990.0, 'min_h': 281.0, 'min_area': 34540.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 190.0, 'median_h': 408.0, 'median_area': 80990.0, 'frames': 493}}, 'selected': {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 428, 'first': 0, 'median_w': 164.5, 'median_h': 393.0, 'median_area': 65316.0, 'min_h': 165.0, 'min_area': 21615.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 164.5, 'median_h': 393.0, 'median_area': 65316.0, 'frames': 428}, 'message': ''}}, target_ids={'dancetrack0063': 4}, target_strategy=stressed, wall_seconds=81.86

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 500 | 428 | 0.0234 | 0.0264 | 0.9766 | 0.3528 | 0.5000 | 5.00 | 9.60 | 11.00 | 0.35400 | 0.91935 | 13 | 14 | 0.0112 | 18 | 2 | 7 | - | - | - | 156.21 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0063 | 4 | observable | partial-view | none | 164x393 | 428 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0063 | vote_off | none | 4 | 500 | 0.0234 | 0.0264 | 0.35400 | 13 | 0.0112 | 18 | 156.21 |
