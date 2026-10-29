# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0063, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0063': {'target_id': 7, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 173.0, 'median_h': 417.0, 'median_area': 72398.0, 'min_h': 277.0, 'min_area': 36828.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.0, 'median_h': 417.0, 'median_area': 72398.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 169.0, 'median_h': 407.0, 'median_area': 65540.0, 'min_h': 180.0, 'min_area': 26912.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 169.0, 'median_h': 407.0, 'median_area': 65540.0, 'frames': 299}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 298, 'first': 0, 'median_w': 157.0, 'median_h': 386.5, 'median_area': 56700.0, 'min_h': 147.0, 'min_area': 22500.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 386.5, 'median_area': 56700.0, 'frames': 298}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 183.0, 'median_h': 431.5, 'median_area': 78076.0, 'min_h': 279.0, 'min_area': 37204.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 183.0, 'median_h': 431.5, 'median_area': 78076.0, 'frames': 296}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 282, 'first': 0, 'median_w': 157.0, 'median_h': 373.0, 'median_area': 60448.5, 'min_h': 165.0, 'min_area': 21615.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 373.0, 'median_area': 60448.5, 'frames': 282}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 186.0, 'median_h': 406.0, 'median_area': 73780.0, 'min_h': 289.0, 'min_area': 43186.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 186.0, 'median_h': 406.0, 'median_area': 73780.0, 'frames': 299}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 290, 'first': 0, 'median_w': 170.0, 'median_h': 377.0, 'median_area': 65301.0, 'min_h': 164.0, 'min_area': 26733.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 170.0, 'median_h': 377.0, 'median_area': 65301.0, 'frames': 290}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 192.0, 'median_h': 433.0, 'median_area': 82670.0, 'min_h': 281.0, 'min_area': 34540.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 433.0, 'median_area': 82670.0, 'frames': 300}}, 'selected': {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 192.0, 'median_h': 433.0, 'median_area': 82670.0, 'min_h': 281.0, 'min_area': 34540.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 192.0, 'median_h': 433.0, 'median_area': 82670.0, 'frames': 300}, 'message': ''}}, target_ids={'dancetrack0063': 7}, target_strategy=stressed, wall_seconds=342.31

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | margin_lost_0.03 | none | 1 | 300 | 300 | 0.0100 | 0.0152 | 0.9900 | 0.1033 | - | - | - | - | 0.09000 | 0.79412 | 4 | 4 | 0.0357 | 3 | 0 | 3 | - | - | - | 1069.18 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0063 | 7 | forced | partial-view | none | 192x433 | 300 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0063 | margin_lost_0.03 | none | 7 | 300 | 0.0100 | 0.0152 | 0.09000 | 4 | 0.0357 | 3 | 1069.18 |
