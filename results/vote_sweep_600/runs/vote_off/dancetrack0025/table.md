# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0025, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0025': {'target_id': 6, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 402, 'first': 0, 'median_w': 127.5, 'median_h': 183.0, 'median_area': 20704.0, 'min_h': 46.0, 'min_area': 6365.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 127.5, 'median_h': 183.0, 'median_area': 20704.0, 'frames': 402}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 402, 'first': 0, 'median_w': 129.5, 'median_h': 194.0, 'median_area': 20952.5, 'min_h': 57.0, 'min_area': 8730.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 129.5, 'median_h': 194.0, 'median_area': 20952.5, 'frames': 402}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 401, 'first': 0, 'median_w': 102.0, 'median_h': 157.0, 'median_area': 14442.0, 'min_h': 36.0, 'min_area': 4140.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 102.0, 'median_h': 157.0, 'median_area': 14442.0, 'frames': 401}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 402, 'first': 0, 'median_w': 125.0, 'median_h': 195.5, 'median_area': 22096.0, 'min_h': 41.0, 'min_area': 5292.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 125.0, 'median_h': 195.5, 'median_area': 22096.0, 'frames': 402}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 374, 'first': 0, 'median_w': 93.5, 'median_h': 151.0, 'median_area': 13011.0, 'min_h': 58.0, 'min_area': 4992.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 93.5, 'median_h': 151.0, 'median_area': 13011.0, 'frames': 374}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 402, 'first': 0, 'median_w': 112.0, 'median_h': 178.5, 'median_area': 17746.5, 'min_h': 41.0, 'min_area': 5289.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 112.0, 'median_h': 178.5, 'median_area': 17746.5, 'frames': 402}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 358, 'first': 0, 'median_w': 87.0, 'median_h': 153.0, 'median_area': 12606.0, 'min_h': 30.0, 'min_area': 4560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 87.0, 'median_h': 153.0, 'median_area': 12606.0, 'frames': 358}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 402, 'first': 0, 'median_w': 116.0, 'median_h': 173.0, 'median_area': 17350.5, 'min_h': 39.0, 'min_area': 4480.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 116.0, 'median_h': 173.0, 'median_area': 17350.5, 'frames': 402}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 400, 'first': 0, 'median_w': 99.5, 'median_h': 157.5, 'median_area': 14448.0, 'min_h': 36.0, 'min_area': 5070.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 99.5, 'median_h': 157.5, 'median_area': 14448.0, 'frames': 400}}, 'selected': {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 358, 'first': 0, 'median_w': 87.0, 'median_h': 153.0, 'median_area': 12606.0, 'min_h': 30.0, 'min_area': 4560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 87.0, 'median_h': 153.0, 'median_area': 12606.0, 'frames': 358}, 'message': ''}}, target_ids={'dancetrack0025': 6}, target_strategy=stressed, wall_seconds=74.29

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_off | none | 1 | 402 | 358 | 0.0112 | 0.0132 | 0.9888 | 0.0894 | 0.0000 | - | - | - | 0.05224 | 0.55556 | 9 | 8 | 0.1600 | 12 | 2 | 2 | - | - | - | 177.33 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0025 | 6 | observable | partial-view | none | 87x153 | 358 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0025 | vote_off | none | 6 | 402 | 0.0112 | 0.0132 | 0.05224 | 9 | 0.1600 | 12 | 177.33 |
