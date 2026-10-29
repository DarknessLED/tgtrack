# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0065, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0065': {'target_id': 0, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 155.5, 'median_h': 330.0, 'median_area': 49236.5, 'min_h': 110.0, 'min_area': 13824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 155.5, 'median_h': 330.0, 'median_area': 49236.5, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 157.5, 'median_h': 358.0, 'median_area': 56756.5, 'min_h': 166.0, 'min_area': 21000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.5, 'median_h': 358.0, 'median_area': 56756.5, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 148.0, 'median_h': 318.0, 'median_area': 48461.0, 'min_h': 159.0, 'min_area': 19971.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 318.0, 'median_area': 48461.0, 'frames': 299}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 166.5, 'median_h': 345.0, 'median_area': 51582.0, 'min_h': 85.0, 'min_area': 18360.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 166.5, 'median_h': 345.0, 'median_area': 51582.0, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 173.0, 'median_h': 331.0, 'median_area': 56984.0, 'min_h': 123.0, 'min_area': 19845.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 173.0, 'median_h': 331.0, 'median_area': 56984.0, 'frames': 299}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 155.5, 'median_h': 330.0, 'median_area': 49236.5, 'min_h': 110.0, 'min_area': 13824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 155.5, 'median_h': 330.0, 'median_area': 49236.5, 'frames': 300}, 'message': ''}}, target_ids={'dancetrack0065': 0}, target_strategy=stressed, wall_seconds=231.85

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | margin_lost_0.03 | none | 1 | 300 | 300 | 0.0800 | 0.0802 | 0.9200 | 0.0600 | - | - | - | - | 0.02000 | 0.14286 | 5 | 5 | 0.8000 | 6 | 6 | 7 | - | - | - | 699.98 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0065 | 0 | forced | partial-view | none | 156x330 | 300 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0065 | margin_lost_0.03 | none | 0 | 300 | 0.0800 | 0.0802 | 0.02000 | 5 | 0.8000 | 6 | 699.98 |
