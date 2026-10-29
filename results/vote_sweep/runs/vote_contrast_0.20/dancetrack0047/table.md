# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0047, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0047': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 177.0, 'median_h': 177.0, 'median_area': 37892.5, 'min_h': 69.0, 'min_area': 14840.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 177.0, 'median_h': 177.0, 'median_area': 37892.5, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 197.5, 'median_h': 171.0, 'median_area': 33037.5, 'min_h': 68.0, 'min_area': 12814.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 197.5, 'median_h': 171.0, 'median_area': 33037.5, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 195.5, 'median_h': 145.5, 'median_area': 27079.5, 'min_h': 52.0, 'min_area': 9063.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 195.5, 'median_h': 145.5, 'median_area': 27079.5, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 260, 'first': 0, 'median_w': 143.5, 'median_h': 251.0, 'median_area': 27327.0, 'min_h': 57.0, 'min_area': 9612.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.5, 'median_h': 251.0, 'median_area': 27327.0, 'frames': 260}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 190.0, 'median_h': 271.0, 'median_area': 47397.0, 'min_h': 69.0, 'min_area': 12972.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 190.0, 'median_h': 271.0, 'median_area': 47397.0, 'frames': 300}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 290, 'first': 0, 'median_w': 202.0, 'median_h': 156.0, 'median_area': 30850.5, 'min_h': 65.0, 'min_area': 11850.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 202.0, 'median_h': 156.0, 'median_area': 30850.5, 'frames': 290}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 161.0, 'median_h': 210.5, 'median_area': 32651.5, 'min_h': 77.0, 'min_area': 14935.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 161.0, 'median_h': 210.5, 'median_area': 32651.5, 'frames': 300}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 147.5, 'median_h': 297.5, 'median_area': 40180.0, 'min_h': 137.0, 'min_area': 23585.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 147.5, 'median_h': 297.5, 'median_area': 40180.0, 'frames': 300}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 260, 'first': 0, 'median_w': 143.5, 'median_h': 251.0, 'median_area': 27327.0, 'min_h': 57.0, 'min_area': 9612.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.5, 'median_h': 251.0, 'median_area': 27327.0, 'frames': 260}, 'message': ''}}, target_ids={'dancetrack0047': 3}, target_strategy=stressed, wall_seconds=66.39

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.20 | none | 1 | 300 | 260 | 0.1462 | 0.1142 | 0.8538 | 0.0731 | 0.0000 | - | - | - | 0.09333 | 0.38806 | 9 | 9 | 0.5484 | 13 | 1 | 36 | - | - | - | 204.50 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0047 | 3 | forced | partial-view | none | 144x251 | 260 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0047 | vote_contrast_0.20 | none | 3 | 300 | 0.1462 | 0.1142 | 0.09333 | 9 | 0.5484 | 13 | 204.50 |
