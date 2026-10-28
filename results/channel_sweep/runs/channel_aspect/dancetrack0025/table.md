# Clicked-target strategy ablation

Config: ablate=channel, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0025, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0025': {'target_id': 6, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 128.0, 'median_h': 187.0, 'median_area': 22004.0, 'min_h': 86.0, 'min_area': 9494.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 128.0, 'median_h': 187.0, 'median_area': 22004.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 129.0, 'median_h': 199.0, 'median_area': 23270.0, 'min_h': 94.0, 'min_area': 8730.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 129.0, 'median_h': 199.0, 'median_area': 23270.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 105.0, 'median_h': 163.0, 'median_area': 16214.0, 'min_h': 83.0, 'min_area': 6665.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 105.0, 'median_h': 163.0, 'median_area': 16214.0, 'frames': 299}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 126.0, 'median_h': 201.0, 'median_area': 24006.5, 'min_h': 129.0, 'min_area': 11242.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 126.0, 'median_h': 201.0, 'median_area': 24006.5, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 279, 'first': 0, 'median_w': 94.0, 'median_h': 153.0, 'median_area': 13447.0, 'min_h': 69.0, 'min_area': 4992.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 94.0, 'median_h': 153.0, 'median_area': 13447.0, 'frames': 279}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 109.5, 'median_h': 181.5, 'median_area': 19403.0, 'min_h': 93.0, 'min_area': 9000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 109.5, 'median_h': 181.5, 'median_area': 19403.0, 'frames': 300}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 266, 'first': 0, 'median_w': 87.0, 'median_h': 156.0, 'median_area': 13674.0, 'min_h': 95.0, 'min_area': 6885.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 87.0, 'median_h': 156.0, 'median_area': 13674.0, 'frames': 266}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 108.5, 'median_h': 179.0, 'median_area': 18562.0, 'min_h': 94.0, 'min_area': 7614.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 108.5, 'median_h': 179.0, 'median_area': 18562.0, 'frames': 300}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 99.0, 'median_h': 162.0, 'median_area': 16268.0, 'min_h': 75.0, 'min_area': 6517.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 99.0, 'median_h': 162.0, 'median_area': 16268.0, 'frames': 299}}, 'selected': {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 266, 'first': 0, 'median_w': 87.0, 'median_h': 156.0, 'median_area': 13674.0, 'min_h': 95.0, 'min_area': 6885.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 87.0, 'median_h': 156.0, 'median_area': 13674.0, 'frames': 266}, 'message': ''}}, target_ids={'dancetrack0025': 6}, target_strategy=stressed, wall_seconds=370.39

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| channel | channel_aspect | none | 1 | 300 | 266 | 0.0150 | 0.0164 | 0.9850 | 0.0977 | 0.0000 | - | - | - | 0.06000 | 0.56667 | 6 | 5 | 0.1818 | 11 | 2 | 2 | - | - | - | 1136.80 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0025 | 6 | forced | partial-view | none | 87x156 | 266 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0025 | channel_aspect | none | 6 | 300 | 0.0150 | 0.0164 | 0.06000 | 6 | 0.1818 | 11 | 1136.80 |
