# Clicked-target strategy ablation

Config: ablate=channel, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=70, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 63, 'first': 6, 'median_w': 125.0, 'median_h': 242.0, 'median_area': 31130.0, 'min_h': 88.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 125.0, 'median_h': 242.0, 'median_area': 31130.0, 'frames': 63}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 70, 'first': 0, 'median_w': 211.5, 'median_h': 325.0, 'median_area': 52501.5, 'min_h': 100.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 211.5, 'median_h': 325.0, 'median_area': 52501.5, 'frames': 70}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 70, 'first': 0, 'median_w': 239.5, 'median_h': 366.0, 'median_area': 74851.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 239.5, 'median_h': 366.0, 'median_area': 74851.0, 'frames': 70}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 70, 'first': 0, 'median_w': 502.0, 'median_h': 257.0, 'median_area': 117999.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 502.0, 'median_h': 257.0, 'median_area': 117999.5, 'frames': 70}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 70, 'first': 0, 'median_w': 502.0, 'median_h': 257.0, 'median_area': 117999.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 502.0, 'median_h': 257.0, 'median_area': 117999.5, 'frames': 70}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=138.9

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| channel | channel_off | none | 1 | 70 | 70 | 0.1000 | 0.1305 | 0.9000 | 0.1571 | - | - | - | - | 0.08571 | 0.33333 | 3 | 3 | 0.5385 | 4 | 3 | 4 | - | - | - | 1171.18 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 3 | observable | partial-view | none | 502x257 | 70 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | channel_off | none | 3 | 70 | 0.1000 | 0.1305 | 0.08571 | 3 | 0.5385 | 4 | 1171.18 |
