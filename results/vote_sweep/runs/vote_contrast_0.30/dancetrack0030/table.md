# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0030, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0030': {'target_id': 0, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 113.0, 'median_h': 188.0, 'median_area': 21552.0, 'min_h': 169.0, 'min_area': 9350.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 113.0, 'median_h': 188.0, 'median_area': 21552.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 150.0, 'median_h': 237.0, 'median_area': 34688.0, 'min_h': 163.0, 'min_area': 13908.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 150.0, 'median_h': 237.0, 'median_area': 34688.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 137.0, 'median_h': 236.0, 'median_area': 31750.5, 'min_h': 152.0, 'min_area': 13640.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 137.0, 'median_h': 236.0, 'median_area': 31750.5, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 128.5, 'median_h': 207.5, 'median_area': 27701.0, 'min_h': 146.0, 'min_area': 11275.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 128.5, 'median_h': 207.5, 'median_area': 27701.0, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 119.0, 'median_h': 220.5, 'median_area': 26831.5, 'min_h': 150.0, 'min_area': 12508.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 119.0, 'median_h': 220.5, 'median_area': 26831.5, 'frames': 300}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 106.0, 'median_h': 204.0, 'median_area': 22880.5, 'min_h': 152.0, 'min_area': 11457.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 106.0, 'median_h': 204.0, 'median_area': 22880.5, 'frames': 300}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 113.0, 'median_h': 188.0, 'median_area': 21552.0, 'min_h': 169.0, 'min_area': 9350.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 113.0, 'median_h': 188.0, 'median_area': 21552.0, 'frames': 300}, 'message': ''}}, target_ids={'dancetrack0030': 0}, target_strategy=stressed, wall_seconds=58.33

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.30 | none | 1 | 300 | 300 | 0.3700 | 0.3393 | 0.6300 | 0.0333 | - | - | - | - | 0.01667 | 0.03333 | 3 | 3 | 0.9565 | 5 | 3 | 75 | - | - | - | 184.12 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0030 | 0 | forced | partial-view | none | 113x188 | 300 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0030 | vote_contrast_0.30 | none | 0 | 300 | 0.3700 | 0.3393 | 0.01667 | 3 | 0.9565 | 5 | 184.12 |
