# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0019, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0019': {'target_id': 6, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 124.0, 'median_h': 202.5, 'median_area': 20752.0, 'min_h': 58.0, 'min_area': 7790.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 124.0, 'median_h': 202.5, 'median_area': 20752.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 123.0, 'median_h': 201.0, 'median_area': 21840.0, 'min_h': 61.0, 'min_area': 11020.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 123.0, 'median_h': 201.0, 'median_area': 21840.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 142.0, 'median_h': 240.0, 'median_area': 32896.5, 'min_h': 75.0, 'min_area': 13200.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 142.0, 'median_h': 240.0, 'median_area': 32896.5, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 150.0, 'median_h': 247.5, 'median_area': 33229.0, 'min_h': 69.0, 'min_area': 17360.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 150.0, 'median_h': 247.5, 'median_area': 33229.0, 'frames': 300}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 127.0, 'median_h': 229.0, 'median_area': 26936.0, 'min_h': 60.0, 'min_area': 9800.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 127.0, 'median_h': 229.0, 'median_area': 26936.0, 'frames': 299}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 111.0, 'median_h': 200.0, 'median_area': 18924.0, 'min_h': 42.0, 'min_area': 9653.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 111.0, 'median_h': 200.0, 'median_area': 18924.0, 'frames': 296}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 279, 'first': 0, 'median_w': 120.0, 'median_h': 213.0, 'median_area': 26418.0, 'min_h': 133.0, 'min_area': 12264.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 120.0, 'median_h': 213.0, 'median_area': 26418.0, 'frames': 279}}, 'selected': {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 279, 'first': 0, 'median_w': 120.0, 'median_h': 213.0, 'median_area': 26418.0, 'min_h': 133.0, 'min_area': 12264.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 120.0, 'median_h': 213.0, 'median_area': 26418.0, 'frames': 279}, 'message': ''}}, target_ids={'dancetrack0019': 6}, target_strategy=stressed, wall_seconds=223.12

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_unanimous | none | 1 | 300 | 279 | 0.2186 | 0.1983 | 0.7814 | 0.0538 | 0.0000 | - | - | - | 0.03667 | 0.14667 | 5 | 5 | 0.8472 | 8 | 1 | 47 | - | - | - | 680.51 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0019 | 6 | forced | partial-view | none | 120x213 | 279 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0019 | vote_unanimous | none | 6 | 300 | 0.2186 | 0.1983 | 0.03667 | 5 | 0.8472 | 8 | 680.51 |
