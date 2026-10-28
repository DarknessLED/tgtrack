# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0097, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0097': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 205.5, 'median_h': 537.5, 'median_area': 110206.0, 'min_h': 277.0, 'min_area': 31857.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 537.5, 'median_area': 110206.0, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 298, 'first': 0, 'median_w': 223.0, 'median_h': 528.5, 'median_area': 111609.5, 'min_h': 376.0, 'min_area': 48645.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 223.0, 'median_h': 528.5, 'median_area': 111609.5, 'frames': 298}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 232.0, 'median_h': 543.5, 'median_area': 124547.0, 'min_h': 364.0, 'min_area': 40768.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 232.0, 'median_h': 543.5, 'median_area': 124547.0, 'frames': 296}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 294, 'first': 0, 'median_w': 176.0, 'median_h': 467.5, 'median_area': 80384.0, 'min_h': 381.0, 'min_area': 35728.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 176.0, 'median_h': 467.5, 'median_area': 80384.0, 'frames': 294}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 294, 'first': 0, 'median_w': 176.0, 'median_h': 467.5, 'median_area': 80384.0, 'min_h': 381.0, 'min_area': 35728.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 176.0, 'median_h': 467.5, 'median_area': 80384.0, 'frames': 294}, 'message': ''}}, target_ids={'dancetrack0097': 3}, target_strategy=stressed, wall_seconds=38.21

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_adaptive_0.20 | none | 1 | 300 | 294 | 0.1599 | 0.2537 | 0.8401 | 0.0000 | 0.0000 | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 5 | 27 | - | - | - | 113.94 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | 3 | forced | partial-view | none | 176x468 | 294 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | vote_adaptive_0.20 | none | 3 | 300 | 0.1599 | 0.2537 | 0.00000 | 0 | 1.0000 | 0 | 113.94 |
