# Clicked-target strategy ablation

Config: ablate=mh, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0010, detector=gt, dry_run=False, embedder=fallback, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['lower']}, jobs=1, masks=['lower'], max_frames=40, subsample=1, target_decisions={'dancetrack0010': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['lower']}, 'observability': {0: {'target_id': 0, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 132.5, 'median_h': 127.0, 'median_area': 17205.75, 'min_h': 113.0, 'min_area': 14136.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 132.5, 'median_h': 127.0, 'median_area': 17205.75, 'frames': 40}, 1: {'target_id': 1, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 94.0, 'median_h': 174.5, 'median_area': 16356.0, 'min_h': 168.0, 'min_area': 14280.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 94.0, 'median_h': 174.5, 'median_area': 16356.0, 'frames': 40}, 2: {'target_id': 2, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 144.0, 'median_h': 141.25, 'median_area': 21225.75, 'min_h': 122.0, 'min_area': 15351.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 144.0, 'median_h': 141.25, 'median_area': 21225.75, 'frames': 40}, 3: {'target_id': 3, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 114.0, 'median_h': 170.0, 'median_area': 19277.5, 'min_h': 164.0, 'min_area': 16550.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 114.0, 'median_h': 170.0, 'median_area': 19277.5, 'frames': 40}, 4: {'target_id': 4, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 133.0, 'median_h': 131.0, 'median_area': 16856.0, 'min_h': 109.5, 'min_area': 14300.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 133.0, 'median_h': 131.0, 'median_area': 16856.0, 'frames': 40}, 5: {'target_id': 5, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 132.5, 'median_h': 133.5, 'median_area': 17403.75, 'min_h': 120.5, 'min_area': 15252.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 132.5, 'median_h': 133.5, 'median_area': 17403.75, 'frames': 40}}, 'selected': {'target_id': 0, 'masks': {'lower': {'mode': 'lower', 'frames': 40, 'first': 0, 'median_w': 132.5, 'median_h': 127.0, 'median_area': 17205.75, 'min_h': 113.0, 'min_area': 14136.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 132.5, 'median_h': 127.0, 'median_area': 17205.75, 'frames': 40}, 'message': ''}}, target_ids={'dancetrack0010': 0}, target_strategy=stressed, wall_seconds=8.61

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | lower | 1 | 40 | 40 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 28.71 |
| mh | mh_dual | lower | 1 | 40 | 40 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 39.31 |
| mh | mh_dual_frozen | lower | 1 | 40 | 40 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 55.22 |
| mh | mh_off | lower | 1 | 40 | 40 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 33.94 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | 0 | observable | partial-view | lower | 132x127 | 40 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [mh] mh_dual (mask=lower): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 1.0000 vs 1.0000 | ms/frame 39.31 vs 28.71 (+10.60)
- [mh] mh_dual_frozen (mask=lower): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 1.0000 vs 1.0000 | ms/frame 55.22 vs 28.71 (+26.51)
- [mh] mh_off (mask=lower): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 1.0000 vs 1.0000 | ms/frame 33.94 vs 28.71 (+5.23)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | baseline | lower | 0 | 40 | 1.0000 | 1.0000 | 0.00000 | 0 | 1.0000 | 0 | 28.71 |
| dancetrack0010 | mh_off | lower | 0 | 40 | 1.0000 | 1.0000 | 0.00000 | 0 | 1.0000 | 0 | 33.94 |
| dancetrack0010 | mh_dual | lower | 0 | 40 | 1.0000 | 1.0000 | 0.00000 | 0 | 1.0000 | 0 | 39.31 |
| dancetrack0010 | mh_dual_frozen | lower | 0 | 40 | 1.0000 | 1.0000 | 0.00000 | 0 | 1.0000 | 0 | 55.22 |
