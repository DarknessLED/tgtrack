# Clicked-target strategy ablation

Config: ablate=mh, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0018, detector=gt, dry_run=False, embedder=fallback, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=40, subsample=1, target_decisions={'dancetrack0018': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 364.0, 'median_h': 579.0, 'median_area': 211750.0, 'min_h': 510.0, 'min_area': 174930.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 364.0, 'median_h': 579.0, 'median_area': 211750.0, 'frames': 40}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 239.5, 'median_h': 515.0, 'median_area': 120861.0, 'min_h': 491.0, 'min_area': 114608.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 239.5, 'median_h': 515.0, 'median_area': 120861.0, 'frames': 40}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 295.5, 'median_h': 571.0, 'median_area': 167717.5, 'min_h': 520.0, 'min_area': 140436.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 295.5, 'median_h': 571.0, 'median_area': 167717.5, 'frames': 40}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 250.0, 'median_h': 471.0, 'median_area': 116138.0, 'min_h': 419.0, 'min_area': 80867.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 250.0, 'median_h': 471.0, 'median_area': 116138.0, 'frames': 40}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 445.5, 'median_h': 834.5, 'median_area': 359336.0, 'min_h': 769.0, 'min_area': 252000.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 445.5, 'median_h': 834.5, 'median_area': 359336.0, 'frames': 40}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 118.0, 'median_h': 330.5, 'median_area': 39029.0, 'min_h': 322.0, 'min_area': 36040.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 118.0, 'median_h': 330.5, 'median_area': 39029.0, 'frames': 40}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 16, 'first': 23, 'median_w': 101.0, 'median_h': 328.0, 'median_area': 33128.0, 'min_h': 316.0, 'min_area': 31916.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 101.0, 'median_h': 328.0, 'median_area': 33128.0, 'frames': 16}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 364.0, 'median_h': 579.0, 'median_area': 211750.0, 'min_h': 510.0, 'min_area': 174930.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 364.0, 'median_h': 579.0, 'median_area': 211750.0, 'frames': 40}, 'message': ''}}, target_ids={'dancetrack0018': 0}, target_strategy=stressed, wall_seconds=11.43

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 40 | 40 | 1.0000 | 0.9713 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 35.23 |
| mh | mh_dual | none | 1 | 40 | 40 | 1.0000 | 0.9713 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 39.49 |
| mh | mh_dual_frozen | none | 1 | 40 | 40 | 1.0000 | 0.9713 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 44.70 |
| mh | mh_off | none | 1 | 40 | 40 | 1.0000 | 0.9713 | 0.0000 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 0 | 40 | - | - | - | 31.27 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0018 | 0 | observable | partial-view | none | 364x579 | 40 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [mh] mh_dual (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.9713 vs 0.9713 | ms/frame 39.49 vs 35.23 (+4.26)
- [mh] mh_dual_frozen (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.9713 vs 0.9713 | ms/frame 44.70 vs 35.23 (+9.46)
- [mh] mh_off (mask=none): TSR 1.0000 vs 1.0000 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.9713 vs 0.9713 | ms/frame 31.27 vs 35.23 (-3.96)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0018 | baseline | none | 0 | 40 | 1.0000 | 0.9713 | 0.00000 | 0 | 1.0000 | 0 | 35.23 |
| dancetrack0018 | mh_off | none | 0 | 40 | 1.0000 | 0.9713 | 0.00000 | 0 | 1.0000 | 0 | 31.27 |
| dancetrack0018 | mh_dual | none | 0 | 40 | 1.0000 | 0.9713 | 0.00000 | 0 | 1.0000 | 0 | 39.49 |
| dancetrack0018 | mh_dual_frozen | none | 0 | 40 | 1.0000 | 0.9713 | 0.00000 | 0 | 1.0000 | 0 | 44.70 |
