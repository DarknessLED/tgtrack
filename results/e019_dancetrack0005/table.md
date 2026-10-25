# Clicked-target strategy ablation

Config: ablate=part_matching, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0005, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, jobs=2, masks=['none', 'upper'], max_frames=200, subsample=2, target_decisions={'dancetrack0005': {'target_id': 0, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 220.25, 'median_area': 53992.5, 'min_h': 141.0, 'min_area': 15933.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 225.0, 'median_h': 220.25, 'median_area': 53992.5, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 212.25, 'median_area': 44539.25, 'min_h': 125.5, 'min_area': 13805.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 189.5, 'median_h': 212.25, 'median_area': 44539.25, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 244.0, 'median_area': 57830.5, 'min_h': 156.5, 'min_area': 24624.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 224.0, 'median_h': 244.0, 'median_area': 57830.5, 'frames': 200}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0005': 0}, target_strategy=stressed, wall_seconds=76.82

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 200 | 0.9950 | 0.9110 | 0.0050 | 0.0050 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 1 | 164 | - | - | - | 200.62 |
| baseline | baseline | upper | 1 | 200 | 200 | 0.8650 | 0.7164 | 0.1350 | 0.0100 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 140 | - | - | - | 169.36 |
| part_matching | part_matching_off | none | 1 | 200 | 200 | 0.9950 | 0.9110 | 0.0050 | 0.0050 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 1 | 164 | - | - | - | 187.95 |
| part_matching | part_matching_off | upper | 1 | 200 | 200 | 0.8650 | 0.7164 | 0.1350 | 0.0100 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 2 | 140 | - | - | - | 155.92 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | 0 | forced | partial-view | upper | 219x233 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.9950 | 0.9950 | +0.0000 | 0.00000 | 0.00000 | +0.00000 | 200.62 | 187.95 |
| upper | 0.8650 | 0.8650 | +0.0000 | 0.00000 | 0.00000 | +0.00000 | 169.36 | 155.92 |

## Claims (delta vs the same-mask baseline)

- [part_matching] part_matching_off (mask=none): TSR 0.9950 vs 0.9950 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.9110 vs 0.9110 | ms/frame 187.95 vs 200.62 (-12.66)
- [part_matching] part_matching_off (mask=upper): TSR 0.8650 vs 0.8650 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.7164 vs 0.7164 | ms/frame 155.92 vs 169.36 (-13.44)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | baseline | none | 0 | 200 | 0.9950 | 0.9110 | 0.00000 | 0 | 1.0000 | 0 | 200.62 |
| dancetrack0005 | baseline | upper | 0 | 200 | 0.8650 | 0.7164 | 0.00000 | 0 | 1.0000 | 0 | 169.36 |
| dancetrack0005 | part_matching_off | none | 0 | 200 | 0.9950 | 0.9110 | 0.00000 | 0 | 1.0000 | 0 | 187.95 |
| dancetrack0005 | part_matching_off | upper | 0 | 200 | 0.8650 | 0.7164 | 0.00000 | 0 | 1.0000 | 0 | 155.92 |
