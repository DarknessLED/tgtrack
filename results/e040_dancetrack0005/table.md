# Clicked-target strategy ablation

Config: ablate=hard_negative, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0005, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0005': {'target_id': 2, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'frames': 200}}, 'selected': {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0005': 2}, target_strategy=stressed, wall_seconds=92.73

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 200 | 0.6950 | 0.6056 | 0.3050 | 0.1200 | - | - | - | - | 0.20500 | 0.15337 | 3 | 4 | 0.7709 | 2 | 1 | 105 | - | - | - | 175.65 |
| hard_negative | hard_negative_cap16 | none | 1 | 200 | 200 | 0.6950 | 0.6056 | 0.3050 | 0.1200 | - | - | - | - | 0.20500 | 0.15337 | 3 | 4 | 0.7709 | 2 | 1 | 105 | - | - | - | 106.67 |
| hard_negative | hard_negative_cap4 | none | 1 | 200 | 200 | 0.6950 | 0.6056 | 0.3050 | 0.1200 | - | - | - | - | 0.20500 | 0.15337 | 3 | 4 | 0.7709 | 2 | 1 | 105 | - | - | - | 150.18 |
| hard_negative | hard_negative_confident | none | 1 | 200 | 200 | 0.6950 | 0.6056 | 0.3050 | 0.1200 | - | - | - | - | 0.20500 | 0.15337 | 3 | 4 | 0.7709 | 2 | 1 | 105 | - | - | - | 150.64 |
| hard_negative | hard_negative_off | none | 1 | 200 | 200 | 0.1700 | 0.1130 | 0.8300 | 0.3950 | - | - | - | - | 0.48500 | 0.70796 | 4 | 6 | 0.2538 | 3 | 0 | 34 | - | - | - | 180.00 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | 2 | forced | partial-view | none | 190x424 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [hard_negative] hard_negative_cap16 (mask=none): TSR 0.6950 vs 0.6950 (+0.0000) | FAR 0.20500 vs 0.20500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.6056 vs 0.6056 | ms/frame 106.67 vs 175.65 (-68.98)
- [hard_negative] hard_negative_cap4 (mask=none): TSR 0.6950 vs 0.6950 (+0.0000) | FAR 0.20500 vs 0.20500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.6056 vs 0.6056 | ms/frame 150.18 vs 175.65 (-25.47)
- [hard_negative] hard_negative_confident (mask=none): TSR 0.6950 vs 0.6950 (+0.0000) | FAR 0.20500 vs 0.20500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.6056 vs 0.6056 | ms/frame 150.64 vs 175.65 (-25.01)
- [hard_negative] hard_negative_off (mask=none): TSR 0.1700 vs 0.6950 (-0.5250) | FAR 0.48500 vs 0.20500 (+0.28000) | FA events 4 vs 3 (+1) | precision 0.1130 vs 0.6056 | ms/frame 180.00 vs 175.65 (+4.35)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | baseline | none | 2 | 200 | 0.6950 | 0.6056 | 0.20500 | 3 | 0.7709 | 2 | 175.65 |
| dancetrack0005 | hard_negative_off | none | 2 | 200 | 0.1700 | 0.1130 | 0.48500 | 4 | 0.2538 | 3 | 180.00 |
| dancetrack0005 | hard_negative_confident | none | 2 | 200 | 0.6950 | 0.6056 | 0.20500 | 3 | 0.7709 | 2 | 150.64 |
| dancetrack0005 | hard_negative_cap4 | none | 2 | 200 | 0.6950 | 0.6056 | 0.20500 | 3 | 0.7709 | 2 | 150.18 |
| dancetrack0005 | hard_negative_cap16 | none | 2 | 200 | 0.6950 | 0.6056 | 0.20500 | 3 | 0.7709 | 2 | 106.67 |
