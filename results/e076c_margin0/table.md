# Clicked-target strategy ablation

Config: ablate=distractor_margin:margin_0.00, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0097, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=40, subsample=2, target_decisions={'dancetrack0097': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 171.0, 'median_h': 294.0, 'median_area': 50297.5, 'min_h': 284.0, 'min_area': 31857.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 171.0, 'median_h': 294.0, 'median_area': 50297.5, 'frames': 40}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 182.5, 'median_h': 398.0, 'median_area': 72541.5, 'min_h': 392.0, 'min_area': 53460.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 182.5, 'median_h': 398.0, 'median_area': 72541.5, 'frames': 40}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 212.5, 'median_h': 391.0, 'median_area': 89144.5, 'min_h': 364.0, 'min_area': 40768.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 212.5, 'median_h': 391.0, 'median_area': 89144.5, 'frames': 40}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 160.5, 'median_h': 396.5, 'median_area': 64213.0, 'min_h': 382.0, 'min_area': 35728.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.5, 'median_h': 396.5, 'median_area': 64213.0, 'frames': 40}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 160.5, 'median_h': 396.5, 'median_area': 64213.0, 'min_h': 382.0, 'min_area': 35728.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.5, 'median_h': 396.5, 'median_area': 64213.0, 'frames': 40}, 'message': ''}}, target_ids={'dancetrack0097': 3}, target_strategy=stressed, wall_seconds=47.56

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 40 | 40 | 0.7750 | 0.6700 | 0.2250 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 1 | 27 | - | - | - | 530.85 |
| distractor_margin | margin_0.00 | none | 1 | 40 | 40 | 0.7750 | 0.6700 | 0.2250 | 0.0000 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 1 | 27 | - | - | - | 258.95 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | 3 | forced | partial-view | none | 160x396 | 40 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=none): TSR 0.7750 vs 0.7750 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.6700 vs 0.6700 | ms/frame 258.95 vs 530.85 (-271.90)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0097 | baseline | none | 3 | 40 | 0.7750 | 0.6700 | 0.00000 | 0 | 1.0000 | 0 | 530.85 |
| dancetrack0097 | margin_0.00 | none | 3 | 40 | 0.7750 | 0.6700 | 0.00000 | 0 | 1.0000 | 0 | 258.95 |
