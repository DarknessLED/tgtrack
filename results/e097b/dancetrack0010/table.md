# Clicked-target strategy ablation

Config: ablate=refute:lag_refute_dump, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0010, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=40, subsample=2, target_decisions={'dancetrack0010': {'target_id': 5, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 145.5, 'median_h': 327.5, 'median_area': 48342.5, 'min_h': 226.0, 'min_area': 30058.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 145.5, 'median_h': 327.5, 'median_area': 48342.5, 'frames': 40}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 118.5, 'median_h': 350.0, 'median_area': 41106.5, 'min_h': 336.0, 'min_area': 28560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 118.5, 'median_h': 350.0, 'median_area': 41106.5, 'frames': 40}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 160.0, 'median_h': 379.5, 'median_area': 50031.0, 'min_h': 244.0, 'min_area': 30702.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 160.0, 'median_h': 379.5, 'median_area': 50031.0, 'frames': 40}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 128.5, 'median_h': 331.0, 'median_area': 44579.0, 'min_h': 322.0, 'min_area': 33100.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 128.5, 'median_h': 331.0, 'median_area': 44579.0, 'frames': 40}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 137.0, 'median_h': 326.5, 'median_area': 38013.0, 'min_h': 220.0, 'min_area': 28600.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 137.0, 'median_h': 326.5, 'median_area': 38013.0, 'frames': 40}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 144.0, 'median_h': 344.5, 'median_area': 44949.0, 'min_h': 243.0, 'min_area': 30618.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 144.0, 'median_h': 344.5, 'median_area': 44949.0, 'frames': 40}}, 'selected': {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 40, 'first': 0, 'median_w': 144.0, 'median_h': 344.5, 'median_area': 44949.0, 'min_h': 243.0, 'min_area': 30618.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 144.0, 'median_h': 344.5, 'median_area': 44949.0, 'frames': 40}, 'message': ''}}, target_ids={'dancetrack0010': 5}, target_strategy=stressed, wall_seconds=12.9

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 40 | 40 | 0.3250 | 0.3020 | 0.6750 | 0.2250 | - | - | - | - | 0.35000 | 0.63636 | 3 | 3 | 0.2632 | 2 | 1 | 9 | - | - | - | 152.78 |
| refute | lag_refute_dump | none | 1 | 40 | 40 | 0.3250 | 0.3020 | 0.6750 | 0.2250 | - | - | - | - | 0.35000 | 0.63636 | 3 | 3 | 0.2632 | 2 | 1 | 9 | - | - | - | 106.17 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | 5 | forced | partial-view | none | 144x344 | 40 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [refute] lag_refute_dump (mask=none): TSR 0.3250 vs 0.3250 (+0.0000) | FAR 0.35000 vs 0.35000 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.3020 vs 0.3020 | ms/frame 106.17 vs 152.78 (-46.61)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | baseline | none | 5 | 40 | 0.3250 | 0.3020 | 0.35000 | 3 | 0.2632 | 2 | 152.78 |
| dancetrack0010 | lag_refute_dump | none | 5 | 40 | 0.3250 | 0.3020 | 0.35000 | 3 | 0.2632 | 2 | 106.17 |
