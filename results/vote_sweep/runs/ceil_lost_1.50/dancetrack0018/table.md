# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0018, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0018': {'target_id': 0, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 407.5, 'median_h': 736.0, 'median_area': 294988.0, 'min_h': 510.0, 'min_area': 167308.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 407.5, 'median_h': 736.0, 'median_area': 294988.0, 'frames': 252}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 317.0, 'median_h': 651.5, 'median_area': 202502.5, 'min_h': 492.0, 'min_area': 115514.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 651.5, 'median_area': 202502.5, 'frames': 252}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 350.0, 'median_h': 727.0, 'median_area': 253875.0, 'min_h': 520.0, 'min_area': 140436.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 350.0, 'median_h': 727.0, 'median_area': 253875.0, 'frames': 252}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 334.5, 'median_h': 631.0, 'median_area': 217872.5, 'min_h': 419.0, 'min_area': 80867.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 334.5, 'median_h': 631.0, 'median_area': 217872.5, 'frames': 252}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 425.0, 'median_h': 821.0, 'median_area': 337872.0, 'min_h': 651.0, 'min_area': 239954.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 425.0, 'median_h': 821.0, 'median_area': 337872.0, 'frames': 252}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 55, 'first': 0, 'median_w': 117.0, 'median_h': 341.0, 'median_area': 39446.0, 'min_h': 302.0, 'min_area': 27784.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 117.0, 'median_h': 341.0, 'median_area': 39446.0, 'frames': 55}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 90, 'first': 28, 'median_w': 110.5, 'median_h': 350.0, 'median_area': 38926.0, 'min_h': 303.0, 'min_area': 31392.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 110.5, 'median_h': 350.0, 'median_area': 38926.0, 'frames': 90}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 66, 'first': 12, 'median_w': 114.0, 'median_h': 335.0, 'median_area': 37447.5, 'min_h': 270.0, 'min_area': 22950.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 114.0, 'median_h': 335.0, 'median_area': 37447.5, 'frames': 66}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 252, 'first': 0, 'median_w': 407.5, 'median_h': 736.0, 'median_area': 294988.0, 'min_h': 510.0, 'min_area': 167308.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 407.5, 'median_h': 736.0, 'median_area': 294988.0, 'frames': 252}, 'message': ''}}, target_ids={'dancetrack0018': 0}, target_strategy=stressed, wall_seconds=201.32

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | ceil_lost_1.50 | none | 1 | 252 | 252 | 0.5675 | 0.5342 | 0.4325 | 0.0675 | - | - | - | - | 0.08730 | 0.10219 | 9 | 9 | 0.8650 | 16 | 6 | 71 | - | - | - | 708.67 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0018 | 0 | forced | partial-view | none | 408x736 | 252 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0018 | ceil_lost_1.50 | none | 0 | 252 | 0.5675 | 0.5342 | 0.08730 | 9 | 0.8650 | 16 | 708.67 |
