# Clicked-target strategy ablation

Config: ablate=tube_velocity, allow_invisible_target=False, data_root=data/raw/_root_dancetrack0007, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0007': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'min_h': 296.0, 'min_area': 27966.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'frames': 199}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'frames': 197}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'min_h': 252.0, 'min_area': 26208.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'frames': 199}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'min_h': 244.0, 'min_area': 20631.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'frames': 200}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'min_h': 260.0, 'min_area': 27040.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'frames': 200}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'min_h': 278.0, 'min_area': 18711.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'frames': 200}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'min_h': 266.0, 'min_area': 24846.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'frames': 200}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'min_h': 279.0, 'min_area': 27808.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'frames': 200}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'frames': 197}, 'message': ''}}, target_ids={'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=118.37

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 197 | 0.4264 | 0.4422 | 0.5736 | 0.5736 | 0.0000 | - | - | - | 0.59500 | 0.59799 | 3 | 3 | 0.3990 | 4 | 1 | 62 | - | - | - | 284.27 |
| tube_velocity | tube_vel_w0.3 | none | 1 | 200 | 197 | 0.4010 | 0.4198 | 0.5990 | 0.4518 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.47500 | 0.53846 | 2 | 3 | 0.4444 | 4 | 1 | 70 | - | - | - | 264.59 |
| tube_velocity | tube_vel_w0.6 | none | 1 | 200 | 197 | 0.3706 | 0.3469 | 0.6294 | 0.2893 | 0.0000 | - | - | - | 0.27500 | 0.39683 | 13 | 16 | 0.5635 | 20 | 3 | 69 | - | - | - | 287.57 |
| tube_velocity | tube_vel_w1.2 | none | 1 | 200 | 197 | 0.3452 | 0.3455 | 0.6548 | 0.3858 | 0.0000 | - | - | - | 0.52000 | 0.59375 | 10 | 16 | 0.3953 | 14 | 2 | 65 | - | - | - | 255.17 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | 1 | forced | partial-view | none | 148x332 | 197 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [tube_velocity] tube_vel_w0.3 (mask=none): TSR 0.4010 vs 0.4264 (-0.0254) | FAR 0.47500 vs 0.59500 (-0.12000) | FA events 2 vs 3 (-1) | precision 0.4198 vs 0.4422 | ms/frame 264.59 vs 284.27 (-19.67)
- [tube_velocity] tube_vel_w0.6 (mask=none): TSR 0.3706 vs 0.4264 (-0.0558) | FAR 0.27500 vs 0.59500 (-0.32000) | FA events 13 vs 3 (+10) | precision 0.3469 vs 0.4422 | ms/frame 287.57 vs 284.27 (+3.30)
- [tube_velocity] tube_vel_w1.2 (mask=none): TSR 0.3452 vs 0.4264 (-0.0812) | FAR 0.52000 vs 0.59500 (-0.07500) | FA events 10 vs 3 (+7) | precision 0.3455 vs 0.4422 | ms/frame 255.17 vs 284.27 (-29.09)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | baseline | none | 1 | 200 | 0.4264 | 0.4422 | 0.59500 | 3 | 0.3990 | 4 | 284.27 |
| dancetrack0007 | tube_vel_w0.3 | none | 1 | 200 | 0.4010 | 0.4198 | 0.47500 | 2 | 0.4444 | 4 | 264.59 |
| dancetrack0007 | tube_vel_w0.6 | none | 1 | 200 | 0.3706 | 0.3469 | 0.27500 | 13 | 0.5635 | 20 | 287.57 |
| dancetrack0007 | tube_vel_w1.2 | none | 1 | 200 | 0.3452 | 0.3455 | 0.52000 | 10 | 0.3953 | 14 | 255.17 |
