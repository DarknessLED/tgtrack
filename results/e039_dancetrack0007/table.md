# Clicked-target strategy ablation

Config: ablate=hard_negative, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0007, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0007': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'min_h': 296.0, 'min_area': 27966.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'frames': 199}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'frames': 197}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'min_h': 252.0, 'min_area': 26208.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'frames': 199}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'min_h': 244.0, 'min_area': 20631.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'frames': 200}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'min_h': 260.0, 'min_area': 27040.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'frames': 200}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'min_h': 278.0, 'min_area': 18711.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'frames': 200}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'min_h': 266.0, 'min_area': 24846.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'frames': 200}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'min_h': 279.0, 'min_area': 27808.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'frames': 200}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'frames': 197}, 'message': ''}}, target_ids={'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=206.38

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 197 | 0.3147 | 0.3541 | 0.6853 | 0.6751 | 0.0000 | - | - | - | 0.69000 | 0.69697 | 2 | 2 | 0.2995 | 3 | 0 | 62 | - | - | - | 277.19 |
| hard_negative | hard_negative_confident | none | 1 | 200 | 197 | 0.3147 | 0.3541 | 0.6853 | 0.6751 | 0.0000 | - | - | - | 0.69000 | 0.69697 | 2 | 2 | 0.2995 | 3 | 0 | 62 | - | - | - | 270.23 |
| hard_negative | hard_negative_far0.1 | none | 1 | 200 | 197 | 0.3147 | 0.3541 | 0.6853 | 0.6751 | 0.0000 | - | - | - | 0.69000 | 0.69697 | 2 | 2 | 0.2995 | 3 | 0 | 62 | - | - | - | 268.82 |
| hard_negative | hard_negative_far0.2 | none | 1 | 200 | 197 | 0.3147 | 0.3541 | 0.6853 | 0.6751 | 0.0000 | - | - | - | 0.69000 | 0.69697 | 2 | 2 | 0.2995 | 3 | 0 | 62 | - | - | - | 263.85 |
| hard_negative | hard_negative_far0.3 | none | 1 | 200 | 197 | 0.3147 | 0.3541 | 0.6853 | 0.6751 | 0.0000 | - | - | - | 0.69000 | 0.69697 | 2 | 2 | 0.2995 | 3 | 0 | 62 | - | - | - | 264.84 |
| hard_negative | hard_negative_far0.5 | none | 1 | 200 | 197 | 0.3147 | 0.3554 | 0.6853 | 0.6802 | 0.0000 | - | - | - | 0.70500 | 0.70352 | 1 | 2 | 0.2950 | 3 | 0 | 62 | - | - | - | 190.57 |
| hard_negative | hard_negative_off | none | 1 | 200 | 197 | 0.4061 | 0.4315 | 0.5939 | 0.5939 | 1.0000 | 0.00 | 0.00 | 0.00 | 0.62500 | 0.62500 | 3 | 3 | 0.3719 | 4 | 2 | 62 | - | - | - | 280.60 |

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

- [hard_negative] hard_negative_confident (mask=none): TSR 0.3147 vs 0.3147 (+0.0000) | FAR 0.69000 vs 0.69000 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.3541 vs 0.3541 | ms/frame 270.23 vs 277.19 (-6.96)
- [hard_negative] hard_negative_far0.1 (mask=none): TSR 0.3147 vs 0.3147 (+0.0000) | FAR 0.69000 vs 0.69000 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.3541 vs 0.3541 | ms/frame 268.82 vs 277.19 (-8.37)
- [hard_negative] hard_negative_far0.2 (mask=none): TSR 0.3147 vs 0.3147 (+0.0000) | FAR 0.69000 vs 0.69000 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.3541 vs 0.3541 | ms/frame 263.85 vs 277.19 (-13.34)
- [hard_negative] hard_negative_far0.3 (mask=none): TSR 0.3147 vs 0.3147 (+0.0000) | FAR 0.69000 vs 0.69000 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.3541 vs 0.3541 | ms/frame 264.84 vs 277.19 (-12.35)
- [hard_negative] hard_negative_far0.5 (mask=none): TSR 0.3147 vs 0.3147 (+0.0000) | FAR 0.70500 vs 0.69000 (+0.01500) | FA events 1 vs 2 (-1) | precision 0.3554 vs 0.3541 | ms/frame 190.57 vs 277.19 (-86.63)
- [hard_negative] hard_negative_off (mask=none): TSR 0.4061 vs 0.3147 (+0.0914) | FAR 0.62500 vs 0.69000 (-0.06500) | FA events 3 vs 2 (+1) | precision 0.4315 vs 0.3541 | ms/frame 280.60 vs 277.19 (+3.41)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | baseline | none | 1 | 200 | 0.3147 | 0.3541 | 0.69000 | 2 | 0.2995 | 3 | 277.19 |
| dancetrack0007 | hard_negative_off | none | 1 | 200 | 0.4061 | 0.4315 | 0.62500 | 3 | 0.3719 | 4 | 280.60 |
| dancetrack0007 | hard_negative_confident | none | 1 | 200 | 0.3147 | 0.3541 | 0.69000 | 2 | 0.2995 | 3 | 270.23 |
| dancetrack0007 | hard_negative_far0.1 | none | 1 | 200 | 0.3147 | 0.3541 | 0.69000 | 2 | 0.2995 | 3 | 268.82 |
| dancetrack0007 | hard_negative_far0.2 | none | 1 | 200 | 0.3147 | 0.3541 | 0.69000 | 2 | 0.2995 | 3 | 263.85 |
| dancetrack0007 | hard_negative_far0.3 | none | 1 | 200 | 0.3147 | 0.3541 | 0.69000 | 2 | 0.2995 | 3 | 264.84 |
| dancetrack0007 | hard_negative_far0.5 | none | 1 | 200 | 0.3147 | 0.3554 | 0.70500 | 1 | 0.2950 | 3 | 190.57 |
