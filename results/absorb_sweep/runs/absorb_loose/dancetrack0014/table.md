# Clicked-target strategy ablation

Config: ablate=absorb, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0014, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0014': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 110.0, 'median_h': 271.5, 'median_area': 28599.5, 'min_h': 173.0, 'min_area': 14904.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 110.0, 'median_h': 271.5, 'median_area': 28599.5, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 275, 'first': 0, 'median_w': 115.0, 'median_h': 255.0, 'median_area': 29264.0, 'min_h': 149.0, 'min_area': 12045.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 115.0, 'median_h': 255.0, 'median_area': 29264.0, 'frames': 275}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 290, 'first': 0, 'median_w': 96.0, 'median_h': 239.5, 'median_area': 22168.5, 'min_h': 165.0, 'min_area': 11340.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 96.0, 'median_h': 239.5, 'median_area': 22168.5, 'frames': 290}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 217, 'first': 0, 'median_w': 96.0, 'median_h': 211.0, 'median_area': 19788.0, 'min_h': 134.0, 'min_area': 10425.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 96.0, 'median_h': 211.0, 'median_area': 19788.0, 'frames': 217}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 277, 'first': 0, 'median_w': 105.0, 'median_h': 238.0, 'median_area': 24288.0, 'min_h': 155.0, 'min_area': 12064.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 105.0, 'median_h': 238.0, 'median_area': 24288.0, 'frames': 277}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 293, 'first': 0, 'median_w': 122.0, 'median_h': 241.0, 'median_area': 28762.0, 'min_h': 182.0, 'min_area': 14528.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 122.0, 'median_h': 241.0, 'median_area': 28762.0, 'frames': 293}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 297, 'first': 0, 'median_w': 109.0, 'median_h': 269.0, 'median_area': 29997.0, 'min_h': 168.0, 'min_area': 15824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 109.0, 'median_h': 269.0, 'median_area': 29997.0, 'frames': 297}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 281, 'first': 0, 'median_w': 112.0, 'median_h': 232.0, 'median_area': 27735.0, 'min_h': 149.0, 'min_area': 10200.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 112.0, 'median_h': 232.0, 'median_area': 27735.0, 'frames': 281}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 240, 'first': 0, 'median_w': 99.0, 'median_h': 210.0, 'median_area': 22021.0, 'min_h': 150.0, 'min_area': 9900.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 99.0, 'median_h': 210.0, 'median_area': 22021.0, 'frames': 240}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 278, 'first': 0, 'median_w': 106.5, 'median_h': 274.0, 'median_area': 26889.5, 'min_h': 176.0, 'min_area': 14553.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 106.5, 'median_h': 274.0, 'median_area': 26889.5, 'frames': 278}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 269, 'first': 0, 'median_w': 93.0, 'median_h': 235.0, 'median_area': 22119.0, 'min_h': 152.0, 'min_area': 11286.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 93.0, 'median_h': 235.0, 'median_area': 22119.0, 'frames': 269}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 202, 'first': 0, 'median_w': 97.0, 'median_h': 256.0, 'median_area': 25033.0, 'min_h': 126.0, 'min_area': 8064.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 97.0, 'median_h': 256.0, 'median_area': 25033.0, 'frames': 202}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 217, 'first': 0, 'median_w': 96.0, 'median_h': 211.0, 'median_area': 19788.0, 'min_h': 134.0, 'min_area': 10425.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 96.0, 'median_h': 211.0, 'median_area': 19788.0, 'frames': 217}, 'message': ''}}, target_ids={'dancetrack0014': 3}, target_strategy=stressed, wall_seconds=68.75

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absorb_sweep | absorb_loose | none | 1 | 300 | 217 | 0.0691 | 0.0815 | 0.9309 | 0.6820 | 0.0000 | - | - | - | 0.65667 | 0.91388 | 10 | 12 | 0.0574 | 18 | 2 | 12 | - | - | - | 205.92 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0014 | 3 | forced | partial-view | none | 96x211 | 217 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0014 | absorb_loose | none | 3 | 300 | 0.0691 | 0.0815 | 0.65667 | 10 | 0.0574 | 18 | 205.92 |
