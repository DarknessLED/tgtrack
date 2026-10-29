# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0090, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0090': {'target_id': 12, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 204.5, 'median_h': 381.5, 'median_area': 73794.0, 'min_h': 197.0, 'min_area': 29988.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 204.5, 'median_h': 381.5, 'median_area': 73794.0, 'frames': 296}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 219.5, 'median_h': 383.0, 'median_area': 80460.0, 'min_h': 175.0, 'min_area': 20125.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.5, 'median_h': 383.0, 'median_area': 80460.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 295, 'first': 0, 'median_w': 205.0, 'median_h': 363.0, 'median_area': 70312.0, 'min_h': 201.0, 'min_area': 24820.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.0, 'median_h': 363.0, 'median_area': 70312.0, 'frames': 295}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 283, 'first': 0, 'median_w': 168.0, 'median_h': 264.0, 'median_area': 38064.0, 'min_h': 167.0, 'min_area': 17672.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 168.0, 'median_h': 264.0, 'median_area': 38064.0, 'frames': 283}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 220.0, 'median_h': 394.5, 'median_area': 87121.0, 'min_h': 227.0, 'min_area': 25608.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 220.0, 'median_h': 394.5, 'median_area': 87121.0, 'frames': 300}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 292, 'first': 0, 'median_w': 172.0, 'median_h': 244.0, 'median_area': 36518.0, 'min_h': 172.0, 'min_area': 19400.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 172.0, 'median_h': 244.0, 'median_area': 36518.0, 'frames': 292}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 298, 'first': 0, 'median_w': 202.5, 'median_h': 343.5, 'median_area': 69799.0, 'min_h': 171.0, 'min_area': 19323.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 202.5, 'median_h': 343.5, 'median_area': 69799.0, 'frames': 298}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 272, 'first': 0, 'median_w': 143.0, 'median_h': 270.0, 'median_area': 32980.0, 'min_h': 152.0, 'min_area': 20008.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.0, 'median_h': 270.0, 'median_area': 32980.0, 'frames': 272}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 283, 'first': 0, 'median_w': 145.0, 'median_h': 280.0, 'median_area': 40086.0, 'min_h': 169.0, 'min_area': 20022.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 145.0, 'median_h': 280.0, 'median_area': 40086.0, 'frames': 283}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 285, 'first': 0, 'median_w': 200.0, 'median_h': 359.0, 'median_area': 64848.0, 'min_h': 186.0, 'min_area': 31515.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 200.0, 'median_h': 359.0, 'median_area': 64848.0, 'frames': 285}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 259, 'first': 0, 'median_w': 163.0, 'median_h': 241.0, 'median_area': 32578.0, 'min_h': 159.0, 'min_area': 16500.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 163.0, 'median_h': 241.0, 'median_area': 32578.0, 'frames': 259}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 251, 'first': 0, 'median_w': 177.0, 'median_h': 343.0, 'median_area': 62812.0, 'min_h': 170.0, 'min_area': 20862.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 177.0, 'median_h': 343.0, 'median_area': 62812.0, 'frames': 251}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 234, 'first': 2, 'median_w': 140.5, 'median_h': 249.5, 'median_area': 28022.5, 'min_h': 150.0, 'min_area': 17982.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 140.5, 'median_h': 249.5, 'median_area': 28022.5, 'frames': 234}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 277, 'first': 0, 'median_w': 146.0, 'median_h': 262.0, 'median_area': 32238.0, 'min_h': 152.0, 'min_area': 19864.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 146.0, 'median_h': 262.0, 'median_area': 32238.0, 'frames': 277}}, 'selected': {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 234, 'first': 2, 'median_w': 140.5, 'median_h': 249.5, 'median_area': 28022.5, 'min_h': 150.0, 'min_area': 17982.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 140.5, 'median_h': 249.5, 'median_area': 28022.5, 'frames': 234}, 'message': ''}}, target_ids={'dancetrack0090': 12}, target_strategy=stressed, wall_seconds=79.37

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.20 | none | 1 | 300 | 234 | 0.0043 | 0.0466 | 0.9957 | 0.5726 | 0.0000 | - | - | - | 0.55369 | 0.97590 | 5 | 8 | 0.0060 | 14 | 0 | 1 | - | - | - | 249.35 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0090 | 12 | forced | partial-view | none | 140x250 | 234 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0090 | vote_contrast_0.20 | none | 12 | 298 | 0.0043 | 0.0466 | 0.55369 | 5 | 0.0060 | 14 | 249.35 |
