# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0014, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=600, subsample=2, target_decisions={'dancetrack0014': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 523, 'first': 0, 'median_w': 108.0, 'median_h': 259.0, 'median_area': 27000.0, 'min_h': 152.0, 'min_area': 11972.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 108.0, 'median_h': 259.0, 'median_area': 27000.0, 'frames': 523}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 567, 'first': 0, 'median_w': 123.0, 'median_h': 277.0, 'median_area': 33019.0, 'min_h': 149.0, 'min_area': 12045.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 123.0, 'median_h': 277.0, 'median_area': 33019.0, 'frames': 567}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 465, 'first': 0, 'median_w': 98.0, 'median_h': 247.0, 'median_area': 23074.0, 'min_h': 165.0, 'min_area': 9870.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 98.0, 'median_h': 247.0, 'median_area': 23074.0, 'frames': 465}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 444, 'first': 0, 'median_w': 101.0, 'median_h': 218.0, 'median_area': 22240.5, 'min_h': 134.0, 'min_area': 10425.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 101.0, 'median_h': 218.0, 'median_area': 22240.5, 'frames': 444}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 524, 'first': 0, 'median_w': 112.0, 'median_h': 247.5, 'median_area': 27398.5, 'min_h': 155.0, 'min_area': 12064.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 112.0, 'median_h': 247.5, 'median_area': 27398.5, 'frames': 524}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 574, 'first': 0, 'median_w': 129.0, 'median_h': 269.5, 'median_area': 31696.0, 'min_h': 157.0, 'min_area': 14219.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 129.0, 'median_h': 269.5, 'median_area': 31696.0, 'frames': 574}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 560, 'first': 0, 'median_w': 122.0, 'median_h': 275.0, 'median_area': 33195.0, 'min_h': 168.0, 'min_area': 15824.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 122.0, 'median_h': 275.0, 'median_area': 33195.0, 'frames': 560}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 517, 'first': 0, 'median_w': 123.0, 'median_h': 252.0, 'median_area': 29904.0, 'min_h': 149.0, 'min_area': 10200.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 123.0, 'median_h': 252.0, 'median_area': 29904.0, 'frames': 517}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 521, 'first': 0, 'median_w': 106.0, 'median_h': 221.0, 'median_area': 23920.0, 'min_h': 143.0, 'min_area': 9900.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 106.0, 'median_h': 221.0, 'median_area': 23920.0, 'frames': 521}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 557, 'first': 0, 'median_w': 112.0, 'median_h': 268.0, 'median_area': 28710.0, 'min_h': 144.0, 'min_area': 12660.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 112.0, 'median_h': 268.0, 'median_area': 28710.0, 'frames': 557}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 567, 'first': 0, 'median_w': 128.0, 'median_h': 262.0, 'median_area': 32058.0, 'min_h': 152.0, 'min_area': 11286.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 128.0, 'median_h': 262.0, 'median_area': 32058.0, 'frames': 567}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 494, 'first': 0, 'median_w': 109.5, 'median_h': 245.0, 'median_area': 27292.0, 'min_h': 123.0, 'min_area': 8064.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 109.5, 'median_h': 245.0, 'median_area': 27292.0, 'frames': 494}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 444, 'first': 0, 'median_w': 101.0, 'median_h': 218.0, 'median_area': 22240.5, 'min_h': 134.0, 'min_area': 10425.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 101.0, 'median_h': 218.0, 'median_area': 22240.5, 'frames': 444}, 'message': ''}}, target_ids={'dancetrack0014': 3}, target_strategy=stressed, wall_seconds=476.79

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.10 | none | 1 | 600 | 444 | 0.0338 | 0.0420 | 0.9662 | 0.4752 | 0.0000 | - | - | - | 0.52167 | 0.93210 | 20 | 21 | 0.0369 | 34 | 2 | 12 | - | - | - | 777.96 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0014 | 3 | forced | partial-view | none | 101x218 | 444 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0014 | vote_contrast_0.10 | none | 3 | 600 | 0.0338 | 0.0420 | 0.52167 | 20 | 0.0369 | 34 | 777.96 |
