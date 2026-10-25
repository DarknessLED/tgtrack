# Clicked-target strategy ablation

Config: ablate=part_matching, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0007, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, jobs=2, masks=['none', 'upper'], max_frames=200, subsample=2, target_decisions={'dancetrack0007': {'target_id': 1, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'min_h': 296.0, 'min_area': 27966.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 180.0, 'median_area': 25525.5, 'min_h': 148.0, 'min_area': 13983.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 156.0, 'median_h': 180.0, 'median_area': 25525.5, 'frames': 199}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'frames': 197}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'min_h': 252.0, 'min_area': 26208.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 172.5, 'median_area': 26484.5, 'min_h': 126.0, 'min_area': 13104.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 155.0, 'median_h': 172.5, 'median_area': 26484.5, 'frames': 199}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'min_h': 244.0, 'min_area': 20631.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 151.0, 'median_area': 18243.25, 'min_h': 122.0, 'min_area': 10315.5, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 126.5, 'median_h': 151.0, 'median_area': 18243.25, 'frames': 200}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'min_h': 260.0, 'min_area': 27040.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 176.0, 'median_area': 22573.5, 'min_h': 130.0, 'min_area': 13520.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 141.0, 'median_h': 176.0, 'median_area': 22573.5, 'frames': 200}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'min_h': 278.0, 'min_area': 18711.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 157.25, 'median_area': 18671.5, 'min_h': 139.0, 'min_area': 9355.5, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 117.0, 'median_h': 157.25, 'median_area': 18671.5, 'frames': 200}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'min_h': 266.0, 'min_area': 24846.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 165.75, 'median_area': 23272.25, 'min_h': 133.0, 'min_area': 12423.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 143.0, 'median_h': 165.75, 'median_area': 23272.25, 'frames': 200}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'min_h': 279.0, 'min_area': 27808.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 167.5, 'median_area': 24166.0, 'min_h': 139.5, 'min_area': 13904.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 140.5, 'median_h': 167.5, 'median_area': 24166.0, 'frames': 200}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}}, 'worst_mask': 'upper', 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'frames': 197}, 'message': ''}}, target_ids={'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=102.12

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 197 | 0.4010 | 0.4154 | 0.5990 | 0.5178 | 0.0000 | - | - | - | 0.55000 | 0.58152 | 3 | 4 | 0.4086 | 5 | 1 | 62 | - | - | - | 254.06 |
| baseline | baseline | upper | 1 | 200 | 197 | 0.3096 | 0.3311 | 0.6904 | 0.5381 | 0.0000 | - | - | - | 0.52000 | 0.59412 | 8 | 9 | 0.3697 | 10 | 0 | 61 | - | - | - | 242.07 |
| part_matching | part_matching_off | none | 1 | 200 | 197 | 0.4010 | 0.4154 | 0.5990 | 0.5178 | 0.0000 | - | - | - | 0.55000 | 0.58152 | 3 | 4 | 0.4086 | 5 | 1 | 62 | - | - | - | 242.68 |
| part_matching | part_matching_off | upper | 1 | 200 | 197 | 0.3096 | 0.3311 | 0.6904 | 0.5381 | 0.0000 | - | - | - | 0.52000 | 0.59412 | 8 | 9 | 0.3697 | 10 | 0 | 61 | - | - | - | 230.95 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | 1 | forced | partial-view | upper | 148x166 | 197 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.4010 | 0.4010 | +0.0000 | 0.55000 | 0.55000 | +0.00000 | 254.06 | 242.68 |
| upper | 0.3096 | 0.3096 | +0.0000 | 0.52000 | 0.52000 | +0.00000 | 242.07 | 230.95 |

## Claims (delta vs the same-mask baseline)

- [part_matching] part_matching_off (mask=none): TSR 0.4010 vs 0.4010 (+0.0000) | FAR 0.55000 vs 0.55000 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.4154 vs 0.4154 | ms/frame 242.68 vs 254.06 (-11.38)
- [part_matching] part_matching_off (mask=upper): TSR 0.3096 vs 0.3096 (+0.0000) | FAR 0.52000 vs 0.52000 (+0.00000) | FA events 8 vs 8 (+0) | precision 0.3311 vs 0.3311 | ms/frame 230.95 vs 242.07 (-11.12)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0007 | baseline | none | 1 | 200 | 0.4010 | 0.4154 | 0.55000 | 3 | 0.4086 | 5 | 254.06 |
| dancetrack0007 | baseline | upper | 1 | 200 | 0.3096 | 0.3311 | 0.52000 | 8 | 0.3697 | 10 | 242.07 |
| dancetrack0007 | part_matching_off | none | 1 | 200 | 0.4010 | 0.4154 | 0.55000 | 3 | 0.4086 | 5 | 242.68 |
| dancetrack0007 | part_matching_off | upper | 1 | 200 | 0.3096 | 0.3311 | 0.52000 | 8 | 0.3697 | 10 | 230.95 |
