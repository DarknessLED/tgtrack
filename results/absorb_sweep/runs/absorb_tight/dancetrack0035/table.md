# Clicked-target strategy ablation

Config: ablate=absorb, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0035, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0035': {'target_id': 5, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 279, 'first': 0, 'median_w': 183.0, 'median_h': 343.0, 'median_area': 61600.0, 'min_h': 220.0, 'min_area': 31518.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 183.0, 'median_h': 343.0, 'median_area': 61600.0, 'frames': 279}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 275, 'first': 0, 'median_w': 186.0, 'median_h': 355.0, 'median_area': 64768.0, 'min_h': 185.0, 'min_area': 26598.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 186.0, 'median_h': 355.0, 'median_area': 64768.0, 'frames': 275}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 296, 'first': 0, 'median_w': 194.0, 'median_h': 355.0, 'median_area': 67560.0, 'min_h': 198.0, 'min_area': 28140.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 194.0, 'median_h': 355.0, 'median_area': 67560.0, 'frames': 296}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 292, 'first': 0, 'median_w': 181.0, 'median_h': 351.0, 'median_area': 63205.0, 'min_h': 216.0, 'min_area': 29808.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 181.0, 'median_h': 351.0, 'median_area': 63205.0, 'frames': 292}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 298, 'first': 0, 'median_w': 208.0, 'median_h': 365.0, 'median_area': 76650.0, 'min_h': 220.0, 'min_area': 29260.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 208.0, 'median_h': 365.0, 'median_area': 76650.0, 'frames': 298}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 228, 'first': 3, 'median_w': 178.0, 'median_h': 346.0, 'median_area': 62332.5, 'min_h': 230.0, 'min_area': 32890.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 346.0, 'median_area': 62332.5, 'frames': 228}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 294, 'first': 0, 'median_w': 181.0, 'median_h': 358.0, 'median_area': 63245.0, 'min_h': 210.0, 'min_area': 31034.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 181.0, 'median_h': 358.0, 'median_area': 63245.0, 'frames': 294}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 187.0, 'median_h': 358.0, 'median_area': 66114.0, 'min_h': 175.0, 'min_area': 32900.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 187.0, 'median_h': 358.0, 'median_area': 66114.0, 'frames': 300}}, 'selected': {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 228, 'first': 3, 'median_w': 178.0, 'median_h': 346.0, 'median_area': 62332.5, 'min_h': 230.0, 'min_area': 32890.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 178.0, 'median_h': 346.0, 'median_area': 62332.5, 'frames': 228}, 'message': ''}}, target_ids={'dancetrack0035': 5}, target_strategy=stressed, wall_seconds=77.91

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absorb_sweep | absorb_tight | none | 1 | 300 | 228 | 0.0219 | 0.0158 | 0.9781 | 0.0175 | 0.9167 | 22.09 | 39.00 | 51.00 | 0.05724 | 0.60000 | 12 | 11 | 0.1500 | 14 | 4 | 1 | - | - | - | 242.93 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0035 | 5 | forced | partial-view | none | 178x346 | 228 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0035 | absorb_tight | none | 5 | 297 | 0.0219 | 0.0158 | 0.05724 | 12 | 0.1500 | 14 | 242.93 |
