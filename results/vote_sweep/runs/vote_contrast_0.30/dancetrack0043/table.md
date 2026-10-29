# Clicked-target strategy ablation

Config: ablate=vote, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0043, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0043': {'target_id': 5, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 65, 'first': 0, 'median_w': 247.0, 'median_h': 465.0, 'median_area': 119250.0, 'min_h': 206.0, 'min_area': 13080.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 247.0, 'median_h': 465.0, 'median_area': 119250.0, 'frames': 65}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 84, 'first': 0, 'median_w': 282.5, 'median_h': 528.0, 'median_area': 150945.0, 'min_h': 224.0, 'min_area': 22950.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 282.5, 'median_h': 528.0, 'median_area': 150945.0, 'frames': 84}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 69, 'first': 0, 'median_w': 171.0, 'median_h': 363.0, 'median_area': 62928.0, 'min_h': 210.0, 'min_area': 12180.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 171.0, 'median_h': 363.0, 'median_area': 62928.0, 'frames': 69}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 80, 'first': 0, 'median_w': 164.0, 'median_h': 408.0, 'median_area': 64083.0, 'min_h': 240.0, 'min_area': 18960.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 164.0, 'median_h': 408.0, 'median_area': 64083.0, 'frames': 80}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 89, 'first': 0, 'median_w': 166.0, 'median_h': 339.0, 'median_area': 57305.0, 'min_h': 301.0, 'min_area': 40560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 166.0, 'median_h': 339.0, 'median_area': 57305.0, 'frames': 89}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 72, 'first': 0, 'median_w': 131.5, 'median_h': 303.5, 'median_area': 39304.0, 'min_h': 273.0, 'min_area': 29975.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 131.5, 'median_h': 303.5, 'median_area': 39304.0, 'frames': 72}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 91, 'first': 0, 'median_w': 282.0, 'median_h': 517.0, 'median_area': 152579.0, 'min_h': 230.0, 'min_area': 25300.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 282.0, 'median_h': 517.0, 'median_area': 152579.0, 'frames': 91}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 92, 'first': 0, 'median_w': 245.0, 'median_h': 440.0, 'median_area': 105161.0, 'min_h': 302.0, 'min_area': 42673.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 245.0, 'median_h': 440.0, 'median_area': 105161.0, 'frames': 92}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 12, 'first': 76, 'median_w': 143.5, 'median_h': 274.5, 'median_area': 38976.0, 'min_h': 245.0, 'min_area': 23845.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 143.5, 'median_h': 274.5, 'median_area': 38976.0, 'frames': 12}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 16, 'first': 64, 'median_w': 96.5, 'median_h': 290.0, 'median_area': 30115.0, 'min_h': 272.0, 'min_area': 22386.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 96.5, 'median_h': 290.0, 'median_area': 30115.0, 'frames': 16}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 24, 'first': 56, 'median_w': 140.5, 'median_h': 318.0, 'median_area': 47651.0, 'min_h': 237.0, 'min_area': 22008.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 140.5, 'median_h': 318.0, 'median_area': 47651.0, 'frames': 24}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 47, 'first': 32, 'median_w': 137.0, 'median_h': 343.0, 'median_area': 45965.0, 'min_h': 278.0, 'min_area': 33124.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 137.0, 'median_h': 343.0, 'median_area': 45965.0, 'frames': 47}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 78, 'first': 0, 'median_w': 157.0, 'median_h': 361.0, 'median_area': 55498.5, 'min_h': 252.0, 'min_area': 19656.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 157.0, 'median_h': 361.0, 'median_area': 55498.5, 'frames': 78}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 18, 'first': 70, 'median_w': 124.5, 'median_h': 329.5, 'median_area': 41335.5, 'min_h': 314.0, 'min_area': 34900.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 124.5, 'median_h': 329.5, 'median_area': 41335.5, 'frames': 18}}, 'selected': {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 72, 'first': 0, 'median_w': 131.5, 'median_h': 303.5, 'median_area': 39304.0, 'min_h': 273.0, 'min_area': 29975.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 131.5, 'median_h': 303.5, 'median_area': 39304.0, 'frames': 72}, 'message': ''}}, target_ids={'dancetrack0043': 5}, target_strategy=stressed, wall_seconds=19.82

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vote_sweep | vote_contrast_0.30 | none | 1 | 92 | 72 | 0.1111 | 0.0681 | 0.8889 | 0.0833 | 0.0000 | - | - | - | 0.04348 | 0.16667 | 2 | 1 | 0.6667 | 3 | 1 | 7 | - | - | - | 186.23 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0043 | 5 | forced | partial-view | none | 132x304 | 72 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0043 | vote_contrast_0.30 | none | 5 | 92 | 0.1111 | 0.0681 | 0.04348 | 2 | 0.6667 | 3 | 186.23 |
