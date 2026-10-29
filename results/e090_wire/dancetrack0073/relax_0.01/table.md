# Clicked-target strategy ablation

Config: ablate=all, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0073, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0073': {'target_id': 7, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 230.5, 'median_h': 316.5, 'median_area': 70699.5, 'min_h': 247.0, 'min_area': 35642.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 230.5, 'median_h': 316.5, 'median_area': 70699.5, 'frames': 300}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 242.0, 'median_h': 393.0, 'median_area': 95164.0, 'min_h': 250.0, 'min_area': 32079.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 242.0, 'median_h': 393.0, 'median_area': 95164.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 311.5, 'median_h': 485.0, 'median_area': 150894.0, 'min_h': 372.0, 'min_area': 81468.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 311.5, 'median_h': 485.0, 'median_area': 150894.0, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 405.0, 'median_h': 573.0, 'median_area': 231256.0, 'min_h': 449.0, 'min_area': 149574.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 405.0, 'median_h': 573.0, 'median_area': 231256.0, 'frames': 299}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 179, 'first': 121, 'median_w': 131.0, 'median_h': 321.0, 'median_area': 42504.0, 'min_h': 249.0, 'min_area': 19635.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 131.0, 'median_h': 321.0, 'median_area': 42504.0, 'frames': 179}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 322.5, 'median_h': 493.5, 'median_area': 163576.0, 'min_h': 371.0, 'min_area': 76266.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 322.5, 'median_h': 493.5, 'median_area': 163576.0, 'frames': 300}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 267.5, 'median_h': 428.0, 'median_area': 113558.0, 'min_h': 304.0, 'min_area': 60450.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 267.5, 'median_h': 428.0, 'median_area': 113558.0, 'frames': 300}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 250, 'first': 0, 'median_w': 237.0, 'median_h': 337.0, 'median_area': 79496.0, 'min_h': 253.0, 'min_area': 40296.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 237.0, 'median_h': 337.0, 'median_area': 79496.0, 'frames': 250}, 8: {'target_id': 8, 'masks': {'none': {'mode': 'none', 'frames': 213, 'first': 80, 'median_w': 153.0, 'median_h': 393.0, 'median_area': 58236.0, 'min_h': 205.0, 'min_area': 22785.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 153.0, 'median_h': 393.0, 'median_area': 58236.0, 'frames': 213}, 9: {'target_id': 9, 'masks': {'none': {'mode': 'none', 'frames': 228, 'first': 72, 'median_w': 186.0, 'median_h': 456.5, 'median_area': 76248.5, 'min_h': 215.0, 'min_area': 31416.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 186.0, 'median_h': 456.5, 'median_area': 76248.5, 'frames': 228}, 10: {'target_id': 10, 'masks': {'none': {'mode': 'none', 'frames': 279, 'first': 21, 'median_w': 189.0, 'median_h': 491.0, 'median_area': 90835.0, 'min_h': 213.0, 'min_area': 27492.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.0, 'median_h': 491.0, 'median_area': 90835.0, 'frames': 279}, 11: {'target_id': 11, 'masks': {'none': {'mode': 'none', 'frames': 257, 'first': 0, 'median_w': 189.0, 'median_h': 460.0, 'median_area': 85100.0, 'min_h': 204.0, 'min_area': 26703.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.0, 'median_h': 460.0, 'median_area': 85100.0, 'frames': 257}, 12: {'target_id': 12, 'masks': {'none': {'mode': 'none', 'frames': 174, 'first': 126, 'median_w': 172.5, 'median_h': 440.5, 'median_area': 72305.0, 'min_h': 350.0, 'min_area': 37471.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 172.5, 'median_h': 440.5, 'median_area': 72305.0, 'frames': 174}, 13: {'target_id': 13, 'masks': {'none': {'mode': 'none', 'frames': 134, 'first': 27, 'median_w': 148.0, 'median_h': 357.0, 'median_area': 55780.0, 'min_h': 188.0, 'min_area': 25273.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 148.0, 'median_h': 357.0, 'median_area': 55780.0, 'frames': 134}}, 'selected': {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 250, 'first': 0, 'median_w': 237.0, 'median_h': 337.0, 'median_area': 79496.0, 'min_h': 253.0, 'min_area': 40296.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 237.0, 'median_h': 337.0, 'median_area': 79496.0, 'frames': 250}, 'message': ''}}, target_ids={'dancetrack0073': 7}, target_strategy=stressed, wall_seconds=240.41

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e090_wire | relax_0.01 | none | 1 | 300 | 250 | 0.5160 | 0.3932 | 0.4840 | 0.1560 | 0.6667 | 26.00 | 38.80 | 42.00 | 0.15333 | 0.25862 | 5 | 5 | 0.7341 | 7 | 4 | 61 | - | - | - | 796.65 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0073 | 7 | forced | partial-view | none | 237x337 | 250 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0073 | relax_0.01 | none | 7 | 300 | 0.5160 | 0.3932 | 0.15333 | 5 | 0.7341 | 7 | 796.65 |
