# Clicked-target strategy ablation

Config: ablate=search_guard, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=100.88

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 200 | 0.2650 | 0.2714 | 0.7350 | 0.2550 | - | - | - | - | 0.22500 | 0.42718 | 7 | 7 | 0.5361 | 10 | 4 | 45 | - | - | - | 191.43 |
| search_guard | search_margin_0.10 | none | 1 | 200 | 200 | 0.3150 | 0.3172 | 0.6850 | 0.2600 | - | - | - | - | 0.21000 | 0.36522 | 6 | 6 | 0.6000 | 5 | 4 | 31 | - | - | - | 165.81 |
| search_guard | search_margin_0.20 | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 154.98 |
| search_guard | search_stm_off | none | 1 | 200 | 200 | 0.0400 | 0.0673 | 0.9600 | 0.6150 | - | - | - | - | 0.54500 | 0.80769 | 12 | 15 | 0.0439 | 10 | 2 | 4 | - | - | - | 192.77 |
| search_guard | search_stm_off_margin_0.10 | none | 1 | 200 | 200 | 0.0550 | 0.1060 | 0.9450 | 0.4850 | - | - | - | - | 0.40000 | 0.74074 | 10 | 10 | 0.1111 | 13 | 5 | 4 | - | - | - | 125.53 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 3 | forced | partial-view | none | 387x446 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [search_guard] search_margin_0.10 (mask=none): TSR 0.3150 vs 0.2650 (+0.0500) | FAR 0.21000 vs 0.22500 (-0.01500) | FA events 6 vs 7 (-1) | precision 0.3172 vs 0.2714 | ms/frame 165.81 vs 191.43 (-25.62)
- [search_guard] search_margin_0.20 (mask=none): TSR 0.2350 vs 0.2650 (-0.0300) | FAR 0.15500 vs 0.22500 (-0.07000) | FA events 3 vs 7 (-4) | precision 0.2176 vs 0.2714 | ms/frame 154.98 vs 191.43 (-36.46)
- [search_guard] search_stm_off (mask=none): TSR 0.0400 vs 0.2650 (-0.2250) | FAR 0.54500 vs 0.22500 (+0.32000) | FA events 12 vs 7 (+5) | precision 0.0673 vs 0.2714 | ms/frame 192.77 vs 191.43 (+1.33)
- [search_guard] search_stm_off_margin_0.10 (mask=none): TSR 0.0550 vs 0.2650 (-0.2100) | FAR 0.40000 vs 0.22500 (+0.17500) | FA events 10 vs 7 (+3) | precision 0.1060 vs 0.2714 | ms/frame 125.53 vs 191.43 (-65.91)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.2650 | 0.2714 | 0.22500 | 7 | 0.5361 | 10 | 191.43 |
| dancetrack0004 | search_stm_off | none | 3 | 200 | 0.0400 | 0.0673 | 0.54500 | 12 | 0.0439 | 10 | 192.77 |
| dancetrack0004 | search_margin_0.10 | none | 3 | 200 | 0.3150 | 0.3172 | 0.21000 | 6 | 0.6000 | 5 | 165.81 |
| dancetrack0004 | search_margin_0.20 | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 154.98 |
| dancetrack0004 | search_stm_off_margin_0.10 | none | 3 | 200 | 0.0550 | 0.1060 | 0.40000 | 10 | 0.1111 | 13 | 125.53 |
