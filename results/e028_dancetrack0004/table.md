# Clicked-target strategy ablation

Config: ablate=anchor_ceiling, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=106.69

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_ceiling | anchor_ceiling_0.45 | none | 1 | 200 | 200 | 0.2550 | 0.2319 | 0.7450 | 0.1650 | - | - | - | - | 0.14500 | 0.34524 | 2 | 2 | 0.6375 | 3 | 4 | 41 | - | - | - | 191.97 |
| anchor_ceiling | anchor_ceiling_0.55 | none | 1 | 200 | 200 | 0.4100 | 0.4022 | 0.5900 | 0.2350 | - | - | - | - | 0.21000 | 0.31783 | 5 | 5 | 0.6613 | 6 | 3 | 35 | - | - | - | 175.81 |
| anchor_ceiling | anchor_ceiling_0.65 | none | 1 | 200 | 200 | 0.1050 | 0.1409 | 0.8950 | 0.0650 | - | - | - | - | 0.03500 | 0.23333 | 5 | 5 | 0.7500 | 8 | 7 | 6 | - | - | - | 141.58 |
| anchor_ceiling | anchor_search_0.55 | none | 1 | 200 | 200 | 0.4100 | 0.4022 | 0.5900 | 0.2350 | - | - | - | - | 0.21000 | 0.31783 | 5 | 5 | 0.6613 | 6 | 3 | 35 | - | - | - | 170.78 |
| anchor_ceiling | anchor_search_0.65 | none | 1 | 200 | 200 | 0.1050 | 0.1409 | 0.8950 | 0.0650 | - | - | - | - | 0.03500 | 0.23333 | 5 | 5 | 0.7500 | 8 | 7 | 6 | - | - | - | 141.36 |
| baseline | baseline | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 177.46 |

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

- [anchor_ceiling] anchor_ceiling_0.45 (mask=none): TSR 0.2550 vs 0.2350 (+0.0200) | FAR 0.14500 vs 0.15500 (-0.01000) | FA events 2 vs 3 (-1) | precision 0.2319 vs 0.2176 | ms/frame 191.97 vs 177.46 (+14.50)
- [anchor_ceiling] anchor_ceiling_0.55 (mask=none): TSR 0.4100 vs 0.2350 (+0.1750) | FAR 0.21000 vs 0.15500 (+0.05500) | FA events 5 vs 3 (+2) | precision 0.4022 vs 0.2176 | ms/frame 175.81 vs 177.46 (-1.65)
- [anchor_ceiling] anchor_ceiling_0.65 (mask=none): TSR 0.1050 vs 0.2350 (-0.1300) | FAR 0.03500 vs 0.15500 (-0.12000) | FA events 5 vs 3 (+2) | precision 0.1409 vs 0.2176 | ms/frame 141.58 vs 177.46 (-35.88)
- [anchor_ceiling] anchor_search_0.55 (mask=none): TSR 0.4100 vs 0.2350 (+0.1750) | FAR 0.21000 vs 0.15500 (+0.05500) | FA events 5 vs 3 (+2) | precision 0.4022 vs 0.2176 | ms/frame 170.78 vs 177.46 (-6.69)
- [anchor_ceiling] anchor_search_0.65 (mask=none): TSR 0.1050 vs 0.2350 (-0.1300) | FAR 0.03500 vs 0.15500 (-0.12000) | FA events 5 vs 3 (+2) | precision 0.1409 vs 0.2176 | ms/frame 141.36 vs 177.46 (-36.11)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 177.46 |
| dancetrack0004 | anchor_ceiling_0.45 | none | 3 | 200 | 0.2550 | 0.2319 | 0.14500 | 2 | 0.6375 | 3 | 191.97 |
| dancetrack0004 | anchor_ceiling_0.55 | none | 3 | 200 | 0.4100 | 0.4022 | 0.21000 | 5 | 0.6613 | 6 | 175.81 |
| dancetrack0004 | anchor_ceiling_0.65 | none | 3 | 200 | 0.1050 | 0.1409 | 0.03500 | 5 | 0.7500 | 8 | 141.58 |
| dancetrack0004 | anchor_search_0.55 | none | 3 | 200 | 0.4100 | 0.4022 | 0.21000 | 5 | 0.6613 | 6 | 170.78 |
| dancetrack0004 | anchor_search_0.65 | none | 3 | 200 | 0.1050 | 0.1409 | 0.03500 | 5 | 0.7500 | 8 | 141.36 |
