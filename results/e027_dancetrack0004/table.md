# Clicked-target strategy ablation

Config: ablate=anchor_ceiling, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=69.11

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_ceiling | anchor_ceiling_0.45 | none | 1 | 200 | 200 | 0.1150 | 0.1183 | 0.8850 | 0.0700 | - | - | - | - | 0.04500 | 0.25000 | 7 | 6 | 0.7188 | 11 | 7 | 6 | - | - | - | 162.78 |
| anchor_ceiling | anchor_ceiling_0.55 | none | 1 | 200 | 200 | 0.1050 | 0.1462 | 0.8950 | 0.1900 | - | - | - | - | 0.16500 | 0.48148 | 10 | 12 | 0.3774 | 11 | 7 | 6 | - | - | - | 151.44 |
| anchor_ceiling | anchor_ceiling_0.65 | none | 1 | 200 | 200 | 0.1050 | 0.1409 | 0.8950 | 0.0650 | - | - | - | - | 0.03500 | 0.23333 | 5 | 5 | 0.7500 | 8 | 7 | 6 | - | - | - | 147.20 |
| baseline | baseline | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 166.64 |

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

- [anchor_ceiling] anchor_ceiling_0.45 (mask=none): TSR 0.1150 vs 0.2350 (-0.1200) | FAR 0.04500 vs 0.15500 (-0.11000) | FA events 7 vs 3 (+4) | precision 0.1183 vs 0.2176 | ms/frame 162.78 vs 166.64 (-3.86)
- [anchor_ceiling] anchor_ceiling_0.55 (mask=none): TSR 0.1050 vs 0.2350 (-0.1300) | FAR 0.16500 vs 0.15500 (+0.01000) | FA events 10 vs 3 (+7) | precision 0.1462 vs 0.2176 | ms/frame 151.44 vs 166.64 (-15.20)
- [anchor_ceiling] anchor_ceiling_0.65 (mask=none): TSR 0.1050 vs 0.2350 (-0.1300) | FAR 0.03500 vs 0.15500 (-0.12000) | FA events 5 vs 3 (+2) | precision 0.1409 vs 0.2176 | ms/frame 147.20 vs 166.64 (-19.43)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 166.64 |
| dancetrack0004 | anchor_ceiling_0.45 | none | 3 | 200 | 0.1150 | 0.1183 | 0.04500 | 7 | 0.7188 | 11 | 162.78 |
| dancetrack0004 | anchor_ceiling_0.55 | none | 3 | 200 | 0.1050 | 0.1462 | 0.16500 | 10 | 0.3774 | 11 | 151.44 |
| dancetrack0004 | anchor_ceiling_0.65 | none | 3 | 200 | 0.1050 | 0.1409 | 0.03500 | 5 | 0.7500 | 8 | 147.20 |
