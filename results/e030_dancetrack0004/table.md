# Clicked-target strategy ablation

Config: ablate=anchor_adaptive, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=136.37

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_adaptive | anchor_adapt_k2 | none | 1 | 200 | 200 | 0.3600 | 0.3262 | 0.6400 | 0.0850 | - | - | - | - | 0.07500 | 0.18750 | 4 | 4 | 0.8256 | 5 | 5 | 43 | - | - | - | 191.93 |
| anchor_adaptive | anchor_adapt_k3 | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 154.99 |
| anchor_adaptive | anchor_adapt_k3_hi0.95 | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 149.32 |
| anchor_adaptive | anchor_adapt_k4 | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 154.91 |
| anchor_adaptive | anchor_adapt_k4_c0.55 | none | 1 | 200 | 200 | 0.1900 | 0.1658 | 0.8100 | 0.0800 | - | - | - | - | 0.07000 | 0.31818 | 7 | 7 | 0.6957 | 8 | 7 | 16 | - | - | - | 142.87 |
| anchor_adaptive | anchor_adapt_k4_c0.60 | none | 1 | 200 | 200 | 0.2600 | 0.2460 | 0.7400 | 0.1400 | - | - | - | - | 0.11500 | 0.29167 | 7 | 7 | 0.6849 | 7 | 6 | 22 | - | - | - | 155.32 |
| anchor_adaptive | anchor_adapt_k4_c0.65 | none | 1 | 200 | 200 | 0.2400 | 0.2203 | 0.7600 | 0.0800 | - | - | - | - | 0.06000 | 0.19672 | 6 | 6 | 0.7966 | 6 | 5 | 32 | - | - | - | 159.68 |
| baseline | baseline | none | 1 | 200 | 200 | 0.2350 | 0.2176 | 0.7650 | 0.1650 | - | - | - | - | 0.15500 | 0.40260 | 3 | 3 | 0.6026 | 3 | 4 | 32 | - | - | - | 182.71 |

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

- [anchor_adaptive] anchor_adapt_k2 (mask=none): TSR 0.3600 vs 0.2350 (+0.1250) | FAR 0.07500 vs 0.15500 (-0.08000) | FA events 4 vs 3 (+1) | precision 0.3262 vs 0.2176 | ms/frame 191.93 vs 182.71 (+9.22)
- [anchor_adaptive] anchor_adapt_k3 (mask=none): TSR 0.2350 vs 0.2350 (+0.0000) | FAR 0.15500 vs 0.15500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.2176 vs 0.2176 | ms/frame 154.99 vs 182.71 (-27.72)
- [anchor_adaptive] anchor_adapt_k3_hi0.95 (mask=none): TSR 0.2350 vs 0.2350 (+0.0000) | FAR 0.15500 vs 0.15500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.2176 vs 0.2176 | ms/frame 149.32 vs 182.71 (-33.39)
- [anchor_adaptive] anchor_adapt_k4 (mask=none): TSR 0.2350 vs 0.2350 (+0.0000) | FAR 0.15500 vs 0.15500 (+0.00000) | FA events 3 vs 3 (+0) | precision 0.2176 vs 0.2176 | ms/frame 154.91 vs 182.71 (-27.80)
- [anchor_adaptive] anchor_adapt_k4_c0.55 (mask=none): TSR 0.1900 vs 0.2350 (-0.0450) | FAR 0.07000 vs 0.15500 (-0.08500) | FA events 7 vs 3 (+4) | precision 0.1658 vs 0.2176 | ms/frame 142.87 vs 182.71 (-39.84)
- [anchor_adaptive] anchor_adapt_k4_c0.60 (mask=none): TSR 0.2600 vs 0.2350 (+0.0250) | FAR 0.11500 vs 0.15500 (-0.04000) | FA events 7 vs 3 (+4) | precision 0.2460 vs 0.2176 | ms/frame 155.32 vs 182.71 (-27.39)
- [anchor_adaptive] anchor_adapt_k4_c0.65 (mask=none): TSR 0.2400 vs 0.2350 (+0.0050) | FAR 0.06000 vs 0.15500 (-0.09500) | FA events 6 vs 3 (+3) | precision 0.2203 vs 0.2176 | ms/frame 159.68 vs 182.71 (-23.03)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 182.71 |
| dancetrack0004 | anchor_adapt_k2 | none | 3 | 200 | 0.3600 | 0.3262 | 0.07500 | 4 | 0.8256 | 5 | 191.93 |
| dancetrack0004 | anchor_adapt_k3 | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 154.99 |
| dancetrack0004 | anchor_adapt_k4 | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 154.91 |
| dancetrack0004 | anchor_adapt_k4_c0.55 | none | 3 | 200 | 0.1900 | 0.1658 | 0.07000 | 7 | 0.6957 | 8 | 142.87 |
| dancetrack0004 | anchor_adapt_k4_c0.60 | none | 3 | 200 | 0.2600 | 0.2460 | 0.11500 | 7 | 0.6849 | 7 | 155.32 |
| dancetrack0004 | anchor_adapt_k4_c0.65 | none | 3 | 200 | 0.2400 | 0.2203 | 0.06000 | 6 | 0.7966 | 6 | 159.68 |
| dancetrack0004 | anchor_adapt_k3_hi0.95 | none | 3 | 200 | 0.2350 | 0.2176 | 0.15500 | 3 | 0.6026 | 3 | 149.32 |
