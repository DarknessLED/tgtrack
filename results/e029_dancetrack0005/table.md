# Clicked-target strategy ablation

Config: ablate=anchor_adaptive, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0005, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0005': {'target_id': 2, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'frames': 200}}, 'selected': {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0005': 2}, target_strategy=stressed, wall_seconds=84.74

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_adaptive | anchor_adapt_k2 | none | 1 | 200 | 200 | 0.6200 | 0.5792 | 0.3800 | 0.2950 | - | - | - | - | 0.29500 | 0.31844 | 5 | 5 | 0.6776 | 5 | 1 | 115 | - | - | - | 143.76 |
| anchor_adaptive | anchor_adapt_k3 | none | 1 | 200 | 200 | 0.1750 | 0.1215 | 0.8250 | 0.5300 | - | - | - | - | 0.63500 | 0.73759 | 6 | 15 | 0.2112 | 3 | 1 | 34 | - | - | - | 134.88 |
| anchor_adaptive | anchor_adapt_k3_hi0.95 | none | 1 | 200 | 200 | 0.1750 | 0.1215 | 0.8250 | 0.7050 | - | - | - | - | 0.71000 | 0.80114 | 3 | 4 | 0.1932 | 3 | 1 | 34 | - | - | - | 121.55 |
| anchor_adaptive | anchor_adapt_k4 | none | 1 | 200 | 200 | 0.6750 | 0.5867 | 0.3250 | 0.1200 | - | - | - | - | 0.20500 | 0.15723 | 3 | 4 | 0.7657 | 2 | 1 | 101 | - | - | - | 125.94 |
| baseline | baseline | none | 1 | 200 | 200 | 0.1800 | 0.1344 | 0.8200 | 0.8050 | - | - | - | - | 0.81000 | 0.82234 | 1 | 1 | 0.1777 | 2 | 1 | 34 | - | - | - | 152.75 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | 2 | forced | partial-view | none | 190x424 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [anchor_adaptive] anchor_adapt_k2 (mask=none): TSR 0.6200 vs 0.1800 (+0.4400) | FAR 0.29500 vs 0.81000 (-0.51500) | FA events 5 vs 1 (+4) | precision 0.5792 vs 0.1344 | ms/frame 143.76 vs 152.75 (-8.99)
- [anchor_adaptive] anchor_adapt_k3 (mask=none): TSR 0.1750 vs 0.1800 (-0.0050) | FAR 0.63500 vs 0.81000 (-0.17500) | FA events 6 vs 1 (+5) | precision 0.1215 vs 0.1344 | ms/frame 134.88 vs 152.75 (-17.87)
- [anchor_adaptive] anchor_adapt_k3_hi0.95 (mask=none): TSR 0.1750 vs 0.1800 (-0.0050) | FAR 0.71000 vs 0.81000 (-0.10000) | FA events 3 vs 1 (+2) | precision 0.1215 vs 0.1344 | ms/frame 121.55 vs 152.75 (-31.20)
- [anchor_adaptive] anchor_adapt_k4 (mask=none): TSR 0.6750 vs 0.1800 (+0.4950) | FAR 0.20500 vs 0.81000 (-0.60500) | FA events 3 vs 1 (+2) | precision 0.5867 vs 0.1344 | ms/frame 125.94 vs 152.75 (-26.81)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | baseline | none | 2 | 200 | 0.1800 | 0.1344 | 0.81000 | 1 | 0.1777 | 2 | 152.75 |
| dancetrack0005 | anchor_adapt_k2 | none | 2 | 200 | 0.6200 | 0.5792 | 0.29500 | 5 | 0.6776 | 5 | 143.76 |
| dancetrack0005 | anchor_adapt_k3 | none | 2 | 200 | 0.1750 | 0.1215 | 0.63500 | 6 | 0.2112 | 3 | 134.88 |
| dancetrack0005 | anchor_adapt_k4 | none | 2 | 200 | 0.6750 | 0.5867 | 0.20500 | 3 | 0.7657 | 2 | 125.94 |
| dancetrack0005 | anchor_adapt_k3_hi0.95 | none | 2 | 200 | 0.1750 | 0.1215 | 0.71000 | 3 | 0.1932 | 3 | 121.55 |
