# Clicked-target strategy ablation

Config: ablate=anchor_adaptive, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0005, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0005': {'target_id': 2, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'frames': 200}}, 'selected': {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0005': 2}, target_strategy=stressed, wall_seconds=135.13

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_adaptive | anchor_adapt_k2 | none | 1 | 200 | 200 | 0.6200 | 0.5792 | 0.3800 | 0.2950 | - | - | - | - | 0.29500 | 0.31844 | 5 | 5 | 0.6776 | 5 | 1 | 115 | - | - | - | 167.26 |
| anchor_adaptive | anchor_adapt_k3 | none | 1 | 200 | 200 | 0.1750 | 0.1215 | 0.8250 | 0.5300 | - | - | - | - | 0.63500 | 0.73759 | 6 | 15 | 0.2112 | 3 | 1 | 34 | - | - | - | 157.72 |
| anchor_adaptive | anchor_adapt_k3_hi0.95 | none | 1 | 200 | 200 | 0.1750 | 0.1215 | 0.8250 | 0.7050 | - | - | - | - | 0.71000 | 0.80114 | 3 | 4 | 0.1932 | 3 | 1 | 34 | - | - | - | 161.26 |
| anchor_adaptive | anchor_adapt_k4 | none | 1 | 200 | 200 | 0.6750 | 0.5867 | 0.3250 | 0.1200 | - | - | - | - | 0.20500 | 0.15723 | 3 | 4 | 0.7657 | 2 | 1 | 101 | - | - | - | 149.01 |
| anchor_adaptive | anchor_adapt_k4_c0.55 | none | 1 | 200 | 200 | 0.2100 | 0.2409 | 0.7900 | 0.0950 | - | - | - | - | 0.12000 | 0.25000 | 7 | 5 | 0.6364 | 7 | 6 | 16 | - | - | - | 140.69 |
| anchor_adaptive | anchor_adapt_k4_c0.60 | none | 1 | 200 | 200 | 0.5700 | 0.5029 | 0.4300 | 0.0500 | - | - | - | - | 0.04500 | 0.06604 | 4 | 3 | 0.9262 | 6 | 8 | 45 | - | - | - | 153.76 |
| anchor_adaptive | anchor_adapt_k4_c0.65 | none | 1 | 200 | 200 | 0.7050 | 0.6041 | 0.2950 | 0.0650 | - | - | - | - | 0.09500 | 0.10714 | 3 | 4 | 0.8797 | 3 | 4 | 69 | - | - | - | 141.87 |
| baseline | baseline | none | 1 | 200 | 200 | 0.1800 | 0.1344 | 0.8200 | 0.8050 | - | - | - | - | 0.81000 | 0.82234 | 1 | 1 | 0.1777 | 2 | 1 | 34 | - | - | - | 178.53 |

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

- [anchor_adaptive] anchor_adapt_k2 (mask=none): TSR 0.6200 vs 0.1800 (+0.4400) | FAR 0.29500 vs 0.81000 (-0.51500) | FA events 5 vs 1 (+4) | precision 0.5792 vs 0.1344 | ms/frame 167.26 vs 178.53 (-11.27)
- [anchor_adaptive] anchor_adapt_k3 (mask=none): TSR 0.1750 vs 0.1800 (-0.0050) | FAR 0.63500 vs 0.81000 (-0.17500) | FA events 6 vs 1 (+5) | precision 0.1215 vs 0.1344 | ms/frame 157.72 vs 178.53 (-20.82)
- [anchor_adaptive] anchor_adapt_k3_hi0.95 (mask=none): TSR 0.1750 vs 0.1800 (-0.0050) | FAR 0.71000 vs 0.81000 (-0.10000) | FA events 3 vs 1 (+2) | precision 0.1215 vs 0.1344 | ms/frame 161.26 vs 178.53 (-17.27)
- [anchor_adaptive] anchor_adapt_k4 (mask=none): TSR 0.6750 vs 0.1800 (+0.4950) | FAR 0.20500 vs 0.81000 (-0.60500) | FA events 3 vs 1 (+2) | precision 0.5867 vs 0.1344 | ms/frame 149.01 vs 178.53 (-29.53)
- [anchor_adaptive] anchor_adapt_k4_c0.55 (mask=none): TSR 0.2100 vs 0.1800 (+0.0300) | FAR 0.12000 vs 0.81000 (-0.69000) | FA events 7 vs 1 (+6) | precision 0.2409 vs 0.1344 | ms/frame 140.69 vs 178.53 (-37.84)
- [anchor_adaptive] anchor_adapt_k4_c0.60 (mask=none): TSR 0.5700 vs 0.1800 (+0.3900) | FAR 0.04500 vs 0.81000 (-0.76500) | FA events 4 vs 1 (+3) | precision 0.5029 vs 0.1344 | ms/frame 153.76 vs 178.53 (-24.78)
- [anchor_adaptive] anchor_adapt_k4_c0.65 (mask=none): TSR 0.7050 vs 0.1800 (+0.5250) | FAR 0.09500 vs 0.81000 (-0.71500) | FA events 3 vs 1 (+2) | precision 0.6041 vs 0.1344 | ms/frame 141.87 vs 178.53 (-36.67)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | baseline | none | 2 | 200 | 0.1800 | 0.1344 | 0.81000 | 1 | 0.1777 | 2 | 178.53 |
| dancetrack0005 | anchor_adapt_k2 | none | 2 | 200 | 0.6200 | 0.5792 | 0.29500 | 5 | 0.6776 | 5 | 167.26 |
| dancetrack0005 | anchor_adapt_k3 | none | 2 | 200 | 0.1750 | 0.1215 | 0.63500 | 6 | 0.2112 | 3 | 157.72 |
| dancetrack0005 | anchor_adapt_k4 | none | 2 | 200 | 0.6750 | 0.5867 | 0.20500 | 3 | 0.7657 | 2 | 149.01 |
| dancetrack0005 | anchor_adapt_k4_c0.55 | none | 2 | 200 | 0.2100 | 0.2409 | 0.12000 | 7 | 0.6364 | 7 | 140.69 |
| dancetrack0005 | anchor_adapt_k4_c0.60 | none | 2 | 200 | 0.5700 | 0.5029 | 0.04500 | 4 | 0.9262 | 6 | 153.76 |
| dancetrack0005 | anchor_adapt_k4_c0.65 | none | 2 | 200 | 0.7050 | 0.6041 | 0.09500 | 3 | 0.8797 | 3 | 141.87 |
| dancetrack0005 | anchor_adapt_k3_hi0.95 | none | 2 | 200 | 0.1750 | 0.1215 | 0.71000 | 3 | 0.1932 | 3 | 161.26 |
