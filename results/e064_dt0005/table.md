# Clicked-target strategy ablation

Config: ablate=audit, allow_invisible_target=False, data_root=data\raw\dancetrack\val\dancetrack0005, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0005': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'frames': 200}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0005': 0}, target_strategy=stressed, wall_seconds=399.71

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audit | audit_1ref_5_m05 | none | 1 | 200 | 200 | 0.2950 | 0.2873 | 0.7050 | 0.7050 | - | - | - | - | 0.70500 | 0.70500 | 1 | 1 | 0.2915 | 4 | 2 | 35 | - | - | - | 514.59 |
| audit | audit_2ref_3_m05 | none | 1 | 200 | 200 | 0.4000 | 0.3700 | 0.6000 | 0.5200 | - | - | - | - | 0.51500 | 0.55978 | 1 | 1 | 0.4341 | 2 | 2 | 42 | - | - | - | 472.31 |
| audit | audit_2ref_5_m05 | none | 1 | 200 | 200 | 0.4000 | 0.3700 | 0.6000 | 0.5200 | - | - | - | - | 0.51500 | 0.55978 | 1 | 1 | 0.4341 | 2 | 2 | 42 | - | - | - | 490.84 |
| audit | audit_3_m05 | none | 1 | 200 | 200 | 0.2000 | 0.1980 | 0.8000 | 0.7950 | - | - | - | - | 0.79000 | 0.79397 | 3 | 3 | 0.1980 | 6 | 2 | 35 | - | - | - | 520.38 |
| audit | audit_5_m05 | none | 1 | 200 | 200 | 0.2950 | 0.2873 | 0.7050 | 0.7050 | - | - | - | - | 0.70500 | 0.70500 | 1 | 1 | 0.2915 | 4 | 2 | 35 | - | - | - | 480.13 |
| audit | audit_5_m10 | none | 1 | 200 | 200 | 0.4000 | 0.3700 | 0.6000 | 0.5200 | - | - | - | - | 0.51500 | 0.55978 | 1 | 1 | 0.4341 | 2 | 2 | 42 | - | - | - | 450.93 |
| audit | audit_off | none | 1 | 200 | 200 | 0.4000 | 0.3700 | 0.6000 | 0.5200 | - | - | - | - | 0.51500 | 0.55978 | 1 | 1 | 0.4341 | 2 | 2 | 42 | - | - | - | 408.34 |
| baseline | baseline | none | 1 | 200 | 200 | 0.4000 | 0.3700 | 0.6000 | 0.5200 | - | - | - | - | 0.51500 | 0.55978 | 1 | 1 | 0.4341 | 2 | 2 | 42 | - | - | - | 408.61 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | 0 | observable | partial-view | none | 219x466 | 200 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [audit] audit_1ref_5_m05 (mask=none): TSR 0.2950 vs 0.4000 (-0.1050) | FAR 0.70500 vs 0.51500 (+0.19000) | FA events 1 vs 1 (+0) | precision 0.2873 vs 0.3700 | ms/frame 514.59 vs 408.61 (+105.97)
- [audit] audit_2ref_3_m05 (mask=none): TSR 0.4000 vs 0.4000 (+0.0000) | FAR 0.51500 vs 0.51500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.3700 vs 0.3700 | ms/frame 472.31 vs 408.61 (+63.69)
- [audit] audit_2ref_5_m05 (mask=none): TSR 0.4000 vs 0.4000 (+0.0000) | FAR 0.51500 vs 0.51500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.3700 vs 0.3700 | ms/frame 490.84 vs 408.61 (+82.23)
- [audit] audit_3_m05 (mask=none): TSR 0.2000 vs 0.4000 (-0.2000) | FAR 0.79000 vs 0.51500 (+0.27500) | FA events 3 vs 1 (+2) | precision 0.1980 vs 0.3700 | ms/frame 520.38 vs 408.61 (+111.77)
- [audit] audit_5_m05 (mask=none): TSR 0.2950 vs 0.4000 (-0.1050) | FAR 0.70500 vs 0.51500 (+0.19000) | FA events 1 vs 1 (+0) | precision 0.2873 vs 0.3700 | ms/frame 480.13 vs 408.61 (+71.52)
- [audit] audit_5_m10 (mask=none): TSR 0.4000 vs 0.4000 (+0.0000) | FAR 0.51500 vs 0.51500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.3700 vs 0.3700 | ms/frame 450.93 vs 408.61 (+42.32)
- [audit] audit_off (mask=none): TSR 0.4000 vs 0.4000 (+0.0000) | FAR 0.51500 vs 0.51500 (+0.00000) | FA events 1 vs 1 (+0) | precision 0.3700 vs 0.3700 | ms/frame 408.34 vs 408.61 (-0.27)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0005 | baseline | none | 0 | 200 | 0.4000 | 0.3700 | 0.51500 | 1 | 0.4341 | 2 | 408.61 |
| dancetrack0005 | audit_off | none | 0 | 200 | 0.4000 | 0.3700 | 0.51500 | 1 | 0.4341 | 2 | 408.34 |
| dancetrack0005 | audit_3_m05 | none | 0 | 200 | 0.2000 | 0.1980 | 0.79000 | 3 | 0.1980 | 6 | 520.38 |
| dancetrack0005 | audit_5_m05 | none | 0 | 200 | 0.2950 | 0.2873 | 0.70500 | 1 | 0.2915 | 4 | 480.13 |
| dancetrack0005 | audit_5_m10 | none | 0 | 200 | 0.4000 | 0.3700 | 0.51500 | 1 | 0.4341 | 2 | 450.93 |
| dancetrack0005 | audit_2ref_3_m05 | none | 0 | 200 | 0.4000 | 0.3700 | 0.51500 | 1 | 0.4341 | 2 | 472.31 |
| dancetrack0005 | audit_2ref_5_m05 | none | 0 | 200 | 0.4000 | 0.3700 | 0.51500 | 1 | 0.4341 | 2 | 490.84 |
| dancetrack0005 | audit_1ref_5_m05 | none | 0 | 200 | 0.2950 | 0.2873 | 0.70500 | 1 | 0.2915 | 4 | 514.59 |
