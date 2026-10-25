# Clicked-target strategy ablation

Config: ablate=trail, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=72.52

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | none | 1 | 200 | 200 | 0.4150 | 0.4136 | 0.5850 | 0.0700 | - | - | - | - | 0.04000 | 0.08602 | 5 | 5 | 0.9111 | 7 | 8 | 67 | - | - | - | 183.53 |
| trail | trail_w0.5 | none | 1 | 200 | 200 | 0.0950 | 0.1153 | 0.9050 | 0.1250 | - | - | - | - | 0.11500 | 0.42857 | 9 | 10 | 0.4390 | 10 | 6 | 7 | - | - | - | 160.31 |
| trail | trail_w1 | none | 1 | 200 | 200 | 0.0800 | 0.1039 | 0.9200 | 0.1350 | - | - | - | - | 0.07500 | 0.36585 | 10 | 10 | 0.5000 | 8 | 5 | 7 | - | - | - | 152.74 |
| trail | trail_w2 | none | 1 | 200 | 200 | 0.0800 | 0.1039 | 0.9200 | 0.1350 | - | - | - | - | 0.07500 | 0.36585 | 10 | 10 | 0.5000 | 8 | 5 | 7 | - | - | - | 151.38 |

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

- [trail] trail_w0.5 (mask=none): TSR 0.0950 vs 0.4150 (-0.3200) | FAR 0.11500 vs 0.04000 (+0.07500) | FA events 9 vs 5 (+4) | precision 0.1153 vs 0.4136 | ms/frame 160.31 vs 183.53 (-23.23)
- [trail] trail_w1 (mask=none): TSR 0.0800 vs 0.4150 (-0.3350) | FAR 0.07500 vs 0.04000 (+0.03500) | FA events 10 vs 5 (+5) | precision 0.1039 vs 0.4136 | ms/frame 152.74 vs 183.53 (-30.79)
- [trail] trail_w2 (mask=none): TSR 0.0800 vs 0.4150 (-0.3350) | FAR 0.07500 vs 0.04000 (+0.03500) | FA events 10 vs 5 (+5) | precision 0.1039 vs 0.4136 | ms/frame 151.38 vs 183.53 (-32.16)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.4150 | 0.4136 | 0.04000 | 5 | 0.9111 | 7 | 183.53 |
| dancetrack0004 | trail_w0.5 | none | 3 | 200 | 0.0950 | 0.1153 | 0.11500 | 9 | 0.4390 | 10 | 160.31 |
| dancetrack0004 | trail_w1 | none | 3 | 200 | 0.0800 | 0.1039 | 0.07500 | 10 | 0.5000 | 8 | 152.74 |
| dancetrack0004 | trail_w2 | none | 3 | 200 | 0.0800 | 0.1039 | 0.07500 | 10 | 0.5000 | 8 | 151.38 |
