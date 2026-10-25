# Clicked-target strategy ablation

Config: ablate=anchor_bootstrap, allow_invisible_target=False, data_root=data\raw\dancetrack\val, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=120, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 113, 'first': 6, 'median_w': 227.0, 'median_h': 205.0, 'median_area': 36352.0, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 227.0, 'median_h': 205.0, 'median_area': 36352.0, 'frames': 113}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 254.0, 'median_h': 254.0, 'median_area': 59910.5, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 254.0, 'median_h': 254.0, 'median_area': 59910.5, 'frames': 120}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 333.0, 'median_h': 340.5, 'median_area': 94708.5, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 333.0, 'median_h': 340.5, 'median_area': 94708.5, 'frames': 120}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 492.5, 'median_h': 326.0, 'median_area': 145021.0, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 492.5, 'median_h': 326.0, 'median_area': 145021.0, 'frames': 120}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 492.5, 'median_h': 326.0, 'median_area': 145021.0, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 492.5, 'median_h': 326.0, 'median_area': 145021.0, 'frames': 120}, 'message': ''}, 'dancetrack0005': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 184.0, 'median_h': 409.5, 'median_area': 75900.0, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 184.0, 'median_h': 409.5, 'median_area': 75900.0, 'frames': 120}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 118, 'first': 2, 'median_w': 186.0, 'median_h': 398.0, 'median_area': 72553.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 186.0, 'median_h': 398.0, 'median_area': 72553.0, 'frames': 118}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 154.0, 'median_h': 363.5, 'median_area': 59405.0, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 154.0, 'median_h': 363.5, 'median_area': 59405.0, 'frames': 120}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 201.0, 'median_h': 438.0, 'median_area': 90216.5, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 201.0, 'median_h': 438.0, 'median_area': 90216.5, 'frames': 120}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 120, 'first': 0, 'median_w': 184.0, 'median_h': 409.5, 'median_area': 75900.0, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 184.0, 'median_h': 409.5, 'median_area': 75900.0, 'frames': 120}, 'message': ''}}, target_ids={'dancetrack0004': 3, 'dancetrack0005': 0}, target_strategy=stressed, wall_seconds=119.47

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_bootstrap | bootstrap_0 | none | 2 | 240 | 240 | 0.3750 | 0.3670 | 0.6250 | 0.1625 | - | - | - | - | 0.12917 | 0.24219 | 4 | 4 | 0.7438 | 5 | 7 | 42 | - | - | - | 193.20 |
| anchor_bootstrap | bootstrap_16 | none | 2 | 240 | 240 | 0.2125 | 0.2443 | 0.7875 | 0.0667 | - | - | - | - | 0.04583 | 0.18966 | 6 | 6 | 0.8226 | 7 | 5 | 38 | - | - | - | 157.16 |
| anchor_bootstrap | bootstrap_4 | none | 2 | 240 | 240 | 0.4792 | 0.4499 | 0.5208 | 0.0625 | - | - | - | - | 0.03750 | 0.06202 | 4 | 4 | 0.9274 | 6 | 8 | 65 | - | - | - | 188.91 |
| anchor_bootstrap | bootstrap_8 | none | 2 | 240 | 240 | 0.4375 | 0.4137 | 0.5625 | 0.1083 | - | - | - | - | 0.07500 | 0.14062 | 8 | 8 | 0.8537 | 11 | 7 | 56 | - | - | - | 198.69 |
| baseline | baseline | none | 2 | 240 | 240 | 0.3750 | 0.3670 | 0.6250 | 0.1625 | - | - | - | - | 0.12917 | 0.24219 | 4 | 4 | 0.7438 | 5 | 7 | 42 | - | - | - | 195.71 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 3 | observable | partial-view | none | 492x326 | 120 |
| dancetrack0005 | 0 | observable | partial-view | none | 184x410 | 120 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

- [anchor_bootstrap] bootstrap_0 (mask=none): TSR 0.3750 vs 0.3750 (+0.0000) | FAR 0.12917 vs 0.12917 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.3670 vs 0.3670 | ms/frame 193.20 vs 195.71 (-2.51)
- [anchor_bootstrap] bootstrap_16 (mask=none): TSR 0.2125 vs 0.3750 (-0.1625) | FAR 0.04583 vs 0.12917 (-0.08333) | FA events 6 vs 4 (+2) | precision 0.2443 vs 0.3670 | ms/frame 157.16 vs 195.71 (-38.55)
- [anchor_bootstrap] bootstrap_4 (mask=none): TSR 0.4792 vs 0.3750 (+0.1042) | FAR 0.03750 vs 0.12917 (-0.09167) | FA events 4 vs 4 (+0) | precision 0.4499 vs 0.3670 | ms/frame 188.91 vs 195.71 (-6.80)
- [anchor_bootstrap] bootstrap_8 (mask=none): TSR 0.4375 vs 0.3750 (+0.0625) | FAR 0.07500 vs 0.12917 (-0.05417) | FA events 8 vs 4 (+4) | precision 0.4137 vs 0.3670 | ms/frame 198.69 vs 195.71 (+2.98)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 120 | 0.1083 | 0.1433 | 0.05000 | 3 | 0.6842 | 4 | 177.55 |
| dancetrack0004 | bootstrap_0 | none | 3 | 120 | 0.1083 | 0.1433 | 0.05000 | 3 | 0.6842 | 4 | 177.55 |
| dancetrack0004 | bootstrap_4 | none | 3 | 120 | 0.1083 | 0.1433 | 0.05000 | 3 | 0.6842 | 4 | 148.61 |
| dancetrack0004 | bootstrap_8 | none | 3 | 120 | 0.0667 | 0.1009 | 0.09167 | 6 | 0.4211 | 7 | 150.20 |
| dancetrack0004 | bootstrap_16 | none | 3 | 120 | 0.0667 | 0.1009 | 0.09167 | 6 | 0.4211 | 7 | 139.64 |
| dancetrack0005 | baseline | none | 0 | 120 | 0.6417 | 0.5908 | 0.20833 | 1 | 0.7549 | 1 | 213.88 |
| dancetrack0005 | bootstrap_0 | none | 0 | 120 | 0.6417 | 0.5908 | 0.20833 | 1 | 0.7549 | 1 | 208.85 |
| dancetrack0005 | bootstrap_4 | none | 0 | 120 | 0.8500 | 0.7565 | 0.02500 | 1 | 0.9714 | 2 | 229.22 |
| dancetrack0005 | bootstrap_8 | none | 0 | 120 | 0.8083 | 0.7265 | 0.05833 | 2 | 0.9327 | 4 | 247.19 |
| dancetrack0005 | bootstrap_16 | none | 0 | 120 | 0.3583 | 0.3877 | 0.00000 | 0 | 1.0000 | 0 | 174.69 |
