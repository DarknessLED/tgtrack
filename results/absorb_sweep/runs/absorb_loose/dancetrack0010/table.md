# Clicked-target strategy ablation

Config: ablate=absorb, allow_invisible_target=False, data_root=C:\Projects\track\data\raw\dancetrack\val\dancetrack0010, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=1, masks=['none'], max_frames=300, subsample=2, target_decisions={'dancetrack0010': {'target_id': 5, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 299, 'first': 0, 'median_w': 181.0, 'median_h': 450.0, 'median_area': 83134.0, 'min_h': 226.0, 'min_area': 28440.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 181.0, 'median_h': 450.0, 'median_area': 83134.0, 'frames': 299}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 174.0, 'median_h': 504.5, 'median_area': 90447.0, 'min_h': 336.0, 'min_area': 28560.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 174.0, 'median_h': 504.5, 'median_area': 90447.0, 'frames': 300}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 300, 'first': 0, 'median_w': 175.0, 'median_h': 474.0, 'median_area': 88971.0, 'min_h': 244.0, 'min_area': 30702.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 175.0, 'median_h': 474.0, 'median_area': 88971.0, 'frames': 300}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 281, 'first': 0, 'median_w': 172.0, 'median_h': 409.0, 'median_area': 74520.0, 'min_h': 270.0, 'min_area': 29667.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 172.0, 'median_h': 409.0, 'median_area': 74520.0, 'frames': 281}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 294, 'first': 0, 'median_w': 176.0, 'median_h': 461.0, 'median_area': 83616.0, 'min_h': 220.0, 'min_area': 28600.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 176.0, 'median_h': 461.0, 'median_area': 83616.0, 'frames': 294}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 272, 'first': 0, 'median_w': 156.0, 'median_h': 461.0, 'median_area': 68489.0, 'min_h': 243.0, 'min_area': 30618.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 156.0, 'median_h': 461.0, 'median_area': 68489.0, 'frames': 272}}, 'selected': {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 272, 'first': 0, 'median_w': 156.0, 'median_h': 461.0, 'median_area': 68489.0, 'min_h': 243.0, 'min_area': 30618.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 156.0, 'median_h': 461.0, 'median_area': 68489.0, 'frames': 272}, 'message': ''}}, target_ids={'dancetrack0010': 5}, target_strategy=stressed, wall_seconds=51.88

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| absorb_sweep | absorb_loose | none | 1 | 300 | 272 | 0.2831 | 0.2413 | 0.7169 | 0.6250 | 0.0000 | - | - | - | 0.68000 | 0.72694 | 6 | 13 | 0.2527 | 5 | 1 | 68 | - | - | - | 151.55 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | 5 | forced | partial-view | none | 156x461 | 272 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Claims (delta vs the same-mask baseline)

(only the baseline was run)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0010 | absorb_loose | none | 5 | 300 | 0.2831 | 0.2413 | 0.68000 | 6 | 0.2527 | 5 | 151.55 |
