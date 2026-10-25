# Clicked-target strategy ablation

Config: ablate=gate,distractor_margin,target_parts,part_matching, data_root=data\raw\mot17\MOT17\train, detector=yolo, dry_run=False, embedder=real, jobs=2, masks=['none', 'upper', 'lower'], max_frames=300, subsample=2, target_ids={'MOT17-02-FRCNN': 19}, target_strategy=stressed, wall_seconds=1606.61

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 293.93 |
| baseline | baseline | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 244.94 |
| baseline | baseline | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 187.63 |
| distractor_margin | margin_0.00 | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01667 | 0.01141 | 5 | 3 | 0.9806 | 7 | 3 | 126 | 381.97 |
| distractor_margin | margin_0.00 | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.01000 | 0.01128 | 3 | 3 | 0.9883 | 5 | 2 | 217 | 353.61 |
| distractor_margin | margin_0.00 | upper | 1 | 300 | 275 | 0.6945 | 0.5098 | 0.3055 | 0.0545 | - | - | - | - | 0.01000 | 0.01422 | 3 | 3 | 0.9845 | 4 | 3 | 97 | 303.16 |
| distractor_margin | margin_0.06 | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 380.09 |
| distractor_margin | margin_0.06 | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 344.98 |
| distractor_margin | margin_0.06 | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 266.50 |
| distractor_margin | margin_0.12 | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01667 | 0.01141 | 4 | 2 | 0.9806 | 6 | 3 | 126 | 376.22 |
| distractor_margin | margin_0.12 | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 352.40 |
| distractor_margin | margin_0.12 | upper | 1 | 300 | 275 | 0.7236 | 0.5446 | 0.2764 | 0.0545 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 5 | 97 | 282.91 |
| gate | gate_growth_unfrozen | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 377.64 |
| gate | gate_growth_unfrozen | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 355.14 |
| gate | gate_growth_unfrozen | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 283.35 |
| gate | gate_legacy | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00766 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 310.39 |
| gate | gate_legacy | none | 1 | 300 | 275 | 0.7891 | 0.6735 | 0.2109 | 0.0145 | - | - | - | - | 0.00333 | 0.00452 | 1 | 1 | 0.9954 | 1 | 0 | 217 | 262.05 |
| gate | gate_legacy | upper | 1 | 300 | 275 | 0.6036 | 0.4422 | 0.3964 | 0.0145 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 3 | 97 | 257.06 |
| gate | gate_search_early | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 367.77 |
| gate | gate_search_early | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 284.44 |
| gate | gate_search_early | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 214.35 |
| gate | gate_search_late | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00766 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 384.54 |
| gate | gate_search_late | none | 1 | 300 | 275 | 0.7891 | 0.6735 | 0.2109 | 0.0145 | - | - | - | - | 0.01000 | 0.01333 | 3 | 3 | 0.9864 | 3 | 0 | 217 | 349.17 |
| gate | gate_search_late | upper | 1 | 300 | 275 | 0.6036 | 0.4422 | 0.3964 | 0.0145 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 3 | 97 | 328.39 |
| part_matching | part_matching_off | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 356.13 |
| part_matching | part_matching_off | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 370.85 |
| part_matching | part_matching_off | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 293.30 |
| target_parts | parts_all | lower | 1 | 300 | 275 | 0.9236 | 0.6976 | 0.0764 | 0.0218 | - | - | - | - | 0.01333 | 0.00760 | 4 | 2 | 0.9844 | 6 | 3 | 126 | 401.66 |
| target_parts | parts_all | none | 1 | 300 | 275 | 0.9236 | 0.7725 | 0.0764 | 0.0327 | - | - | - | - | 0.00667 | 0.00752 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 367.74 |
| target_parts | parts_all | upper | 1 | 300 | 275 | 0.7527 | 0.5617 | 0.2473 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 292.58 |
| target_parts | parts_full_only | lower | 1 | 300 | 275 | 0.7745 | 0.5973 | 0.2255 | 0.0291 | - | - | - | - | 0.02000 | 0.00897 | 5 | 2 | 0.9725 | 8 | 8 | 126 | 273.34 |
| target_parts | parts_full_only | none | 1 | 300 | 275 | 0.9200 | 0.7709 | 0.0800 | 0.0327 | - | - | - | - | 0.00667 | 0.00755 | 2 | 2 | 0.9922 | 4 | 2 | 217 | 263.24 |
| target_parts | parts_full_only | upper | 1 | 300 | 275 | 0.6691 | 0.5157 | 0.3309 | 0.0473 | - | - | - | - | 0.00000 | 0.00000 | 0 | 0 | 1.0000 | 0 | 6 | 97 | 207.94 |

## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lower | 0.9236 | 0.9236 | +0.0000 | 0.01333 | 0.01333 | +0.00000 | 293.93 | 356.13 |
| none | 0.9236 | 0.9236 | +0.0000 | 0.00667 | 0.00667 | +0.00000 | 244.94 | 370.85 |
| upper | 0.7527 | 0.7527 | +0.0000 | 0.00000 | 0.00000 | +0.00000 | 187.63 | 293.30 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01667 vs 0.01333 (+0.00333) | FA events 5 vs 4 (+1) | precision 0.6976 vs 0.6976 | ms/frame 381.97 vs 293.93 (+88.04)
- [distractor_margin] margin_0.00 (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01000 vs 0.00667 (+0.00333) | FA events 3 vs 2 (+1) | precision 0.7725 vs 0.7725 | ms/frame 353.61 vs 244.94 (+108.67)
- [distractor_margin] margin_0.00 (mask=upper): TSR 0.6945 vs 0.7527 (-0.0582) | FAR 0.01000 vs 0.00000 (+0.01000) | FA events 3 vs 0 (+3) | precision 0.5098 vs 0.5617 | ms/frame 303.16 vs 187.63 (+115.52)
- [distractor_margin] margin_0.06 (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 380.09 vs 293.93 (+86.17)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 344.98 vs 244.94 (+100.04)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.7527 vs 0.7527 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5617 vs 0.5617 | ms/frame 266.50 vs 187.63 (+78.87)
- [distractor_margin] margin_0.12 (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01667 vs 0.01333 (+0.00333) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 376.22 vs 293.93 (+82.29)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 352.40 vs 244.94 (+107.45)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.7236 vs 0.7527 (-0.0291) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5446 vs 0.5617 | ms/frame 282.91 vs 187.63 (+95.28)
- [gate] gate_growth_unfrozen (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 377.64 vs 293.93 (+83.71)
- [gate] gate_growth_unfrozen (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 355.14 vs 244.94 (+110.20)
- [gate] gate_growth_unfrozen (mask=upper): TSR 0.7527 vs 0.7527 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5617 vs 0.5617 | ms/frame 283.35 vs 187.63 (+95.72)
- [gate] gate_legacy (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 310.39 vs 293.93 (+16.46)
- [gate] gate_legacy (mask=none): TSR 0.7891 vs 0.9236 (-0.1345) | FAR 0.00333 vs 0.00667 (-0.00333) | FA events 1 vs 2 (-1) | precision 0.6735 vs 0.7725 | ms/frame 262.05 vs 244.94 (+17.10)
- [gate] gate_legacy (mask=upper): TSR 0.6036 vs 0.7527 (-0.1491) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.4422 vs 0.5617 | ms/frame 257.06 vs 187.63 (+69.42)
- [gate] gate_search_early (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 367.77 vs 293.93 (+73.85)
- [gate] gate_search_early (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 284.44 vs 244.94 (+39.50)
- [gate] gate_search_early (mask=upper): TSR 0.7527 vs 0.7527 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5617 vs 0.5617 | ms/frame 214.35 vs 187.63 (+26.71)
- [gate] gate_search_late (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 384.54 vs 293.93 (+90.61)
- [gate] gate_search_late (mask=none): TSR 0.7891 vs 0.9236 (-0.1345) | FAR 0.01000 vs 0.00667 (+0.00333) | FA events 3 vs 2 (+1) | precision 0.6735 vs 0.7725 | ms/frame 349.17 vs 244.94 (+104.23)
- [gate] gate_search_late (mask=upper): TSR 0.6036 vs 0.7527 (-0.1491) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.4422 vs 0.5617 | ms/frame 328.39 vs 187.63 (+140.75)
- [part_matching] part_matching_off (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 356.13 vs 293.93 (+62.21)
- [part_matching] part_matching_off (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 370.85 vs 244.94 (+125.91)
- [part_matching] part_matching_off (mask=upper): TSR 0.7527 vs 0.7527 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5617 vs 0.5617 | ms/frame 293.30 vs 187.63 (+105.66)
- [target_parts] parts_all (mask=lower): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.01333 vs 0.01333 (+0.00000) | FA events 4 vs 4 (+0) | precision 0.6976 vs 0.6976 | ms/frame 401.66 vs 293.93 (+107.74)
- [target_parts] parts_all (mask=none): TSR 0.9236 vs 0.9236 (+0.0000) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7725 vs 0.7725 | ms/frame 367.74 vs 244.94 (+122.80)
- [target_parts] parts_all (mask=upper): TSR 0.7527 vs 0.7527 (+0.0000) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5617 vs 0.5617 | ms/frame 292.58 vs 187.63 (+104.94)
- [target_parts] parts_full_only (mask=lower): TSR 0.7745 vs 0.9236 (-0.1491) | FAR 0.02000 vs 0.01333 (+0.00667) | FA events 5 vs 4 (+1) | precision 0.5973 vs 0.6976 | ms/frame 273.34 vs 293.93 (-20.59)
- [target_parts] parts_full_only (mask=none): TSR 0.9200 vs 0.9236 (-0.0036) | FAR 0.00667 vs 0.00667 (+0.00000) | FA events 2 vs 2 (+0) | precision 0.7709 vs 0.7725 | ms/frame 263.24 vs 244.94 (+18.30)
- [target_parts] parts_full_only (mask=upper): TSR 0.6691 vs 0.7527 (-0.0836) | FAR 0.00000 vs 0.00000 (+0.00000) | FA events 0 vs 0 (+0) | precision 0.5157 vs 0.5617 | ms/frame 207.94 vs 187.63 (+20.31)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-FRCNN | baseline | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 244.94 |
| MOT17-02-FRCNN | baseline | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 187.63 |
| MOT17-02-FRCNN | baseline | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 293.93 |
| MOT17-02-FRCNN | gate_legacy | none | 19 | 300 | 0.7891 | 0.6735 | 0.00333 | 1 | 0.9954 | 1 | 262.05 |
| MOT17-02-FRCNN | gate_legacy | upper | 19 | 300 | 0.6036 | 0.4422 | 0.00000 | 0 | 1.0000 | 0 | 257.06 |
| MOT17-02-FRCNN | gate_legacy | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 310.39 |
| MOT17-02-FRCNN | gate_search_early | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 284.44 |
| MOT17-02-FRCNN | gate_search_early | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 214.35 |
| MOT17-02-FRCNN | gate_search_early | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 367.77 |
| MOT17-02-FRCNN | gate_search_late | none | 19 | 300 | 0.7891 | 0.6735 | 0.01000 | 3 | 0.9864 | 3 | 349.17 |
| MOT17-02-FRCNN | gate_search_late | upper | 19 | 300 | 0.6036 | 0.4422 | 0.00000 | 0 | 1.0000 | 0 | 328.39 |
| MOT17-02-FRCNN | gate_search_late | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 384.54 |
| MOT17-02-FRCNN | gate_growth_unfrozen | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 355.14 |
| MOT17-02-FRCNN | gate_growth_unfrozen | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 283.35 |
| MOT17-02-FRCNN | gate_growth_unfrozen | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 377.64 |
| MOT17-02-FRCNN | margin_0.00 | none | 19 | 300 | 0.9236 | 0.7725 | 0.01000 | 3 | 0.9883 | 5 | 353.61 |
| MOT17-02-FRCNN | margin_0.00 | upper | 19 | 300 | 0.6945 | 0.5098 | 0.01000 | 3 | 0.9845 | 4 | 303.16 |
| MOT17-02-FRCNN | margin_0.00 | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01667 | 5 | 0.9806 | 7 | 381.97 |
| MOT17-02-FRCNN | margin_0.06 | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 344.98 |
| MOT17-02-FRCNN | margin_0.06 | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 266.50 |
| MOT17-02-FRCNN | margin_0.06 | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 380.09 |
| MOT17-02-FRCNN | margin_0.12 | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 352.40 |
| MOT17-02-FRCNN | margin_0.12 | upper | 19 | 300 | 0.7236 | 0.5446 | 0.00000 | 0 | 1.0000 | 0 | 282.91 |
| MOT17-02-FRCNN | margin_0.12 | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01667 | 4 | 0.9806 | 6 | 376.22 |
| MOT17-02-FRCNN | parts_full_only | none | 19 | 300 | 0.9200 | 0.7709 | 0.00667 | 2 | 0.9922 | 4 | 263.24 |
| MOT17-02-FRCNN | parts_full_only | upper | 19 | 300 | 0.6691 | 0.5157 | 0.00000 | 0 | 1.0000 | 0 | 207.94 |
| MOT17-02-FRCNN | parts_full_only | lower | 19 | 300 | 0.7745 | 0.5973 | 0.02000 | 5 | 0.9725 | 8 | 273.34 |
| MOT17-02-FRCNN | parts_all | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 367.74 |
| MOT17-02-FRCNN | parts_all | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 292.58 |
| MOT17-02-FRCNN | parts_all | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 401.66 |
| MOT17-02-FRCNN | part_matching_off | none | 19 | 300 | 0.9236 | 0.7725 | 0.00667 | 2 | 0.9922 | 4 | 370.85 |
| MOT17-02-FRCNN | part_matching_off | upper | 19 | 300 | 0.7527 | 0.5617 | 0.00000 | 0 | 1.0000 | 0 | 293.30 |
| MOT17-02-FRCNN | part_matching_off | lower | 19 | 300 | 0.9236 | 0.6976 | 0.01333 | 4 | 0.9844 | 6 | 356.13 |
