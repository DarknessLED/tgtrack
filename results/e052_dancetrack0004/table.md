# Clicked-target strategy ablation

Config: ablate=all, allow_invisible_target=False, data_root=data\raw\_root_dancetrack0004, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, jobs=2, masks=['none'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'forced', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}}, 'worst_mask': 'none', 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'frames': 200}, 'message': ''}}, target_ids={'dancetrack0004': 3}, target_strategy=stressed, wall_seconds=1283.45

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_forget | adaptive_thr_0.25 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 204.19 |
| adaptive_forget | adaptive_thr_0.30 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 204.59 |
| adaptive_forget | adaptive_thr_0.35 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 206.65 |
| adaptive_forget | anchor_drift_0.25 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 205.63 |
| adaptive_forget | anchor_drift_0.35 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 210.72 |
| adaptive_forget | anchor_drift_0.50 | none | 1 | 200 | 200 | 0.4800 | 0.4186 | 0.5200 | 0.0900 | - | - | - | - | 0.11500 | 0.18750 | 7 | 7 | 0.7830 | 17 | 6 | 39 | - | - | - | 227.59 |
| anchor_adaptive | anchor_adapt_k2 | none | 1 | 200 | 200 | 0.4150 | 0.3756 | 0.5850 | 0.2100 | - | - | - | - | 0.21500 | 0.34426 | 4 | 4 | 0.6261 | 9 | 8 | 37 | - | - | - | 229.87 |
| anchor_adaptive | anchor_adapt_k3 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 211.16 |
| anchor_adaptive | anchor_adapt_k3_hi0.95 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 216.78 |
| anchor_adaptive | anchor_adapt_k4 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 214.91 |
| anchor_adaptive | anchor_adapt_k4_c0.55 | none | 1 | 200 | 200 | 0.0600 | 0.1039 | 0.9400 | 0.2100 | - | - | - | - | 0.23500 | 0.66038 | 8 | 10 | 0.1754 | 9 | 4 | 5 | - | - | - | 161.75 |
| anchor_adaptive | anchor_adapt_k4_c0.60 | none | 1 | 200 | 200 | 0.2900 | 0.2875 | 0.7100 | 0.1400 | - | - | - | - | 0.10500 | 0.25926 | 10 | 10 | 0.7162 | 15 | 11 | 20 | - | - | - | 187.50 |
| anchor_adaptive | anchor_adapt_k4_c0.65 | none | 1 | 200 | 200 | 0.3100 | 0.2889 | 0.6900 | 0.2000 | - | - | - | - | 0.22000 | 0.44000 | 4 | 4 | 0.5368 | 11 | 9 | 16 | - | - | - | 207.57 |
| anchor_ceiling | anchor_ceiling_0.45 | none | 1 | 200 | 200 | 0.0300 | 0.0651 | 0.9700 | 0.0550 | - | - | - | - | 0.05000 | 0.37500 | 7 | 5 | 0.3750 | 7 | 1 | 5 | - | - | - | 156.51 |
| anchor_ceiling | anchor_ceiling_0.55 | none | 1 | 200 | 200 | 0.2600 | 0.2577 | 0.7400 | 0.2150 | - | - | - | - | 0.22500 | 0.47872 | 5 | 5 | 0.4706 | 11 | 3 | 40 | - | - | - | 200.21 |
| anchor_ceiling | anchor_ceiling_0.65 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 219.17 |
| anchor_ceiling | anchor_search_0.55 | none | 1 | 200 | 200 | 0.3700 | 0.3467 | 0.6300 | 0.1200 | - | - | - | - | 0.09000 | 0.15957 | 6 | 8 | 0.8000 | 9 | 7 | 27 | - | - | - | 209.21 |
| anchor_ceiling | anchor_search_0.65 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 231.33 |
| baseline | baseline | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 189.22 |
| distractor_margin | margin_0.00 | none | 1 | 200 | 200 | 0.4150 | 0.3585 | 0.5850 | 0.2400 | - | - | - | - | 0.25000 | 0.38462 | 6 | 6 | 0.5868 | 15 | 6 | 37 | - | - | - | 165.66 |
| distractor_margin | margin_0.06 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 164.29 |
| distractor_margin | margin_0.12 | none | 1 | 200 | 200 | 0.3950 | 0.3691 | 0.6050 | 0.1700 | - | - | - | - | 0.23000 | 0.38532 | 8 | 8 | 0.5660 | 17 | 8 | 39 | - | - | - | 170.17 |
| drift | drift_w4_0.35 | none | 1 | 200 | 200 | 0.1050 | 0.1167 | 0.8950 | 0.0650 | - | - | - | - | 0.03500 | 0.18750 | 5 | 5 | 0.7308 | 9 | 7 | 10 | - | - | - | 170.36 |
| drift | drift_w4_0.50 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 205.44 |
| drift | drift_w8_0.35 | none | 1 | 200 | 200 | 0.1050 | 0.1179 | 0.8950 | 0.0750 | - | - | - | - | 0.05000 | 0.22222 | 8 | 7 | 0.6667 | 13 | 6 | 10 | - | - | - | 180.97 |
| drift | drift_w8_0.50 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 213.12 |
| freeze | freeze_off | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 167.26 |
| freeze | freeze_on | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 165.97 |
| gallery_hygiene | hl_150 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 207.81 |
| gallery_hygiene | hl_5000 | none | 1 | 200 | 200 | 0.3450 | 0.3140 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 207.13 |
| gallery_hygiene | hl_none | none | 1 | 200 | 200 | 0.3450 | 0.3140 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 207.11 |
| gallery_hygiene | minq_0.50 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 207.30 |
| gallery_hygiene | minq_0.70 | none | 1 | 200 | 200 | 0.0550 | 0.0939 | 0.9450 | 0.1850 | - | - | - | - | 0.15500 | 0.61702 | 10 | 11 | 0.2051 | 15 | 6 | 3 | - | - | - | 164.53 |
| gate | gate_growth_unfrozen | none | 1 | 200 | 200 | 0.2500 | 0.2559 | 0.7500 | 0.1400 | - | - | - | - | 0.10500 | 0.27273 | 7 | 7 | 0.6866 | 12 | 8 | 18 | - | - | - | 166.80 |
| gate | gate_legacy | none | 1 | 200 | 200 | 0.0750 | 0.0826 | 0.9250 | 0.0800 | - | - | - | - | 0.02000 | 0.13333 | 3 | 3 | 0.7895 | 4 | 6 | 4 | - | - | - | 133.56 |
| gate | gate_search_early | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 163.00 |
| gate | gate_search_late | none | 1 | 200 | 200 | 0.0800 | 0.0871 | 0.9200 | 0.0800 | - | - | - | - | 0.02000 | 0.12903 | 3 | 3 | 0.8000 | 5 | 7 | 4 | - | - | - | 136.76 |
| hard_negative | hard_negative_cap12 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 161.07 |
| hard_negative | hard_negative_cap6 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 160.84 |
| hard_negative | hard_negative_confident | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 162.62 |
| hard_negative | hard_negative_off | none | 1 | 200 | 200 | 0.4200 | 0.3637 | 0.5800 | 0.0700 | - | - | - | - | 0.09000 | 0.18947 | 4 | 4 | 0.8000 | 13 | 10 | 16 | - | - | - | 162.70 |
| ltm_anchor | anchor_16 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 210.60 |
| ltm_anchor | anchor_32 | none | 1 | 200 | 200 | 0.3450 | 0.3140 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 205.65 |
| ltm_anchor | anchor_4 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 206.53 |
| ltm_capacity | ltm_cap_1 | none | 1 | 200 | 200 | 0.0500 | 0.0979 | 0.9500 | 0.0450 | - | - | - | - | 0.02500 | 0.22222 | 3 | 3 | 0.6154 | 6 | 4 | 5 | - | - | - | 119.00 |
| ltm_capacity | ltm_cap_384 | none | 1 | 200 | 200 | 0.3150 | 0.3066 | 0.6850 | 0.2550 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 13 | 6 | 31 | - | - | - | 162.02 |
| ltm_capacity | ltm_cap_96 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 163.01 |
| motion_band | motion_band_k2 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 230.18 |
| motion_band | motion_band_k3 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 212.29 |
| motion_band | motion_band_k4 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 176.17 |
| motion_guard | vclamp_2 | none | 1 | 200 | 200 | 0.3800 | 0.3652 | 0.6200 | 0.2000 | - | - | - | - | 0.23000 | 0.39655 | 4 | 4 | 0.5818 | 11 | 6 | 36 | - | - | - | 158.60 |
| motion_guard | vclamp_4 | none | 1 | 200 | 200 | 0.1150 | 0.1271 | 0.8850 | 0.1450 | - | - | - | - | 0.09000 | 0.35294 | 7 | 7 | 0.5610 | 9 | 6 | 10 | - | - | - | 139.04 |
| motion_guard | vclamp_off | none | 1 | 200 | 200 | 0.0300 | 0.0389 | 0.9700 | 0.1150 | - | - | - | - | 0.05000 | 0.34483 | 6 | 6 | 0.3750 | 6 | 3 | 3 | - | - | - | 143.26 |
| part_aware | part_aware_off | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 164.52 |
| part_matching | part_matching_off | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 189.22 |
| partial_fusion | partial_m0.2_a2.4 | none | 1 | 200 | 200 | 0.2500 | 0.2466 | 0.7500 | 0.1000 | - | - | - | - | 0.05500 | 0.14706 | 8 | 7 | 0.8103 | 12 | 13 | 11 | - | - | - | 203.94 |
| partial_fusion | partial_m0.4_a2.2 | none | 1 | 200 | 200 | 0.3850 | 0.3386 | 0.6150 | 0.1950 | - | - | - | - | 0.22000 | 0.38053 | 3 | 3 | 0.5963 | 9 | 7 | 37 | - | - | - | 208.95 |
| partial_fusion | partial_m0.5_a2.0 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 213.54 |
| partial_fusion | partial_m1.0_a1.6 | none | 1 | 200 | 200 | 0.1500 | 0.1683 | 0.8500 | 0.2050 | - | - | - | - | 0.25000 | 0.59420 | 3 | 4 | 0.3243 | 10 | 8 | 13 | - | - | - | 185.95 |
| search_guard | search_margin_0.10 | none | 1 | 200 | 200 | 0.3300 | 0.3148 | 0.6700 | 0.2500 | - | - | - | - | 0.21500 | 0.36697 | 9 | 10 | 0.5905 | 13 | 12 | 23 | - | - | - | 195.16 |
| search_guard | search_margin_0.20 | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 207.12 |
| search_guard | search_stm_off | none | 1 | 200 | 200 | 0.2550 | 0.2537 | 0.7450 | 0.2300 | - | - | - | - | 0.24000 | 0.48958 | 7 | 7 | 0.4545 | 16 | 7 | 22 | - | - | - | 206.79 |
| search_guard | search_stm_off_margin_0.10 | none | 1 | 200 | 200 | 0.3800 | 0.3641 | 0.6200 | 0.2150 | - | - | - | - | 0.20500 | 0.34821 | 7 | 8 | 0.6095 | 14 | 15 | 23 | - | - | - | 210.09 |
| stm_ltm | balanced | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 165.55 |
| stm_ltm | ltm_only | none | 1 | 200 | 200 | 0.3350 | 0.3354 | 0.6650 | 0.2200 | - | - | - | - | 0.21000 | 0.37037 | 7 | 7 | 0.5882 | 9 | 6 | 38 | - | - | - | 151.60 |
| stm_ltm | stm_only | none | 1 | 200 | 200 | 0.2000 | 0.2106 | 0.8000 | 0.0800 | - | - | - | - | 0.09500 | 0.28846 | 9 | 8 | 0.6122 | 17 | 7 | 13 | - | - | - | 143.96 |
| target_parts | parts_all | none | 1 | 200 | 200 | 0.3450 | 0.3137 | 0.6550 | 0.2250 | - | - | - | - | 0.24500 | 0.41964 | 7 | 7 | 0.5377 | 15 | 5 | 39 | - | - | - | 164.57 |
| target_parts | parts_full_only | none | 1 | 200 | 200 | 0.0750 | 0.1201 | 0.9250 | 0.2000 | - | - | - | - | 0.19000 | 0.70588 | 5 | 4 | 0.2692 | 9 | 4 | 5 | - | - | - | 109.05 |

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
| none | 0.3450 | 0.3450 | +0.0000 | 0.24500 | 0.24500 | +0.00000 | 189.22 | 189.22 |

## Claims (delta vs the same-mask baseline)

- [adaptive_forget] adaptive_thr_0.25 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 204.19 vs 189.22 (+14.96)
- [adaptive_forget] adaptive_thr_0.30 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 204.59 vs 189.22 (+15.37)
- [adaptive_forget] adaptive_thr_0.35 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 206.65 vs 189.22 (+17.42)
- [adaptive_forget] anchor_drift_0.25 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 205.63 vs 189.22 (+16.41)
- [adaptive_forget] anchor_drift_0.35 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 210.72 vs 189.22 (+21.50)
- [adaptive_forget] anchor_drift_0.50 (mask=none): TSR 0.4800 vs 0.3450 (+0.1350) | FAR 0.11500 vs 0.24500 (-0.13000) | FA events 7 vs 7 (+0) | precision 0.4186 vs 0.3137 | ms/frame 227.59 vs 189.22 (+38.36)
- [anchor_adaptive] anchor_adapt_k2 (mask=none): TSR 0.4150 vs 0.3450 (+0.0700) | FAR 0.21500 vs 0.24500 (-0.03000) | FA events 4 vs 7 (-3) | precision 0.3756 vs 0.3137 | ms/frame 229.87 vs 189.22 (+40.65)
- [anchor_adaptive] anchor_adapt_k3 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 211.16 vs 189.22 (+21.94)
- [anchor_adaptive] anchor_adapt_k3_hi0.95 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 216.78 vs 189.22 (+27.56)
- [anchor_adaptive] anchor_adapt_k4 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 214.91 vs 189.22 (+25.68)
- [anchor_adaptive] anchor_adapt_k4_c0.55 (mask=none): TSR 0.0600 vs 0.3450 (-0.2850) | FAR 0.23500 vs 0.24500 (-0.01000) | FA events 8 vs 7 (+1) | precision 0.1039 vs 0.3137 | ms/frame 161.75 vs 189.22 (-27.48)
- [anchor_adaptive] anchor_adapt_k4_c0.60 (mask=none): TSR 0.2900 vs 0.3450 (-0.0550) | FAR 0.10500 vs 0.24500 (-0.14000) | FA events 10 vs 7 (+3) | precision 0.2875 vs 0.3137 | ms/frame 187.50 vs 189.22 (-1.72)
- [anchor_adaptive] anchor_adapt_k4_c0.65 (mask=none): TSR 0.3100 vs 0.3450 (-0.0350) | FAR 0.22000 vs 0.24500 (-0.02500) | FA events 4 vs 7 (-3) | precision 0.2889 vs 0.3137 | ms/frame 207.57 vs 189.22 (+18.35)
- [anchor_ceiling] anchor_ceiling_0.45 (mask=none): TSR 0.0300 vs 0.3450 (-0.3150) | FAR 0.05000 vs 0.24500 (-0.19500) | FA events 7 vs 7 (+0) | precision 0.0651 vs 0.3137 | ms/frame 156.51 vs 189.22 (-32.71)
- [anchor_ceiling] anchor_ceiling_0.55 (mask=none): TSR 0.2600 vs 0.3450 (-0.0850) | FAR 0.22500 vs 0.24500 (-0.02000) | FA events 5 vs 7 (-2) | precision 0.2577 vs 0.3137 | ms/frame 200.21 vs 189.22 (+10.99)
- [anchor_ceiling] anchor_ceiling_0.65 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 219.17 vs 189.22 (+29.95)
- [anchor_ceiling] anchor_search_0.55 (mask=none): TSR 0.3700 vs 0.3450 (+0.0250) | FAR 0.09000 vs 0.24500 (-0.15500) | FA events 6 vs 7 (-1) | precision 0.3467 vs 0.3137 | ms/frame 209.21 vs 189.22 (+19.98)
- [anchor_ceiling] anchor_search_0.65 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 231.33 vs 189.22 (+42.11)
- [distractor_margin] margin_0.00 (mask=none): TSR 0.4150 vs 0.3450 (+0.0700) | FAR 0.25000 vs 0.24500 (+0.00500) | FA events 6 vs 7 (-1) | precision 0.3585 vs 0.3137 | ms/frame 165.66 vs 189.22 (-23.56)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 164.29 vs 189.22 (-24.93)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.3950 vs 0.3450 (+0.0500) | FAR 0.23000 vs 0.24500 (-0.01500) | FA events 8 vs 7 (+1) | precision 0.3691 vs 0.3137 | ms/frame 170.17 vs 189.22 (-19.05)
- [drift] drift_w4_0.35 (mask=none): TSR 0.1050 vs 0.3450 (-0.2400) | FAR 0.03500 vs 0.24500 (-0.21000) | FA events 5 vs 7 (-2) | precision 0.1167 vs 0.3137 | ms/frame 170.36 vs 189.22 (-18.86)
- [drift] drift_w4_0.50 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 205.44 vs 189.22 (+16.21)
- [drift] drift_w8_0.35 (mask=none): TSR 0.1050 vs 0.3450 (-0.2400) | FAR 0.05000 vs 0.24500 (-0.19500) | FA events 8 vs 7 (+1) | precision 0.1179 vs 0.3137 | ms/frame 180.97 vs 189.22 (-8.26)
- [drift] drift_w8_0.50 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 213.12 vs 189.22 (+23.89)
- [freeze] freeze_off (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 167.26 vs 189.22 (-21.96)
- [freeze] freeze_on (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 165.97 vs 189.22 (-23.25)
- [gallery_hygiene] hl_150 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 207.81 vs 189.22 (+18.59)
- [gallery_hygiene] hl_5000 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3140 vs 0.3137 | ms/frame 207.13 vs 189.22 (+17.91)
- [gallery_hygiene] hl_none (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3140 vs 0.3137 | ms/frame 207.11 vs 189.22 (+17.89)
- [gallery_hygiene] minq_0.50 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 207.30 vs 189.22 (+18.08)
- [gallery_hygiene] minq_0.70 (mask=none): TSR 0.0550 vs 0.3450 (-0.2900) | FAR 0.15500 vs 0.24500 (-0.09000) | FA events 10 vs 7 (+3) | precision 0.0939 vs 0.3137 | ms/frame 164.53 vs 189.22 (-24.69)
- [gate] gate_growth_unfrozen (mask=none): TSR 0.2500 vs 0.3450 (-0.0950) | FAR 0.10500 vs 0.24500 (-0.14000) | FA events 7 vs 7 (+0) | precision 0.2559 vs 0.3137 | ms/frame 166.80 vs 189.22 (-22.43)
- [gate] gate_legacy (mask=none): TSR 0.0750 vs 0.3450 (-0.2700) | FAR 0.02000 vs 0.24500 (-0.22500) | FA events 3 vs 7 (-4) | precision 0.0826 vs 0.3137 | ms/frame 133.56 vs 189.22 (-55.67)
- [gate] gate_search_early (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 163.00 vs 189.22 (-26.22)
- [gate] gate_search_late (mask=none): TSR 0.0800 vs 0.3450 (-0.2650) | FAR 0.02000 vs 0.24500 (-0.22500) | FA events 3 vs 7 (-4) | precision 0.0871 vs 0.3137 | ms/frame 136.76 vs 189.22 (-52.47)
- [hard_negative] hard_negative_cap12 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 161.07 vs 189.22 (-28.16)
- [hard_negative] hard_negative_cap6 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 160.84 vs 189.22 (-28.39)
- [hard_negative] hard_negative_confident (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 162.62 vs 189.22 (-26.60)
- [hard_negative] hard_negative_off (mask=none): TSR 0.4200 vs 0.3450 (+0.0750) | FAR 0.09000 vs 0.24500 (-0.15500) | FA events 4 vs 7 (-3) | precision 0.3637 vs 0.3137 | ms/frame 162.70 vs 189.22 (-26.52)
- [ltm_anchor] anchor_16 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 210.60 vs 189.22 (+21.38)
- [ltm_anchor] anchor_32 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3140 vs 0.3137 | ms/frame 205.65 vs 189.22 (+16.42)
- [ltm_anchor] anchor_4 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 206.53 vs 189.22 (+17.31)
- [ltm_capacity] ltm_cap_1 (mask=none): TSR 0.0500 vs 0.3450 (-0.2950) | FAR 0.02500 vs 0.24500 (-0.22000) | FA events 3 vs 7 (-4) | precision 0.0979 vs 0.3137 | ms/frame 119.00 vs 189.22 (-70.22)
- [ltm_capacity] ltm_cap_384 (mask=none): TSR 0.3150 vs 0.3450 (-0.0300) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3066 vs 0.3137 | ms/frame 162.02 vs 189.22 (-27.20)
- [ltm_capacity] ltm_cap_96 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 163.01 vs 189.22 (-26.22)
- [motion_band] motion_band_k2 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 230.18 vs 189.22 (+40.96)
- [motion_band] motion_band_k3 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 212.29 vs 189.22 (+23.07)
- [motion_band] motion_band_k4 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 176.17 vs 189.22 (-13.05)
- [motion_guard] vclamp_2 (mask=none): TSR 0.3800 vs 0.3450 (+0.0350) | FAR 0.23000 vs 0.24500 (-0.01500) | FA events 4 vs 7 (-3) | precision 0.3652 vs 0.3137 | ms/frame 158.60 vs 189.22 (-30.63)
- [motion_guard] vclamp_4 (mask=none): TSR 0.1150 vs 0.3450 (-0.2300) | FAR 0.09000 vs 0.24500 (-0.15500) | FA events 7 vs 7 (+0) | precision 0.1271 vs 0.3137 | ms/frame 139.04 vs 189.22 (-50.18)
- [motion_guard] vclamp_off (mask=none): TSR 0.0300 vs 0.3450 (-0.3150) | FAR 0.05000 vs 0.24500 (-0.19500) | FA events 6 vs 7 (-1) | precision 0.0389 vs 0.3137 | ms/frame 143.26 vs 189.22 (-45.97)
- [part_aware] part_aware_off (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 164.52 vs 189.22 (-24.70)
- [part_matching] part_matching_off (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 189.22 vs 189.22 (-0.00)
- [partial_fusion] partial_m0.2_a2.4 (mask=none): TSR 0.2500 vs 0.3450 (-0.0950) | FAR 0.05500 vs 0.24500 (-0.19000) | FA events 8 vs 7 (+1) | precision 0.2466 vs 0.3137 | ms/frame 203.94 vs 189.22 (+14.71)
- [partial_fusion] partial_m0.4_a2.2 (mask=none): TSR 0.3850 vs 0.3450 (+0.0400) | FAR 0.22000 vs 0.24500 (-0.02500) | FA events 3 vs 7 (-4) | precision 0.3386 vs 0.3137 | ms/frame 208.95 vs 189.22 (+19.72)
- [partial_fusion] partial_m0.5_a2.0 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 213.54 vs 189.22 (+24.32)
- [partial_fusion] partial_m1.0_a1.6 (mask=none): TSR 0.1500 vs 0.3450 (-0.1950) | FAR 0.25000 vs 0.24500 (+0.00500) | FA events 3 vs 7 (-4) | precision 0.1683 vs 0.3137 | ms/frame 185.95 vs 189.22 (-3.28)
- [search_guard] search_margin_0.10 (mask=none): TSR 0.3300 vs 0.3450 (-0.0150) | FAR 0.21500 vs 0.24500 (-0.03000) | FA events 9 vs 7 (+2) | precision 0.3148 vs 0.3137 | ms/frame 195.16 vs 189.22 (+5.93)
- [search_guard] search_margin_0.20 (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 207.12 vs 189.22 (+17.89)
- [search_guard] search_stm_off (mask=none): TSR 0.2550 vs 0.3450 (-0.0900) | FAR 0.24000 vs 0.24500 (-0.00500) | FA events 7 vs 7 (+0) | precision 0.2537 vs 0.3137 | ms/frame 206.79 vs 189.22 (+17.56)
- [search_guard] search_stm_off_margin_0.10 (mask=none): TSR 0.3800 vs 0.3450 (+0.0350) | FAR 0.20500 vs 0.24500 (-0.04000) | FA events 7 vs 7 (+0) | precision 0.3641 vs 0.3137 | ms/frame 210.09 vs 189.22 (+20.87)
- [stm_ltm] balanced (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 165.55 vs 189.22 (-23.67)
- [stm_ltm] ltm_only (mask=none): TSR 0.3350 vs 0.3450 (-0.0100) | FAR 0.21000 vs 0.24500 (-0.03500) | FA events 7 vs 7 (+0) | precision 0.3354 vs 0.3137 | ms/frame 151.60 vs 189.22 (-37.62)
- [stm_ltm] stm_only (mask=none): TSR 0.2000 vs 0.3450 (-0.1450) | FAR 0.09500 vs 0.24500 (-0.15000) | FA events 9 vs 7 (+2) | precision 0.2106 vs 0.3137 | ms/frame 143.96 vs 189.22 (-45.26)
- [target_parts] parts_all (mask=none): TSR 0.3450 vs 0.3450 (+0.0000) | FAR 0.24500 vs 0.24500 (+0.00000) | FA events 7 vs 7 (+0) | precision 0.3137 vs 0.3137 | ms/frame 164.57 vs 189.22 (-24.65)
- [target_parts] parts_full_only (mask=none): TSR 0.0750 vs 0.3450 (-0.2700) | FAR 0.19000 vs 0.24500 (-0.05500) | FA events 5 vs 7 (-2) | precision 0.1201 vs 0.3137 | ms/frame 109.05 vs 189.22 (-80.18)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 189.22 |
| dancetrack0004 | part_matching_off | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 189.22 |
| dancetrack0004 | part_aware_off | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 164.52 |
| dancetrack0004 | margin_0.00 | none | 3 | 200 | 0.4150 | 0.3585 | 0.25000 | 6 | 0.5868 | 15 | 165.66 |
| dancetrack0004 | margin_0.06 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 164.29 |
| dancetrack0004 | margin_0.12 | none | 3 | 200 | 0.3950 | 0.3691 | 0.23000 | 8 | 0.5660 | 17 | 170.17 |
| dancetrack0004 | hard_negative_off | none | 3 | 200 | 0.4200 | 0.3637 | 0.09000 | 4 | 0.8000 | 13 | 162.70 |
| dancetrack0004 | hard_negative_confident | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 162.62 |
| dancetrack0004 | hard_negative_cap6 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 160.84 |
| dancetrack0004 | hard_negative_cap12 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 161.07 |
| dancetrack0004 | stm_only | none | 3 | 200 | 0.2000 | 0.2106 | 0.09500 | 9 | 0.6122 | 17 | 143.96 |
| dancetrack0004 | balanced | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 165.55 |
| dancetrack0004 | ltm_only | none | 3 | 200 | 0.3350 | 0.3354 | 0.21000 | 7 | 0.5882 | 9 | 151.60 |
| dancetrack0004 | ltm_cap_1 | none | 3 | 200 | 0.0500 | 0.0979 | 0.02500 | 3 | 0.6154 | 6 | 119.00 |
| dancetrack0004 | ltm_cap_96 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 163.01 |
| dancetrack0004 | ltm_cap_384 | none | 3 | 200 | 0.3150 | 0.3066 | 0.24500 | 7 | 0.5377 | 13 | 162.02 |
| dancetrack0004 | freeze_on | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 165.97 |
| dancetrack0004 | freeze_off | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 167.26 |
| dancetrack0004 | parts_full_only | none | 3 | 200 | 0.0750 | 0.1201 | 0.19000 | 5 | 0.2692 | 9 | 109.05 |
| dancetrack0004 | parts_all | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 164.57 |
| dancetrack0004 | gate_legacy | none | 3 | 200 | 0.0750 | 0.0826 | 0.02000 | 3 | 0.7895 | 4 | 133.56 |
| dancetrack0004 | gate_search_early | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 163.00 |
| dancetrack0004 | gate_search_late | none | 3 | 200 | 0.0800 | 0.0871 | 0.02000 | 3 | 0.8000 | 5 | 136.76 |
| dancetrack0004 | gate_growth_unfrozen | none | 3 | 200 | 0.2500 | 0.2559 | 0.10500 | 7 | 0.6866 | 12 | 166.80 |
| dancetrack0004 | vclamp_off | none | 3 | 200 | 0.0300 | 0.0389 | 0.05000 | 6 | 0.3750 | 6 | 143.26 |
| dancetrack0004 | vclamp_2 | none | 3 | 200 | 0.3800 | 0.3652 | 0.23000 | 4 | 0.5818 | 11 | 158.60 |
| dancetrack0004 | vclamp_4 | none | 3 | 200 | 0.1150 | 0.1271 | 0.09000 | 7 | 0.5610 | 9 | 139.04 |
| dancetrack0004 | drift_w4_0.35 | none | 3 | 200 | 0.1050 | 0.1167 | 0.03500 | 5 | 0.7308 | 9 | 170.36 |
| dancetrack0004 | drift_w4_0.50 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 205.44 |
| dancetrack0004 | drift_w8_0.35 | none | 3 | 200 | 0.1050 | 0.1179 | 0.05000 | 8 | 0.6667 | 13 | 180.97 |
| dancetrack0004 | drift_w8_0.50 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 213.12 |
| dancetrack0004 | search_stm_off | none | 3 | 200 | 0.2550 | 0.2537 | 0.24000 | 7 | 0.4545 | 16 | 206.79 |
| dancetrack0004 | search_margin_0.10 | none | 3 | 200 | 0.3300 | 0.3148 | 0.21500 | 9 | 0.5905 | 13 | 195.16 |
| dancetrack0004 | search_margin_0.20 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 207.12 |
| dancetrack0004 | search_stm_off_margin_0.10 | none | 3 | 200 | 0.3800 | 0.3641 | 0.20500 | 7 | 0.6095 | 14 | 210.09 |
| dancetrack0004 | anchor_4 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 206.53 |
| dancetrack0004 | anchor_16 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 210.60 |
| dancetrack0004 | anchor_32 | none | 3 | 200 | 0.3450 | 0.3140 | 0.24500 | 7 | 0.5377 | 15 | 205.65 |
| dancetrack0004 | hl_none | none | 3 | 200 | 0.3450 | 0.3140 | 0.24500 | 7 | 0.5377 | 15 | 207.11 |
| dancetrack0004 | hl_150 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 207.81 |
| dancetrack0004 | hl_5000 | none | 3 | 200 | 0.3450 | 0.3140 | 0.24500 | 7 | 0.5377 | 15 | 207.13 |
| dancetrack0004 | minq_0.50 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 207.30 |
| dancetrack0004 | minq_0.70 | none | 3 | 200 | 0.0550 | 0.0939 | 0.15500 | 10 | 0.2051 | 15 | 164.53 |
| dancetrack0004 | adaptive_thr_0.30 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 204.59 |
| dancetrack0004 | adaptive_thr_0.25 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 204.19 |
| dancetrack0004 | adaptive_thr_0.35 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 206.65 |
| dancetrack0004 | anchor_drift_0.25 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 205.63 |
| dancetrack0004 | anchor_drift_0.35 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 210.72 |
| dancetrack0004 | anchor_drift_0.50 | none | 3 | 200 | 0.4800 | 0.4186 | 0.11500 | 7 | 0.7830 | 17 | 227.59 |
| dancetrack0004 | anchor_ceiling_0.45 | none | 3 | 200 | 0.0300 | 0.0651 | 0.05000 | 7 | 0.3750 | 7 | 156.51 |
| dancetrack0004 | anchor_ceiling_0.55 | none | 3 | 200 | 0.2600 | 0.2577 | 0.22500 | 5 | 0.4706 | 11 | 200.21 |
| dancetrack0004 | anchor_ceiling_0.65 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 219.17 |
| dancetrack0004 | anchor_search_0.55 | none | 3 | 200 | 0.3700 | 0.3467 | 0.09000 | 6 | 0.8000 | 9 | 209.21 |
| dancetrack0004 | anchor_search_0.65 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 231.33 |
| dancetrack0004 | anchor_adapt_k2 | none | 3 | 200 | 0.4150 | 0.3756 | 0.21500 | 4 | 0.6261 | 9 | 229.87 |
| dancetrack0004 | anchor_adapt_k3 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 211.16 |
| dancetrack0004 | anchor_adapt_k4 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 214.91 |
| dancetrack0004 | anchor_adapt_k4_c0.55 | none | 3 | 200 | 0.0600 | 0.1039 | 0.23500 | 8 | 0.1754 | 9 | 161.75 |
| dancetrack0004 | anchor_adapt_k4_c0.60 | none | 3 | 200 | 0.2900 | 0.2875 | 0.10500 | 10 | 0.7162 | 15 | 187.50 |
| dancetrack0004 | anchor_adapt_k4_c0.65 | none | 3 | 200 | 0.3100 | 0.2889 | 0.22000 | 4 | 0.5368 | 11 | 207.57 |
| dancetrack0004 | anchor_adapt_k3_hi0.95 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 216.78 |
| dancetrack0004 | partial_m1.0_a1.6 | none | 3 | 200 | 0.1500 | 0.1683 | 0.25000 | 3 | 0.3243 | 10 | 185.95 |
| dancetrack0004 | partial_m0.4_a2.2 | none | 3 | 200 | 0.3850 | 0.3386 | 0.22000 | 3 | 0.5963 | 9 | 208.95 |
| dancetrack0004 | partial_m0.5_a2.0 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 213.54 |
| dancetrack0004 | partial_m0.2_a2.4 | none | 3 | 200 | 0.2500 | 0.2466 | 0.05500 | 8 | 0.8103 | 12 | 203.94 |
| dancetrack0004 | motion_band_k2 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 230.18 |
| dancetrack0004 | motion_band_k3 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 212.29 |
| dancetrack0004 | motion_band_k4 | none | 3 | 200 | 0.3450 | 0.3137 | 0.24500 | 7 | 0.5377 | 15 | 176.17 |
