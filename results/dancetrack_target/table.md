# Clicked-target strategy ablation

Config: ablate=gate,distractor_margin,target_parts,part_matching, allow_invisible_target=False, data_root=data\raw\dancetrack\val, detector=yolo, dry_run=False, embedder=real, guard={'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper']}, jobs=2, masks=['none', 'lower', 'upper'], max_frames=200, subsample=2, target_decisions={'dancetrack0004': {'target_id': 3, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 253.0, 'median_area': 46439.5, 'min_h': 76.0, 'min_area': 15708.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 126.5, 'median_area': 23219.75, 'min_h': 38.0, 'min_area': 7854.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 186, 'first': 6, 'median_w': 205.5, 'median_h': 126.5, 'median_area': 23219.75, 'min_h': 38.0, 'min_area': 7854.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 205.5, 'median_h': 126.5, 'median_area': 23219.75, 'frames': 186}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 363.0, 'median_area': 70419.0, 'min_h': 80.0, 'min_area': 22898.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 181.5, 'median_area': 35209.5, 'min_h': 40.0, 'min_area': 11449.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 198, 'first': 0, 'median_w': 269.0, 'median_h': 181.5, 'median_area': 35209.5, 'min_h': 40.0, 'min_area': 11449.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 269.0, 'median_h': 181.5, 'median_area': 35209.5, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 427.0, 'median_area': 108703.0, 'min_h': 109.0, 'min_area': 35397.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 213.5, 'median_area': 54351.5, 'min_h': 54.5, 'min_area': 17698.5, 'visible_fraction': 0.5000037434348816}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 317.0, 'median_h': 213.5, 'median_area': 54351.5, 'min_h': 54.5, 'min_area': 17698.5, 'visible_fraction': 0.49999624545700294}}, 'worst_mask': 'lower', 'median_w': 317.0, 'median_h': 213.5, 'median_area': 54351.5, 'frames': 197}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'frames': 200}}, 'selected': {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 446.0, 'median_area': 146531.5, 'min_h': 160.0, 'min_area': 58680.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'min_h': 80.0, 'min_area': 29340.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 387.0, 'median_h': 223.0, 'median_area': 73265.75, 'frames': 200}, 'message': ''}, 'dancetrack0005': {'target_id': 0, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'frames': 200}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 440.5, 'median_area': 107985.0, 'min_h': 282.0, 'min_area': 31866.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 220.25, 'median_area': 53992.5, 'min_h': 141.0, 'min_area': 15933.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 198, 'first': 2, 'median_w': 225.0, 'median_h': 220.25, 'median_area': 53992.5, 'min_h': 141.0, 'min_area': 15933.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 225.0, 'median_h': 220.25, 'median_area': 53992.5, 'frames': 198}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 424.5, 'median_area': 89078.5, 'min_h': 251.0, 'min_area': 27610.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 212.25, 'median_area': 44539.25, 'min_h': 125.5, 'min_area': 13805.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 189.5, 'median_h': 212.25, 'median_area': 44539.25, 'min_h': 125.5, 'min_area': 13805.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 189.5, 'median_h': 212.25, 'median_area': 44539.25, 'frames': 200}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 488.0, 'median_area': 115661.0, 'min_h': 313.0, 'min_area': 49248.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 244.0, 'median_area': 57830.5, 'min_h': 156.5, 'min_area': 24624.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 224.0, 'median_h': 244.0, 'median_area': 57830.5, 'min_h': 156.5, 'min_area': 24624.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 224.0, 'median_h': 244.0, 'median_area': 57830.5, 'frames': 200}}, 'selected': {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 465.5, 'median_area': 103367.5, 'min_h': 291.0, 'min_area': 28595.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'min_h': 145.5, 'min_area': 14297.5, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 219.0, 'median_h': 232.75, 'median_area': 51683.75, 'frames': 200}, 'message': ''}, 'dancetrack0007': {'target_id': 1, 'reason': 'observable', 'invisible': False, 'control': 'partial-view', 'guard_passed': True, 'guard': {'min_visible_h': 80.0, 'min_visible_area': 0.0, 'masks': ['none', 'lower', 'upper']}, 'observability': {0: {'target_id': 0, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 360.0, 'median_area': 51051.0, 'min_h': 296.0, 'min_area': 27966.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 180.0, 'median_area': 25525.5, 'min_h': 148.0, 'min_area': 13983.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 199, 'first': 0, 'median_w': 156.0, 'median_h': 180.0, 'median_area': 25525.5, 'min_h': 148.0, 'min_area': 13983.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 156.0, 'median_h': 180.0, 'median_area': 25525.5, 'frames': 199}, 1: {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'frames': 197}, 2: {'target_id': 2, 'masks': {'none': {'mode': 'none', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 345.0, 'median_area': 52969.0, 'min_h': 252.0, 'min_area': 26208.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 172.5, 'median_area': 26484.5, 'min_h': 126.0, 'min_area': 13104.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 199, 'first': 0, 'median_w': 155.0, 'median_h': 172.5, 'median_area': 26484.5, 'min_h': 126.0, 'min_area': 13104.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 155.0, 'median_h': 172.5, 'median_area': 26484.5, 'frames': 199}, 3: {'target_id': 3, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 302.0, 'median_area': 36486.5, 'min_h': 244.0, 'min_area': 20631.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 151.0, 'median_area': 18243.25, 'min_h': 122.0, 'min_area': 10315.5, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 126.5, 'median_h': 151.0, 'median_area': 18243.25, 'min_h': 122.0, 'min_area': 10315.5, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 126.5, 'median_h': 151.0, 'median_area': 18243.25, 'frames': 200}, 4: {'target_id': 4, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 352.0, 'median_area': 45147.0, 'min_h': 260.0, 'min_area': 27040.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 176.0, 'median_area': 22573.5, 'min_h': 130.0, 'min_area': 13520.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 141.0, 'median_h': 176.0, 'median_area': 22573.5, 'min_h': 130.0, 'min_area': 13520.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 141.0, 'median_h': 176.0, 'median_area': 22573.5, 'frames': 200}, 5: {'target_id': 5, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 314.5, 'median_area': 37343.0, 'min_h': 278.0, 'min_area': 18711.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 157.25, 'median_area': 18671.5, 'min_h': 139.0, 'min_area': 9355.5, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 117.0, 'median_h': 157.25, 'median_area': 18671.5, 'min_h': 139.0, 'min_area': 9355.5, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 117.0, 'median_h': 157.25, 'median_area': 18671.5, 'frames': 200}, 6: {'target_id': 6, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 331.5, 'median_area': 46544.5, 'min_h': 266.0, 'min_area': 24846.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 165.75, 'median_area': 23272.25, 'min_h': 133.0, 'min_area': 12423.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 143.0, 'median_h': 165.75, 'median_area': 23272.25, 'min_h': 133.0, 'min_area': 12423.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 143.0, 'median_h': 165.75, 'median_area': 23272.25, 'frames': 200}, 7: {'target_id': 7, 'masks': {'none': {'mode': 'none', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 335.0, 'median_area': 48332.0, 'min_h': 279.0, 'min_area': 27808.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 167.5, 'median_area': 24166.0, 'min_h': 139.5, 'min_area': 13904.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 200, 'first': 0, 'median_w': 140.5, 'median_h': 167.5, 'median_area': 24166.0, 'min_h': 139.5, 'min_area': 13904.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 140.5, 'median_h': 167.5, 'median_area': 24166.0, 'frames': 200}}, 'selected': {'target_id': 1, 'masks': {'none': {'mode': 'none', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 332.0, 'median_area': 48042.0, 'min_h': 181.0, 'min_area': 15416.0, 'visible_fraction': 1.0}, 'lower': {'mode': 'lower', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}, 'upper': {'mode': 'upper', 'frames': 197, 'first': 0, 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'min_h': 90.5, 'min_area': 7708.0, 'visible_fraction': 0.5}}, 'worst_mask': 'lower', 'median_w': 148.0, 'median_h': 166.0, 'median_area': 24021.0, 'frames': 197}, 'message': ''}}, target_ids={'dancetrack0004': 3, 'dancetrack0005': 0, 'dancetrack0007': 1}, target_strategy=stressed, wall_seconds=2694.22

## Pooled results (frame-weighted over sequences)

| group | variant | mask | seqs | frames | visible | TSR | prec | loss | loss_lock | recq_rate | recq_lat | recq_p90 | recq_max | FAR | FAR_lock | FA_ev | FA_ev_lock | id_cons | switches | frag | longest | coast_ok | worst_id | detect | ms/frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | lower | 3 | 600 | 597 | 0.3132 | 0.2791 | 0.6868 | 0.1407 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.10500 | 0.18216 | 27 | 25 | 0.7470 | 24 | 11 | 60 | - | - | - | 291.64 |
| baseline | baseline | none | 3 | 600 | 597 | 0.2228 | 0.2266 | 0.7772 | 0.5008 | 0.0000 | - | - | - | 0.49333 | 0.67816 | 13 | 13 | 0.3035 | 11 | 4 | 62 | - | - | - | 309.85 |
| baseline | baseline | upper | 3 | 600 | 597 | 0.5729 | 0.4803 | 0.4271 | 0.0888 | 0.0000 | - | - | - | 0.00833 | 0.01039 | 5 | 4 | 0.9856 | 8 | 33 | 59 | - | - | - | 283.68 |
| distractor_margin | margin_0.00 | lower | 3 | 600 | 597 | 0.3484 | 0.2990 | 0.6516 | 0.2312 | 0.0000 | - | - | - | 0.19833 | 0.32464 | 21 | 22 | 0.6350 | 18 | 10 | 65 | - | - | - | 267.83 |
| distractor_margin | margin_0.00 | none | 3 | 600 | 597 | 0.2998 | 0.2897 | 0.7002 | 0.5176 | 0.0000 | - | - | - | 0.51667 | 0.62729 | 12 | 10 | 0.3568 | 15 | 10 | 62 | - | - | - | 290.26 |
| distractor_margin | margin_0.00 | upper | 3 | 600 | 597 | 0.5678 | 0.4792 | 0.4322 | 0.0838 | 0.0000 | - | - | - | 0.01333 | 0.01837 | 7 | 7 | 0.9769 | 11 | 27 | 59 | - | - | - | 276.59 |
| distractor_margin | margin_0.06 | lower | 3 | 600 | 597 | 0.3132 | 0.2791 | 0.6868 | 0.1407 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.10500 | 0.18216 | 27 | 25 | 0.7470 | 24 | 11 | 60 | - | - | - | 261.78 |
| distractor_margin | margin_0.06 | none | 3 | 600 | 597 | 0.2228 | 0.2266 | 0.7772 | 0.5008 | 0.0000 | - | - | - | 0.49333 | 0.67816 | 13 | 13 | 0.3035 | 11 | 4 | 62 | - | - | - | 286.11 |
| distractor_margin | margin_0.06 | upper | 3 | 600 | 597 | 0.5729 | 0.4803 | 0.4271 | 0.0888 | 0.0000 | - | - | - | 0.00833 | 0.01039 | 5 | 4 | 0.9856 | 8 | 33 | 59 | - | - | - | 270.57 |
| distractor_margin | margin_0.12 | lower | 3 | 600 | 597 | 0.3869 | 0.3331 | 0.6131 | 0.0771 | 1.0000 | 5.00 | 5.00 | 5.00 | 0.02000 | 0.04396 | 10 | 10 | 0.9506 | 10 | 12 | 67 | - | - | - | 273.29 |
| distractor_margin | margin_0.12 | none | 3 | 600 | 597 | 0.2529 | 0.2524 | 0.7471 | 0.4774 | 0.0000 | - | - | - | 0.47333 | 0.64908 | 12 | 12 | 0.3395 | 11 | 6 | 62 | - | - | - | 288.50 |
| distractor_margin | margin_0.12 | upper | 3 | 600 | 597 | 0.5243 | 0.4437 | 0.4757 | 0.0804 | 0.0000 | - | - | - | 0.00500 | 0.00570 | 3 | 2 | 0.9905 | 5 | 28 | 59 | - | - | - | 260.81 |
| gate | gate_growth_unfrozen | lower | 3 | 600 | 597 | 0.2915 | 0.2587 | 0.7085 | 0.1491 | 0.0000 | - | - | - | 0.10833 | 0.19615 | 29 | 27 | 0.7269 | 25 | 9 | 60 | - | - | - | 259.82 |
| gate | gate_growth_unfrozen | none | 3 | 600 | 597 | 0.2228 | 0.2240 | 0.7772 | 0.5578 | 0.0000 | - | - | - | 0.55333 | 0.70576 | 11 | 11 | 0.2798 | 9 | 4 | 62 | - | - | - | 291.92 |
| gate | gate_growth_unfrozen | upper | 3 | 600 | 597 | 0.5745 | 0.4813 | 0.4255 | 0.0838 | 0.0000 | - | - | - | 0.00833 | 0.01044 | 4 | 3 | 0.9856 | 8 | 34 | 59 | - | - | - | 261.53 |
| gate | gate_legacy | lower | 3 | 600 | 597 | 0.2864 | 0.2296 | 0.7136 | 0.0570 | 0.0000 | - | - | - | 0.02667 | 0.07426 | 12 | 12 | 0.9144 | 11 | 6 | 60 | - | - | - | 254.77 |
| gate | gate_legacy | none | 3 | 600 | 597 | 0.1759 | 0.1720 | 0.8241 | 0.4523 | 0.0000 | - | - | - | 0.44667 | 0.71200 | 6 | 6 | 0.2737 | 5 | 3 | 62 | - | - | - | 287.21 |
| gate | gate_legacy | upper | 3 | 600 | 597 | 0.4255 | 0.3452 | 0.5745 | 0.0436 | 0.0000 | - | - | - | 0.00333 | 0.00741 | 2 | 2 | 0.9922 | 4 | 15 | 59 | - | - | - | 242.30 |
| gate | gate_search_early | lower | 3 | 600 | 597 | 0.3132 | 0.2791 | 0.6868 | 0.1407 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.10500 | 0.18216 | 27 | 25 | 0.7470 | 24 | 11 | 60 | - | - | - | 243.14 |
| gate | gate_search_early | none | 3 | 600 | 597 | 0.2228 | 0.2266 | 0.7772 | 0.5008 | 0.0000 | - | - | - | 0.49333 | 0.67816 | 13 | 13 | 0.3035 | 11 | 4 | 62 | - | - | - | 268.28 |
| gate | gate_search_early | upper | 3 | 600 | 597 | 0.5729 | 0.4803 | 0.4271 | 0.0888 | 0.0000 | - | - | - | 0.00833 | 0.01039 | 5 | 4 | 0.9856 | 8 | 33 | 59 | - | - | - | 275.16 |
| gate | gate_search_late | lower | 3 | 600 | 597 | 0.3116 | 0.2530 | 0.6884 | 0.0486 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.02167 | 0.05634 | 9 | 9 | 0.9347 | 12 | 10 | 60 | - | - | - | 274.43 |
| gate | gate_search_late | none | 3 | 600 | 597 | 0.1910 | 0.1901 | 0.8090 | 0.4003 | 0.0000 | - | - | - | 0.39333 | 0.65730 | 9 | 8 | 0.3179 | 13 | 6 | 62 | - | - | - | 279.01 |
| gate | gate_search_late | upper | 3 | 600 | 597 | 0.3668 | 0.3031 | 0.6332 | 0.0787 | 0.0000 | - | - | - | 0.03500 | 0.06250 | 8 | 8 | 0.9125 | 10 | 20 | 59 | - | - | - | 260.29 |
| part_matching | part_matching_off | lower | 3 | 600 | 597 | 0.3132 | 0.2791 | 0.6868 | 0.1407 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.10500 | 0.18216 | 27 | 25 | 0.7470 | 24 | 11 | 60 | - | - | - | 254.34 |
| part_matching | part_matching_off | none | 3 | 600 | 597 | 0.2228 | 0.2266 | 0.7772 | 0.5008 | 0.0000 | - | - | - | 0.49333 | 0.67816 | 13 | 13 | 0.3035 | 11 | 4 | 62 | - | - | - | 281.49 |
| part_matching | part_matching_off | upper | 3 | 600 | 597 | 0.5729 | 0.4803 | 0.4271 | 0.0888 | 0.0000 | - | - | - | 0.00833 | 0.01039 | 5 | 4 | 0.9856 | 8 | 33 | 59 | - | - | - | 252.15 |
| target_parts | parts_all | lower | 3 | 600 | 597 | 0.3132 | 0.2791 | 0.6868 | 0.1407 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.10500 | 0.18216 | 27 | 25 | 0.7470 | 24 | 11 | 60 | - | - | - | 269.47 |
| target_parts | parts_all | none | 3 | 600 | 597 | 0.2228 | 0.2266 | 0.7772 | 0.5008 | 0.0000 | - | - | - | 0.49333 | 0.67816 | 13 | 13 | 0.3035 | 11 | 4 | 62 | - | - | - | 278.39 |
| target_parts | parts_all | upper | 3 | 600 | 597 | 0.5729 | 0.4803 | 0.4271 | 0.0888 | 0.0000 | - | - | - | 0.00833 | 0.01039 | 5 | 4 | 0.9856 | 8 | 33 | 59 | - | - | - | 287.81 |
| target_parts | parts_full_only | lower | 3 | 600 | 597 | 0.2814 | 0.2376 | 0.7186 | 0.1173 | 1.0000 | 7.00 | 7.00 | 7.00 | 0.07667 | 0.15745 | 22 | 25 | 0.7850 | 18 | 11 | 58 | - | - | - | 196.00 |
| target_parts | parts_full_only | none | 3 | 600 | 597 | 0.2077 | 0.2078 | 0.7923 | 0.4908 | 0.0000 | - | - | - | 0.47167 | 0.67381 | 14 | 14 | 0.2978 | 12 | 10 | 62 | - | - | - | 222.57 |
| target_parts | parts_full_only | upper | 3 | 600 | 597 | 0.4288 | 0.3883 | 0.5712 | 0.0704 | 0.0000 | - | - | - | 0.03167 | 0.03831 | 9 | 8 | 0.9309 | 12 | 27 | 59 | - | - | - | 196.05 |

Observability guard: the target must keep a median post-mask visible box of at least 80 px of height and 0 px^2 under **every** requested mask; runs that fail it are excluded from every table above and listed in the control section below.

Attribution note: coast_ok counts frames in a blind state that are still on the target (success by inertia, not evidence of a lock); worst_id counts frames where a **non-blind** state reports a box overlapping a different real identity. The metric layer books false_acquisition only at an IoU of at least 0.5 with that other identity, so a confident wrong lock sitting at IoU 0.22 is counted by worst_id and not by FAR -- read the two columns together.

## Invisible-target control (excluded from the headline)

(no unobservable-target run)

## Target selection (per sequence)

| sequence | target | reason | control | worst mask | visible median | frames |
| --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | 3 | observable | partial-view | lower | 387x223 | 200 |
| dancetrack0005 | 0 | observable | partial-view | lower | 219x233 | 200 |
| dancetrack0007 | 1 | observable | partial-view | lower | 148x166 | 197 |


## Partial-view cross-tab (part matching ON vs OFF)

| mask | part_matching ON TSR | OFF TSR | dTSR | ON FAR | OFF FAR | dFAR | ON ms | OFF ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lower | 0.3132 | 0.3132 | +0.0000 | 0.10500 | 0.10500 | +0.00000 | 291.64 | 254.34 |
| none | 0.2228 | 0.2228 | +0.0000 | 0.49333 | 0.49333 | +0.00000 | 309.85 | 281.49 |
| upper | 0.5729 | 0.5729 | +0.0000 | 0.00833 | 0.00833 | +0.00000 | 283.68 | 252.15 |

## Claims (delta vs the same-mask baseline)

- [distractor_margin] margin_0.00 (mask=lower): TSR 0.3484 vs 0.3132 (+0.0352) | FAR 0.19833 vs 0.10500 (+0.09333) | FA events 21 vs 27 (-6) | precision 0.2990 vs 0.2791 | ms/frame 267.83 vs 291.64 (-23.81)
- [distractor_margin] margin_0.00 (mask=none): TSR 0.2998 vs 0.2228 (+0.0771) | FAR 0.51667 vs 0.49333 (+0.02333) | FA events 12 vs 13 (-1) | precision 0.2897 vs 0.2266 | ms/frame 290.26 vs 309.85 (-19.59)
- [distractor_margin] margin_0.00 (mask=upper): TSR 0.5678 vs 0.5729 (-0.0050) | FAR 0.01333 vs 0.00833 (+0.00500) | FA events 7 vs 5 (+2) | precision 0.4792 vs 0.4803 | ms/frame 276.59 vs 283.68 (-7.09)
- [distractor_margin] margin_0.06 (mask=lower): TSR 0.3132 vs 0.3132 (+0.0000) | FAR 0.10500 vs 0.10500 (+0.00000) | FA events 27 vs 27 (+0) | precision 0.2791 vs 0.2791 | ms/frame 261.78 vs 291.64 (-29.86)
- [distractor_margin] margin_0.06 (mask=none): TSR 0.2228 vs 0.2228 (+0.0000) | FAR 0.49333 vs 0.49333 (+0.00000) | FA events 13 vs 13 (+0) | precision 0.2266 vs 0.2266 | ms/frame 286.11 vs 309.85 (-23.74)
- [distractor_margin] margin_0.06 (mask=upper): TSR 0.5729 vs 0.5729 (+0.0000) | FAR 0.00833 vs 0.00833 (+0.00000) | FA events 5 vs 5 (+0) | precision 0.4803 vs 0.4803 | ms/frame 270.57 vs 283.68 (-13.11)
- [distractor_margin] margin_0.12 (mask=lower): TSR 0.3869 vs 0.3132 (+0.0737) | FAR 0.02000 vs 0.10500 (-0.08500) | FA events 10 vs 27 (-17) | precision 0.3331 vs 0.2791 | ms/frame 273.29 vs 291.64 (-18.35)
- [distractor_margin] margin_0.12 (mask=none): TSR 0.2529 vs 0.2228 (+0.0302) | FAR 0.47333 vs 0.49333 (-0.02000) | FA events 12 vs 13 (-1) | precision 0.2524 vs 0.2266 | ms/frame 288.50 vs 309.85 (-21.35)
- [distractor_margin] margin_0.12 (mask=upper): TSR 0.5243 vs 0.5729 (-0.0486) | FAR 0.00500 vs 0.00833 (-0.00333) | FA events 3 vs 5 (-2) | precision 0.4437 vs 0.4803 | ms/frame 260.81 vs 283.68 (-22.87)
- [gate] gate_growth_unfrozen (mask=lower): TSR 0.2915 vs 0.3132 (-0.0218) | FAR 0.10833 vs 0.10500 (+0.00333) | FA events 29 vs 27 (+2) | precision 0.2587 vs 0.2791 | ms/frame 259.82 vs 291.64 (-31.83)
- [gate] gate_growth_unfrozen (mask=none): TSR 0.2228 vs 0.2228 (+0.0000) | FAR 0.55333 vs 0.49333 (+0.06000) | FA events 11 vs 13 (-2) | precision 0.2240 vs 0.2266 | ms/frame 291.92 vs 309.85 (-17.93)
- [gate] gate_growth_unfrozen (mask=upper): TSR 0.5745 vs 0.5729 (+0.0017) | FAR 0.00833 vs 0.00833 (+0.00000) | FA events 4 vs 5 (-1) | precision 0.4813 vs 0.4803 | ms/frame 261.53 vs 283.68 (-22.15)
- [gate] gate_legacy (mask=lower): TSR 0.2864 vs 0.3132 (-0.0268) | FAR 0.02667 vs 0.10500 (-0.07833) | FA events 12 vs 27 (-15) | precision 0.2296 vs 0.2791 | ms/frame 254.77 vs 291.64 (-36.87)
- [gate] gate_legacy (mask=none): TSR 0.1759 vs 0.2228 (-0.0469) | FAR 0.44667 vs 0.49333 (-0.04667) | FA events 6 vs 13 (-7) | precision 0.1720 vs 0.2266 | ms/frame 287.21 vs 309.85 (-22.64)
- [gate] gate_legacy (mask=upper): TSR 0.4255 vs 0.5729 (-0.1474) | FAR 0.00333 vs 0.00833 (-0.00500) | FA events 2 vs 5 (-3) | precision 0.3452 vs 0.4803 | ms/frame 242.30 vs 283.68 (-41.38)
- [gate] gate_search_early (mask=lower): TSR 0.3132 vs 0.3132 (+0.0000) | FAR 0.10500 vs 0.10500 (+0.00000) | FA events 27 vs 27 (+0) | precision 0.2791 vs 0.2791 | ms/frame 243.14 vs 291.64 (-48.51)
- [gate] gate_search_early (mask=none): TSR 0.2228 vs 0.2228 (+0.0000) | FAR 0.49333 vs 0.49333 (+0.00000) | FA events 13 vs 13 (+0) | precision 0.2266 vs 0.2266 | ms/frame 268.28 vs 309.85 (-41.57)
- [gate] gate_search_early (mask=upper): TSR 0.5729 vs 0.5729 (+0.0000) | FAR 0.00833 vs 0.00833 (+0.00000) | FA events 5 vs 5 (+0) | precision 0.4803 vs 0.4803 | ms/frame 275.16 vs 283.68 (-8.52)
- [gate] gate_search_late (mask=lower): TSR 0.3116 vs 0.3132 (-0.0017) | FAR 0.02167 vs 0.10500 (-0.08333) | FA events 9 vs 27 (-18) | precision 0.2530 vs 0.2791 | ms/frame 274.43 vs 291.64 (-17.22)
- [gate] gate_search_late (mask=none): TSR 0.1910 vs 0.2228 (-0.0318) | FAR 0.39333 vs 0.49333 (-0.10000) | FA events 9 vs 13 (-4) | precision 0.1901 vs 0.2266 | ms/frame 279.01 vs 309.85 (-30.83)
- [gate] gate_search_late (mask=upper): TSR 0.3668 vs 0.5729 (-0.2060) | FAR 0.03500 vs 0.00833 (+0.02667) | FA events 8 vs 5 (+3) | precision 0.3031 vs 0.4803 | ms/frame 260.29 vs 283.68 (-23.39)
- [part_matching] part_matching_off (mask=lower): TSR 0.3132 vs 0.3132 (+0.0000) | FAR 0.10500 vs 0.10500 (+0.00000) | FA events 27 vs 27 (+0) | precision 0.2791 vs 0.2791 | ms/frame 254.34 vs 291.64 (-37.30)
- [part_matching] part_matching_off (mask=none): TSR 0.2228 vs 0.2228 (+0.0000) | FAR 0.49333 vs 0.49333 (+0.00000) | FA events 13 vs 13 (+0) | precision 0.2266 vs 0.2266 | ms/frame 281.49 vs 309.85 (-28.35)
- [part_matching] part_matching_off (mask=upper): TSR 0.5729 vs 0.5729 (+0.0000) | FAR 0.00833 vs 0.00833 (+0.00000) | FA events 5 vs 5 (+0) | precision 0.4803 vs 0.4803 | ms/frame 252.15 vs 283.68 (-31.53)
- [target_parts] parts_all (mask=lower): TSR 0.3132 vs 0.3132 (+0.0000) | FAR 0.10500 vs 0.10500 (+0.00000) | FA events 27 vs 27 (+0) | precision 0.2791 vs 0.2791 | ms/frame 269.47 vs 291.64 (-22.17)
- [target_parts] parts_all (mask=none): TSR 0.2228 vs 0.2228 (+0.0000) | FAR 0.49333 vs 0.49333 (+0.00000) | FA events 13 vs 13 (+0) | precision 0.2266 vs 0.2266 | ms/frame 278.39 vs 309.85 (-31.45)
- [target_parts] parts_all (mask=upper): TSR 0.5729 vs 0.5729 (+0.0000) | FAR 0.00833 vs 0.00833 (+0.00000) | FA events 5 vs 5 (+0) | precision 0.4803 vs 0.4803 | ms/frame 287.81 vs 283.68 (+4.13)
- [target_parts] parts_full_only (mask=lower): TSR 0.2814 vs 0.3132 (-0.0318) | FAR 0.07667 vs 0.10500 (-0.02833) | FA events 22 vs 27 (-5) | precision 0.2376 vs 0.2791 | ms/frame 196.00 vs 291.64 (-95.65)
- [target_parts] parts_full_only (mask=none): TSR 0.2077 vs 0.2228 (-0.0151) | FAR 0.47167 vs 0.49333 (-0.02167) | FA events 14 vs 13 (+1) | precision 0.2078 vs 0.2266 | ms/frame 222.57 vs 309.85 (-87.28)
- [target_parts] parts_full_only (mask=upper): TSR 0.4288 vs 0.5729 (-0.1441) | FAR 0.03167 vs 0.00833 (+0.02333) | FA events 9 vs 5 (+4) | precision 0.3883 vs 0.4803 | ms/frame 196.05 vs 283.68 (-87.63)

## Per-run detail

| sequence | variant | mask | target_id | frames_processed | target_success_rate | target_precision | false_acquisition_rate | false_acquisition_events | id_consistency | target_switch_count | ms_per_frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dancetrack0004 | baseline | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 306.63 |
| dancetrack0004 | baseline | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 282.96 |
| dancetrack0004 | baseline | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 264.35 |
| dancetrack0004 | gate_legacy | none | 3 | 200 | 0.0350 | 0.0389 | 0.00500 | 1 | 0.8750 | 1 | 249.15 |
| dancetrack0004 | gate_legacy | lower | 3 | 200 | 0.0200 | 0.0232 | 0.02000 | 2 | 0.5000 | 2 | 190.91 |
| dancetrack0004 | gate_legacy | upper | 3 | 200 | 0.0350 | 0.0389 | 0.00000 | 0 | 1.0000 | 0 | 185.79 |
| dancetrack0004 | gate_search_early | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 191.06 |
| dancetrack0004 | gate_search_early | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 167.92 |
| dancetrack0004 | gate_search_early | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 246.80 |
| dancetrack0004 | gate_search_late | none | 3 | 200 | 0.0800 | 0.0784 | 0.02000 | 4 | 0.8000 | 7 | 241.29 |
| dancetrack0004 | gate_search_late | lower | 3 | 200 | 0.0250 | 0.0272 | 0.03000 | 3 | 0.4545 | 3 | 231.19 |
| dancetrack0004 | gate_search_late | upper | 3 | 200 | 0.1350 | 0.1269 | 0.00500 | 1 | 0.9643 | 2 | 238.37 |
| dancetrack0004 | gate_growth_unfrozen | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 241.20 |
| dancetrack0004 | gate_growth_unfrozen | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 218.16 |
| dancetrack0004 | gate_growth_unfrozen | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 245.45 |
| dancetrack0004 | margin_0.00 | none | 3 | 200 | 0.4050 | 0.3957 | 0.10500 | 9 | 0.7900 | 13 | 272.09 |
| dancetrack0004 | margin_0.00 | lower | 3 | 200 | 0.0550 | 0.0778 | 0.09000 | 7 | 0.3571 | 10 | 236.96 |
| dancetrack0004 | margin_0.00 | upper | 3 | 200 | 0.3200 | 0.3186 | 0.01500 | 2 | 0.9552 | 5 | 273.68 |
| dancetrack0004 | margin_0.06 | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 259.15 |
| dancetrack0004 | margin_0.06 | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 240.33 |
| dancetrack0004 | margin_0.06 | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 265.13 |
| dancetrack0004 | margin_0.12 | none | 3 | 200 | 0.2650 | 0.2663 | 0.04000 | 7 | 0.8667 | 7 | 267.31 |
| dancetrack0004 | margin_0.12 | lower | 3 | 200 | 0.1350 | 0.1528 | 0.02500 | 4 | 0.8438 | 3 | 237.80 |
| dancetrack0004 | margin_0.12 | upper | 3 | 200 | 0.1800 | 0.2055 | 0.00500 | 1 | 0.9730 | 2 | 219.49 |
| dancetrack0004 | parts_full_only | none | 3 | 200 | 0.1350 | 0.1504 | 0.09000 | 8 | 0.5909 | 9 | 188.13 |
| dancetrack0004 | parts_full_only | lower | 3 | 200 | 0.0250 | 0.0573 | 0.07500 | 9 | 0.2500 | 6 | 172.87 |
| dancetrack0004 | parts_full_only | upper | 3 | 200 | 0.2250 | 0.2649 | 0.04500 | 4 | 0.8333 | 7 | 192.73 |
| dancetrack0004 | parts_all | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 232.85 |
| dancetrack0004 | parts_all | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 220.83 |
| dancetrack0004 | parts_all | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 249.26 |
| dancetrack0004 | part_matching_off | none | 3 | 200 | 0.1750 | 0.1918 | 0.22500 | 6 | 0.4375 | 5 | 253.89 |
| dancetrack0004 | part_matching_off | lower | 3 | 200 | 0.0350 | 0.0859 | 0.15500 | 12 | 0.1622 | 10 | 240.39 |
| dancetrack0004 | part_matching_off | upper | 3 | 200 | 0.3050 | 0.3001 | 0.00500 | 1 | 0.9839 | 2 | 250.19 |
| dancetrack0005 | baseline | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 321.45 |
| dancetrack0005 | baseline | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 272.99 |
| dancetrack0005 | baseline | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 268.18 |
| dancetrack0005 | gate_legacy | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 311.38 |
| dancetrack0005 | gate_legacy | lower | 0 | 200 | 0.3000 | 0.2246 | 0.03500 | 5 | 0.8955 | 2 | 269.53 |
| dancetrack0005 | gate_legacy | upper | 0 | 200 | 0.4650 | 0.3981 | 0.01000 | 2 | 0.9789 | 4 | 240.96 |
| dancetrack0005 | gate_search_early | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 326.90 |
| dancetrack0005 | gate_search_early | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 268.48 |
| dancetrack0005 | gate_search_early | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 253.69 |
| dancetrack0005 | gate_search_late | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 303.16 |
| dancetrack0005 | gate_search_late | lower | 0 | 200 | 0.3050 | 0.2295 | 0.02000 | 3 | 0.9385 | 3 | 282.49 |
| dancetrack0005 | gate_search_late | upper | 0 | 200 | 0.1950 | 0.1868 | 0.09000 | 5 | 0.6842 | 6 | 225.05 |
| dancetrack0005 | gate_growth_unfrozen | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 317.45 |
| dancetrack0005 | gate_growth_unfrozen | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 257.16 |
| dancetrack0005 | gate_growth_unfrozen | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 244.61 |
| dancetrack0005 | margin_0.00 | none | 0 | 200 | 0.1800 | 0.1780 | 0.80500 | 1 | 0.1744 | 1 | 298.24 |
| dancetrack0005 | margin_0.00 | lower | 0 | 200 | 0.3000 | 0.2387 | 0.47500 | 8 | 0.3871 | 2 | 262.74 |
| dancetrack0005 | margin_0.00 | upper | 0 | 200 | 0.6050 | 0.5227 | 0.01000 | 2 | 0.9837 | 3 | 244.50 |
| dancetrack0005 | margin_0.06 | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 307.67 |
| dancetrack0005 | margin_0.06 | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 253.79 |
| dancetrack0005 | margin_0.06 | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 244.37 |
| dancetrack0005 | margin_0.12 | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 297.34 |
| dancetrack0005 | margin_0.12 | lower | 0 | 200 | 0.3300 | 0.2656 | 0.03000 | 5 | 0.9167 | 5 | 283.90 |
| dancetrack0005 | margin_0.12 | upper | 0 | 200 | 0.6200 | 0.5317 | 0.00500 | 1 | 0.9920 | 2 | 265.42 |
| dancetrack0005 | parts_full_only | none | 0 | 200 | 0.1750 | 0.1750 | 0.69500 | 3 | 0.2011 | 2 | 243.99 |
| dancetrack0005 | parts_full_only | lower | 0 | 200 | 0.2800 | 0.2170 | 0.13500 | 9 | 0.6747 | 4 | 214.96 |
| dancetrack0005 | parts_full_only | upper | 0 | 200 | 0.2850 | 0.2971 | 0.04000 | 3 | 0.8769 | 4 | 158.90 |
| dancetrack0005 | parts_all | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 323.28 |
| dancetrack0005 | parts_all | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 278.39 |
| dancetrack0005 | parts_all | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 282.10 |
| dancetrack0005 | part_matching_off | none | 0 | 200 | 0.1800 | 0.1799 | 0.80000 | 2 | 0.1795 | 3 | 291.24 |
| dancetrack0005 | part_matching_off | lower | 0 | 200 | 0.3000 | 0.2491 | 0.14500 | 12 | 0.6742 | 8 | 213.52 |
| dancetrack0005 | part_matching_off | upper | 0 | 200 | 0.6400 | 0.5450 | 0.01000 | 2 | 0.9846 | 4 | 215.69 |
| dancetrack0007 | baseline | none | 1 | 200 | 0.3147 | 0.3080 | 0.45500 | 5 | 0.3933 | 3 | 301.47 |
| dancetrack0007 | baseline | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 318.97 |
| dancetrack0007 | baseline | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 318.51 |
| dancetrack0007 | gate_legacy | none | 1 | 200 | 0.3147 | 0.2970 | 0.53500 | 3 | 0.3554 | 1 | 301.09 |
| dancetrack0007 | gate_legacy | lower | 1 | 200 | 0.5431 | 0.4410 | 0.02500 | 5 | 0.9554 | 7 | 303.88 |
| dancetrack0007 | gate_legacy | upper | 1 | 200 | 0.7817 | 0.5987 | 0.00000 | 0 | 1.0000 | 0 | 300.15 |
| dancetrack0007 | gate_search_early | none | 1 | 200 | 0.3147 | 0.3080 | 0.45500 | 5 | 0.3933 | 3 | 286.88 |
| dancetrack0007 | gate_search_early | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 293.01 |
| dancetrack0007 | gate_search_early | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 325.00 |
| dancetrack0007 | gate_search_late | none | 1 | 200 | 0.3147 | 0.3119 | 0.36000 | 3 | 0.4504 | 3 | 292.58 |
| dancetrack0007 | gate_search_late | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 309.60 |
| dancetrack0007 | gate_search_late | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 317.44 |
| dancetrack0007 | gate_growth_unfrozen | none | 1 | 200 | 0.3147 | 0.3003 | 0.63500 | 3 | 0.3172 | 1 | 317.11 |
| dancetrack0007 | gate_growth_unfrozen | lower | 1 | 200 | 0.5431 | 0.4410 | 0.02500 | 5 | 0.9554 | 7 | 304.13 |
| dancetrack0007 | gate_growth_unfrozen | upper | 1 | 200 | 0.7817 | 0.5987 | 0.01000 | 1 | 0.9872 | 2 | 294.54 |
| dancetrack0007 | margin_0.00 | none | 1 | 200 | 0.3147 | 0.2953 | 0.64000 | 2 | 0.3155 | 1 | 300.44 |
| dancetrack0007 | margin_0.00 | lower | 1 | 200 | 0.6954 | 0.5806 | 0.03000 | 6 | 0.9580 | 6 | 303.81 |
| dancetrack0007 | margin_0.00 | upper | 1 | 200 | 0.7817 | 0.5963 | 0.01500 | 3 | 0.9809 | 3 | 311.58 |
| dancetrack0007 | margin_0.06 | none | 1 | 200 | 0.3147 | 0.3080 | 0.45500 | 5 | 0.3933 | 3 | 291.51 |
| dancetrack0007 | margin_0.06 | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 291.24 |
| dancetrack0007 | margin_0.06 | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 302.20 |
| dancetrack0007 | margin_0.12 | none | 1 | 200 | 0.3147 | 0.3109 | 0.58000 | 3 | 0.3371 | 1 | 300.86 |
| dancetrack0007 | margin_0.12 | lower | 1 | 200 | 0.7005 | 0.5810 | 0.00500 | 1 | 0.9928 | 2 | 298.18 |
| dancetrack0007 | margin_0.12 | upper | 1 | 200 | 0.7766 | 0.5938 | 0.00500 | 1 | 0.9935 | 1 | 297.52 |
| dancetrack0007 | parts_full_only | none | 1 | 200 | 0.3147 | 0.2980 | 0.63000 | 3 | 0.3189 | 1 | 235.58 |
| dancetrack0007 | parts_full_only | lower | 1 | 200 | 0.5431 | 0.4386 | 0.02000 | 4 | 0.9640 | 8 | 200.15 |
| dancetrack0007 | parts_full_only | upper | 1 | 200 | 0.7817 | 0.6029 | 0.01000 | 2 | 0.9872 | 1 | 236.53 |
| dancetrack0007 | parts_all | none | 1 | 200 | 0.3147 | 0.3080 | 0.45500 | 5 | 0.3933 | 3 | 279.06 |
| dancetrack0007 | parts_all | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 309.18 |
| dancetrack0007 | parts_all | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 332.08 |
| dancetrack0007 | part_matching_off | none | 1 | 200 | 0.3147 | 0.3080 | 0.45500 | 5 | 0.3933 | 3 | 299.35 |
| dancetrack0007 | part_matching_off | lower | 1 | 200 | 0.6091 | 0.5023 | 0.01500 | 3 | 0.9756 | 6 | 309.12 |
| dancetrack0007 | part_matching_off | upper | 1 | 200 | 0.7766 | 0.5958 | 0.01000 | 2 | 0.9871 | 2 | 290.57 |
