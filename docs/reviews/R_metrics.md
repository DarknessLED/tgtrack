# R_metrics — independent audit of the metric claims in `README.md` and `docs/EXPERIMENTS.md`

Audit date: 2026-10-25. Auditor: independent pass (no writes outside this file; no GPU
training/sweeps run). Workspace `C:\Projects\track`.

## Method

Every claim below was resolved against a machine-readable artifact (`results/**/summary.json`,
`summary.md`, `table.csv`, `per_run.json`, `metrics.json`), not against prose. Where a claim
needed arithmetic (pooled means, pp deltas) the arithmetic was recomputed from the artifact.
Where a claim needed a chronology judgement, the result directory's filesystem timestamps were
used ("epochs" below) and cross-checked against the ledger's own experiment order.

Re-run for the test-count claim:

```
python -m pytest -q -o addopts=""
471 passed, 3 skipped, 2 warnings in 274.94s (0:04:34)      # exit code 0
```

Epoch table used throughout (directory mtime = when the artifact was produced):

| artifact | produced | what it is |
|---|---|---|
| `results/partial_view_224`, `partial_view_448` | 2026-10-24 19:45 | embedder-resolution control |
| `results/target_obs19` | 2026-10-24 22:11 | 11-variant grid, MOT17-02 / target 19 / 300 f / sub 2 |
| `results/target_obs19_shipped` | 2026-10-25 01:44 | lone-bound re-run, same protocol |
| `results/mot_ignore4` | 2026-10-25 02:36 | MOT17 7-seq, shipped detector |
| `results/e031b_mot17` | 2026-10-25 07:15 | partial-weight grid (criterion 3) |
| `results/e041_mot17_upper` | 2026-10-25 09:08 | hard-negative grid, upper mask |
| `results/e051_ch_partial` | 2026-10-25 12:20 | CrowdHuman detector on the same benchmark |
| `results/e062` | 2026-10-25 14:40 | authoritative probe re-measurement (post-retraction) |
| `results/e063b_dt0004/0005/0007` | 2026-10-25 15:35 | DanceTrack audit guard table |

## Claim audit

| claim | source line | artifact | verdict | evidence |
|---|---|---|---|---|
| Test suite is green at the current size | `README.md:115` ("300+ tests") | live run | **OK** | 471 passed, 3 skipped, 274.94 s, exit 0. "300+" is true but understated by >150 tests. |
| "440 passed, 3 skipped" | `docs/EXPERIMENTS.md:741` | live run | **STALE** | Current suite is 471/3. The 440 figure is an epoch note with no date or commit, so it reads as a standing claim. |
| MOT17 all-person table (HOTA 0.4980 → 0.5050, DetA, AssA, MOTA 0.4950 → 0.5586, IDF1 0.6125 → 0.6531, ms/frame 49.68 → 41.88) and every delta | `README.md:593-602` | `results/mot_ignore4/summary.md`, `results/mot17_ch_det/summary.md` | **OK** | Both rows match the artifacts digit for digit, 7 sequences each; deltas recompute correctly (+0.70 HOTA points, +2.20 DetA, −0.90 AssA, +6.36 MOTA, +4.06 IDF1, −15.7 % ms/frame). `mot17_ch_det` pipeline line confirms `crowdhuman_yolov8n_best.pt`, `imgsz=1280`, `subsample=3`, `max_frames=200` — the protocol the README states. |
| Detector swap raises pooled MOT17 recall 0.731 → 0.803 (+7.2 pp) | `README.md:589-591` | `docs/EXPERIMENTS.md:1718-1719` (E054) | **OK** | E054 records 0.7306 → 0.8029; rounding and delta are correct. |
| All 19 shipped knob values (`imgsz 1280`, embedder `448`, `hard_negative_capacity 8`, `hard_negative_max_iou 0.30`, `motion_gate_adaptive False`, `distractor_margin 0.06`, `search_margin 0.20`, `anchor_adaptive True`/k 4.0/0.35/0.75/30, `partial_motion_weight 0.5`, `partial_appearance_weight 2.0`, `partial_fusion_shape_dev 0.45`, `lone_candidate_accept_cost None`, `stm_weight 0.6`, `ltm_weight 0.4`, `audit_interval 0`) | `docs/EXPERIMENTS.md` programme positions + `README.md` §5 | `src/tgtrack/config.py:58,114,264,276,343,357,372,385,475-479,506,507,512,514,518,519,557` | **OK** | 19/19 match. No ledger-vs-config contradiction found. |
| Ablation deltas in the MOT17-02 grid: gate legacy +13.5/+14.9 pp, `margin 0.00` inflates FAR on 2 masks, `margin 0.12` −2.9 pp TSR, `parts_full_only` −8.4/−14.9 pp, part-matching inert, hallucination 0.1467/0.3100/0.1433 | `README.md:406-459` | `results/target_obs19/table.csv`, `per_run.json` | **OK** | 33 runs, 0 errors. Baseline 0.9236/0.7527/0.9236; gate_legacy 0.7891/0.6036/0.9236; margin_0.00 FAR 0.01000/0.01000 vs 0.00667/0.00000; margin_0.12 upper 0.7236; parts_full_only 0.9200/0.6691/0.7745 with 8 vs 6 lower switches; `part_matching_off` and `parts_all` bit-identical. Hallucination rates come from `per_run.json` (44/93/43 frames), not from `table.csv` — but they are exact. |
| The grid's `baseline` row is "shipped defaults" (`0.9236 / 0.7527 / 0.9236`) | `README.md:406-412`, `README.md:466` | `results/target_obs19` (22:11) vs `target_obs19_shipped` (01:44) vs `e031b_mot17` (07:15) vs `e041_mot17_upper` (09:08) | **STALE** | Under the *same* command (MOT17-02, target 19, 300 f, sub 2, `mask=upper`) four runs labelled baseline/shipped measure **0.7527, 0.7164, 0.7818 and 0.8545** — a 14 pp spread with no epoch label anywhere in the README. The grid predates the E031b shipping of the partial weights (07:15) by 9 hours, so its baseline is not today's default. |
| `lone_candidate_accept_cost` costs 3.6 pp of upper-mask TSR (0.7527 → 0.7164) on real MOT17 and 0.0 pp elsewhere; therefore shipped off | `README.md:676-679` | `results/target_obs19`, `results/target_obs19_shipped` | **UNSUPPORTED** | No artifact contains a controlled A/B of the bound on MOT17: `target_obs19`'s ablate groups are `gate,distractor_margin,target_parts,part_matching` (no lone group) and `target_obs19_shipped` contains only `baseline` and `part_matching_off` — both at the default `None`. 0.7164 is that directory's *baseline* upper value, and 0.7527 is a different run 3.5 h earlier. The 3.6 pp "cost" is the difference between two baselines, not a bound effect. The decision (keep `None`) may still be right, but the number does not measure what it says. |
| Lone-bound sweep: 100 runs / 10 scenarios × 10 cells, peak false acquisition 76.9 % → 75.6 %, clean-family cost 2.6 pp at 0.5 and 5.4 pp at 0.4 | `README.md:669-673` | `results/motion_ablation/lone_bound.md` | **OK** | Header says "runs: 100 (10 scenarios x 10 cells), errors: 0"; `lnone` peak 76.9 vs `l020..l050` 75.6; clean families 98.9 → 96.3 (−2.6) and 93.5 (−5.4). |
| "hijack count goes 3/4 runs → 1/4 at 0.5 and **→ 0/4 at ≤ 0.4**" | `README.md:670` | `results/motion_ablation/lone_bound.md` | **WRONG** | At `l020`–`l040` the count is **1/4** in `mode=missing` (`n_dets>=2: 1`) and **3/4** in `mode=orthogonal`. Only the `n_dets=1` sub-count is 0. The "0/4" overstates the fix and contradicts `README.md:25(c)`, which says "3/4 → 1/4". |
| Unidentifiable-target pooled result: 39.2 % success / 54.6 % false acquisition with bystander (30 runs) vs 58.7 % / 0.0 % solo (10 runs) | `README.md:662` | `results/motion_ablation/table.md:267-268`, `readme.md:96-97` | **OK** | Exact, including the run counts. |
| Demo click, MOT17-02 id 9 is 36 × 110 px, TSR 0.06, `wrong_box_rate` 0.843, `false_acquisition_rate_locked` 0.357, `id_consistency` 0.419, 22 switches, hallucination 0.841 | `README.md:686-696` | `results/demo_real_id9/metrics.json`, MOT17-02 `gt/gt.txt` | **OK** | Frame-1 GT row is `1,9,1056,484,36,110` exactly; metrics 0.06 / 0.84333 / 0.35714 / 0.41860 / 22 / 0.84133. The `botsort.yaml` in that summary is the demo script's own default, not the shipped MOT tracker, but the README does not claim otherwise there. |
| Partial-view retrieval: every per-mask cell (7 rows × 5 strategies) and both margin tables | `README.md:344-352`, `359-362` | `results/partial_view_real/summary.json` | **OK** | 25 identities / 414 queries; all 35 cells match to 3 dp; upper60 margins −0.0044/+0.0032/+0.0302/+0.0383/+0.0390 and lower +0.0762/+0.1358/+0.1573/+0.1664/+0.1592 are exact. |
| Aggregate "Mean identity Rank-1 **over the seven occluded masks**": 0.672 / 0.803 / 0.809 / 0.840 / 0.855 and "+27 %" | `README.md:332-340`, `README.md:20` | `results/partial_view_real/summary.json` | **OK** (wording defect) | The five values are exactly the means over the **six** occluded masks (0.6719 / 0.8027 / 0.8088 / 0.8398 / 0.8555); the seventh row, `none`, is the clean control and is excluded from the mean. 0.855/0.672 = 1.272 is consistent. The label "seven occluded masks" is off by one. |
| 448-not-224: "clean 0.886 → **1.000**, occluded **0.780 → 0.989**" | `README.md:570-574` | `results/partial_view_224/summary.json`, `results/partial_view_448/summary.json` | **WRONG** | Recomputing the six-mask means: `full` goes **0.6553 → 0.7803**, `quality_w` **0.9205 → 0.9848**, `mean_parts`/`visible_gate` **0.9508/0.9205 → 0.9886/0.9886**. So 0.780 is the *post-fix* whole-body value and 0.989 is the *post-fix* part-fused value: the sentence splices a strategy axis into a resolution axis, implying the 448 fix bought 0.78 → 0.989 when the same quantity moved 0.655 → 0.780. `src/tgtrack/config.py:109-110` states the correct pair ("224 -> clean 0.886, occluded 0.655; 448 -> clean 1.000, occluded 0.780"), which is how the README line should read. |
| VisDrone: tracktrack HOTA 0.3800 vs bytetrack 0.3009, DetA 0.306, AssA 0.476, ~4× fewer ID switches, and 0.0000 on the two 4K sequences (3840×2160, 1904×1071) | `README.md:320-321`, `README.md:128` | `results/visdrone_mot/summary.md`, source frames | **OK** | 7 sequences, 14 pairs, 0 failed; 0.3800/0.3055/0.4761 and IDSW 152 vs 604. The two zero-HOTA sequences are `uav0000268_05773_v` (3840×2160) and `uav0000305_00000_v` (1904×1071), measured from the decoded first frame. |
| Criterion ② "asked for TSR ≥ 0.55 with FAR < 0.20. We reached **0.512 / 0.280**" | `README.md:612`; repeated at `docs/EXPERIMENTS.md:1467,1504,1543,1690,1744,1818,1870` | `results/e041_dancetrack0004/0005/0007/table.csv`; `results/e063b_dt0005/table.csv` | **STALE** | 0.512 = mean(0.4150, 0.6950, 0.4264) and 0.280 = mean(0.040, 0.205, 0.595) **at E041**, where dancetrack0005 was run with `target_ids={'dancetrack0005': 2}` and `target_reason=forced`. The same command today selects target **0** (`target_reason=observable`) and the current shipped baseline measures 0004 0.4150/0.0400, 0005 **0.4000/0.5150**, 0007 0.4264/0.5950 → pooled **0.4138 / 0.3833**, i.e. 9.8 pp TSR better than the programme start and 9.8 pp *worse* than the quoted figure. The E041 number is not reproducible with the current harness and is never labelled as an E041-epoch, different-target measurement. |
| "Starting point at programme outset was TSR 0.2228 / FAR 0.4933, so this is **+28.9 pp TSR and −21.3 pp FAR**" | `docs/EXPERIMENTS.md:1401-1403` | `results/dancetrack_target/table.csv`, `results/e041_*`, `results/e063b_*` | **WRONG** | 0.2228/0.4933 come from the 99-run grid (2026-10-24, targets 3/0/1, "observable"); 0.512/0.280 come from E041 (target **2** on 0005). Different target *and* different epoch, so the subtraction mixes both axes. The honest same-target, latest-epoch comparison is 0.2228 → 0.4138 (+19.1 pp) with FAR 0.4933 → 0.3833. |
| "Programme position after E060/E062: criterion 2 **0.512 / 0.280 pooled** but 0007 alone is 0.426 / 0.595" | `docs/EXPERIMENTS.md:1870-1871` (also `1818`, `1744`, `1690`) | `results/e063b_dt0004/0005/0007/table.csv` | **STALE** | This sentence sits in the same ledger as E063b's own guard table (`:1940-1945`), whose shipped rows (0.4150/0.0400, 0.4000/0.5150, 0.4264/0.5950) pool to 0.4138/0.3833. The ledger therefore asserts 0.512 at line 1870 and publishes the data that refutes it at line 1943, 70 lines later, with no reconciliation note. |
| Criterion ③ **MET**: partial-view upper-mask TSR 0.7527 → **0.8545 (+10.18 pp)** | `docs/EXPERIMENTS.md:1020,1051,1139`, `:1871` "criterion 3 MET 0.8545"; `README.md:612` region | `results/e031b_mot17/table.csv`, `results/e041_mot17_upper/table.csv` | **WRONG** (attribution) | The 0.8545 endpoint is real and *latest-epoch*: `e041_mot17_upper` (09:08) baseline upper = 0.8545, FAR 0.00000, idcons 1.0000. But `e031b_mot17` contains the controlled same-run pair: baseline **0.7818** vs `partial_m0.5_a2.0` **0.8545** → the mechanism's real gain is **+7.27 pp**, not +10.18. The +10.18 comes from subtracting the *programme-start* baseline (0.7527, measured at an earlier epoch with different shipped weights) from a later run. The criterion "≥ +10 pp over 0.7527" is therefore met against an epoch-mismatched baseline; against the run's own control it is 7.3 pp and the criterion would read NOT MET. |
| `yolo26m (shipped)` lower-mask TSR = **0.8545** | `docs/EXPERIMENTS.md:1650` | `results/e051_ch_partial/table.csv` (CH column only), `results/target_obs19/table.csv` | **WRONG** | 0.8545 on the lower mask appears in no artifact. The same-protocol shipped runs measure 0.9236 (`target_obs19`, `target_obs19_shipped`). The cell duplicates the upper-mask value, which is also the only cell in that table whose "shipped" side has no artifact in `e051_ch_partial`. |
| Criterion ① "0.4827 baseline, 0.4980 after the ignore fix" | `docs/EXPERIMENTS.md:1630` | `results/mot_ignore`, `mot_ignore2`, `mot_ignore3`, `mot_ignore4` | **UNSUPPORTED** | 0.4827 appears in no result artifact; the three pre-fix runs all read **0.4856** and the post-fix run 0.4980. Either the pre-fix number is stale/typo'd or the run that produced it was never kept. |
| E062 retraction of E020 / E060 after the `gt_at()` bug | `docs/EXPERIMENTS.md:1873-1882` | `results/e062/summary.json`, `results/e061_probe7/trace.json` | **OK** (honest) | The retraction names the defect precisely (`seq.gt_at(i)[0]` returns an arbitrary annotated box, not the target), names both affected claims, keeps the text instead of deleting it, and re-measures with the authoritative resolver: `frames_scored 197`, `tsr 0.4264`, `wrong_frames 113`, `wrong_frames_true_target_detected 108`, `iou≥0.3 112`, `det_hit_rate_all_frames 0.9695`, `wrong_det_iou_p50 0.9336`, area ratios 0.766/1.022 — every number matches the ledger table, and 0.4264 is bit-identical to the harness baseline in `e063b_dt0007`. The superseded probe trace (`e061_probe7`, 199 rows, TSR 0.3015) is retained separately and is not cited as evidence. This is a model retraction. |
| The retracted E020 result is still quoted as live evidence: "appearance (true target ranked ≥ 2 in **74/111** wrong frames, `results/e020_*`)" | `README.md:614` | no artifact; `results/e020_*` does not exist | **WRONG** | The 74/111 histogram is one of the two claims E062 retracts by name. The README — the deliverable, written after the retraction — reproduces it as the measured ceiling of criterion ②, and cites a result directory that was never kept (the tree has `e019_*` and `e021_*`, no `e020_*`). The ledger also repeats 74/111 at `:1113`, `:1496`, `:1532` and `:1543` (all before E062, but none annotated as superseded), and E020's own section at `:609` carries no retraction banner, so a reader who arrives there first sees a clean claim. |
| E060's 113-wrong-frame table survives the retraction | `docs/EXPERIMENTS.md:1852-1867` | `results/e062/summary.json` | **OK** | `wrong_frames 113`, all in state `tracking` — independently reproduced by the authoritative probe. Only E060's "target never detected after f70" premise is retracted, and that is stated. |

## Verdict counts

| verdict | count |
|---|---|
| OK | 12 |
| STALE | 4 |
| WRONG | 6 |
| UNSUPPORTED | 2 |
| **total claims audited** | **24** |

## What must be fixed, ranked by severity

1. **`README.md:676-679` quotes a controlled A/B that does not exist.** The "3.6 pp cost of the
   lone bound (0.7527 → 0.7164)" compares two *baseline* runs from different directories 3.5 hours
   apart; neither directory contains an armed variant of `lone_candidate_accept_cost`. Either run
   the A/B (bound `None` vs `0.5`, one grid, same epoch) or restate the claim as "two same-protocol
   baseline runs 3.5 h apart differ by 3.6 pp, which is the size of the effect we are trying to
   measure — hence the shipped-off decision rests on the synthetic sweep, not on MOT17."
2. **Criterion ②'s headline number (0.512 / 0.280) is not reproducible.** It depends on
   `dancetrack0005` target id **2** (E041, `target_reason=forced`); the current harness selects id
   **0** on the same command and measures 0.4000 / 0.5150, pooling to 0.4138 / 0.3833. The
   "+28.9 pp / −21.3 pp" arithmetic (`EXPERIMENTS.md:1401-1403`) then subtracts across both the
   target-id axis and the epoch axis. This is the single largest numeric overstatement in the
   repo: quote 0.4138 / 0.3833 as current, keep 0.512 / 0.280 as "E041, forced target 2", and note
   that `--target-strategy` did not resolve to the same person in both epochs.
3. **Criterion ③ is "MET" only against an epoch-mismatched baseline.** The controlled pair inside
   `results/e031b_mot17` is 0.7818 → 0.8545 = **+7.27 pp**, which misses the stated ≥ +10 pp
   requirement. The 0.8545 endpoint itself is sound (`e041_mot17_upper`, latest epoch); the "+10.18
   pp" and the MET label are not. Re-state as "+7.3 pp against the run's own control, +10.2 pp
   against the programme-start 0.7527 which was measured with pre-E031b defaults", and decide
   explicitly which of the two the criterion meant.
4. **`README.md:614` still sells a retracted number.** "74/111" is retracted by E062 and the cited
   `results/e020_*` does not exist. Delete the parenthetical or replace it with an E062-consistent
   statement (detector sees the true target in 108/113 wrong frames at median IoU 0.93 — which
   supports the same argument *better*). Also add a one-line RETRACTED banner at `EXPERIMENTS.md:609`
   (E020) and at `:1113/:1496/:1532` so the ledger cannot be read selectively.
5. **Four different values are labelled "shipped defaults" for the same benchmark**
   (0.7527 / 0.7164 / 0.7818 / 0.8545 upper-mask TSR on MOT17-02 target 19, 300 f, sub 2). The
   README's §5 grid is a 10-24 22:11 artifact that predates the E031b shipping of the partial
   weights; its internal deltas are self-consistent but its absolute baseline is stale. Either
   re-run the grid at the current epoch or label every row with its date.
6. **`README.md:570-574` conflates the resolution axis with the strategy axis.** Write
   "whole-body 0.655 → 0.780, structure-weighted 0.921 → 0.985 (six-mask means)" — the 0.886 → 1.000
   clean-crop pair is correct as written.
7. **`EXPERIMENTS.md:1650` lower-mask cell (0.8545) is a duplicate of the upper cell** and is
   contradicted by every lower-mask artifact (0.9236). Fix to 0.9236 or delete the row.
8. **`README.md:670` "→ 0/4 at ≤ 0.4"** contradicts both the sweep (1/4, and 3/4 in
   `mode=orthogonal`) and `README.md:25(c)` ("3/4 → 1/4"). Fix to 1/4 and note the `n_dets=1`
   sub-count is 0/4.
9. **`README.md:332` "seven occluded masks"** should be "six occluded masks (the seventh row is the
   clean control)". Numbers are right; the label is not.
10. Minor: `EXPERIMENTS.md:741` "440 passed, 3 skipped" (now 471/3) and `:1630` "0.4827 baseline"
    (artifacts say 0.4856) are stale/unsupported epoch notes; `README.md:115` "300+ tests" is true
    but 171 short.

## Not verified (stated, not hidden)

- GPU-side throughput claims (`ms/frame` values are taken from the artifacts as recorded, not
  re-timed) — re-timing would change hardware, and the audit was not allowed to run GPU work.
- `README.md:694-695` TSR 0.964 for id 2 and the 0.924 for id 19: the id-19 figure is consistent
  with `target_obs19` baseline (`none` 0.9236) but the id-2 run's artifact was not located in this
  pass, so it is neither confirmed nor disputed here.
- The E058 bootstrap and E059 trail effect sizes were not re-derived; they were outside the
  claim budget and are internally consistent with their own tables.
