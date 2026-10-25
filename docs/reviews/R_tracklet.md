# R_tracklet — is the all-person MOT layer an external re-acquisition channel?

**Question (E065 hypothesis, from `docs/reviews/README.md` F-03).** When the single-target layer is
wrong, does the all-person MOT layer still hold a tracklet on the same person, and for how long? Yes
would make the MOT tracklet id a free, external identity signal (computed without the tracked
person's own recent appearance, so immune to the contamination documented in `docs/METHOD.md` §8).
No kills the direction cheaply.

**Verdict: no — not for the losses that matter.** The MOT layer *does* keep boxes on the target
(83.6 % of measurable loss frames at IoU ≥ 0.5, 84.4 % with appearance on), but the tracklet **id**
fragments exactly during the long losses: only **6 of 14** loss runs of ≥ 5 frames are anchored
end-to-end by one id (42.9 %; 28.6 % fully), and those anchored runs hold **108 of 1113** frames
(9.7 %). Every one of the five ≥ 50-frame runs is unanchored (single-id fraction 0.21–0.64). The
channel is stable precisely where the target layer already succeeds (median single-id stretch 103–150
frames on the two TSR > 0.85 sequences, 2–4 frames on the three catastrophic ones). Useful as a
re-acquisition anchor: **not usable as shipped**.

## Measurement

Artifacts (everything below is read from these files):
`results/tracklet_evidence/run_evidence.py` (runner), `analyze_evidence.py` (analysis),
`runs/run_manifest.json` (commands + provenance), `runs/<seq>/run_status.json`,
`runs/<seq>/{target,mot,mot_reid}.log`, `analysis_reid.json` (primary, appearance on) and
`analysis.json` (appearance off).

Six sequences from `results/heldout_dancetrack/report.json`, target ids pinned to the ones the
held-out run used (the runner refuses to start if they disagree): the three worst
`dancetrack0090` (tid 12, TSR 0.004), `dancetrack0025` (tid 6, 0.015), `dancetrack0035` (tid 5, 0.022),
plus all three sequences above TSR 0.30 — `dancetrack0030` (tid 0, 0.370), `dancetrack0094` (tid 0,
0.880), `dancetrack0079` (tid 2, 0.913). 300 frames each (`--subsample 2 --max-frames 300`, the
held-out protocol), 1800 frames total.

Per sequence, two harness runs, one GPU process at a time:
1. **target layer** `scripts/run_target.py --debug-csv`, variant pinned to `baseline` only. The CLI
   cannot express that (`--ablate` defaults to `"all"`, 84 variants), so the runner re-enters itself
   as a child process that replaces `run_target.build_variants` with a baseline-only builder and then
   asserts `meta.variants == ["baseline"]` on the written summary — the `scripts/heldout_report.py`
   pattern, METHOD §9. No file under `src/`, `scripts/`, `tests/` was edited.
2. **all-person layer** `scripts/eval_mot.py --trackers botsort.yaml --save-tracks` (one
   comma-separated `--trackers` value), the shipped `MotCfg` tracker (`botsort.yaml`, `with_reid=True`).

### Frame alignment — verified, not assumed
Both harnesses call `tgtrack.data.load_mot_sequence` with identical `subsample=2, max_frames=300`
(parameter provenance in `runs/run_manifest.json`), so original frame `o` is kept iff `o % 2 == 0` and
renumbered to `o // 2`; `eval_mot.write_mot_tracks` writes `index + 1`, so `index = file_frame - 1`.
Four checks per sequence in `analysis*.json → alignment`, all recorded, all passing:

- **A1** the debug CSV covers exactly frames `0..299` (no holes, no offset).
- **A2** its `gt_xyxy` column is bit-comparable (atol 0.6 px) to
  `make_target_gt(load_mot_sequence(seq, subsample=2, max_frames=300), target_id)` for all 300 frames,
  **0 mismatches in 6 × 300 frames** — the CSV's frame axis and the loader parameters are proven, not
  inferred. (GT never comes from `seq.gt_at(i)[0]`.)
- **A3** the MOT track file spans indices `0..299` and `num_frames == 300` in both harnesses.
- **A4** lag test: over loss frames with a visible target, the median best IoU against MOT boxes is
  highest at offset 0 (e.g. 0030 0.820 vs 0.745/0.746; 0035 0.809 vs 0.766/0.728). One case is a tie
  (0090 with appearance: 0.637 vs 0.638 at −1, a near-static target); it is reported as a tie, with a
  0.02 IoU tolerance, and the strict comparison is kept in the file.

## Results — shipped configuration (`botsort.yaml`, `with_reid=True`)

Per-sequence, appearance on (`analysis_reid.json`). TSR is recomputed from the debug CSV; loss frames
are `correct == False`; "measurable" = the target is visible in GT.

| sequence | tid | TSR | MOT HOTA | loss frames (meas / all) | covered | cov % | cov/all % | runs | runs ≥5 | anchored ≥80 % | fully anchored | median anchored run | median id stretch | max id stretch | id switches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dancetrack0090 | 12 | 0.003 | 0.458 | 233 / 299 | 162 | 69.5 | 54.2 | 2 | 1 | 0 | 0 | – | 2.0 | 20 | 39 |
| dancetrack0025 | 6 | 0.013 | 0.744 | 262 / 296 | 215 | 82.1 | 72.6 | 3 | 3 | 2 | 1 | 9.5 | 3 | 57 | 22 |
| dancetrack0035 | 5 | 0.017 | 0.346 | 223 / 295 | 211 | 94.6 | 71.5 | 6 | 5 | 1 | 0 | 54 | 2.5 | 39 | 33 |
| dancetrack0030 | 0 | 0.370 | 0.674 | 189 / 189 | 174 | 92.1 | 92.1 | 4 | 2 | 1 | 1 | 5 | 4.5 | 112 | 21 |
| dancetrack0094 | 0 | 0.880 | 0.506 | 36 / 36 | 36 | 100.0 | 100.0 | 5 | 2 | 2 | 2 | 15.0 | 103 | 195 | 2 |
| dancetrack0079 | 2 | 0.913 | 0.506 | 26 / 26 | 20 | 76.9 | 76.9 | 5 | 1 | 0 | 0 | – | 33 | 257 | 2 |
| **pooled** | | | | **969 / 1141** | **938** | **84.4** | **71.7** | **25** | **14** | **6 (42.9 %)** | **4 (28.6 %)** | **9.5** | | | **119** |

`cov/all %` counts the 172 loss frames where GT has no target box (15 %) as uncovered; they are not
measurable at all, which is why both numbers are given. Anchoring per run uses the all-frames
denominator; with a GT-visible denominator the pooled figure is 6/14 (42.9 %).

The appearance-off pass (`analysis.json`) is within noise of this: coverage 0.8359 vs 0.8442,
6/14 anchored both ways, median anchored run 9.5 both ways, fully anchored 4 both ways. The
conclusion is therefore not an artifact of how the MOT layer was configured.

### The ≥5-frame runs, one by one (appearance on)

| sequence | run (frames) | len | target visible | anchor id | hits | frac | anchored |
|---|---|---|---|---|---|---|---|
| dancetrack0030 | 75–79 | 5 | 5 | 5 | 5 | 1.00 | yes |
| dancetrack0030 | 121–299 | 179 | 179 | 6 | 115 | 0.64 | no |
| dancetrack0079 | 40–53 | 14 | 14 | 4 | 11 | 0.79 | no |
| dancetrack0094 | 34–57 | 24 | 24 | 7 | 24 | 1.00 | yes |
| dancetrack0094 | 197–202 | 6 | 6 | 7 | 6 | 1.00 | yes |
| dancetrack0025 | 2–12 | 11 | 11 | 9 | 10 | 0.91 | yes |
| dancetrack0025 | 14–21 | 8 | 8 | 9 | 8 | 1.00 | yes |
| dancetrack0025 | 23–299 | 277 | 243 | 55 | 78 | 0.28 | no |
| dancetrack0035 | 4–20 | 17 | 4 | 7 | 3 | 0.18 | no |
| dancetrack0035 | 22–61 | 40 | 15 | 26 | 11 | 0.28 | no |
| dancetrack0035 | 63–116 | 54 | 53 | 7 | 46 | 0.85 | yes |
| dancetrack0035 | 118–266 | 149 | 128 | 41 | 54 | 0.36 | no |
| dancetrack0035 | 268–299 | 32 | 23 | 30 | 13 | 0.41 | no |
| dancetrack0090 | 3–299 | 297 | 233 | 142 | 64 | 0.21 | no |

Run-length distribution over all 25 runs: 1 frame ×3, 2–4 ×8, 5–9 ×3, 10–19 ×3, 20–49 ×3, ≥50 ×5.
Anchoring is essentially a function of run length: **short runs are anchored, long runs are not.**
The five ≥ 50-frame runs (149, 179, 277, 297, plus 32/149/268 on 0035's tail) contain 88 % of all
long-run frames and contribute **0** anchored animations beyond 0035's 54-frame run.

### Identity stability, independently of the losses

Greedy continuation (keep the previous id while it still reaches IoU ≥ 0.5) over all GT-visible
frames, so a tie between two overlapping qualifying boxes cannot fake a switch:

| sequence | TSR | median single-id stretch | max stretch | id switches |
|---|---|---|---|---|
| dancetrack0094 | 0.880 | 103 | 195 (frames 0–194) | 2 |
| dancetrack0079 | 0.913 | 33 | 257 (frames 43–299) | 2 |
| dancetrack0030 | 0.370 | 4.5 | 112 (frames 0–111) | 21 |
| dancetrack0025 | 0.013 | 3 | 57 | 22 |
| dancetrack0035 | 0.017 | 2.5 | 39 | 33 |
| dancetrack0090 | 0.003 | 2.0 | 20 | 39 |

The correlation runs the wrong way for a rescue channel: the MOT layer's identity is durable exactly
where the target layer is already right, and churns every 2–4 frames where it is not.

## What this means

- The MOT layer is **not** blind to the target — a well-fitting box exists on 84 % of loss frames,
  median best IoU 0.64–0.97. The information bottleneck is identity, not geometry.
- As a re-acquisition anchor the id is the thing you need, and the id is what fails. 42.9 % of
  ≥5-frame runs and 9.7 % of their frames are single-id anchored, with a median anchored run of 9.5
  frames (19 source frames) — i.e. the channel buys a re-attach window only for losses that were
  already short.
- Consuming those boxes anyway would require an appearance model to bind them to the target. Binding
  against the target layer's own memory is exactly what METHOD §8 rules out; binding against the MOT
  track's own accumulated crops is a *different* experiment (an external gallery), not this one.
- Cost of the negative: 17.5 min of GPU. Worth it — the direction is now closed rather than queued.

## Limits (stated, not hidden)

1. Six sequences, 300 frames each, one MOT backend (`botsort.yaml`), one detector configuration.
   DanceTrack crowds are extreme for an IoU/motion tracker, so the churn is a property of *this* MOT
   layer, not of all-person tracking in general.
2. 15 % of loss frames have no target annotation at all; they are unmatchable and are counted as
   uncovered in `cov/all %`, excluded from coverage and from the anchor denominator.
3. `anchor_frac` in the run table uses the all-frames denominator, so a run spanning unannotated
   frames is penalised; the GT-visible variant is in `analysis*.json → runs.table →
   anchor_frac_visible` and changes the pooled count only from 6/14 to 7/14 in the appearance-off pass.
4. A4's lag test is a redundant sanity check on top of the exact A2/A3 checks; the 0090 tie is
   reported rather than suppressed.
5. `eval_mot.py` defaults to `--no-embed`, i.e. BoT-SORT without appearance, while the shipped
   `MotCfg.with_reid` is `True`. Both configurations were measured; they agree to within 2.6 pp of
   coverage, so the verdict does not depend on this choice.

## Traps hit (all cost real time)

1. `--ablate` defaults to `"all"` and `build_variants("")` also means `"all"` — baseline-only needs
   the child-process pin, and the assertion must be on the written `summary.json`, not on the argv.
2. `n_report_frames` (298 for 0090) is the count of frames the *target layer reported on*, not the
   frame axis; comparing it against `eval_mot`'s `num_frames` (300) produced a false alignment alarm.
   The frame axis is `n_frames`.
3. `eval_mot.py`'s default is the appearance-off path; the first pass silently measured a weaker MOT
   layer than the shipped configuration. Caught by reading `config.with_reid` in the pair JSON.
4. Per-frame argmax for "which id is the target" flips between two qualifying overlapping boxes and
   inflates id switches (129 → 108 after greedy continuation). Any tracklet-continuity number is
   meaningless without this fix.
5. `--trackers a.yaml b.yaml` fails: it is one comma-separated value.

## Actions

1. **Close E065 as a re-acquisition channel** and record this negative in `docs/EXPERIMENTS.md`; do
   not spend a round on tracklet-id audits of the shipped MOT layer.
2. If the direction is ever revisited, the admissible variant is an *external gallery* built from the
   MOT track's own crops (long-stretch tracks exist: 195 and 257 frames on the two good sequences),
   which must be measured as a new experiment, not as a re-use of this one.
3. Keep `results/tracklet_evidence/` as the reproducible harness: `run_evidence.py` (with the
   baseline pin and reuse), `analyze_evidence.py --mot-variant mot|mot_reid`, and the alignment block
   as the admission gate for any future tracklet claim.
