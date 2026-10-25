# DanceTrack — all-person MOT smoke (step 1–2)

What was run (exactly this command):

```
python scripts/eval_mot.py --data-root data/raw/dancetrack/val \
  --trackers botsort.yaml,ocsort.yaml,deepocsort.yaml,fasttrack.yaml,tracktrack.yaml,bytetrack.yaml \
  --limit-sequences 2 --max-frames 150 --subsample 2 --out results/dancetrack_mot --jobs 1
```

Scale: 2 sequences (`dancetrack0004`, `dancetrack0005`), 150 original frames each
(subsampled by 2 → 150 kept frames per sequence, 300 frames total), 1183 GT boxes,
detector `yolo26n` at the project default, no ReID embedding (`--no-embed`), one
worker so the shared GPU is not oversubscribed.

## Loader validation (step 1)

`tgtrack.data.load_mot_sequence` handles DanceTrack **with its default columns**
(`conf_col=6, class_col=7, visibility_col=8`) — no adjustment needed. Verified:

- 25 sequences discovered under `data/raw/dancetrack/val`; each has `img1/`, `gt/gt.txt`, `seqinfo.ini`.
- `dancetrack0004`: `seqLength=1203`, `imWidth=1920`, `imHeight=1080`, `frameRate=20`, 4586 GT rows, 4 track ids (ids run 0…3), all rows `conf=1.0`, `cls=1`, `visibility=1.0`.
- `dancetrack0005`: same shape — 4739 GT rows, 4 ids (0…3), `conf/cls/visibility` all constant.
- `seq.frame(0)` decodes: `uint8` `(1080, 1920, 3)` for both sequences.
- `sequence_summary` on a 20-frame subsampled slice: 74 (0004) / 78 (0005) GT boxes, 4 ids each.

Two DanceTrack-specific notes that matter for later steps:

1. `visibility` is a constant `1.0`, so `--min-visible` cannot filter anything (as warned).
2. Track ids start at **0**, and these two sequences contain only 4 identities each; ids are
   per-sequence, so they are not comparable across sequences.
3. `discover_sequences(root)` returns paths that **already include the root** — pass them
   straight to `load_mot_sequence(dir)`; do not join with the root again.

## Results (pooled over the 2 sequences, sorted by HOTA)

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | ms/frame |
|---|---|---|---|---|---|---|---|---|
| tracktrack | **0.6539** | 0.7324 | **0.5866** | 0.8216 | **0.6711** | 21 | 36 | 34.5 |
| deepocsort | 0.6062 | 0.7336 | 0.5042 | 0.8284 | 0.6064 | 23 | 34 | 33.0 |
| botsort | 0.5871 | 0.7435 | 0.4656 | 0.8326 | 0.6165 | 23 | 34 | 35.4 |
| fasttrack | 0.5713 | 0.7427 | 0.4411 | 0.8216 | 0.6202 | 29 | 38 | 21.5 |
| bytetrack | 0.5682 | 0.7375 | 0.4395 | 0.8259 | 0.6057 | 22 | 36 | 21.5 |
| ocsort | 0.5445 | 0.7407 | 0.4021 | 0.8250 | 0.5521 | 29 | 35 | 21.4 |

## What the numbers mean

- **Detection is not the variable.** `DetA` spans 0.7324–0.7435 (1.1 pp) and `MOTA`
  0.8216–0.8326, i.e. every tracker sees the same detections; all the spread is association.
- **Association is the variable.** `AssA` spans 0.4021–0.5866 (**18.4 pp**) and HOTA
  0.5445–0.6539 (**10.9 pp**). The ranking is almost exactly the association ranking, which is
  the expected DanceTrack signature: uniform appearance (so ReID cannot separate people) plus
  nonlinear, non-gridded dancer motion puts the load on the motion model.
- **The motion-centric tracker wins.** `tracktrack` is first on HOTA, AssA and IDF1, +10.9 pp
  HOTA over `ocsort` and +18.4 pp `AssA`. On MOT17 the same comparison separated much less, so
  the gap widens on DanceTrack as predicted — this is a smoke result (2 sequences, 300 frames)
  and the full 25-sequence campaign (task-9 step 3, `--jobs 2`, with and without `--embed`) is
  what turns it into a defensible number.
- `deepocsort` is second and is the strongest single-sequence result (`dancetrack0005` HOTA
  0.7703 vs `tracktrack` 0.7700), so the 2-sequence ordering is not yet stable at the top.

## Files

- `summary.csv` / `summary.md` — the pooled and per-(sequence, tracker) tables.
- `pairs/<pair>.json` — per-(sequence, tracker) metrics and full TrackEval result dict.
- `run.json` — the run plan, timings and the same summary rows.
- `plan.json` — the 12 (sequence × tracker) pairs.
