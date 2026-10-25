# R_c1 — Is "appearance does not buy association" an artefact of an inert switch?

**Status:** DONE (cheap falsification run, 1 GPU, `--jobs 1`)
**Date:** 2026-10-25
**Reviewer hypothesis (C1):** the ledger's conclusion *"appearance does not buy MOT association"*
(E047/E048/E053) is an artefact — `results/` runs used `--no-embed`, so `scripts/eval_mot.py:598`
never built an embedder, and `with_reid: True` alone installs ultralytics' `_auto_encoder`
pass-through that receives `feats=None` and therefore never engages. E053's tracktrack+reid row
being bit-identical to tracktrack is the tell.

**Answer:** the *mechanism* part of the hypothesis is **CONFIRMED** (the flag was inert, and I can
show exactly where). The *consequence* part — that engaging appearance would move AssA by
> +0.5 pp — is **REJECTED**: AssA moved **−0.068 pp** (2 sequences, 400 frames).

---

## 1. Code path (read-only, `scripts/eval_mot.py`)

| line | what it does |
|---|---|
| `scripts/eval_mot.py:1091-1092` | `--embed` / `--no-embed` (dest `embed`, **default False** → `--no-embed`) |
| `scripts/eval_mot.py:313` | `cfg.mot.with_reid = bool(opts.with_reid or opts.embed)` |
| `scripts/eval_mot.py:598` | `embedder = emb_factory(cfg, opts) if opts.embed else None` ← **the inert switch** |
| `scripts/eval_mot.py:628-631` | only when `embedder is not None`: `embedder.encode(...)` → `feats_from_embeddings(...)` |
| `scripts/eval_mot.py:635` | `tracker.update(img, dets, feats=feats)` |
| `scripts/eval_mot.py:696` | counter `num_missing_features` (rows the embedder could not fill) |
| `src/tgtrack/mot_layer.py:307-320, 368-370` | `_resolve_feats` → `self._tracker.update(container, img_bgr, feats=per_det_feats)`; `None` when no feats |
| `ultralytics/trackers/utils/reid.py:150-159` (8.4.185) | `build_encoder(with_reid, "auto")` → `_auto_encoder(feats, _dets)` pass-through, `None` when `with_reid` False |
| `ultralytics/trackers/track_tracker.py:491-504` | `feats = kwargs.get("feats")`; `if use_native: encoder_input = feats[high_mask] if feats is not None and len(feats) else None` → **`None` ⇒ `dets_high` built without `feat` ⇒ no cosine term at all** |

So there are two independent ways to get a vacuous "reid" run: `--no-embed` (no embedder at all),
and `--with-reid` without `--embed` (encoder installed, zero feature rows). E053 used the second
for `trk_reid.yaml`; E047/E048 used the first.

## 2. Exact commands

All three runs: MOT17 `train`, `--subsample 3 --max-frames 200 --ignore-regions --jobs 1`,
tracker `tracktrack.yaml`, detector `data/weights/crowdhuman_yolov8n_best.pt` @ `imgsz 1280`
(= E047's configuration), `--limit-sequences 2` → **MOT17-02-FRCNN, MOT17-04-FRCNN** (the first two
of `discover_sequences`; not cherry-picked).

```powershell
$env:PYTHONIOENCODING="utf-8"
# (a) measurement: appearance ENGAGED
python results/c1_assoc_engaged/probe_embed.py embed
#   == python scripts/eval_mot.py --data-root data/raw/mot17/MOT17/train --subsample 3 --max-frames 200 `
#        --trackers tracktrack.yaml --weights data/weights/crowdhuman_yolov8n_best.pt --imgsz 1280 `
#        --ignore-regions --limit-sequences 2 --out results/c1_assoc_engaged/run_embed --jobs 1 --embed
# (b) control: same thing with --no-embed
python results/c1_assoc_engaged/probe_embed.py control
# (c) inert-switch reproduction: --with-reid, no --embed  (E053's trk_reid.yaml condition)
python results/c1_assoc_engaged/probe_embed.py with_reid
```

`probe_embed.py` is a **read-only instrumentation harness**: it imports `scripts/eval_mot.py`,
patches four already-imported objects at runtime (nothing in `scripts/` or `src/` is touched) and
calls `eval_mot.main(argv)` with the real CLI args. `--jobs 1` takes the serial in-process branch
(`scripts/eval_mot.py:800-803`), so the hooks are on the measured code path. Counters land in
`results/c1_assoc_engaged/run_*/proof.json`.

Hooked layers: **L1** `tgtrack.appearance.Embedder.encode`, **L2**
`eval_mot.feats_from_embeddings`, **L3** `tgtrack.mot_layer.AllPersonTracker.update`, **L4**
`ultralytics.trackers.track_tracker.TRACKTRACK.update` (also counts tracks whose `curr_feat` is set
after the update — that field can only be filled through the ReID feature path).

## 3. Did the mechanism fire? Yes (run a), No (runs b, c)

| counter | (a) `--embed` | (b) `--no-embed` | (c) `--with-reid` |
|---|---|---|---|
| L1 `Embedder.encode` calls | **400** (= every frame) | 0 | 0 |
| L1 rows encoded / dim / mean L2 norm | **14 400** / 512 / 1.000 | – | – |
| L2 `feats_from_embeddings` calls with a real `(N,D)` array | **400** | 0 | 0 |
| L2 rows handed on, `(N,D)`, alignment failures | **14 400**, 512-d, **0** | 0 | 0 |
| L2 missing (unfillable) rows | **0** | 0 | 0 |
| L3 `AllPersonTracker.update` calls with `feats is not None` | **400 / 400** | 0 | 0 |
| L4 `TRACKTRACK.update` calls receiving `feats` | **400 / 400** (14 400 rows) | 0 | 0 |
| L4 `self.encoder is not None` | **True** (`_auto_encoder`) | False | **True** (`_auto_encoder`) |
| L4 frames where the cosine ReID term entered the cost matrix | **400** | 0 | **0** |
| L4 max tracks holding `curr_feat` (post-update) | **60** | 0 | **0** |

Interpretation:

* Run (a): 14 400 unit-norm 512-d vectors were extracted from the CrowdHuman detection crops
  (`data/weights/yolo26n-reid.onnx`, ONNX Runtime 1.20.2 + CUDAExecutionProvider), fed 1:1 with
  every detection to the tracker, and **60 tracks carried an embedding** — with `reid_weight = 0.5`
  that cosine term is in the cost matrix on all 400 frames. **The switch is live.** Detections and
  predictions changed (`num_pred_dets` 3 951 → 4 006 on MOT17-02), i.e. the run is not a no-op.
* Run (c) is the exact mechanism of the E053 artefact: encoder **installed**, features **never
  supplied** (`feats=None` → `encoder_input=None` at `track_tracker.py:496`), so the ReID term is
  never computed and the metrics are **bit-identical** to (b) at full float64 precision
  (`MOT17-02: HOTA 0.37556733221819333, AssA 0.393963071114671`; `MOT17-04: HOTA 0.6028919663743704,
  AssA 0.6388611990970576` in both runs). E053's "with_reid is inert, bit-identical" observation is
  therefore **reproduced and explained** — it was measuring the pass-through, not appearance.
* Run (b) also reproduces E047's per-sequence numbers **bit-identically**
  (`results/mot17_ch_det/pairs.csv`, same 2 sequences), even though E047 ran `--jobs 2`: the pair
  loop is per-sequence and independent, so the E047 baseline *is* a valid same-sequence comparison
  basis. The fresh control is kept anyway to rule out environment drift.

## 4. Metrics — same 2 sequences, appearance engaged vs not

Combined over the 2 sequences (`combine_mot`, 400 frames, 15 112 GT boxes):

| metric | (b) no-embed control | (a) `--embed` | Δ (pp) |
|---|---|---|---|
| HOTA | 0.520845 | 0.520478 | **−0.037** |
| DetA | 0.485057 | 0.484668 | −0.039 |
| **AssA** | **0.563505** | **0.562821** | **−0.068** |
| MOTA | 0.573584 | 0.571930 | −0.165 |
| IDF1 | 0.681943 | 0.681558 | −0.039 |
| IDSW | 144 | 148 | +4 |
| FP | 1 523 | 1 569 | +46 |
| FN | 4 777 | 4 752 | −25 |
| ms/frame | 70.61 | **826.97** | **+756.4 (11.7×)** |
| pipeline FPS | 14.16 | 1.21 | −12.95 |

Per sequence (so the sign flip is visible):

| sequence | AssA no-embed | AssA embed | Δ AssA (pp) | HOTA Δ (pp) | MOTA Δ (pp) | IDF1 Δ (pp) | IDSW (no→emb) |
|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | 0.393963 | 0.387493 | **−0.647** | −0.443 | −1.002 | −0.521 | 99 → 118 |
| MOT17-04-FRCNN | 0.638861 | 0.640667 | **+0.181** | +0.203 | +0.415 | +0.304 | 45 → 30 |

`num_frames` 200 per sequence; both runs 2/2 pairs ok, 0 failures.

## 5. Verdict

* **C1 as stated — REJECTED.** With appearance genuinely engaged (400/400 frames, 14 400 rows,
  60 tracks carrying embeddings, cosine term live on every frame, `reid_weight 0.5`), **AssA moves
  −0.068 pp** (bar was > +0.5 pp). HOTA −0.04 pp, MOTA −0.17 pp, IDF1 −0.04 pp, IDSW +4. The
  ledger's substantive conclusion — appearance does not buy association on this CrowdHuman stream —
  survives a *non-vacuous* test, so it is no longer an untested claim. The only positive signal is
  sequence-local and small: on MOT17-04 AssA +0.18 pp with 15 fewer ID switches, paid for by
  MOT17-02 AssA −0.65 pp and 19 more ID switches.
* **The "inert switch" observation — CONFIRMED and now pinned to a line.** E053's tracktrack+reid
  row (and, by the same code path, any `with_reid`-only run) is a pass-through: the encoder exists,
  `feats` is `None`, `encoder_input` is `None`, no cosine term. Numbers calling it "appearance
  assisted" should be re-labelled **inert / not appearance-assisted**; they are valid as
  "with_reid is a no-op for tracktrack", not as evidence about appearance.
* **Cost caveat:** appearance costs **+756 ms/frame** (11.7× wall) at ~36 detections/frame with
  this per-full-image `Embedder` — the naive ONNX call path is not viable for the pipeline even
  before the accuracy question. Any future appearance test should first batch the crops (or cache
  by track) so the comparison is not confounded by a 12× throughput collapse.

## 6. Housekeeping

* `scripts/`, `src/` untouched. Writes: `results/c1_assoc_engaged/**` (probe script, 3 run dirs with
  `pairs.csv`/`summary.csv`/`summary.md`/`run.json`/`proof.json`) and this report.
  `docs/EXPERIMENTS.md` and `README.md` were **not** edited (Lead folds this in).
* Side effect: the `--dry-run` plan check wrote `results/mot_eval/plan.json` (new file, 17:13, the
  directory previously held nothing else). No metric artefacts were touched.
* GPU shared with another job; all runs `--jobs 1`, serial in-process.
