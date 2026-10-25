# R_perf — independent performance review of the single-target path

Reviewer: delegated perf/eng reviewer. Target: `C:\Projects\track`, one shared
RTX 4060 Laptop 8 GB. Artifacts: `results/perf/` (scripts + raw JSON, listed in
§7). No repo source was modified; this document is the only file added outside
`results/perf/`.

Everything below is measured on this box, not quoted from the docs, unless marked
*anchor* (a number the task statement supplied).

---

## 0. TL;DR

| Question | Verdict |
|---|---|
| What dominates the target path? | **Appearance encoding: 79 % of wall** (225.1 of 286.5 ms/frame). Detector 11.9 %, JPEG decode 4.4 %, association python 3.4 %, motion/IMM 1.4 %, memory writes 0.3 %. |
| Biggest zero-risk win | **Stop re-encoding the winning box.** `_features_for` is called on 40/40 frames and costs 94.0 ms/frame = **33 % of wall**. Those features were already computed by `_ensure_features` two lines earlier. |
| Second win | **Call `session.run` directly instead of `AutoBackend`**: 1.16–1.7× on the forward, **parity verified bit-identical (`max_abs_diff = 0.0`)**. |
| Cost-ineffective shipped knob | `cost.part_matching=True`: **dTSR = 0.0000 in 43/43 paired ledger rows** across 7 masks. It costs ~0 ms (pure numpy), so it is a complexity/noise win to delete, not a speed win — **the mechanism that actually burns the clock is the 4-part × multi-candidate encode plus the duplicate encode** (`target_parts`, 43.8 ms/frame for a mean dTSR of +0.0414 with 33/48 rows exactly zero). |
| `--jobs 2`? | **No.** Same grid: 38.8 s @ jobs 1 → 36.1 s @ jobs 2 (−7 % grid wall, startup overlap only) while per-run latency went **443.7 → 641.9 ms/frame** and VRAM **4.2 → 6.5 GB**. Rule: **one GPU worker at a time**. |

---

## 1. Measurement conditions (read before trusting any absolute number)

This box is shared: while I measured, other agent processes were running
YOLO/ORT grids — `nvidia-smi` showed 13–60 % GPU utilisation and 4.4–7.8 GB of
8.1 GB in use, with up to four compute PIDs present. Consequence:

* Wall-clock numbers here carry a **~1.5–3× contention inflation**. The internal
  calibration anchor is the detector: measured **34.2–70.5 ms/frame** against the
  supplied anchor of **22.8 ms/frame** → inflation factor **1.5–3.1×** in the same
  run. Where a result depends on an absolute time I say so; where it depends on a
  *share within one run* (encoder vs detector) the contention largely cancels,
  because every stage is sampled in the same process on the same frames.
* Protocol: instrument in-process (monkeypatch only), **warm the model before
  starting the clock** (the first YOLO call pays ~10 s of load + CUDA init), then
  take the minimum/wall of repeated runs. The ledger's own `ms_per_frame` is
  *not* usable for cross-run comparison — see §5.
* `results/perf/profile_target.py` produced the stage tables; the same script was
  run for the shipped config and two ablation variants.

**A trap worth knowing about (it cost me two bad runs).** `_ensure_features`
wraps the encoder in `except Exception: self._last_diag["encode_error"] = repr(exc)`
(`src/tgtrack/target_tracker.py:816-820`, comment: *"a broken encoder must not
kill tracking"*). Any instrumentation bug that makes `embedder.encode` raise turns
the encoder into a **silent no-op**: the run gets *faster*, TSR stays plausible
because geometry alone still tracks, and the stage table reads "encoder = 2 % of
wall". I hit exactly this by patching `_crops_to_tensor` on the instance instead of
the class (rebinding `self` → `TypeError` → swallowed). Two consequences:

1. All profiling must assert `embedder.forward_calls == n_forwards` (my script now
   records `encode_sizes` and a per-instance snapshot for this reason).
2. **Production risk**: today a genuinely broken encoder degrades every run to
   geometry-only tracking with no visible failure in the run record. That is a
   larger correctness hazard than anything in this report; surface
   `_last_diag["encode_error"]` in `per_run.json` and warn loudly.

---

## 2. Q1 — Stage budget, dancetrack0007, 40 frames

Config: `--subsample 2 --mask none`, detector yolo26m@1280 half, embedder real
(CUDA EP, 448², dim 512), `--jobs 1`, target 1, 40 frames / 40 hits.
Source: `results/perf/prof_shipped.json`.

| Stage | ms/frame | share | note |
|---|---:|---:|---|
| **Appearance encoder (`Embedder.encode`)** | **225.1** | **79.0 %** | 79 forward calls / 40 frames = **2.0 per frame** |
| — ONNX forward (`session.run` via AutoBackend) | 182.9 | 63.8 % | 772 crops total = **19.3 crops/frame** |
| — CPU crop→tensor prep (`_crops_to_tensor`, 448² bilinear, fp32) | 39.6 | 13.8 % | pure CPU, on the critical path |
| — bookkeeping / normalization | 2.6 | 0.9 % | |
| **Detector (yolo26m@1280)** | 34.2 | 11.9 % | anchor 22.8 → 1.5× inflation in this run |
| **IO / JPEG decode** | 12.5 | 4.4 % | `Sequence.frame()` |
| **Association python** (cost matrices, gate, select) | 9.7 | 3.4 % | includes `compute_costs` twice/frame |
| **Motion / IMM** (predict+update+mahalanobis) | 4.0 | 1.4 % | 3 models |
| **Memory ops** (STM update + LTM add) | 0.8 | 0.3 % | 160 LTM adds / 40 frames |
| Tracker other | 0.1 | 0.0 % | |
| **Total** | **286.5** | 100 % | sum of rows = 286.4 |

Encoder internals (same run), split by *caller*:

| Caller | ms/frame | share of encoder | calls |
|---|---:|---:|---:|
| `_ensure_features` (candidate crops, top-k by cheap cost) | 132.7 | 59 % | 39 |
| **`_features_for` (the chosen box, re-encoded in `_absorb`)** | **94.0** | **42 %** | **40** |

The two overlap (99 %), so the sum exceeds the total; the point is that **42 % of
encoder time recomputes features for a box whose `part_feats` were populated
microseconds earlier in the same frame** (`_ensure_features` sets
`d.part_feats` / `d.feat`; `_absorb` → `_features_for` then calls
`self.embedder.encode(frame_bgr, box.reshape(1,4), parts)` again —
`src/tgtrack/target_tracker.py:1279`).

Crop demand per forward (histogram of `len(crops)`): `{4: 40, 12: 5, 16: 32, 20: 2}`
— i.e. typically 4 parts × 1–4 surviving candidates. All are far below
`max_batch=64`, so the adaptive chunking never triggers.

### Calibration of the ReID forward (why "3.2 ms / 64 crops" is not what the pipeline sees)

Isolated micro-benchmarks (`results/perf/bench_direct_session2.json`,
`bench_batching.json`, `micro_bench.json`), GPU-quietest window:

| Path | 4 crops | 16 crops | 64 crops | per crop (64) |
|---|---:|---:|---:|---:|
| `AutoBackend` (what the pipeline uses) | 20.7 ms | 53.7 ms | 180.3 ms | 2.90 ms |
| **direct `session.run`** | 12.9 ms | 24.2 ms | 105.5 ms | **1.73 ms** |
| 1 crop, either path | — | — | — | 11.4–15.0 ms (single-call floor) |

Two facts follow, and they matter more than the absolute values:

1. **There is almost no batching gain.** Per-crop cost is flat from 16 → 64 crops
   (1.51 → 1.65 ms direct; 3.35 → 2.82 ms via AutoBackend). Cost = *fixed
   per-call overhead* (≈7–10 ms measured: launch + H2D of a **fp32 448² tensor**,
   154 MB at 64 crops) + ≈1.5 ms/crop. Cutting the crop count barely helps;
   cutting the *number of calls* helps a lot.
2. **`fp16=True` in `Config` is a no-op today.** `embedder.info` reports
   `fp16: true`, but `AutoBackend.fp16` is `False` and the session input is
   `tensor(float)` — the shipped `yolo26n-reid.onnx` is an fp32 export. Every
   forward therefore moves 4 bytes/pixel and runs fp32 kernels.
3. The pipeline's in-run forward cost (≈9 ms/crop at 10 crops/call) is 3–5× the
   isolated steady state, because each frame issues two *small* calls, each paying
   the fixed cost on an already-contended GPU.

The supplied anchor "3.2 ms per 64-crop batch" is ~1.8× faster than the fastest
*steady-state* 64-crop forward I could measure on this box (105.5 ms direct), i.e.
it is not reproducible here even with the AutoBackend overhead removed. Treat it as
a target, not a measurement.

---

## 3. Q2 — Ranked speedups that do not change decisions

Ranked by measured wall saving on the 286.5 ms/frame budget. "Parity" is the
*verification* you must run, not a promise.

### S1. Delete the duplicate encode of the winning box — save 94.0 ms/frame (**−33 % wall**)

`_features_for` re-encodes the box that `_ensure_features` already encoded in the
same `step()`. Fix: in `_absorb`, reuse the matched `Detection.part_feats` when the
box is the one that was just encoded (pass the `Detection`, not a bare box; fall
back to encoding only when `force=True` or the detection carries no features —
e.g. after a global-search jump to an unencoded candidate).

* Expected: −94.0 ms/frame on this sequence (286.5 → ~192 ms, −33 %). On a dense
  MOT17 frame the absolute saving is larger (the re-encode is a full 4-part
  forward there too).
* Risk: **low but not zero**. The two paths can ask for different part sets:
  `_features_for` uses `("full",)` when `force is False and not self.ltm.entries`,
  otherwise `_query_parts()`, while `_ensure_features` used `_query_parts()` *and*
  `crop_quality`/`visible_ratio` side effects. Reuse is only legal when the part
  set and the frame are identical.
* Parity: assert `part_feats` keys/values are identical per frame (they should be
  the *same arrays*), TSR/FAR/precision identical to 4 dp over ≥3 sequences × 200
  frames, and `forward_calls` drops from 2.0 to 1.0 per frame. If the reused dict
  differs in keys, fail loudly rather than silently falling back.

### S2. Bypass `AutoBackend`; call the ORT session directly — save ~40 ms/frame (**−14 % wall**), **bit-identical**

`reid.model(chunk)` goes through `AutoBackend.forward`, which re-reads metadata,
maybe permutes, and round-trips the output tensor through `from_numpy`. The
Embedder only needs `session.run(None, {input_name: batch})`.

* Measured: 64 crops 180.3 → 105.5 ms (**1.71×**), 16 crops 53.7 → 24.2 ms
  (**2.22×**), 4 crops 20.7 → 12.9 ms (1.60×).
* **Parity verified: `max_abs_diff = 0.000000`, `bit_identical = true`**
  (`results/perf/bench_direct_session2.json` → `parity`). Same session, same
  weights, same input buffer ordering.
* Risk: low. Keep `_run_onnx`'s chunking/OOM-halving behaviour around the direct
  call, and keep the static-batch padding path for fixed-batch exports.

### S3. Coalesce the frame's encodes into one forward — ~10–90 ms/frame (**3–30 %** of the encoder)

Two calls per frame (candidates + chosen box) each pay the ~7–10 ms fixed cost, and
S1 removes one of them anyway; after S1, coalesce the remaining candidate batch with
any other pending crop demand (e.g. audit, `_part_ema` updates) into a single call
per frame.

* Expected: −1 fixed cost/frame ≈ **7–10 ms/frame** (measured single-call floor
  11.4 ms at 1 crop vs 1.73 ms/crop at 64).
* Risk: **batch-size-dependent kernels can change the last bits**. Parity plan:
  `max|Δembedding| ≤ 1e-6` on 200 frames plus **decision-level** equality
  (identical matches, TSR/FAR) — do not claim bit-identity here.
* Same lever, bigger: raise `max_batch` only if you also coalesce; on its own it
  changes nothing (all current batches are ≤20 crops).

### S4. Real fp16 (needs a new export) — expected 1.4–2× on the forward

`embedder.fp16` currently only flips a flag that the fp32 ONNX cannot honour
(§2). Either export `yolo26n-reid.fp16.onnx` or leave the knob off — as shipped it
is dead config that misleads every reader of `embedder.info`.

* Expected: input tensor 154 → 77 MB at 64 crops, plus fp16 kernels; ≈1.4–2× on
  the forward (−60 to −90 ms/frame at today's encoder share).
* Risk: **medium** — embedding drift. Parity: cosine ≥ 0.999 per dimension-wise
  comparison, TSR/FAR/precision within ±0.005, and a distractor-switch audit.
  Not bit-identical; report it as an approximation.

### S5. Session options / IO binding — unmeasured, low risk, do after S1–S3

`SessionOptions` are left at defaults; nothing sets
`intra_op_num_threads`/`inter_op_num_threads`, arena growth, or
`cudnn_conv_algo_search`. IO binding (preallocate the output `OrtValue`, keep the
input tensor on the GPU and copy crops into it) would remove the per-call H2D of
the whole 154 MB batch — this is the structural version of S3 and is where the
remaining fixed cost lives.

* Expected: 5–15 % on the forward, unmeasured here — **needs an E-number**.
* Risk: low; note `_row_array` already documents why output buffers must be copied
  out (`src/tgtrack/appearance.py:666-679`) — keep that when binding.

### S6. Skip the appearance pass when the geometry winner is unambiguous — up to −100 % of the encoder on easy frames, **changes decisions**

On this 40-frame slice the target tracked geometry-only with TSR 1.0 (when the
encoder was accidentally disabled, §1) — the appearance stage bought nothing there,
but it *is* what the whole LTM/parts machinery is for on hard frames. If you ship
this, gate it hard: skip only when exactly one candidate survives the motion gate
**and** its cost margin exceeds a configurable multiple of the runner-up.

* Expected: −132.7 ms/frame (the candidate encode) on frames that qualify.
* Risk: **high** — it is a decision change by construction. Parity is not
  achievable; require an E-number showing dTSR ≥ 0 across masks before shipping,
  and log the skip rate so it cannot silently become the default path.
* Note `_encoding_set` already caps the work (`top_k = max(app_topk*3, 12) = 15`,
  `MAX_ENCODE = 64` on global search), so the cheap version of this is a **smaller
  `top_k`**, not a new mechanism.

### What I would *not* spend time on

* `cost.part_matching` — 0 ms of wall time (pure numpy post-process,
  `src/tgtrack/association.py:501`). Deleting it saves code, not seconds.
* `--jobs 2` as a speed lever (§6).
* Enlarging `max_batch` (§3 S3).
* Rewriting association python: 9.7 ms/frame total (3.4 %) puts a hard ceiling of
  3 % on any such work.

---

## 4. Q3 — Cost-effectiveness of the shipped knobs

Method: mined **every** `results/*/per_run.json` (115 directories, 959 pooled
rows), paired each ablation row with the `baseline` row of the *same directory and
mask* → **815 paired rows / 111 dirs**
(`results/perf/mine_ledger.py` → `ledger_paired.csv`, `results/perf/ledger_report.py`).

### 4.1 The ledger's wall-time column is not comparable — and that is itself a finding

Same nominal baseline, `mask=none`, across directories:
**min 9.5, median 189.2, max 980.4 ms/frame, σ ≈ 145.5** (n = 485). `mask=lower`
is worse (min 8.5, max 2061.1, σ ≈ 287). Cause: different `--max-frames`,
`--jobs`, concurrent agents, no quiescence check, no repetition. Any per-knob
"costs N ms" claim built on this column is noise — including the tempting
"`part_matching` is −22.8 ms" reading.

Recommendation: record `jobs`, `max_frames`, GPU utilisation/VRAM at start, and run
timing variants **serially, best-of-3**, then key on a `timing_run` flag. Until
then, only `TSR`/`FAR`/precision deltas from the ledger are usable.

### 4.2 Metric deltas *are* trustworthy — and they say which mechanisms did nothing

`dTSR` mean per group vs same-mask baseline (n = paired rows):

| group | n | mean dTSR | rows with dTSR == 0 | mean dms (noise) |
|---|---:|---:|---:|---:|
| **part_matching** | **43** | **+0.0000** | **43 / 43** | −22.8 |
| freeze | 42 | +0.0000 | 42 / 42 | −8.0 |
| part_aware | 18 | −0.0026 | 17 / 18 | −5.2 |
| hard_negative | 104 | −0.0181 | 72 / 104 | −18.2 |
| stm_ltm | 63 | +0.0144 | 44 / 63 | −11.1 |
| ltm_capacity | 54 | −0.0176 | 45 / 54 | −11.8 |
| gate | 68 | +0.0146 | 45 / 68 | −6.0 |
| target_parts | 48 | +0.0414 | 33 / 48 | −15.7 |
| distractor_margin | 81 | +0.0132 | 55 / 81 | +0.6 |
| trail | 9 | **−0.1100** | 6 / 9 | +14.1 |
| tube_velocity | 6 | **−0.1612** | 0 / 6 | −39.9 |

`part_matching` is the standout: **43 paired comparisons, 7 masks, every single one
exactly 0.0000** — and in the e019 receipt grid the underlying numbers are
`0.4010 → 0.4010` (mask=none) and `0.3096 → 0.3096` (mask=upper), with FAR, FA
events and precision bit-for-bit equal too. A knob that changes nothing in 43/43
runs is either dead code or never exercised.

### 4.3 The verdict

* **Cheapest thing to remove: `cost.part_matching` (default `True`,
  `src/tgtrack/config.py` `CostCfg.part_matching`)** — zero measured gain in 43/43
  rows, zero measured cost. It is *not* a wall-time hog; removing it buys clarity
  and one fewer score path, not speed. Ship the deletion with an E-number that
  states "dTSR = 0.0000, n = 43, 7 masks" so the ledger stops re-testing it.
* **Most expensive mechanism for the least measured gain: the multi-part encode
  together with the duplicate encode** — `target_parts = ("full","upper","lower",
  "head")` costs **43.8 ms/frame** on this sequence (225.1 → 181.3 ms/frame when
  cut to `("full",)` in `results/perf/prof_single_part.json`, crops 772 → 193),
  for a mean dTSR of only +0.0414 with **33/48 paired rows at exactly zero**.
  Note the shape of that cost: a 4× crop reduction buys only **19 %** of encoder
  time, because the forward is fixed-cost-bound (§2) — which is precisely why S1/S3
  (fewer *calls*, no duplicate) beat "encode fewer parts".
* Mechanisms whose gain is **unmeasured — needs an E-number**: `audit`
  (ships off, `audit_interval=0`), `trail` (ships off, `trail_weight=0.0`),
  `tube_velocity` (ships off, `tube_velocity_weight=0.0`). The ledger rows that
  exist for `trail`/`tube_velocity` show them *hurting* (dTSR −0.11 / −0.16), but
  they are not shipped on, so they are not part of this verdict.
* `freeze` (42/42 zero) and `part_aware` (17/18 zero) deserve the same treatment as
  `part_matching`: an E-number that either kills them or explains when they matter.

---

## 5. Q4 — Parallel efficiency and the safe concurrency rule

Same grid both times: `--ablate part_matching` (2 variants) × 1 sequence × 40
frames, subsample 2, mask none, `--data-root data\raw\_root_dancetrack0007`.
Sources: `results/perf/jobs1/per_run.json`, `results/perf/jobs2/per_run.json`.

| | `--jobs 1` | `--jobs 2` | delta |
|---|---:|---:|---:|
| grid wall (s) | 38.84 | 36.08 | **−7 %** |
| run 1 wall (s) / ms per frame | 17.77 / **443.7** | 25.70 / **641.9** | **+45 % latency** |
| run 2 wall (s) / ms per frame | 12.02 / **300.2** | 25.61 / **639.5** | **+113 % latency** |
| p95 (ms) | 374.8 / 348.9 | 691.4 / 692.8 | ~2× worse tail |
| peak VRAM seen (`nvidia-smi`, sampled) | ~4.2 GB | **6.5 GB** | **+2.3 GB** |

Reading: `--jobs 2` shaved 7 % off *grid* wall only because two ~15 s
model-load startups overlapped. Every individual run got **1.4–2.1× slower** and
the tail doubled, because both workers share one GPU's SMs and memory bandwidth.

**Safe concurrency rule for this box.**

1. **One GPU process at a time: `--jobs 1` for every detector/embedder grid.**
   The GPU is the serial resource; splitting it multiplies VRAM and start-up cost
   without adding throughput.
2. **VRAM budget**: one worker ≈ 4.2 GB resident (detector + ORT arena 269 MB +
   CUDA contexts + desktop). A second worker adds ≈ 2.3 GB → 6.5 GB of 8.1 GB,
   which is what produced the recorded
   `BFCArena::AllocateRawInternal` failure at **269,746,176 B**. Anything else on
   the card (browser, another agent) then pushes it over. Two embedder workers on
   an 8 GB card are *not* safe even when they "fit" at start, because the arena
   grows per allocation event.
3. If you need wall-clock parallelism, parallelise **CPU-only variants**
   (`--no-embed`, `detector=gt`, analysis/plots) or shard across *time* — run
   sequences sequentially in one process and reuse the cached models
   (`get_detector`/`get_embedder` cache by config tuple, so a single process over
   N sequences pays one load).
4. Keep the existing adaptive OOM-halving in `_run_onnx`
   (`src/tgtrack/appearance.py:710-741`) as the backstop, and **grep runs for
   `oom_retries > 0`** — a non-zero counter invalidates that run's timing and
   should invalidate its batch-size assumption too.

---

## 6. Recommended action order

1. **S1** (drop duplicate encode) — 33 % of wall, decision-neutral if gated on
   identical part sets. Biggest win, smallest code change (`_absorb` /
   `_features_for`).
2. **S2** (direct `session.run`) — a further ~14 %, provably bit-identical.
3. **S3** (coalesce to one forward/frame) — 3–30 % of the encoder, verify with
   `max|Δ| ≤ 1e-6` + decision parity.
4. **Instrumentation hygiene** — expose `encode_error`, `forward_calls`, GPU
   state and `jobs` in `per_run.json`; mark timing runs and use best-of-3.
5. **Config truth-up** — `fp16` is dead on the ONNX path (fix or remove);
   `part_matching` deleted with its E-number; `freeze`/`part_aware` given a
   deadline to justify themselves.
6. **Ranked last**: `target_parts` reduction (19 % of encoder for a real part of
   the design), fp16 export (medium risk, needs quality parity), session options
   (measure first).

Expected combined effect of S1+S2 on this budget, holding decisions fixed:
286.5 → **~150 ms/frame (−48 %)**, with S1 verified by identical feature arrays and
S2 verified bit-identical.

---

## 7. Artifacts (`results/perf/`)

| File | What it is |
|---|---|
| `profile_target.py` | In-process stage profiler (monkeypatch only). Emits stage table, budget, call/crop counts, per-instance encode snapshot. Variants: `shipped`, `part_matching_off`, `single_part`, `no_audit`, `no_absorb_reencode`. |
| `prof_shipped.json` | The §2 budget run (40 frames, shipped config, wall 286.5 ms/frame). `prof_shipped4.json` is the same config from a heavily contended window (wall 661.5 ms/frame, encoder share 82.2 %) — keep it as the spread indicator. **Note**: `prof_shipped2/3.json` are invalidated by the instance-level `_crops_to_tensor` patch (§1); keep them only as evidence of the silent-encoder-failure mode. |
| `prof_single_part.json` | `target_parts=("full",)` — prices the parts multiplier (crops 772 → 193). |
| `micro_bench.py` / `micro_bench.json` | Detector + `encode()`/`encode_crops()` scaling, 1…64 boxes × 4 parts. |
| `bench_batching.py` / `bench_batching.json` | Fixed-N vs alternating-N forwards; shows the missing batching gain and the first-call plan cost. |
| `bench_direct_session.py` / `bench_direct_session2.json` | `AutoBackend` vs direct `session.run`, plus the **bit-identical parity check**. |
| `mine_ledger.py`, `ledger_report.py` | Ledger miner/analyser → `ledger_rows.csv` (959 rows), `ledger_paired.csv` (815 paired rows). |
| `jobs1/`, `jobs2/` | `run_target.py` grid outputs (per_run.json / summary.json / table.*) for the §5 comparison. |
