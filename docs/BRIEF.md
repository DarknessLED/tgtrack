# Shared Brief — Human Target Tracking (DJI-ActiveTrack-style)

## Machine / environment facts (verified)
- Working dir: `C:\Projects\track` (Windows, PowerShell)
- GPU: **NVIDIA RTX 4060 Laptop, 8 GB VRAM** (sm_89), driver 595.97, CUDA 13.2 driver
- Python **3.12.7**; CUDA toolkit 11.8 at `C:\Software\CUDA\bin\nvcc.exe`
- Installed & VERIFIED: **torch 2.7.0+cu126 / torchvision 0.22.0+cu126** (`cuda_available True`, cc 8.9, 7.9956 GB; 2048^3 matmul x20 = 9.6 ms)
- Installed: **ultralytics 8.4.185**, lap 0.5.13, **onnxruntime-gpu 1.20.2**, onnx 1.23.2, numpy 1.26.4,
  opencv-python 4.11.0.86, scipy 1.13.1, pandas 2.2.2, matplotlib 3.9.2, timm 1.0.26
- pip works via `-i https://pypi.tuna.tsinghua.edu.cn/simple` (9.3 MB/s). PyTorch cu126 index works:
  `pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126`
- `huggingface.co` is blocked; **`hf-mirror.com` works** (set `HF_ENDPOINT=https://hf-mirror.com`).
  **github.com release downloads are unreliable/blocketed**; `https://ghproxy.net/https://github.com/...` works.
- **`ffmpeg` IS available, but the canonical-looking one is a trap.** `C:\Software\Trae CN\resources\app\bin\ffmpeg.exe`
  reports 6.1.1 yet is built `--disable-everything` (no `image2` demuxer, no `mjpeg` encoder, no `select`
  filter) and cannot read a JPEG sequence. Use **`C:\Software\oopz\ffmpeg.exe`** (gyan.dev) with
  `-start_number 1 -vsync 0`. `ffprobe.exe` exists only beside the Trae build (fine for mp4).
  `git` 2.49 available. Disk C: 213 GB free.
- Console default encoding is GBK -> always set `$env:PYTHONIOENCODING="utf-8"` before python that prints Unicode.

## Model weights — use `Config()` defaults, never hardcode or re-download
- `config.py` resolves weights automatically via `resolve_weight(name)`: existing absolute path ->
  `data/weights/<basename>` -> `./<name>`. Both `DetectorCfg.weights` and `EmbedderCfg.model` use it.
- `yolo26n.pt` 5,544,453 B / 2,572,280 params; `yolo26m.pt` 44,255,705 B / ~21.9 M params.
- `yolo26n-reid.onnx` 9,873,245 B (md5 `6D01B3F9CD3F166A4FDCA8CC3EC3039F`), 2,432,284 params,
  512-d L2-normalised output, 1280-d penultimate -> `C:\Projects\track\data\weights\yolo26n-reid.onnx`
- **Both of these are AGPL-3.0** (`license` in the ONNX metadata; ultralytics for the detector).
  A networked product would trigger AGPL source obligations — one licensing decision, not two.
  Apache-2.0 alternatives exist: OpenCV Zoo `person_reid_youtu_2021nov.onnx`, RTMPose, SAM 2.1.
- The ReID ONNX declares **`imgsz=[448,448]`** — it was TRAINED at 448. `EmbedderCfg.imgsz` is now 448.

### Detector default is `yolo26m` at `imgsz=1280` — measured, not guessed
On this GPU the detector is NOT network-bound (letterbox + NMS + host transfer dominate), so n/s/m cost
nearly the same per frame while recall differs enormously. `scripts/probe_recall.py --confirm`, pooled over 5
MOT17 sequences, 2,658 GT boxes with visibility >= 0.25, greedy 1-1 at IoU 0.5:

| setting | pooled recall | det/frame | ms/frame | FPS |
|---|---|---|---|---|
| `yolo26n@640` | 0.368 (978/2658) | 7.6 | 33.5 | 29.9 |
| `yolo26m@1280` | **0.733** (1947/2658) | 19.1 | 42.9 | 23.3 |

**2.0x the recall for +28% frame time.** This is the whole explanation for the all-person MOT layer's
low DetA (~0.25 was recall, not tracking). `yolo26n@640` stays available via `Config().clone(...)` for
fast sweeps. MOT17-05 (113 GT, easy) is 0.858 for BOTH — the gain is concentrated in crowds.

## ONNX Runtime GPU — solved, do not re-debug
- `onnxruntime` (CPU) and `onnxruntime-gpu` must NOT both be installed; they shadow each other.
  Only **onnxruntime-gpu 1.20.2** is installed (CUDA 12 + cuDNN 9 line, matches torch 2.7).
- The CUDA provider is a plugin DLL that links `cublasLt64_12.dll` / `cudnn64_9.dll`, which live in
  `C:\Software\Anaconda3\Lib\site-packages\torch\lib` and are NOT on PATH -> the plugin silently fails
  and ORT silently falls back to CPU.
- Fix: `src/tgtrack/runtime.py` -> call **`ensure_ort_cuda()` before constructing the InferenceSession**,
  and pass `providers=ort_providers(device)` (always ends with CPUExecutionProvider).
- ReID latency: **bs1 2.74 ms, bs40 12.7 ms (0.317 ms/crop) on CUDA** vs 4.25 ms / ~2.7 ms per crop on CPU (~8x).


## ultralytics 8.4.185 facts (verified by inspecting the wheel)
- YOLO26 configs exist: `cfg/models/26/yolo26.yaml`, `yolo26-p2.yaml`, `yolo26-p6.yaml`,
  `yolo26-seg/pose/obb/cls/depth/sem.yaml`, plus `yoloe-26.yaml` (open-vocabulary/promptable).
- **ReID assets**: `ultralytics/trackers/utils/reid.py` defines
  `REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")`
  => `yolo26n-reid.onnx`, `yolo26s-reid.onnx`, `yolo26m-reid.onnx`, `yolo26l-reid.onnx`, `yolo26x-reid.onnx`
  auto-downloaded by `attempt_download_asset()`. Default crop size 224.
- ReID API: `ReID(model, imgsz=224, device=None, fp16=False)`; `__call__(img_bgr, dets_xywh) -> list[np.ndarray|None]`;
  `build_encoder(with_reid, model, device)` where `model="auto"` reuses detector backbone features;
  `smooth_feature(feat, smooth, alpha)` does L2-norm + EMA.
  `.pt` checkpoints route through the YOLO predictor's `embed=[...]` path; `.onnx`/`.engine`/OpenVINO route through `AutoBackend`.
- Built-in tracker backends (`cfg/trackers/*.yaml`), all usable via `model.track(..., tracker="x.yaml")`:
  - `bytetrack.yaml`
  - `botsort.yaml` — `gmc_method: sparseOptFlow`, `with_reid`, `model: auto`, `proximity_thresh`, `appearance_thresh`
  - `ocsort.yaml`
  - `deepocsort.yaml` — CVPR2023 Deep OC-SORT; `alpha_fixed_emb: 0.95`, `with_reid`, `use_byte`, `inertia`, `delta_t`
  - `fasttrack.yaml` — arXiv 2508.14370; occlusion-aware, Kalman rollback, `occ_cover_thresh`, `occ_reappear_window`, `enlarge_bbox_occ`, `dampen_motion_occ`, `active_occ_to_lost_thresh`, `init_iou_suppress`
  - `tracktrack.yaml` — CVPR 2025 "Focusing on Tracks for Online MOT"; HMIoU, iterative assignment,
    Track-Aware Initialization (`tai_thr`, `min_track_len`), weights `iou_weight/reid_weight/conf_weight/angle_weight`,
    `lost_match_thr` for long-occlusion rebind, `penalty_p/penalty_q/reduce_step`

## Task (from the user, verbatim intent)
1. Download/use **ultralytics YOLO26** person detection + ReID models, **n** variants; better param-matched models welcome.
2. Download **person-tracking datasets** for experiments; diverse; **total < 10 GB**; images may be compressed.
3. **Survey** tracking strategies of the last decade, especially last 5 years: top journals/conferences, big tech,
   research institutes, competitions, blogs/forums.
4. Final system must: detect **all** persons; track the **click-selected** person (DJI-like). The selected target needs
   **long-term + short-term ReID**; must track effectively when only **part** is visible (upper body, lower body, face,
   body, ...). Other persons need only **ID distinction**. Must handle **long/short-term out-of-frame**, **fast motion**,
   **nonlinear motion**.
5. Build the experiment environment; **parallel** experiments; **no model fine-tuning** — optimize the **strategy**.
6. Subagents allowed.

## Non-negotiable design pillars
- Detection: YOLO26 (person class) — all persons.
- All-person ID: multi-object tracker layer (choose/compare backends).
- **Target layer (the novel part)**: a dedicated single-target tracker for the clicked person, decoupled from the
  all-person MOT layer, with:
  - Short-term memory: recent EMA appearance + motion filter (Kalman / IMM / OCM direction consistency)
  - Long-term memory: quality-weighted, diversity-aware embedding gallery for recovery after long occlusion / out-of-frame
  - Partial-view robustness: part-aware crops (upper body / lower body / head-face / torso) matched against a part-aware bank
  - Out-of-frame: coast motion, freeze/guard appearance bank, expanded search, then global re-detection
  - Distractor suppression: hard-negative mining among similar-looking persons
- Evaluation: MOT metrics (MOTA/IDF1/HOTA/AssA) for the all-person layer; **target-centric metrics**
  (target success rate, precision, lost rate, re-acquisition rate/latency, ID-switch-free rate) for the target layer.

## Evidence-backed strategy decisions (do not re-litigate without new measurements)

### Part-aware matching is the mechanism that survives occlusion (HEADLINE RESULT)
Measured by `scripts/exp_partial_view.py`: paint out a region of a real person crop, re-encode, and rank
all gallery identities. The tracker is **not** told which mask was applied — visibility is estimated from
image structure energy alone, exactly as the real tracker does. Mean identity Rank-1 over the occluded
masks (3 identities, 44 queries):

| strategy | occluded Rank-1 |
|---|---|
| `full` (whole-body descriptor only) | 0.655 |
| `mean_parts` | **0.951** |
| `quality_w` (structure-weighted) | 0.920 |
| `visible_gate` | 0.920 |
| `max_part` | 0.898 |

Control (clean crops; same-identity minus different-identity cosine) proves **every** part carries
identity evidence, so `head` is NOT a degenerate embedding: full +0.194, upper +0.168, lower +0.174,
**head +0.297**. Independently, R2 measured that when only the upper body is visible, the same person
scores full 0.757 but upper 0.920, while a DIFFERENT person's upper scores 0.48.
CAVEAT: the table above is 3 synthetic sprite identities. Re-run on real MOT17/DanceTrack before
quoting it as final, and treat "head is not degenerate" as data-dependent until then.

### Input resolution
Embeddings at different `imgsz` are NOT comparable (arXiv 2403.10022), so galleries must be stamped with
`EmbedderCfg.spec` (e.g. `yolo26n-reid.onnx@imgsz448@v2`, see `EMBED_SPEC_VERSION`) and a mismatch must
be refused. Identity Rank-1, clean / occluded, full-only / part-based:

| imgsz | clean full | occluded full | occluded parts | ms/crop batched |
|---|---|---|---|---|
| 224 | 0.886 | 0.655 | 0.951 | 0.295 |
| 448 | **1.000** | **0.780** | **0.989** | 1.069 |

448 costs ~3.6x batched but only ~1.27x at batch size 1. Encode parts only for the target + candidates.

### Architecture (R3 recommendation + R1's MOT evidence)
- Keep detect-all-persons -> MOT layer as the *awareness* layer; the clicked target is a **decoupled**
  single-target layer. Do NOT let an end-to-end SOT net (OSTrack/MixFormer/ODTrack) be the decision
  maker: they are trained for "same object", not "same person", and cannot express "absent".
- **Refusal is a first-class outcome**: accept only if `S(c*) > tau_acc` AND `S(c*) - S(c2) > tau_margin`.
  The margin test, not appearance alone, is the anti-latch mechanism (our `cost.distractor_margin`).
- R1's transferable numbers: OC-SORT's observation-centric correction is +7.3 HOTA over ByteTrack with
  identical detections (54.6 vs 47.3, DanceTrack); HMIoU is the only built-in geometry cue that survives
  vertical truncation; FastTracker's occlusion rollback (>0.7 coverage -> restore position 3 frames back,
  velocity 5 back, h*1.1, v*0.5); `lost_match_thr` ships disabled (0) in `tracktrack.yaml` and enabling it
  is free occlusion recovery.
- DJI Air 3S manual, verbatim: on loss the aircraft "*will keep flying at the current speed and
  orientation to try re-identifying the subject*" — commercial re-acquisition moves the *platform*
  instead of running a global re-detector — and DJI admits the subject "*may be inadvertently swapped to
  another subject if they pass nearby each other*". That swap is precisely what the margin gate targets.
- Face cues are NOT viable at detector imgsz 640: a full-height person yields ~85 px face, ~20 px
  inter-eye, below the 30-60 px commercial requirement. Adopt only as a gated auxiliary vote.

## Front doors (run these)
- `python scripts/demo_click.py --source <video|dir|MOT seq> [--click X,Y] --save-video` — the actual
  product behaviour: detect all persons, click one, follow it, colour-code the tracker's own state, and
  score the run with `evaluate_target` when ground truth exists.
- `python scripts/exp_partial_view.py --source <video|dir>` — the partial-view appearance experiment.
- `python scripts/eval_mot.py --data-root data/raw/mot17/MOT17/train --out results/mot_eval --jobs 4 --subsample 3`
  — the all-person MOT layer over real MOT-format sequences (process pool over (sequence, tracker) pairs).
- `python scripts/run_target.py --data-root data/raw/mot17/MOT17/train --out results/target --ablate all --mask none,lower,upper`
  — the clicked-target strategy-ablation grid.
- `python scripts/ablate_motion.py` — motion-model ablation driven through `TargetTracker` as a black box.
- `python scripts/run_video.py ...` — single-video detections + annotated mp4.
- `python scripts/probe_recall.py [--confirm]` — detector recall vs model size and input resolution.
- `python scripts/pick_observable_target.py` — ranks identities by their post-mask visible box, so a masked
  run can be pointed at a target that is genuinely still detectable. Read-only; this is the tool that
  exposed the "38x54 px invisible target" defect.
- `python scripts/probe_mask_detect.py` — does the mask remove the person from the DETECTOR too? Uses no
  tracker at all. Read-only.
- `python scripts/probe_why_refuse.py` — replays `run_sequence` and prints the per-frame accept/refuse
  reason with each detection's IoU against the visible AND the full GT. Read-only.

NOTE: dataset sequences live at `data/raw/mot17/MOT17/train/<SEQ>` and `data/raw/dancetrack/val/<SEQ>`;
VisDrone was converted to MOT layout at `data/sequences/visdrone/<SEQ>`. There is no MOT tree under
`data/sequences/` — pass the real root to `--data-root`.

## Datasets (acquired; see `docs/research/R4_datasets.md`)
Byte-verified against origin:

| archive | bytes | licence | what we use |
|---|---|---|---|
| `data/raw/MOT17.zip` | 5,860,214,001 | CC BY-NC-SA 3.0 | train, `-FRCNN` variant only (DPM/SDP are byte-identical duplicates) |
| `data/raw/dancetrack_val.zip` | 4,209,785,614 | annotations CC BY 4.0, **non-commercial research only** | `val/` (25 sequences) |
| `data/raw/visdrone_mot_val.zip` | 1,602,586,611 | CC BY-NC-SA 3.0 | closes the aerial/UAV viewpoint gap |

**The 10 GB cap is unit-ambiguous.** MOT17+DanceTrack raw archives are 10,069,999,615 B = 9.378 GiB
(6.2% under a 10 GiB cap) but 10.070 GB decimal (0.7% **over**). The deliberately-used, extracted set is
851 MB (MOT17 FRCNN) + 4,091 MB (DanceTrack val) = **4.94 GB**, so the deliverable is comfortably inside
the cap on either reading; the raw archives are redundant packaging and are pruned once extracted.

Two coverage facts that change how experiments must be designed:
- **DanceTrack's `visibility` column is the constant `1`** across all 225,148 val rows — it carries no
  occlusion signal. Every visibility / partial-view ablation must run on **MOT17 only**.
- **Neither dataset has GT during a gap**, so re-acquisition latency cannot be scored from either. This
  is a protocol gap, not a data gap, and it is bridged by masking a real target's box for N frames and
  using its own reappearance as GT (`scripts/run_target.py --mask`), at zero download cost.

## Search radius while blind (defect found by motion-dev, fixed)
`_gate_override()` used to call `MotionModel.gate_radius()`, which returns `gate_base + gain*speed` in
bare **pixels**; the canonical `association.gate_radius()` multiplies `gate_base` by the frame diagonal.
On 1080p that is ~15 px where it should be ~147 px, and the growth term used the coast-damped speed
(0.98^fsu), so the radius peaked (~264 px at fsu=49) and then *decayed*. `_global_search()` also armed
only past `lost_long_after=60` frames (~2 s). Net effect: a target that reappeared a few hundred pixels
away was rejected as `outside_radius` until fsu=61. Fixed by (a) using the canonical radius with
`frame_diag`, (b) freezing the growth term on `_v0_speed` — the speed at the last ACCEPTED observation —
so the radius is strictly monotone, and (c) `StateCfg.global_states` arms global search at `lost_short`.
The appearance-evidence guard is unchanged, so widening the search does not license a geometry-only latch.
Pinned by four regressions in `tests/test_target_tracker.py`.

## All-person MOT baseline on real MOT17 (detector-limited — see the recall table above)
`scripts/eval_mot.py`, 7 FRCNN sequences, 42 (sequence, tracker) pairs, `--subsample 3` -> 1772 frames,
37,424 GT dets, 0 failures, 193.6 s wall vs 747.2 s serial = **3.86x at `--jobs 4`**.  Ordering:

| tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|---|
| tracktrack | 0.3589 | 0.2644 | 0.4900 | 0.8310 | 0.2853 | 0.4179 | 184 |
| botsort | 0.3434 | 0.2618 | 0.4538 | 0.8325 | 0.2818 | 0.3857 | 275 |
| deepocsort | 0.3415 | 0.2616 | 0.4490 | 0.8329 | 0.2820 | 0.3834 | 270 |
| ocsort | 0.3254 | 0.2501 | 0.4268 | 0.8353 | 0.2619 | 0.3559 | 490 |
| bytetrack | 0.3247 | 0.2488 | 0.4269 | 0.8366 | 0.2611 | 0.3525 | 503 |
| fasttrack | 0.3114 | 0.2485 | 0.3936 | 0.8357 | 0.2602 | 0.3369 | 550 |

These are with `yolo26n@640` (recall 0.368), which is why DetA is ~0.26 everywhere and the ordering is
driven by AssA/IDSW rather than detection. NOT leaderboard-comparable: `--subsample 3`, and GT rows with
`conf=0` (official ignore regions) are dropped rather than handled as ignores. No GPU leak: allocated/
reserved stayed 32.0/32.0 MB flat across all 42 pairs (48 MB with `--embed`).

**Re-run with the shipped `yolo26m@1280` default (the numbers to quote):** 42 pairs, 0 failed,
294.6 s wall vs 1156.0 s serial = **3.92x at `--jobs 4`**, `results/mot_eval_m1280/summary.md`.
tracktrack HOTA **0.4827** / DetA 0.4415 / AssA 0.5322 / MOTA **0.4329** / IDF1 0.5860 / IDSW 346 /
105.44 ms; botsort 0.4728 / 0.4402 / 0.5129 / 0.4334 / 0.5649 / 431; deepocsort 0.4706; ocsort 0.4425;
bytetrack 0.4421; fasttrack 0.4325. Same grid with `--embed` (`results/mot_eval_m1280_embed/`):
tracktrack 0.4880 / 0.4420 / 0.5431 / 0.4306 / 0.5944 / 299 but **377.02 ms/frame (3.6x)**; botsort
0.4815; deepocsort 0.4701 (**worse** than without ReID); ocsort/bytetrack/fasttrack HOTA *identical*
to no-ReID at 329-345 ms because they do not consume appearance features. Conclusion: **ReID in the
all-person layer buys +0.005 HOTA for 3.6x cost; it belongs in the target layer.**

## Click-target measurement on real MOT17 (`scripts/run_target.py`)

**Do not trust a masked-run TSR without checking the target is observable.** `build_mask_plan`
rewrites the target's GT to the visible rectangle, so `--target-strategy stressed` can select an
identity whose post-mask box no detector ever sees: on MOT17-02, `--max-frames 60 --subsample 2`
picked id 17 = **38x54 px**, best IoU 0.000 on all 60 frames, and `TSR = 0.2333` measured an
*invisible* target, not partial-view ReID. `--target-strategy longest` picks the same id.
`scripts/pick_observable_target.py` ranks identities by post-mask visible area (under `mask=upper`: id 19
304x190 = 55,480 px2, id 3 227x160, id 23 248x134, id 17 54x38 = 2,071 px2).

Valid grid — `--target-id 19`, 11 variants x 3 masks = 33 runs, 0 errors, real `yolo26m@1280` + real
ReID, 300 frames, `results/target_obs19/`:

| variant | none | upper | lower |
|---|---|---|---|
| baseline | 0.9236 | 0.7527 | 0.9236 |
| gate_legacy / gate_search_late | 0.7891 | 0.6036 | 0.9236 |
| gate_growth_unfrozen | 0.9236 | 0.7527 | 0.9236 |
| margin_0.00 / margin_0.12 | 0.9236 | 0.6945 / 0.7236 | 0.9236 |
| parts_full_only | 0.9200 | 0.6691 | 0.7745 |
| part_matching_off | 0.9236 | 0.7527 | 0.9236 |

1. The **early global search** gate fix is worth **+13.5 pp (none) / +14.9 pp (upper)**; the
   frozen-speed growth term alone changes nothing on all three masks — do not credit both.
2. **Part banks earn their cost**: `parts_full_only` costs 8.36 pp (upper) and 14.91 pp (lower), with
   more ID switches (8 vs 6) and fragments (8 vs 3) on `lower`, for only 0.36 pp when nothing is
   occluded. This REVERSES the earlier synthetic `parts_full_only` win (that run used a GT-oracle
   detector + fallback descriptor against the invisible target, and is not evidence).
3. **`distractor_margin = 0.06` is right on real data**: 0.00 raises FAR (0.0100 vs 0.0067 none;
   0.0100 vs 0.0000 upper), 0.12 costs 2.9 pp TSR on `upper` for no FAR gain. The synthetic grid had
   suggested 0.12 was far better — it is not.
4. **`part_matching_off` is inert** (identical to baseline to 4 decimals on all masks) while
   `target_parts` clearly is not: the gallery's part *banks* do the work; cost-side part routing never
   fires because `estimate_partial_state` only detects FRAME-EDGE truncation — a person whose upper
   body is blacked out mid-frame still looks like a small complete person. Improvement direction:
   derive partial state from the crop's own structure energy.
5. **TSR is not evidence of observation.** Baseline hallucination_rate 0.1467 (none) / **0.3100**
   (upper) / 0.1433 (lower); `parts_full_only` 0.3867 (upper) / 0.2733 (lower); `margin_0.00` 0.3533
   (upper). Quote "holds the identity 0.75-0.92 of frames, observing it ~70% of the time", not 0.92.

### task-10 harness guard (eval-dev) — the shipped measurement path
`scripts/run_target.py`: `--min-visible-h` (default **80 px**) / `--min-visible-area` select using the
**median post-mask visible box** across every `--mask` mode; `--target-id` still wins (`forced`).
`--min-visible-h 3000` -> **exit 3, no runs started**, with `[guard] ... largest candidates: ...`.
`summary.json` gains `guard`, `targets`, `n_headline_runs`, `n_invisible_runs`, `n_errors`, and per
sequence+mask `median_w/h/area`, `frames`, `fraction`, `detect_rate`, `guard_passed`, `match_mode`
(`detector` = measured against real detections, `plan` = GT-oracle, so plan numbers cannot masquerade).
`--allow-invisible-target` / `--target-strategy invisible` runs the unobservable case as a **labelled
control** (`control="invisible"`, separate table, never pooled with the headline). Two new metrics:
`wrong_identity_locked` (confident lock on another identity at IoU >= 0.1 — what `false_acquisition`
misses because it needs IoU >= 0.5) and `coasting_correct_frames` (frames booked correct while blind
with nothing accepted).

## Known open hole: identity handover to a lone similar bystander (measured, not hidden)
`scripts/ablate_motion.py`, `unidentifiable_target` family, identical-appearance bystander:
- bystander present, target INVISIBLE -> `target_success_rate` **39.2% / false_acquisition 54.6%**
- target alone -> `58.7% / 0.0%`
- handover occurs on the FIRST frame the target is unreadable, while the state is still `tracking`
  with `fsu = 0`; median accepted margin **0.88-1.11**, i.e. 15x `distractor_margin`, so the margin
  test cannot see it.
- Aggregate FA: 0.0% for appearance similarity <= 0.92, 7.6% at margin 0.12 for sim 0.97/0.99, and
  6.7-19.2% for sim 1.00. Median margin (runner-up - best_cost) = 1.42/0.61/0.20/0.12/0.08/0.07/0.057
  for sim 0.00/0.60/0.85/0.92/0.97/0.99/1.00 -> the shipped 0.06 sits exactly ON the refusal cliff.
  2x2 peel control shows the swap is caused by the REFUSAL (coasting onto the bystander), not an
  appearance mis-decision. Recommendation: keep 0.06.
- ROOT CAUSE 1 (`src/tgtrack/association.py`): absent evidence is priced
  `missing_appearance_cost = 0.5`, i.e. **0.8 of the 1.0 accept budget**, while a genuinely different
  person at cosine 0.0 is priced 1.6 — "we could not read this person" scores BETTER than "this is
  someone else". Now a real `CostCfg` field (default 0.5, unchanged).
- ROOT CAUSE 2 (`select_candidate`): with ONE finite candidate the margin stays `inf`, so
  `margin < margin_thr` is **vacuously false** — the ambiguity test cannot refuse a lone candidate no
  matter the threshold. This bites exactly when the bystander is the only detection.

**MEASURED FOLLOW-UP (motion-dev, `--unid-cost`, `results/motion_ablation/unid_cost.{md,csv}`):**
- `handover_split` over a 60-run pricing grid: **n=1 at 36 of 36 handovers, n=2 at 0** (12 runs never
  handed over). Every handover happened in the GAP, not the blind window. So Hole 2 is not merely
  dominant — it is the *only* hole exercised at the handover frame, and no `distractor_margin` value
  can touch it.
- The pricing sweep is a **FLAT NULL**. `missing_appearance_cost` in {0.5, 1.0, 1.2} gives identical
  results for kf and imm: TSR 29.5% / FAR 76.9% / 60 FA frames / id_consistency 0.23 / handover at
  frame 18. The only quantity that moves is the median ACCEPTED margin in the blind window
  (0.88 -> 1.68 -> 2.00). The price changes the cost surface but never flips an accept/reject
  decision, because the decision only needs "best candidate" and there is exactly one.
  **Do NOT raise `missing_appearance_cost`** — it buys nothing here. Left at 0.5.
- Frame-18 trace (`--trace unidentifiable_missing_s060 --trace-spec kf-shipped`, columns best_cost /
  thr / runner_up / accepted / iou_target / iou_bystander): f15-f17 n_dets 2, best 0.29, runner_up
  1.21/1.13/1.02, ACCEPT, iou_target 1.00; **f18 n_dets 1, best 0.90, runner_up "-", ACCEPT** (target
  stops being detected, iou_bystander 1.00) <- the handover; f34 best 0.61 iou_target 0.05; f36 n_dets
  2, best 0.61 vs runner_up 1.49 -> **once latched the TRUE target is 2.4x MORE expensive than the
  bystander**, so no later frame can outbid the latch (hence id_consistency 0.23, FAR 76.9%).
- Calibration: legitimate lone-candidate accepts in the no-bystander control cost **0.02**; the
  handover accept costs **0.90**; the sustained latch 0.61-0.63. A lone bound near **0.4** separates
  them with ~2x headroom on both sides.
- New knob: **`CostCfg.lone_candidate_accept_cost`**, applied in `select_candidate` whenever exactly one
  finite candidate exists. Two implementation facts matter, and both were WRONG at first:
  1. **It was exempted while lost** (`if require_margin and not lost:`), i.e. skipped exactly when the
     tracker goes blind — which is where every handover happens. That exemption is what reduced the
     guard to a 2-frame deferral (handover 18 -> 20) and made the first sweep read as a flat null.
  2. **It must require readable appearance.** The refusal also fired when the lone candidate's
     appearance could not be read at all (a placeholder `missing_appearance_cost`), so a transient
     encoder failure killed tracking instead of protecting it. The bound now applies only when
     `appearance_valid` is True: it bounds EVIDENCE and refuses to reason about a placeholder.
  Both are pinned by tests (`test_lone_candidate_bound_refuses_a_lone_stranger_but_not_the_target`,
  `test_encoder_failure_does_not_kill_tracking`).
- **THE SWEEP, AND WHY IT IS SHIPPED OFF.** `results/motion_ablation/lone_bound.md` re-run with
  a genuine off-control (100 runs, 7,800 frames, 0 errors; `None` now sets an EXPLICIT override to None,
  because "no override" had silently become a second copy of the treatment once the default went live).
  Unidentifiable family, per mode/model: **off -> 3/4 runs hand over, all at `n_dets=1` (the lone
  hijack), frame 18, peak false acquisition 76.9%**; bounds 0.2/0.3/0.4 -> **`n_dets=1: 0`**, every
  survivor a frame-36 PRICED comparison, peak FA 75.6%; bound 0.5 -> one lone hijack survives in 1 of 4
  cells (frame 31). So the lost-path fix converts a lone hijack into a priced comparison — it does not
  prevent the handover, and aggregate false acquisition barely moves (76.9% -> 75.6%).
- COST: clean-family success with the rule off is **98.9%** -> 96.3% at 0.5 (**-2.6 pp**), 93.5% at 0.4
  (-5.4 pp), 88.1% at 0.3, 85.7% at 0.2. Clean false acquisition stays 0.0% and id_consistency 1.000 at
  every bound; the solo control (no bystander) is 55.1% / FAR 0.0% at every bound including off.
- Decision: **the shipped default is `None` (off)**, decided on real data. `results/target_obs19_shipped` vs
  `results/target_obs19` — identical command (MOT17-02-FRCNN, `--target-id 19`, 300 frames, `--subsample 2`,
  masks none/upper/lower): the bound costs **3.6 pp of target success on the upper-body-only mask**
  (0.7527 -> 0.7164), **0.0 pp** on `none` and `lower`, and moves false acquisition **not at all**
  (0.00667 / 0.0 / 0.01333 both ways). Real cost, zero real-data benefit, synthetic-only win -> the knob,
  the sweep and the negative result are kept and the default is off. Set it to a float to enable.
- Reclassified after the default went back to `None`: the stub-world low-side closure only appears once the
  lone rule is ARMED. With the rule off, price moves the RECOVERY but never the handover — every price hands
  over at frame 8 with `n_dets=1` (`test_pricing_the_absent_evidence_moves_recovery_not_the_handover`).
- Promising but NOT yet validated: `scripts/ablate_motion.py`'s `MISSING_COSTS` axis now also samples the
  LOW side `(0.0, 0.1, 0.25, 0.5, 1.0)`, because raising the price provably cannot help (it makes the
  unreadable TRUE target more expensive). On the 30-frame stub world, with the bound on at 0.5, prices
  <= 0.25 close the handover entirely (TSR 0.4333 -> 0.9667, FAR 0.7333 -> 0.2333). That is one small
  synthetic world, not the 10-scenario grid — do not quote it as a result.
- `CostCfg.distractor_margin_lost` (default **0.0** = inert) is available to tighten the margin while
  lost but is NOT validated; leave at 0.
- HONESTY NOTE kept: the synthetic `unid` calibration (stub embedder) and the live MOT17 id-17 rebind
  (real margin 0.0635, n_dets >= 2) are **two different mechanisms**; the guard is calibrated on the
  synthetic family and is not claimed to fix the live low-IoU rebind.

## Conventions
- Write artifacts under `C:\Projects\track`. Source in `src/`, configs in `configs/`, scripts in `scripts/`,
  results in `results/`, docs in `docs/`, data in `data/`.
- Every claim about performance must come with a command that reproduces it and the measured number.
- Use `$env:PYTHONIOENCODING="utf-8"` in PowerShell before python.

## Research programme status (E019-E022, round 2-3)

- **A *stale baseline* was found and killed.** `results/dancetrack_target` (FAR 0.49) predates
  the current `association.py`/`motion.py`; the fresh, target-pinned measurement over the three
  DanceTrack sequences is `results/e019_dancetrack000*`.  Rule: a baseline older than the newest
  source file is not a baseline.
- **DanceTrack criterion (FAR < 0.20, TSR >= 0.55):** pooled TSR **0.544**, pooled FAR **0.197**
  with the shipped `search_margin = 0.20` (`results/e019_*` baseline vs `results/e022_dancetrack*`).
  FAR met; TSR is 0.006 short; 0005 is already 0.995/0.000 and the whole gap is 0004/0007.
- **Partial-view criterion (upper-body TSR +10 pp):** +2.91 pp so far (0.7527 -> 0.7818 with the
  E016 velocity clamp, `motion.velocity_clamp_ratio = 3.0`, shipped).  `search_margin = 0.20`
  moves it to 0.7782 -- do not spend it twice, quote 0.7818 for the clamp and note the margin's
  -0.36 pp.
- **Detector cap:** E010 proved tracker output gating costs only 1-3 % of recall (emitted 0.527 vs
  matched 0.556 on MOT17-02); every detection-side route was measured and rejected (E001 tiling,
  E002 thresholds, E003 bigger/higher-res model, E005 SPRT promotion, E011 ensemble).  All-person
  MOT17 HOTA is **0.4980** with the ignore convention and cannot reach 0.55 on this detector.
- **Where the programme is stuck, precisely:** the FAR/TAS trade is now a *single-sequence* problem
  (dancetrack0007), and the loss there is not "we cannot find the target" (139/141 wrong frames had
  the target detected, E020) but "the memory prefers the wrong person" -- i.e. the next lever is
  gallery hygiene / long-term bank partitioning, not another association knob.
