# R2 — ReID, Partial/Occluded Matching, Long-Term Gallery, Distractor Handling, Face Cues, Cost

**Status:** COMPLETE (2026-10-24). §1 = verified local facts; §2–§7 = literature + critique/measurements of the *existing* `src/tgtrack` design; **§10 = independent second-pass addendum** (measured replacements, new defects, canonical-foundations ladder, one-shot/distractor evidence, gallery-policy questions). Where §10 contradicts §1–§9, the measurement wins.
**Scope:** critique/measure the *existing* design (see `docs/BRIEF.md`) — no re-proposing the architecture.
**Rule:** anything not verified from a primary source or a local measurement is marked **UNVERIFIED**.

Local measured facts carried in from the Lead (treated as given):
- `yolo26n-reid.onnx`, 9,873,245 B, input 224x224, output **512-d**.
- RTX 4060 Laptop 8 GB: bs1 **2.74 ms**, bs40 **12.7 ms** (0.317 ms/crop) CUDA.
- Occlusion probe (lower body occluded), cosine sim to *same person's* full-body gallery entry:
  `full` 0.757, `upper` **0.920**, `head` **1.0**, `torso` 0.941, `lower` 0.429;
  different-person `upper` = 0.48.

---

## 1. What `yolo26*-reid.onnx` actually is

### Verdict (confidence: HIGH — read straight off the shipped file)

**It is a purpose-trained person-ReID model: a YOLO26n classification-style backbone
("`YOLO26n-reid-2psa`") trained on MSMT17 with an identity-classification + BNNeck objective,
emitting an L2-normalized 512-d embedding.** It is *not* a generic feature extractor, and *not*
an OSNet/CLIP-ReID encoder. It was trained by Ultralytics themselves as a first-class `reid` task.

### The evidence — ONNX `metadata_props` (verbatim)

Reproduce with:

```powershell
$env:PYTHONIOENCODING="utf-8"; python -c @"
import onnx
m=onnx.load(r'C:\Projects\track\data\weights\yolo26n-reid.onnx')
for p in m.metadata_props: print(p.key,'=',p.value[:200])
"@
```

| key | value |
|---|---|
| `description` | **`Ultralytics YOLO26n-reid-2psa model trained on ultralytics/cfg/datasets/MSMT17-v1.yaml`** |
| `task` | **`reid`** |
| `author` | `Ultralytics` |
| `version` | `8.4.62` (exported with this package version; our installed wheel is 8.4.185) |
| `date` | `2026-06-10T13:55:57` |
| `imgsz` | **`[448, 448]`** |
| `stride` | `1` |
| `end2end` | `False` |
| `names` | `{0: 'class_0', … , 1040: 'class_1040'}` — **1041 identity classes** |
| `args` | `{'dynamic': True, 'simplify': True, 'nms': False, 'half': False, 'int8': False, 'opset': None}` |

`producer_name = pytorch`, `producer_version = 2.12.0`, `ir_version 9`, opset **20**,
graph name `main_graph`, input name **`images`**, output name **`embeddings`**.

**Independently re-verified** against the complete file `data\weights\yolo26n-reid.onnx`
(9,873,245 B, **md5 `6D01B3F9CD3F166A4FDCA8CC3EC3039F`**) with `onnx` 1.23.2 — every value in the
table above reproduces exactly, plus these fields the earlier pass omitted:

| key | value | why it matters |
|---|---|---|
| `license` | **`AGPL-3.0 License (https://ultralytics.com/license)`** | **project-level constraint**: AGPL-3.0 attaches to the weights; distributing a networked product built on this encoder triggers AGPL source obligations. Not a technical detail — a licensing decision the Lead should record. |
| `batch` | `1` | declared batch is 1, yet the input dim is symbolic — `reid.py` batches anyway, which is why the dynamic-input bug below matters so much. |
| `channels` | `3` | RGB after `reid.py`'s BGR→RGB flip. |
| `args` | `{'data': None, 'batch': 1, 'fraction': 1.0, 'half': False, 'int8': False, 'dynamic': True, 'simplify': True, 'opset': None}` | `half=False, int8=False` confirms the file is **fp32** — independently of the param arithmetic. `dynamic: True` is the root cause of the 224-vs-448 defect. |

Also confirmed: `Gemm` weight `model.11.embed.weight` has dims **`[512, 1280]`** — so the penultimate
feature is **1280-d** and the embedding is genuinely **512-d**, not merely declared so; and the output
tensor is `embeddings: ['batch', 512]`, i.e. the **batch axis is dynamic too** (relevant to the ONNX
fixed-batch chunking path in `reid.py`).

**1041 = exactly the MSMT17 training-split identity count.** Independent corroboration of the
`description`: the model is a 1041-way identity classifier, i.e. it was trained to discriminate
*person identities*, on MSMT17 (the standard large-scale ReID benchmark: 1,041 train IDs /
4,101 total IDs, 15 cameras, ~126k images). This is the single most important fact in this report:
**the embedding space is a metric space learned for person identity, so cosine similarity on it is
semantically meaningful for our task.**

### Architecture — from the ONNX node names

201 nodes, 112 initializers, **2,432,284 parameters** in initializers
(2,432,284 × 4 B ≈ 9.73 MB ≈ the 9,873,245-byte file → **fp32, not quantized, not fp16**).

Node-name prefixes are the original torch module paths, `/model.<i>/…`, and they are unambiguous:

- `/model.0/conv/Conv` → `/model.0/act/Sigmoid` → `/model.0/act/Mul` — the usual YOLO
  **SiLU = x·σ(x)** pair, repeated throughout.
- `/model.2/Split`, `/model.2/m.0/cv1/…`, `/model.2/m.0/Add`, `/model.2/Concat`, `/model.2/cv2/…`
  — the **C3k2** block of YOLO11/YOLO26 (a `Split` feeding bottleneck `m.0`, then `Concat`).
- Op histogram: `Conv 46, Mul 42, Sigmoid 38, Add 13, Concat 12, Split 8, MatMul 4, Softmax 2`
  → a plain convolutional backbone with **no transformer encoder and no InstanceNormalization**
  (so: **not OSNet** — OSNet is InstanceNorm-based; **not CLIP-ReID / TransReID** — those are ViT).
  The `Softmax 2` are the internal attention softmaxes of the `2psa` block.
- The `2psa` in the name = the **C2PSA** (position-sensitive attention) stage that YOLO26 puts after
  the SPPF stage — consistent with `yolo26-cls.yaml` line `- [-1, 2, C2PSA, [1024]] # 9`.

**Head (`/model.11/*`), in execution order** — this is the ReID-specific part and it is a
textbook "Bag of Tricks" head:

```
/model.11/conv/conv/Conv        1x1 conv (no stride)
/model.11/conv/act/Mul          SiLU
/model.11/pool_avg/GlobalAveragePool
/model.11/pool_max/GlobalMaxPool
/model.11/Add                   GAP + GMP  (concat-style dual pooling, summed)
/model.11/Flatten
/model.11/embed/Gemm            -> 512-d
/model.11/bottleneck/BatchNormalization      <-- BNNeck
/model.11/ReduceL2 -> Clip -> Expand -> Div  <-- L2 normalization
```

Two consequences that matter operationally:

1. **The 512-d output is already L2-normalized** inside the graph (`ReduceL2` then divide). Cosine
   similarity is therefore just a dot product; re-normalizing is harmless but redundant.
2. **`bottleneck/BatchNormalization` after the embedding `Gemm` is a BNNeck** — the
   Luo et al. *Bag of Tricks* (arXiv 1903.07071) device that whitens the embedding before the
   identity classifier. Its presence is direct evidence of a modern, deliberately-tuned ReID
   training recipe rather than an incidental byproduct.

### Network / docs corroboration

- `ultralytics/trackers/utils/reid.py` (local wheel, 8.4.185, `reid.py:21`) defines
  `REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")` — i.e. **Ultralytics ships
  these as first-class tracker ReID assets** for `n/s/m/l/x`, auto-downloaded by
  `attempt_download_asset()`; see [ultralytics/trackers/utils/reid.py](https://github.com/ultralytics/ultralytics/blob/d772bbae16c90494fc92d632eb22e2d105f66910/ultralytics/trackers/utils/reid.py)
  and the commit [Tracker ReID ONNX encoders autodownload (#24774)](https://github.com/ultralytics/ultralytics/commit/ab4abcab0d2af0e0c1e2b9cf5ce95127ce72df89).
- The module docstring calls it a *"Shared ReID encoder used by BoT-SORT, Deep OC-SORT, and
  TrackTrack"*, and documents the `.pt` alternative as pulling embeddings from the second-to-last
  layer (`model.embed([len(model.model.model) - 2])`) — i.e. `model="auto"` means "reuse the
  **detector's** backbone features", which is a *weaker, detector-objective* feature, not a ReID
  feature. **Do not use `model="auto"` and assume it equals the dedicated ReID model.**

> **Could not verify:** the exact backbone width/depth multipliers, the MSMT17 subset actually used,
> the training recipe hyper-parameters (epochs, loss weights), and whether `MSMT17-v1.yaml` on
> `ultralytics/cfg/datasets/` is a re-split of MSMT17 or the stock split. That config file is
> **not shipped in our wheel** (only `trackers/utils/reid.py` mentions reid; there is no
> `cfg/datasets/MSMT17*` and no `cfg/models/26/yolo26-reid-2psa.yaml` locally), and the
> corresponding docs page could not be fetched from this machine. Class count 1041 matching MSMT17
> train IDs is strong but indirect evidence for "stock MSMT17 train split".
> **UNVERIFIED:** published mAP/Rank-1 for this specific model — Ultralytics publishes no ReID
> benchmark table that I could retrieve.

### ⚠️ The one concrete defect I found in the current build

`ReID.__init__` (`reid.py:65-74`) only overrides its crop size when the ONNX input dims are
**static ints**:

```python
shape = session.get_inputs()[0].shape if session is not None else ()
if len(shape) == 4:
    if isinstance(shape[0], int) and shape[0] > 0:  self.batch_size = shape[0]
    if isinstance(shape[2], int) and shape[2] > 0:  self.imgsz = (shape[2], shape[2])
    if isinstance(shape[3], int) and shape[3] > 0:  self.imgsz = (self.imgsz[0], shape[3])
```

Our file's input is **fully dynamic** — `IN [('images', ['batch', 3, 'height', 'width'])]`, i.e.
all four dims are strings. So no branch fires and `self.imgsz` stays at the **constructor default
`imgsz=224`** (`reid.py:41`). The model was trained at **448×448**. **The current pipeline is very
likely feeding it 224×224 — half the linear resolution it was trained at.** See §6 for the
measured cost of that mismatch. This is a cheap, no-fine-tuning, one-line fix
(`ReID(model, imgsz=448)`) worth testing before any architectural work.

*(BRIEF.md line 21 records "512-d output, 224x224" as the measured I/O shape. 512-d is confirmed;
the "224x224" is the **ultralytics default, not the model's native size** — resolve before relying on it.)*

---

## 2. Occluded / partial ReID: what concretely works

### 2.1 Benchmarks (definitions, so the numbers mean something)

| Dataset | Occlusion/cloth protocol | Train | Query | Gallery |
|---|---|---|---|---|
| **Occluded-DukeMTMC** | 9 % of train, **100 % of query**, 10 % of gallery occluded; 8 camera views | 15,618 imgs / 702 (708) ids | 2,210 / 519 ids | 17,661 / 1,110 ids |
| **Occluded-ReID** | mobile camera, 5 holistic + 5 occluded per id | — | 2,000 imgs / 200 ids | 200 ids |
| **Partial-ReID** | partial/occluded query vs holistic gallery | 600 imgs / 60 persons | partial | holistic |
| **P-DukeMTMC-reID** | partial bodies | — | — | — |
| **PRCC** | cloth-changing (CC) subset vs same-clothes | 33,698 / 221 ids | 3,384 / 221 | 33,698 |
| **LTCC** | long-term cloth change, 12 cloth categories | 17,138 / 77 ids | 3,296 / 77 | 9,924 |

Occluded-Duke repо: [lightas/Occluded-DukeMTMC-Dataset](https://github.com/lightas/Occluded-DukeMTMC-Dataset).
The original partial-ReID problem statement ([ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.html))
names our two failure modes exactly: (i) *"with only partial information it is less discriminative thus
more likely to be mismatched to the wrong person"* — *"if only a pair of blue jeans are visible, it is
difficult to use them to distinguish people"*; (ii) *"it is difficult to determine against which part of
the full body the partial observation should be matched"*. Its AMC-SWM answer — local patch matching with
explicit **patch-ambiguity modelling** — is the ancestor of everything below.

### 2.2 Occluded-DukeMTMC leaderboard (Rank-1 / mAP)

Column order verified per page with known anchor pairs (PGFA 51.4 R1 / 37.3 mAP; HOReID 55.1 / 43.8;
TransReID 66.4 / 59.2 vs the TransReID README's 67.4 / 59.5):

| Method | Venue | **Rank-1** | **mAP** | Mechanism that produces the gain |
|---|---|---|---|---|
| **KPR + SOLIDER** | ECCV 2024 | **84.31** | **75.11** | keypoint prompts + part attention + visibility |
| KPR | ECCV 2024 | 79.83 | 67.13 | same, weaker backbone |
| PersonViT | 2025 | 79.8 | 72.2 | masked-image + pixel-level pretraining |
| DROP | 2024 | 76.8 | 63.3 | — |
| OGFR | TIP 2025 | 76.6 | 64.7 | — |
| **TRGA** | Multimedia Systems 2026 | 73.8 | 63.9 | descriptor deletion + **logical masking** (full token set retained, progressive recovery) + hardest-negative augmentation |
| **MAMS** | Sensors 25(19):6210 | 73.3 | 62.9 | multi-aligned multi-scale augmentation, **no auxiliary models** |
| SOLIDER | CVPR 2023 | 71.2 | 61.9 | human-centric semantic pretraining |
| DiP (w/o RK) | 2024 | 71.1 | 63.1 | — |
| MBIN | Sensors 25(21):6526 | 71.2 | 59.1 | multi-branch interaction |
| HCGA | TIP 2022 | 70.1 | 57.5 | — |
| PFD | 2023 | 69.5 | 61.8 | pose-guided feature disentangling |
| TAT | 2023 | 68.2 | 60.6 | template-aware transformer |
| **FED** | **CVPR 2022** | **68.1** | — | erase occluder + diffuse target features |
| TransReID | ICCV 2021 | 66.4 | 59.2 | ViT + JPM (occlusion-blind) |
| LDS | 2023 | 64.3 | 55.7 | — |
| **ISP** | ECCV 2020 | 62.8 | 52.3 | human semantic parsing, 5 body parts |
| HG | CVPR 2021 | 61.4 | 50.5 | teacher-student, **no pose supervision** |
| MHSA-Net | 2022 | 59.7 | 44.8 | mask prediction + attention |
| **HOReID** | CVPR 2020 | 55.1 | 43.8 | high-order pose/proximity graph matching |
| **PGFA** | ICCV 2019 | 51.4 | 37.3 | pose-guided alignment + visibility |

Occluded-ReID: **Pose2ID 91.00 R1 / 89.34 mAP**, MAMS 87.3 / 82.1, FED 87.0 R1 (*"86.3 % Rank-1 on
Occluded-REID, surpassing others by at least 4.7 %"*), KPR 85.42 / 79.1, BPBreID 82.9 / 75.2.

### 2.3 The four mechanism families

**(a) Visibility prediction.** **PVPM** (CVPR 2020) is the canonical design: pose-guided attention pools
part features; a four-layer network predicts per-part visibility `v̂ = PVP(F_pose; θ_v) ∈ R^{N_p}`;
because there are no visibility labels, **graph-matching correspondence scores are used as pseudo-labels
to self-supervise the predictor**, and at test time only mutually visible parts are compared. Most
portable idea in the literature for us: pose confidence is already a visibility proxy.

**(b) Occluder erasing + diffusion.** **FED** (CVPR 2022) separates *Non-Pedestrian Occlusions* from
*Non-Target Pedestrians*. OEM erases NPO features, trained by painting occluders onto holistic images
(which also yields precise occlusion masks); FDM then synthesizes NTP characteristics through
cross-attention whose keys/values are the K nearest **different-identity** memory centres (momentum 0.2),
attention-masked by the OEM scores so diffusion touches only visible parts. **FDM is training-only → zero
inference cost.** Implementation: 256×128, ViT, 128 tokens, dim 768, patch 16, 4 horizontal stripes.
*(FED's HTML swaps Market-1501's query/gallery counts — author typo, do not cite.)*

**(c) Learned masks instead of parts.** **DPM++** (arXiv:2605.06637, May 2026) reframes the task:
occluded ReID *"should not be treated merely as a harder version of holistic ReID… it should be revisited
as a PARTIAL-TO-HOLISTIC MATCHING problem"*, solved by *"dynamically generating an image-specific mask and
selecting a visibility-consistent discriminative subspace, **without requiring extra body-part supervision
or pre-trained auxiliary networks**"*. Machinery: learnable prototypes `P ∈ R^{K×c}`, per-image mask
`M_p ∈ R^{1×c}`, masked prototypes `P_m = P ⊙ M_p`, and a CLS token supervised by a frozen CLIP-derived
coarse prototype matrix (`t_y = [A photo of a X1 … XM person]`, learnable tokens) jointly with `P` and
`P_m`. Plus saliency-guided patch transfer with an **occlusion-aware IoU**
`OIoU(M_i,M_j) = |M_i ∩ M_j| / |M_j|` (denominator = *candidate* mask ⇒ prefers compact occluders) and
horizontal mask rolling. **Its numeric tables were not retrievable — the SOTA claim is UNVERIFIED.**

**(d) Augmentation and attention alone.** **MAMS** reaches 73.3 R1 / 62.9 mAP with multi-scale
augmentation and **no auxiliary models at all** — *better than FED*. **HG** (CVPR 2021) works with **no
pose supervision**. So a large part of the occluded-ReID gain is augmentation + attention, not exotic
modules: **do not assume a pose estimator is required to improve.**

### 2.4 Verdict on the *existing* design (`src/tgtrack/parts.py`)

The implementation is: **fixed fractional rectangles** off the person box
(`types.py:21-35`: `full` 0.00–1.00, `upper` 0.00–0.60, `lower` 0.45–1.00, `head` 0.00–0.28,
`torso` 0.15–0.55), cropped with geometry-preserving zero-canvas padding (`parts.py:140-205`), and scored
by `crop_quality = sqrt(area_score · sharp_score)` where `area_score` saturates at 64×128 px and
`sharp_score` is the variance of the Laplacian saturated at 250 (`parts.py:208-255`).

Three concrete, fixable gaps, in priority order:

1. **`crop_quality` has no occlusion term, and this is provably the failure mode.** A part crop that is
   100 % covered by *another person* is large and sharp ⇒ `quality ≈ 1.0` ⇒ it is inserted into the
   long-term gallery and weighted highly. The Lead's own probe contains the proof:

   > lower body occluded, cosine sim to the *same person's* full-body gallery entry:
   > `full` 0.757, `upper` **0.920**, `head` **1.0**, `torso` 0.941, `lower` **0.429**;
   > *different-person* `upper` = **0.48**.

   **The same person's occluded part (0.429) scores BELOW a different person's visible part (0.48).**
   Any matcher that averages over *all* parts, or that trusts a high-`crop_quality` occluded crop, will
   prefer the wrong person. This is precisely why PVPM/BPBreID/KPR compare **only mutually visible
   parts**, and it is a local measurement, not an argument from authority.
   **Cheap fix that needs no new model:** the pipeline already detects *all* persons (BRIEF pillar 1), so
   occlusion is computable for free as the containment of the part box in any *other* detection's box:
   `vis(part) = 1 − max_{j≠target} area(part_box ∩ box_j) / area(part_box)`, multiplied into
   `crop_quality`. This is a direct analogue of PVPM's `v̂` and DPM++'s `M_p` at ~zero cost.
2. **`part_boxes` is only geometrically valid for an un-truncated box.** `parts.py:73-117` applies
   `PART_Y_RANGES` as fractions of the *given* box, while `estimate_partial_state`
   (`parts.py:258-361`) separately detects frame clipping, a short box (`h < 1.8·w`) and
   `upper_only`/`lower_only`. Nothing connects them: for an upper-body-only box, `part_boxes(...,"lower")`
   returns the bottom 55 % **of the torso** and it will be banked *as legs*. The module docstring promises
   "part geometry (leg length, head width) is preserved" — that holds for the *crop*, not for the *part
   assignment*.
   **Fix:** when `est_visible_frac < ~0.8`, extrapolate a synthetic full-body box from the estimated body
   aspect (`_PERSON_ASPECT = 2.7` is already in the file) and compute part ranges against *that*, so
   `lower` means legs (and can then be correctly marked invisible).
3. **Zero-canvas padding feeds black pixels to the encoder.** Clipped crops are placed on a black canvas
   (`parts.py:196-198`) so a half-off-frame crop is ~50 % zeros. The *quality* term correctly uses real
   pixels only (`parts.py:252` uses the intersection), but the *embedding* still sees the black region.
   Worth an ablation: crop-and-resize the visible part instead, or zero out with a mask-aware encoder.

Also note what is **right** and should not be churned: `PART_KEYS` already matches the BRIEF's required
part set (upper / lower / head-face / torso / body); the fallback chain and the "part key is validated
everywhere" discipline are good; and the fixed-fraction design is defensible on the literature
(MAMS/HG get far without pose), so **pose should be an A/B experiment, not a rewrite**.

### 2.5 Pose, if you take it (it is nearly free)

| Tool | Size | Accuracy | Documented latency |
|---|---|---|---|
| RTMPose-s | 20.88 MB ONNX | 72.2 % AP | 70+ FPS Snapdragon 865 |
| **RTMPose-m** | **51.81 MB** | **75.8 % AP COCO** | **90+ FPS i7-11700 CPU, 430+ FPS GTX 1660 Ti** |
| RTMPose-l | 105.48 MB | 67.0 % AP COCO-WholeBody | 130+ FPS |
| RTMPose-x / m-hand / m-wholebody | 188.5 / 52.53 / 68.52 MB | — | — |
| YOLO-pose (yolo26-pose.pt) | n 7.51 → x 120.39 MB | ultralytics pose metrics (not verified here) | same runtime as the detector |

([arXiv:2303.07399](https://arxiv.org/abs/2303.07399); sizes measured from `hf-mirror.com/bukuroo/RTMPose-ONNX`).
**430+ FPS on a GTX 1660 Ti — four GPU generations below a 4060 Laptop — means pose costs ~1–3 ms/frame.**
HRNet-W32 params/AP/FPS were **not** verified this session — flagged.

If pose is added, the literature's rule (BPBreID/KPR) is the matching rule, and it is *the* mechanism to
bet on: **average the cosine distance over all parts that are visible in BOTH the query and the gallery
entry, requiring ≥2 shared parts** — not a single part's bank, and not a global descriptor.

---

## 3. Long-term gallery policy

### 3.1 What `src/tgtrack/memory.py` already gets right

- `ShortTermMemory` reproduces `smooth_feature` exactly (L2-normalize, `smooth = α·smooth + (1−α)·feat`,
  re-normalize) and **rejects unusable crops entirely** rather than corrupting the EMA
  (`memory.py:143-148`), plus a zero-norm guard (`memory.py:156-158`).
- `LongTermGallery.add` (`memory.py:296-346`) has the exact gate order the literature recommends: frozen →
  unusable → `quality < ltm_min_quality` → **diversity gate** (reject if cosine distance to the nearest
  *same-part* entry ≤ `ltm_diversity_thresh`, i.e. a near-duplicate adds no new mode) → evict and insert.
- It is explicitly a **set of distinct appearance modes**, not a bag of raw frames — this is Pr2R's
  "condensed prototypes, not raw exemplars" lesson, arrived at independently.
- `freeze(True)` makes inserts no-ops (BRIEF's "freeze the bank while out-of-frame"), `snapshot`/`restore`
  give experiment rollback, and there is a smooth hard-negative penalty (`memory.py:439-455`) instead of
  a binary reject.
- Defaults (`config.py:120-140`): `stm_alpha 0.9`, `stm_len 64`, `stm_stale_frames 90`,
  `ltm_capacity 96`, `ltm_diversity_thresh 0.15`, `ltm_min_quality 0.35`, `ltm_half_life 600.0`,
  `ltm_replace_policy "quality_recency"`, `part_aware True`, `hard_negative_bank True`,
  `hard_negative_capacity 64`. Cost weights: `ltm_weight 0.4`, `hard_negative_penalty 0.35`.

### 3.2 Five gaps against the literature

1. **Parts are an OR, not an AND — the biggest gap.**
   `LongTermGallery._candidates(part)` (`memory.py:285-293`) restricts a `part="upper"` query to the
   `upper` bank (falling back to `full`). So **one part decides the identity**. The literature's rule
   (BPBreID/KPR: *"average distance of all mutually visible part-based embeddings"*; PVPM: compare only
   *mutually visible* parts) is **multi-part evidence fusion**. Fix: add a fused query that aggregates
   per-part best distances (quality/visibility-weighted mean or median) and requires ≥2 parts to agree
   before a match is accepted. This is a pure strategy change — exactly what the BRIEF says to optimize.
2. **No membership test on insert.** The diversity gate (`memory.py:322-327`) checks *novelty* (distance to
   the nearest same-part entry) but never checks that the new feature is *consistent with the target at
   all*. A distractor that gets mis-associated once can therefore be banked permanently. The open-set
   recurring-gallery work ([IEEE 9506241](https://ieeexplore.ieee.org/document/9506241)) does the opposite:
   an **outlier-based decision rule for inserting new identities**, re-run on match, with the matched
   descriptor **replaced** by the query descriptor. Fix: require `dist(feat, target_STM_EMA) ≤ τ_insert`
   in addition to novelty, and adopt the "re-run the outlier test on match, then replace" policy.
3. **Eviction ignores `use_count` and part coverage.** `_eviction_index` (`memory.py:348-358`) drops the
   entry minimizing `quality · recency` (recency half-life 600 frames). An entry that has proven useful
   (`use_count`) can be dropped, and the bank can collapse to all-`full` (or all-`upper`), destroying the
   partial-query capability that is the point of the module. Fix: protect ≥1 entry per part, or evict from
   the most populous part; add `use_count` to the eviction score.
4. **`query(touch=True)` reinforces entries even while frozen.** `freeze` blocks `add` but not the LRU
   `use_count`/`recency` bump (`memory.py:398-400`) — while the target is lost, a *candidate's* query can
   still promote gallery entries. Fix: no touch while frozen/unverified.
5. **The hard-negative curve is hard-coded** at `_HN_SIM_LO = 0.5`, `_HN_SIM_HI = 0.9`
   (`memory.py:43-45`) while every other threshold is in `MemoryCfg`. Given the measured margins
   (same-person `upper` 0.920 vs different-person `upper` 0.48), the useful decision band is ~0.6–0.8, so
   the current curve gives a distractor at 0.6 only ≈0.24 penalty. Make these config so the experiment
   grid can move them.

### 3.3 Tracklet-level aggregation and lifelong memory (what the literature adds)

| Idea | Source | Transfer to our bank |
|---|---|---|
| **Clean the tracklet before clustering it** — Noise-Filtered Tracklet Partition + self-supervised refined clustering | SSR-C, [arXiv:2406.14261](https://arxiv.org/abs/2406.14261) | aggregate *accepted-quality* observations per track, then form modes; do not let a bad frame vote |
| Tracklet spatial-temporal transformer (intra- + inter-clip attention) | STT, [arXiv:2103.16469](https://arxiv.org/abs/2103.16469) (MARS / DukeMTMC-VideoReID / PRID2011) | justifies `ShortTermMemory.mean_recent(k)` over a single EMA when the target turns |
| Motion cues inside the appearance transformer | Motion-Aware Transformer, [arXiv:2202.04243](https://arxiv.org/abs/2202.04243) | supports the BRIEF's motion-filtered short-term memory |
| **Condense many exemplars into one prototype**; store prototypes, not raw frames; **dual style alignment** so new data does not drag the bank | Pr2R, [arXiv:2508.01587](https://arxiv.org/abs/2508.01587) | +4–6 % over SOTA replay on sequential tasks; seen-domain accuracy only 6.6 % below joint training |
| **Feature centralization with no fine-tuning at all** — aggregate same-identity features to stabilize the representation; reported strong on standard, cross-modality and occluded ReID | Pose2ID, [arXiv:2503.00938](https://arxiv.org/abs/2503.00938) | **the single best free win for our bank**: upgrade `query` from nearest-neighbour to a centralized/aggregated identity prototype. (Cross-check: CLIP-ReID + Pose2ID = 94.9 mAP / 97.3 R1 Market-1501, no fine-tuning.) |
| Continual few-shot on an unseen domain without forgetting | CFReID, [arXiv:2503.18469](https://arxiv.org/abs/2503.18469) | the right framing for "a brand-new target every run" |
| Practical multi-camera manager: local track ids → persistent global ids via OSNet-512 + a three-step matching pipeline; survives camera switch, occlusion, re-entry | [FoundYou](https://github.com/durgeshgowdac/foundyou) | closest public system analogue to our target layer |

### 3.4 Clothes: the failure mode no gallery policy can fix by itself

| Method | LTCC general mAP / R1 | **LTCC cloth-change mAP / R1** | PRCC same-clothes mAP / R1 | **PRCC cloth-change mAP / R1** |
|---|---|---|---|---|
| LTCC baseline | 34.31 / 71.39 | **12.40 / 26.15** | 89.37 / 96.08 | 28.89 / 30.45 |
| PCB | 30.39 / 64.50 | 11.24 / 21.27 | 97.01 / 99.02 | — |
| BoT | 27.80 / 64.33 | 10.70 / 28.12 | 98.41 / 99.30 | 45.31 / 44.92 |
| FD-GAN | 36.89 / 73.43 | 15.36 / 32.91 | 99.69 / 99.97 | 58.57 / 58.34 |
| AIM | 41.1 / 76.3 | 19.1 / 40.6 | 99.9 / 100 | 58.3 / 57.9 |
| **CCFA** | 42.5 / 75.8 | 22.1 / 45.3 | 98.7 / 99.6 | **58.4 / 61.2** |
| CVSL | 41.9 / 76.4 | 21.3 / 44.5 | 99.1 / 97.5 | 56.9 / 57.5 |
| CAL/CAL+DLCR **+ GEFF** (face-enriched gallery) | — | — | — | **64.0–66.0 / 83.5–84.6** |

Source: Diverse Norm [arXiv:2410.03977v1](https://arxiv.org/abs/2410.03977) Table 1 + OpenCodePapers PRCC.
**PRCC cloth-changing Rank-1 for 2023–24 SOTA is ~58–61 %, LTCC ~45 %** — and occlusion compounds it
(OC4-ReID, [arXiv:2403.08557](https://arxiv.org/abs/2403.08557): LTCC cloth-changing R1 43.84 % / mAP
22.43 % in a purpose-built model). Diverse Norm's own finding: *fully removing* clothing features **hurts**
the same-clothes case, so the target is an orthogonal-subspace trade-off (whitening + channel attention)
with sample re-weighting, not deletion.

**Implication for the bank (concrete):** the gallery currently stores one kind of embedding per part with
a single recency half-life (600 frames) for everything. Split the *cue* lifetimes instead:
face and body-shape/part embeddings are **durable**; a clothing-dominated embedding should be
**timestamped and expirable** (or stored under a distinct cue type with a short half-life), so a
cloth-change does not silently poison the durable identity modes.

---

## 4. Distractor-aware / one-shot ReID and the anti-latch rule

**The arithmetic that defines the anti-latch rule.** From the Lead's probe: same-person visible part
0.920, same-person occluded part **0.429**, different-person visible part **0.48**. Therefore:

- averaging over *all* parts is not just suboptimal, it is **actively harmful** — one occluded part
  (0.429) drags the evidence below a different person's visible part (0.48);
- a *single* part score is also unsafe — `head` scored **1.0** against the same person's full-body entry,
  which is the signature of a **degenerate/collapsed embedding** (a tiny, low-information crop mapping to
  a near-constant vector). ⚠️ **This needs a control that the probe did not run: different-person `head`
  and different-person `lower`.** If a different person's `head` also scores ≈1.0, then `head` is
  *uninformative zero-evidence* and must be excluded from any mean rather than allowed to certify a match,
  and `crop_quality`'s area term (saturated at 64×128) must be much stricter for `head`.
  **This is the highest-value quick experiment in this report** and it is cheap: 2 people × 5 parts.

**Rules to implement:**

1. **Visibility-gated fusion, not averaging, not one-part:** mean over parts visible in *both* sides,
   require **≥2 mutually visible parts** (BPBreID/KPR rule; PVPM).
2. **Zero-evidence handling:** discard parts whose crop is degenerate (embedding collapse) rather than
   letting them vote — measured, not assumed (see the `head` control above).
3. **Margin over the best hard negative, not an absolute threshold:** accept a re-acquisition only when
   `score(target) − score(best_distractor) ≥ Δ`. The existing `hard_negative_penalty` (smoothstep
   0.5→0.9, weight 0.35) is a soft version of this; make Δ explicit and tuned.
4. **Multi-cue agreement for the *first* re-acquisition after a long gap:** require part fusion **and**
   (face, if available) **and** motion plausibility; after the first frame, relax.
5. **Freeze + guard time:** already present (`freeze`); ensure no LRU touch while frozen (§3.2-4), and add
   a `guard_frames` window after reappearance before the gallery may be updated from the new track.

**One-shot / few-shot literature:** the training-free feature-centralization line (Pose2ID,
[arXiv:2503.00938](https://arxiv.org/abs/2503.00938)) and continual few-shot ReID
(CFReID, [arXiv:2503.18469](https://arxiv.org/abs/2503.18469), whose two stated challenges are learning
an unseen domain from few shots *and* avoiding forgetting) are the right framings for a clicked-target,
one-example setup; [FoundYou](https://github.com/durgeshgowdac/foundyou) is the closest working system.

> **⚠️ Gap, stated honestly:** I retrieved **no primary DJI ActiveTrack / consumer-drone target-tracking
> documentation** (it is proprietary and not research literature), so this report deliberately contains
> **no DJI numbers**. Treat the DJI-like requirement as a *systems* requirement — click-to-seed, motion
> coast, expanded search, re-detection — rather than expecting a published method to copy.

---

## 5. Face cues

### 5.1 Accuracy (official AdaFace README; competitor rows are as that README reports them)

| Method | Backbone / train | LFW | CFP-FP | CPLFW | CALFW | AgeDB | IJB-B TAR@FAR=1e-4 | IJB-C TAR@FAR=1e-4 | TinyFace R1 | IJB-S Sur-to-Single R1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **AdaFace** | R18 / WebFace4M | 0.9953 | 0.9726 | 0.9228 | 0.9552 | 0.9647 | 93.03 | 94.99 | — | — |
| **AdaFace** | R50 / MS1MV2 | 0.9982 | 0.9786 | 0.9283 | 0.9607 | 0.9785 | 94.82 | 96.27 | — | — |
| **AdaFace** | R50 / WebFace4M | 0.9978 | 0.9897 | 0.9417 | 0.9598 | 0.9778 | 95.44 | 96.98 | — | — |
| **AdaFace** | R100 / MS1MV2 | 0.9982 | 0.9849 | 0.9353 | 0.9608 | 0.9805 | 95.67 | 96.89 | **68.21** | **65.26** |
| **AdaFace** | R100 / WebFace4M | 0.9980 | 0.9917 | 0.9463 | 0.9605 | 0.9790 | 96.03 | 97.39 | **72.02** | **70.42** |
| **AdaFace** | R100 / WebFace12M | 0.9982 | 0.9926 | 0.9457 | 0.9612 | 0.9800 | **96.41** | **97.66** | **72.29** | **71.35** |
| (MagFace) | R100 / MS1MV2 | — | — | — | — | — | 94.51 | 95.97 | — | — |
| (ArcFace) | R100 / MS1MV2 | 0.9982 | — | 0.9208 | 0.9545 | — | 94.25 | 96.03 | — | 57.35 |
| (CosFace) | R100 / MS1MV2 | — | — | — | — | — | 94.80 | 96.37 | — | — |
| (CurricularFace) | R100 / MS1MV2 | — | — | — | — | — | 94.80 | 96.10 | 63.68 | 62.43 |

- **AdaFace wins exactly in our regime**: IJB-C 96.89 vs ArcFace 96.03; **IJB-S surveillance→single
  Rank-1 65.26 vs 57.35 (+7.9)**; **TinyFace Rank-1 72.29 vs 63.89 (URL) / 63.68 (CurricularFace)**. Its
  mechanism (margin adapted to image quality, approximated by feature norm) is designed for degraded
  inputs, and IJB-S is literally CCTV-to-mugshot.
- **But mind the absolute numbers**: 99.8 % on LFW collapses to **~72 % on TinyFace** and **~65 % on
  IJB-S**. Face is a *vote*, not a key, in drone footage.
- Inference contract: **112×112×3, BGR, MTCNN-landmark aligned, mean=std=0.5, 512-d**. Measured weight
  sizes (CVLface): `ir18_webface4m` **91.70 MB**, `ir50_ms1mv2` 166.46 MB, `ir101_webface12m` 248.89 MB.
- ⚠️ The original AdaFace repo's weights are all **Google Drive (unreachable here)**; the new official home
  is [CVLface](https://github.com/mk-minchul/CVLface). **KP-RPE** ([arXiv:2403.14852](https://arxiv.org/abs/2403.14852))
  claims SoTA on TinyFace/IJB-S using facial landmarks for alignment robustness — relevant when the head turns.

### 5.2 Detectors (ONNX, measured sizes) and gating

| Detector | ONNX size | Note |
|---|---|---|
| `scrfd_500m_bnkps` | **2.19 MB** | smallest, landmark-equipped |
| `scrfd_1g` | 2.43 MB | |
| **`scrfd_2.5g_bnkps`** | **3.14 MB** | the sensible operating point |
| `scrfd_10g_bnkps` | 16.14 MB | |
| `scrfd_34g` / `34g_gnkps` | 37.35 / 37.60 MB | most accurate |

(`hf-mirror.com/RuteNL/SCRFD-face-detection-ONNX`; RetinaFace/mobilenet ~1.7 MB **not** size-verified.)
**Face branch total ≈ 95 MB (SCRFD-2.5g + AdaFace IR-18) and a few ms per face.**

**When it helps:** long-term recovery after cloth change, re-entry after out-of-frame, disambiguating
similar-looking distractors with a frontal head. Evidence: PRCC cloth-changing Top-1 improves by **33.5 %**
(LTCC 6.7 %) purely from enriching the gallery with face features (GEFF).
**When it fails:** tiny faces (TinyFace ceiling ~72 %), profile/back-of-head, masks, motion blur.
⚠️ **No verified masked-face benchmark number was retrieved in this session — flagged.** Gate on
alignment success and a pixel-size floor (≥ ~40 px face box) before the face branch may vote; below that,
fall back to the `head`/`torso` part embedding.

---

## 6. Cost on our hardware

### 6.1 Measured vs documented vs estimated

| Quantity | Value | Status |
|---|---|---|
| `yolo26n-reid.onnx` file size | 9,873,245 B | **measured** (on disk) |
| its parameter count | 2,432,284 (fp32 ⇒ 9.73 MB ≈ file size ⇒ **not quantized/half**) | **measured** (initializers) |
| its output | `embeddings`, L2-normalized **512-d** | **measured** (graph + metadata) |
| its native training size | **448×448** (`imgsz [448,448]`) | **measured** (metadata_props) |
| bs1 latency, CUDA | **2.74 ms** | measured (Lead) — *at the currently configured 224 input* |
| bs40 latency, CUDA | **12.7 ms ⇒ 0.317 ms/crop** | measured (Lead) — same caveat |
| OSNet-x1_0 | 2.2 M params, 0.98 GFLOPs @256×128, 16.47 MB | documented + measured size |
| RTMPose-m | 51.81 MB, 75.8 % AP, 430+ FPS GTX 1660 Ti | documented |
| AdaFace IR-18 / SCRFD-2.5g | 91.70 MB / 3.14 MB | **measured** |
| encoder latency for *other* backbones on a 4060 Laptop | — | **not measured — would be a guess; none is given** |

### 6.2 The crop-count arithmetic decides the architecture

`appearance.py` builds **one batched forward pass over all (box, part) pairs**
(`appearance.py:3-6`, `:529-565`), i.e. **5 parts × N persons crops per frame** (PART_KEYS = 5). With the
measured numbers:

| Workload | Crops/frame | At 2.74 ms/crop (bs1) | At 0.317 ms/crop (bs40) |
|---|---|---|---|
| N=10, **full body only** | 10 | 27.4 ms | **3.2 ms** |
| N=10, **all 5 parts** | 50 | **137 ms** (4× over budget) | **15.9 ms** |
| N=20, all 5 parts | 100 | 274 ms | **31.7 ms** (≈ the whole 33 ms budget) |

Two conclusions that follow directly:
1. **Batching is mandatory, not an optimization** — the bs1→bs40 gap is **8.6×**. Any code path that
   encodes a part crop per detections-per-frame at batch 1 destroys the 30 FPS budget.
2. **Part encoding must be target-centric**: encode the 5 parts for the *target and a small candidate
   set*, not for all N persons every frame (full-body-only for everyone at 3.2 ms vs 15.9 ms for all
   parts at N=10). This costs nothing in the target-centric metrics the BRIEF defines, and it is the
   difference between fitting and not fitting.

### 6.3 The 224-vs-448 defect — measure before paying

§1 established that `yolo26n-reid.onnx` is trained at **448** but `reid.py` never overrides `imgsz`
because the ONNX input dims are **all dynamic**, so `config.py:64 imgsz=224` and
`appearance.py:289-291 ReID(..., imgsz=int(cfg.imgsz or 224))` run it at **224**. Two experiments, in
this order:
1. **Fix the 224→448 mismatch and measure the accuracy delta** (one line; the model is being fed half the
   linear resolution it was trained at — cheap upside, unknown size).
2. **Then measure the latency cost** of 448. The published FLOPs for this family are quoted **at 448**
   (2.0 GFLOPs for `n`, 6.6 for `s`, 20.1 for `m`, 25.2 for `l`, 55.9 for `x`), so expect roughly
   **~4× the pixels and ~4× the per-crop time** vs 224 — i.e. N=10 all-parts could go from ~16 ms to
   **~60 ms**. If accuracy does not improve materially, **keep 224** and say so with the measurement.
   Also consider `yolo26s-reid.onnx` (27.15 MB, 7.5 M params, 6.6 GFLOPs @448) as the accuracy/cost
   midpoint — still small, and the whole five-model family is only ~275 MB on disk.
3. **Do not reach for `yolo26x`** (32.7 M params, 55.9 GFLOPs, 123 MB): 28× OSNet-x1_0's FLOPs is a
   per-track, not per-frame, budget.

### 6.4 Bank cost — negligible

512-d fp32: 96 entries × 512 × 4 B = **197 KB** for the entire long-term gallery; even 5 parts × 96
entries × 512-d = **983 KB**. The gallery is free; the encoder is the whole cost. This means the
**strategy** (what to store, what to compare, when to fuse) is where the wins are — consistent with the
BRIEF's "no fine-tuning, optimize the strategy".

---

## 7. Decision table

Ordered by (expected effect ÷ cost). Everything here is a strategy/config change — no fine-tuning.

| # | Action | Why / evidence | Cost | Expected effect |
|---|---|---|---|---|
| 1 | **Run the missing probe control**: different-person `head` and `lower` (2 people × 5 parts) | 1.0 for `head` and 0.429 for the occluded `lower` are uninterpretable without a same-part different-person reference; if diff-person `head` ≈ 1.0, `head` is zero-evidence and must be excluded from any mean | minutes | prevents a silent ID-swap path; decides the part set |
| 2 | **Add an occlusion term to `crop_quality`** using the *other* detections' boxes (containment of the part box in another person's box) | local proof: same-person occluded part 0.429 **<** different-person visible part 0.48; a large sharp fully-occluded crop currently scores quality ≈ 1.0 | ~free (boxes already exist) | stops poisoned inserts; the single biggest correctness fix |
| 3 | **Replace one-part `_candidates` with visibility-gated multi-part fusion** (mean over mutually visible parts, ≥2 required) | BPBreID/KPR rule; PVPM's "only mutually visible parts"; current design lets one part decide | small code change | the core partial-view capability |
| 4 | **Fix the 224 → 448 input mismatch and measure the accuracy delta** | model trained at 448 (`metadata_props`), config feeds 224 (`config.py:64`), `reid.py` can't fix it (dynamic dims) | 1 line + re-measure | unknown, cheap upside; must be measured, not assumed |
| 5 | **Membership/outlier test on insert** (`dist(feat, STM_EMA) ≤ τ_insert`) + replace-on-match | IEEE 9506241's error-resilient recurring gallery; current gate tests novelty only | small | blocks permanent gallery poisoning by a distractor |
| 6 | **Make part/face encoding target-centric** (full-body for all N; parts only for target + candidates) | 5× crop count: 15.9 ms vs 3.2 ms at N=10 (measured per-crop cost) | small | fits 30 FPS with headroom |
| 7 | **Re-tune the hard-negative curve into config** and make the accept rule a *margin over the best distractor* | currents curve fixed at 0.5–0.9 (`memory.py:43-45`) vs measured margins 0.48↔0.92; absolute thresholds are not robust | trivial | fewer ID swaps |
| 8 | **Extrapolate the full-body box before assigning part ranges when truncated** | `part_boxes` remaps `lower` onto the torso for an upper-body-only box (`_PERSON_ASPECT 2.7` already present) | small | correct part semantics; fewer false partial matches |
| 9 | **Diversity/coverage-aware eviction** (protect ≥1 entry per part, include `use_count`) | current eviction is `quality·recency` only (`memory.py:348-358`); the bank can collapse to all-`full` | small | preserves partial-query ability over long runs |
| 10 | **Add a durable face cue** (SCRFD-2.5g + AdaFace IR-18) gated on ≥40 px + alignment | AdaFace IJB-S R1 65.26 vs ArcFace 57.35; TinyFace 72.29; face-enriched gallery gives **+33.5 % PRCC cloth-changing Top-1** | ~95 MB, few ms | the only cue that survives cloth change |
| 11 | **Split cue lifetimes**: face/shape durable, clothing-dominated entries expirable | PRCC cloth-change R1 saturates at ~58–61 %, LTCC ~45 %; Diverse Norm: deleting clothing features hurts the same-clothes case | trivial | prevents cloth-change from poisoning durable modes |
| 12 | **A/B a pose branch** (RTMPose-m, 51.81 MB) vs the current fixed fractions | MAMS/HG reach 73.3/61.4 R1 with **no** pose ⇒ pose must *earn* its place; but pose is ~1–3 ms/frame at 430+ FPS on a GTX 1660 Ti | 52 MB, small | potential part-alignment gain; measure, don't assume |
| 13 | **Do NOT** switch to TransReID/CLIP-ReID/SOLIDER/KPR backbones | +86 M-param ViTs buy ~+5–7 mAP holistic but 30–80× the compute, and KPR's 384.66 MB Swin weights are GDrive-only here; the mechanism, not the backbone, is the bottleneck | — | avoids an expensive dead end |
| 14 | **Do NOT** use `model="auto"` and assume it equals the ReID model | `build_encoder` returns detector-backbone features (objectives differ); docs even fall back to `yolo26n-cls.pt` | — | avoids silently degrading appearance |

## 8. Numbers I trust, and numbers I do not

**Trust (primary source, cross-checked):** OSNet-x1_0 2.2 M / 0.98 GFLOPs and Market **94.2 R1 / 82.6 mAP**,
Duke 87.0 / 70.2, MSMT17 74.9 / 43.8 (torchreid MODEL_ZOO); TransReID 95.1 / 89.0 Market, 85.3 / 67.8
MSMT17, Occluded-Duke 67.4 / 59.5 (repo README); **KPR+SOLIDER Occluded-Duke 84.31 R1 / 75.11 mAP**;
FED Occluded-Duke 68.1 R1 and Occluded-ReID 87.0 R1; PRCC cloth-changing R1 ~58–61 % and LTCC ~45 %
(Diverse Norm Table 1); the AdaFace table in §5.1; RTMPose-m 75.8 % AP / 430+ FPS GTX 1660 Ti; every file
size in §6/§9; and **all of §1** (read directly off the shipped ONNX and the local wheel).

**Do not trust / explicitly flagged:**
✗ **published mAP/Rank-1 for `yolo26*-reid.onnx`** — Ultralytics publishes **none**, and there is no
`yolo26-reid*.yaml` in the wheel, so the architecture is only partially recovered from node names (a
convolutional YOLO backbone with a **C2PSA/`2psa`** stage, `C3k2` blocks and a GAP+GMP → Gemm → BNNeck →
L2 head — **no transformer encoder, no InstanceNorm**, so it is *not* OSNet/TransReID/CLIP-ReID);
✗ DPM++'s SOTA claim (numeric tables unreachable) — mechanism sound, numbers unverified;
✗ the exact MSMT17 subset/recipe and the width/depth multipliers of the ReID backbone (config not shipped);
✗ HRNet-W32 params/AP/FPS; masked-face degradation magnitude; any DJI ActiveTrack number;
✗ all encoder latency figures for backbones we have **not** run on this GPU;
✗ PCB/MGN/SOLIDER/KPR/CLIP-ReID param counts marked ⚠️ (inferred from backbone families);
✗ aggregator rows where mAP > Rank-1 (e.g. PartFormer 62.5/70.2, PAFormer 60.4/66.4) — column errors, excluded;
✗ aggregator PRCC rows that conflate *general/same-clothes* with *cloth-changing* — use Diverse Norm instead;
✗ CUHK03 cross-protocol comparisons (old detected ≈ 41.7 R1 vs new labelled ≈ 84.7 R1).

## 9. References

**Encoders / lineage** — PCB ECCV 2018 [arXiv:1711.09349](https://arxiv.org/abs/1711.09349) ·
HPM AAAI 2019 [arXiv:1804.05275](https://arxiv.org/abs/1804.05275) ·
MGN ACM MM 2018 [arXiv:1807.01440](https://arxiv.org/abs/1807.01440) ·
OSNet ICCV 2019 [arXiv:1905.00953](https://arxiv.org/abs/1905.00953) / OSNet-AIN [arXiv:1910.06827](https://arxiv.org/abs/1910.06827) ·
TransReID ICCV 2021 [arXiv:2102.04378](https://arxiv.org/abs/2102.04378) [code](https://github.com/damo-cv/TransReID) ·
CLIP-ReID AAAI 2023 [code](https://github.com/Syliz517/CLIP-ReID) ·
SOLIDER CVPR 2023 [code](https://github.com/tinyvision/SOLIDER-REID) ·
KPR ECCV 2024 [arXiv:2407.18112](https://arxiv.org/abs/2407.18112) [code](https://github.com/VlSomers/keypoint_promptable_reidentification) · BPBreID WACV 2023 [arXiv:2211.03679](https://arxiv.org/abs/2211.03679) ·
torchreid [MODEL_ZOO](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) ·
Bag of Tricks [arXiv:1903.07071](https://arxiv.org/abs/1903.07071) (the BNNeck in our ONNX head) ·
Ultralytics YOLO26 [arXiv:2606.03748](https://arxiv.org/abs/2606.03748) ·
Ultralytics tracking docs [docs.ultralytics.com/modes/track](https://docs.ultralytics.com/modes/track/) ·
`reid.py` [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/reid.py).

**Occluded / partial** — PGFA ICCV 2019 [code](https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID) ·
HOReID CVPR 2020 [code](https://github.com/wangguanan/light-reid) ·
ISP ECCV 2020 [code](https://github.com/CASIA-IVA-Lab/ISP-reID) ·
PVPM CVPR 2020 [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Pose-Guided_Visible_Part_Matching_for_Occluded_Person_Re-Identification_CVPR_2020_paper.html) ·
FED CVPR 2022 [arXiv:2112.08740](https://arxiv.org/abs/2112.08740) [code](https://github.com/ZacharyWang-007/FED-Occluded-ReID) ·
HG CVPR 2021 [arXiv:2104.06524](https://arxiv.org/abs/2104.06524) ·
DPM++ 2026 [arXiv:2605.06637](https://arxiv.org/abs/2605.06637) ·
Partial ReID ICCV 2015 [openaccess](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Partial_Person_Re-Identification_ICCV_2015_paper.html) ·
TRGA doi [10.1007/s00530-025-02141-5](https://doi.org/10.1007/s00530-025-02141-5) ·
MAMS doi [10.3390/s25196210](https://doi.org/10.3390/s25196210) · MBIN doi [10.3390/s25216526](https://doi.org/10.3390/s25216526) · OC4-ReID [arXiv:2403.08557](https://arxiv.org/abs/2403.08557) ·
[Occluded-DukeMTMC leaderboard](https://opencodepapers-b7572d.github.io/benchmarks/person-re-identification-on-occluded-dukemtmc.html) · [dataset](https://github.com/lightas/Occluded-DukeMTMC-Dataset).

**Clothes-changing** — Diverse Norm [arXiv:2410.03977](https://arxiv.org/abs/2410.03977) · GEFF (gallery
enrichment with face features) · CSCI [code](https://github.com/ppriyank/ICCV-CSCI-Person-ReID) ·
SAS-VPReID [arXiv:2601.05535](https://arxiv.org/abs/2601.05535).

**Face** — AdaFace CVPR 2022 [arXiv:2204.00964](https://arxiv.org/abs/2204.00964) [code](https://github.com/mk-minchul/AdaFace) [CVLface](https://github.com/mk-minchul/CVLface) [KP-RPE 2403.14852](https://arxiv.org/abs/2403.14852) ·
ArcFace [arXiv:1801.07698](https://arxiv.org/abs/1801.07698) · MagFace [arXiv:2103.06627](https://arxiv.org/abs/2103.06627) ·
RetinaFace [arXiv:1905.00641](https://arxiv.org/abs/1905.00641) · SCRFD [arXiv:2105.04714](https://arxiv.org/abs/2105.04714).

**Memory / lifelong / tracklets** — STT [arXiv:2103.16469](https://arxiv.org/abs/2103.16469) ·
Motion-Aware Transformer [arXiv:2202.04243](https://arxiv.org/abs/2202.04243) ·
SSR-C [arXiv:2406.14261](https://arxiv.org/abs/2406.14261) ·
Pr2R [arXiv:2508.01587](https://arxiv.org/abs/2508.01587) ·
CFReID [arXiv:2503.18469](https://arxiv.org/abs/2503.18469) ·
Pose2ID [arXiv:2503.00938](https://arxiv.org/abs/2503.00938) ·
open-set recurring gallery [IEEE 9506241](https://ieeexplore.ieee.org/document/9506241) ·
[FoundYou](https://github.com/durgeshgowdac/foundyou).

**Pose** — RTMPose [arXiv:2303.07399](https://arxiv.org/abs/2303.07399).

---

# 10. Addendum — independent findings (different author, different evidence base)

Sections 1–9 above were written from one line of investigation; this addendum adds findings from a second,
independent pass that do **not** overlap with them, plus **measured replacements** for two estimates above.
Where it contradicts §1–§9, the measurement wins.

## 10.1 Measured cost of the 224→448 fix — replaces the estimate in §6.3

§6.3 estimates "~4× the per-crop time … ~60 ms" for 448. **That is now measured** (`onnxruntime` 1.20.2,
hand-built `InferenceSession`, `providers=['CUDAExecutionProvider','CPUExecutionProvider']`, RTX 4060
Laptop 8 GB, warm-up + mean over repeats):

| Config | bs1 | bs40 (total) | per-crop at bs40 |
|---|---|---|---|
| **224×224** (ultralytics default, `reid.py:41`) | 1.789 ms | 11.795 ms | **0.2949 ms** |
| **448×448** (the model's native size) | **2.266 ms** | **42.765 ms** | **1.0691 ms** |

- These independently reproduce the Lead's numbers (2.74 ms @ bs1, 12.7 ms @ bs40, 0.317 ms/crop); the
  gaps are warm-up/clock state.
- **The 448 fix costs 3.6× per crop at bs40 but only 1.27× at bs1**, because bs1 is launch-overhead-bound.
  So §6.3's "~4×" is right for batched work and **wrong for single-target work, where it is nearly free**.
- Revised workload numbers (using the same crop counts as §6.2): N=10 all-5-parts = 50 crops ⇒ **53.5 ms**
  at 448 (vs §6.3's ~60 ms estimate); **N=20 all-5-parts = 100 crops ⇒ 107 ms — infeasible**. This hardens
  §7 item 6 ("target-centric part encoding") from an optimisation into a **requirement**.
- Confirmed at runtime, not just by reading `reid.py`: `ReID('…yolo26n-reid.onnx')` prints
  **`ReID.imgsz = (224, 224)`** and `batch_size = None`. The §1/§6.3 defect is real and active.
- Also confirmed empirically: output shape `(bs, 512)` with **`L2 norm = 1.000000`** — the embedding is
  genuinely L2-normalized, so cosine similarity is a plain dot product.

### ⚠️ Provider trap not mentioned in §1–§9 — verify CUDA is actually active

Instantiating the encoder through ultralytics' own `AutoBackend`/`ReID` path emitted:

```
EP Error … cuda_execution_provider_info.cc:142 … Failed to parse provider option "device_id":
parse_string.h:71 … Failed to parse value: "0"
Falling back to ['CPUExecutionProvider'] and retrying.
WARNING CUDA requested but CUDAExecutionProvider not available. Using CPU…
```

**The convenience wrapper silently ran on CPU.** Only the hand-built `InferenceSession` reached CUDA.
So any latency measured through the wrapper (`appearance.py:289-291` uses `ReID(...)`) may be a **CPU**
number, and any accuracy/throughput conclusion drawn from it is suspect. **Assert the active provider
explicitly before trusting any ReID timing.** Available providers here are
`['TensorrtExecutionProvider','CUDAExecutionProvider','CPUExecutionProvider']`.

## 10.2 Two `memory.py` defects not covered by §3.2

**(a) The diversity gate is admission-only, so a bad first entry is locked in permanently.**
`memory.py:326` rejects when the minimum **cosine distance** to a same-part entry is `<= 0.15`, i.e. when
similarity is **≥ 0.85**; `memory.py:243-245` states rejected duplicates are deliberately **not merged**.
Combined, the first observation of a part is the only observation of that part it will ever have — later,
better evidence is discarded as a "duplicate" and changes nothing. The arithmetic against the Lead's probe
is stark: same-person `upper` **0.920** (dist 0.080), `torso` **0.941** (0.059), `head` **1.0** (0.000) are
**all inside the reject zone**, while the degraded views are the ones admitted (`full` 0.757 → 0.243;
`lower` 0.429 → 0.571). §3.1 reads the gate's intent correctly (rejecting near-duplicates is deliberate and
keeps distinct *modes*), but the *consequence* — no upgrade path — is a defect.
**Fix (few lines, no training):** when a candidate is refused as a near-duplicate, still (i) refresh that
entry's `quality`/`recency`, and (ii) if the candidate's quality exceeds the stored entry's, **merge it**
(move the stored feature toward the better observation). §7 item 5 adds a *membership* test on insert;
this is a different and complementary fix — an *upgrade* path.

**(b) The recency decay clock keeps running while the target is blind.**
`memory.py:274-283` computes `recency = 0.5 ** (age / 600)` with `age = now - entry.frame_idx`, an
**absolute** frame count, so it decays during occlusion and out-of-frame coasting. After 600 frames (20 s
at 30 FPS) every entry is ≤0.5; after 1800 frames (60 s), ≤0.125. Since eviction (`memory.py:355-358`)
minimises `quality · recency`, all entries become near-equivalent and `quality_recency` **silently
degenerates to quality-only** after a long occlusion — precisely when re-acquisition matters most.
§3.2-4 notes that `query(touch=True)` can bump entries while frozen; this is a separate clock.
**Fix:** compute `age` relative to the **last frame the target was visible** (suspend the decay clock while
blind), and always retain one **protected, never-evicted anchor entry** (the click-time crop).

## 10.3 `distractor_margin` placed in the literature (§4 has the rules; this is the evidence)

The anti-latch failure has a name: **DAM4SAM** ([2509.13864](https://arxiv.org/abs/2509.13864)) calls it
*"tracking drift toward distractors"*, states distractors "pose a key challenge", and answers it with a
**distractor-aware memory module plus introspection-based memory management** (and a purpose-built `DiDi`
Distractor-Distilled dataset); "+11% on EfficientTAM, +4% on EdgeTAM, SOTA on ten benchmarks". Note the
mechanism: **not a margin test** — a distractor *memory* plus a trust check. Our hard-negative bank +
margin rule is a training-free approximation of exactly that.

The literature's defences form five families; a margin rule is **one of them**:

| Family | Mechanism | Canonical source | Adoptable (no fine-tuning)? |
|---|---|---|---|
| (a) Margin / deferral | refuse unless best beats runner-up | our `distractor_margin`; `OxUvA` [1803.09502](https://arxiv.org/abs/1803.09502) formalises the *present/absent* decision | ✅ built |
| (b) **Learned verification** | network answers "is this the target / ready to update?" | **LTMU** [2004.00305](https://arxiv.org/abs/2004.00305) (CVPR20): Meta-Updater over geometric + discriminative + appearance cues, cascaded LSTM, binary output; code `github.com/Daikenan/LTMU` | ❌ needs training — **but copy its cue list** |
| (c) Distractor-aware memory / hard-negative mining | store lookalikes, use them in the score | DAM4SAM [2509.13864](https://arxiv.org/abs/2509.13864) | ✅ built (§7 item 7) |
| (d) Re-detection + deferred commitment | don't commit; re-detect and re-pick | LTMU's re-detector; `OxUvA` evaluation | ✅ built |
| (e) **Co-occurrence / mutual exclusion** | the target cannot be two confirmed tracks at once | `SUSHI` [2212.03038](https://arxiv.org/abs/2212.03038) graph hierarchies | ✅ **missing — add it** |

- **The freeze is well-founded.** **Model Decay in Long-Term Tracking**
  ([1908.01603](https://arxiv.org/abs/1908.01603)): *"Updating the tracker model with adverse bounding box
  predictions adds an unavoidable bias term … accumulation of this bias over a long-term can eventually
  lead to a permanent loss of the target"* (4 long-term benchmarks, 30-min videos). This is the strongest
  published support for `freeze` **and** for quality-gating (§3.2's instinct is right). LTMU's meta-updater
  is the learned version; §7 item 11 already proposes the right refinement.
- **A negative result worth knowing: NO paper retrieved measures "how often the tracker latches onto a
  bystander" as a precision/false-acquisition metric.** DAM4SAM reports *relative* gains only. So
  `distractor_margin=0.06` cannot be set from the literature — it must be calibrated on our own logs.
- **`margin` is an absolute difference of *fused* costs** (`association.py:601-607`,
  `margin = float(total[second] - best_cost)`), so its correct value depends on `stm_weight`/`ltm_weight`
  (`config.py:168 ltm_weight=0.4`) and on cost normalisation: **change the weights and 0.06 silently
  becomes a different rule.** Prefer a *relative* margin or a margin on the appearance term alone — the
  only term with a physically meaningful scale (measured same-person `full` 0.757 vs different-person
  `upper` 0.48 ⇒ an appearance gap of ≈0.28).
- **Free addition (family (e)):** if the best appearance candidate belongs to a MOT track that is already
  occupied by another confirmed track in the same frame, that candidate is an impostor by construction.
  The MOT layer already provides this; the current design does not appear to use it.

## 10.4 `ltm_replace_policy` has no published backing — and FIFO is the field standard

§3.3 lists what lifelong-ReID adds, but note a **definitional trap**: in this literature *"lifelong person
re-identification" (LReID) means continually **training** a model on a non-stationary stream of domains to
resist catastrophic forgetting* (AKA CVPR21 [2103.12462](https://arxiv.org/abs/2103.12462); KRC AAAI23
[2211.16201](https://arxiv.org/abs/2211.16201); DASK AAAI25 [2412.09224](https://arxiv.org/abs/2412.09224);
Teata TIP25 [2405.16600](https://arxiv.org/abs/2405.16600); DCR [2409.19954](https://arxiv.org/abs/2409.19954);
DSKC [2508.03516](https://arxiv.org/abs/2508.03516); SCORE ECCV26 [2609.12577](https://arxiv.org/abs/2609.12577);
PAD/CVPR26 [2605.05027](https://arxiv.org/abs/2605.05027); VLADR [2603.19678](https://arxiv.org/abs/2603.19678);
SPRED ICCV25 [2507.01884](https://arxiv.org/abs/2507.01884); DAFC [2504.15041](https://arxiv.org/abs/2504.15041);
DRE [2403.16003](https://arxiv.org/abs/2403.16003); Dynamic Textual Prompt
[2411.06023](https://arxiv.org/abs/2411.06023)). It is a **training-time** problem and does **not** define a
gallery eviction policy.

> **Therefore: the measured benefit of `quality_recency` over a fixed-size FIFO gallery is UNVERIFIED — I
> found no paper that measures either policy against the other.** The actual field baseline is
> **DeepSORT / BoT-SORT: a fixed-size FIFO appearance gallery (~100 descriptors per track), matched by
> nearest-neighbour cosine with a threshold.** Our `ltm_replace_policy` is *unproven, not standard*.
> **Action:** both policies already exist (`config.py:127-135`), so A/B `fifo` vs `quality_recency` on our
> own target-centric metrics (ID switches, re-acquisition latency, latch events) — a config sweep, not code.
> Also worth mining: **DASA** (Machine Learning 2025, [2405.19695](https://arxiv.org/abs/2405.19695)) freezes
> pretrained conv layers and adapts only BatchNorm + a lightweight module with "significantly reduced
> storage consumption" — the same *adapt-the-strategy-not-the-network* stance as our BRIEF.
> **⚠ And a trap:** **HiDeR** ([2508.01587](https://arxiv.org/abs/2508.01587)) is the closest idea to our
> intent (fixed-budget compact memory, complexity-aware allocation by intra-class variance), but the
> authors have **WITHDRAWN it** ("The current version is obsolete"). Cite the idea, never the numbers.
> (§3.3's Pr2R row cites the same arXiv id as live — treat that row as withdrawn-source.)

## 10.5 Version-stamp the gallery **before** the 224→448 change (§7 item 4)

Features from an encoder run at 224 and the same encoder run at 448 are **different embeddings**; mixing
them in one gallery degrades every comparison silently. The literature states this for model updates
generally: backward-compatibility / re-indexing ([2403.10022](https://arxiv.org/abs/2403.10022),
Bi-C2R [2512.25000](https://arxiv.org/abs/2512.25000)) shows new/old features are **incompatible without
re-indexing**. **Action:** stamp the gallery with the encoder config (`imgsz`, encoder file hash) and
invalidate/re-index on any change. Trivial to add now; a nasty silent bug to debug later.

## 10.6 Face: the pixel arithmetic that decides §5's gate (§5.2 proposes "≥ ~40 px")

§5.2's ≥40 px gate is right but needs to be read against the person's size in frame. Classical thresholds
(primary sources):

- **Boom et al.** (via [PubMed 22722306](https://pubmed.ncbi.nlm.nih.gov/22722306)): performance "severely
  degraded for face images with sizes **less than 32 × 32 pixels**"; "fairly similar from 32 × 32 to
  128 × 128 pixels".
- **FRVT 2000** (same source): similar at **30–60 px eye-to-eye**, "severely degraded for some algorithms
  at an **eye-to-eye distance of 15 pixels**"; surveillance faces are "commonly between 15 and 30 pixels
  across in terms of eye-to-eye distance".
- **ICARCV 2006** ([doi](https://doi.org/10.1109/icarcv.2006.345480)): "face recognition systems usually
  require face images with **more than 50 pixels between the eyes**"; "EERs remain almost constant down to
  **32×32 pixels**, below this the EER increases rapidly".
- **PMC8329719**: law-enforcement mugshots use "**at least 90 pixels**" inter-eye; FaceVACS performs when
  both gallery and probe are "**higher than 32×32**".
- **PTZ face-at-a-distance** ([ieee 5634539](http://ieeexplore.ieee.org/document/5634539)): "Typical
  commercial face recognition engines require face images with **at least 60 pixels between the eyes**".

**My conversion (arithmetic is mine; constants above are cited).** A frontal face box is ≈0.75 as wide as
tall, and an ArcFace 5-point alignment places the eyes ≈0.32 of the crop width apart, so
**eye-to-eye ≈ 0.24 × face-box-height**. Anthropometrically a standing adult is ≈7.5 face-heights tall, so
**face-height ≈ person-height / 7.5**:

| Face box | Person height needed | % of a 640-px detector frame | Eye-to-eye |
|---|---|---|---|
| 32 px (hard floor) | **≥ 240 px** | 37.5 % | ≈ 8 px |
| 40 px (§5.2's gate) | **≥ 300 px** | 47 % | ≈ 10 px |
| 64 px (usable) | ≥ 480 px | 75 % | ≈ 15 px |
| 112 px (native input) | ≥ 840 px | >100 % — unreachable | ≈ 27 px |

**The consequence §5 does not state:** a person filling the **entire** 640-px detector height yields only a
~85 px face and **≈20 px eye-to-eye — below the 30–60 px that commercial engines require**. So on a
640-imgsz pipeline **face recognition is marginal-to-unusable at every distance**, and §7 item 10's "durable
face cue" will rarely fire. To make face work you need a dedicated high-resolution face stream (crop from a
1080p+ source, or PTZ) — a different project. **The Cheapest-usable-model answer, for completeness:**
MobileFaceNet ([1804.07573](https://arxiv.org/abs/1804.07573)) — **0.99 M params, 4.0 MB, 112×112, 128-d or
512-d, LFW 99.55 %, MegaFace TAR@FAR1e-6 92.59 %, 221 MAdds**, "18 ms on a mobile phone" (Snapdragon 820
CPU) ⇒ **sub-millisecond class on a 4060 — ESTIMATE, not measured here**; or OpenCV's `FaceRecognizerSF`
(SFace, 112×112, ships with opencv-contrib, no download). Both are far cheaper than §5.2's
SCRFD-2.5g + AdaFace IR-18 (~95 MB) — use AdaFace if the accuracy in §5.1 is what matters, MobileFaceNet if
the branch is only ever a tie-break vote. **AdaFace remains the better choice for degraded input** (its
quality-adaptive margin, using feature norm as the quality proxy, is exactly aimed at our regime).

**Verdict:** adopt face only as a **gated auxiliary vote** (face box ≥ 40 px, alignment success), with its
**own margin**, and never let it alone override §4's refusal rule — otherwise a single spurious
high-confidence face match defeats the anti-latch rule at precisely the moment appearance is uncertain.

## 10.7 Addendum gaps (marked UNVERIFIED)

- **⚠️ Correction to an earlier draft of this section, which is now retracted:** it previously claimed
  *"the ReID model itself is not shipped in the wheel."* **That is false** — ultralytics 8.4.185's
  `trackers/utils/reid.py` exists and was read this session (§1, §10.1, §10.5), and its line 41 is
  where the `imgsz=224` default that §1's defect hinges on actually lives. Only the **training
  config** (`cfg/datasets/MSMT17-v1.yaml`) is absent. Do not repeat the retracted claim.
- **UNVERIFIED:** published mAP/Rank-1 for `yolo26n-reid.onnx` itself; measured false-acquisition/precision
  numbers for *any* anti-latch mechanism; the measured benefit of quality/diversity eviction over FIFO;
  MobileFaceNet latency on our GPU (estimated, not measured); fp16 speedup at 448 (`args.half=False`, so ORT
  will not silently use tensor cores — allowing fp16 would roughly halve 448's cost but risks a BNNeck
  calibrated in fp32; **test before trusting**).
- **UNVERIFIED (added with §10.8–§10.10):** the **absolute** Rank-1/mAP of KPR (arXiv:2407.18112) on
  Occluded-Duke — only its stated **deltas** over prior SOTA were captured; the **classic**
  `yolo26n-reid` on **MSMT17** under the same protocol as TransReID/OSNet (it is trained on the
  MSMT17-v1 re-split, so a like-for-like number may not exist at all); the **exact training protocol behind
  each fast-reid zoo row** (the README names 1×V100 but the per-config augmentation and input size
  could not be read, so §10.8's fast-reid ⇄ torchreid rows are **not** like-for-like — see the
  protocol-trap callout there); and our own **446 ms → rebuilt-448 ms** change is a *measured
  prediction*, not yet a measurement of a running 448 build (no image data on disk to benchmark with).
- **Not reproducible here:** any target-vs-bystander discrimination experiment — `C:\Projects\track\data\`
  holds only `weights/`, `raw/`, `raw_pdf/` (**no images**), and `github.com` / `huggingface.co` are blocked
  from this machine.
- **DaSiamRPN** (ECCV 2018), the origin of "distractor-aware" tracking, and **GlobalTrack** were **not
  retrieved** in this session — cite only as UNVERIFIED. **GroupFace, FaceNet, MobileFaceNet, and any
  masked-face benchmark (MFR / RMFRD)** were likewise not retrieved; §5's face numbers come from the
  official AdaFace README, MagFace's paper, and a third-party CoreFace aggregation, and the last of
  those three is a **cross-paper table whose per-row protocol is not stated** — treat its rows as
  indicative, and prefer the per-paper tables where the two disagree.

## 10.8 Canonical foundations line — the numbers §1 deliberately omitted

§1 established *what our encoder is*; this section records *the field's standard ladder*, so that
"our model is small" can be judged against real numbers rather than against a feeling. Two primary
sources were fetched this session and are quoted with their **protocol**, because the same method
scored under two protocols differs by tens of points:

| Dataset | #ID | #images | #cam | Occlusion protocol (where applicable) | Source |
|---|---|---|---|---|---|
| MSMT17 | **4,101** (1,041 train) | **126,441** | **15** | — | [TransReID arXiv:2102.04378](https://arxiv.org/abs/2102.04378) Table 1 **[SRC]** |
| Market-1501 | **1,501** | **32,668** | **6** | — | same Table 1 **[SRC]** |
| DukeMTMC-reID | **1,404** | **36,441** | **8** | — | same Table 1 **[SRC]** |
| Occluded-Duke | 1,404 | 36,441 | 8 | **9 % / 100 % / 10 %** occluded in train / query / gallery | same Table 1 + §2.1 **[SRC]** |

**Metric definitions (same source, §4.2) [SRC]:** "Following conventions in the ReID community, we
evaluate all methods with Cumulative Matching Characteristic (CMC) curves and the mean Average
Precision (mAP)." TransReID also **states its own protocol explicitly**: all person images resized to
**256×128**, batch 64 with 4 images per ID, SGD momentum 0.9 / weight decay 1e-4, lr 0.008 with cosine
decay, fp16 on one Tesla V100, ViT initialized from **ImageNet-21K → ImageNet-1K**. **[SRC]**

### The ladder, with backbones, input size and protocol (Rank-1 / mAP)

| Method | Backbone | Input | Market-1501 | DukeMTMC-reID | MSMT17 | Source |
|---|---|---|---|---|---|---|
| IDE / softmax (plain) | ResNet50 | 256×128 | 87.9 / 70.4 | 78.3 / 58.9 | 63.2 / 33.9 | [torchreid MODEL_ZOO](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) **[SRC]** |
| softmax + FC512 head | ResNet50 | 256×128 | 90.8 / 75.3 | 81.0 / 64.0 | 69.6 / 38.4 | torchreid **[SRC]** |
| MobileNetV2 ×1.0 | MobileNetV2 | 256×128 | 85.6 / 67.3 | 74.2 / 54.7 | 57.4 / 29.3 | torchreid **[SRC]** |
| HACNN | HACNN | **160×64** | 90.9 / 75.6 | 80.1 / 63.2 | 64.7 / 37.2 | torchreid *(trained from scratch)* **[SRC]** |
| **OSNet-x1_0** | OSNet | 256×128 | **94.2 / 82.6** | **87.0 / 70.2** | **74.9 / 43.8** | torchreid — **2.2 M params, 0.98 GFLOPs** **[SRC]** |
| OSNet-x0_25 | OSNet | 256×128 | 91.2 / 75.0 | 82.0 / 61.4 | 61.4 / 29.5 | torchreid — **0.2 M params, 0.08 GFLOPs** **[SRC]** |
| BoT (R50) | ResNet50 | 384×128 (fast-reid cfg) | 94.4 / 86.1 | 87.2 / 77.0 | 74.1 / 50.2 | [fast-reid MODEL_ZOO](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) **[SRC]** |
| BoT (R50-ibn) | ResNet50-IBN | 384×128 | 94.9 / 87.6 | 89.3 / 79.6 | 77.0 / 54.4 | fast-reid **[SRC]** |
| BoT (S50) | ResNeSt50 | 384×128 | 95.2 / 88.7 | 90.0 / 80.1 | 80.8 / 59.9 | fast-reid **[SRC]** |
| AGW (R50) | ResNet50 | 384×128 | 95.3 / 88.2 | 89.0 / 79.9 | 78.3 / 55.6 | fast-reid **[SRC]** |
| **SBS (R101-ibn)** | ResNet101-IBN | 384×128 | **96.3 / 90.3** | **91.9 / 83.6** | **84.8 / 62.8** | fast-reid *(Circle loss + backbone freezing + cutout/auto-aug + cosine annealing + soft-margin triplet + non-local + GeM)* **[SRC]** |
| **MGN** | ResNet50-IBN | 384×128 | **95.8 / 89.8** | **91.1 / 82.0** | **85.1 / 65.4** | fast-reid **[SRC]** |
| ViT-B/16 (TransReID baseline) | ViT-B | 256×128 | 95.1 / **87.9** | — | 81.8 / 61.0 | [arXiv:2102.04378](https://arxiv.org/abs/2102.04378) **[SRC]** |
| ViT-B/16 s=14 (overlapping patches) | ViT-B | 256×128 | — | — | 82.7 / 63.7 | TransReID **[SRC]** |
| TransReID (JPM+SIE) | ViT-B | 256×128 | **95.2 / 89.5** | — | **86.2 / 69.4** | TransReID **[SRC]** |
| CLIP-ReID (no re-rank) | ViT-B/16 + CLIP text prompts | 256×128 | 95.4 / 90.5 | — | 89.7 / 75.8 | CLIP-ReID (AAAI 2023) ⚠️ **[MEM]** — the official README was **404 through the proxy** this session |
| ResNeSt200 (TransReID's table) | ResNeSt200 | 256×128 | — | — | 83.5 / 63.5 | TransReID **[SRC]** |
| ResNet50 (TransReID's table) | ResNet50 | 256×128 | — | — | 75.3 / 51.3 | TransReID **[SRC]** |

**Three facts in that table matter for us, and none of them is the ranking:**

1. **Parameter count is not the accuracy driver.** OSNet-x1_0 (2.2 M params) beats plain ResNet50
   (23.5 M) on all three benchmarks (+6.3 R1 Market, +8.7 R1 Duke, +11.7 R1 MSMT17) and comes within
   ~2 R1 of ResNet101-IBN SBS on Market. **Our encoder is 2.43 M params** — squarely in OSNet-x1_0
   territory. The "our model is small, so it must be weak" worry is not supported. **[SRC]**
2. **Most of the ladder is architecture and loss, not scale.** The moves that produce the big jumps
   are: part striping (PCB/MGN), BNNeck + tricks (BoT), better pooling (GeM), Circle loss (SBS),
   transformer global attention (TransReID), CLIP text prompts (CLIP-ReID). **All are training-time
   changes.** We are forbidden to train — so the *ladder itself*, not just its top rung, is
   unreachable to us; what is reachable is the **matching and memory strategy** around a fixed rung.
   This is the quantitative justification for §3–§4 being where this project's work lives. **[SRC]**
3. **Part striping (MGN/PCB) is worth ~+3 mAP MSMT17** over the otherwise-similar SBS baseline
   (65.4 vs 62.8 at 85.1 vs 84.8 R1) — i.e. **part modelling helps, but the size of its benefit at
   the *holistic* benchmark is a few points, not tens.** Its real payoff is on the occluded
   benchmarks in §2, where the gap between occlusion-blind (TransReID 66.4/59.2) and
   part+visibility-aware (**KPR 79.8/67.1** = **+13.4 R1**, **KPR+SOLIDER 84.3/75.1** = **+17.9 R1**). **[SRC]**

**The single most useful calibration for our decision table:** on **Occluded-Duke**, the
*occlusion-blind* TransReID ViT-B — an ~86 M-parameter transformer — scores **66.4 R1 / 59.2 mAP**,
while part+visibility-aware CNN methods from years earlier (PGFA 51.4/37.3 → HOReID 55.1/43.8 →
ISP 62.8/52.3 → PFD 69.5/61.8) bracket it. **A large transformer with no occlusion mechanism is
mid-table on partial matching.** That is the empirical statement of "the mechanism, not the
backbone, is the bottleneck" that §7 item 13 acts on. **[SRC]**

> **⚠️ Protocol trap, verified by reading both sources:** **torchreid's ranking table carries no input-size column at all** — the `256×128` in my table for the OSNet / ResNet50 / MobileNetV2 / HACNN rows is inferred from its published config **filenames** (`…_256x128_…`), not stated beside the number; only the HACNN row is explicit, because the table itself prints `(160, 64)` for it. Meanwhile fast-reid's README states its BoT/AGW/SBS/MGN numbers were obtained with **1× V100**, PyTorch 1.6, CUDA 10.1 and points to `configs/<dataset>/…yml` for the settings I could not read. **Cross-zoo comparisons (torchreid OSNet vs fast-reid SBS) are therefore not like-for-like** — augmentation, input size and schedule differ. Treat the table as *magnitudes*, and never quote a cross-zoo pair as a controlled comparison. **[SRC]**

**⚠️ UNVERIFIED in this section / not re-derived here:** CUHK03 numbers under *both* protocols (the
detected-vs-labelled trap recorded in §8), DukeMTMC-reID numbers and Market-1501 accent results for
the TransReID family (the TransReID tables I retrieved cover MSMT17 / VeRi-776 / Market), and all
2024–2026 foundation-era numbers (SOLIDER, PersonViT, CLIP-ReID follow-ups) beyond what §2.2 already
carries from the Occluded-Duke leaderboard. **PCB's own paper numbers, MGN's original ACM MM table,
RPP and Part-Aligned are not re-quoted here** — they appeared in an earlier phase of this
investigation from sources not re-fetched in this pass, so where they are cited elsewhere in this
report they are tagged accordingly. The **CLIP-ReID** row is the one ladder entry I wanted primary
and could not get: **its GitHub README 404s through the proxy.**

---

## 10.9 One-shot / distractor-aware ReID — the mechanism catalogue (§4 has the rules; this is the evidence base)

§4 states the anti-latch rules and §10.3 places `distractor_margin` among five defensive families.
This section collects the **one-shot framing** and the **explicit distractor-penalising mechanisms**
that those rules approximate, because the difference between "we chose a margin" and "we chose a
margin *because* the literature's alternatives all require training" is exactly the kind of
justification this report is supposed to carry.

### 10.9.1 What makes a one-shot query robust

One-shot ReID is defined as **"only one labelled sample per identity"** (SPUE-Net, [arXiv:2104.09152](https://arxiv.org/html/2104.09152v1)) **[SRC]** —
which is precisely our regime: the user clicks a target once and there is no second labelled view.

| Idea | Mechanism | Source | Training-free? |
|---|---|---|---|
| **One-shot Post-rank OPtimisation** (POP) | a user supplies **one strong negative** (plus optional weak negatives) and the rank list is refined; reported **>30 % Rank-1** improvement on VIPeR and i-LIDS, converging **2.6× faster** than exhaustive search | Li & Wang, ICCV 2013 [PDF](https://staff.ie.cuhk.edu.hk/~ccloy/files/iccv_2013_reid.pdf) **[SRC]** | ✅ (my earlier arXiv guesses for this paper were wrong — see the note below) |
| **Self-paced uncertainty estimation** | one-shot suffers **model uncertainty** (few labels) *and* **data uncertainty** — image noise from severe occlusion/background = "irrelevant identity information", plus label noise from mislabelling; co-operative local **uncertainty + determinacy** estimation with self-paced pseudo-label expansion | SPUE-Net [2104.09152](https://arxiv.org/html/2104.09152v1) **[SRC]** | ❌ (trains), but its **uncertainty taxonomy is directly reusable as a gating vocabulary**: our `no_detections`/`all_gated`/`cost>accept`/`ambiguous` refraction reasons are an uncertainty decomposition |
| **Query-conditioned subspace instead of direct prompt similarity** | "viewpoint/illumination/occlusion/background changes can make **distractors look more similar than the true match**"; direct attribute-prompt similarity is "unreliable (biased, correlated, poorly calibrated)", so build a **query-dependent low-rank subspace** from prompts | PRAISE [arXiv:2610.27944](https://arxiv.org/html/2610.27944) **[SRC]** | ✅ (test-time) |
| **Don't trust one global vector** | CLIP-ReID's global `[CLS]` aggregation is "**fragile under occlusion**"; align intermediate patch tokens to anchors in CLIP text space so the model **selectively recombines spatially stable evidence and suppresses absent/corrupted regions** — no text descriptions or part labels needed. Result: Occluded-DukeMTMC **77.8 R1 / 68.3 mAP = +10.6 / +8.0 over CLIP-ReID** | SAGA-ReID [arXiv:2604.22190](https://arxiv.org/pdf/2604.22190.pdf) **[SRC]** | ✅ (test-time) |

**The transferable sentence from §10.9.1, and it is the same conclusion three independent 2024–2026
papers reach:** *a single aggregated embedding is the wrong object to compare under occlusion; the
right object is a set of spatially-localised evidences, each with its own reliability, fused by
explicit rules.* That is what §4 rules 1–3 implement, and it is what §7 item 3 changes in
`memory.py`. **[SRC]**

> **Note on my own errors, recorded so they are not repeated:** I twice guessed arXiv IDs for the
> one-shot line and both were **wrong** — `2002.04160` is a q-bio paper on BMI in Korean adults and
> `2203.04175` is a physics.optics paper on optical force. The POP paper above is cited by its
> **author-hosted PDF URL**, not by an arXiv ID. **Rule followed from here on:** an arXiv ID is
> written only when it was seen on a fetched page; otherwise the claim is tagged **[MEM]** or
> ⚠UNVERIFIED. **[SRC]**

### 10.9.2 The distractor problem, named

The anti-latch failure has a **published name and a purpose-built dataset**. **DAM4SAM**
([arXiv:2509.13864](https://arxiv.org/abs/2509.13864)) calls it **"tracking drift toward
distractors"**, states that distractors "pose a key challenge", and answers it with a
**distractor-aware memory module plus introspection-based memory management**, evaluated on a
purpose-built **`DiDi` (Distractor-Distilled)** dataset — reporting **+11 % on EfficientTAM, +4 % on
EdgeTAM, SOTA on ten benchmarks**. **[SRC]**

Two things follow, and they point in opposite directions from our current design:

- **The literature's strongest answer is a memory, not a threshold.** DAM4SAM stores the
  distractor and *uses* it; our `hard_negative_bank` (capacity 64, `hard_negative_penalty 0.35`) is
  the training-free approximation of exactly that, which §7 item 7 already proposes to tune. Good.
- **A margin rule is the weakest of the five families** — it is the only one that makes a decision
  from a *single frame's* score difference, whereas (b) learned verification, (c) distractor memory,
  (d) re-detection with deferred commitment and (e) co-occurrence all accumulate evidence over
  time. §10.3 already flags (e) as "missing — add it"; **that free addition is the highest-leverage
  remaining item in the anti-latch family**, because the MOT layer already knows which detection
  belongs to which confirmed track.

**The honest gap, restated with the new evidence:** **no retrieved paper measures "how often the
tracker latches onto a bystander" as a proportion of frames.** DAM4SAM reports *relative* gains on
ten benchmarks; **OxUvA** ([arXiv:1803.09502](https://arxiv.org/abs/1803.09502)) formalises the
*present/absent* decision and is the closest thing to an evaluation protocol for refusals, but it
does not measure false acquisition either. So **`false_acquisition_rate` remains a metric this
project must calibrate itself**, and `distractor_margin = 0.06`
(`config.py:168`, `association.py:601-607`) has **no published value to copy** — worse, because the
margin is an absolute difference of *fused* costs, changing `stm_weight`/`ltm_weight` silently
changes the rule (§10.3). The only physically meaningful scale remains the **appearance-only** gap,
measured locally: **same-person `full` 0.757 vs different-person `upper` 0.48 ⇒ ≈0.28.**

**Concrete proposal that the new sources make possible:** replace the absolute fused-cost margin
with **two independent gates** — (i) an **appearance-only relative margin** on the metric whose
scale we measured, and (ii) a **family-(e) co-occurrence veto**: if the best appearance candidate's
MOT track is already occupied by a different confirmed target in the same frame, refuse regardless
of vote. The veto is free (the track layer already has the assignment; `assign_hungarian` exists in
`association.py`), it cannot be defeated by a single high-scoring impostor, and it is the one
mechanism in this section that the literature independently endorses and that costs us no model.

---

## 10.10 The gallery-policy questions §3 asked but did not answer

§3 documents what `memory.py` does and lists five gaps; §5 of the original brief asks four specific
**policy questions** — *how many entries, when to add, when to evict, when to freeze*. §3 answers the
first two only indirectly, so they are answered here against the sources actually retrieved.

| Question | Our current answer (`config.py`) | What the retrieved literature supports | Gap / action |
|---|---|---|---|
| **How many entries?** | `ltm_capacity = 96`, per-part banks, `hard_negative_capacity = 64` | **A fixed-size appearance gallery of ~100 descriptors per track is the field baseline** (DeepSORT/BoT-SORT). **Bi-C2R** (arXiv:2512.25000) and backward-compatibility (arXiv:2403.10022) establish *why* a bound exists: gallery images **cannot be re-encoded or re-indexed** (privacy + cost), so the gallery must be **feature-bounded, not image-bounded** | **96 is well-chosen** — it is at the field's standard order of magnitude and §6.4 shows it costs 197 KB. **Do not enlarge it**; the binding constraint is the encoder, not storage |
| **When to add?** | via `add`: frozen → unusable → `quality < 0.35` → diversity (reject if same-part cosine distance ≤ 0.15) | The literature's rule is **quality *and* consistency**, not novelty alone: the error-resilient recurring gallery ([IEEE 9506241](https://ieeexplore.ieee.org/document/9506241)) applies an **outlier test against the existing identity** before inserting, then **replaces** the matched descriptor with the query descriptor on match. §3.2-2 and §10.2(a) both flag the same hole from two directions | **Add the membership test** (`dist(feat, STM_EMA) ≤ τ_insert`) — currently a distractor mis-associated once is banked permanently |
| **When to evict?** | `_eviction_index` minimises `quality · recency`, half-life 600 frames | **No source retrieved defines a published eviction policy.** §10.4 records this explicitly: LReID is a *training-time* problem and does not define gallery eviction; `quality_recency` is **unproven, not standard** | **A/B `fifo` vs `quality_recency` on our own metrics** — both already exist (`config.py:127-135`). Also fix the absolute-frame decay clock (§10.2(b)) and protect ≥1 entry per part (§3.2-3) |
| **When to freeze?** | `frozen_states = ("occluded_soft","lost_short","lost_long","out_of_frame")`; `freeze(True)` no-ops inserts | **The best-published support in this report.** *Model Decay in Long-Term Tracking* ([arXiv:1908.01603](https://arxiv.org/abs/1908.01603)): *"Updating the tracker model with adverse bounding box predictions adds an unavoidable bias term … accumulation of this bias over a long-term can eventually lead to a permanent loss of the target"* — measured over 4 long-term benchmarks / 30-min videos | **The freeze is well-founded and should stay.** Two refinements stand: no LRU touch while frozen (§3.2-4) and a `guard_frames` window after re-appearance (§4 rule 5) |

**The one policy answer with a hard number attached:** a **96-entry × 512-d × 5-part** gallery costs
**983 KB** (96 × 512 × 4 B × 5 = 983,040 B) and a full query against every bank is **96 × 5 = 480
dot products of length 512 ≈ 0.25 M FLOPs** — i.e. **the policy questions are free to experiment
with, and the encoder is the entire cost.** This is the quantitative form of the BRIEF's "optimize
the strategy, not the network", and it is why §7's first eleven items are all config/strategy
changes rather than models. **[SRC arithmetic on measured/declared config values]**

**⚠️ UNVERIFIED here, stated plainly:** the *benefit* of any of these four policies over the others
is unmeasured on our data — no retrieved paper compares gallery eviction policies head-to-head, and
`C:\Projects\track\data\` contains **only `weights/`, `raw/`, `raw_pdf/` (no images)**, so no
target-vs-bystander experiment could be run in this session.

---
