# R3 — Single-Target (VOT) Tracking + Consumer-Drone "ActiveTrack" Setting

**Scope:** one user-clicked *person* tracked among many; the system must stay aware of all other persons as
distractors and must re-acquire the clicked target after occlusion / out-of-frame / fast nonlinear motion.
**Project constraints (from `docs/BRIEF.md`):** RTX 4060 Laptop 8 GB VRAM, torch 2.7.0+cu126,
ultralytics 8.4.185, onnxruntime-gpu 1.20.2; **no model fine-tuning allowed**; `huggingface.co` blocked
(`hf-mirror.com` works), GitHub release downloads unreliable (`https://ghproxy.net/https://github.com/...` works).

**Evidence policy.** Facts I read from primary sources in this session are marked **[SRC]**.
Numbers I am confident about but did **not** re-verify here are **[MEM]** (from the literature as I know it).
Anything I could not verify is marked **⚠ UNVERIFIED**. Marketing copy is labelled **[MKTG]** and never
presented as engineering fact.

---

## 0. Executive summary (the answer first)

1. **Architecture: two-layer, decoupled.** Keep a *detect-all-persons → MOT* layer for "awareness of all
   persons" (already available: BoT-SORT / Deep OC-SORT / `tracktrack.yaml` in ultralytics 8.4.185, all using
   `yolo26n-reid.onnx`), and add a **separate single-target tracking layer** that owns the clicked target.
   The two layers share the detector and the ReID embedder — this is what makes the system cheap.

2. **The target layer should be a *re-detection-first* tracker, not a template-matching tracker.**
   Every frame: (a) run YOLO26 person detection, (b) build candidate boxes, (c) score each candidate with
   *appearance* (ReID cosine to a quality-weighted gallery), *motion* (Kalman/IMM predictive IoU) and
   *distractor* terms, (d) pick the assignment. This is exactly the **GlobalTrack** philosophy (2019,
   arXiv 1912.08531) plus **DiMP/ATOM**-style online discriminative scoring — and it is the only family that
   survives "target absent for 200 frames", because there is no state to drift, only a score to exceed.

3. **Do NOT make an end-to-end SOT net (OSTrack / MixFormer / ODTrack) the primary tracker.** Reasons, all
   verifiable: (i) all of them are trained on GOT-10k/LaSOT/TrackingNet with **generic objects**, not
   person-specific identity — they will happily lock onto a similar person (they have no notion of
   "identity", only "same object"); (ii) they output one box and **cannot** express "absent" without a
   hand-tuned confidence threshold; (iii) they cost VRAM and latency competing with your detector. They are
   excellent **optional specialists** to add as a *verifier* or a *short-term gap filler* (see §7.3).

4. **Off-the-shelf weights that actually matter for this project (all verified in this session):**
   `yolo26n-reid.onnx` (already on disk, 9,873,245 B) for person identity; SAM 2.1 hiera-tiny/small
   (156 MB / 184 MB, Apache-2.0, reachable via `hf-mirror.com`) for **part-level / pixel-level** re-anchoring
   after partial occlusion. See §8 for URLs, sizes, licences.

5. **What DJI/Skydio practice does that papers miss:** the commercial systems are **not** pure trackers.
   They run a *re-detector* as the ground truth of "where is the subject", use the tracker only to bridge
   frames, and — critically — they **move the camera / gimbal and the aircraft** to keep the subject in a
   favourable regime (kept at a fixed apparent size, kept centered, kept away from the frame edge). The
   control loop is part of the tracker. Papers assume the camera is a passive observer (§9).

6. **Recommended architecture in one paragraph.** Keep the existing **detect-all-persons → MOT** layer
   (BoT-SORT / Deep OC-SORT / `tracktrack.yaml` / `fasttrack.yaml`, all driven by `yolo26n-reid.onnx`) as
   the "awareness" layer. On top of it, add a **decoupled single-target layer** that owns the clicked person
   and has four parts: **(1) a candidate scorer** over the detections already produced — `s_app` (cosine to
   a quality-weighted, diversity-aware embedding gallery), `s_mot` (IoU to an **IMM/Singer** predictive
   filter *after* camera-motion compensation), and `− s_dis` (an explicit distractor penalty against the
   other persons in frame); **(2) a margin-based accept gate** — accept only if `S(c*) > τ_acc` **and**
   `S(c*) − S(c₂) > τ_margin`, which is precisely what prevents hijacking a look-alike; **(3) a state
   machine** (§14) that separates *partial occlusion*, *short loss*, *long loss* and *re-acquisition* into
   distinct states with distinct actions and gates every gallery update; **(4) an on-demand SAM 2.1 tiny
   pass** on keyframes / during global search, giving part-level masks so that "only the upper body is
   visible" can still match and so that truncated crops cannot poison the appearance model. **No end-to-end
   SOT network (OSTrack / MixFormer / ODTrack) should be the decision-maker** — they are trained for "same
   object", not "same person", and cannot express "absent"; at most they are optional box/mask refiners
   operating on the candidate the target layer already chose.

---

## 1. Why the "single target among many persons" problem is different from MOT

| | MOT (all persons) | Single-target VOT (this project's target layer) |
|---|---|---|
| Output | N tracks, globally consistent IDs | 1 box + a validity/absent flag |
| Failure mode | ID switch between two persons | **Lock-on / hijack** onto the wrong person |
| Occlusion | Motion model coasts; ReID re-binds | Must decide *absent*, then **re-search globally** |
| Supervision signal | Detections from frame 0 | Only the **first-frame click** (one template) |
| Re-acquisition | Implicit — the track keeps being updated | Explicit state transition + global search |

The unit of failure is **identity**, not localisation. A tracker that reports a perfect box on the *wrong*
person has failed catastrophically; a tracker reporting *no* box for 20 frames and then the right person has
merely been conservative. Every design decision below is biased toward "no box" over "wrong box".

---

## 2. The correlation-filter era (2010–2019) — what is still worth stealing

| Tracker | Venue / year | Key idea | What survives into 2026 engineering |
|---|---|---|---|
| **MOSSE** | CVPR 2010 (Bolme et al.) **[MEM]** | Adaptive correlation filter, grayscale, ~600 FPS | The *base-template + running-average* update rule. **The failure mode it introduced is the one you must fix:** naive per-frame filter update ⇒ filter drift ⇒ lock-on. |
| **KCF** | TPAMI 2015 (Henriques et al.) **[MEM]** | Circulant structure + kernels, HOG | Sub-window search + peak-to-sidelobe ratio (PSR) as a *confidence* signal. PSR is still a decent cheap "am I locked?" test. |
| **DSST / fDSST** | BMVC 2014 / TPAMI 2017 (Danelljan et al.) **[MEM]** | Separate translation and **scale** filters (scale pyramid) | Decoupling scale from translation — your person's apparent size changes fast with drone range. Modern replacement: regress scale from the detection box. |
| **ECO** | CVPR 2017 — [arXiv 1611.09224](https://arxiv.org/abs/1611.09224) **[SRC]** | Factorised convolution operator + **Gaussian Mixture Model** of training samples + conservative update | The **sample-space sparsity** idea: keep a *diverse, small* set of templates rather than a dense buffer. This is exactly the right prior for a "long-term memory gallery" (§7.2). |
| **STRCF** | CVPR 2018 — [arXiv 1803.08679](https://arxiv.org/abs/1803.08679) **[SRC]** | Spatial-temporal regularisation; penalises deviation from the *previous* filter | Temporal smoothness of the *model*, not just the box. Cheap to emulate: L2-regularise your gallery update toward the previous embedding. |

**Zone-out on CF trackers.** They are 10–600 FPS on CPU, which is why OpenCV still ships them
(`cv2.legacy.TrackerMOSSE/KCF/CSRT` in `opencv-contrib-python`; the modern `cv2.Tracker*` API is the
`vittrack` model — see §8). For this project they are useful only as a **CPU-side fallback** when the GPU is
saturated, or as a **drift detector** (a CF response map collapsing is an early warning before ReID fails).
They cannot re-acquire. Never let one own the target.

---

## 3. The Siamese era (2016–2021) — the cross-correlation template matcher

| Tracker | Venue / year | Link | Contribution | Relevance here |
|---|---|---|---|---|
| **SiamFC** | ECCVW 2016 | [arXiv 1606.09549](https://arxiv.org/abs/1606.09549) **[SRC]** | Fully-convolutional cross-correlation of a template with a search region; no online update | The origin of "template ≠ identity". Its score map *is* a distractor map — multi-modal peaks = "there are several look-alikes". |
| **SiamRPN** | CVPR 2018 (Li et al.) **[MEM]** | — | Adds a **region-proposal** head (classification + regression) on top of Siamese features | Decouples *where* (RPN) from *what* (template). Multi-peak → multiple proposals → you can score each against ReID. |
| **DaSiamRPN** | ECCV 2018 | [arXiv 1808.06048](https://arxiv.org/abs/1808.06048) **[SRC]** | **Distractor-aware** training: hard negative mining of *similar* objects in the same frame + distractor-aware incremental learning | **The single most relevant Siamese paper for this project.** It is the first one that explicitly says "another instance of the same class is the hard negative". |
| **SiamRPN++** | CVPR 2019 | [arXiv 1812.11703](https://arxiv.org/abs/1812.11703) **[SRC]** | Depth-wise cross-correlation + layer-wise aggregation ⇒ ResNet backbones finally help; uniform sampling to break the centre bias | Still the default lightweight baseline; exported as ONNX in OpenCV's legacy `TrackerDaSiamRPN`. |
| **SiamDW** | CVPR 2019 | [arXiv 1901.01660](https://arxiv.org/abs/1901.01660) **[SRC]** | *Deeper and Wider* Siamese networks: CIR/RecRes units to stop the padding-induced centre bias | Architectural rule for any backbone swap you attempt without fine-tuning. |
| **SiamMask** | CVPR 2019 / TPAMI 2022 | [arXiv 1812.05050](https://arxiv.org/abs/1812.05050), [arXiv 2207.02088](https://arxiv.org/abs/2207.02088) **[SRC]** | Adds a **mask** branch ⇒ pixel-level target extent | Mask = the thing you need for "only the upper body is visible". This is the conceptual ancestor of the SAM-2 trackers (§6). |
| **SiamBAN** | CVPR 2020 | [arXiv 2003.06761](https://arxiv.org/abs/2003.06761) **[SRC]** | Anchor-free box-adaptive head for the Siamese family | Removes anchor hyper-parameters; the ancestor of most 2021+ box heads. |
| **SiamCAR** | CVPR 2020 | [arXiv 1911.07241](https://arxiv.org/abs/1911.07241) **[SRC]** | Fully-convolutional classification + regression, per-pixel | Simple, fast; useful as an ONNX-friendly head design. |
| **Ocean** | ECCV 2020 | [arXiv 2006.10721](https://arxiv.org/abs/2006.10721) **[SRC]** | Object-aware anchor-free: learns *target-aware* features via ground-truth-guided feature alignment | The label-alignment trick that later shows up in DiMP/ATOM-style offline training. |

**Where the whole family breaks for this project (important, and the reason §7 recommends otherwise):**
a Siamese tracker has *no concept of "this specific person"*. Given two people in the same jacket it produces
two equal correlation peaks; its online update is either "none" (SiamFC) or a short-term template average
(SiamRPN family), both of which drift onto the distractor after a couple of frames of occlusion. DaSiamRPN's
distractor suppression is a *training-time* mechanism — you cannot fine-tune, so you cannot adopt it directly.
You can, however, adopt its *inference-time* idea: **explicitly score every other detected person against your
target's appearance and suppress boxes that score high.**

---

## 4. The modern discriminative / transformer era (2019–2026)

### 4.1 Online discriminative model prediction — and why ATOM/DiMP are special here

| Tracker | Venue / year | Link | Key idea |
|---|---|---|---|
| **ATOM** | CVPR 2019 | [arXiv 1904.07220](https://arxiv.org/abs/1904.07220) **[SRC]** | "Learning Discriminative Model Prediction": decompose tracking into **target estimation** (IoU-Net) + **classification**; the classifier is *optimised per sequence at test time* |
| **DiMP** | ICCV 2019 (Bhat et al.) **[MEM]** | — | End-to-end learnable **optimiser** (meta-learning the gradient descent) for the discriminative classifier; per-sequence online adaptation |
| **PrDiMP** | CVPR 2020 (Danelljan et al.) **[MEM]** | — | Probabilistic formulation: predicts a **distribution** over the target, not a point ⇒ principled confidence and "absent" probability |
| **KeepTrack** | ICCV 2021 | [arXiv 2103.16556](https://arxiv.org/abs/2103.16556) **[SRC]** | *"Learning Target Candidate Association to Keep Track of What Not to Track"* — the paper refuses to let the target be defined by a template; it maintains **target-candidate associations** and can explicitly report "all candidates are distractors" |

> **The critical property for a no-fine-tuning project:** ATOM/DiMP/PrDiMP/KeepTrack do
> **test-time / online adaptation** — they fit a small discriminative model to *your* clicked target during
> the sequence. That is *not* fine-tuning; it requires no labels and no retraining. `KeepTrack` is the one
> that also answers "how do I know the target is gone?" — it explicitly models candidates and can reject
> all of them.

### 4.2 Transformer trackers

| Tracker | Venue / year | Link | Key idea / relevance |
|---|---|---|---|
| **SimTrack** | ECCV 2022 | [arXiv 2203.05328](https://arxiv.org/abs/2203.05328) **[SRC]** | *"Backbone is All Your Need"* — a plain ViT with a **single** target-search token, no correlation at all |
| **OSTrack** | ECCV 2022 | [arXiv 2203.11991](https://arxiv.org/abs/2203.11991) **[SRC]** | *"Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework"* — the de-facto strong generic baseline; early candidate elimination prunes background tokens |
| **MixFormer** | CVPR 2022 (ext. [arXiv 2302.02814](https://arxiv.org/abs/2302.02814)) | [arXiv 2203.11082](https://arxiv.org/abs/2203.11082) **[SRC]** | Mixed Attention Module (MAM) fuses feature extraction and target integration; asymmetric attention over templates. Reported **[SRC from README]**: MixViT-L **LaSOT AUC 73.3**, **TrackingNet AUC 86.1**, **VOT2020 EAO 0.584**, **GOT-10k AO 75.7** |
| **MixFormerV2** | NeurIPS 2023 | [arXiv 2305.15896](https://arxiv.org/abs/2305.15896) **[SRC]** | *"Efficient Fully Transformer Tracking"* — distils MixFormer into a **fully transformer, no-CV**, small model; the [MixFormerV2 README](https://github.com/MCG-NJU/MixFormerV2) quotes a **~3.6 M-parameter** variant **[MEM]** |
| **SeqTrack** | CVPR 2023 | [arXiv 2304.14394](https://arxiv.org/abs/2304.14394) **[SRC]** | *"Unified Sequence-to-Sequence Learning"* — tracking as **autoregressive box-sequence generation**; makes the tracker a sequence model, which is the right shape for "target disappears for k frames" |
| **ARTrack / ARTrackV2** | CVPR 2023 / 2024 | [arXiv 2312.17133](https://arxiv.org/abs/2312.17133) **[SRC]** | ARTrack: autoregressive *trajectory* prediction; **V2** adds prompting "where to look and how to describe" |
| **ODTrack** | AAAI 2024 | [arXiv 2401.01686](https://arxiv.org/abs/2401.01686) **[SRC]** | *"Online Dense Temporal Token Learning"* — carries tokens across frames; 2025 extension to universal modal tracking ([arXiv 2507.20177](https://arxiv.org/abs/2507.20177)) **[SRC]** |
| **EVPTrack** | Information Fusion 2024 | [arXiv 2401.03142](https://arxiv.org/abs/2401.03142) **[SRC]** | *"Explicit Visual Prompts"* — feeds the *backbone reference points* of the exemplar into the search features; surprisingly strong with a frozen backbone (attractive for no-fine-tune reuse) |
| **LoRAT** | CVPR 2024 | [arXiv 2403.05231](https://arxiv.org/abs/2403.05231) **[SRC]** | *"Tracking Meets LoRA"* — tunes a large ViT **with LoRA** instead of redesigning the architecture; CLIP-based, so the features are semantically rich. Also produced the Perception Test 2024 winning note ([arXiv 2410.16329](https://arxiv.org/abs/2410.16329)) |
| **HIPTrack** | CVPR 2024 | [arXiv 2311.02072](https://arxiv.org/abs/2311.02072) **[SRC]** | *"Visual Tracking with Historical Prompts"* — mines hard and *diverse* historical samples to keep the prompt set discriminative. Directly transferable: your long-term gallery needs **diversity**, not just recency |
| **SUTrack** | AAAI 2025 | [arXiv 2412.19138](https://arxiv.org/abs/2412.19138) **[SRC]** | *"Simple and Unified SOT"* — one model for RGB / RGB-D / RGB-T / RGB-Event / RGB-Language. The RGB-T and RGB-D variants matter for a drone that also has depth/thermal |
- ⚠ **FastTracker** (`2508.14370`, "FastTracker: Real-Time and Accurate Visual Tracking", 2025-08) **[SRC]** is
  the arXiv record for what this project's BRIEF lists as `fasttrack.yaml` (arXiv 2508.14370) — a
  *real-time, occlusion-aware* tracker with Kalman rollback. Note the arXiv title is "FastTracker", while
  ultralytics names the config `fasttrack.yaml`.
- ⚠ **"DropTrack" (CVPR 2025)** — ⚠ **UNVERIFIED** — I could not confirm an arXiv entry under this name; arXiv `all:"DropTrack"` returns only a *microfluidics* droplet-tracking paper ([2205.02568](https://arxiv.org/abs/2205.02568)) **[SRC]**. Treat the CVPR 2025 "DropTrack" citation as unconfirmed; do not cite it without the proceedings page. |
| **KYS** | CVPR 2022 (Zhou et al.) **[MEM]** | ⚠ **UNVERIFIED** — `ti:"KYS"` did not return it in this session | "Know Your Surroundings": exemplar + surrounding-region tokens. Cite the CVPR 2022 proceedings page, not an arXiv ID, until verified. |

### 4.3 2025–2026 successors worth watching

| Work | Link | Why it matters |
|---|---|---|
| **Distractor-Aware Memory-Based VOT** | [arXiv 2509.13864](https://arxiv.org/abs/2509.13864) **[SRC]** | Names *your* problem exactly: memory banks must be **distractor-aware**. Read this before designing your gallery update rule. |
| **A Distractor-Aware Memory for VOT with SAM2** | [arXiv 2411.17576](https://arxiv.org/abs/2411.17576) **[SRC]** | SAM-2 memory + explicit distractor handling |
| **Towards Universal Modal Tracking with Online Dense Temporal Token Learning** | [arXiv 2507.20177](https://arxiv.org/abs/2507.20177) **[SRC]** | ODTrack's 2025 extension; multi-modal robustness |
| **Exploiting Lightweight Hierarchical ViT and Dynamic Framework for Efficient Visual Tracking** | [arXiv 2506.20381](https://arxiv.org/abs/2506.20381) **[SRC]** | Directly targets the *efficiency* axis you are constrained by |
| **Drift-Resilient Temporal Priors for Visual Tracking** | [arXiv 2604.02654](https://arxiv.org/abs/2604.02654) **[SRC]** | 2026-era explicit anti-drift design — the failure mode of every online-update tracker |
| **SAM2Long** | [arXiv 2410.16268](https://arxiv.org/abs/2410.16268) **[SRC]** | Training-free **memory tree** over SAM-2, with a constrained tree-search memory-selection rule; fixes SAM-2's "error accumulation in long videos" |
| **SAMURAI** | [arXiv 2411.11922](https://arxiv.org/abs/2411.11922) **[SRC]** | Zero-shot SAM-2 tracking with **motion-aware memory** (motion scores + affinity) — no training, and it handles distractors |
| **NT-VOT211** | [arXiv 2410.20421](https://arxiv.org/abs/2410.20421) **[SRC]** | Large-scale **night-time** VOT benchmark (211k videos) — relevant to real drone use, which is where appearance-only ReID dies |
| **SurgSLOT** | [arXiv 2511.16618](https://arxiv.org/abs/2511.16618) **[SRC]** | Long-term tracking with SAM semantics in a **different domain** — useful as a template for how to bolt semantics onto long-term tracking |
| **ReID-SAM (SkiTB 2025)** | [arXiv 2503.01907](https://arxiv.org/abs/2503.01907) **[SRC]** | A challenge report that combines **ReID + SAM** for tracking — the same two ingredients this project already has |
| **UAUTrack / anti-UAV 2025–2026** | [arXiv 2512.02668](https://arxiv.org/abs/2512.02668), [arXiv 2607.26511](https://arxiv.org/abs/2607.26511) **[SRC]** | Unified multi-modal anti-UAV tracking; the 2026 paper explicitly models **both camera platform and target moving simultaneously** — the airborne-observer regime |

---

## 5. SAM-based trackers (SAM 2 / SAMURAI / SAM2Long)

**SAM 2** ("Segment Anything in Images and Videos") — [arXiv 2408.00714](https://arxiv.org/abs/2408.00714) **[SRC via HF model card tag `arxiv:2408.00714`]**, Apache-2.0, published on HF as `facebook/sam2.1-hiera-*` **[SRC]**.

What it gives this project that nothing else does:

- **Promptable, mask-based, part-level.** You can prompt with the *upper-body* box and get a mask that says
  "this is the visible part of that person". That is the direct answer to BRIEF pillar
  *"must track effectively when only part is visible"*.
- **A memory bank with a per-frame memory-attention mechanism** — the architectural template for your
  long-term gallery, including the *masked attention* trick: only attend to memory tokens that were
  positively matched.
- **Zero-shot.** No fine-tuning needed. This fits the "no model fine-tuning" rule exactly.

Its failure mode, and the reason SAMURAI/SAM2Long exist: **naive memory writes accumulate errors**; once a
distractor is written into memory, the tracker locks on. SAMURAI fixes it with a **motion-aware memory**
(motion score + motion-aware affinity); SAM2Long fixes it with a **memory tree + constrained search** over
which memory to read, plus a "memory quality" gate. Both are **training-free** — they are inference-time
algorithms, i.e. exactly the kind of thing this project is allowed to do.

**Cost.** On 8 GB VRAM, `sam2.1_hiera_tiny` (155.9 MB of weights) is the only comfortable choice; `hiera_small`
(184.4 MB) is feasible at reduced resolution; `base_plus` / `large` are not. Do **not** run SAM-2 per frame as
the main tracker — run it only on a **keyframe / on-demand** basis, driven by the state machine (§10).

---

## 6. Long-term tracking — "the target left / is fully occluded / came back"

This is the core of the project and the place where the academic literature is most directly usable.

| Tracker / benchmark | Venue / year | Link | Mechanism for the *absent* state |
|---|---|---|---|
| **GlobalTrack** | AAAI 2020 | [arXiv 1912.08531](https://arxiv.org/abs/1912.08531) **[SRC]** | *"A Simple and Strong Baseline"*: **no online update at all**; every frame is a *global* search with the first-frame template. Absence is implicit — no candidate exceeds the threshold |
| **SPLT** | ICCV 2019 | [arXiv 1909.01840](https://arxiv.org/abs/1909.01840) **[SRC]** | *"Skimming-Perusal Tracking"*: an **explicit two-stage** design — a *skimmer* proposes candidate regions, a *verifier* decides. This is the cleanest published precedent for a LOST→RE_ACQUIRE state machine |
| **LTMU** | CVPR 2020 | [arXiv 2004.00305](https://arxiv.org/abs/2004.00305) **[SRC]** | *"High-Performance Long-Term Tracking with Meta-Updater"*: a **meta-updater** decides *when to update* the appearance model — the central anti-drift control. The paper's own framing is "update or not", which is a binary control policy you can implement directly |
| **KeepTrack** | ICCV 2021 | [arXiv 2103.16556](https://arxiv.org/abs/2103.16556) **[SRC]** | Target-candidate association; can reject *all* candidates ("what not to track") |
| **MFT** | WACV 2024 | [arXiv 2305.12998](https://arxiv.org/abs/2305.12998) **[SRC — arXiv comment: "accepted to WACV 2024", code github.com/serycjon/MFT]** | *"Long-Term Tracking of Every Pixel"* — pixel-level long-term tracking; the mask version of GlobalTrack |
| **TCTrack / TCTrack++** | CVPR 2022 | [arXiv 2203.01885](https://arxiv.org/abs/2203.01885) **[SRC — arXiv comment: "To appear in CVPR2022"]** | **Aerial** tracking: temporal context via online temporal adaptive convolution + adaptive temporal transformer; handles *abrupt camera motion* by *explicitly* modelling it |
| **LaSOT** | CVPR 2019 | [arXiv 1803.09502](https://arxiv.org/abs/1803.09502) **[SRC]** | The long-term benchmark whose videos include full occlusion and out-of-view; it defines the average-length-≥1000-frames regime |
| **VOT-LT** | VOT2020 challenge track | [votchallenge.net](https://www.votchallenge.net/) **[MEM]** | The actual *long-term* protocol: reports **precision**, **recall** and **F1** instead of IoU, because IoU is undefined when the target is absent — **adopt this metric choice** for the target layer |
| **Model Decay in Long-Term Tracking** | 2019 | [arXiv 1908.01603](https://arxiv.org/abs/1908.01603) **[SRC]** | The paper that names your #1 enemy: template/model decay over long sequences |

### 6.1 How the literature represents "target absent"

Three distinct conventions — know which one you are using:

1. **Implicit / energy-threshold** (GlobalTrack, SiamFC-family): the score map has no peak above τ ⇒ report
   absence. Cheap, but τ is dataset-dependent and the box is still emitted internally (drift risk).
2. **Explicit state / verifier** (SPLT, LTMU): a separate classifier decides "target / background" for the
   proposed region; absence is a *classification output*.
3. **Probabilistic** (PrDiMP): the tracker predicts a *distribution*; absence = low total probability mass in
   a plausible region. The most principled, and the easiest to threshold with a calibrated confidence.

### 6.2 The literature's numbers you should NOT blindly trust

Long-term benchmarks are dominated by **LaSOT with generic objects**. A high LaSOT AUC does **not** imply
"will not hijack a similar person". Two concrete reasons: (a) LaSOT distractors are other object classes;
(b) most long-term protocols evaluate IoU on frames where the target *is* present, so a tracker that silently
hijacks a distractor during the occlusion and comes back is only penalised through the few post-occlusion
frames. **This is the single most important caveat in this whole report.**

---

## 7. Distractor-aware tracking — the exact failure this project must not have

The task statement is explicit: *"The project must not latch onto a similar-looking person."* Four bodies of
work address this directly.

### 7.1 DaSiamRPN (ECCV 2018) — distractor-aware *training*

[arXiv 1808.06048](https://arxiv.org/abs/1808.06048) **[SRC]**. Two mechanisms: (a) **hard negative mining**
of same-class, similar-appearance instances within the same frame during training; (b) **distractor-aware
incremental learning** at inference — when the target is lost, the model suppresses the distractors it can
now see. Since you cannot fine-tune, take (b)'s *inference-time* form: build an explicit
**distractor set** `D = {other detected persons}` each frame, and subtract their appearance from your
target's matching score.

### 7.2 LTMU (CVPR 2020) — the "when to update" control problem

[arXiv 2004.00305](https://arxiv.org/abs/2004.00305) **[SRC]**. LTMU's meta-updater is a *learned gate*
that answers: given the current candidate, should I update the appearance model? The engineering
translation, without training, is a **hand-written gate** on your gallery update:

```
update_gallery  iff  (match_score > τ_high) AND (motion_residual < τ_mot) AND (not occluded)
no-update       iff  match_score ∈ [τ_low, τ_high)          # ambiguous → freeze
purge + search  iff  match_score < τ_low for k consecutive frames
```

That is LTMU's policy with a hand-tuned gate instead of a meta-network, and it is the highest-value
single idea in the long-term literature for a no-fine-tuning system.

### 7.3 DiMP / PrDiMP hard-negative mining and the discriminative formulation

DiMP **[MEM]** and PrDiMP **[MEM]** train the *discriminative classifier* with hard negatives drawn from
*the same sequence* — i.e. they optimise "target vs. this specific background/distractors", not "target vs.
random background". PrDiMP's probabilistic head gives the "absent" test a principled form.

**Practical inference-time transplant (no training needed):**
1. Each frame, from the detect-all-persons output, form the candidate set `C = {c_1..c_N}`.
2. `s_app(c_i) = cos( emb(c_i), G )` where `G` is your quality-weighted gallery.
3. `s_mot(c_i) = IoU(c_i, Kalman/IMM-predicted box)`.
4. `s_dis(c_i) = max_{j≠i, j∈C} cos( emb(c_i), emb(c_j) )` — how *confusable* this candidate is.
5. Target score `S(c_i) = w_a·s_app + w_m·s_mot − w_d·s_dis` (the last term is DaSiamRPN, executed at
   inference rather than in training).
6. Accept iff `S(c*) > τ_acc` **and** `S(c*) − S(c_2) > τ_margin` (the margin test is what prevents
   swapping between two near-identical people).

The **margin term** is the piece most published single-target trackers do not have, because they were never
evaluated on "two identical jackets in one frame".

### 7.4 The 2024–2025 papers that name this problem

- **Distractor-Aware Memory-Based VOT** — [arXiv 2509.13864](https://arxiv.org/abs/2509.13864) **[SRC]**
- **A Distractor-Aware Memory for VOT with SAM2** — [arXiv 2411.17576](https://arxiv.org/abs/2411.17576) **[SRC]**
- **KeepTrack** (ICCV 2021) — [arXiv 2103.16556](https://arxiv.org/abs/2103.16556) **[SRC]**
- **HIPTrack** (CVPR 2024) — [arXiv 2311.02072](https://arxiv.org/abs/2311.02072) **[SRC]**: *diverse*
  historical prompts, not merely recent ones.

**Cross-domain confirmation that distractor-awareness is the real difficulty:** the event-based branch has an
explicit *"Distractor-aware Event-based Tracking"* paper ([arXiv 2310.14194](https://arxiv.org/abs/2310.14194))
**[SRC]**, and a *transparent-object* distractor-aware architecture/data set
([arXiv 2401.03872](https://arxiv.org/abs/2401.03872)) **[SRC]**. The problem is recognised across
modalities; the solution each time is **explicit negative modelling**, never a better backbone.

### 7.5 Additional lever: multi-object global assignment as a distractor filter

Because your project already runs a *full* MOT layer, you can use the **global assignment** solution as a
constraint: the clicked target occupies exactly one person slot. If your target-matching says "the target is
person #7" but the MOT layer's Hungarian assignment has already claimed #7 as an existing different track,
the conflict is evidence of a possible hijack — a cross-check that single-object trackers structurally
cannot perform. This is free, and it is not in any single-object tracking paper.

---

## 8. UAV / drone tracking: benchmarks and SOTA

### 8.1 The benchmarks

| Benchmark | Venue / year | What it is | Why it matters here |
|---|---|---|---|
| **UAV123** | ECCV 2016 workshop (Mueller, Smith, Ghanem) **[MEM]** — primary arXiv entry not retrievable this session | 123 short aerial sequences, 12 attributes (aspect change, background clutter, camera motion, fast motion, partial/full occlusion, out-of-view, scale variation, viewpoint change, …) | **The four attributes that matter to you are all present**: camera motion, fast motion, full occlusion, out-of-view. Adopt the attribute breakdown as your *test suite design*. |
| **UAV20L** | ECCV 2016 workshop **[MEM]** | 20 **long** aerial sequences | The long-term aerial protocol — the closest published analogue to "drone follows one person for minutes" |
| **DTB70** | ICVS 2017 (Li, Yeung et al.) **[MEM]** — corroborated this session: numerous UAV trackers evaluate on DTB70 (e.g. [arXiv 2308.10262](https://arxiv.org/abs/2308.10262), [arXiv 2308.11450](https://arxiv.org/abs/2308.11450), [arXiv 2104.03114](https://arxiv.org/abs/2104.03114), [arXiv 2207.01768](https://arxiv.org/abs/2207.01768)) **[SRC]** | 70 drone sequences, generic object tracking from a drone | Small targets + fast camera motion |
| **VisDrone** | ECCV 2018 workshop / TPAMI 2021 **[MEM]** — corroborated: [arXiv 2107.08766](https://arxiv.org/abs/2107.08766), [arXiv 2202.11983](https://arxiv.org/abs/2202.11983) **[SRC]** | Detection (10 classes incl. pedestrian) + multi-object tracking, from drones | **Your training/eval source for "detect all persons from a drone"**; it is *the* drone MOT benchmark |
| **Anti-UAV** | 1st–3rd workshops 2020–2023, [arXiv 2108.09909](https://arxiv.org/abs/2108.09909) / [arXiv 2305.07290](https://arxiv.org/abs/2305.07290) **[SRC]** | RGB/TIR tracking of a *flying* UAV, with **both camera and target moving** | The "dual-moving" regime; the 2026 paper [arXiv 2607.26511](https://arxiv.org/abs/2607.26511) **[SRC]** states explicitly that "both the camera platform and the target move simultaneously" *differs from conventional UAV tracking* — i.e. the field itself admits general UAV tracking benchmarks under-model your regime |
| **NT-VOT211** | ICCV? 2024, [arXiv 2410.20421](https://arxiv.org/abs/2410.20421) **[SRC]** | 211k night-time VOT videos | Drone tracking at night is a real deployment case; appearance-only ReID degrades badly |
| **VOT-LT** | VOT challenge 2020 **[MEM]** | Long-term protocol; **precision / recall / F1** | Use these three metrics, plus re-acquisition latency, for the target layer |

### 8.2 What "UAV-tuned" actually changes

The UAV tracker literature ([TCTrack, AAAI 2022, arXiv 2203.01885](https://arxiv.org/abs/2203.01885) **[SRC]**,
AutoTrack [arXiv 2003.12949](https://arxiv.org/abs/2003.12949) **[SRC]**, and the many DTB70 papers above)
converges on the same four modifications:

1. **Temporal context / abrupt-motion modelling.** TCTrack's whole thesis is that aerial footage has
   *camera-induced* apparent motion that a single-frame template matcher cannot handle; it adds
   online temporal adaptive convolution + an adaptive temporal transformer.
2. **Spatio-temporal regularisation** (AutoTrack) to suppress the "filter learns the background" failure when
   the camera moves.
3. **Lightweight backbones** because the compute is on the aircraft.
4. **Scale handling** because a drone's range changes fast.

> **Mapping to your project:** you have an *additional* signal no VOT benchmark provides — the **drone's own
> IMU and gimbal attitude**. TCTrack *learns* the camera-induced motion; you can *measure* most of it and
> subtract it before running any appearance model. That is a strictly easier problem, and it is exactly
> what §9 shows the commercial systems doing.

### 8.3 SOTA numbers you may quote (with the right caveat)

- MixViT-L (supervised pre-training): **LaSOT AUC 73.3 / TrackingNet AUC 86.1 / GOT-10k AO 75.7 /
  VOT2020 EAO 0.584** — **[SRC]**, read from the MixFormer README table reproduced in this session.
- SAM 2.1 Tiny J&F: **SA-V val 75.1 / SA-V test 76.3 / LVOS v2 81.6**; **SAM2Long** raises Tiny to
  **78.9 / 79.0 / 82.4** and Small to **79.6 / 80.4 / 84.3** — **[SRC]**, from the SAM2Long README tables.
  (Average improvement ≈ **3 J&F** across 24 head-to-head comparisons on SA-V and LVOS **[SRC]**.)
- ⚠ **UNVERIFIED / do not quote without a source:** specific UAV123 AUC values for individual trackers.
  I did not retrieve a verified leaderboard in this session. Any "UAV123 AUC = x" number in this report's
  successor documents must be traced to a paper table, not to memory.

---

## 9. DJI / Skydio / consumer-drone industry practice — what is verifiable

This section deliberately separates **[SRC]** (read from official or primary material) from **[MKTG]**
(vendor marketing) and **[INF]** (my inference from the two).

### 9.1 DJI — verifiable, from DJI's own support documentation

**Verified [SRC]** (DJI support article on FocusTrack/ActiveTrack, `support.dji.com`, article
`en-us03400006832`):

- **FocusTrack is an umbrella with three sub-modes: Spotlight, ActiveTrack, and Point of Interest.**
  - **Spotlight**: locks the gimbal on the subject; *the pilot flies manually*. The tracker's only output is
    camera pointing.
  - **ActiveTrack**: the aircraft follows. Two sub-behaviours: **Trace** ("tracks the subject at a constant
    distance") and **Parallel** ("follows beside").
  - **Point of Interest**: orbits a point.
- **Explicit obstacle-avoidance behaviour during tracking**: *"During the flight, the drone will
  automatically calculate the distance from obstacles, plan the route, and avoid the obstacles."*
- **Explicit documented failure mode**: the article ends with *"if issues such as **inability to recognize
  the tracking target** or **tracking target loss** occur, you can troubleshoot by referring to the FAQs
  about Using FocusTrack/ActiveTrack."* ⇒ DJI ships a **documented LOST state** and a user-facing recovery
  procedure. This is confirmation from the vendor that target loss and re-recognition are first-class states.
- The DJI GO 4 (legacy) article documents the **same three modes** on older products (Mavic 2/Mavic Pro/
  Spark/Phantom 4) ⇒ the *mode decomposition* is a long-lived design, not a per-product gimmick.
- Historically documented in this session's research (verbatim from DJI support material): **SLOW** tracking
  setting capped at **12 m/s with obstacle sensing**, **FAST** up to **20 m/s without obstacle avoidance**
  **[SRC]**; and **Subject Scanning** enabled through a gesture (arm-wave) **[SRC]**.

**Verified DJI patents [SRC]** (found via patent aggregator in this session; Google Patents itself was
unreachable — direct fetch timed out):

- **US11216958B2 — "Tracking method and device"**: obtain *first* characteristic information of the target
  through a **first camera**; obtain *second* characteristic information through a **second camera**; track
  the target using **at least one of** the two. [patents.searchlight.law/doc/US11216958](https://patents.searchlight.law/doc/US11216958)
  → **The claim is literally "multi-camera appearance fusion for tracking"**, i.e. the vendor's answer to
  "the target disappeared from camera A" is "look at camera B", not "improve the tracker".
- **US11776139B2 — "Tracking method and device"**: track through camera 1; read the **gimbal attitude** of
  camera 1; **adjust camera 2's gimbal to maintain attitude consistency** between the two cameras.
  [patents.searchlight.law/doc/US11776139](https://patents.searchlight.law/doc/US11776139)
  → Gimbal attitude is an explicit, claimed state used for *cross-camera handoff*.
- **US9164506B1** — a DJI target-tracking-and-following patent (seen in search results this session; title
  and number recorded **[SRC]**, claim text not read ⇒ treat the specific claims as **⚠ UNVERIFIED**).

**Marketing claims [MKTG] — do not treat as verified:** ActiveTrack 360 / "omnidirectional sensing",
"ActiveTrack 5.0 ... combines machine learning prediction with depth mapping to anticipate where subjects
will move", "Subject Scanning", and the various "AI re-identification" phrasings on DJI product pages.
These come from DJI product/support marketing and third-party blogs; no technical whitepaper with numbers was
retrieved. The *mode structure* and the *loss/recovery behaviour* are verified; the *algorithm* is not.

### 9.2 Skydio — verifiable, from Skydio's own material

**Verified [SRC]** (fetched from `skydio.com/skydio-autonomy` and Skydio support in this session):

- Skydio's autonomy is described as an **architecture, not a feature**: the aircraft *"continuously reason[s]
  in 3D, building an understanding of the environment mid-air"*, *"constantly map[s] the environment as they
  fly through it"*, avoids *"obstacles as small as a ½-inch wire"*.
- Two named subsystems: **Pathfinder** ("autonomously plans and executes the best flight path", factoring
  terrain and buildings) and **Foresight** ("anticipates the right move"; looks ahead along the route and
  re-plans while there is still distance to manoeuvre).
- Skydio support documents **"Shadow Subject Track and Follow"**: *"The drone autonomously identifies and
  maintains visual lock on a selected subject while managing camera framing and orientation"* **[SRC]**.
- A search-engine snippet quoted Skydio describing reacquisition *"even through brief occlusions"* by
  combining speed, direction and visual appearance cues **[SRC - secondary]**. Treat the *wording* as
  secondary; the *capability claim* is consistent with the support documentation.
- ⚠ Skydio's deeper engineering blog pages were **not fetchable** in this session (blocked/timeout).

**Inference [INF]:** Skydio's subject tracking is *one client of a 3D-reconstructed world model*. The
subject is a *3D object with an estimated position and velocity*, and camera pointing + flight planning are
solved **jointly** with tracking. This is why their system can fly behind a building and still come out
pointing at the subject: it never needed to see the subject; it needed to predict where it would be.

### 9.3 The two industrial moves that papers systematically omit

1. **Act on the camera, not just the box.** Both DJI (gimbal / aircraft) and Skydio (flight path) treat
   *where the camera points* and *how far away the aircraft is* as control variables of the tracking loop.
   Academic VOT is defined on a **passive** camera with a fixed field of view; the closest academic analogue
   is *active tracking* / *active vision*, and it is a much smaller literature. **Consequence for this
   project:** if your eventual deployment has a gimbal, expose a control output (yaw/pitch rate, desired
   apparent size) from the tracker state machine — a "TRACKING" state that also emits a gimbal command has
   far fewer loss events than one that only emits a box.
2. **Maintain a geometry/range state, not just a 2D box.** A subject at constant *apparent size* is a
   subject whose appearance statistics stay stationary — which is precisely the condition under which a
   ReID gallery stays valid. DJI's "Trace ... at a constant distance" is not a convenience feature; it is
   **appearance-model maintenance**. Papers rarely state this because their camera cannot move.
3. **A documented LOST state with a user-visible recovery path** (DJI's own FAQ) is a product requirement
   that most papers do not even evaluate. Your state machine should be *user-legible*: "searching",
   "re-acquired", "give up" are UI states.

---

## 10. Nonlinear / fast motion handling

### 10.1 What the tracking literature assumes, and where it breaks

Almost every modern SOT tracker either (a) has **no** motion model (Siamese/transformer family — just a
search window around the previous box), or (b) uses a **constant-velocity Kalman filter** (SAMURAI
explicitly: *"The Kalman filter is used to estimate the current and future state (bounding box location and
scale) of a moving object based on measurements over time ... it is a common approach that had been adopted
in the field of tracking for a long time, which does not require any training"* **[SRC]**, from the SAMURAI
FAQ). At **camera frame rate** constant velocity is a decent approximation *for a person walking*; it is
bad for: direction reversal, sudden acceleration, and any motion that is really the camera's.

### 10.2 Motion models worth considering (ordered by value/cost for this project)

| Model | What it is | When it wins | Cost |
|---|---|---|---|
| **Constant velocity (CV) KF** | state `[x,y,w,h,vx,vy]`, constant process noise | Steady walking | ~0. Always have it. |
| **Constant acceleration (CA) KF** | adds `[ax,ay]` | Smooth but accelerating motion (person starting to run) | ~0 |
| **Singer / exponentially-decaying acceleration** | acceleration is an OU process; interpolates CV↔CA via a manoeuvre-time constant | The **best single-value choice** for a person: zero lag when steady, responsive when manoeuvring | ~0, one tunable τ |
| **Constant-turn (CT) model** | adds a turn-rate `ω` | Orbiting / circling subject (very common in drone footage!) | low |
| **IMM (Interacting Multiple Model)** | runs CV+CA+CT (+ stationary) in parallel with a Markov mode transition, outputs the probability-weighted mixture. **Stationary is a critical mode**: a person who stops is *not* CV, and a stopped person is the hardest case for a motion-gated re-detector | Abrupt switches between walking / stopping / turning. **This is the principled answer to "handle abrupt acceleration / direction reversal"** | ~N× the cost of one KF, N≈3–4; still microseconds |
| **Particle filter (PF)** | non-parametric posterior over the state | Multimodal posteriors — i.e. *"the target is either behind that car or in that doorway"*. The **only** model that represents ambiguity natively. Use a PF *on top of detection candidates* (each detection seeds/weights particles) rather than a blind PF over pixels | higher, and needs careful resampling; feasible at low particle counts (100–500) |
| **Learned motion predictors** | e.g. autoregressive trajectory heads in **ARTrack/V2** ([arXiv 2312.17133](https://arxiv.org/abs/2312.17133)), sequence prediction in **SeqTrack** ([arXiv 2304.14394](https://arxiv.org/abs/2304.14394)) **[SRC]** | They learn the *statistics* of object motion from LaSOT/TrackingNet | Requires running the model; their motion prior is *generic-object*, not pedestrian |
| **Gaussian-process / continuous-time motion** | Regress a smooth trajectory with calibrated uncertainty; the uncertainty *is* the search-region size | Gives a principled, growing search radius over time — exactly the "coast then search wider" behaviour | low–moderate; GP inference is cheap at these dimensions |
| **OCM direction consistency** (already in this project's brief) | Consistency of the optical-flow/centre-direction over a window | Detects *decisions* like "the subject turned around" earlier than a KF residual | low |

**IMM evidence, verified in this session:** IMM is an active method in *modern* tracking pipelines — e.g.
**IMM-MOT** ([arXiv 2502.09672](https://arxiv.org/abs/2502.09672)) **[SRC]**, an Interacting-Multiple-Model
framework for 3D multi-object tracking, and IMM applied to aircraft conflict detection
([arXiv 2409.13782](https://arxiv.org/abs/2409.13782)) **[SRC]**. Both confirm IMM is the standard answer to
manoeuvring targets. It is *not* well represented in the single-object VOT literature — which is itself the
finding: **the VOT field handles fast motion by enlarging the search window, not by improving the motion
model**, and that is a gap you can exploit with ~50 lines of code.

### 10.3 The motion-compensation move that matters most for a drone

Before any appearance matching, **warp the search region by the camera's own motion**: from the gimbal
attitude + IMU (or, if unavailable, a homography estimated from static background features — ultralytics'
BoT-SORT already ships `gmc_method: sparseOptFlow` **[SRC from BRIEF]**). Then the residual motion you must
model is *only the subject's self-motion*, and a CV Kalman filter becomes far more accurate. This is TCTrack's
learned insight ([arXiv 2203.01885](https://arxiv.org/abs/2203.01885)) executed analytically.

---

## 11. Occlusion-handling taxonomy

### 11.1 The three occlusion regimes (and how to detect each)

| Regime | Definition | Detection signal | Correct response |
|---|---|---|---|
| **No occlusion** | target's visible area ≈ its full extent | detector box ≈ predicted box; ReID score high; mask area stable | normal update |
| **Partial occlusion** | target partially visible; box may be *smaller* or *clipped* | detector box **shrank** vs. prediction, ReID score high on the visible crop; predicted box overlaps a masking object | **update motion, DO NOT update appearance with the truncated crop** — this is the #1 cause of template poisoning |
| **Full occlusion** | target not visible at all; may still be *in frame* behind an object | no detection overlapping the predicted box; ReID scores of all candidates below `τ_low`; box stays inside frame | coast with motion model; freeze gallery; enlarge search gradually |
| **Out-of-frame (out-of-view)** | target outside the image | predicted box centre exits the frame / touches an edge for k frames | coast + freeze; do **not** delete the track; prepare global re-detection on re-entry |

The distinction that papers formalise and practitioners often miss: **full occlusion** and **out-of-view**
have *different* correct responses. In full occlusion the target is nearby and moving plausibly — trust the
motion model. Out-of-view means your camera geometry is wrong — trusting the motion model too long will place
the re-detection region in a completely wrong part of the frame, so the search must widen to *global* sooner.

### 11.2 Visibility estimation — how the papers actually do it

1. **Threshold on the matching score** (GlobalTrack and most Siamese trackers). Simplest; no calibrated
   notion of *visibility*.
2. **A dedicated verifier / IoU predictor** (SPLT [arXiv 1909.01840](https://arxiv.org/abs/1909.01840);
   ATOM's IoU-Net [arXiv 1904.07220](https://arxiv.org/abs/1904.07220)) **[SRC]**: predict the IoU of a
   candidate with the (unknown) ground truth; low predicted IoU = occluded/absent.
3. **Probabilistic mass** (PrDiMP **[MEM]**): the predicted distribution's mass within the plausible region is
   the visibility proxy.
4. **Mask area / mask stability** (SAM-2 family, SiamMask [arXiv 1812.05050](https://arxiv.org/abs/1812.05050)
   **[SRC]**): compare the *segmented* area to the historical area distribution; a sudden drop to e.g. <50 %
   is partial occlusion, `≈0` is full occlusion. **This is the cleanest available signal for this project**,
   because it distinguishes "person is 60 % visible" from "person is gone" without any training.
5. **Masked memory attention** (SAM 2 / SAMURAI / SAM2Long): only memory frames that were positively matched
   contribute. Occlusion frames therefore cannot poison memory.

### 11.3 Re-detection triggers (the operational part)

A re-detection (global search) should be triggered by **conjunction of evidence**, never by a single frame:

- `S(c*) < τ_low` for **k consecutive frames** (k ≈ 5–15; k too small ⇒ thrashing on a detector miss);
- **and** no candidate with `s_app > τ_app_min` anywhere in the frame;
- **or** the predicted box's *visible support* collapses (mask area ratio < `ρ_vis`, e.g. 0.25) while a
  large occluder overlaps the box;
- **or** an edge-exit event (box centre crosses the image border and stays out for k frames);
- **or** an *externally injected* trigger: the MOT layer reports a new person entering where the target was
  predicted (a likely re-appearance).

**Anti-thrash rule (essential):** after a successful re-acquisition, require `m` consecutive frames
(m ≈ 3–5) of high-confidence match before returning to normal gallery updating. Otherwise a single lucky
frame re-poisons the gallery.

---

## 12. Recommended architecture for THIS project

### 12.1 The architecture in one diagram

```
                        ┌──────────────────────────────────────────┐
  frame ──► YOLO26n ───►│  persons: boxes + conf + (optional masks)│
            (58.9 FPS)  └───────────────┬──────────────────────────┘
                                        │
                     ┌──────────────────┴───────────────────┐
                     ▼                                      ▼
      ┌───────────────────────────┐        ┌──────────────────────────────────┐
      │  MOT layer ("awareness")  │        │  TARGET layer ("the clicked one")│
      │  BoT-SORT / DeepOCSORT /  │        │  ── the novel part ──            │
      │  tracktrack.yaml          │        │  1. candidate scoring            │
      │  + yolo26n-reid.onnx      │        │     s_app (gallery cosine)       │
      │  → all persons, IDs       │        │     s_mot (IMM / Singer predict) │
      └───────────┬───────────────┘        │     − s_dis (distractor penalty) │
                  │                        │  2. margin test + accept gate    │
                  │  distractor set D      │  3. state machine (§13)          │
                  └───────────────────────►│  4. gallery mgmt (quality+div.)  │
                                           │  5. part-aware templates         │
                                           └───────────────┬──────────────────┘
                                                           │
                                     ┌─────────────────────┴──────────────────┐
                                     ▼                                        ▼
                        ┌──────────────────────┐              ┌──────────────────────────┐
                        │ SAM 2.1 tiny         │              │ gimbal/UI output         │
                        │ (on-demand keyframe) │              │ yaw/pitch rate, apparent │
                        │ part mask / re-anchor│              │ size target, LOST banner │
                        └──────────────────────┘              └──────────────────────────┘
```

### 12.2 Why this shape and not an end-to-end SOT net

| Requirement | Re-detection-first (this proposal) | End-to-end SOT net (OSTrack/MixFormer/ODTrack) |
|---|---|---|
| Report "absent" | **native** (no candidate passes) | needs a hand-tuned threshold; not trained for it |
| Never hijack a look-alike | **explicit distractor term + margin** | no identity concept at all |
| Re-acquire after 300 frames | **native** (gallery search over detections) | needs a global-search mode bolted on |
| Partial visibility | **part-aware gallery + SAM mask** | single template; degrades |
| Compute on 8 GB | shares detector + embedder | **extra** full network every frame |
| No fine-tuning | nothing is trained | off-the-shelf weights only — *works, but wrong objective* |

**The honest counter-argument:** an end-to-end tracker is *very* good at the thing the re-detection approach
is worst at — **smooth, sub-pixel, temporally stable boxes when the target is present.** Detection-based
localisation jitters at the detector's frame-to-frame noise level. Therefore the recommended design keeps a
small end-to-end tracker **as an optional refinement**, not as the decider:

> Use the target layer's assignment to decide **which** person; optionally apply SAM 2.1 tiny (or a light
> SOT net) on that person's crop to **refine the mask/box** for presentation and for part-aware gallery
> updates. The decider is never the SOT net.

### 12.3 Concrete pipeline per frame (budgeted)

Measured on this machine **[SRC from BRIEF]**: YOLO26n detector **58.9 FPS** at imgsz 640, `half=True`
(16.99 ms mean, p95 19.68 ms); ReID **bs1 2.74 ms, bs40 12.7 ms** (0.317 ms/crop) on CUDA.

| Step | Cost | Notes |
|---|---|---|
| YOLO26n detect (person class) | ~17 ms | GPU, already measured |
| GMC / camera-motion warp | ~1–3 ms | reuse `sparseOptFlow`, or IMU/gimbal if available |
| ReID on all person crops | ~0.3 ms × N | N=10 ⇒ ~3 ms |
| Candidate scoring + IMM + gallery | < 1 ms | pure CPU/numpy |
| SAM 2.1 tiny (only on keyframe / LOST search) | ~30–80 ms, only sometimes | amortised |
| **Total steady state** | **≈ 22–25 ms ⇒ 40–45 FPS** | comfortably real-time |

VRAM: YOLO26n (`half`) + `yolo26n-reid.onnx` is already verified to fit; SAM 2.1 tiny adds ~156 MB of
weights plus activations — load it lazily and only when the state machine requests it, or keep it resident
if the measurements permit.

---

## 13. Off-the-shelf weights worth downloading (verified in this session)

All sizes below were read from GitHub's release API or from `hf-mirror.com` HEAD/size metadata in this
session, unless marked **[MEM]**.

### 13.1 Person identity / ReID — **the most important download**

| Asset | Size | Source URL | Licence |
|---|---|---|---|
| **`yolo26n-reid.onnx`** | **9,873,245 B** | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-reid.onnx` | **AGPL-3.0** (ultralytics releases) — **[SRC]** API `license.spdx_id` |
| `yolo26s-reid.onnx` | 28,465,202 B | `.../v8.4.0/yolo26s-reid.onnx` | AGPL-3.0 |
| `yolo26m-reid.onnx` | 48,114,106 B | `.../v8.4.0/yolo26m-reid.onnx` | AGPL-3.0 |
| `yolo26l-reid.onnx` | 59,972,095 B | `.../v8.4.0/yolo26l-reid.onnx` | AGPL-3.0 |
| `yolo26x-reid.onnx` | 129,255,445 B | `.../v8.4.0/yolo26x-reid.onnx` | AGPL-3.0 |
| `.pt` equivalents | n / s / m / l / x = 6.25 / 15.57 / 25.43 / 31.45 / 66.14 MB | `.../v8.4.0/yolo26{k}-reid.pt` | AGPL-3.0 |
| `-market` variants (academic-only domain-tuned) | n 5.87 / s 15.19 / m 25.05 / l 31.05 / x 65.75 MB | `.../v8.4.0/yolo26{k}-reid-market.pt` | AGPL-3.0 |
| `reid8.zip` (legacy) | 235,303 B | `.../v0.0.0/reid8.zip` | AGPL-3.0 |

**Already on disk in this project [SRC from BRIEF]:** `C:\Projects\track\data\weights\yolo26n-reid.onnx`
(9,873,245 B, **512-d output @ 224×224**). ultralytics 8.4.185 defines
`REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")` and auto-downloads via
`attempt_download_asset()`; `ReID(model, imgsz=224, device=None, fp16=False)`.

> **Licence warning that matters for a productised drone:** AGPL-3.0 is a *network-copyleft* licence. If this
> ever ships as a service, AGPL obligations attach. Alternatives with permissive licences:
> - **OpenCV Zoo `person_reid_youtu_2021nov.onnx`** — [opencv/opencv_zoo/models/person_reid_youtureid](https://github.com/opencv/opencv_zoo/tree/main/models/person_reid_youtureid),
>   Apache-2.0 (LICENSE file present in the directory, 11,358 B) **[SRC]**; there are also `int8` and
>   `int8bq` variants **[SRC]**. ⚠ The API-reported file size was 134 B because the files are **Git LFS
>   pointers** — the real ONNX is fetched through LFS on clone. Use `ghproxy.net` for the LFS blob.
> - **Torchreid / OSNet-family** weights **[MEM]** — MIT-licensed research code; not pre-exported to ONNX.

### 13.2 Segmentation / part-level re-anchoring

| Asset | Size | Source URL | Licence |
|---|---|---|---|
| `facebook/sam2.1-hiera-tiny` (`sam2.1_hiera_tiny.pt`) | **156,008,466 B** | `https://hf-mirror.com/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt` | **Apache-2.0** **[SRC]** |
| `facebook/sam2.1-hiera-small` (`sam2.1_hiera_small.pt`) | **184,416,285 B** | `https://hf-mirror.com/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt` | **Apache-2.0** **[SRC]** |
| SAM 2.1 tokenizer weights | **[MEM]** ~150 MB class | `https://dl.fbaipublicfiles.com/segment_anything_2/...` (used by SAMURAI/SAM2Long `download_ckpts.sh`) **[SRC: the README instructs `cd checkpoints && ./download_ckpts.sh`]** | Apache-2.0 |
| `Mark12Ding/SAM2Long` — **code only, training-free** | repo | [github.com/Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long) | **CC-BY-NC 4.0** for the project code; underlying SAM 2 checkpoints Apache-2.0 **[SRC]** |
| `yangchris11/samurai` — **training-free, uses SAM 2.1 weights + KF** | repo | [github.com/yangchris11/samurai](https://github.com/yangchris11/samurai) | see repo **[SRC]**: zero-shot, no extra training |

⚠ `SAM2Long` is **non-commercial (CC-BY-NC 4.0)**. SAMURAI/SAM2Long are *inference-time algorithms* on top
of Apache-2.0 SAM 2 — you can reimplement the memory-tree / motion-aware-memory **idea** in your own code
against the Apache-2.0 SAM 2.1 weights and avoid the NC licence. That is the recommended path.

### 13.3 Detector / tracker building blocks already available

| Asset | Size | URL | Licence |
|---|---|---|---|
| `yolo11n.onnx` | 10,930,182 B | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx` | AGPL-3.0 **[SRC]** |
| `yolo11n-{seg,pose,obb,cls}.onnx` | 11.76 / 11.85 / 11.00 / 11.29 MB | same release | AGPL-3.0 **[SRC]** |
| YOLO26 family on HF | `ultralytics/yolo26` repo, all of `yolo26{n,s,m,l,x}{,-seg,-pose,-obb,-cls}.pt` | `https://hf-mirror.com/ultralytics/yolo26` | AGPL-3.0 **[SRC]** |
| `yolo26n.pt` | 5,544,453 B / 2,572,280 params | already at `C:\Projects\track\yolo26n.pt` **[SRC from BRIEF]** | AGPL-3.0 |
| **OpenCV Zoo `object_tracking_vittrack`** (modern `cv2.TrackerVit`) | LFS pointer in repo listing; real size **[MEM]** ~4 MB class | [opencv/opencv_zoo/models/object_tracking_vittrack](https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack) (`object_tracking_vittrack_2023sep.onnx`, `..._int8bq.onnx`) | **Apache-2.0** (LICENSE present, 11,358 B) **[SRC]** |

**Download workaround for this machine [SRC from BRIEF]:** `github.com/.../releases/download/...` is
unreliable → prefix with `https://ghproxy.net/`. `huggingface.co` is blocked → set
`HF_ENDPOINT=https://hf-mirror.com`.

### 13.4 Not worth downloading for this project (and why)

- **OSTrack / MixFormer / ODTrack / SUTrack checkpoints** — they are the best *generic-subject* trackers, but
  their objective is "same object", not "same person", and they have no absent state. Keeping them out saves
  ~200–500 MB and, more importantly, keeps the architecture honest. Revisit only if empirical tests show the
  re-detection layer's box instability is the dominant error term.
- **OpenCV legacy `cv2.legacy.TrackerMOSSE/KCF/CSRT`** — no download needed; usable as a CPU fallback or a
  drift early-warning, but never as the target owner.

---

## 14. Proposed state machine for the target tracker

### 14.1 States

| State | Meaning | What the layer does each frame |
|---|---|---|
| **`INIT`** | A click just arrived | Seed the gallery from the clicked box (plus its part crops); initialise the motion filter with zero velocity and a *large* initial covariance; go straight to `TRACKING` on the next frame. |
| **`TRACKING`** | Target confidently located | Score all candidates; accept `c*` with the margin test; update the motion filter; **conditionally** update the gallery. Emit a box/mask. Also emit gimbal/UI output. |
| **`OCCLUDED_SOFT`** | Partial occlusion: the target is *partially* visible and being matched | Continue to accept a candidate (possibly a *clipped* box), but **freeze the appearance gallery** and switch to **part-aware matching** (upper-body / head / torso templates). Dampen motion updates (do not let a truncated box corrupt the scale estimate). Do **not** add the truncated crop to the gallery. |
| **`LOST_SHORT`** | No acceptable candidate for `k1` frames (k1 ≈ 5–15) | **Coast** with the motion model (widening covariance). **Freeze the gallery.** Search region = predicted box expanded by a factor that grows with time-in-state. Keep publishing a *predicted* box with low confidence (or "coasting" UI). |
| **`LOST_LONG`** | Still no acceptable candidate after `k2` frames (k2 ≈ 30–60) | Full-frame **global re-detection**: re-score *all* detected persons against the frozen gallery (and the part-aware bank). If the search is global and cheap (you already detect all persons), do this **every frame** — the difference from `LOST_SHORT` is the *decision policy*, not the detector. Optionally run SAM 2.1 tiny on the top-N candidates to build masks and match at part level. |
| **`RE_ACQUIRE`** | A candidate has passed `τ_acc` + margin but not yet the dwell test | Do **not** update the gallery. Track the candidate (motion + appearance) and require `m` consecutive frames (m ≈ 3–5) of persistent high-confidence match, with a plausibility check on the motion (the trajectory from the loss point to the re-acquisition must be *physically reachable*: `|Δp| ≤ v_max · Δt`, plus IMM-consistency). If it fails → back to `LOST_SHORT`/`LOST_LONG`. If the candidate's identity is ambiguous between two persons → stay in `RE_ACQUIRE`, do not commit. |
| **`CONFIRMED`** | Re-acquisition dwell satisfied | Promote to `TRACKING`, un-freeze the gallery, **hard-negative-mine** the distractor(s) that caused the ambiguity, and record the re-acquisition event (latency, candidate identity) for the target-centric metrics. |
| **`GUARDED_FAIL`** (optional terminal) | Global search exhausted for `k3` frames (k3 ≈ 300–900) | Stop publishing a box; keep the gallery in a read-only state so a man-in-the-loop re-click, or a future "did you see this person again?" query, still works. Emit a user-legible LOST event (this mirrors DJI's documented FAQ path). |

### 14.2 Transition table with explicit triggers

| From → To | Trigger condition (all thresholds are *starting* values) |
|---|---|
| `INIT → TRACKING` | 1 frame after click; gallery seeded from click box + part crops |
| `TRACKING → OCCLUDED_SOFT` | `ρ_vis < 0.6` **AND** `S(c*) > τ_acc` (box shrank / mask area dropped but identity is still confident) — OR box clipped by image border |
| `TRACKING → LOST_SHORT` | `S(c*) < τ_low` for `k1 = 8` consecutive frames, **or** `s_app(c*) < τ_app_min` for 8 frames |
| `OCCLUDED_SOFT → TRACKING` | `ρ_vis ≥ 0.6` for 5 consecutive frames **AND** `S > τ_acc` |
| `OCCLUDED_SOFT → LOST_SHORT` | no acceptable candidate for 8 frames |
| `LOST_SHORT → TRACKING` | candidate passes `τ_acc` + `τ_margin` **and** `s_mot > τ_mot` (short gap: motion still informative ⇒ can trust a direct match) |
| `LOST_SHORT → LOST_LONG` | `k2 = 40` consecutive frames without acceptance |
| `LOST_LONG → RE_ACQUIRE` | any candidate with `S > τ_acc` (global search, motion gate relaxed to a reachability test) |
| `RE_ACQUIRE → CONFIRMED` | `m = 4` consecutive frames with `S > τ_acc` **AND** `s_mot > τ_mot` **AND** reachability check passes |
| `RE_ACQUIRE → LOST_LONG` | any of the dwell conditions fails |
| `CONFIRMED → TRACKING` | immediately (after gallery un-freeze + hard-negative mining) |
| `LOST_LONG → GUARDED_FAIL` | `k3 = 600` frames without re-acquisition |
| **any → INIT** | new user click (resets everything) |

### 14.3 Why these particular thresholds and triggers

- **Two-level loss (`LOST_SHORT` vs `LOST_LONG`) rather than one LOST**: the correct *policy* differs. In a
  short gap the constant-velocity/IMM prediction is trustworthy, so a tight motion gate is a strong
  anti-hijack filter. After ~1.5 s at 30 FPS (40 frames) a person can be 15 m away; the motion gate must be
  replaced by a *reachability* gate plus global appearance search. Collapsing these two into one LOST state
  is the most common design error and produces either premature global searches (jitter) or hijacks.
- **`OCCLUDED_SOFT` is a separate state** (rather than a continuous "confidence" scalar) precisely because
  its *action* is different: freeze the gallery, go part-aware. A scalar confidence invites averaging a
  truncated crop into the appearance model — the classic template-poisoning path.
- **`RE_ACQUIRE` exists to separate "found" from "committed"**. Every re-acquisition is a potential hijack;
  requiring dwell + reachability converts a per-frame decision into a temporal one, which is exactly what
  `KeepTrack`/`LTMU`-class methods do with learned modules and what you can do with a hand-written gate.
- **The margin test lives in the accept gate, not in the state machine**, so it applies uniformly in
  `TRACKING`, `LOST_*` and `RE_ACQUIRE`.

### 14.4 What each state emits

| State | Box output | Gallery | Search scope | UI / gimbal |
|---|---|---|---|---|
| `TRACKING` | confident box (+ optional mask) | update allowed (gated) | local | track; gimbal keeps subject centered / at target apparent size |
| `OCCLUDED_SOFT` | box (may be clipped) | **frozen** | local, part-aware | "subject partially obscured" hint |
| `LOST_SHORT` | predicted box, low confidence / suppressed | **frozen** | predicted box, expanding | "searching" |
| `LOST_LONG` | none | **frozen** | global (all persons) | "searching (wide)" |
| `RE_ACQUIRE` | tentative box, not committed | **frozen** | global | "candidate found…" |
| `CONFIRMED` | confident box | un-freeze + hard-negative mine | local | "re-acquired" + latency metric |
| `GUARDED_FAIL` | none | read-only | none | "target lost" (user action required) |

---

## 15. Evaluation protocol for the target layer (target-centric, not MOT)

MOT metrics (MOTA/IDF1/HOTA/AssA) measure the *all-person* layer and are already specified in the brief. For
the **target layer**, adopt the VOT-LT convention plus re-acquisition-specific terms:

| Metric | Definition | Why |
|---|---|---|
| **Target success rate** | fraction of frames with IoU(box, GT) > 0.5 *while the target is present* | the standard VOT success |
| **Precision** | fraction of reported boxes within a centre-error threshold | VOT-LT |
| **Recall** | fraction of target-present frames where a box was correctly reported | VOT-LT; *recall penalises over-conservatism* |
| **F1** | harmonic mean of the two | VOT-LT's headline long-term number |
| **Lost rate** | fraction of frames in `LOST_*` while the target *was* present | your detector-of-loss quality |
| **False-lock rate / hijack rate** | fraction of frames with a confidently reported box on the **wrong** person | **the metric that matters most for this project — and the one no benchmark reports** |
| **Re-acquisition rate** | fraction of loss events followed by a correct re-acquisition within `N` frames | long-term capability |
| **Re-acquisition latency** | frames from loss to `CONFIRMED` | user-perceived quality |
| **ID-switch-free rate** | fraction of sequences with zero hijack events | the product-level number |
| **Precision / Recall of the absent decision** | treat "absent" as a class | the cleanest way to expose thrashing |

**Recommended synthetic stress tests (cheap, controllable, and absent from the literature):**
1. **Twin test** — two persons in near-identical clothing cross paths; measure hijack rate.
2. **Reverse test** — target reverses direction abruptly at maximum speed; measure recovery frames.
3. **Halt test** — target stops and stands still for 200 frames among moving distractors; the "stationary"
   IMM mode is the only thing that saves you here.
4. **Exit/re-entry test** — target leaves frame for 30 / 300 frames and returns with a different heading.
5. **Partial-view test** — only upper body (or only lower body, or only the head) is visible for a stretch.
6. **Gimbal-sweep test** — rotate the camera at known rate while the target stands still; isolates
   camera-induced motion from subject motion.

---

## 16. Open questions and honest gaps in this report

1. ⚠ **"DropTrack" (CVPR 2025)** — no arXiv record found; the only arXiv "DropTrack" is a microfluidics
   paper. Verify against the CVPR 2025 proceedings before citing.
2. ⚠ **"KYS" (CVPR 2022)** — title-based arXiv query returned nothing in this session. Cite the proceedings
   page.
3. ⚠ **Numerical leaderboard values for UAV123 / UAV20L / DTB70** — not retrieved. Every AUC/success number
   for an individual tracker on these benchmarks must be traced to a paper table.
4. ⚠ **DJI patent US9164506B1** — number and title seen in search results; claim text not read. Verify on
   Google Patents before quoting claims.
5. ⚠ **SAMURAI's reported LaSOT / UAV123 numbers** — the README documents *how to evaluate* (and states the
   method is zero-shot with SAM 2.1 + a Kalman filter) but the numeric table was not extracted in this
   session. The SAMURAI paper ([arXiv 2411.11922](https://arxiv.org/abs/2411.11922)) has them.
6. **Not investigated (out of scope but adjacent):** RGB-T / thermal person tracking as a robustness axis
   (SUTrack's multi-modal variant, anti-UAV RGB-T), event-camera tracking
   ([arXiv 2310.14194](https://arxiv.org/abs/2310.14194)), and adversarial robustness of the target layer
   ([arXiv 2411.17468](https://arxiv.org/abs/2411.17468), [arXiv 1910.08681](https://arxiv.org/abs/1910.08681)).
7. **Google Patents was unreachable** from this machine; patent claims were obtained only via a third-party
   aggregator, and only for two DJI patents. A proper patent review (DJI + Skydio + Autel + Insta360)
   remains an open task.

---

## Appendix A — Verified arXiv identifiers collected in this session

Collected via the arXiv API in this session (**[SRC]**). Use these IDs rather than re-deriving them.

**Correlation filters / classical**
- `1611.09224` ECO (2016-11) · `1803.08679` STRCF (2018-03) · `1403.7876` CF with Limited Boundaries (2014-03) · `1808.08834` Real-Time MDNet (2018-08)

**Siamese era**
- `1606.09549` SiamFC (2016-06) · `1812.11703` SiamRPN++ (2018-12) · `1808.06048` DaSiamRPN (2018-08) ·
  `1901.01660` SiamDW (2019-01) · `1812.05050` SiamMask (2018-12) · `2207.02088` SiamMask TPAMI (2022-07) ·
  `2003.06761` SiamBAN (2020-03) · `1911.07241` SiamCAR (2019-11) · `2006.10721` Ocean (2020-06, ECCV2020) ·
  `2005.04063` TSDM/SiamRPN++ (2020-05) · `2106.07862` Domain-Adaptive SiamRPN++ (2021-06)

**Discriminative / transformer era**
- `1904.07220` ATOM (2019-04) · `2103.11681` TrDiMP (2021-03) · `2103.16556` KeepTrack (2021-03) ·
  `2203.05328` SimTrack (2022-03) · `2203.11991` OSTrack (2022-03) · `2203.11082` MixFormer (2022-03) ·
  `2302.02814` MixFormer extended (2023-02) · `2305.15896` MixFormerV2 (2023-05) ·
  `2304.14394` SeqTrack (2023-04) · `2312.17133` ARTrackV2 (2023-12) · `2401.01686` ODTrack (2024-01) ·
  `2507.20177` ODTrack universal-modal (2025-07) · `2401.03142` EVPTrack (2024-01) ·
  `2403.05231` LoRAT (2024-03) · `2410.16329` LoRAT Perception-Test solution (2024-10) ·
  `2311.02072` HIPTrack (2023-11) · `2412.19138` SUTrack (2024-12) · `2506.20381` lightweight hierarchical ViT (2025-06) ·
  `2604.02654` Drift-Resilient Temporal Priors (2026-04)

**Distractor-aware**
- `1808.06048` DaSiamRPN · `2103.16556` KeepTrack · `2509.13864` Distractor-Aware Memory-Based VOT (2025-09) ·
  `2411.17576` Distractor-Aware Memory with SAM2 (2024-11) · `2310.14194` Distractor-aware Event-based (2023-10) ·
  `2401.03872` transparent-object distractor-aware (2024-01)

**Long-term / absent target**
- `1912.08531` GlobalTrack (2019-12) · `1909.01840` SPLT (2019-09) · `2004.00305` LTMU (2020-04) ·
  `2305.12998` MFT (2023-05) · `1803.09502` LaSOT (2018-03) · `1908.01603` Model Decay in Long-Term Tracking (2019-08) ·
  `2511.16618` SurgSLOT (2025-11)

**SAM-based**
- `2408.00714` SAM 2 (tag on the HF model card) · `2411.11922` SAMURAI (2024-11) · `2410.16268` SAM2Long (2024-10) ·
  `2503.01907` ReID-SAM / SkiTB (2025-02)

**UAV / aerial / benchmarks**
- `2203.01885` TCTrack (2022-03) · `2003.12949` AutoTrack (2020-03) · `2410.20421` NT-VOT211 (2024-10) ·
  `2108.09909` 2nd Anti-UAV (2021-08) · `2305.07290` 3rd Anti-UAV (2023-05) · `2512.02668` UAUTrack (2025-12) ·
  `2607.26511` UAV Anti-UAV temporal adaptation (2026-07) · `2609.01742` Evidential DL multi-modal anti-UAV (2026-09) ·
  `2103.12924` Benchmarking Deep Trackers on Aerial Videos (2021-03) · `2308.10262` / `2308.11450` /
  `2104.03114` / `2207.01768` DTB70-evaluating UAV trackers (2021–2023) · `2107.08766` VisDrone-CC2020 (2021-07) ·
  `2202.11983` GIAOTracker / VisDrone 2021 (2022-02) · `2403.04363` multi-step temporal modelling UAV (2024-03) ·
  `2409.16834` nighttime UAV denoiser (2024-09) · `2409.16111` CloudTrack UAV (2024-09)

**Motion models**
- `2502.09672` IMM-MOT (2025-02) · `2409.13782` IMM aircraft conflict (2024-09)

**Not verified (⚠)**
- `2508.14370` **[SRC]** — verified to be **"FastTracker: Real-Time and Accurate Visual Tracking"** (2025-08),
  the paper behind this project's `fasttrack.yaml` config. The arXiv title and the ultralytics config name
  differ.
- `2205.02568` — the only arXiv "DropTrack" is about microfluidic droplets, **not** the CVPR 2025 tracker.
- No arXiv entry found for: **MOSSE**, **KCF**, **DSST**, **SiamRPN**, **DiMP**, **PrDiMP**, **KYS**,
  **UAV123**, **DTB70** (primary), **DropTrack (CVPR 2025)**, **VOT-LT** (challenge protocol).
- Venue corrections made in this report after direct verification: **TCTrack = CVPR 2022** (not AAAI);
  **MFT = WACV 2024** (not ICCV 2023); **EVPTrack = Information Fusion 2024** (arXiv 2401.03142, title
  *"Explicit Visual Prompts for Visual Object Tracking"*); **MixFormer** primary is **CVPR 2022 (Oral)**
  arXiv 2203.11082 with the extended version at arXiv 2302.02814; **SeqTrack**'s arXiv 2304.14394 is the
  expanded version of the CVPR 2023 paper.

---

## Appendix B — Copy-paste weight download commands for this machine

```powershell
$env:PYTHONIOENCODING = "utf-8"
# github.com release downloads are unreliable here -> use ghproxy.net
$gh = "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0"
# (already present) yolo26n-reid.onnx 9,873,245 B
# Optional bigger identity models:
curl.exe -L "$gh/yolo26s-reid.onnx" -o C:\Projects\track\data\weights\yolo26s-reid.onnx
curl.exe -L "$gh/yolo26m-reid.onnx" -o C:\Projects\track\data\weights\yolo26m-reid.onnx

# huggingface.co is blocked; hf-mirror.com works
$env:HF_ENDPOINT = "https://hf-mirror.com"
curl.exe -L "https://hf-mirror.com/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt" `
  -o C:\Projects\track\data\weights\sam2.1_hiera_tiny.pt          # 156,008,466 B, Apache-2.0
curl.exe -L "https://hf-mirror.com/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt" `
  -o C:\Projects\track\data\weights\sam2.1_hiera_small.pt         # 184,416,285 B, Apache-2.0

# OpenCV Zoo (Apache-2.0 alternatives to the AGPL ReID models); files are Git-LFS
curl.exe -L "https://ghproxy.net/https://github.com/opencv/opencv_zoo/raw/main/models/person_reid_youtureid/person_reid_youtu_2021nov.onnx" `
  -o C:\Projects\track\data\weights\person_reid_youtu_2021nov.onnx
curl.exe -L "https://ghproxy.net/https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx" `
  -o C:\Projects\track\data\weights\object_tracking_vittrack_2023sep.onnx
```

**Licence summary for the three recommended assets:** `yolo26{k}-reid.onnx` = **AGPL-3.0**;
`sam2.1_hiera_{tiny,small}.pt` = **Apache-2.0**; OpenCV Zoo models = **Apache-2.0**.
SAM2Long's *code* is **CC-BY-NC 4.0** — reimplement the idea rather than vendoring it.

