# R1 — Multi-Object Tracking (people): association strategies, 2013–2026

**Author:** research-analyst subagent · **Date:** 2026-10-24
**Scope:** MOT association for people, weighted to 2021–2026. Detection-based (TBD) lineage, motion models, appearance/ReID integration, end-to-end & query-based trackers, association machinery, benchmark SOTA, transferable ablations — plus a decision table for *this* project.

**Evidence policy.** Every number is attributed to a source URL. Facts about ultralytics 8.4.185 are read directly from the installed wheel (`C:\Projects\track\_p2\ul_wheel\`) and are labelled **[SRC]** (verified by reading source). Items I could not verify in this session are labelled **⚠ UNVERIFIED**.

**Companion artefact:** the ultralytics implementation audit is §10 (the highest-value section for this project).

---

## 0. TL;DR

1. **For nonlinear/fast motion, "observation-centric" corrections are the single biggest lever** in the heuristic (TBD) family: on DanceTrack test, ByteTrack 47.3 HOTA → OC-SORT 54.6 HOTA with *the same detections* ([Hybrid-SORT, AAAI 2024](https://arxiv.org/html/2308.00783v2)).
2. **Weak cues (confidence state, height state, velocity direction, HMIoU) buy almost as much as ReID**, and cost nothing: Hybrid-SORT = OC-SORT + 7.6 HOTA on DanceTrack with *identical association inputs* and nearly identical compute (same source).
3. **Appearance/ReID does not pay on uniform-appearance data and can be left off**; when used, it should be *gated* and *min-fused* (BoT-SORT / Deep OC-SORT) so it can never override a motion-valid match — the exact design shipped in ultralytics **[SRC]**.
4. **The bottleneck on DanceTrack/SportsMOT is association, not detection** — DetA on DanceTrack is *higher* than on MOT17 while HOTA/AssA/IDF1 are much lower ([DanceTrack, CVPR 2022](https://dancetrack.github.io/); [SportsMOT, ICCV 2023](https://arxiv.org/html/2304.05170v2)).
5. **Kalman rollback + occlusion-aware search expansion (FastTracker)** and **relaxed lost-track rebinding (TrackTrack `lost_match_thr`)**, **enough-buffer IoU cascades (C-BIoU)**, and **unbounded track buffers (Deep HM-SORT)** are the four occlusion/re-entry mechanisms that transfer directly into a *single-target* tracker.

---

## 1. Detection-based tracking (TBD) lineage

All of these are "detector + hand-written association". The lineage is a sequence of *corrections to the Kalman-filter + IoU + Hungarian recipe*.

| Year | Method | Venue | Core association idea | Link |
|---|---|---|---|---|
| 2016 | **SORT** | ICIP 2016 | Linear KF (XYAH, constant velocity) + IoU + Hungarian. No appearance. | [arXiv:1602.00763](https://arxiv.org/abs/1602.00763) |
| 2017 | **DeepSORT** | ICIP 2017 | Adds a deep cosine-appearance cost; cascade matching by track age; Mahalanobis gating. | [arXiv:1703.07402](https://arxiv.org/abs/1703.07402) |
| 2022 | **ByteTrack** | ECCV 2022 | Two-stage association: high-score dets vs tracks, then *low-score* dets vs the still-unmatched tracks (IoU-only, no score fusion). | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) |
| 2023 | **OC-SORT** | CVPR 2023 | **OCM** velocity-direction cost + **OCR** (associate against the *last observation*, not the KF prediction) + **ORU** (roll the KF back to the last observation and replay virtual observations across the gap). | [arXiv:2203.14360](https://arxiv.org/abs/2203.14360) |
| 2022 | **BoT-SORT** | arXiv 2022 | ByteTrack + **camera-motion compensation (GMC)** applied to KF states + optional ReID fused by `min()`. | [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) |
| 2023 | **StrongSORT / StrongSORT++** | IEEE TMM 2023 | Detection refresh (YOLOX-X + tuned NMS/reID), **NSA Kalman** (scale measurement noise by detection confidence), EMA feature bank, AFLink/GSI post-processing. | [arXiv:2202.13514](https://arxiv.org/abs/2202.13514) · [arXiv:2210.15975](https://arxiv.org/abs/2210.15975) |
| 2023 | **Deep OC-SORT** | ICIP 2023 | OC-SORT + **adaptive appearance**: embedding weight and EMA rate driven by detection score; appearance as a *gated min-fusion*. | [arXiv:2302.11813](https://arxiv.org/abs/2302.11813) |
| 2024 | **Hybrid-SORT** | AAAI 2024 | **Weak cues**: tracklet confidence state (with linear extrapolation on occlusion onset), height state, and **HMIoU**; **ROCM** = OCM over multiple Δt and all 4 box corners. Cost = `C_HMIoU + λ1·C_Vel + λ2·C_Conf + λ3·C_Appr`. | [arXiv:2308.00783](https://arxiv.org/abs/2308.00783) |
| 2023 | **GHOST** | CVPR 2023 | "Simple cues": tuned IoU + ReID + *camera-motion/feature-normalised* association; strong but heavy post-processing. | [arXiv:2206.04656](https://arxiv.org/abs/2206.04656) |
| 2023 | **C-BIoU** | WACV 2023 | **Cascaded Buffered IoU**: enlarge boxes by a *small* buffer for a first cascade, a *large* buffer for a second, so identical-but-non-overlapping objects in adjacent frames still match and motion-estimation bias is absorbed. 2nd place CVPR'22 SoccerNet + ECCV'22 DanceTrack challenge solution. | [arXiv:2211.14317](https://arxiv.org/abs/2211.14317) |
| 2025 | **FastTracker** | arXiv 2025 | ByteTrack + **occlusion detection by coverage**, **Kalman rollback** to pre-occlusion state, motion dampening, bbox enlargement, **init-IoU suppression**, extended re-find window. MOT17 test HOTA 66.4 / MOT20 65.7. | [arXiv:2508.14370](https://arxiv.org/abs/2508.14370) |
| 2025 | **TrackTrack** | CVPR 2025 | **TPA** (track-perspective association over HMIoU + optional ReID + confidence-projection + corner-angle, solved by *iterative* relaxing-threshold assignment) + **TAI** (track-aware NMS before spawning an ID). MOT17 test HOTA **67.1** / MOT20 **65.7**. | [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) |

Notes that matter:

- **The lineage's real axis is "when the Kalman filter is wrong, stop trusting it."** OC-SORT trusts the last *observation*; Hybrid-SORT trusts a *confidence state*; FastTracker trusts a *history buffer*; TrackTrack trusts *confidence extrapolation + corner angle*.
- Ultralytics ships exactly six of the above (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTracker, TrackTrack) — see §10. `tracktrack.yaml` is the **default** when no `tracker=` argument is passed ([ultralytics docs](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md)).

---

## 2. Motion models

### 2.1 Kalman filter variants actually shipped in the open-source mainstream

| Variant | What changes | Where |
|---|---|---|
| SORT/DeepSORT KF — 8-dim XYAH (x,y,a,h,vx,vy,va,vh), constant velocity, `std_weight_position=1/20`, `std_weight_velocity=1/160` | baseline | [SRC] `ultralytics/trackers/utils/kalman_filter.py` |
| **XYWH KF** — state (x,y,w,h,vx,vy,vw,vh); noise scaled by w,h | avoids aspect-ratio coupling; used by BoT-SORT & TrackTrack | [SRC] same file, `KalmanFilterXYWH` |
| **NSA Kalman** (StrongSORT) — measurement noise scaled by detection confidence: `R ← R · max(1 − c, 0.05)` | low-confidence detections no longer yank the state | [SRC] `KalmanFilterXYAH.project(mean, cov, confidence)`; also [arXiv:2202.13514](https://arxiv.org/abs/2202.13514) |
| **Observation-centric re-update (ORU)** — restore the KF to the state at the last observation, then replay *virtual* linearly-interpolated observations across the gap, then predict to now | removes the KF's accumulated drift through an occlusion | [SRC] `OCSortTrack.apply_oru`; [arXiv:2203.14360](https://arxiv.org/abs/2203.14360) |
| **Kalman rollback** — copy `mean[0:4]` + full covariance from *N frames ago*, copy velocity from *M frames ago*, scale h by 1.1, damp velocity by 0.5 | re-centres the prediction on where the target *was* when it disappeared | [SRC] `FastSTrack` / `FASTTracker._handle_occlusions`; [arXiv:2508.14370](https://arxiv.org/abs/2508.14370) |
| **Zeroing velocity on Lost** | stop coasting with a stale velocity | [SRC] ByteTrack `STrack.predict()` zeroes `vh`; BoT-SORT zeroes `vw` **and** `vh` |
| **Mamba/SSM motion predictor (MTP)** — bi-Mamba over spatial-temporal location dynamics, autoregressive over missing observations | learned nonlinear motion, ~linear complexity | [MambaTrack, ACM MM 2024, arXiv:2408.09178](https://arxiv.org/abs/2408.09178) |

### 2.2 Nonlinear-motion corrections that are *not* new filters

- **OCM (observation-centric momentum)**: cost = `arccos(cos(direction_of_track, direction_to_detection))/π`, multiplied by `inertia`. Penalises matches that require a direction reversal. [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)
- **ROCM (Hybrid-SORT)**: same idea but over *all four box corners* and multiple Δt ∈ {1,2,3} instead of the centre. [arXiv:2308.00783](https://arxiv.org/abs/2308.00783)
- **C-BIoU buffered IoU**: tolerates displacement up to the buffer, which is what fast/variable-speed motion produces. [arXiv:2211.14317](https://arxiv.org/abs/2211.14317)
- **Camera motion compensation (GMC)**: an affine warp estimated per frame and applied to *every track's* mean/covariance before matching. Three estimators: sparse optical flow (Lucas–Kanade + RANSAC), ECC, feature (ORB/SIFT). [arXiv:2206.14651](https://arxiv.org/abs/2206.14651), and [SRC] `ultralytics/trackers/utils/gmc.py`.

### 2.3 IMM and Ornstein–Uhlenbeck — honest coverage statement

- **IMM (Interacting Multiple Model)**: no IMM-based tracker appears in the 2021–2026 top-venue TBD lineage surveyed here. The functional role IMM would play (switch between a constant-velocity and a manoeuvre model) is instead played by *confidence/velocity-consistency costs* (OCM/ROCM), *velocity zeroing on Lost*, and *rollback*. **⚠ UNVERIFIED**: I did not find a citable 2021–2026 top-venue MOT tracker whose association stage is a full IMM. Treat "IMM for MOT" as an open implementation opportunity rather than a recipe to copy.
- **Ornstein–Uhlenbeck**: appears in MOT-adjacent literature as a continuous-time motion prior (e.g. trajectory-scoring/re-ranking work) rather than inside a shipped TBD tracker. **⚠ UNVERIFIED** in this session — I did not read a primary source, and no ultralytics backend uses it.
- **Practical substitute that *does* exist and ships**: noise that adapts to evidence quality — NSA scaling by score, EMA rate by score, rollback on occlusion, dampening while occluded. If you want "noise-adaptive", these four are the evidence-backed mechanisms; OU/IMM are speculative here.

---

## 3. Appearance / ReID inside MOT

### 3.1 How embeddings are integrated (the three designs that ship)

1. **Cosine distance with a gated minimum fusion (BoT-SORT, Deep OC-SORT).**
   `cost = min(IoU_cost_fused, emb_cost)` where `emb_cost = cosine_distance/2` and pairs are invalidated when `IoU_cost > 1 − proximity_thresh` or `emb > 1 − appearance_thresh`. Consequence: **appearance can only ever make a match easier for spatially-plausible pairs; it can never break a motion-valid match.** **[SRC]** `BOTSORT.get_dists`, `DeepOCSORT._fuse_appearance`.
2. **Track-perspective weighted sum (TrackTrack).** `cost = iou_weight·HMIoU_dist + reid_weight·cosine` for pairs that have a feature, with `conf_weight·|proj_score − det_score|` added always and `angle_weight·corner_angle·det_score` only for pairs with `IoU > 0.10`; pairs lacking a feature fall back to the motion cost alone. **[SRC]** `TRACKTRACK._cost_matrix`.
3. **Harmonic mean (Deep HM-SORT).** `HM(d1,d2) = 2/(1/d1 + 1/d2)` over the expansion-IoU distance and the cosine distance. Reported advantage over min/mean fusion: it *cannot* be dominated by one channel, which is exactly what you want when both channels are unreliable. Ablation: HM barely moves HOTA (77.5 → 77.21) but cuts ID switches 2721 → 2146 (−21%); combined with an unbounded track buffer it reaches 80.1 HOTA / 1703 IDSW on SportsMOT. [arXiv:2406.12081](https://arxiv.org/html/2406.12081v1)

### 3.2 EMA of the appearance bank — and the "appearance hurts" debate

- **Fixed EMA**: `s ← α·s + (1−α)·f`, then renormalise. BoT-SORT uses α = 0.9 hard-coded; Deep OC-SORT and TrackTrack default `alpha_fixed_emb = 0.95`. **[SRC]**
- **Score-adaptive EMA** (the fix for "appearance hurts"): make the bank update *slowly when the detection is weak*.
  - Deep OC-SORT: `trust = (score − det_thresh)/(1 − det_thresh)`; `α = α_fixed + (1 − α_fixed)(1 − trust)`. If `score ≤ det_thresh`, `α = 1.0` — i.e. **the old embedding is kept untouched**. **[SRC]** `DeepOCSortTrack.update_features`
  - TrackTrack `TTSTrack`: `β = α + (1−α)(1 − score)`. **[SRC]**
- **Why appearance loses on DanceTrack**: the dataset was explicitly built with *uniform appearance and diverse motion*, so appearance features are near-uninformative while motion is highly informative ([DanceTrack, CVPR 2022](https://dancetrack.github.io/)). The DanceTrack paper's own analysis shows the bottleneck is **association**, and that *better motion models* (KF, LSTM) beat naive IoU by a large margin.
- **Why appearance still wins on MOT17/MOT20**: pedestrians have discriminative clothing; appearance is what gets you through long occlusions. BoT-SORT-ReID 65.0 vs ByteTrack 63.1 HOTA on MOT17 test.
- **SportsMOT is the middle case**: t-SNE of ReID features puts SportsMOT *between* MOT17 (most separable) and DanceTrack (least separable) — "similar yet distinguishable" (jersey numbers, postures). The dataset authors' conclusion is that sports needs **both** a better motion model and a more discriminative appearance model. [arXiv:2304.05170](https://arxiv.org/html/2304.05170v2)
- **Practical rule for this project**: enable ReID when it is nearly free (ultralytics `model: auto` reuses detector backbone features), gate it hard (`proximity_thresh`, `appearance_thresh`), and **use it for the clicked target's long-term memory, not for the all-person layer's frame-to-frame association** if the scene has look-alike people.

---

## 4. End-to-end / query-based trackers

Two paradigms: **tracking-by-propagation** (DETR track queries carried across frames) and **decoupled detection + learned association**.

| Method | Venue | Paradigm / mechanism | Link |
|---|---|---|---|
| **TrackFormer** | CVPR 2022 | Track queries added to a DETR decoder; identity by query persistence. | [arXiv:2101.02702](https://arxiv.org/abs/2101.02702) |
| **TransTrack** | arXiv 2020 | Transformer over the previous frame's object features + learned queries. | [arXiv:2012.15460](https://arxiv.org/abs/2012.15460) |
| **MOTR** | ECCV 2022 | Track query + **tracklet-aware label assignment** + temporal aggregation. DanceTrack test 54.2 HOTA. | [arXiv:2105.03247](https://arxiv.org/abs/2105.03247) |
| **MOTRv2** | CVPR 2023 | MOTR + a separate YOLOX detector, queries initialised from detections. DanceTrack test **69.9 HOTA / 71.7 IDF1 / 91.9 MOTA**. | [arXiv:2303.15780](https://arxiv.org/abs/2303.15780) |
| **MOTRv3** | arXiv 2023 | Release/learn queries, VRT (visual relation transformer) to coordinate detection/tracking heads. ⚠ UNVERIFIED this session. | [arXiv:2305.13383](https://arxiv.org/abs/2305.13383) |
| **MeMOT** | CVPR 2022 | Memory bank of per-object embeddings + query-based detection. | [arXiv:2203.16761](https://arxiv.org/abs/2203.16761) |
| **MeMOTR** | ICCV 2023 | **Long-term memory-augmented** track embeddings via a memory-attention layer; +7.9 HOTA / +13.0 AssA over the then-SOTA on DanceTrack. | [arXiv:2307.15700](https://arxiv.org/abs/2307.15700) |
| **SiamMOT** | CVPR 2021 | Region-based Siamese tracker; explicit vs implicit motion modelling; 17 FPS @720p. | [arXiv:2105.11595](https://arxiv.org/abs/2105.11595) |
| **MambaTrack** | ACM MM 2024 | TBD + **MTP**: a bi-Mamba *motion predictor* replacing/backing up the KF; used autoregressively to fill missing observations. | [arXiv:2408.09178](https://arxiv.org/abs/2408.09178) |
| **SambaMOTR** | **ICLR 2025 Spotlight** | Samba: synchronized set-of-sequences selective state space — one SSM per tracklet, memories synchronized across tracklets; **MaskObs** masks uncertain observations out of the propagation. | [arXiv:2410.01806](https://arxiv.org/abs/2410.01806) |
| **MOTIP** | **CVPR 2025** (pp. 27883–27893) | Reframes association as **in-context ID prediction**: a learnable ID dictionary of K+1 tokens; tracklet token = `concat(DETR object embedding, ID embedding)`; a transformer ID-decoder predicts which ID each current detection belongs to, trained with cross-entropy. Trajectory augmentation = random token dropping (occlusion) + random ID swap (simulated assignment error). | [arXiv:2403.16848](https://arxiv.org/abs/2403.16848) · [code](https://github.com/MCG-NJU/MOTIP) |
| **SUSHI** | ICCV 2023 | Offline: GNN matcher over tracklet proposals. DanceTrack test 63.3 HOTA. ⚠ venue/link not re-verified. | [arXiv:2308.12714](https://arxiv.org/abs/2308.12714) |
| **uMOT** | arXiv 2023 | Unsupervised MOT via contrastive similarity learning — no ID labels at training time. | [arXiv:2309.00942](https://arxiv.org/abs/2309.00942) |

**What transfers from this family into a single-target tracker.** Not the architecture — the *temporal memory* idea:

- MOTIP's **"predict the ID from a context window"** is essentially a learned version of "match the clicked target's gallery against candidates", with the key trick that **the label is in-context (relative), not global** — which is exactly the right formulation for a *single* target tracked over a long video, where the global ID is meaningless but consistency is everything.
- SambaMOTR's **MaskObs** (do not let an uncertain observation update the memory) is the same principle as Deep OC-SORT's `α = 1.0` for low-trust detections, and it is the single most important guard for a target's long-term gallery.
- MOTIP's **trajectory augmentation** (drop tokens, swap IDs) is a ready-made recipe if you ever train a small matcher on your own data.

---

## 5. Association machinery (the boring parts that decide everything)

| Axis | Options | Evidence / where it ships |
|---|---|---|
| **Solver** | Hungarian (`lap.lapjv`, `lap>=0.5.12`) vs **iterative/greedy mutually-nearest with a relaxing threshold** | ByteTrack/BoT-SORT/OC-SORT use LAP; **TrackTrack uses iterative greedy** with `reduce_step = 0.05` per iteration **[SRC]**. Deep HM-SORT caps the assignment cost at 0.8 (and appearance cost at 0.3) — an explicit "refuse to match" rule. |
| **Score fusion** | Multiplicative: `cost' = 1 − (1 − cost)·score` | `fuse_score()` in ultralytics **[SRC]**; default True for all except it is *deliberately disabled* in ByteTrack's low-score second pass (fusing low scores pushes cost above the 0.5 gate) **[SRC]** comment in `byte_tracker.py`. |
| **Two-stage / low-score matching** | ByteTrack stage 2 | Yes for ByteTrack/BoT-SORT/FastTracker; **off by default** for OC-SORT/Deep OC-SORT (`use_byte: False`) **[SRC]**; TrackTrack instead keeps low-score dets in the pool *with a cost penalty* `penalty_p = 0.2` **[SRC]**. |
| **Extra detection bucket** | TrackTrack re-runs postprocess at loose NMS IoU 0.95, keeps boxes with max-IoU < 0.97 vs the tight results, and charges them `penalty_q = 0.4` | **[SRC]** `track_tracker.py` |
| **Track-aware initialisation (TAI)** | Suppress a new ID if it overlaps an existing track | TrackTrack `tai_thr = 0.55` **[SRC]**; FastTracker `init_iou_suppress = 0.7` **[SRC]**; both default-documented in [ultralytics docs](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md). |
| **Duplicate-ID suppression** | If two tracks have IoU > 0.85 (cost < 0.15), **drop the shorter-lived one** | **[SRC]** `utils/stracks.py::remove_duplicate_stracks(dup_thresh=0.15)` |
| **Cost geometry** | IoU (xyxy) · rotated ProbIoU (xywha) · **HMIoU = HIoU · IoU** where `HIoU = h_overlap/h_union` · centre distance · EIoU with bbox expansion | HMIoU is the partial-visibility cue: it rewards *vertical* overlap, so a track whose legs are occluded still matches its upper body. **[SRC]** `_hmiou_distance`; [arXiv:2308.00783](https://arxiv.org/abs/2308.00783) |
| **Lost-track re-identification** | (a) keep lost tracks for `track_buffer` frames; (b) **relaxed rebind pass** with a looser gate `lost_match_thr`; (c) **unbounded buffer**; (d) OCR pass on Lost tracks *after* active tracks | (b) TrackTrack `lost_match_thr = 0.0` (disabled) — docs advise raising above `match_thresh` (e.g. 0.9) for long occlusion **[SRC]**; (c) Deep HM-SORT: keeping all tracklets ≈ +2.7 HOTA and −300 IDSW by itself [arXiv:2406.12081](https://arxiv.org/html/2406.12081v1); (d) OC-SORT runs OCR twice, active first, "so a recently-lost track cannot outbid an active one" **[SRC]**. |
| **Gating** | Mahalanobis gate (SORT `gating_distance`, χ² thresholds 9.4877 for 4 DoF / 5.9915 for 2), appearance gate, proximity gate | **[SRC]** `kalman_filter.py` |

---

## 6. Benchmarks — concrete numbers

### 6.1 MOT17 / MOT20 (test, private detections) — association-focused view

| Tracker | MOT17 HOTA | MOT17 IDF1 | MOT17 MOTA | MOT20 HOTA |
|---|---|---|---|---|
| DeepSORT (with YOLOX-X) | 61.2 | 74.5 | 78.0 | — |
| ByteTrack | 63.1 | 77.3 | 80.3 | — |
| OC-SORT | 63.15 | 77.5 | 78.0 | 62.45 |
| Hybrid-SORT-ReID | 64.0 | — | — | 63.9 |
| StrongSORT++ | 64.3 | — | — | 63.65 |
| FineTrack | 64.4 | — | — | 62.65 |
| Deep OC-SORT | 64.95 | — | — | 63.9 |
| SparseTrack | 65.1 | — | — | 63.4 |
| BoT-SORT-ReID | 65.0 | 80.2 | 80.5 | — |
| UCMCTrack+ | 65.75 | — | — | 62.85 |
| FastTracker | 66.4 | — | — | 65.7 |
| **TrackTrack** | **67.1** | **83.1** | **81.8** (AssA 68.2) | **65.7** (IDF1 80.9, MOTA 78.0, AssA 67.3) |

Sources: TrackTrack MOT17/MOT20 rows and the prior-work HOTA pairs are from the CVPR 2025 paper; DeepSORT/ByteTrack/OC-SORT/BoT-SORT-ReID MOT17 rows also appear in the industry survey ([forasoft tracker blog, accessed 2026-10](https://www.forasoft.com/blog/article/multi-object-tracking-algorithms)) — that blog is a **secondary source**, flagged as such. FastTracker MOT17/MOT20 from [arXiv:2508.14370](https://arxiv.org/abs/2508.14370).

> **Read this table sceptically.** MOT17/MOT20 are motion-saturated pedestrian benchmarks; ByteTrack and OC-SORT differ by **0.1 HOTA** there but by **7.4 HOTA** on DanceTrack. Do not pick a tracker for this project from MOT17 numbers.

### 6.2 DanceTrack (test) — this project's "uniform appearance, nonlinear motion" case

Dataset: 100 videos, ~105k frames, 20 fps; 40/25/35 train/val/test; uniform appearance, diverse non-linear motion ([dancetrack.github.io](https://dancetrack.github.io/), [arXiv:2111.14690](https://arxiv.org/abs/2111.14690)).

| Tracker | HOTA | IDF1 | MOTA | Note |
|---|---|---|---|---|
| FairMOT | 39.7 | 40.8 | 82.2 | TBD+ReID |
| CenterTrack | 41.8 | 35.7 | 86.8 | |
| QDTrack | 45.7 | — | — | |
| DeepSORT | 45.6 | — | — | |
| SORT | 47.9 | — | — | |
| ByteTrack | 47.3 | 52.5 | 89.5 | no appearance, no OCM |
| FineTrack | 52.7 | — | — | |
| MOTR | 54.2 | — | — | E2E |
| OC-SORT | 54.6 | 54.6 | 89.6 | +7.3 HOTA over ByteTrack, same detections |
| GHOST | 56.7 | 57.7 | 91.3 | |
| Hybrid-SORT | 62.2 | 63.0 | 91.6 | weak cues, no ReID |
| SUSHI | 63.3 | 63.4 | 88.7 | offline GNN |
| **Hybrid-SORT-ReID** | **65.7** | 67.4 | 91.8 | best *heuristic* tracker |
| SambaMOTR (D-DETR) | 67.2 | 71.3 | 88.1 | E2E/SSM |
| SambaMOTR (DAB-D-DETR) | 69.0 | 74.2 | 89.1 | E2E/SSM |
| **MOTIP** | **69.6** (DetA 80.4, AssA 60.4) | — | — | E2E, official repo table |
| MOTRv2 | 69.9 | 71.7 | 91.9 | E2E + YOLOX |

Sources: Hybrid-SORT table rows ([arXiv:2308.00783](https://arxiv.org/html/2308.00783v2)); SambaMOTR rows from its official README ([github](https://github.com/mattiasegu/sambamotr), [project page](https://sambamotr.github.io/)); MOTIP row from the official model zoo ([docs/MODEL_ZOO.md](https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md)) — ⚠ the model-zoo file does not label the split in the table itself; the linked output paths are `.../test/tracker`, so these are **test** numbers, but treat as "official, not independently verified".
⚠ The live DanceTrack leaderboard ([Codabench comp. 14885](https://www.codabench.org/competitions/14885/)) is JS-rendered and could not be scraped — the current #1 entry is **not** verified here. MOTIP/SambaMOTR are the strongest *published* numbers I could confirm.

### 6.3 SportsMOT (test) — this project's "fast motion" case

Dataset: 240 videos (45/45/150 train/val/test), 150,379 frames, 1.63 M boxes, 3,401 tracks, 720p@25fps, 3 sports. Its two defining properties: **fast and variable-speed motion** and **similar-yet-distinguishable appearance**. Empirically it has the **lowest adjacent-frame IoU *and* the lowest Kalman-filter-based adjacent-frame IoU** of MOT17 / DanceTrack / SportsMOT (football worst) — i.e. a constant-velocity KF is provably inadequate. [arXiv:2304.05170](https://arxiv.org/html/2304.05170v2)

| Tracker | Setup | HOTA | IDF1 | AssA | MOTA | DetA | IDs |
|---|---|---|---|---|---|---|---|
| FairMOT | Train | 49.3 | 53.5 | 34.7 | 86.4 | 70.2 | 9928 |
| QDTrack | Train | 60.4 | 62.3 | 47.2 | 90.1 | 77.5 | 6377 |
| CenterTrack | Train | 62.7 | 60.0 | 48.0 | 90.8 | 82.1 | 10481 |
| ByteTrack | Train | 62.8 | 69.8 | 51.4 | 94.5 | 77.1 | 3267 |
| ByteTrack | Train+Val | 64.1 | 71.4 | 52.3 | 95.9 | 78.5 | 3089 |
| MixSort-Byte | Train+Val | 65.7 | 74.1 | 54.8 | 96.2 | 78.8 | 2472 |
| BoT-SORT | Train | 68.7 | 70.0 | 55.9 | 94.5 | 84.4 | 6729 |
| MeMOTR (D-DETR) | Train | 68.8 | — | 57.8 | — | 82.0 | — |
| TransTrack | Train | 68.9 | 71.5 | 57.5 | 92.6 | 82.7 | 4992 |
| MeMOTR | Train | 70.0 | — | 59.1 | — | 83.1 | — |
| **OC-SORT** | Train | **71.9** | 72.2 | 59.4 | 95.4 | 86.4 | **3093** |
| OC-SORT | Train+Val | 73.7 | 74.0 | 61.5 | 96.5 | 88.5 | 2728 |
| MotionTrack | Train | 74.0 | — | 61.7 | — | — | — |
| MixSort-OC | Train+Val | 74.1 | 74.4 | 62.0 | 96.5 | 88.5 | 2781 |
| Deep-EIoU | Train | 74.1 | 75.0 | 63.1 | 95.1 | 87.2 | 3066 |
| MoveSORT | Train+Val | 74.6 | — | 63.7 | — | — | — |
| Deep-EIoU | Train+Val | 77.2 | 79.8 | 67.7 | 96.3 | 88.2 | 2659 |
| **Deep HM-SORT** | Train+Val | **80.1** | **85.2** | **72.7** | **96.6** | 88.3 | **1703** |

Table reproduced from [Deep HM-SORT, arXiv:2406.12081](https://arxiv.org/html/2406.12081v1) Table 1 (SportsMOT test set).

For reference, the two E2E SOTA trackers on SportsMOT: **MOTIP 72.6 / DetA 83.5 / AssA 63.2** ([model zoo](https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md)) and **SambaMOTR 70.5 (D-DETR) / 72.1 (DAB-D-DETR)** ([README](https://github.com/mattiasegu/sambamotr)). Note these are *well below* the sports-specialised heuristic trackers — the sports benchmark rewards task-specific association engineering over generic E2E models.

**Deep HM-SORT's ablation is the most transferable experiment in this section** (SportsMOT test, same hyperparameters):

| HM fusion | Keep all tracks | HOTA | IDSW |
|---|---|---|---|
| — | — | 77.5 | 2721 |
| — | ✓ | **80.2** | 2418 |
| ✓ | — | 77.21 | **2146** |
| ✓ | ✓ | 80.1 | **1703** |

→ **An unbounded (or very long) track buffer is the HOTA lever; harmonic-mean fusion is the ID-switch lever.** Both are cheap.

### 6.4 Other benchmarks (stated scope, less depth)

- **BFT (bird flock tracking)**: MOTIP 70.5 HOTA (DetA 69.6, AssA 71.8); SambaMOTR 69.6 (D-DETR) → 72.1 (DAB-D-DETR). Sources as above.
- **BDD100K MOT / VisDrone / KITTI tracking / PersonPath22**: named in the task, but I did not collect comparable HOTA tables for them in this session (KITTI/ VisDrone are dominated by vehicles/small objects; PersonPath22 ⚠ I could not verify a canonical arXiv entry). **Not carried into the decision table** — none of them matches this project's failure modes (people, handheld/moving camera, occlusion, partial body) better than DanceTrack/SportsMOT/MOT17.

---

## 7. Ablations that actually transfer

1. **Same detections, better association = +7.3 HOTA.** ByteTrack → OC-SORT on DanceTrack (47.3 → 54.6). The lever is observation-centric logic, not a bigger model. [arXiv:2308.00783](https://arxiv.org/html/2308.00783v2)
2. **Weak cues ≈ free accuracy.** Hybrid-SORT vs OC-SORT: **+7.6 HOTA with identical association inputs and nearly identical compute**. Confidence state + height state + velocity direction are cheap and strong. (same source)
3. **Extending the track buffer / never deleting tracklets is the top occlusion knob.** Deep HM-SORT: +2.7 HOTA and −300 IDSW from "keep all tracks" alone. (SportsMOT ablation above)
4. **Harmonic-mean fusion trades HOTA for ID stability.** −21% ID switches for ≈0 HOTA. If your KPI is "no ID swap on the clicked target", HM fusion is the right cost. (same)
5. **Low-confidence second-stage matching helps only when it is IoU-only and score-fusion is off.** Ultralytics' own comment: fusing low scores pushes the cost above the 0.5 gate **[SRC]**.
6. **Init suppression / TAI kills duplicate IDs** cheaply: FastTracker `init_iou_suppress`, TrackTrack `tai_thr` + `min_track_len = 3` **[SRC]**; docs advise lowering to 0.45–0.5 in crowds.
7. **Rollback beats coasting through an occlusion.** FastTracker restores position+covariance from 3 frames back and velocity from 5 frames back, damps velocity ×0.5 and enlarges height ×1.1 **[SRC]**.
8. **Appearance must be gated or it hurts**: BoT-SORT's `proximity_thresh` / `appearance_thresh` gates exist precisely because ungated appearance costs ID switches in look-alike crowds. Deep OC-SORT's `α = 1.0` for low-trust detections is the same lesson applied to the EMA **[SRC]**.
9. **MOTIP's trajectory augmentation** (random token drop + random ID swap during training) improves DanceTrack HOTA 65.2 → 69.6 in the official ablation — i.e. **training the matcher to survive its own ID errors** is worth ~4.4 HOTA. ([model zoo](https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md))
10. **Detector NMS IoU is a hidden tracker knob.** TrackTrack re-runs postprocess at NMS IoU 0.95 and feeds the recovered boxes in as a *penalised third bucket* **[SRC]** — a cheap way to recover detections that crowd-NMS deletes.

---

## 8. Decision table for THIS project (ultralytics 8.4.185, RTX 4060 Laptop 8 GB VRAM)

### 8.0 What we have to work with

Six backends, all reachable as `model.track(..., tracker="<name>.yaml", persist=True)`, mapped in `ultralytics/trackers/track.py`:

```
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT, "tracktrack": TRACKTRACK,
               "fasttrack": FASTTracker, "ocsort": OCSORT, "deepocsort": DeepOCSORT}
```
**[SRC]** `C:\Projects\track\_p2\ul_wheel\ultralytics\trackers\track.py`

`tracktrack.yaml` is the **default** (docs). All six are pure-Python/NumPy + OpenCV — CPU cost is negligible next to the detector on a 4060.

### 8.1 (a) All-person ID layer

| Rank | Config | Why | Watch out |
|---|---|---|---|
| **1** | `tracktrack.yaml` (default) with `tai_thr: 0.45`, `track_buffer: 60–90`, `lost_match_thr: 0.9`, `min_track_len: 2` | Newest of the six (CVPR 2025), multi-cue (HMIoU + confidence-projection + corner angle), iterative assignment, TAI duplicate suppression, and — uniquely — **an explicit relaxed lost-rebind pass** for long occlusion. Best public MOT17/MOT20 HOTA (67.1 / 65.7). Its defaults (`track_high 0.6`, `new_track_thresh 0.7`) are conservative: if the detector is only `yolo26n`, they will starve the tracker. | Raise `track_high_thresh` sensitivity: lower to ~0.3–0.4 for a small model, or accept fewer but cleaner tracks. `lost_match_thr = 0` by default — **you must turn the rebind pass on**. |
| **2** | `botsort.yaml` + `gmc_method: sparseOptFlow` (+ `with_reid: True, model: auto` if look-alikes) | The only backend whose GMC is genuinely designed for it, and the highest-motivation choice for a **handheld/moving camera** (this project is DJI-ActiveTrack-like). BoT-SORT-ReID = 65.0 HOTA on MOT17 vs ByteTrack 63.1. | GMC adds real per-frame cost (optical flow + RANSAC) and can fail on low-texture frames (falls back to identity). `with_reid` on DanceTrack-like uniform-appearance data will *hurt*. |
| **3** | `ocsort.yaml` (optionally `use_byte: True`) | **The nonlinear-motion specialist at zero extra cost**: OCM + OCR + ORU; DanceTrack 54.6 vs ByteTrack 47.3 with the same detections. No GMC and no appearance ⇒ also the most robust to weird appearance distributions. | No camera-motion compensation ⇒ on a moving camera the IoU stage degrades. No ReID ⇒ long occlusions are only survived via `track_buffer` + ORU. |
| 4 | `deepocsort.yaml` + `gmc_method: sparseOptFlow` | OC-SORT + *trust-weighted* appearance + CMC — the "crowded moving camera, ID swaps dominate" pick per the docs. | The stock YAML ships `gmc_method: none` (a trap for this project) and `appearance_thresh: 0.9` (stricter than BoT-SORT's 0.8). |
| 5 | `fasttrack.yaml` | Occlusion-aware ByteTrack; the Kalman-rollback machinery is the most valuable *idea* here, but as a full MOT layer it is ByteTrack-class on nonlinear motion (no OCM/OCR). | No GMC, no appearance. Its knobs are effectively a superset of what you want in the single-target layer. |
| 6 | `bytetrack.yaml` | Minimum overhead; useful as the *control* baseline only. DanceTrack 47.3 HOTA. | No appearance, no CMC, no observation-centric correction. |

**Concrete first experiment (cheap, decisive).** Run all six on this project's own people footage (moving camera + crowd + fast motion), same `yolo26n` detections, same `conf`, with `persist=True`, and score HOTA/IDF1/AssA with TrackEval plus a "no-ID-swap" rate. Predicted ranking, from the evidence above: `tracktrack` (with `lost_match_thr` on) ≥ `botsort`+sparseOptFlow > `deepocsort`+sparseOptFlow > `ocsort` > `fasttrack` > `bytetrack` on crowds; `ocsort` can jump to #1 if the camera is static and motion is nonlinear. **ReID should be evaluated as an on/off switch per config, not assumed to help.**

### 8.2 (b) Component/idea source for the dedicated single-target tracker

| Rank | Source | What to lift |
|---|---|---|
| **1** | `fast_tracker.py` (`FastSTrack`, `FASTTracker._handle_occlusions`) | The **occlusion state machine** + Kalman rollback + search expansion + init suppression. This is *already written for a per-track perspective* and is ~8 hyperparameters. Direct copy targets: coverage-based occlusion detection (`bbox_ioa` of a track against other unmatched boxes, threshold 0.7), `mean[0:4]`/covariance restore, velocity restore, `h *= 1.1`, `v *= 0.5`, `max_frames_lost + occ_reappear_window`. |
| **2** | `oc_sort.py` (`apply_oru`, `_velocity_direction_cost`) | **ORU** is the correct way to bridge a gap: restore → replay virtual observations → predict. **OCM** is a cheap direction-consistency cost for re-bind decisions after a loss. |
| **3** | `kalman_filter.py::project(..., confidence)` | **NSA scaling** — one line, free, and it is exactly the "trust the observation proportionally" behaviour you want for a target detected at varying confidence (partial body, motion blur). |
| **4** | `deep_oc_sort.py::DeepOCSortTrack.update_features` | **Trust-weighted EMA** for the target's short-term appearance memory (`α = 0.95 + 0.05(1 − trust)`, `α = 1.0` when the detection is below threshold). This is the guard that keeps a partial-view crop from poisoning the gallery. |
| **5** | `track_tracker.py` (`TTSTrack.update_features`, `_confidence_distance`, `lost_match_thr` pass) | Score-adaptive EMA, **confidence-projection distance** `|s + (s − s_prev) − s_det|`, and the **relaxed lost-rebind gate** — the closest thing in the wheel to "re-acquire the target after a long absence". |
| **6** | `utils/gmc.py` (`GMC.apply`) + `utils/stracks.py::multi_gmc` | Camera-motion compensation for a *single* target's KF state. **Critical caveat [SRC]**: `multi_gmc`'s docstring says *"This assumes an XYWH state layout; XYAH trackers must override this."* `DeepOCSortTrack.multi_gmc` is the correct XYAH version (`R8x8[:2,:2] = R`, `R8x8[4:6,4:6] = R`, a/h left identity). If your single-target KF is XYAH, port that one, not the shared helper. |
| **7** | `track_tracker.py::_hmiou_distance` | **HMIoU** (`HIoU·IoU`, `HIoU = h_over/h_union`) — the only built-in cost that is *specifically designed to survive vertical truncation*, i.e. the upper-body-only / lower-body-only case that the project brief calls out. |
| **8** | `utils/reid.py` (`ReID`, `build_encoder`, `smooth_feature`) | The ready-made crop→embedding→L2-norm→EMA pipeline, plus `model: auto` (detector backbone features, near-free) and the ONNX encoders `yolo26{n,s,m,l,x}-reid.onnx` (2.8 M params / 2.0 GFLOPs for `n`, 448² input per the docs table). On 8 GB VRAM, `yolo26n-reid.onnx` is the only sane choice at frame rate. |

### 8.3 What is **not** worth importing

- Full MOT association machinery (Hungarian over N×M, unbounded tracklet pools, per-frame duplicate-ID suppression) — the single-target problem is a *detection-selection* problem, not an assignment problem. Keep the **cost formulation** and the **state machine**, drop the solver.
- Offline post-processing (GSI/AFLink interpolation, SUSHI-style GNN re-matching) — incompatible with online, frame-by-frame latency budget.
- E2E trackers (MOTIP/SambaMOTR/MOTRv2) as *components*: they need GPU training and a DETR backbone; the transferable part is only the **memory-update discipline** (MaskObs, trajectory augmentation) and the **in-context ID formulation**, both of which you can approximate with a gated gallery + a small cosine/MLP scorer.

### 8.4 8 GB VRAM budget sanity

- Detector `yolo26n` (person) at 640² + `yolo26n-reid.onnx` at 448² per crop is comfortably within 8 GB; the real risk is **batching many crops** — `ReID` pads fixed-batch ONNX models with `chunk[-1:]`, so a large crop batch inflates memory. Cap the number of persons you embed per frame (the clicked target + a bounded candidate set) rather than embedding every person every frame. **[SRC]** `utils/reid.py`.
- `gmc_method: sparseOptFlow` costs CPU (400 corners, LK, RANSAC), not VRAM.

---

## 9. Open questions / flagged unverified

1. **DanceTrack live leaderboard** — could not scrape ([Codabench 14885](https://www.codabench.org/competitions/14885/) is JS). MOTIP 69.6 / MOTRv2 69.9 / SambaMOTR 69.0 are the best *published* values I verified; a newer entry may exist.
2. **MOTIP model-zoo split labelling** — table does not state train/val/test; linked output paths imply test. Numbers: DanceTrack 69.6 (65.2 w/o traj-aug), SportsMOT 72.6 (70.9), BFT 70.5 (71.3 w/o).
3. **MambaTrack numbers** — the abstract confirms DanceTrack/SportsMOT gains but I did not extract its result tables. ⚠ no numbers quoted.
4. **IMM / Ornstein–Uhlenbeck in MOT** — no verified 2021–2026 top-venue TBD tracker uses either. Do not treat them as validated recipes.
5. **MOTRv3, SUSHI, SparseTrack, UCMCTrack, McByte, FineTrack, PersonPath22** — cited for lineage/completeness; arXiv IDs quoted without re-verifying the current version.
6. **SportsMOT `Deep HM-SORT` 80.1 HOTA** relies on the authors' reading of Papers-With-Code/leaderboard numbers; the same table lists OC-SORT at 71.9/73.7 which is consistent with the SportsMOT paper's own benchmark, so I treat the table as trustworthy but second-hand.
7. **DanceTrack "better motion beats better appearance"** — supported by the dataset's own analysis page and by the Hybrid-SORT ablation (same detections, +7.6 HOTA from weak cues). I did not read the DanceTrack paper's full appendix in this session.

---

## 10. Ultralytics 8.4.185 implementation audit (verified from source)

All paths below are inside `C:\Projects\track\_p2\ul_wheel\ultralytics\`. Line counts are total file lengths; symbol names are exact. This is the section to grep when implementing.

### 10.1 `trackers/utils/kalman_filter.py` (485 lines)

- `class KalmanFilterXYAH` — 8-dim state `(x, y, a, h, vx, vy, va, vh)`. `_motion_mat = eye(8) + I[0:4, 4:8]·dt` with `ndim, dt = 4, 1.0`; `_update_mat = np.eye(4, 8)`; `_std_weight_position = 1.0/20`; `_std_weight_velocity = 1.0/160` (SORT/DeepSORT values).
- `initiate(measurement) -> (mean, cov)`; std vector `[2·swp·h, 2·swp·h, 1e-2, 2·swp·h, 10·swv·h, 10·swv·h, 1e-5, 10·swv·h]`.
- `predict`, `project(mean, covariance, confidence=None)`, `multi_predict`, `update(mean, covariance, measurement, confidence=None)`, `gating_distance(mean, covariance, measurements, only_position=False, metric="maha")` (χ² thresholds 9.4877 for 4 DoF / 5.9915 for 2 in the docstring).
- **NSA Kalman is present**: inside `project()`, `if confidence is not None: innovation_cov *= max(1.0 - float(confidence), 0.05)`.
- `class KalmanFilterXYWH(KalmanFilterXYAH)` — state `(x, y, w, h, vx, vy, vw, vh)`; overrides `initiate/predict/project/multi_predict`; noise scaled by `measurement[2]`/`measurement[3]` (w,h).

### 10.2 `trackers/utils/matching.py` (168 lines)

- `linear_assignment(cost_matrix, thresh, use_lap=True)` → `lap.lapjv(..., extend_cost=True, cost_limit=thresh)`, else `ultralytics.utils.ops.linear_sum_assignment` + `cost <= thresh` filter. `lap >= 0.5.12` auto-installed by `check_requirements`.
- `iou_distance(atracks, btracks)` → `1 − iou`; uses `track.xywha` when `track.angle is not None` → `batch_probiou` (rotated IoU), else `bbox_ioa(..., iou=True)`.
- `embedding_distance(tracks, detections)` → `1 − T@D.T / max(‖T‖·‖D‖, eps)` clipped at ≥ 0, on `t.smooth_feat` vs `d.curr_feat`; **missing features are set to `2.0`**, callers divide by 2 so a missing feature is an automatic non-match.
- `fuse_score(cost_matrix, detections)` → `1 − (1 − cost)·det_score` (multiplicative).

### 10.3 `trackers/utils/gmc.py` (355 lines)

`class GMC(method="sparseOptFlow", downscale=2)`; `apply(raw_frame, detections=None) -> np.ndarray(2,3)`; `reset_params()`.
- `sparseOptFlow`: `goodFeaturesToTrack({maxCorners:400, qualityLevel:0.01, minDistance:0, blockSize:3, useHarrisDetector:False, k:0.04})` → `calcOpticalFlowPyrLK` → `estimateAffinePartial2D(prev, curr, cv2.RANSAC)`, requires `prevPoints.shape[0] > 4`; translation rescaled by `downscale`.
- `orb`: `FastFeatureDetector_create(20)` + `ORB_create()` + `BFMatcher(NORM_HAMMING)`. `sift`: `SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)` + `BFMatcher(NORM_L2)`.
- `ecc`: `MOTION_EUCLIDEAN`, criteria `(EPS|COUNT, 5000, 1e-6)`, GaussianBlur(3,3) σ=1.5.
- Feature paths use a 2 %–98 % border mask, zero out detection boxes from the mask, Lowe ratio `m.distance < 0.9·n.distance`, spatial outlier filter `≤ 2.5·std`.
- `method in {"none", "None", None}` → identity warp. ECC failure logs `findTransformECC failed; using identity warp.`

### 10.4 `trackers/byte_tracker.py` (549 lines) — ByteTrack + the hook architecture

- `class STrack(BaseTrack)`; `shared_kalman = KalmanFilterXYAH()`; `__init__(xywh, score, cls)` accepts `(x,y,w,h,idx)` or `(x,y,w,h,angle,idx)`; `predict()` **zeroes `mean_state[7]` (= vh) when state != Tracked**; `re_activate(new_track, frame_id, new_id=False)`; `update(new_track, frame_id)`; `tlwh_to_xyah`; `result = [*xyxy, track_id, score, cls, idx]`.
- `class BYTETracker` (`track_class = STrack`), `update(self, results, img=None, feats=None, **kwargs) -> np.ndarray`, orchestrated through overridable hooks — this is the extension point to use:
  `_split_detections` (high: `conf >= track_high_thresh`; low: `track_low_thresh < conf < track_high_thresh`; boxes with `w <= 0 or h <= 0` dropped) → `_input_for` → `_split_tracked` → `_pre_first_associate` (applies `multi_gmc` to `strack_pool` and `unconfirmed` when `self.gmc.method is not None and img is not None`) → `_first_association` (`thresh = match_thresh`) → `_post_first_association` (identity in the base class) → `_second_association` (remaining Tracked vs low-score dets; **IoU-only + `fuse_score`, `thresh = 0.5` hard-coded**; then mark_lost) → `_unconfirmed_association` (`thresh = 0.7` hard-coded) → `_init_new_tracks` (`score >= new_track_thresh`) → `_remove_stale_lost` (`frame_id - end_frame > max_frames_lost`) → `_format_output`.
- `max_frames_lost = args.track_buffer`; `get_dists` = `iou_distance` + optional `fuse_score`; `get_kalmanfilter() -> KalmanFilterXYAH()`; `reset_id()` uses `iter(range(1, sys.maxsize))`.
- Source comment at `_second_association`: *"IoU-only by design (ByteTrack paper sec. 3.2): fusing low scores pushes costs above the 0.5 threshold"*.

### 10.5 `trackers/bot_sort.py` — BoT-SORT

- `class BOTrack(STrack)`, `shared_kalman = KalmanFilterXYWH()`, `self.alpha = 0.9` (hard-coded EMA), `update_features(feat)` → `smooth_feature(feat, self.smooth_feat, self.alpha)`; `predict()` zeroes `mean_state[6]` (vw) **and** `[7]` (vh) when not Tracked.
- `class BOTSORT(BYTETracker)`: builds `self.gmc = GMC(method=args.gmc_method)`, `self.encoder = build_encoder(args.with_reid, getattr(args, "model", "auto"), ...)`.
- `get_dists` cost pipeline: `dists = iou_distance(...)`; `dists_mask = dists > (1 - proximity_thresh)`; `fuse_score`; if ReID: `emb = embedding_distance(...)/2`, `emb[emb > (1 - appearance_thresh)] = 1`, `emb[dists_mask] = 1`, `dists = np.minimum(dists, emb)` — **gated min-fusion**.

### 10.6 `trackers/oc_sort.py` (356 lines) — OC-SORT

- `class OCSortTrack(STrack)`: `last_observation` (xyxy, init `[-1,-1,-1,-1]`), `observations: dict[int, xyxy]` pruned to `delta_t + 2`, `velocity` (unit dx,dy), `_saved_mean`/`_saved_covariance`.
- `_compute_velocity()`: centre at the max frame key; most recent observation with `frame < current_frame - delta_t + 1`, else the earliest; normalised (None if < 2 observations; zeros if ‖·‖ < 1e-6).
- `apply_oru(new_observation_xyxy, current_frame_id)`: restore `mean/covariance` from `_saved_*`, then for `t in 1..gap-1` build `virtual_xyxy = (1-α)·last_obs + α·new_obs` with `α = t/gap`, convert xyxy→tlwh→xyah, `predict` + `update` with the virtual measurement, then a final `predict`.
- `class OCSORT(BYTETracker)`, `track_class = OCSortTrack`; defaults `delta_t = 3`, `inertia = 0.2`, `use_byte = False`.
- `get_dists` = `iou_distance` (+ `fuse_score`) `+ inertia · _velocity_direction_cost(...)` then the `_fuse_appearance` hook. `_velocity_direction_cost` = `arccos(clip(dot))/π` between track velocity and (last_observation→det) direction.
- `_ocr_associate`: `_ocr_distance` computes IoU distance against **`last_observation`**, not the KF prediction (OBB falls back to predicted `xywha`); uses `linear_assignment(thresh=args.match_thresh)`; on a Lost-track match: `apply_oru(...)` then `re_activate(..., new_id=False)`.
- `_post_first_association` runs OCR **twice** — Tracked-unmatched first, then Lost-unmatched on the remaining detections.
- `_second_association`: if `use_byte` is False, just `mark_lost` (no low-score pass).

### 10.7 `trackers/deep_oc_sort.py` (267 lines) — Deep OC-SORT

- `class DeepOCSortTrack(OCSortTrack)`: `alpha_fixed_emb = 0.95`, `det_thresh = 0.25`; `update_features(feat, score=None)`: if `score > det_thresh`: `trust = (score - det_thresh)/(1 - det_thresh)`, `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb)·(1 - trust)`; **else `alpha = 1.0`** (keep the old embedding). Applied on `update` and `re_activate`.
- `DeepOCSortTrack.multi_gmc(stracks, H)`: the XYAH-correct GMC — `R8x8 = eye(8)` with only `R8x8[:2,:2] = R` and `R8x8[4:6,4:6] = R`, `mean[:, :2] += t`; also rotates `last_observation` centres so OCR/ORU stay consistent.
- `class DeepOCSORT(OCSORT)`: `proximity_thresh = 0.5`, `appearance_thresh = 0.75` (getattr default; the YAML ships 0.9), `det_thresh = args.track_high_thresh`; `_pre_first_associate` applies `DeepOCSortTrack.multi_gmc`; `_fuse_appearance` = BoT-SORT-style gated min-fusion.

### 10.8 `trackers/fast_tracker.py` — FastTracker

- `class FastSTrack(STrack)`: `mean_history: deque` (`history_len = 16` default), `not_matched`, `is_occluded`, `occluded_len`, `last_occluded_frame`, `was_recently_occluded`; `_push_history()` appends `(mean.copy(), covariance.copy())` on activate/update/re_activate.
- `class FASTTracker(BYTETracker)`, `track_class = FastSTrack`; `_history_len = max(reset_velocity_offset_occ, reset_pos_offset_occ) + 4` (= 9 with defaults).
- `_handle_occlusions(r_tracked, u_track, activated_stracks, lost_stracks)`: `cov = bbox_ioa(active_boxes, unmatched_boxes)` (intersection / unmatched area), self-pairs zeroed by track id, `max_cov = cov.max(axis=0)`; if `max_cov[i] > occ_cover_thresh (0.7)` and the track is Tracked and not already occluded → set `is_occluded`, then **rollback**:
  `mean[4:8] = hist[-reset_velocity_offset_occ][0][4:8]` (velocity from 5 frames back);
  `mean[0:4] = hist[-reset_pos_offset_occ][0][0:4]` **and** `covariance = hist[-reset_pos_offset_occ][1].copy()` (position + covariance from 3 frames back);
  one-shot `mean[3] *= enlarge_bbox_occ (1.1)` (height; width follows from `w = a·h`) and `mean[4:8] *= dampen_motion_occ (0.5)`.
- Lost transition: `track.not_matched > 2 and (not is_occluded or occluded_len > active_occ_to_lost_thresh (10))`.
- `_remove_stale_lost` keeps recently-occluded lost tracks re-findable for `occ_reappear_window (40)` frames beyond `track_buffer`. `_init_new_tracks` suppresses a spawn if `bbox_ioa(det.xyxy, active_stack, iou=True).max() >= init_iou_suppress (0.7)`. `_format_output` emits only tracks with `frame_id == self.frame_id`.

### 10.9 `trackers/track_tracker.py` (607 lines) — TrackTrack (CVPR 2025)

Module constants: `_CORNER_DX_IDX = [0,0,2,2]`, `_CORNER_DY_IDX = [1,3,1,3]`, `_LOOSE_NMS_IOU = 0.95`, `_LOOSE_NMS_DEDUP_IOU = 0.97`.

- `_hmiou_distance(tracks_a, tracks_b) -> (iou_sim, 1 - HMIoU)` with `HMIoU = HIoU · IoU`, `HIoU = h_over/h_union` clipped to [0,1].
- `_angle_distance(tracks, dets, frame_id, pairs, delta_t=3)`: per-corner unit vectors `det_box − track_box`, dotted with `track.velocity` (4,2); `dist = mean(|arccos(clip(dot))|)/π`; multiplied by the detection score.
- `_confidence_distance(tracks, dets)`: `|track_proj_score − det_score|`, `track_proj_score = score + (score − prev_score)`.
- `_iterative_associate(cost, match_thr, reduce_step=0.05)`: greedy mutually-nearest; matched rows/cols set to `inf`, `match_thr -= reduce_step` each iteration. **No Hungarian.**
- `_track_aware_nms(tracks, dets, tai_thr, new_track_thresh)`: keep `score > new_track_thresh`; drop dets with `iou[det, track].max() > tai_thr`; then greedy NMS among dets at `tai_thr`.
- `_cosine_distance(tracks, dets)`: cosine in [0,1] on `smooth_feat or curr_feat`; **returns NaN where a feature is missing** so the caller falls back to motion.
- `attach_raw_preds_hook(predictor)` + `compute_dets_del(predictor)`: re-run `predictor._orig_postprocess` at loose NMS IoU 0.95 and keep boxes with `max_iou < 0.97` vs the tight results → the "recovered" detection bucket.
- `class TTSTrack(BOTrack)`: `min_track_len = 3`, `_alpha = 0.95`, `_delta_t = 3`; `prev_score`, `velocity` (4,2), `_history: deque(maxlen=delta_t+1)` of `(frame_id, xyxy)`; **score-adaptive EMA** `beta = 0.95 + 0.05·(1 − score)`; `update()` recomputes corner velocity as `(1/delta_t)·Σ_{dt=1..delta_t} unit(curr_box − get_history_box(frame_id, dt))/dt`; Kalman `update`/`re_activate` pass `confidence=new_track.score` (NSA); promoted to Tracked when `tracklet_len + 1 >= min_track_len`.
- `class TRACKTRACK` — **not** a `BYTETracker` subclass; `max_time_lost = args.track_buffer`; `self.kalman_filter = KalmanFilterXYWH()`; `self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))`; `self.encoder = build_encoder(with_reid, model, device)`.
- `_cost_matrix(tracks, dets)`: `iou_sim, hmiou_dist = _hmiou_distance(...)`; with encoder `cost = where(isnan(cos), hmiou_dist, iou_weight·hmiou_dist + reid_weight·cos)`, else `hmiou_dist`; `cost += conf_weight·_confidence_distance(...)`; only for `iou_sim > 0.10` add `angle_weight·_angle_distance(...)`; unsupported pairs `= 1.0`; `np.clip(cost, 0, 1)`.
- `update(results, img=None, dets_del=None, **kwargs)`: three buckets (high / low / recovered) with `cost[:, n_high:n_high+n_low] += penalty_p (0.2)` and recovered `+= penalty_q (0.4)`; GMC via `_apply_gmc`; main `_iterative_associate(cost, match_thr)`; then unconfirmed vs leftover-high; then the **optional relaxed lost rebind** when `lost_match_thr > 0` (`_cost_matrix` on still-Lost tracks); then `_track_aware_nms` for spawning; lost pruned at `frame_id - end_frame > max_time_lost`; output only `is_activated and frame_id == self.frame_id`.
- `setup_predictor`/`compute_frame_extras` classmethods restrict the loose-NMS recovery to `{detect, obb}` + `DetectionPredictor` (Segment/Pose skip it so masks/keypoints stay aligned).

### 10.10 `trackers/utils/reid.py` (186 lines), `utils/stracks.py` (180), `basetrack.py`, `track.py`

- `REID_ASSETS = frozenset(f"yolo26{k}-reid.onnx" for k in "nsmlx")`; `class ReID(model, imgsz=224, device=None, fp16=False)`.
  - `.pt` → `YOLO(model)` + `model.embed(np.zeros((32,32,3), np.uint8), device, ...)`.
  - non-`.pt` → `AutoBackend`, auto-download via `attempt_download_asset`, reads the ONNX input shape from `session.get_inputs()[0].shape` (static batch/imgsz supported), pads the last chunk with `chunk[-1:]`.
  - `__call__(img, dets) -> list[np.ndarray | None]`: crops with `save_one_box(det, img, save=False, BGR=True)` on `xywh2xyxy(dets[:, :4])`, BGR→RGB, normalise to BCHW at `self.imgsz`; `None` for empty crops.
  - Docs list the ONNX encoders at **448²** (n 2.8 M / 2.0 GFLOPs; s 7.5/6.6; m 12.4/20.1; l 15.3/25.2; x 32.7/55.9) while the Python default is `imgsz=224` — for ONNX the shape comes from the file, so the effective input is 448; flag as an inconsistency to be aware of. [docs](https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md)
- `build_encoder(with_reid, model, device=None)`: `False → None`; `"auto" → _auto_encoder(feats, _dets)` (pass-through of detector backbone features); else `ReID(model, device=device)`.
- `smooth_feature(feat, smooth, alpha) -> (curr, smooth)`: float32; `‖feat‖ < 1e-12 → (None, smooth)`; L2-normalise; first call returns `(feat, feat.copy())`; else `smooth = alpha·smooth + (1-alpha)·feat` then re-normalise.
- `utils/stracks.py`: `joint_stracks, merge_track_pools, multi_gmc, parse_bboxes, remove_duplicate_stracks, sub_stracks`.
  - `parse_bboxes(results) = concat([results.xywhr or results.xywh, arange(N)])` — the appended column is the **detection index**, and `STrack.result` carries it through to the returned array.
  - `remove_duplicate_stracks(atracks, btracks, dup_thresh=0.15)`: pairs with `iou_distance < 0.15` (IoU > 0.85) are duplicates; **the shorter-lived track (smaller `frame_id - start_frame`) is dropped**.
  - `multi_gmc(stracks, H)`: XYWH version, `R8x8 = np.kron(np.eye(4), R)`, then `R8x8[2:4,2:4] = R8x8[6:8,6:8] = np.abs(R)` ("sizes never flip sign"), centre translated. Docstring warns: *"This assumes an XYWH state layout; XYAH trackers must override this."*
- `basetrack.py`: `class TrackState: New=0, Tracked=1, Lost=2, Removed=3`; `class BaseTrack` with `_count`, `next_id()`, `mark_lost()`, `mark_removed()`, `end_frame`.
- `track.py`: `TRACKER_MAP` (quoted in §8.0); `on_predict_start` loads the YAML via `check_yaml` + `IterableSimpleNamespace`, sets `cfg.device = predictor.device`, and raises `AssertionError(f"Only {sorted(TRACKER_MAP)} are supported for now, but got '{cfg.tracker_type}'")` for anything else.
  - **ReID "auto" wiring**: for `tracker_type in {"botsort","tracktrack","deepocsort"}` with `with_reid: True` and `model: auto`, if the detector head is not a plain (non-end2end) `Detect` it **falls back to `cfg.model = "yolo26n-cls.pt"`**; otherwise it registers a forward-pre-hook on `predictor.model.model.model[-1]` capturing `input[0]` into `predictor._feats` (native backbone features).
  - `on_predict_postprocess_end` calls `tracker.update(det, result.orig_img, feats=getattr(result, "feats", None), dets_del=...)`, hides unconfirmed tracks via `result[:0]`, and writes back `tracks[:, :-1]`.

### 10.11 Verbatim YAML defaults

| key | bytetrack | botsort | ocsort | deepocsort | fasttrack | tracktrack |
|---|---|---|---|---|---|---|
| `track_high_thresh` | 0.25 | 0.25 | 0.25 | **0.3** | 0.25 | **0.6** |
| `track_low_thresh` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | **0.25** |
| `new_track_thresh` | 0.25 | 0.25 | 0.25 | **0.3** | 0.25 | **0.7** |
| `track_buffer` | 30 | 30 | 30 | 30 | 30 | 30 |
| `match_thresh` | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | **0.7** |
| `fuse_score` | True | True | True | True | True | — (n/a) |
| `gmc_method` | — | sparseOptFlow | — | **none** | — | sparseOptFlow |
| `with_reid` | — | False | — | False | — | False |
| `model` | — | auto | — | auto | — | auto |
| `proximity_thresh` | — | 0.5 | — | 0.5 | — | — |
| `appearance_thresh` | — | **0.8** | — | **0.9** | — | — |
| `alpha_fixed_emb` | — | — | — | 0.95 | — | — |
| `delta_t` / `inertia` / `use_byte` | — | — | 3 / 0.2 / False | 3 / 0.2 / False | — | — |
| tracker-specific | — | — | — | — | `reset_velocity_offset_occ 5`, `reset_pos_offset_occ 3`, `enlarge_bbox_occ 1.1`, `dampen_motion_occ 0.5`, `active_occ_to_lost_thresh 10`, `occ_cover_thresh 0.7`, `occ_reappear_window 40`, `init_iou_suppress 0.7` | `lost_match_thr 0.0`, `iou_weight 0.5`, `reid_weight 0.5`, `conf_weight 0.1`, `angle_weight 0.05`, `penalty_p 0.2`, `penalty_q 0.4`, `reduce_step 0.05`, `tai_thr 0.55`, `min_track_len 3` |

**[SRC]** `C:\Projects\track\_p2\ul_wheel\ultralytics\cfg\trackers\*.yaml`

---

## 11. References

Primary papers

- SORT — ICIP 2016 — https://arxiv.org/abs/1602.00763
- DeepSORT — ICIP 2017 — https://arxiv.org/abs/1703.07402
- ByteTrack — ECCV 2022 — https://arxiv.org/abs/2110.06864
- OC-SORT — CVPR 2023 — https://arxiv.org/abs/2203.14360
- BoT-SORT — 2022 — https://arxiv.org/abs/2206.14651
- StrongSORT — IEEE TMM 2023 — https://arxiv.org/abs/2202.13514 ; StrongSORT++ — https://arxiv.org/abs/2210.15975
- Deep OC-SORT — ICIP 2023 — https://arxiv.org/abs/2302.11813
- Hybrid-SORT — AAAI 2024 — https://arxiv.org/abs/2308.00783
- GHOST — CVPR 2023 — https://arxiv.org/abs/2206.04656
- C-BIoU — WACV 2023 — https://arxiv.org/abs/2211.14317
- FastTracker — 2025 — https://arxiv.org/abs/2508.14370 (code: https://github.com/Hamidreza-Hashempoor/FastTracker)
- TrackTrack — CVPR 2025 — https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf
- Deep HM-SORT — 2024 — https://arxiv.org/abs/2406.12081
- MOTR — ECCV 2022 — https://arxiv.org/abs/2105.03247 ; MOTRv2 — CVPR 2023 — https://arxiv.org/abs/2303.15780 ; MOTRv3 — https://arxiv.org/abs/2305.13383
- MeMOT — CVPR 2022 — https://arxiv.org/abs/2203.16761 ; MeMOTR — ICCV 2023 — https://arxiv.org/abs/2307.15700
- TrackFormer — CVPR 2022 — https://arxiv.org/abs/2101.02702 ; TransTrack — https://arxiv.org/abs/2012.15460 ; SiamMOT — CVPR 2021 — https://arxiv.org/abs/2105.11595
- MambaTrack — ACM MM 2024 — https://arxiv.org/abs/2408.09178
- SambaMOTR — ICLR 2025 Spotlight — https://arxiv.org/abs/2410.01806 — https://sambamotr.github.io/ — https://github.com/mattiasegu/sambamotr
- MOTIP — CVPR 2025 — https://arxiv.org/abs/2403.16848 — https://github.com/MCG-NJU/MOTIP (model zoo: https://github.com/MCG-NJU/MOTIP/blob/main/docs/MODEL_ZOO.md)
- SUSHI — ICCV 2023 — https://arxiv.org/abs/2308.12714 (⚠ not re-verified)
- uMOT — https://arxiv.org/abs/2309.00942

Datasets / benchmarks

- DanceTrack — CVPR 2022 — https://arxiv.org/abs/2111.14690 — https://dancetrack.github.io/ — leaderboard https://www.codabench.org/competitions/14885/
- SportsMOT — ICCV 2023 — https://arxiv.org/abs/2304.05170
- MOT17 — https://arxiv.org/abs/1603.00831 ; MOT20 — https://arxiv.org/abs/2003.09003
- BDD100K — CVPR 2020 — https://arxiv.org/abs/1805.04687 ; KITTI tracking — http://www.cvlibs.net/datasets/kitti/eval_tracking.php ; VisDrone — https://arxiv.org/abs/2001.06303 (⚠ latter two not used for numbers)

Tooling / secondary

- Ultralytics tracking docs (main branch) — https://raw.githubusercontent.com/ultralytics/ultralytics/main/docs/en/modes/track.md
- forasoft MOT survey (secondary, 2026) — https://www.forasoft.com/blog/article/multi-object-tracking-algorithms
- Roboflow `trackers` comparison (referenced, not fetched) — https://trackers.roboflow.com/latest/trackers/comparison
