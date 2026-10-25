# tg-track — click-to-track a person, DJI-ActiveTrack style

Detect **every** person in a frame, let an operator click one, and then follow **that**
person and only that person — through occlusion, partial visibility, leaving the frame,
and fast or nonlinear motion.

The interesting part of this problem is not "track a bounding box". It is **not
switching to somebody else**. A consumer tracker that silently locks onto a
similar-looking bystander is worse than one that admits it is lost, because the
operator cannot see the difference until the drone is following a stranger. This
repository is built around that failure mode: refusing to match is a first-class
outcome, and every claim below is backed by a script and a number.

---

## TL;DR — the six results that matter

| # | Finding | Evidence |
|---|---|---|
| 1 | **Part descriptors survive occlusion; a whole-body descriptor does not.** Identity Rank-1 on real occluded MOT17 crops: `full` 0.672 → `quality_w` **0.855** (+27%) — and on the hardest mask the whole-body margin goes *negative* while the part-based margin stays positive. | `scripts/exp_partial_view.py` |
| 2 | **The detector, not the tracker, was the bottleneck.** `yolo26m@1280` finds **2.0×** the people `yolo26n@640` does (recall 0.368 → 0.733, 5 sequences) for +28% frame time, because this GPU is preprocessing-bound, not network-bound. | `scripts/probe_recall.py --confirm` |
| 3 | Switching the detector default moved all-person HOTA **0.359 → 0.483** and DetA **0.264 → 0.442** on real MOT17. | `scripts/eval_mot.py` |
| 4 | **ReID in the all-person layer is not worth it**: +0.005 HOTA for **3.6×** the frame time. It is an on/off switch, not a default. | `scripts/eval_mot.py --embed` |
| 5 | An IMM motion model beats a plain Kalman filter by **25%** on curved paths and **61%** under constant acceleration. | `scripts/ablate_motion.py` |
| 6 | **Two of our own claims died under their controls, and a third was broken, re-swept, and then rejected on real data.** (a) A grid that said part descriptors *hurt* was silently tracking a 38×54 px target that no detection ever touched — the real re-run points the other way (0.672 → 0.855). (b) The tracker's one remaining hole — a lone similar bystander taking the identity over in a single frame — is isolated and measured rather than hidden. (c) The guard written to close that hole had two bugs (it was skipped exactly when the tracker goes blind, and it fired on unreadable evidence); fixed and re-swept, it cuts lone-candidate hijacks from 3/4 runs to 1/4 — then real MOT17 (3.6 pp of upper-body TSR lost, zero false-acquisition gained) turned it back OFF, so the shipped default is None and the negative result is kept. | `scripts/pick_observable_target.py`, `results/target_obs19/`, `results/motion_ablation/lone_bound.md` |

---

## The six requirements, and where each is met

| Requirement (user's own wording, condensed) | Where it is satisfied |
|---|---|
| 1. Ultralytics **YOLO26** person detection + ReID, `n` variants | `src/tgtrack/detector.py` (`yolo26n/s/m` benchmarked), `src/tgtrack/appearance.py` (`yolo26n-reid.onnx`, 512-d). Default is `yolo26m@1280` on measurement — the user explicitly welcomed better parameter-matched models. |
| 2. Person-tracking datasets, diverse, **total < 10 GB** | 3 datasets, **6.51 GB** on disk: MOT17 (level crowd), DanceTrack (uniform appearance, nonlinear), VisDrone (aerial/UAV). See *Datasets*. |
| 3. **Survey** of tracking strategies, last decade / last 5 years | `docs/research/R1_mot_association.md` (MOT association), `R2_reid_partial_occlusion.md` (ReID + partial/occluded), `R3_single_target_uav.md` (single-object, long-term, UAV industry), `R4_datasets.md`. 2,400+ lines, source-linked, with explicit UNVERIFIED markers. |
| 4. Detect all persons; click-track one; long+short-term ReID; partial view; out-of-frame; fast and nonlinear motion | `src/tgtrack/` — `detector.py` → `mot_layer.py` (all-person IDs) + `target_tracker.py` (the clicked person). Long-term = `memory.LongTermGallery`, short-term = `memory.ShortTermMemory`, partial view = `parts.py` + part-affinity in `association.py`, out-of-frame = `StateCfg` + coasting, motion = `motion.py` (IMM/NSA). |
| 5. Experiment environment, **parallel**, **no fine-tuning** | `scripts/eval_mot.py`, `run_target.py`, `ablate_motion.py`, `exp_partial_view.py` — all process-pooled, all strategy-only. **No weights were trained or fine-tuned anywhere in this repository.** |
| 6. Subagents allowed | Yes — used for the four surveys and for several implementation tasks. |

---

## Architecture

Three layers, deliberately decoupled:

```
frame ──▶ PersonDetector (YOLO26)          all persons, every frame
             │
             ├──▶ AllPersonTracker           per-person IDs — "other people need only an ID"
             │      (ultralytics backend: botsort / tracktrack / ...)
             │
             └──▶ TargetTracker              the clicked person — the actual product
                    ├── ShortTermMemory      EMA appearance + ring buffer
                    ├── LongTermGallery      quality/diversity-weighted part banks
                    ├── MotionModel          IMM (cv/ca/ct) + NSA adaptive noise
                    └── Association          motion · appearance · direction · size, with a margin gate
```

**Why not an end-to-end SOT network** (OSTrack / MixFormer / ODTrack)? Because they are
trained to answer "*the same object*", not "*the same person*", and they cannot express
"the target is absent" — they always emit a box. R3 established this; they are only
useful as optional box refiners. The decision-maker here is an explicit cost with an
**explicit refusal**.

**The anti-latch mechanism.** A candidate is accepted only if

```
S(c*) > tau_accept        AND        S(c*) − S(c2) > tau_margin
```

The second term is the important one. It says: *if somebody else in the frame scores
nearly as well, do not commit.* Without it, a similar-looking bystander wins by a hair
and the tracker walks off with them. Measured effect in the ablation grid.

**Refusal has teeth.** The target state machine
(`INIT → TRACKING → SOFT_OCCLUSION → LOST_SHORT → LOST_LONG → OUT_OF_FRAME → GAVE_UP`,
plus `REACQUIRING`) gates every appearance write: while the tracker is blind, the
gallery is **frozen**. This is not a heuristic — it is the best-supported decision in
the long-term-tracking literature (arXiv:1908.01603, *Model Decay in Long-Term
Tracking*: adverse-box updates "add an unavoidable bias term … can eventually lead to a
permanent loss of the target"). A tracker that learns from the occluder becomes
confidently wrong.

**Two-stage association.** Geometry gates cheaply first; the ReID encoder only runs on
candidates that survive. Encoding every detection every frame would cost more than the
detector.

---

## Environment

Verified on this machine:

- Windows, PowerShell. **Python 3.12.7**, `C:\Software\Anaconda3`
- **NVIDIA RTX 4060 Laptop, 8 GB VRAM**, sm_89
- **torch 2.7.0+cu126 / torchvision 0.22.0+cu126** — the cu126 index is required; plain
  PyPI gives you a CPU build. `cuda_available True`, verified with a 2048³ matmul.
- **ultralytics 8.4.185**, **onnxruntime-gpu 1.20.2**, lap 0.5.13, onnx 1.23.2

Two traps, both solved and both documented in `docs/BRIEF.md`:

1. `onnxruntime` (CPU) and `onnxruntime-gpu` **cannot coexist** — same module name, the
   CUDA build gets shadowed and you silently run on CPU. Also the CUDA provider is a
   plugin DLL linking `cublasLt64_12.dll` / `cudnn64_9.dll`, which ship *inside
   `torch/lib`* and are not on `PATH`. `src/tgtrack/runtime.py::ensure_ort_cuda()` fixes
   this; **call it before constructing an InferenceSession** or you are timing CPU
   (2.74 ms vs 4.25 ms at batch 1, ~8× at batch 40).
2. Network: `huggingface.co`, `github.com` and `drive.google.com` are unreachable here.
   `hf-mirror.com` works, and `https://ghproxy.net/https://github.com/...` proxies GitHub.

```powershell
$env:PYTHONIOENCODING="utf-8"     # console is GBK; Unicode output crashes without this
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pytest -q               # 300+ tests
```

---

## Datasets

Three datasets, chosen for **diversity of failure mode**, not for leaderboard appeal:

| dataset | what it adds | used for experiments |
|---|---|---|
| **MOT17** (train, FRCNN) | level-view crowd, the canonical benchmark, per-box visibility labels | all-person MOT; all partial-view and visibility work |
| **DanceTrack** (val) | near-identical appearance + nonlinear motion + long occlusion — the hard case for ID | all-person MOT on a second distribution, and the **target layer's hardest test**: with `mask=none` the clicked-person FAR is **0.493** (§4b) |
| **VisDrone2019-MOT** (val) | **aerial/UAV**, looking down at people — the actual product viewpoint | all-person MOT on a third camera geometry — 14 pairs, tracktrack HOTA 0.3800, but **0.0000 on the two 4K sequences** (§3c: this is where our system meets its limit) |

**Size: 6.51 GB on disk** (MOT17 851 MB + DanceTrack 4,091 MB + VisDrone 1,565 MB),
against a 10 GB budget. The raw archives are 10.07 GB decimal / 9.38 GiB — the
budget's unit is ambiguous, so the report quotes the deliberately-used set, which is
unambiguous on either reading.

Two facts that shaped the experiment design:

- **DanceTrack's `visibility` column is the constant `1`** across all 225,148 val rows.
  It carries no occlusion signal at all, so every visibility-dependent ablation runs on
  MOT17 only. (Finding this early prevented a whole class of silently-meaningless
  experiments.)
- **Neither dataset has ground truth while a target is absent**, so re-acquisition
  latency cannot be scored from either. That is a protocol gap, not a data gap: the
  harness masks a real target's box for N frames and uses its known reappearance as GT
  (`scripts/run_target.py --mask`), at zero download cost.

VisDrone ships a different annotation format, so `scripts/convert_visdrone.py` converts
it to MOTChallenge (hardlinking images, not copying them) — 7 sequences, 2,846 frames,
49,848 person GT rows.

---

## Survey — what the literature actually says

Four reports in `docs/research/`, ~2,400 lines, every claim source-linked, unverifiable
claims explicitly marked `UNVERIFIED` rather than smoothed over. What changed our design:

**Adopted**

- **Observation-centric correction** (OC-SORT): +7.3 HOTA over ByteTrack *with identical
  detections* (54.6 vs 47.3, DanceTrack). Matching the last *observation* rather than the
  Kalman prediction beats better appearance models.
- **HMIoU** (Hybrid-SORT): the only built-in geometry cue that survives **vertical
  truncation** — directly relevant to partial visibility.
- **Rollback on occlusion** (FastTracker): coverage > 0.7 → restore position from 3
  frames back, velocity from 5 back, `h*1.1`, `v*0.5`.
- **Freeze the gallery while lost** (Model Decay, arXiv:1908.01603).
- **The margin gate**, which R3 identified as "the actual anti-hijack mechanism" — our
  `cost.distractor_margin`.
- **NSA adaptive noise** (SORT-family): scale process noise by detection confidence.

**Rejected, with reasons**

- **End-to-end SOT nets as the decision-maker.** Trained for "same object", cannot say
  "absent". Useful only as refiners.
- **Face recognition as a primary cue.** Measured: a full-height person at detector
  `imgsz=640` yields ~85 px face and **~20 px inter-eye**, below the 30–60 px commercial
  requirement. AdaFace does beat ArcFace on surveillance (IJB-S 65.26 vs 57.35) but that
  gap is irrelevant when the pixels are not there. Adoptable only as a gated auxiliary vote.
- **Heavier ReID backbones** (TransReID, CLIP-ReID, SOLIDER/KPR). 30–80× the compute to
  close a *backbone* gap, when our measured gap is a *mechanism* gap — TransReID scores
  95.2 Rank-1 on Market-1501 and collapses to 64.2 on Occluded-Duke.
- **`ltm_replace_policy="quality_recency"` as though it were validated.** It has no
  published backing; FIFO is the field standard. Kept as a config knob to A/B, not as a
  default claim.

**Industry reality check.** DJI's own Air 3S manual says that on loss the aircraft "*will
keep flying at the current speed and orientation to try re-identifying the subject*" —
commercial re-acquisition moves the *platform* as the search policy rather than running a
global re-detector — and admits the subject "*may be inadvertently swapped to another
subject if they pass nearby each other*". That swap is exactly the failure this
repository's margin gate targets. Skydio's advantage is 6-camera obstacle avoidance in
the control loop, not a better tracker.

---

## Experiments

All numbers below are reproducible with the command shown; nothing is quoted from a
paper as if we had measured it.

### 1. Detector recall — the bottleneck nobody had measured

`python scripts/probe_recall.py --confirm` — pooled over 5 MOT17 sequences, 2,658 GT boxes
with visibility ≥ 0.25, greedy 1-1 matching at IoU 0.5:

| setting | pooled recall | det/frame | ms/frame | FPS |
|---|---|---|---|---|
| `yolo26n@640` | 0.368 (978/2658) | 7.6 | 33.5 | 29.9 |
| `yolo26m@1280` | **0.733** (1947/2658) | 19.1 | 42.9 | 23.3 |

Per sequence: 02 0.348→0.721, 04 0.317→0.822, **05 0.858→0.858**, 11 0.537→0.578,
13 0.297→0.662. The easy sequence is unchanged and the hard ones double — the gain is
exactly where it is needed. Latency is nearly flat across `n/s/m` because letterbox +
NMS + host transfer dominate, so this is a **2× recall improvement for +28% time**.

This single change explains the all-person layer's previously poor DetA. The tracker
was being blamed for the detector's misses.

### 2. All-person MOT on real MOT17

`python scripts/eval_mot.py --data-root data/raw/mot17/MOT17/train --out results/mot_eval_m1280 --jobs 4 --subsample 3`
— 7 sequences × 6 trackers = 42 pairs, 0 failures, 294.6 s wall vs 1156.0 s serial
(**3.92× at `--jobs 4`**).

| tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | ms/frame |
|---|---|---|---|---|---|---|---|---|
| **tracktrack.yaml** | **0.4827** | 0.4415 | **0.5322** | 0.8272 | 0.4329 | **0.5860** | **346** | 105.4 |
| botsort.yaml | 0.4728 | 0.4402 | 0.5129 | 0.8287 | **0.4334** | 0.5649 | 431 | 101.9 |
| deepocsort.yaml | 0.4706 | 0.4391 | 0.5093 | 0.8284 | 0.4304 | 0.5613 | 500 | 110.1 |
| ocsort.yaml | 0.4425 | 0.4218 | 0.4697 | 0.8313 | 0.3949 | 0.5129 | 1148 | 74.1 |
| bytetrack.yaml | 0.4421 | 0.4201 | 0.4706 | 0.8324 | 0.3965 | 0.5092 | 1114 | 70.3 |
| fasttrack.yaml | 0.4325 | 0.4206 | 0.4499 | 0.8324 | 0.3938 | 0.4953 | 1210 | 72.8 |

With the previous default (`yolo26n@640`) the same harness gave tracktrack
**HOTA 0.3589 / DetA 0.2644 / MOTA 0.2853 / IDF1 0.4179**. So the detector change alone
is worth **+0.124 HOTA, +0.178 DetA, +0.148 MOTA**.

`tracktrack` (CVPR 2025, HMIoU + track-aware init + relaxed lost-rebind) wins on
accuracy and — importantly — has **half the ID switches** of ByteTrack/OC-SORT. Speed
runs the other way: ByteTrack is the fastest at 70 ms/frame, OC-SORT 74 ms.

**Not leaderboard-comparable**: `--subsample 3`, and GT rows with `conf=0` (the official
ignore regions) are dropped rather than handled as ignores. The ordering is what we
learned from; the absolute values are not a claim of state-of-the-art MOT.

### 3. Is ReID worth it in the all-person layer? No.

`... --embed` (same 42 pairs). ReID is only consumed by the backends that implement it —
`eval_mot.py` warns loudly for the rest, which is how the last three rows below are
explained:

| tracker | HOTA no-ReID | HOTA +ReID | Δ | IDF1 | IDSW | ms/frame | slowdown |
|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 0.4827 | 0.4880 | **+0.0053** | 0.5860 → 0.5944 | 346 → **299** | 105.4 → **377.0** | **3.6×** |
| botsort.yaml | 0.4728 | 0.4815 | +0.0087 | 0.5649 → 0.5803 | 431 → 481 | 101.9 → 382.6 | 3.8× |
| deepocsort.yaml | 0.4706 | 0.4701 | −0.0005 | 0.5613 → 0.5614 | 500 → 525 | 110.1 → 394.1 | 3.6× |
| ocsort.yaml | 0.4425 | 0.4425 | 0.0000 | unchanged | unchanged | 74.1 → 345.8 | **4.7×, pure waste** |
| bytetrack.yaml | 0.4421 | 0.4421 | 0.0000 | unchanged | unchanged | 70.3 → 329.9 | 4.7×, pure waste |
| fasttrack.yaml | 0.4325 | 0.4325 | 0.0000 | unchanged | unchanged | 72.8 → 339.5 | 4.7×, pure waste |

**+0.005 HOTA for 3.6× the frame time** on the layer where it helps at all, and on the
three trackers that ignore appearance features it is **4.7× slower for exactly zero
change**. R1 predicted this ("ReID is an on/off switch, not a default"); now it is
measured on our own data. ReID earns its cost in the *target* layer, where identity is
the whole point — not in the layer that only needs other people to keep distinct IDs.

### 3b. The same grid on real DanceTrack — 25 sequences, 150 pairs

MOT17 is a crowd benchmark with real clothing; DanceTrack is the opposite stressor:
uniform appearance, nonlinear motion, heavy occlusion. Same command shape, `--no-embed`:

```
python scripts/eval_mot.py --data-root data/raw/dancetrack/val --trackers \
  tracktrack.yaml botsort.yaml bytetrack.yaml deepocsort.yaml ocsort.yaml fasttrack.yaml \
  --subsample 2 --out results/dancetrack_mot --jobs 2 --no-embed
```

150 pairs, **0 failed**, 25 sequences, wall 2531.6 s vs 5051.9 s serial = 2.00× at `--jobs 2`.

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | ms/frame |
|---|---|---|---|---|---|---|---|
| tracktrack.yaml | **0.4514** | 0.7087 | **0.2897** | 0.7737 | **0.4280** | 2520 | 67.01 |
| botsort.yaml | 0.4273 | 0.7064 | 0.2602 | 0.7762 | 0.4042 | 2911 | 65.76 |
| bytetrack.yaml | 0.4173 | 0.7063 | 0.2485 | 0.7714 | 0.3949 | 3175 | 42.75 |
| deepocsort.yaml | 0.4086 | 0.7046 | 0.2387 | 0.7683 | 0.3803 | 3600 | 67.51 |
| ocsort.yaml | 0.4084 | 0.7018 | 0.2393 | 0.7689 | 0.3855 | 3517 | 43.91 |
| fasttrack.yaml | 0.4047 | 0.7056 | 0.2337 | 0.7663 | 0.3834 | 3791 | 42.09 |

The spread is the point. **DetA is flat across all six backends (0.7 pp) and MOTA within
1.0 pp, while AssA spans 5.6 pp and HOTA 4.7 pp.** On DanceTrack the detector is not the
variable — association is, which is precisely the layer the target tracker replaces.

One trap in reading across the two datasets: DanceTrack's MOTA (0.77) is *higher* than
MOT17's (0.43) while its HOTA is *lower*. That is not the tracker doing better; DanceTrack
has almost no ignore regions, whereas MOT17's MOTA is depressed by the official `conf=0`
rows that we drop instead of handling as ignores. **HOTA and AssA are comparable across the
two datasets here, MOTA is not.**

### 3c. The same grid on VisDrone — aerial, and the detector breaks first

```
python scripts/eval_mot.py --data-root data/sequences/visdrone \
  --trackers tracktrack.yaml,bytetrack.yaml --subsample 3 --out results/visdrone_mot --jobs 1 --no-embed
```

14 pairs, **0 failed**, 7 sequences, wall 232.0 s.

| tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | MT | ML | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | **0.3800** | 0.3055 | **0.4761** | 0.7651 | 0.2415 | **0.4937** | 152 | 73 | 137 | 102.92 |
| bytetrack.yaml | 0.3009 | 0.2639 | 0.3455 | 0.7735 | 0.2281 | 0.3542 | 604 | 39 | 172 | 57.49 |

The ordering survives the domain change (tracktrack +6.9 pp HOTA and **4× fewer ID
switches**), but the per-sequence spread is the real finding: **0.4590, 0.4356, 0.4294,
0.2429, 0.1449, 0.0000, 0.0000**. The two zeros are exactly the two highest-resolution
sequences — `uav0000268_05773_v` at 3840×2160 and `uav0000305_00000_v` at 1904×1071 — and
`DetA 0.3055` against `AssA 0.4761` says the same thing from the other direction: **on aerial
footage at `imgsz=1280` the detector finds almost nobody, so association quality stops
mattering.** At 4K a person is roughly 15 px tall. No strategy layer can recover a detection
that never fired, which is why §1's recall numbers, not the tracker, are the binding
constraint on the drone use case.

### 4. Partial view — the headline mechanism

`python scripts/exp_partial_view.py --source data/raw/mot17/MOT17/train/MOT17-02-FRCNN/img1 --out results/partial_view_real --max-frames 80 --subsample 2`
poses it as pure identity retrieval: no tracking, no training, no GT. **Real MOT17
person crops** (25 gallery identities, 414 query crops) are labelled by their MOT track
id, split into gallery/query, and each query is re-encoded under a synthetic occlusion
mask. The tracker is **not told** which mask was applied — visibility is estimated from
image structure energy alone, exactly as the live tracker does.

Mean identity Rank-1 over the seven occluded masks (whole-body on clean crops = 0.903):

| strategy | occluded Rank-1 | vs. `full` |
|---|---|---|
| `full` (whole-body descriptor only — the naive baseline) | 0.672 | — |
| `max_part` | 0.803 | +19% |
| `mean_parts` | 0.809 | +20% |
| `visible_gate` (only still-structured parts) | 0.840 | +25% |
| **`quality_w`** (structure-weighted) | **0.855** | **+27%** |

Per mask, the ordering holds where it is hardest:

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | 0.903 | 0.983 | 0.928 | **0.983** | 0.983 |
| lower | 0.792 | 0.932 | 0.884 | **0.957** | 0.923 |
| upper | 0.681 | 0.807 | 0.845 | **0.870** | 0.855 |
| lower60 | 0.669 | 0.889 | 0.920 | **0.942** | 0.930 |
| upper60 | 0.483 | 0.524 | 0.606 | **0.674** | 0.664 |
| left | 0.710 | 0.833 | 0.742 | **0.831** | 0.831 |
| right | 0.696 | 0.867 | 0.819 | **0.860** | 0.836 |

**The margin is the part that matters.** The refusal gate consumes
`S(correct) − S(best wrong)`, and on the hardest mask the whole-body descriptor's margin
goes **negative** — the correct identity is outranked — while the part-based strategies
stay positive:

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| upper60 | **−0.0044** | +0.0032 | +0.0302 | **+0.0383** | +0.0390 |
| lower | +0.0762 | +0.1358 | +0.1573 | **+0.1664** | +0.1592 |

So parts do not merely rank better; they give the anti-latch test real headroom on
exactly the frames where a whole-body descriptor would confidently latch onto the wrong
person.

Control — is the `head` descriptor actually informative, or does it return the same
vector for everybody? Same-identity vs different-identity mean cosine on clean crops
(n=414):

| part | same-id cos | diff-id cos | **gap** |
|---|---|---|---|
| full | +0.6664 | +0.4632 | +0.2032 |
| upper | +0.6402 | +0.4249 | **+0.2153** |
| lower | +0.6073 | +0.4088 | +0.1985 |
| head | +0.5860 | +0.4097 | +0.1763 |

Every part carries real identity evidence. `head` has the smallest gap but it is clearly
positive (+0.176), so it is informative rather than degenerate — an earlier 3-identity
synthetic run had suggested its gap was the *largest*, which is exactly why the control
had to be re-run on real data before being believed.

The visibility estimator itself is validated by the same run: painting out the lower body
drops measured `lower` activity from 0.650 to **0.069**, painting the upper body drops
`upper` from 0.693 to **0.111**, and a half-height mask lands in between (0.490).

#### The tracking-side experiment, and a harness trap worth recording

The retrieval table above is the clean measurement. The *tracking* version of the same
claim — mask the target in the video and watch the tracker hold on — is much easier to
get wrong, and our first attempt got it wrong in an instructive way.

`scripts/run_target.py` picks the target identity from track statistics alone and then
rewrites that target's ground truth to whatever survives the mask. Nothing checked that
the survivor was still *detectable*. On MOT17-02 with `--max-frames 60 --subsample 2
--mask upper` it picked id 17, whose post-mask visible box is **38×54 px**, and no
detection overlapped it on **any** of the 60 frames (best IoU 0.000, with 17–20
detections per frame). That run was measuring an *invisible* target, not a partial one,
and its `TSR = 0.2333` says nothing about part-based ReID. `--target-strategy longest`
picks the same identity, so the flag cannot dodge it. `scripts/pick_observable_target.py`
ranks identities by post-mask visible size and shows why: under `mask=upper` the largest
are id 19 (304×190 = 55,480 px²), id 3 (227×160) and id 23 (248×134), while id 17 is
54×38 = 2,071 px² — and the harness was landing on the smallest.

**Pointed at a target that is still observable (id 19), the same grid finally
discriminates.** 11 variants × 3 masks = 33 runs, 0 errors, real `yolo26m@1280` + real
ReID, `results/target_obs19/`:

| variant | mask=none | mask=upper | mask=lower |
|---|---|---|---|
| **baseline** (shipped defaults) | **0.9236** | **0.7527** | **0.9236** |
| `gate_legacy` (no early global search) | 0.7891 | 0.6036 | 0.9236 |
| `gate_search_late` | 0.7891 | 0.6036 | 0.9236 |
| `gate_growth_unfrozen` | 0.9236 | 0.7527 | 0.9236 |
| `margin_0.00` | 0.9236 | 0.6945 | 0.9236 |
| `margin_0.12` | 0.9236 | 0.7236 | 0.9236 |
| `parts_full_only` | 0.9200 | 0.6691 | 0.7745 |
| `part_matching_off` | 0.9236 | 0.7527 | 0.9236 |

TSR alone overstates that. The same runs record how often the tracker *reports a box it
has no observation for* (`hallucination_rate`) and how often the reported box is simply
wrong (`wrong_box_rate`) — the inertia a success metric cannot see:

| variant | mask | TSR | hallucination | wrong box | distinct IDs followed |
|---|---|---|---|---|---|
| baseline | none | 0.9236 | **0.1467** | 0.0764 | 2 |
| baseline | upper | 0.7527 | **0.3100** | 0.2473 | 1 |
| baseline | lower | 0.9236 | 0.1433 | 0.0764 | 5 |
| `parts_full_only` | upper | 0.6691 | 0.3867 | 0.3309 | 1 |
| `parts_full_only` | lower | 0.7745 | 0.2733 | 0.2255 | 6 |
| `margin_0.00` | upper | 0.6945 | 0.3533 | 0.3055 | 4 |
| `gate_legacy` | none | 0.7891 | 0.2733 | 0.2109 | 2 |

So roughly **15% of the baseline's "correct" frames with no mask, and 31% with the upper
body masked, are coasting rather than measurement.** Dropping the part banks raises wrong
boxes from 0.2473 to 0.3309 under `mask=upper` and from 0.0764 to 0.2255 under
`mask=lower` — the part mechanism showing up in a second, independent metric instead of
only in TSR.

Five things this settles:

1. **The early-search gate fix is worth +13.5 to +14.9 pp TSR.** Disabling it
   (`gate_legacy`, i.e. `state.global_states=()`) costs 13.45 pp with no mask and
   14.91 pp with the upper body masked. `gate_growth_unfrozen` equals the baseline on all
   three masks, so it is the *earlier search*, not the frozen-speed growth term, that
   carries the win — reported as such rather than crediting both.
2. **Part banks earn their cost.** Removing them (`parts_full_only`) costs 8.36 pp under
   `mask=upper` and 14.91 pp under `mask=lower`, with more ID switches (8 vs 6) and
   fragments (8 vs 3) on `mask=lower`, while costing only 0.36 pp when nothing is
   occluded. This **reverses** an earlier synthetic result in which `parts_full_only`
   *won*; that run used a GT-oracle detector and the fallback descriptor against the
   invisible target above, and is not published as evidence.
3. **`distractor_margin = 0.06` is right on real data.** 0.00 raises false acquisitions
   (FAR 0.0100 vs 0.0067 with no mask; 0.0100 vs 0.0000 under `mask=upper`), and 0.12
   costs 2.9 pp TSR under `mask=upper` for no FAR gain. The synthetic grid had suggested
   0.12 was far better; on an observable target it is not.
4. **`part_matching_off` changes nothing** — identical to the baseline to four decimals
   on all three masks — while `target_parts` clearly does. So the gallery's part *banks*
   do the work and the per-frame cost-side part routing is inert here. The reason is that
   `estimate_partial_state` only recognises truncation by the *frame edge*; a person whose
   upper body is blacked out mid-frame still looks like a small complete person to the
   detector, so a partial view is never declared. That is a concrete improvement
   direction: derive the partial state from the crop's own structure energy, which the
   tracker already computes for the visibility estimate.
5. **A success metric is not evidence of tracking.** The baseline's 0.9236 with no mask
   coexists with a 14.7% hallucination rate; under `mask=upper` the TSR is 0.7527 while
   **31%** of frames carry no observation at all. The honest headline for the masked runs
   is therefore "holds the identity for 0.75–0.92 of frames and is actually observing the
   target for about 70% of them", not "0.92 accuracy". Two blind spots are mitigated in
   the harness rather than hidden: `evaluate_target` counts a coasting frame as correct, and
   its `false_acquisition` requires IoU ≥ 0.5 against another identity, so a confident
   low-IoU lock on a stranger is invisible to it. task-10 added `wrong_identity_locked`
   (confident lock on another identity at IoU ≥ 0.1) and `coasting_correct_frames`, both
   reported as their own columns.

`mask=upper` is much harder than `mask=lower` (0.7527 vs 0.9236). Part of that is tracker
loss and part is detector recall on a legs-only crop; this run cannot separate them.

### 4b. DanceTrack — where the target layer actually hurts

```
python scripts/run_target.py --data-root data/raw/dancetrack/val --limit-sequences 3 --max-frames 200 `
  --subsample 2 --mask none,lower,upper --ablate gate,distractor_margin,target_parts,part_matching `
  --target-strategy stressed --out results/dancetrack_target --jobs 2
```

99 runs (11 variants × 3 masks × 3 sequences), **0 errors**, wall 2,694 s at `--jobs 2`.

DanceTrack is uniform appearance + nonlinear motion, and it breaks the target layer in a way
MOT17 never did:

| variant | mask | TSR | prec | FAR | FA events | id_consistency |
|---|---|---|---|---|---|---|
| baseline (`margin 0.06`) | none | 0.2228 | 0.2266 | **0.49333** | 13 | **0.3035** |
| `margin_0.12` | none | 0.2529 | 0.2524 | 0.47333 | 12 | 0.3395 |
| `gate_search_late` | none | 0.1910 | 0.1901 | 0.39333 | 9 | 0.3179 |
| `gate_legacy` | none | 0.1759 | 0.1720 | 0.44667 | 6 | 0.2737 |
| baseline | lower | 0.3132 | 0.2791 | 0.10500 | 27 | 0.7470 |
| `margin_0.12` | lower | **0.3869** | 0.3331 | **0.02000** | 10 | **0.9506** |
| baseline | upper | **0.5729** | 0.4803 | 0.00833 | 5 | 0.9856 |
| `margin_0.12` | upper | 0.5243 | 0.4437 | 0.00500 | 3 | 0.9905 |
| `parts_full_only` | upper | 0.4288 | 0.3883 | 0.03167 | 9 | 0.9309 |

Read the columns, not the rows: `--target-strategy stressed` picks a *different* identity per
mask, so the three mask columns are three different targets and only variants **within** a
column are comparable.

Four things this settles:

1. **On DanceTrack the target layer fails about half the time.** With `mask=none`, FAR is
   **0.493** and `id_consistency` is **0.304** — the clicked person is the one being followed
   in under a third of frames. This is exactly what the survey predicted (appearance is nearly
   useless when everyone looks alike), and it is the strongest statement in this document that
   requirement 4 is **met on MOT17 and only partly met here**.
2. **The margin default is dataset-sensitive, and DanceTrack wants it stricter.**
   `margin 0.12` beats 0.06 on two of three masks — on `lower` it is +7.4 pp TSR, **−8.5 pp FAR**
   and **+20.4 pp id_consistency** — and it never loses on FAR anywhere (MOT17 included). What it
   does lose is TSR when only the *upper* body is visible (−4.9 pp here, −2.9 pp on MOT17), because
   a torso-and-head crop genuinely resembles other people's torsos and heads. We ship **0.06**
   and record the disagreement, because that lost case is precisely requirement 4's scenario; the
   knob is one line of config.
3. **The occlusion-search fix is validated on real nonlinear data.** Reverting it
   (`gate_search_late`, i.e. arming global search only after `lost_long`) costs TSR on all three
   masks (−3.2 / −0.2 / −11.6 pp) and quadruples FAR on `upper` (0.00833 → 0.03500), and
   `gate_growth_unfrozen` (using the coast-damped speed in the gate-growth term instead of the
   speed frozen at occlusion onset) is worse on FAR on two of three. Both shipped defaults hold.
4. **The part banks earn their place here — the opposite of the MOT17 grid, and the reason that
   grid was wrong.** `parts_full_only` costs TSR on every mask and **−14.4 pp on `upper`**
   (0.5729 → 0.4288, id_consistency 0.9856 → 0.9309): drop the part banks and the one case where
   the target is a head and a torso is exactly the case that collapses. The partial-view mechanism
   reproduces on a second, harder, real dataset.

One defect is visible in both datasets and worth naming: **`cost.part_matching` is inert.**
`part_matching_off` reproduces `baseline` *exactly* on all six (dataset × mask) columns, because
`parts.estimate_partial_state` only detects truncation by the **frame edge** and therefore never
fires for a rectangle painted inside the frame. The mechanism works anyway — through part-affinity
scoring in `association.py` — but the intended "notice the partial view, then switch bank" path
never activates, and any future work should start there.

### 5. Motion model

`python scripts/ablate_motion.py` — one-step-ahead centre error, px, 160-frame
trajectories:

| trajectory | Kalman (kf) | **IMM** | gain |
|---|---|---|---|
| circle r=40, v=4 | 4.085 | **3.062** | −25.0% |
| circle r=80, v=8 | 8.169 | **6.145** | −24.8% |
| figure-8 r=60, v=5 | 9.214 | **8.335** | −9.5% |
| constant accel 0.5 px/f² | 6.121 | **2.365** | **−61.4%** |

Worst 40-frame IMM/KF segment ratio is 0.785, so the gain is not an artefact of
averaging over oscillation. `nsa_kf` equals `kf` at confidence 1.0 by design.

---

## Design decisions and why

**The default detector is large, and that is deliberate.** `yolo26m@1280` is not the
"obvious" choice for a project that started by asking for YOLO26-*n*. It is the choice
the measurement demanded: 2× recall for 28% more time, and requirement (4) says *detect
all persons*. `yolo26n@640` remains one `Config().clone(...)` away for fast sweeps.

**Embeddings are stamped.** Features from the same network at a different input
resolution are **not** commensurable (arXiv 2403.10022), so `EmbedderCfg.spec` returns
e.g. `yolo26n-reid.onnx@imgsz448@v2` and a persisted gallery must record it and refuse a
mismatch. This is the kind of bug that silently degrades a system for months.

**448, not 224.** The shipped ONNX declares `imgsz=[448,448]` — it was *trained* at 448 —
and ultralytics' wrapper only honours that metadata when the ONNX dims are static ints.
Ours are fully dynamic, so the wrapper silently ran at 224 and threw away half the
training resolution. On the retrieval task the resolution pair alone is clean **0.655 → 0.780**
and occluded **0.921 → 0.985** (the earlier 0.886/1.000 and 0.780/0.989 pair spliced the part-aware
strategy axis into this resolution claim; the strategy gain is measured separately, see E013).

**ReID is encoded only for the target and gated candidates.** At 448, encoding 20 people
× 5 parts every frame is ~107 ms/frame — infeasible. Target-centric encoding is a
requirement, not an optimisation.

**Refusal is a feature.** See *Architecture*. The whole state machine exists to decide
when *not* to update.

---

## The detector swap: capacity was never the constraint, training distribution was

Under the user's later allowance (third-party pretrained weights, no fine-tuning by us, comparable
parameter count) the general-purpose detector was replaced with a **CrowdHuman person-specialist**
(`crowdhuman_yolov8n_best.pt`, **3.01 M parameters** vs yolo26m's 21.90 M). On the identical E003
protocol it raises pooled MOT17 recall **0.731 -> 0.803 (+7.2 pp)** with **7.3x fewer parameters** and
**5.2x lower latency**; at 1536 px it is *worse* (0.790), repeating that more pixels is not the lever.

On the real MOT17 HOTA protocol (`results/mot17_ch_det/`, 7 sequences):

| metric | yolo26m | crowdhuman yolov8n | delta |
|---|---|---|---|
| HOTA | 0.4980 | **0.5050** | +0.70 pp |
| DetA | 0.4609 | 0.4829 | +2.20 pp |
| AssA | 0.5431 | 0.5341 | -0.90 pp |
| MOTA | 0.4950 | **0.5586** | **+6.36 pp** |
| IDF1 | 0.6125 | **0.6531** | **+4.06 pp** |
| ms/frame | 49.68 | 41.88 | -15.7 % |

Read this as: recall was the binding constraint on MOTA/IDF1, and it bought +6.4/+4.1 pp there, while
HOTA only gains 0.7 pp because the extra detections arrive with +428 FP and +56 IDSW. Lowering the
MOT layer's entry thresholds for the richer stream (E048) and sweeping confidence (E049) both
**lost** HOTA, so the defaults were left alone. The swapped detector is documented as a measured
option, not yet the shipped default, because criteria ② and ③ have not been re-validated with it.

## The held-out measurement, and what it says about everything above

All seven shipped target-layer knobs were selected **and** reported on the same three DanceTrack
sequences (`0004/0005/0007`), which the method audit flagged as the campaign's most serious defect.
The first held-out measurement — the **22 DanceTrack-val sequences outside the frozen dev split**,
shipped defaults, no ablation — is in `results/heldout_dancetrack/` and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) E068:

| quantity | dev split (n=3) | **held-out (n=22)** | 95% cluster bootstrap CI |
|---|---|---|---|
| target success rate | 0.512 | **0.2104** | **[0.1157, 0.3254]** |
| false acquisition rate | 0.280 | **0.2472** | [0.1489, 0.3487] |

**The dev figure does not generalise.** 0.512 lies outside the held-out CI. The FAR is statistically
indistinguishable, so on unseen sequences the system's problem is **losing the target**, not following
the wrong person. The spread is bimodal (median TSR 0.153; 7 of 22 sequences below 0.05, 2 above 0.80).

Consequences that are now stated plainly: criterion ② is **0.21 [0.12, 0.33] over 22 sequences**, not
0.512; the dev-split intervals were ±0.394 (TSR) and ±0.708 (FAR), so those three sequences never
certified anything; and E022/E029/E031b/E038/E041 are **unvalidated** on held-out data.

## Corrections after the first independent audit (round A)

An independent audit re-derived 24 numeric claims in this file from the `results/` artifacts and found
**6 wrong, 4 stale, 2 unsupported**. The ones that change a conclusion are corrected here instead of
being silently edited, so the record shows what was believed and why it did not hold.

- **Criterion ③ (partial view, +10 pp) is NOT met.** The "+10.18 pp" figure compared runs from two
  different epochs. The controlled same-run pair (`results/e031b_mot17`, identical command, baseline
  vs shipped weights) is **0.7818 -> 0.8545 = +7.27 pp**, below the >= +10 pp bar. The endpoint
  (0.8545) and the mechanism are real; the attribution of +10.18 pp is not. **Criterion ③ is
  therefore open at +7.27 pp of a required +10 pp.**
- **Criterion ②'s baseline is target-dependent** (see the next section): 0.512/0.280 was measured with
  `--target-id 2` forced on `dancetrack0005`; today's automatic pick yields 0.4138/0.3833. Both are
  now quoted together with their target id, and the criterion is scored on the target-2 basis.
- **The lone-bound cost of 3.6 pp is retracted**: `results/target_obs19_shipped` contains no armed
  variant, so 0.7164 is that grid's own baseline, not a measurement of the bound.
- **The lone-bound sweep reaches 1/4 hijacks at <= 0.4, not 0/4.**
- **The 448 resolution pair is 0.655 -> 0.780 clean and 0.921 -> 0.985 occluded**, not the spliced
  0.886/1.000 and 0.780/0.989, which mixed the resolution axis with the part-aware strategy axis.
- **The VisDrone lower-mask cell is 0.9236**, not a duplicate of the upper cell.
- **E020's "74/111" ceiling evidence is retracted** (E062): that probe read the wrong GT identity. With
  the correct mapping the true target was detected in **108 of the 113** wrong frames, which is the
  opposite kind of failure — detection is fine, identity assignment is what breaks.
- Four upper-mask TSR values appear in this file (0.7527 / 0.7164 / 0.7818 / 0.8545). They are **not**
  the same benchmark: the first two predate the E031b partial-weight shipping. Compare 0.7818 against
  0.8545; treat 0.7527/0.7164 as historical.

## Two criteria we did not reach, and the measurements that say why

Criterion ② asked for DanceTrack TSR >= 0.55 with FAR < 0.20. The best figure is **0.512 / 0.280**,
measured with `--target-id 2` forced on `dancetrack0005` (E041). **The benchmark is not stable across
runs:** under today's automatic target pick the harness selects id 0 for that sequence and the pooled
figure becomes **0.4138 / 0.3833** (E063b), so the two numbers are different people, not a regression.
Every comparison in this file now names its target id, and the criterion is scored on the target-2
basis until that is fixed in the harness.
The gap is not reachable by strategy: four independent signals were measured and each landed at or
below chance — an added motion axis (0.000 joint movement, E032b), pose skeleton structure
(**0.432**, E045), and the inter-person neighbourhood computed from GT boxes (**0.333**, median margin
-0.0035, E045b). The earlier "appearance ranked the true target >= 2 in 74/111 wrong frames" evidence
(E020) was **RETRACTED in E062**: that probe read the wrong GT identity, and with the correct mapping
the true target was detected in **108 of the 113** wrong frames.
Criterion ① asked for MOT17 HOTA 0.55; the detector caps DetA at 0.4609, and every route to more
recall was measured and rejected (tiling E001, threshold E002, capacity E003, temporal promotion
E004/E005, stitching E007, gating E010, ensemble E011, Hybrid-SORT E012, box fusion E055). Both
ceilings are documented in `docs/EXPERIMENTS.md` with reproduction commands.

## Honest limitations

- **AGPL-3.0.** Both the YOLO26 detector weights and `yolo26*-reid.onnx` are AGPL-3.0
  (the licence is in the ONNX metadata). A networked product built on these triggers
  AGPL source obligations. Apache-2.0 alternatives exist (OpenCV Zoo
  `person_reid_youtu_2021nov.onnx`, RTMPose, SAM 2.1) but were not benchmarked here.
- **Not SOTA MOT.** `--subsample 3`, and we drop rather than ignore `conf=0` GT regions.
  The numbers are internally consistent and reproducible; they are not leaderboard
  entries. Published DanceTrack HOTA for reference: ByteTrack 47.3, OC-SORT 54.6,
  Hybrid-SORT 62.2, MOTIP 69.6.
- **The all-person layer is 9–14 FPS on this GPU** at `yolo26m@1280`. The target layer is
  cheap; the detector dominates. Real-time needs a smaller model or a smaller `imgsz`,
  and the recall table lets you price that trade explicitly.
- **Requirement 4 is met on MOT17 and only partly met on DanceTrack.** With `mask=none` there,
  the clicked-person false-acquisition rate is **0.493** and `id_consistency` is **0.304**
  (§4b): when everybody looks alike, the anti-latch machinery does not save us. What does help
  on that dataset is the part banks (+14.4 pp TSR on the `upper` mask) and a **stricter** margin
  (0.12: −8.5 pp FAR and +20.4 pp id_consistency on `lower`) — both reported, neither hidden.
- **Aerial footage fails before tracking does.** On VisDrone the tracker ranks correctly
  (tracktrack HOTA 0.3800 vs bytetrack 0.3009, 4× fewer ID switches) but **two of seven
  sequences score HOTA 0.0000** — the two at 3840×2160 and 1904×1071 — because at
  `imgsz=1280` the detector finds almost nobody in a 4K frame; `DetA` (0.306) trails `AssA`
  (0.476) across the whole dataset. At that scale a person is ~15 px tall. The fix for the
  drone use case is a tiled/sliced detector, **which we have not built**, not a better
  association strategy.
- **The margin threshold needs calibration.** No paper measures false-acquisition rate,
  so `distractor_margin` cannot be copied from the literature. It must be tuned against
  our own logs — and it should arguably be *relative* rather than an absolute fused-cost
  gap.
- **Face cues do not work at this resolution** and are not implemented.
- **A similar-looking bystander can take the identity over in a single frame.**
  motion-dev's `unidentifiable_target` family isolates the mechanism: the target walks,
  halts, and stops being *readable* while still detected and still present in the ground
  truth. The handover happens on the **first** frame the target is unreadable, and it
  happens while the state is still `tracking` with `frames_since_update = 0` — so no
  blind state is entered, global search never arms, and the `search_needs_appearance`
  guard is never consulted. With a bystander at appearance similarity 0.40–0.85 the median
  *accepted* margin is **0.88–1.11**, i.e. 15× `distractor_margin`, so no margin threshold
  can refuse it: the runner-up is the feature-less *true* target, which
  `missing_appearance_cost` prices *below* a clean-descriptor stranger. Pooled over 30
  runs that is **39.2 % target success / 54.6 % false acquisition** with the bystander,
  against **58.7 % / 0.0 %** in the solo control. The lever written for this hole is
  `CostCfg.lone_candidate_accept_cost`, and it took two fixes before it worked at all: it
  had been **exempted while lost** — skipped exactly when the tracker goes blind, which is
  where every handover happens — and it also fired when the candidate's appearance was
  unreadable, so a transient encoder failure killed tracking instead of protecting it. Both
  are fixed: the bound now applies in both directions and only to readable evidence.
  Re-swept over 100 runs with a genuine off-control (`results/motion_ablation/lone_bound.md`),
  the lone-candidate hijack count goes **3/4 runs → 1/4 at 0.5 and → 1/4 at ≤ 0.4**, and every
  surviving handover is a frame-36 *priced* comparison instead of a single-frame takeover. It
  still does not close the hole: peak false acquisition only moves 76.9 % → 75.6 %, because the
  aggregate is dominated by that priced comparison; on the synthetic clean families the price is
  **2.6 pp of success at 0.5** (98.9 % → 96.3 %) and 5.4 pp at 0.4. **Real MOT17 then decided the
  default against enabling it.** On the observability-guarded grid (`results/target_obs19_shipped`
  vs `results/target_obs19` — identical command, MOT17-02, target id 19, 300 frames, subsample 2)
  the comparison originally quoted here — **3.6 pp of target success on the upper-body-only mask
  (0.7527 → 0.7164) — is RETRACTED**: `results/target_obs19_shipped` contains no armed variant, so
  0.7164 is that grid's own baseline and the two figures are not a controlled pair. What is actually
  established is weaker: the armed rule has a measured synthetic cost (**2.6 pp** of clean-family
  success at 0.5) and **no controlled real-data benefit has ever been demonstrated**. On that
  evidence the shipped default stays `None`; the knob, the
  100-run sweep and the negative result are all kept. `CostCfg.missing_appearance_cost` is the
  remaining lever: once the lone rule is armed its low side (≤ 0.25) closes the handover
  completely on a 30-frame stub world (TSR 0.43 → 0.97 / FAR 0.73 → 0.23), but with the rule off
  price only moves the *recovery*, never the handover itself — both facts are pinned by tests.
  This is the sharpest remaining gap against requirement 4 and it is stated, not hidden.
- **The end-to-end path is strongly target-size dependent, and on a small person it fails.**
  `scripts/demo_click.py` on MOT17-02 clicking the most persistent identity in the sequence
  (id 9, **36 × 110 px**, visible in 300/300 GT frames) reports
  `target_success_rate` **0.06**, `wrong_box_rate` **0.843**,
  `false_acquisition_rate_locked` **0.357**, `id_consistency` 0.419, `target_switch_count` 22
  — it follows the wrong person for most of the clip. It never gives up (`reacquire_rate`
  1.0, latency 29 frames) and it *knows* it is unsure (`hallucination_rate` 0.841), but
  "unsure most of the time" is not the product promise. The same harness on a large,
  clearly-visible person reports TSR **0.964** (id 2 — but that identity is only GT-visible
  in 28 of 300 frames, so treat 0.964 as a 28-frame sample) and **0.924** in the
  observability-guarded grid (`results/target_obs19`, id 19, 607 × 190 px, 300 frames).
  A person barely larger than the ReID crop cannot be pinned down by a 512-d whole-body
  descriptor; this is the concrete outer boundary of requirement 4.
  Metric note: `target_precision` divides by *all* frames including the ones where the
  target is absent (run 1 reads 0.079 while `target_precision_visible` is 0.851) — read the
  `_visible` variant.
- **`reacquiring` counts as a confident lock** in the target metrics. That is deliberate:
  the state is reached only after a candidate has been *accepted*, so it is a tentative
  identity claim rather than a blind coast (the blind bucket is derived from
  `StateCfg.frozen_states`). A wrong claim there must therefore show up in
  `false_acquisition_rate_locked`, and it does.
- **`false_acquisition` needs IoU ≥ 0.5 with another ground-truth identity**, so a wrong
  lock on a person it only half-overlaps is not counted (the live id-17 trace sat at IoU
  0.224). `evaluate_target` also reports `correct=True` for coasting frames, which is
  success by inertia, not by observation. Read TSR together with the per-frame CSV.
- **DanceTrack visibility labels are useless** (constant 1), so cross-dataset
  partial-view validation is limited to MOT17 + VisDrone.
- **Some survey claims are marked UNVERIFIED** in the reports — chiefly DJI patent
  *claims* (patent sites were unreachable; only snippet-level evidence) and a few
  conference numbers that exist only inside PDFs. They are labelled, not laundered.

---

## Repository map

```
src/tgtrack/
  types.py           the contract: Detection, TargetStatus, part ranges, IoU/cosine helpers
  config.py          all knobs, dataclasses, dotted-path clone(); embed-spec stamping
  runtime.py         ensure_ort_cuda() — call before any InferenceSession
  parts.py           part crops, visibility + structure estimation, partial-state inference
  appearance.py      Embedder (ONNX ReID, batched, part-aware)
  memory.py          ShortTermMemory (EMA) + LongTermGallery (quality/diversity/part banks)
  motion.py          Kalman / NSA-Kalman / IMM (cv, ca, ct), rollback, adaptive gate
  association.py     cost terms, part affinity, margin gate, candidate selection
  target_tracker.py  the clicked-target state machine — the core deliverable
  detector.py        YOLO26 person detector, multi-model benchmark
  mot_layer.py       all-person tracking over the ultralytics backends
  data.py            MOTChallenge / YOLO / VisDrone loaders, click simulation
  evaluate.py        HOTA/MOTA/IDF1 + target-centric metrics, from scratch (NumPy)
scripts/
  demo_click.py        the product: detect all, click one, follow it, score it
  eval_mot.py          parallel all-person MOT evaluation over real sequences
  run_target.py        clicked-target strategy-ablation grid, incl. masking
  ablate_motion.py     motion-model ablation through TargetTracker as a black box
  exp_partial_view.py  the partial-view retrieval experiment
  convert_visdrone.py  VisDrone -> MOTChallenge
  run_video.py         single-video detections + annotated mp4
docs/
  BRIEF.md             the shared context file every agent read first
  research/R1..R4      the four surveys
```

## Reproducing

```powershell
$env:PYTHONIOENCODING="utf-8"      # the console is GBK; unicode in a log will crash without this
python -m pytest -q                # full suite, no GPU needed

# datasets (one-time; 5.59 GB + 4.01 GB, inside the 10 GB budget)
curl.exe -L -C - -o data/raw/MOT17.zip https://motchallenge.net/data/MOT17.zip
curl.exe -L -C - -o data/raw/dancetrack_val.zip https://hf-mirror.com/datasets/noahcao/dancetrack/resolve/main/val.zip
python scripts/convert_visdrone.py     # VisDrone -> MOT layout (hardlinks images, no extra disk)

# 1. detector recall, and 4. partial view as identity retrieval
python scripts/probe_recall.py --confirm
python scripts/exp_partial_view.py --source data/raw/mot17/MOT17/train/MOT17-02-FRCNN/img1 `
    --out results/partial_view_real --max-frames 80 --subsample 2

# the read-only diagnostics that the claims in sections 1 and 4 rest on
python scripts/pick_observable_target.py --max-frames 300 --subsample 2   # which identities stay observable after masking
python scripts/probe_mask_detect.py --max-frames 30 --subsample 1         # does the mask remove the person from the DETECTOR too?
python scripts/probe_why_refuse.py --mask upper --frames 22               # accept/refuse reason, frame by frame

# 2 / 3. all-person MOT on all three datasets
python scripts/eval_mot.py --data-root data/raw/mot17/MOT17/train --subsample 3 `
    --trackers tracktrack.yaml,botsort.yaml,bytetrack.yaml,deepocsort.yaml,ocsort.yaml,fasttrack.yaml `
    --out results/mot_eval_m1280 --jobs 4
python scripts/eval_mot.py --data-root data/raw/dancetrack/val --subsample 2 `
    --trackers tracktrack.yaml,botsort.yaml,bytetrack.yaml,deepocsort.yaml,ocsort.yaml,fasttrack.yaml `
    --out results/dancetrack_mot --jobs 2 --no-embed
python scripts/eval_mot.py --data-root data/sequences/visdrone --trackers tracktrack.yaml,bytetrack.yaml `
    --subsample 3 --out results/visdrone_mot --jobs 1 --no-embed

# the target layer: observability guard, ablations, the trap in section 4
python scripts/run_target.py --data-root data/raw/mot17/MOT17/train --target-id 19 `
    --limit-sequences 1 --max-frames 300 --subsample 2 --mask none,upper,lower `
    --ablate gate,distractor_margin,target_parts,part_matching --out results/target_obs19 --jobs 2
python scripts/run_target.py --data-root data/raw/mot17/MOT17/train --dry-run --ablate all `
    --mask none,lower,upper,left,right,edge --out results/target_eval_guard

# 5. motion model, the identity-handover family, and the pricing sweep
python scripts/ablate_motion.py
python scripts/ablate_motion.py --unid
python scripts/ablate_motion.py --unid-cost

# the product: click one person and follow only them
python scripts/demo_click.py --source data/raw/mot17/MOT17/train/MOT17-02-FRCNN `
    --save-video --out results/demo
# ... or pin the click, which is what the limitation bullet above was measured with
python scripts/demo_click.py --source data/raw/mot17/MOT17/train/MOT17-02-FRCNN `
    --click 1074,539 --frame 0 --max-frames 300 --subsample 2 `
    --save-video --out results/demo_real_id9
```

Four things worth knowing before you re-run any of it:

- **`--target-id` is not optional for a masked run.** Without it `run_target.py` now picks
  an identity and *refuses to start* if none of them is observable (`exit 3`), because the
  first version silently selected a 38×54 px target that no detection ever touched.
- **Read `TSR` next to `hallucination_rate` and `wrong_box_rate`.** A coasting frame counts
  as success, and `false_acquisition` needs IoU ≥ 0.5 against another identity, so a
  confident low-IoU lock on a stranger is invisible to it.
- **On this 8 GB card, keep the total concurrent ONNX batch under ~64 per worker.** Four
  workers at `EmbedderCfg.max_batch=128` died inside the first Conv
  (`BFCArena::AllocateRawInternal ... 269746176`, which is 128×3×448×448×4 B for one
  activation). The default is now 64, and `Embedder.encode` halves-and-retries on an
  allocation failure rather than killing the run.
- **`--jobs` reports its own honesty**: `eval_mot.py` prints the measured speedup next to the
  serial time (3.92× at `--jobs 4`, 2.00× at `--jobs 2`), and per-worker GPU memory.
- **`demo_click.py --click X,Y` scores against the person you actually meant**: the point is
  resolved to the smallest detection containing it, then matched to a ground-truth identity.
  The `auto (largest)` default picks whoever is biggest at frame 0, which on MOT17-02 is an
  identity that is only GT-visible in 28 of 300 frames — cheap to score well on, and not a
  measurement of anything.

Every performance claim in this document comes with the command that produced it and
the measured number. Where a number is quoted from the literature rather than measured
here, it says so.

## Research ledger and methodology

The strategy layer was built by an experiment loop, and both halves of that loop are committed
next to the code:

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) -- the ledger: 36 numbered experiments, each with
  hypothesis, exact command, measured table, verdict, and the reason for the verdict. It includes
  the rejected branches (`E021` drift monitor, `E023` pinned anchors, `E024` gallery hygiene,
  `E025` adaptive forgetting, `E026` frozen-anchor drift, `E027`/`E028` identity ceiling,
  `E030` capped band, `E032`/`E032b` motion band) so the knowledge survives without the code.
- [docs/METHOD.md](docs/METHOD.md) -- how experiments are run here so the numbers can be
  believed: epoch discipline, the fresh-baseline and nondiscrimination checks, controls, the
  five verdicts, the shipping rules, and a catalogue of the nine ways this programme actually
  got a wrong answer (silent no-op patches, dropped CLI flags, off-by-one annotation frames,
  metric sign inversion, fusion normalisation, unexercised mechanisms...).

The single most reusable finding is in METHOD.md section 7: three different distance bands
(appearance, capped appearance, motion) and their conjunction all failed to separate "the target
changed" from "this is someone else", because on the hard sequence the impostor is *closer to the
target's own past* than the target is. Any further gain needs an independent identity signal.
