# R_direction — exploration-direction review (round A, after E063)

Reviewer: exploration-direction subagent, dispatched 2026-10-25 by the round-A gate
(`docs/reviews/README.md:35-43`). Question: which evidence axes are exhausted, which are open,
which 5 experiments to run next, and what to stop.

Method note: **no experiment was run by this review.** Every number below is either quoted from
`docs/EXPERIMENTS.md` (E-id) or derived read-only from an existing artifact under `results/`
(paths given). One analysis is new: the swap structure of the 0007 baseline debug CSV
(`results/e059_diag7/runs/dancetrack0007__baseline__none.csv`), reported in §2.5.

---

## 1. Three measurement-integrity facts that constrain every ranking below

These are not style points; each one invalidates a number that the programme currently quotes.

### 1.1 The criterion-2 benchmark silently changed target on 0005

| artifact | `meta.target_ids` | `target_decisions.reason` | baseline TSR / FAR |
|---|---|---|---|
| `results/e041_dancetrack0005/summary.json` | `{dancetrack0005: 2}` | `forced` | **0.695 / 0.205** |
| `results/e063b_dt0005/summary.json` | `{dancetrack0005: 0}` | `observable` | **0.400 / 0.515** |
| `results/e041_dancetrack0007` vs `results/e063b_dt0007` | `{...: 1}` both | forced → observable | 0.4264 / 0.595 (unchanged) |
| `results/e041_dancetrack0004` vs `results/e063b_dt0004` | `{...: 3}` both | forced → observable | 0.415 / 0.040 (unchanged) |

So the ledger's "shipped 0.512 / 0.280" (E041, and repeated in the position notes after E049–E063)
and the current "0.4138 / 0.3833" (E063 guard, and the criterion-2 baseline in the campaign brief)
**are two different benchmarks**, differing only in which person is target in 0005. The 9.8 pp TSR /
10.3 pp FAR gap between them is not a regression, an epoch effect, or a mechanism effect — it is a
target-id re-pick that happened when `--target-id` stopped being passed (`target_decisions.reason`
went `forced` → `observable`). E063's pooled "+1.2 pp TSR / −1.7 pp FAR" therefore averages one
sequence's target-A score with two other sequences' scores.

Consequence: E063's guard verdict ("keep `audit_interval = 0`") is *probably* still right, but it was
decided on a mixed benchmark, and the campaign brief's criterion-2 baseline (0005 = 0.400/0.515) is
**the harder** of the two targets. Any of the five experiments below that reports a pooled criterion-2
number must first re-state it on one frozen target set (§6).

### 1.2 E053's "appearance does not buy association" was measured with the mechanism provably off

E053 recorded `tracktrack + with_reid` as **bit-identical** to no-ReID (0.5050 / 0.5586 / 0.6531) and
then concluded "(b) appearance does not buy association on this stream". The bit-identity is not a
null result, it is proof that no appearance term entered the cost matrix:

- `configs/trk_reid.yaml` sets `with_reid: true, model: auto, reid_weight: 0.5`.
- `ultralytics/trackers/utils/reid.py:139-161` — `build_encoder(with_reid, model)` with
  `model == "auto"` returns `_auto_encoder`, a **pass-through for `feats`**: it does not load a model.
- `ultralytics/trackers/track_tracker.py:438` — `cost = np.where(np.isnan(cos), hmiou_dist, …)`, so
  when `feats is None` the cosine block is NaN and the cost collapses to HMIoU.
- `scripts/eval_mot.py:598` — `embedder = emb_factory(cfg, opts) if opts.embed else None`, and
  `scripts/eval_mot.py:627-632` only build `feats` when the embedder exists. E047's protocol (which
  E048 and E053 reused) ran with `--no-embed`.

Nothing in the ledger ever ran the all-person tracker with features actually supplied. The claim
"appearance does not buy association" is a fourth victim of the `docs/METHOD.md` §6 "silent flag
drop" pattern, self-documented in the entry that states it.

### 1.3 The two tracker layers never talk — verified in code, not inferred

`src/tgtrack/target_tracker.py:61-66` imports `association`, `config`, `memory`, `motion`, `parts`,
`types`; it does **not** import `mot_layer`, and `scripts/run_target.py` never constructs an
`AllPersonTracker` (`grep -n AllPersonTracker scripts/run_target.py` → no match). The single-person
layer therefore re-derives identity from raw detections + one embedding, and the all-person layer's
track ids (`mot_layer.apply_track_ids`, `mot_layer.py:94`) are never used as evidence anywhere in the
product. This is the largest structural gap in the programme and §4.3 is built on it.

---

## 2. Axis map — what is closed, and the strongest measurement that closed it

Legend: **CLOSED** = further work on this axis needs a new measurement, not a new knob;
**OPEN** = under-tested or untested.

| # | axis | status | strongest closing measurement |
|---|---|---|---|
| 2.1 | detection recall / tiling / threshold / ensemble / resolution | CLOSED | E055: fusing a 0.8303-recall union (E054) into the CH stream gives **HOTA 0.4944**, worse than CH alone 0.5050, at 89.2 vs 41.9 ms/f |
| 2.2 | detector combination (add boxes / replace geometry) | CLOSED | E056: geometry snapping **0.4966** vs 0.5050; snapping only fires where CH was already right |
| 2.3 | appearance distance-to-own-past as the decision rule | CLOSED *under the one descriptor ever tested* | E032b: motion ∧ appearance conjunction is **bit-identical** (never fires) and the 0007 impostor is *closer* to that past (0.07–0.17) than the target (0.22–0.30) |
| 2.4 | motion bands / velocity gates / Mahalanobis band | CLOSED | E032: Mahalanobis band pooled **0.307 / 0.135** (0004 0.255 → 0.050); E016's velocity clamp buys +2.91 pp on partial-view and −0.36 pp elsewhere |
| 2.5 | tube / trail heading (analytic motion signature) | CLOSED | E059: 0004 **0.4150 → 0.0950** (w=0.5) and 0.0800 (w=1,2); 0007 bit-identical because search-only terms never fire |
| 2.6 | gallery policy (STM/LTM, caps, half-life, anchor set) | CLOSED | E043: `ltm_only` 0.417/0.352, `stm_only` 0.224/0.428 vs shipped 0.512/0.280 — the shipped mix is a knife-edge local optimum; E042 caps 6/8/12 plateau |
| 2.7 | bank guards (hard negatives, admission gates) | CLOSED | E036: `hard_negative_off` **+35 pp TSR / −12.5 pp FAR on 0004** and −52.5 pp on 0005, i.e. the bank helps and hurts with no policy that separates the two; E037's confident-only gate is bit-identical (never fires) |
| 2.8 | section / partial-view snapping, part weights | CLOSED (and criterion 3 is MET on it) | E031: MOT17 upper **0.7818 → 0.8545** (+10.18 pp over the 0.7527 baseline); E034/E035 buy more on MOT17 (0.2/2.4 → 0.2684) and lose DanceTrack |
| 2.9 | post-hoc association repair (eviction, stitching) | CLOSED | E007: 80 merges over 630 fragments → **+0.0005 HOTA**; E006: 0–20 of 20,951 boxes evicted |
| 2.10 | all-person backend / motion-model geometry | CLOSED | E012: purpose-built hybridsort **0.4831** with IDSW 475 vs tracktrack 0.4980 / 299 at 1.8× the latency |
| 2.11 | **appearance association in the all-person layer** | **OPEN** | E053's null is an inert-flag artifact (§1.2); no run in the ledger supplies features |
| 2.12 | **identity by tracklet membership (global association)** | **OPEN** | E060 claimed it and was retracted (E062), which removed the *attribution*, not the mechanism; nothing measures tracklet purity on DanceTrack |
| 2.13 | **co-visibility / neighbourhood structure** | **OPEN, under-tested** | E045b: one 39-frame probe, win rate 0.333, median margin −0.0035 — with a descriptor whose content differs between the click frame and later frames (see §3.2) |
| 2.14 | **cross-domain descriptor transfer** | **OPEN** | E046 proved the doctrine on detection (+7.2 pp recall from a 3.01 M-param specialist) and it was never applied to ReID |
| 2.15 | **temporal cadence / frame-gap scaling of the motion machinery** | **OPEN** | nothing in the ledger varies the evaluation's frame gap; all motion constants are expressed in processed frames |
| 2.16 | active re-observation at ambiguity (plan H8) | **OPEN** | never tested; E035 tested tiling for *recall* (rejected), not for *discriminative margin* |

### 2.5 New read-only measurement: the 0007 swaps are spatially smooth and long-lived

From `results/e059_diag7/runs/dancetrack0007__baseline__none.csv` (200 rows, shipped config,
target id 1; 116 rows `correct=False`, of which 113 are scored — E062):

| frames | followed GT id | n | wrong |
|---|---|---|---|
| 0–58 | 1 (target) | 59 | 0 |
| 59–102 | 3 | 44 | **41** |
| 103–156 | 6 | 54 | **54** |
| 162–181 | 1 (target) | 20 | 0 |
| 183–199 | 3 | 17 | **17** |

Swap onsets with the centre displacement normalised by body height: f59 **0.19**, f103 **0.143**,
f162 0.096 (recovery, correct), f183 0.104. The largest is 0.19 body heights.

Two consequences, both load-bearing for the ranking:

1. **Any "implausible jump / teleport" gate is dead on arrival.** The tracker does not teleport onto
   the wrong person; it drifts onto an adjacent dancer over 1–3 frames with sub-0.2-height motion,
   and then holds a *coherent* wrong track for 44 and 54 frames. Consistently, E062 measures the
   followed box at 0.766× the target's area while the target is still detected at median IoU 0.9336.
2. **The state that must be discriminated is "a stable wrong tube", not "a noisy frame".** That is
   why appearance margins are thin (E063: `audit_margin` 0.05 fires, 0.10 is inert) and why an
   evidence source that is *categorical* rather than *metric* would be worth more than a better
   threshold: a binary "this box belongs to a different tracklet than the one I am on" has no margin
   to tune.

---

## 3. What is open, and which open axis has the highest expected value

### 3.1 The shape of the problem after E062/E063

Criterion 1 needs HOTA 0.4827 → 0.55. HOTA = sqrt(DetA · AssA) and the best current pair is
DetA 0.4829 / AssA 0.5341 (E047, CH@1280). 0.55² = 0.3025, so:

- with AssA frozen at 0.5341, DetA must reach **0.5664** (+8.35 pp);
- with DetA frozen at 0.4829, AssA must reach **0.6263** (+9.22 pp).

Neither single-term target has ever been approached: the best detection move in the campaign (E046/E047)
bought +2.2 pp DetA and **cost** 0.9 pp AssA. So criterion 1 is not "one more detector" away — it is
an association problem with a detection problem attached. Every detector-shaped experiment (2.1, 2.2,
2.10) moved DetA up and HOTA down. The one association-shaped experiment that was run (E053) measured
nothing (§1.2).

Criterion 2 needs, on the current (harder) target set, TSR 0.4138 → 0.55 and FAR 0.3833 → <0.20.
E062 localises the loss precisely: 113/113 wrong frames are in `tracking`, 108 have the true target
detected at IoU ≥ 0.5 (median 0.9336). This is a *choice* failure with the answer in the detection
list, not a sensing failure.

### 3.2 Both remaining deficits are "give the tracker a discriminator with margin", and the
orthogonal-evidence axis is under-tested rather than refuted

The brief asks specifically for identity evidence that is **not** distance-to-own-past and **not**
detection recall. Two candidate families in the ledger claim to have closed that, and neither claim
survives inspection:

**(a) E045b (neighbourhood structure) is one biased probe, not an axis closure.** The probe
(`scripts/probe_neighbour_identity.py`) builds the click descriptor with the impostor *inside* the
neighbour set (`desc(T, others)`) and every later descriptor with the impostor *excluded*
(`rest_t = [b for b in others if b is not imp]`), so the target's own descriptor content changes
between the reference frame and the test frames by construction; k=4 slots are filled by
distance-sorted order, so slot *j* is not the same neighbour across frames; there is one click frame,
39 sampled frames (STEP=5 over 200), no confidence interval on a median margin of −0.0035; and the
"impostor" is the GT person spatially nearest the target, which E062 shows is **not** the person the
tracker actually follows (the followed box is 0.766× area, adjacent not nearest). A 39-frame probe with
an asymmetric descriptor and the wrong adversarial pair cannot close an axis.

**(b) E045/E013 (pose) failed, but for a reason that leaves a different pose test open.** E013's ROC-AUC
for `pose_full` is 0.7247 (MOT17-02) but 0.4778 / 0.3250 on DanceTrack 0004/0005 — below chance. That
is a statement about *one* keypoint descriptor (box-height-normalised joints, no limb symmetry, no
viewpoint handling) evaluated on *matched* crops. It is not a statement about limb geometry as identity
evidence in the regime where appearance fails. And E045's own median detection IoU of 0.893 on the
target says the crops were good — the descriptor was not.

**(c) E032b's null is descriptor-relative and nobody has re-parameterised the descriptor.** "The
impostor is closer to the target's past than the target is" was measured with `yolo26n-reid.onnx`.
E046's doctrine — *capacity was never the constraint, training distribution was* — was worth +7.2 pp
recall on detection and has never been applied to ReID. Under `docs/METHOD.md` §6, a null from one
instrument is not a null.

**Verdict on the open axis with highest EV.** Two families:

1. **Structural identity: tracklet membership.** The programme owns a working all-person tracker
   (HOTA 0.5050, IDF1 0.6531, 41.9 ms/f) whose track-ids it never gives the target layer (§1.3). A
   swap like the f103 one (id 1 → id 6, jump 0.143 heights, held 54 frames) is, to a global tracker,
   a *different object*; "accept only detections belonging to my tracklet" is categorical, has no
   margin to tune, and has **zero latency** — which matters because E063 proved the swap timescale is
   1–2 frames and that audit latency was the difference between +20.3 pp and inert. This is the only
   evidence source in the project that is neither a distance to the target's past nor a property of
   the detector.
2. **Instrument quality: cross-domain descriptor transfer + effective resolution of the evidence.**
   Cheapest decisive test in the whole list (embeddings only, no tracking loop), and it gates the
   entire appearance family (16 experiments already spent on it).

Highest EV *per GPU-minute* is neither: it is the unrun control on E053 (§1.2), because 5 GPU-min
either recovers 9 pp of AssA or finally closes criterion 1's association term with an engaged control.
I rank by that, and say so explicitly below.

---

## 4. The five ranked experiments

Costs are GPU-minutes on the local RTX 4060, derived from measured artefact timings:
target-layer runs are **≈1.27 s per processed frame** (E063: 636.87 s wall for 5 variants × 100
processed frames), all-person runs are **41.9–49.7 ms per frame** (E047) and ≈3× that with the
embedder engaged.

### 4.1 (1) `C1-ASSOC-ENGAGED` — run the appearance-association test that E053 did not run

- **Mechanism.** Supply real `yolo26n-reid` features to the CH-stream all-person tracker and sweep the
  appearance weight. `scripts/eval_mot.py` already builds `feats` whenever `--embed` is on and
  `src/tgtrack/mot_layer.py:301-327` already forwards them; `configs/trk_reid.yaml` already sets
  `with_reid: true, reid_weight: 0.5`. Nothing has to be implemented — only run without `--no-embed`,
  and *assert the term engaged* (count non-NaN cosine entries in `_multi_cue_cost`, or log the number
  of detections that arrived with a feature row).
- **Why the ledger does not refute it.** E053's null is bit-identical, i.e. produced by a cost matrix
  with no appearance block (§1.2). E012 is not a counter-argument: it also ran feature-free. The
  relevant prior is positive — E017 measured the descriptor separating *matched* crops at AUC 0.9654
  (MOT17-02) / 0.7499 (dance 0005) with win rate 0.996, so the descriptor carries identity at exactly
  the crops the all-person layer would compare.
- **Cost.** ≈5 GPU-min (2 configs × 7 seqs × 67 frames at ≈150 ms/f, plus one 7-seq control pass).
- **Expected delta.** Criterion 1. Plausible band: AssA 0.5341 → 0.55–0.58 (HOTA 0.5050 → 0.51–0.53);
  the large outcome (AssA 0.6263) is what 0.55 actually requires and would be a surprise, but the
  *sign* of the appearance term on this stream has never been measured, and E047 lost 0.9 pp AssA to
  CH's extra FPs (E055 attributes it to precision), so even a modest recovery is worth having.
- **Cheapest falsification.** Two sequences (MOT17-02, -04), one variant, with the fired-mechanism
  assertion. Kill the axis if **AssA moves < +0.5 pp while the assertion shows >0 features were
  actually used**. That is a real null, and it closes criterion 1's association term for good.
- **Secondary product.** It re-derives the AssA cost of CH's FPs directly, which is the number the
  "detector-capped" story has been resting on since E047 without a controlled read.

### 4.2 (2) `C2-DISC-EXIST` — does a discriminator exist against the boxes the tracker actually follows?

- **Mechanism.** Build the adversarial set from `results/e059_diag7/runs/dancetrack0007__baseline__none.csv`:
  for each of the 113 wrong frames, the true target's crop (from `gt_xyxy`) and the crop of the box
  the tracker actually reported (`pred_xyxy`). Then measure, per candidate descriptor, the AUC
  separating those two sets, plus the per-frame margin (target-to-click-anchor distance minus
  rival-to-anchor distance). Descriptors: `yolo26n-reid` (control), a person-pretrained ReID swap
  (E046 doctrine — `m04393` allows third-party pretrained weights at comparable parameter count), and
  a 2× upscaled crop arm (does the evidence improve with more pixels, which E001 never asked because
  it measured *recall*).
- **Why the ledger does not refute it.** Every appearance number in the ledger was measured either on
  *matched GT crops* (E013, E017 — which is why they look good) or on a *hypothetical nearest-neighbour
  impostor* (E045/E045b), never on the real followed impostor. E062 is the first artefact that puts the
  two populations side by side, and E063's thin 0.05 margin is the first hint of the real number. A
  null from one instrument (E032b) is not a null from the space of instruments.
- **Cost.** ≈4 GPU-min (2 × ~120 crops × 4 arms + 2 encodes).
- **Expected delta.** Criterion 2, indirectly but decisively: pass (AUC ≥ 0.85, median margin ≥ 0.15)
  re-opens the audit family with a usable margin and makes 4.5 worth tuning; fail (AUC ≤ 0.65) closes
  appearance-based identity for DanceTrack *with an engaged control against the correct adversarial
  pair*, which the ledger currently does not have, and redirects the effort to 4.3.
- **Cheapest falsification.** The control arm is the falsification: if `yolo26n-reid` on the real
  followed crop reproduces the E045b shape (win rate near 0.5, median margin near 0) *and* the
  pretrained arm and the 2× arm do not move it, the axis is closed. Note the discrimination must be
  measured against E063's own operating point (margin 0.05) to be comparable.
- **Caveat to report with the result.** AUC on 113 frames has a ±≈0.09 confidence half-width at this
  sample size; report bootstrap CI and do not declare a 0.03 AUC difference.

### 4.3 (3) `C2-TRACKLET-ID` — give the target layer the all-person layer's track ids

- **Mechanism.** Run `AllPersonTracker` (already shipped, `src/tgtrack/mot_layer.py`, 41.9 ms/f at
  CH@1280) in the target harness; bind the click frame's track id; then (a) refuse to accept a
  detection belonging to a different tracklet than the current one while `state == tracking`, and
  (b) allow a switch only to a detection in the click-time tracklet, or — if that tracklet is dead —
  to a tracklet whose first appearance is consistent with the target's motion prediction. This is a
  structural veto with no appearance term and no latency: it changes the *state machine*, not a
  threshold.
- **Why the ledger does not refute it.** E060 asserted exactly this transition ("accepting a detection
  that belongs to a different tube while TRACKING") and E062 retracted the *measurement* because both
  probes resolved GT with `seq.gt_at(i)[0]`. The retraction removed the per-frame identity attribution;
  it did not test the mechanism, and E062's replacement numbers make the mechanism *more* attractive:
  the target is in the detection list on 108/113 wrong frames (median IoU 0.9336) and the followed box
  is a different, adjacent object (0.766× area) held for 44–54 frame runs (§2.5). Nothing in the ledger
  measures tracklet purity on DanceTrack, and §1.3 confirms the wiring does not exist.
- **Cost.** ≈10 GPU-min. Wiring is the expensive part (a flag on the target harness + one audit
  branch), but the *measurement* is cheap: one all-person pass over 3 DanceTrack sequences is
  ≈0.5 GPU-min.
- **Expected delta.** Criterion 2, and it *converts* criterion 2's identity problem into criterion 1's
  association problem — one problem instead of two, on the criterion that has visible headroom. If
  tracklet purity holds, the upper bound is large: 116/116 wrong rows in the 0007 baseline are on a
  non-target GT id, so a perfect tracklet oracle catches all of them; the realistic number is set by
  the MOT layer's IDSW rate.
- **Cheapest falsification (and the number nobody has).** Read-only probe, no tracker change: run the
  all-person tracker over 0007's 200 frames, take the target's track id at the click frame, then
  compute (a) **coverage** = fraction of frames in which a detection with GT-target IoU ≥ 0.5 carries
  the click track id, and (b) **separation** = fraction of the 113 wrong frames whose followed box
  carries a *different* track id. Pass: coverage ≥ 0.85 and separation ≥ 0.70. Fail: coverage < 0.60
  (the tracklets fragment in the crowd) → drop the direction, because the veto would then be reported
  as a loss and would fail E021's lesson that FAR comes from re-acquisition. ≈2 GPU-min.
- **Explicitly not claimed.** That DanceTrack all-person MOT is good. Nobody has measured it; that is
  why the falsification above comes first.

### 4.4 (4) `C1-FRAME-GAP` — is AssA being measured through a 3-frame sampling artefact?

- **Mechanism.** Every motion constant in the project (`track_buffer` 30, `match_thresh`, gate radii,
  KF process noise, `velocity_clamp`) is expressed in *processed* frames, but the criterion-1 protocol
  feeds every **third** source frame (E047: 7 seqs, subsample 3, 200 frames). At a 3-frame stride the
  prediction is up to 3 frames stale and the effective per-step displacement triples. Measure the
  frame-gap sensitivity directly: same 200-frame window, subsample 1 vs subsample 3 on 3 sequences,
  comparing AssA / IDSW / MT per processed frame; then, if the gap matters, scale the KF process noise
  and gate radius by the measured stride.
- **Why the ledger does not refute it.** No entry in the ledger varies the evaluation stride, and E007
  ("`track_buffer` 60–90 already bridges gaps") varied *buffer length*, not the gap itself — if the
  motion model is mis-scaled, a longer buffer does not help and E007's null is consistent with the
  artefact rather than evidence against it. E012's IDSW 299–475 at HOTA ≈0.49 is exactly the signature
  a mis-scaled motion model would produce.
- **Cost.** ≈8 GPU-min (3 seqs × 200 frames at subsample 1 is 600 processed frames × 50 ms).
- **Expected delta.** Criterion 1. Diagnostic-first: if AssA at stride 1 exceeds stride 3 by ≥2 pp,
  the campaign has been tuning against an artefact and the *legitimate* fix (stride-aware gates, or
  running the tracker at source cadence and scoring every third frame) is worth more than any
  association mechanism. If the delta is <0.5 pp, criterion 1's ceiling argument becomes sound and the
  number should be quoted in `docs/METHOD.md` as a protocol invariant.
- **Cheapest falsification.** One sequence (MOT17-02), two strides, AssA and IDSW only: ≈2 GPU-min.
  If AssA is flat, the axis closes immediately and cheaply.
- **Guard.** This must not become a benchmark change. The scored frame set stays the same; only the
  tracker's internal cadence changes.

### 4.5 (5) `C2-AUDIT-2REF` — E064, but as a 2×2 rather than one conjunctive rule

- **Mechanism.** E063's audit compares each rival to the frozen click anchor and switches when the
  rival is `audit_margin` closer (0.05 fires, 0.10 inert). Its guard failure on 0005 is a stale-reference
  veto: the target's look legitimately changes (E024: 0005 needs half-life 150), so "closest to the
  click crop" stops being the target. The named fix is a second reference (the most recent confident
  crop) with a disagreement test.
- **Why the ledger does not refute it.** E063's own diagnosis names it and it is not built. But the
  conjunctive form as written in `docs/reviews/README.md:42` is likely to **block its own win**: at the
  swap the click anchor prefers the *impostor* (E063's diagnostic run: `audit_best 0.3405 < audit_cur
  0.4638`), and the recent crop — taken while the track was still correct — prefers the *true target*.
  Requiring both references to prefer the rival therefore suppresses the 0007 switch (+20.3 pp) while it
  correctly suppresses the 0005 false switch. Run four arms on the same three sequences:
  `click-only` (= E063 i3/m0.05, the +20.3 pp reference), `recent-only`, `AND` (the dispatched E064),
  `recent-AND-NOT-click`.
- **Cost.** ≈15 GPU-min (3 seqs × 4 arms × 100 processed frames × 1.27 s).
- **Expected delta.** Criterion 2. Target: 0007 ≥ 0.5685 while 0005 stays ≥ 0.40 and 0004 ≥ 0.415, i.e.
  a pooled Pareto gain over 0.4138 / 0.3833. On E063's own numbers, `recent-only` is the most likely
  winner (a recent reference cannot go stale the way the click crop does), and if `recent-only` ≈
  `click-only` on 0007 but protects 0005, the mechanism ships.
- **Cheapest falsification.** Drop 0007 to a second stage: run 0005 only (3 arms) and kill any arm that
  does not cut 0005's false-acquisition frames (baseline 103 of 200, `results/e063b_dt0005/summary.json`)
  below ~60 with TSR ≥ 0.40. ≈5 GPU-min. Only arms that pass 0005 earn the 0007 run.
- **Note for the implementer already dispatched.** If the 2×2 is not run, at minimum keep
  `click-only` as a control arm in the same harness, otherwise the E063 gain and the E064 mechanism
  are not comparable and the round will not be able to tell "fixed" from "disabled".

### 4.6 Ranking rationale (why this order)

| rank | id | criterion | cost (GPU-min) | why at this rank |
|---|---|---|---|---|
| 1 | `C1-ASSOC-ENGAGED` | 1 | 5 | cheapest; the only unrun hypothesis aimed at HOTA's binding term; corrects a false null already in the ledger |
| 2 | `C2-DISC-EXIST` | 2 | 4 | gates 16 experiments' worth of appearance work; second cheapest; returns a number (margin) the programme has never had |
| 3 | `C2-TRACKLET-ID` | 2 (→1) | 10 | highest-EV *new axis*; converts identity into association; falsified in 2 GPU-min before any wiring |
| 4 | `C1-FRAME-GAP` | 1 | 8 | cheap, and either kills the artefact or kills the ceiling argument — decision value either way |
| 5 | `C2-AUDIT-2REF` | 2 | 15 | on the critical path and already dispatched, but it is an appearance-audit continuation whose margin may not exist until 4.2 reports |

If the round can only afford one: **`C1-ASSOC-ENGAGED`**. If it can only afford one *new axis*:
**`C2-TRACKLET-ID`**.

---

## 5. Stop doing this

### 5.1 Stop buying detectors and substitution geometry for criterion 1

Six independent negatives with mechanisms, and the ledger keeps returning to it:

| id | move | result |
|---|---|---|
| E001 | 2×2 tiling @1280 | recall 0.721 unchanged, 73.8 vs 24.8 ms/f |
| E002 | conf 0.25 → 0.10 | DetA +0.9 pp, HOTA ±0.3 pp, marginal precision 0.553 |
| E003 | yolo26l@1280 / m@1920 | recall 0.707 / 0.687 vs 0.733 |
| E011 | n+m+l ensemble | recall 0.5911 vs 0.5607, marginal precision 0.21 |
| E055 | box fusion with the 0.8303-recall union | **HOTA 0.4944** vs 0.5050, 2× frame time |
| E056 | CrowdHuman primacy + yolo26m geometry snap | **HOTA 0.4966** vs 0.5050 |
| E057 | CH @960 | 0.4847 vs 0.5050 |

Past ≈0.80 recall this stream's HOTA is set by precision and association (E055's own diagnosis), and
the one detector move that ever paid was a *distribution* change (E046), which bought +0.70 pp HOTA for
+7.2 pp recall — i.e. the exchange rate between recall and HOTA here is ≈1:10. Any further detector
experiment should be refused unless it predicts an AssA mechanism, which no detector experiment can.

### 5.2 Stop tuning the appearance bank / band / margin family

E024, E025, E026, E027, E028, E029, E030, E036, E037, E038, E039, E040, E041, E042, E043, E044 — 16
experiments on one discriminator, and the family's own end state is E032b: the conjunction is
**bit-identical** (the mechanisms never fire together) and the impostor is measurably *closer* to the
target's past than the target is. E044 shows the decision surface is sharply bimodal
(`margin` 0.00 → 0.224/0.428, 0.06 → 0.512/0.280, 0.12 → 0.355/0.408), which is the signature of a
threshold sitting on a discriminator with no margin, not of an untuned knob. E063 is the family's one
real gain and it is *also* the demonstration that the lever is the audit's timing/reference structure,
not the bank: interval 15 / margin 0.10 / persist 2 are all inert for structural reasons.

Concretely, do not run "E063 with more interval/margin grid points". That is E044 again. The next
audit experiment must change the *evidence* (4.2, 4.3) or the *reference structure* (4.5), and any new
audit knob must ship with a fired-mechanism assertion, because this family has already produced two
inert-by-construction runs (E023's anchors, E063's first sweep).

### 5.3 Stop quoting 0.512 / 0.280 as the criterion-2 baseline

It is 0005-target-2; the current harness auto-picks 0005-target-0 (§1.1). The honest statement is that
the campaign has **two** criterion-2 baselines and must pick one in writing before any of the five
experiments above is scored. This is the same class of error as E033's stale baseline, one level down:
not a stale file, a stale *benchmark identity*.

### 5.4 Two smaller ones

- **Do not re-propose camera-motion compensation as unexplored.** `gmc_method: sparseOptFlow` is
  default in `MotCfg` (`src/tgtrack/config.py:152`), is forwarded to every backend
  (`mot_layer.py:230`), and the shipped `TRACKTRACK` applies it every frame
  (`ultralytics/trackers/track_tracker.py:405,521-522`). It is already in the shipped number.
- **Do not re-run the retracted probes.** `probe_decision.py` / `probe_merged7.py` still resolve GT
  with `seq.gt_at(i)[0]`; anything measured with them is inadmissible, and `R_method` should require a
  test that the resolver is `tgtrack.data.make_target_gt`.

---

## 6. Precondition (not ranked, but blocking): freeze the criterion-2 target set

Before 4.2, 4.3 or 4.5 is scored, write the frozen set into the harness and the ledger:

```
dancetrack0004 --target-id 3
dancetrack0005 --target-id 2   # NOT 0; the E041–E044 epoch target, TSR 0.695 / FAR 0.205
dancetrack0007 --target-id 1
```

Either choice is defensible (target 2 is the "stressed" forced pick the bank/hard-negative family was
tuned on; target 0 is the harder auto-pick). What is not defensible is switching between them inside a
comparison, which is what E063's pooled table does. Re-baseline cost: 3 seqs × 1 variant ≈ 6 GPU-min.
If the team prefers target 0, then the criterion-2 bar (TSR ≥ 0.55, FAR < 0.20) must be re-derived
against 0.400/0.515 and the E041-era gains re-stated as "on a different target".

---

## Appendix — read-only evidence used

- `results/e059_diag7/runs/dancetrack0007__baseline__none.csv` — 200 rows; columns include `gt_xyxy`,
  `pred_xyxy`, `target_person_id`, `followed_id`, `best_gt_id`, `correct`. §2.5 table and onsets are
  computed from it.
- `results/e041_dancetrack0005/summary.json`, `results/e063b_dt0005/summary.json`,
  `results/e041_dancetrack0004`, `results/e063b_dt0004`, `results/e041_dancetrack0007`,
  `results/e063b_dt0007` — `meta.target_ids`, `target_decisions[*].reason`, baseline rows.
- `results/e063_dt0004`, `results/e063_dt0007`, `results/e063b_dt0007` — audit grid rows.
- Code: `src/tgtrack/target_tracker.py:61-66` (imports; no `mot_layer`), `src/tgtrack/mot_layer.py:94,
  230, 301-327` (`apply_track_ids`, `gmc_method` forwarding, `_resolve_feats`), `scripts/eval_mot.py:598,
  627-632` (`opts.embed` gate on `feats`), `src/tgtrack/config.py:152` (`gmc_method` default),
  `C:\Software\Anaconda3\Lib\site-packages\ultralytics\trackers\utils\reid.py:139-161` (`build_encoder`
  pass-through), `...\ultralytics\trackers\track_tracker.py:405, 438, 521-522` (GMC on, NaN cosine
  fallback to HMIoU, per-frame warp).
- Datasets present and used for the cost estimates: `data/raw/mot17/MOT17`, 7 DanceTrack val sequences
  (`data/raw/dancetrack/val/dancetrack0004|0005|0007|0010|0014|0018|0019`). GPU: RTX 4060 Laptop, 8 GB.

---

### Numbered actions, in severity order

1. Freeze the criterion-2 target set (`--target-id 3,2,1`) and re-baseline; annotate E063's pooled row
   as measured on a mixed target set (§1.1, §6).
2. Run `C1-ASSOC-ENGAGED` with a fired-mechanism assertion; record AssA + the number of detections that
   carried a feature row (§4.1). If AssA < +0.5 pp with features actually used, write the null into the
   ledger and stop.
3. Run `C2-DISC-EXIST` before any further audit tuning, and publish the AUC/margin with a bootstrap CI
   (§4.2).
4. Run the 2-GPU-min tracklet-purity probe (coverage/separation on 0007) before implementing
   `C2-TRACKLET-ID`; only wire the MOT layer if coverage ≥ 0.85 (§4.3).
5. Add `--all-person-track` (off by default) to the target harness only after step 4 passes.
6. Re-run E064 as a 2×2 with `click-only` retained as a control arm, 0005 first (§4.5).
7. Add the frame-gap diagnostic (stride 1 vs 3, one sequence) to `docs/METHOD.md` as a protocol
   invariant (§4.4).
8. Refuse further detector-substitution experiments for criterion 1 (§5.1); mark the axis CLOSED in
   `docs/RESEARCH_PLAN.md`'s hypothesis board.
9. Add a test/guard that a `with_reid: true` config run must assert a non-empty appearance block, so
   the E053 class of null cannot recur (§1.2, `R_method` overlap).
