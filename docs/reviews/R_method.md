# R_method.md — independent methodological audit

**Scope.** The *method*, not the results: how verdicts are decided, whether the shipped defaults were
selected on data independent of the numbers that certify them, and whether the harness can even tell
the difference. I did not modify repository source and ran no experiments; the only executions were
read-only inspections (`results/*/summary.json`, the ledger, `scripts/*.py`, and one 200-frame GT
lookup per sequence through the public loader), plus the repo's own read-only
`scripts/check_baseline_freshness.py`.

**Evidence base actually parsed (not sampled).**

| what | count |
|---|---|
| ledger sections (`##`/`###`/`####` E0xx headings) | 66 (E001–E063, with `E031b`/`E032b`/`E032c`/`E045b`/`E063 guard`; **no E061 section**) |
| ledger `**Verdict` / heading verdicts | 38 verdict lines; 7 changes shipped to defaults |
| `results/*/summary.json` parsed | 122 dirs, **905 runs**, 96 distinct variant names |
| `results/*` directories total | 179, of which **57 have no `summary.json`** |
| headline table rows on DanceTrack (criterion 2) | **514** |
| headline rows on MOT17 (criteria 1/3) | 120, of which 81 are `mask=upper` (criterion 3) |
| DanceTrack sequences ever used | **3 of 25** (`0004`, `0005`, `0007`) |
| VisDrone target runs ever recorded | **0 of 7** |
| result dirs passing the repo's own freshness gate | **4 of 180** (176 STALE) |
| VCS | none — `C:\Projects\track` is not a git repository |

**Verdict in one paragraph.** The written protocol in `docs/METHOD.md` is better than the practice it
describes: it already contains the epoch rule, the freshness rule, the sample-size caveat and an
anti-pattern catalogue. What it does **not** contain is a held-out split, and every acceptance number
in the programme is computed on the same three DanceTrack sequences — and in the case of criterion 3,
the same single MOT17 sequence — on which its knobs were tuned. The ledger is unusually honest about
individual failures, but the *aggregate* claim ("criterion 2: 0.512 / 0.280") is a mean of three
hand-picked sequences whose 95 % interval over sequences spans 0.12–0.91, and which is not comparable
to the same-epoch number the ledger prints 46 experiments later (`0.4138 / 0.3833`) because the two
are computed on **different people** in `dancetrack0005`. None of this is a data problem: the fix costs
about ten minutes of compute.

### Global severity order (used by every section's action list)

| rank | finding | section |
|---|---|---|
| **S1** | No held-out data: every shipped knob is selected on the sequences that certify it; sequence-level CI of criterion 2 is ±0.39 | §1 |
| **S2** | Target identity and epoch are recorded but not *pinned*: `dancetrack0005` is person 0 in some epochs and person 2 in others, which alone moves criterion 2 FAR from 0.280 to 0.383 | §2 |
| **S3** | The `seq.gt_at(i)[0]` class of bug is still live in `scripts/probe_decision.py`, and the unit test that should catch it asserts the buggy convention | §4 |
| **S4** | 96 variants / 514 criterion-2 cells → ~3.5 expected false positives at α=0.05; the last shipped step is z≈1.3 | §3 |
| **S5** | Criteria 2/3 artifacts record no command line, no resolved config, no weights/source identity; the ledger and `results/` have already diverged (`e061_*`, `e064_*`) | §5 |

---

## 1. Statistical soundness

### 1.1 What one "verdict" is actually an average over

The measurement unit is **1 target × 1 sequence × 200 kept frames** (`--subsample 2`, so 400 original
frames), recorded per dir as `meta.subsample=2, meta.max_frames=200` and `targets{seq: target_id}`.
Criterion 2 is therefore a mean over **n = 3 sequences / 600 frames / 1 target each**. Criterion 3 is
`MOT17-02-FRCNN`, target 19, `--subsample 2`, 300 frames — **one sequence**, and every criterion-3
number in the ledger (0.7527, 0.7818, 0.8545, 0.8636, 0.9850) is that same window. Criterion 1
(`results/mot_eval_m1280/run.json`) is the only multi-sequence quantity: 7 sequences at
`--subsample 3`, 42 pairs.

**There is no RNG in the pipeline.** `grep -n 'random\|seed' src/tgtrack` returns exactly one hit,
`config.py:631 seed: int = 0`, and it is never read. So the campaign's "one seed" is not a variance
problem — every rerun is bit-exact (the ledger even uses that as a control: E032b and E058's
`bootstrap_0` are "bit-identical to baseline"). The variance that matters is **over sequences**, and
that is the one the design has 3 of.

### 1.2 The pooled criterion-2 number cannot support a sequence-level claim

Shipped per-sequence TSRs are 0.4150 (`0004`), 0.4264 (`0007`), 0.6950 (`0005`, person 2).

- pooled mean **0.5121**, between-sequence sd **0.1585**, se **0.0915** → t₉₅ (2 df) = **±0.394**,
  95 % interval **[0.118, 0.906]**. As a statement about DanceTrack, "0.512" is indistinguishable from
  0.15 and from 0.85.
- The only way to get a tight interval is to treat the 600 frames as independent: se = **2.0 pp**.
  That is pseudo-replication — frames of one target in one shot are strongly autocorrelated, and the
  design effect is unmeasured (no per-frame traces are retained for these runs).
- Even under that *favourable* model, the last shipped criterion-2 step (E041, cap 8, "pooled TSR
  0.475 → 0.512") is **+3.7 pp, z ≈ 1.28, p ≈ 0.20** two-sided — before multiplicity (§3).
- `docs/METHOD.md:49-50` already states the resolution limit ("200 frames × subsample 2 on one
  sequence is enough to reject a mechanism that moves 5 pp, and not enough to rank two mechanisms that
  differ by 1 pp"). Yet E042 ranks cap 6/8/12 at 0.507/0.512/0.512 and ships 8; E043 calls the
  memory mix a "local optimum" on ≤1 pp. Those are rankings below the protocol's own resolution.

### 1.3 Exposure: which knobs were chosen on the data that certifies them

"Selected on" = the sequences in the grid that decided the ship. "Reported on" = the sequences in the
acceptance number the programme quotes.

| shipped knob (current value, `config.py`) | selected on | reported on | exposure |
|---|---|---|---|
| `CostCfg.search_margin = 0.20` (:372) | 0004+0007 (+MOT17-02) | 0004, 0005, 0007 | **100 %** |
| `AnchorCfg.anchor_adaptive = True, k=4.0, floor .35, ceil .75, window 30` (:475-479) | 0007+0004+0005 | same 3 | **100 %** |
| `CostCfg.partial_motion_weight 0.5 / partial_appearance_weight 2.0` (:506-507) | MOT17-02 upper + 3 DT seqs | MOT17-02 upper | **100 %** |
| `CostCfg.partial_fusion_shape_dev = 0.45` (:512) | MOT17-02 upper | MOT17-02 upper | **100 %** |
| `BankCfg.hard_negative_max_iou = 0.30` (:276) | 0004+0005+0007 | same 3 | **100 %** |
| `BankCfg.hard_negative_capacity = 8` (:264) | 0004+0005+0007 | same 3 | **100 %** |
| `MotionCfg.velocity_clamp_ratio = 3.0` (:229) | MOT17-02 only | MOT17-02 (criterion 3) | **100 %** (1 seq) |
| `AuditCfg.audit_interval = 0` (:557) | 0004+0005+0007 | — | kept off (E063) |
| ignore-region convention (E009) | MOT17 ×7 | MOT17 ×7 (HOTA) | **100 %** |

**All five criterion-2 knobs (E022, E029, E031b, E038, E041) and the sole criterion-3 knob
(E016) were selected on exactly the sequences whose numbers certify them — 7 of 7 shipped
changes, counting the E009 ignore convention.** Criterion 1 is the only headline with any independence at all,
and even there the detector/threshold choices (E049's conf sweep, E057's `imgsz`) were made on the
same 7 sequences that produce 0.4827 / 0.5050. The ledger's own ship rule
(`METHOD.md:91-93`, "conservative-margin re-examination") is what institutionalised this: E022's tie
between margin 0.10 and 0.20 was broken on "the wider grid" — MOT17-02, one sequence, which is
criterion 3's only test sequence.

### 1.4 What a defensible split looks like, given three datasets

- **Dev (burned, keep forever):** `dancetrack0004`, `0005`, `0007`; `MOT17-02`; any VisDrone seq used
  for debugging. All 514 existing criterion-2 cells live here; nothing measured on these can certify
  anything.
- **Report (held out until the end):** the other 22 DanceTrack val sequences
  (`0010 0014 0018 0019 0025 0026 0030 0034 0035 0041 0043 0047 0058 0063 0065 0073 0077 0079 0081
  0090 0094 0097`), with the target chosen by a **pre-registered rule** (e.g. `--target-strategy
  observable`, target id written into the ledger before the run), and `MOT17-04/05/09/10/11/13` for
  criteria 1/3 (criterion 3 needs ≥3 MOT17 sequences, not one).
- **Out-of-domain:** VisDrone 7 sequences — currently **zero** target runs exist, and E016's open
  action item ("re-run on DanceTrack and VisDrone", `EXPERIMENTS.md:515-517`) has been open for 46
  experiments.
- **Inference:** cluster bootstrap over sequences (B = 1000, resample sequences with replacement), not
  over frames. With 22 sequences a ±0.10 interval on pooled TSR is reachable; with 3 it is not.

**Minimum protocol change that removes the confound** (one change, machine-enforced): a `--split
dev|report` flag on `run_target.py` that (i) tags every artifact with the split and (ii) **refuses to
run** when `split=report` is combined with `--variants`/`--ablate`/`--guard` (i.e. no sweeps on the
report set), plus one pre-registration JSON per experiment listing variants, criterion, dev sequences
and the decision rule. Cost of the one re-report that this implies: 22 sequences × 1 shipped config ×
200 frames at the measured 194 ms/frame ≈ **15 min single-threaded, ≈4 min with `--jobs 4`** — about 4 %
of the DanceTrack compute already spent on three sequences.

### Actions (by severity)

1. **(S1)** Add `--split {dev,report}` to `scripts/run_target.py`, write it into `summary.json` meta,
   and make the process exit non-zero if a sweep flag (`--variants`, `--ablate`, `--guard`) is combined
   with `--split report`. Then re-report criterion 2 **once**, shipped defaults, no variants, on the 22
   held-out DanceTrack sequences; record per-sequence TSR/FAR and a cluster bootstrap CI in the ledger.
2. **(S1)** Write `docs/PROTOCOL.md` with the dev/report sequence lists, the target-selection rule per
   split, and the inference unit (sequence, not frame); delete `METHOD.md:91-93`'s "re-examine on the
   wider grid" as a ship justification unless the wider grid is a report-split sequence.
3. **(S1)** Stop quoting "criterion 2 = 0.512 / 0.280" as a programme number. Replace it in
   `RESEARCH_PLAN.md` and in the ledger header with "dev-set: 0.512 / 0.280 (3 of 25 sequences,
   n=3, CI ±0.39)" until action 1 lands.
4. **(S1)** Re-measure criterion 3 on ≥3 held-out MOT17 sequences (`04/05/09`) at the criterion-3
   protocol (`--subsample 2 --max-frames 300`, forced `--target-id`), and declare `MOT17-02` dev.

---

## 2. Confound audit

### 2.1 The epochs, reconstructed from artifacts (there is no VCS to diff)

| epoch | marker | `0005` target | `0005` TSR/FAR | `0004` | `0007` |
|---|---|---|---|---|---|
| pre-E029 | E019–E028 | person **0** (E019, E018, E058 Grid A, E063 guard) | **0.995 / 0.000** (E019) → **0.400 / 0.515** (E063) | 0.265 / 0.225 | 0.401 / 0.550 |
| post-E029 (adaptive band) | E029–E037 | person **2** | **0.180 / 0.810** (E027) → **0.695 / 0.205** (E041) | 0.235 / 0.155 → 0.415 / 0.040 | 0.401 → **0.4264 / 0.5950** |
| post-E041 (cap 8) | E041 → | person 2 (headline), person 0 (E063 guard) | 0.695 / 0.205 | 0.4150 / 0.0400 | 0.4264 / 0.5950 |
| E063 epoch | E063/E063b | **person 0** in `e063b_dt0005` | **0.400 / 0.515** | 0.4150 / 0.0400 | 0.4264 / 0.5950 |

The decisive fact is that the two numbers the programme prints for the *same* criterion, in the *same*
session, differ only in which person on `dancetrack0005` counts as the target:

- `EXPERIMENTS.md:1818` (E058 position): "criterion 2 **0.512 / 0.280**" = mean over
  (0.4150, 0.4264, **0.695**) / (0.040, 0.595, **0.205**) — 0005 = person 2.
- `EXPERIMENTS.md:1947` (E063 guard): "Pooled … **0.4138 → 0.4262**, FAR **0.3833**" = mean over
  (0.4150, 0.4264, **0.400**) / (0.040, 0.595, **0.515**) — 0005 = person 0.

`0.4138` is *not* a different epoch of the same measurement; it is a different target. Nothing in the
directory says so: `e063b_dt0005/summary.json` records `targets.dancetrack0005 = {target_id: 0,
reason: "observable"}`, while the criterion-2 dirs record `target_id: 2, reason: "forced"`. The
criterion itself quietly changed identity, and the guard was run against a target the criterion does
not use.

### 2.2 Named cases where a quoted delta is cross-epoch, cross-target or cross-window

| # | quoted delta | where | why it is confounded | exact re-run needed |
|---|---|---|---|---|
| 1 | "FAR criterion failing → met (pooled 0.258 → **0.197**)" — the sole justification for shipping `search_margin=0.20` | `EXPERIMENTS.md:703-709, 737-739` | the pool is stated as "0005 **unchanged at 0.995 / 0.000**", i.e. it carries the person-0 target forward by assumption, in the same table that decides the ship | `python scripts/run_target.py --seq data/raw/dancetrack/val/dancetrack0005 --target-id 2 --subsample 2 --max-frames 200 --ablate search_guard` → compare `search_margin=None` vs `0.20` on the person-2 set |
| 2 | "Pooled 0.4138 → 0.4262" guard vs "criterion 2 = 0.512/0.280" | `:1947` vs `:1818` | different target on 0005 (0 vs 2) | `--seq dancetrack0005 --target-id 2 --variants audit_3,audit_5,shipped` (the run exists at person 0 only) |
| 3 | "**K=4 decision-identical to baseline** … the anchor is not the binding constraint on **0007**" | `:1796-1804` | `results/e058_dt0007/summary.json` actually ran **`dancetrack0004`, target 3, 200 frames** (its `bootstrap_0` id-consistency 0.9111 equals 0004's shipped 0.9111, and 0007's is 0.3990) | `run_target.py --seq dancetrack0007 --target-id 1 --subsample 2 --max-frames 200 --variants anchor_bootstrap` |
| 4 | "0005 alone: TSR **0.642 → 0.850**" (E058 Grid A) | `:1794` | `e058_bootstrap_dt` ran 0005 at **target 0**, and at `max_frames=120` | re-run with `--target-id 2 --max-frames 120` and `200` |
| 5 | "Grid C — MOT17-02 upper mask, **200 frames**: baseline **0.9850**" used as the criterion-3 guard | `:1806-1808` | criterion 3 is 300 frames (`e031b_mot17`, `e016_vclamp`, `e051_ch_partial` all `max_frames=300`); the baseline itself moves 0.985 → 0.8545 from the window alone | re-run bootstrap grid at `--max-frames 300 --subsample 2` so the guard is on the criterion's own baseline |
| 6 | "criterion 3 re-measured **OUT OF SEQUENCE** … all seven MOT17-FRCNN sequences (+2.71 pp)" (E034) | `:1176-1203` | the artifact contains **six** sequences (`02,05,09,10,11,13`; `04` absent), `e034_MOT17-04-FRCNN` and `e034_MOT17-05-FRCNN` are **two copies of the same 35-run sweep** (summaries differ only in `wall_seconds`, identical headline rows), `--subsample 1` (criterion 3 is 2), and targets were **auto-picked** (`reason: "observable"`, ids 23/1/20/11/25/2) rather than forced | re-run with an explicit 7-sequence list, `--target-id` forced, `--subsample 2 --max-frames 300`; delete the duplicate dir |
| 7 | "velocity clamp: upper TSR **+2.91 pp**" | `:503-517` | one sequence (MOT17-02), self-declared "Evidence quality: ONE sequence", open action item never done | `--ablate motion` (vclamp None/2.0/3.0) on 0004+0005+0007 and on 2 VisDrone sequences |

Also cross-window rather than cross-epoch: the E058 grids A (120 frames) and B (200) are quoted in the
same table as the criterion-2 numbers (200), and grid C (200) against a criterion measured at 300.

### 2.3 Why the mixing is invisible at the artifact level

`run_target.py` writes `meta.target_strategy`, `meta.target_ids`, `meta.variants`, `meta.guard`,
`meta.ablate`, `meta.subsample`, `meta.max_frames` and `targets{seq: {target_id, reason, guard_passed}}`.
That is enough to *detect* drift by hand (as done above) but there is no field that says "this artifact
is comparable to criterion X", no epoch tag, and no default-set identifier. `METHOD.md:32-35` mandates
"name the epoch … any table that mixes epochs is wrong", but nothing in the harness enforces the naming.

### Actions (by severity)

5. **(S2)** Add `meta.epoch` (a hash of the resolved config's default-set) and `meta.criterion`
   (`c2` / `c3-hota` / `surrogate`) to every `run_target.py` summary, and add
   `docs/EPOCHS.md` mapping each shipped-default epoch to its date, its knobs and the artifacts
   measured under it. Any ledger table that spans two epochs must cite two rows.
6. **(S2)** Re-run the eight cases in §2.2 — in particular #1 (the `search_margin` ship justification),
   #2 (the E063 guard at target 2) and #3 (E058 on the sequence it claims). Each is one digit change on
   an existing command; all eight are ≈30 runs ≈ 25 min at `--jobs 4`.
7. **(S2)** Make `--target-strategy observable` print a warning and record `meta.criterion_target_mismatch`
   when the auto-picked id differs from the id in `configs/criteria/*.json` (new file: the criterion
   definitions, one target id per sequence, checked in). Then E063's guard could not have silently
   switched 0005's target.
8. **(S2)** Correct the ledger in place for #3/#6 (strike "0007" → "0004" in E058 Grid B; strike
   "seven" → "six, two duplicate dirs" in E034; "200 frames" → "300 frames" in E058 Grid C) and mark
   them `detached` under the taxonomy in `METHOD.md:77`.

---

## 3. Multiple comparisons / novelty inflation

**How many variants were tried.** 905 runs across 122 summarized dirs, **96 distinct variant names**.
Against criterion 2 specifically: **514 headline cells**; against criterion 3: **81** upper-mask cells.
Knob families swept on the dev three: `search_margin` (3 values), adaptive band k (3) + floor/ceil (4) +
window (3), partial motion/appearance (3) + shape_dev (4), hard-negative `max_iou` (5) + `capacity` (7),
motion veto (5), distractor margin (3), memory half-life (4) + admission quality (3), eviction policy (3),
anchor pinning (4), anchor bootstrap (4), trail weight (2), audit interval/margin/persist (5),
`stm_weight`/`ltm_weight` (3) — roughly **70 independent criterion-2 cells over ~15 families**.

**Honest false-positive risk.** At an uncorrected α = 0.05 over ~70 cells, expected spurious "wins" =
**3.5**, family-wise error ≈ **97 %**; the Šidák threshold for 70 looks is α ≈ 7×10⁻⁴. The programme
shipped **7** default changes (E009, E016, E022, E029, E031b, E038, E041) out of 63 numbered
experiments and 38 explicit verdicts — i.e. an apparent 11 % hit rate, in a family where ~3.5 winners
are expected by chance alone. Two of the three criterion-2 ships are the same step measured twice
(E038 `max_iou` 0.30: 0.437 → 0.475; E041 cap 8: 0.475 → 0.512, **+3.7 pp, z≈1.3**), and E039/E042 then
showed both are *plateaus* — the value is arbitrary inside a flat band, which is exactly what selection
noise looks like.

**Who actually survived a guard.** Of the 7 ships, **none** has been re-validated on a sequence outside
the dev three. Guards that did their job and killed a candidate: E030 (band cap, +0007/−0004), E032
(motion veto, TSR for FAR), E035 (0.2/2.4 collapses 0004), E040 (cap 4), E051/E052 (CrowdHuman),
E058 (bootstrap), E059 (trail), E063 (audit interval — the ledger's best moment: it refuses a +20.3 pp
single-sequence gain because it costs 0004 and 0005). That is a healthy pattern *within* the dev set and
none of it touches the external-validity question.

### Actions (by severity)

9. **(S4)** Pre-register a variant budget per experiment (`docs/pre_reg/E0xx.json`: variants, criterion,
   dev sequences, decision rule, max N cells) and report a family-wise-adjusted threshold for the whole
   campaign so far: at minimum, annotate every shipped knob in `config.py` with the number of cells that
   were swept before it was chosen, and require **≥2 sequences outside the dev three** before any new
   ship.
10. **(S4)** Re-classify E038/E041 as one experiment with one family of 12 cells (not two Pareto moves),
   and downgrade "cap 8 is a Pareto point" to "cap 6/8/12 are indistinguishable; 8 is chosen for
   headroom, not because it measured better".

---

## 4. Instrumentation validity

**The bug class.** `data.Sequence.gt_at(i)` returns **every** annotated person in frame `i`
(`src/tgtrack/data.py:202-204`), in `gt.txt` row order. The target-facing resolver is
`make_target_gt(seq, target_id)` (`data.py:661-678`), which filters by `track_id`. `gt_at(i)[0]` is
therefore "the first person in the annotation file", not the target. I measured how often that is a
*different* person, on the exact (sequence, target) pairs the criterion uses, 200 frames, subsample 2:

| sequence | criterion target | frames where `gt_at(i)[0].track_id != target` | note |
|---|---|---|---|
| `dancetrack0004` | 3 | **200 / 200 (100 %)** | first row is always id 1 |
| `dancetrack0005` | **2** | **200 / 200 (100 %)** | first row is always id 0 |
| `dancetrack0005` | 0 (E063's guard) | 0 / 200 | the one case where `[0]` is right |
| `dancetrack0007` | 1 | **199 / 200 (99.5 %)** | plus 3 frames where the target is absent but another person is annotated |

So the retraction class is not an edge case — it is the **default outcome**, and it silently swaps
"target absent" for "different person present". The reason `0005` at person 0 looked coherent is that
it *is* the first row of `gt.txt`.

**It is still live.** `scripts/probe_decision.py:74-78` still resolves GT with
`g = seq.gt_at(i); return g[0].xyxy`, and that is the probe that produced E020's premise and E060's
retracted "the target is never detected after frame 70". The post-retraction probe
(`scripts/probe_recovery7.py:34,54`) uses `make_target_gt` — the fix was applied to the *new* probe,
not to the one that was wrong. (Probes that score *all* persons — `probe_miss.py`, `probe_recall.py`,
`probe_tiling.py`, `probe_temporal.py`, `pick_observable_target.py` — are legitimately using `gt_at`
and should not be "fixed".)

**The test encodes the bug.** `tests/test_run_target.py:274` asserts
`np.testing.assert_allclose(plan.gt_target[i], seq.gt_at(i)[0].xyxy)` — and every fixture in that file
puts the target in box 0 (`_seq(N)` builds `track_id=1` first). A resolver that regressed to `[0]`
would pass the entire suite. There is also `test_data.py:248` (`test_make_target_gt`) but its fixture
cannot distinguish "filtered by id" from "took the only/first box" for the failure mode above.

### The guard to add (test-level, so the class cannot recur)

a. **Adversarial fixture, not a convenient one.** A sequence whose frames contain decoy id **1** and
   target id **7**, decoy first, in every frame; assert `make_target_gt(seq, 7)` returns exactly the
   id-7 box, `len(...)==1` per frame, and — the canary — that
   `seq.gt_at(i)[0].track_id != 7` for at least one frame, so any `[0]`-style implementation fails.
b. **Invariant test over real data (cheap, no detector).** For every (sequence, target) pair listed in
   the new `configs/criteria/*.json`: for every frame, `len(make_target_gt(seq,tid)) ≤ 1`, and every
   returned array equals the box whose `track_id == tid`; count absent frames separately and compare
   with the ledger's stated `n_frames`.
c. **Runtime assertion in the harness, not just in the probe.** In `run_target.build_mask_plan` /
   `RunPlan` construction, assert for every frame that each target box came from
   `make_target_gt` and that `box.track_id == target_id` (the plan already stores `all_gt`;
   `tests/test_run_target.py:273` shows the plumbing exists).
d. **Remove the affordance.** Rename `Sequence.gt_at` → `gt_all` and add
   `Sequence.gt_target(frame, target_id)` as the only target-facing accessor; add a lint rule (ruff
   `TID`/custom flake8) banning `[0]` on `gt_all(...)` outside a whitelist of all-person analyses.
e. **Split the metric.** Report `n_target_absent_frames` (target not annotated) separately from
   `loss_frames`, so "not in GT" can never masquerade as "tracker lost the person" — the E060
   confusion.

### Actions (by severity)

11. **(S3)** Add the invariant test (b) and the canary fixture (a) to `tests/test_data.py` and
    `tests/test_run_target.py`, and fix `tests/test_run_target.py:274` to resolve GT with
    `make_target_gt(seq, 1)` instead of `gt_at(i)[0]`.
12. **(S3)** Fix `scripts/probe_decision.py:74-78` to use `make_target_gt(seq, args.target_id)`, then
    re-run E020/E060's surviving claims with it; strike any probe output that depended on `[0]`.
13. **(S3)** Add `n_target_absent_frames` to the per-run summary and the `RunPlan` assertion in (c), so
    a future probe cannot re-introduce the substitution unnoticed.

---

## 5. Reproducibility

### 5.1 What is recorded

Per `results/<dir>/summary.json`: `meta{data_root, detector, embedder, masks, ablate, jobs, subsample,
max_frames, target_strategy, target_ids, variants, guard, allow_invisible, target_decisions,
wall_seconds}`, `targets{seq: {target_id, reason, guard_passed, visible_median_w/h}}`, and per-row
metrics including `n_frames`, `n_sequences`, `ms_per_frame`. That is a good deal more than most
projects. `scripts/eval_mot.py:1016-1037` goes further and writes `run.json` with `argv`, resolved
`options`, `env{platform, python, tgtrack_version}` and per-pair JSON — so criterion 1 *is* fully
reproducible (`results/mot_eval_m1280/run.json` even reveals `--subsample 3`).

### 5.2 What is missing

- `run_target.py` — the harness behind **every** criterion-2/3 verdict — records **no `argv`**, no
  resolved config dump, no weights path or hash, no source identity, no frame indices, no seed
  declaration. `eval_mot.py` already has the recorder; it was never shared.
- The ledger: **39 of 66 sections quote no command line**, **33 of 66 quote no frame count**. Where a
  command is given it is often deictic ("identical command: …", "same command as the criterion-3 grid"),
  so reproducibility requires chasing a chain of references.
- **57 of 180 result dirs have no `summary.json`** — including all `mot_eval_*`, `mot_ignore*`,
  `mot17_ch_*` and `mot_hybrid_*` dirs. E009's +0.0124 HOTA and E047/E055/E056/E057's detector numbers
  live in those.
- The repo is **not under version control** (`git log` → "not a git repository"), so "epoch" cannot be
  reconstructed from commits; it can only be inferred from prose and mtimes.
- Freshness: running the repo's own gate gives **176 stale of 180 directories**, including every
  criterion-2 artifact (`e063b_dt0004/0005/0007`, `e063_dt0004/0007`). By `METHOD.md:37-41` ("never
  trust a baseline artifact from an earlier default-set") essentially the entire evidence base is
  unquotable. The gate is also unfixable in its current form: it compares against the newest `.py`
  anywhere under `src/`, `scripts/`, `configs/`, so editing an unrelated probe invalidates everything —
  which is why it is ignored. It needs to be per-artifact (`sources.json` written at run time).
- **Ledger/artifact divergence.** `results/e061_probe7`, `e061_tv_dt0004`, `e061_tv_dt0007` exist but
  **there is no E061 in the ledger** (the numbering skips it). `results/e064_dt0004`, `e064_dt0007`,
  `e064_logs` exist while the ledger's last line says E064 "is not built yet". Whatever their status,
  `results/` currently contains measurements that the ledger does not account for. The ledger was also
  being written *during* this audit (`docs/EXPERIMENTS.md` grew from 1959 to 1965 lines and three
  `e064_*` dirs appeared mid-session), which is exactly the condition under which "the ledger is the
  record" stops being true; all line references in this review are to the 1959/1965-line state.

### 5.3 Accepted results that cannot be reproduced from the ledger alone

| result | ledger | what is missing |
|---|---|---|
| E016 velocity clamp **+2.91 pp** | `:503-517` | no command; "identical command" is a deictic pointer at E014, and the numbers are from the pre-E029 HOTA-epoch tree |
| E022 DanceTrack half (the pooled FAR 0.258→0.197 that shipped `search_margin`) | `:690-709` | the command is given but the pooled row has **no single artifact**: it is a mean over two dirs plus an assumed-unchanged `0005`, and the target id / epoch are not recorded |
| E029 adaptive band (**SHIPPED**) | `:946-972` | no command (frames and dirs are recorded) |
| E031/E031b partial fusion (**SHIPPED**) | `:1003-1054` | no command, no frame count |
| E038 `max_iou=0.30` (**SHIPPED**) | `:1297-1326` | no command; the 0.475 baseline is only recoverable by re-deriving it from `e040_*`/`e041_*` |
| E041 `capacity=8` (**SHIPPED**) | `:1379-1407` | no command; and the "MOT17 check **0.8636**" has no traceable artifact — no `summary.json` in `results/` carries 0.8636 as a target-success rate (the only 0.8636 strings are in `results/partial_view_224`, a discriminability probe) |
| E058 grid A/B/C | `:1785-1808` | no command; B ran the wrong sequence; C ran the wrong window |
| E063 "0007 → 0.6294" | `:1905-1933` | no command; the 0.6294 row is in `e063b_dt0007`, not `e063_dt0007` (which contains only the inert 15/30 intervals) |
| E034/E035 out-of-sequence sweep | `:1176-1232` | no command; artifact has 6 seqs, two copies of one sweep, `--subsample 1`, auto targets |
| E046–E049, E054–E057 (detector work) | `:1546-1774` | no command or no `summary.json`; `_x/` probe scripts are outside `scripts/` |

### Actions (by severity)

14. **(S5)** Port `eval_mot.py`'s recorder into `run_target.py`: write `run.json` with `argv`,
    resolved `Config` (all knobs), `env{python, platform, tgtrack_version, torch, ultralytics}`,
    `sources{path: mtime, sha256}` for `src/tgtrack/*.py` + the weights file, and the frame index list.
    This single change closes actions 15 and 16 for all future runs.
15. **(S5)** Add `make ledger-lint`: for every ledger row with an accept/ship verdict, require
    `{command, frame count, subsample, target_id, epoch, results dir, ✓fresh}`; fail CI on a missing
    field. Run it once over E001–E063 and record the debt as an appendix (expected: ~25 shipped/unreproducible
    rows, listed in §5.3).
16. **(S5)** `git init` + commit the tree so "epoch" has a referent, and reconcile `results/` with the
    ledger: either document `e061_*` as the missing E061 or move it to `_orphan/`, and only write
    `e064_*` once E064 exists as a ledger section.
17. **(S5)** Replace the mtime gate with a per-artifact `sources.json` comparison (the freshness number
    must mean "a file this run imported changed", not "any `.py` anywhere changed"), and gate the
    report on it.

---

## The single change I would mandate first

**Split the data and make the split machine-enforced (action 1).** Declare `dancetrack0004/0005/0007`
and `MOT17-02` permanently dev; add `--split {dev,report}` to `run_target.py` that aborts any
sweep/guard flag on `report`; re-report criterion 2 once on the 22 held-out DanceTrack sequences with
shipped defaults and a cluster-bootstrap CI (~15 min of wall clock, 4 min with `--jobs 4`). Everything
else in this review becomes *measurable* once that exists: the epoch/target drift of §2, the
multiplicity of §3 and the reproducibility gaps of §5 all matter because there is no independent number
to check them against. Until then, "criterion 2: 0.512 / 0.280" is a mean of three sequences that were
used to choose every knob involved, and its 95 % interval over sequences contains both 0.12 and 0.91.
