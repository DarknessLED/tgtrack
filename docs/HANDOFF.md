# HANDOFF — read this first

You are inheriting a **click-to-track single-person tracking system** built in 6 days on one
8 GB laptop GPU, with no model fine-tuning. This note is written for an agent in a fresh
workspace: it states what exists, what is measured, what is *not* measured, and what to do
next. Everything here is backed by a file in this package.

---

## 1. What the system does

Given a video and a click on one person:

1. **detects all persons** every frame (`ultralytics` YOLO, `yolo26m@1280` in production,
   `crowdhuman_yolov8n` as the params-matched alternative);
2. **tracks all persons** for ID distinction only (6 pluggable MOT backends; `tracktrack.yaml`
   ships);
3. **tracks the clicked person** with a dedicated `TargetTracker`: IMM + NSA Kalman motion,
   part-aware long-term gallery, short-term EMA, hard-negative bank, distractor margin, honest
   refusal, and a re-acquisition search that only accepts identity-verifiable candidates.

The interesting part is not the pipeline, it is the **refusal logic**: the system is built to
say "I am not sure this is the same person" rather than to guess, because a wrong latch is
invisible to the user until it matters.

## 2. Where to look (in this order)

| file | what it gives you |
|---|---|
| `docs/REPORT.md` | the full report, method-oriented, with literature coordinates |
| `docs/EXPLORATION_TREE.md` | **the whole exploration as a tree**, including 31 pruned branches grouped by why they died, 3 self-retractions, and the "if we started over" path |
| `docs/EXPERIMENTS.md` | the ledger: ~70 experiments, each with hypothesis → config → measured delta → verdict |
| `docs/METHOD.md` | the evaluation protocol that makes the numbers trustworthy |
| `src/tgtrack/` | the library (`target_tracker.py` is the core, ~1.4 kLOC) |
| `tests/` | 480+ tests; they encode the measured behaviour, including the negative results |
| `results/` | every number quoted in the docs, as markdown/csv/small json |

## 3. The four acceptance criteria and where they stand

The campaign was aimed at four numbers. **Two met, two not** — state this honestly if you
report on it.

| # | criterion | status | best measured |
|---|---|---|---|
| 1 | all-person MOT: MOT17 HOTA from 0.4827 to ≥0.55 | **not met** | 0.5050 (`crowdhuman_yolov8n` + ignore-region convention). The detector, not the strategy, is the ceiling: emitted recall is ~0.55–0.74 and gating only costs 1–3 pp of it |
| 2 | single-target on DanceTrack: FAR from 0.49 to <0.20 with TSR ≥0.55 | **not met** | dev 0.512 / 0.280 but **held-out 22 sequences: TSR 0.2104 / FAR 0.2472** (E070, 6-point gate sweep, 132 pairs, 0 excluded; CIs [0.116, 0.325] / [0.149, 0.349]). The dev number does not generalise, and no gate axis point reaches the pre-registered +5 pp bar |
| 3 | partial-view (upper/lower body) TSR +10 pp | **met** | +10.18 pp in a same-round controlled comparison (0.7527 → 0.8545), from reliability-weighted part fusion behind a shape-deviation gate |
| 4 | reusable experiment ledger + methodology | **met** | `docs/EXPERIMENTS.md` + `docs/METHOD.md` |

## 4. What is PROVEN (do not re-derive, cite instead)

- **Detection is the all-person ceiling.** Tiling, lower thresholds, bigger models, native
  1920, temporal accumulation and box-level detector fusion were all measured and all failed
  to raise HOTA (E001–E003, E054–E055). Past ~0.80 recall, HOTA is limited by detection
  *precision* and association.
- **The refusal logic is where the single-target gains came from**: distractor margin (E044),
  search-side identity gate (E022), hard-negative bank with an IoU admission gate and cap 8
  (E038/E041), self-calibrating identity band (E029/E030). Together: dev TSR 0.12 → 0.51.
- **Several plausible mechanisms are measurably dead**: feature whitening, pose identity,
  neighbourhood topology, trajectory shape, tube velocity, MOT-tracklet as an external witness,
  appearance-guided association, Hybrid-SORT backend, adaptive forgetting, gallery hygiene,
  pinned anchors, single-candidate guard. Each has a ledger entry with numbers and a reason.
- **The gate cannot be fixed by moving the threshold.** Lowering the acceptance margin buys
  TSR and pays FAR at roughly 1:9 (paired, held-out: +2.6 pp TSR for +23.2 pp FAR). This is
  the single most important negative result for whoever continues: **stop sweeping the
  threshold, change the decision rule.**

## 5. What is NOT measured / open

1. **Held-out performance is poor** (TSR 0.21): the shipped configuration does not generalise.
   Any claim must come from the 22 held-out sequences, never the frozen dev three
   (`dancetrack0004/0005/0007`).
2. **A decision-rule change is designed but not implemented**: `docs/E071_PLAN.md`
   (consensus-before-commit + temporal pooling of challenger appearance). This is the
   highest-value next experiment, with a pre-registered acceptance rule.
3. **The full 5-point gate sweep** (`margin` 0.00/0.10/0.15/0.20/0.30) was still running at
   handoff; `results/gate_sweep/` holds the completed pairs and the split-analysis tooling
   (`scripts/gate_sweep_split.py`, select-on-one-half protocol).
4. **Face / gait identity**: gait needs multi-frame sequences (not single-frame descriptors) and
   was never tested; face was rejected on resolution grounds. Both remain open with a stated
   precondition.

## 6. Traps that will save you days (all of these cost real time here)

- **`gt_at(i)[0]` is not the target.** It returns an arbitrary person with an *internal dense*
  id. The authoritative target ground truth is `run_target.py`'s `plan.gt_target` /
  `make_target_gt` in `src/tgtrack/data.py`. This trap produced two retracted conclusions.
- **`--target-id` is mandatory for DanceTrack target runs.** Without it the harness picks a
  target that is often not observable and TSR collapses to ~0.05 for reasons unrelated to the
  method. Always quote `meta.target_ids` next to a table.
- **`with_reid: True` in ultralytics is an empty switch** when the encoder is not supplied
  (bit-identical metrics). Never conclude "appearance doesn't help" from that flag.
- **`--ablate` defaults to all groups** in `run_target.py`; a "baseline" row you did not
  request may be an average over an axis.
- **`eval_mot.py` defaults to `--no-embed`** although `MotCfg.with_reid` ships True; the two
  paths differ by 2.6 pp.
- **Never run two GPU workers** (8 GB card; a second worker OOMs the BFCArena). `gate_sweep`
  now has a single-instance lock for this reason.
- **"Bit-identical" must be checked element-wise**, not asserted: a claimed parity was really
  `max|Δ| = 3.0e-4` (CUDA batch-shape nondeterminism), with 0/40 decisions changed.
- **Silent no-ops**: `snapshot()/restore()`, `set_half_life`, `snap_geometry`, a dataclass
  option field that was never copied into `RunOptions`. Every mechanism needs a counter that
  proves it fired before its metric is read ("inert ≠ harmless").
- **Environment**: PowerShell console is GBK — set `$env:PYTHONIOENCODING="utf-8"`; always
  read files with `encoding="utf-8"`.

## 7. Conventions to keep

- Every claim carries a measured number **and** the command that reproduces it.
- Negative results are first-class: they go in `docs/EXPERIMENTS.md` with a verdict.
- Selection and reporting use **disjoint** sequence sets; report cluster-bootstrap CIs.
- No model fine-tuning (hard constraint from the original request). Only strategy/inference.
- Stage your work like this campaign did: diagnose → literature → mechanism → ablation →
  held-out validation → ledger entry.

## 8. Reproduce in 10 minutes

```powershell
$env:PYTHONIOENCODING="utf-8"
pip install -r requirements.txt          # torch must come from the cu126 index (see file)
python -m pytest -q                      # 480+ tests, ~2 min, no GPU needed for most
# after placing datasets + weights (see MANIFEST.md §excluded):
python scripts/eval_mot.py --data-root data\raw\mot17\MOT17\train --subsample 3 `
    --max-frames 200 --trackers tracktrack.yaml --ignore-regions
python scripts/run_target.py --data-root data\raw\dancetrack\val --limit-sequences 1 `
    --target-id 1 --mask none,upper --ablate search_guard
```

## Update 2026-10-28 — E079 and E081 both closed (KILL), and what that changed

Both experiments targeted the same place: the frozen-click-crop veto, which is where 51 % of the
single-target loss is born. Neither shipped, and the reason is worth carrying forward:

- **E079** (k-of-3 reference vote over the crop veto) is the campaign's best signal: +5.88 pp
  paired TSR and **-5.65 pp FAR (to 0.1912)**, id_consistency +6.64 pp — but its TSR CI is
  [-1.61, +14.79], so it fails the pre-registered bar. Engagement is real (12,329 vetoed
  candidates examined; only 391 of 6,714 rescues survive the downstream refusal stack).
- **E081** (arm that vote only while the tracker is distressed) **falsified its own hypothesis**:
  it keeps 85-102 % of the FAR benefit and only 7-45 % of the TSR benefit, so the good rescues
  happen in confident frames. Its `tau = 1.00` null control is bit-identical to the shipped arm
  on 22/22 held-out sequences, which is the cleanest plumbing proof in the ledger.
- Current criterion-2 position: **TSR 0.2104 / FAR 0.2472** shipped; best-measured variant
  **0.2362 / 0.1896**. The FAR bound (< 0.20) is provably reachable, the TSR bound (>= 0.55) is
  not, on this protocol.
- Read `docs/E082_PLAN.md` before touching the vote again: the next test is a *contrast*
  requirement, not another gate or threshold.
