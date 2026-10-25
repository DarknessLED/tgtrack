
### E070 result — the gate axis is a smooth TSR/FAR trade-off that never reaches the pre-registered bar (6/6 points NO-SHIP)

The (built, running) entry above described the harness; this is the completed measurement on the full
held-out split. Hypothesis under test (`docs/NEXT_ROUND.md`): the held-out gap (TSR 0.2102 / FAR
0.2364, E068) is **gate calibration, not missing evidence**, because E069 detected the target in 84.4 %
of the frames the tracker loses. One knob was swept end to end, then the conditional second knob,
because the primary axis did show a monotone trade.

Split, printed at startup and asserted dev-free: 25 DanceTrack-val sequences minus the frozen dev three
(`dancetrack0004/0005/0007`) = **22 held-out sequences**, `dancetrack0010 0014 0018 0019 0025 0026 0030
0034 0035 0041 0043 0047 0058 0063 0065 0073 0077 0079 0081 0090 0094 0097`. Protocol per point: 300
frames, subsample 2, `--mask none`, `--jobs 1`, shipped defaults everywhere else. **132 (sequence, point)
pairs, 0 excluded at any point, 6181 report frames per point.** Target identity is pinned per sequence to
the shipped-point pick and each run's own `meta.target_ids` is asserted against it; no run uses
`seq.gt_at(i)[0]`.

| point | axis value | pooled TSR | 95 % CI | pooled FAR | 95 % CI | TSR vs shipped | seqs >5 pp | verdict |
|---|---|---|---|---|---|---|---|---|
| margin_0.20 (**shipped, reference**) | 0.20 | 0.2104 | [0.1157, 0.3254] | 0.2472 | [0.1489, 0.3487] | — | 0/22 | REFERENCE |
| margin_0.00 | 0.00 | **0.2460** | [0.1491, 0.3568] | 0.4166 | [0.3182, 0.5070] | **+3.56 pp** | 4/22 | **NO-SHIP** |
| margin_0.10 | 0.10 | 0.2282 | [0.1358, 0.3437] | 0.3433 | [0.2357, 0.4511] | +1.77 pp | 3/22 | **NO-SHIP** |
| margin_0.15 | 0.15 | 0.2163 | [0.1236, 0.3295] | 0.2407 | [0.1410, 0.3476] | +0.59 pp | 1/22 | **NO-SHIP** |
| margin_0.30 | 0.30 | 0.1931 | [0.0972, 0.3120] | 0.1443 | [0.0654, 0.2463] | −1.74 pp | 1/22 | **NO-SHIP** |
| band_k3.0 (secondary) | 3.0 | 0.2304 | [0.1283, 0.3485] | 0.3017 | [0.2060, 0.3989] | +2.00 pp | 2/22 | **NO-SHIP** |

Frame-weighted pooled means, sequence-level cluster bootstrap, 2000 reps, seed 0, resampling whole
sequences (same estimator as `scripts/heldout_report.py`, so the reference row is directly comparable to
E068). The pre-registered rule applied verbatim — *ship only if pooled TSR rises >= +5 pp and the upper
bound of the candidate's FAR CI does not exceed the shipped FAR CI upper bound* — gives:

- **margin_0.00: NO-SHIP.** TSR +3.56 pp misses the +5 pp bar; FAR CI upper 0.5070 vs shipped 0.3487.
- **margin_0.10: NO-SHIP.** +1.77 pp; FAR CI upper 0.4511.
- **margin_0.15: NO-SHIP.** +0.59 pp — this is the only candidate that keeps FAR statistically where it
  was (0.3476 vs 0.3487, a 0.11 pp margin), and it buys ~nothing in TSR.
- **margin_0.30: NO-SHIP.** TSR *falls* 1.74 pp and FAR more than halves (0.1443): a tighter gate is not
  free TSR, it is lost target.
- **band_k3.0: NO-SHIP.** +2.00 pp and FAR CI upper 0.3989.

**No candidate passes either conjunct cleanly, and none passes the first conjunct at all.** The gate is a
monotone dial (TSR up, FAR up as the margin loosens: 0.30 < 0.15 ~= 0.20 < 0.10 < 0.00) whose reachable
set along this axis simply does not contain (+5 pp TSR, FAR held). The trade is steep: the +3.56 pp of
TSR at margin 0.00 is bought with **+16.94 pp of FAR**, and the dial is not even one-directional in cost:
at margin 0.30 `dancetrack0010` collapses from 0.283 to 0.033 (−25.0 pp), i.e. tightening loses target it
was holding.

The per-sequence counts are the part a pooled mean hides, and they matter here: the pooled movement is
not a uniform shift. At margin 0.00, **4/22 sequences move >5 pp on TSR** — `dancetrack0041` **+28.72 pp**
(0.174 -> 0.462), `dancetrack0019` +11.11 pp (0.219 -> 0.330), `dancetrack0063` +8.00 pp (0.030 -> 0.110),
`dancetrack0026` +5.65 pp (0.081 -> 0.137) — while **14/22 move >5 pp on FAR**, headed by
`dancetrack0025` +51.0 pp, `dancetrack0065` +43.33, `dancetrack0058` +42.33, `dancetrack0035` +41.41,
`dancetrack0026` +40.4, `dancetrack0030` +34.0. So the global gain is concentrated in a handful of
sequences that happen to be losing the target early, paid for by the sequences that were already fine.
margin_0.10 movers: `dancetrack0081` +11.42, `dancetrack0065` +7.00, `dancetrack0034` +6.45.
margin_0.15: `dancetrack0035` +8.33. margin_0.30: `dancetrack0010` −25.0.

The secondary axis is worse than "small": `band_k3.0` moves *both directions violently* —
`dancetrack0010` TSR +36.76 pp with FAR −53.67 (the one genuine win of the whole sweep), against
`dancetrack0063` FAR +63.0 pp, `dancetrack0030` +52.67, `dancetrack0065` +32.0, `dancetrack0081` +26.0.
`anchor_adaptive_k` is not a calibration dial, it is a lottery on which of the two failure modes each
sequence lands in, and the pooled FAR CI upper (0.3989) proves it.

**Verdict: SHIP nothing. Defaults stay `cost.search_margin = 0.20` and `cost.anchor_adaptive_k = 4.0`.**
The diagnosis in `NEXT_ROUND.md` survives — the evidence *is* there and the gate *is* the thing throwing
it away — but the fix cannot be a global constant on this axis: no constant buys TSR without paying FAR
somewhere else. That is exactly the shape E072 (input-dependent `adaptive_search_margin`) is designed
for, and this sweep is its kill-check baseline.

**Spec defects found while executing (they do not change the numbers, but they change what a reader
would think was shipped):**

1. `docs/NEXT_ROUND.md` names the secondary knob **`cost.anchor_band_k`**. No such key exists in
   `src/tgtrack/config.py`; the adaptive-band knob is **`cost.anchor_adaptive_k`**, which is what this
   sweep overrides.
2. The doc calls 3.0 the *shipped* band k. The tree ships **4.0**
   (`CostCfg.anchor_adaptive_k = 4.0`), so the doc's {3.0, 4.0} is {candidate, shipped} — 3.0 was
   measured here as a candidate against 4.0, not the other way round.
3. `scripts/run_target.py --ablate` defaults to `"all"` (84 variants) and cannot express
   baseline-only, so each pair is a child process with `run_target.build_variants` pinned to one
   variant and `run_grid` wrapped to refuse anything but exactly one spec with exactly the intended
   overrides; `meta.variants` is asserted to be exactly the one variant.

**In-sweep reference vs E068 (why FAR reads 0.2472 here and 0.2364 there).** `src/tgtrack/target_tracker.py`
(mtime 2026-10-25 18:40:11) and `src/tgtrack/appearance.py` (18:54:48) were both modified **after** E068
finished (18:25:51), so E068 is not reproducible today. The shipped point re-run reproduces E068 exactly
on **20/22** sequences and diverges on two: `dancetrack0047` FAR 0.2633 -> 0.4700 (TSR identical at
0.1462) and `dancetrack0041` TSR 0.1692 -> 0.1744, FAR 0.0300 -> 0.0467. That, and not pooling, is the
1.1 pp FAR gap flagged as a "known inconsistency" in E070-prep. **Quote E068 for the external criterion;
quote this sweep's own margin_0.20 row for the paired rule**, because the rule must compare like with
like. `src/tgtrack/consensus.py` (E071) exists in the tree but nothing imports it, so it is inert for
these runs.

**Provenance.** Two sweep processes ran concurrently in this workspace at one point (a second agent
edited this driver mid-run), which corrupted one `summary.json` (`JSONDecodeError: Extra data: line 283
column 2 (char 7766)`, a single stray `}`; recovered with `raw_decode`, recorded as `json_recovered`).
Every number here comes from a run directory stamped `.mine` by this sweep — 47 of the pre-guard runs
were backfilled only after their logs were checked against the stored summary. The driver now takes a
single-instance lock (`results/gate_sweep/.sweep.lock`) and refuses to start beside a live sibling.
Per-pair summary, `specs.json` and logs are under `results/gate_sweep/runs/<point>/<seq>/`; every raw
number is in `results/gate_sweep/report.json`; the human table is `results/gate_sweep/REPORT.md`.

**Cost.** 132 pairs, summed measured GPU wall **9144.4 s (152.4 min)**, mean 69.3 s/pair (margin_0.00
66.9, 0.10 51.8, 0.15 65.5, 0.20 63.8, 0.30 90.7, band_k3.0 77.0), single process at `--jobs 1`
throughout; total session wall including the aborted first attempt and the re-runs after the provenance
guard was ~3 h.

Exact commands:

    $env:PYTHONIOENCODING="utf-8"
    python scripts/gate_sweep.py --points margin_0.00,margin_0.10,margin_0.15,margin_0.20,margin_0.30,band_k3.0

the driver spawns, per (sequence, point), a child equivalent to:

    $env:GATESWEEP_POINT="margin_0.00" ; $env:GATESWEEP_VARIANT="sm_0.00"
    $env:GATESWEEP_OVERRIDES='{"cost.search_margin": 0.0}'
    python scripts/gate_sweep.py     # child mode: pins build_variants, then calls scripts/run_target.py main()
      -> run_target argv: --data-root data\raw\dancetrack\val\dancetrack0010
                          --out results\gate_sweep\runs\margin_0.00\dancetrack0010
                          --mask none --max-frames 300 --subsample 2 --jobs 1
                          --ablate gate_sweep [--target-id <pinned>]

