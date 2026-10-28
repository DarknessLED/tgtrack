# The TSR / FAR frontier for click-to-track identity (criterion 2), measured

Every number below is a **held-out DanceTrack-val** pooled measurement at the E070 protocol
(22 sequences, dev `dancetrack0004/0005/0007` excluded and asserted absent, 300 frames,
`--subsample 2 --mask none --jobs 1`), frame-weighted pooling, 95 % CIs from a 2000-rep cluster
bootstrap over sequences (seed 0). Rows marked ⚠ are partial coverage and say so.

The point of this table is not any single row. It is that **nine independent attempts to move the
operating point trace out one frontier, and no point on it comes close to the criterion** (TSR ≥ 0.55
with FAR < 0.20).

| # | mechanism / setting | TSR | FAR | source |
|---|---|---|---|---|
| — | **shipped default** (`search_margin 0.20`, band on, consensus off) | **0.2104** [0.1157, 0.3254] | **0.2472** [0.1489, 0.3487] | E070 |
| 1 | `search_margin 0.00` (no search-side margin) | 0.2460 [0.1491, 0.3568] | 0.4166 [0.3182, 0.5070] | E070 |
| 2 | `search_margin 0.10` | 0.2282 [0.1358, 0.3437] | 0.3433 [0.2357, 0.4511] | E070 |
| 3 | `search_margin 0.15` | 0.2163 [0.1236, 0.3295] | **0.2407** [0.1410, 0.3476] | E070 |
| 4 | `search_margin 0.30` | 0.1931 [0.0972, 0.3120] | **0.1443** [0.0654, 0.2463] | E070 |
| 5 | `anchor_adaptive_k 3.0` (looser adaptive band) | 0.2304 [0.1283, 0.3485] | 0.3017 [0.2060, 0.3989] | E070 |
| 6 | commit-consensus 2-frame gate | 0.1549 [0.0964, 0.2156] | 0.2118 [0.1206, 0.3089] | E073 |
| 7 | absorb guard, tight (ratio 0.90) | **0.1977** | **0.2234** | E078 |
| 8 | channel-conditional fusion ⚠ 7/22 seqs | +0.12 pp (CI incl. 0) | CI upper +33 pp | E077 |
| 9 | bounded-aspect coasting ⚠ 7/22 seqs | +0.07 pp (CI incl. 0) | — | E077 |
| 10 | three-reference vote, 2-of-3 (all 22 seqs) | **0.2668** | **0.1912** | E079 |

## What the frontier says

1. **TSR and FAR are almost perfectly anti-correlated along every axis we tried.** Moving the gate
   looser buys ~+3.6 pp TSR for +16.9 pp FAR (row 1); moving it tighter costs ~1.7 pp TSR for
   −10.3 pp FAR (row 4). There is no free direction, and the polynomials do not bend.
2. **The best achievable point is bounded — with one exception now visible.** Row 10 (the 2-of-3
   reference vote) is the first point measured anywhere in this campaign that moves TSR **and** FAR
   **and** id_consistency the right way simultaneously, and it puts FAR below the criterion's 0.20 for
   the first time (0.1912) while raising TSR to 0.2668. It still fails the pre-registered rule because
   the paired CI includes zero ([-1.85, +15.04] pp on 22 sequences, 7 up / 5 down / 10 flat), and it
   is concentrated: +74.2 pp on 0018 versus -23.5 pp on 0010. So the frontier *can* be pushed outward
   — the missing ingredient is selectivity (E081: apply the vote only in observable distress), not a
   better threshold.
3. **Refusal is not timidity, it is the measured optimum.** Row 10 (and E079's mechanism engagement)
   is the sharpest evidence: relaxing the hard anchor veto to a 2-of-3 vote does not "recover
   refused-but-correct evidence", it latches the tracker onto the first permissive impostor and TSR
   collapses to 0.048. The `all_gated:anchor_ceiling` veto that causes 51 % blindness (E076/E077) is
   *protecting* the 0.21 TSR we have.
4. **Below ~0.20 FAR there is nothing on the frontier but rows 4 and 6, both of which pay more TSR
   than they buy FAR.** The criterion as written therefore needs an information source this
   architecture does not have, not a better threshold.
5. **The only remaining structural change in the queue is E080** (let the all-person MOT layer carry
   the target between re-identifications). It is the one idea that changes *which* decisions the
   identity layer makes rather than how hard it gates — and it is the last one, because every
   threshold-shaped idea is exhausted (E032b: a whole 16-experiment family bit-identical).

## Honest caveats

- Row 7's numbers are absolute pooled values recomputed from the 66 per-pair records; the same
  recomputation gives row "shipped default" as 0.2104 / 0.2472 / id 0.4358 for its `absorb_off`
  reference arm — **bit-identical to the E070 reference**, which is a strong cross-script no-op
  proof. Row 7 is also the only mechanism so far that improved *identification quality* at all
  (id_consistency 0.4358 → 0.4528, and FAR −2.4 pp) while paying 1.3 pp of TSR; it failed the
  pre-registered bar, so it is not shipped, but it is the best-quality point on the frontier.
- Row 7's `absorb_loose` arm was **bit-identical** to the reference on all 22 sequences (dTSR =
  dFAR = dID = 0.00, CI [0,0]), i.e. that ratio never binds — a useful inert-probe, not a result.
- Rows 8-10 are partial coverage. Row 10 is a single sequence and is quoted only because its
  direction is unambiguous (TSR ×0.17).
- ms/frame is deliberately absent: the box ran 4-5× slower for parts of this epoch (SM clock pinned
  480-750 MHz), so timings are not comparable across rows.
