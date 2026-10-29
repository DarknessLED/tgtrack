# E102 -- pre-registration: can the tracker *withhold* a report instead of coasting a wrong box?

_Written 2026-10-30 02:55, before any number in this experiment was computed._

## The observation that motivates it

E088 measured the shipped tracker's per-frame behaviour on three sequences: on **91.2 %** of the frames
where the target is visible and not reported correctly, the tracker *is* reporting -- it is coasting a
stale prediction onto the wrong person (55 % of those frames at `frames_since_update >= 60`).  The
tracker never returns "no answer": `no_report` is 0.0 % on visible frames.

That is the design flaw behind the near-1:1 TSR/FAR exchange rate of E082/E089/E092/E098/E099/E101: with
one commit per frame, refusing a candidate and reporting a wrong person are the same choice.  A tracker
that *could* say "unassigned" would pay for abstention only where it abstains, and abstention on a frame
that would have been **wrong** is free for TSR (a wrong report earns no TSR) while improving FAR twice
(the numerator drops and the denominator drops).

## Mechanism

`report_suppress_after` (new state knob, default 0 = off, bit-identical): while the tracker is coasting
(`frames_since_update >= K`), do not emit a reported box for that frame -- keep the internal state,
identity and lock exactly as they are, only the *report* is withheld.  This changes no decision the
tracker makes; it changes what it claims.

## Prediction

Correction: abstention is not free in general -- the baseline warns that a coast frame can be *correct*.
So the vertical question is the purity of the coast population: over coast frames with
`frames_since_update >= K`, what fraction are correct?  If it is below ~50 %, then abstaining there
should raise FAR more than it costs TSR.

Pre-registered predictions:

- **P1**: for K = 30, fewer than 50 % of reported coast frames are correct (i.e. abstention there is
  TSR-cheap).
- **P2**: pooled FAR over the diagnostic set falls by >= 5 pp with pooled TSR falling by <= 3 pp at some
  K in {15, 30, 45, 60}.
- **P3**: the gain is concentrated in the sequences whose shipped FAR is high (`0010` 0.68, `0073`
  0.58/0.38, `0035` ...), not in already-clean sequences (`0097` FAR 0.000).

## Diagnostic protocol (offline, from recorded per-frame traces)

Use the shipped-configuration per-frame CSVs already on disk -- `results/e097/<seq>/runs/*__none.csv`
(5 sequences) and `results/e088_ship/<seq>/runs/*__none.csv` (3 sequences) -- and replay the abstention
rule combinatorially: the tracker's decisions are identical, only `reported` changes, so TSR and FAR can
be recomputed exactly from the trace without re-running the tracker.

**Promote to the tracker only if** P2 holds on the diagnostic set; then implement the knob, verify the
K = 0 control is bit-identical, and measure on the 22-sequence protocol with the campaign rule
(paired per-sequence dTSR bootstrap CI + the E089 constraints: no sequence worse than 2.0 pp TSR, P2/P3).

**Kill if** P1 is false (coast frames are mostly correct) -- that would mean the tracker's stale
predictions are better than they look and abstention can only lose.
