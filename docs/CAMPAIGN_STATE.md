# Campaign state -- the one page to read first

_Written 2026-10-30 02:35 (Asia/Shanghai).  Everything below is measured, on the protocols named, with
the artifacts under `results/`; the long-form record is `docs/EXPERIMENTS.md` and the round log is
`docs/NEXT_ROUND.md`._

## What is shipped (the default configuration)

| knob | value | measured effect |
|---|---|---|
| detector | `yolo26m@1280` | DanceTrack TSR 0.368 -> 0.733 in the recall probe (E059) |
| MOT17 all-person | `tracktrack.yaml` + WideDeep (E046) | **HOTA 0.5050**, DetA 0.4415, AssA 0.5322 |
| anchor vote | `off` | E079's +5.88 pp did not survive the held-out protocol |
| lost-scale | `anchor_band_lost_scale = 1.25` | TSR 0.2104 -> **0.2395**, FAR 0.2472 -> **0.2089** (E089, shipped) |
| partial view | upper-mask fusion | **TSR 0.8545**, FAR 0.000 (MOT17-02, target 19) |
| second channel | none | E095/E095b measured/designed, see below |

Release verification: E094 re-ran the shipped arm on the 22-sequence protocol and got **22/22
bit-identical** TSR/FAR/target-id versus the E089 records (max |d| 0.0).  `vote_off` is pinned to the
pre-E089 epoch (`anchor_band_lost_scale = 1.0`) so every historical comparison stays valid.

## Criteria (the standing objective)

1. **MOT17 HOTA >= 0.55** -- best **0.5050** (E046). Not met; detector-capped (DetA 0.42-0.44 across all
   six ultralytics backends).
2. **DanceTrack TSR >= 0.55 with FAR < 0.20** -- shipped **0.2395 / 0.2089**. Not met (FAR missed by
   0.9 pp, TSR by 31 pp). Best measured arm ever: `vote_contrast_0.10` at 0.2770 / 0.1830 (E082), which
   the pre-registered rule killed because its CI included 0.
3. **Partial view +10 pp** -- endpoint 0.8545 is real; the audit's +10.18 pp attribution is not. Open at
   **+7.27 pp of the required +10 pp**.
4. **Ledger + methodology** -- met: 4376-line append-only ledger, `docs/METHOD.md` lessons, this page.

## What the last eleven experiments established (E090-E101)

Every identity-side knob moves TSR and FAR in opposite directions at a near-1:1 exchange rate:

| experiment | mechanism | result |
|---|---|---|
| E090 | relax the ambiguity margin while lost | KILL: -1.28 pp TSR, 3 constraint violations, 0073 +27.6 pp at 0.01 vs +5.6 pp at 0.03 (non-monotone) |
| E091 | widen the ceiling only for near candidates | KILL: +50.3 pp on 0097, -20.0 pp on 0077; near-not-far is the dangerous candidate |
| E092 / E092b | delay re-acquisition until it persists | KILL: -8.11 pp FAR paid at -8.87 pp TSR; gap tolerance did not repair it |
| E093 | stack the two surviving Pareto points | KILL: substitutes, +0.92 pp combined |
| E096 | the unmeasured dose 1.50 | KILL: -5.17 pp TSR; the dose axis is a bifurcation switch |
| E098 | the same dose re-measured on the shipped path | KILL: asymptotic, non-monotone, slope sign flips per sequence |
| E099 | put the offline selector inside the tracker | KILL: pooled -1.40 pp; +23.13 on 0097, -23.90 on 0010 |
| E100 | normalise appearance prequentially | no effect: pooled hit 18.8-19.5 %, wrong 16.0-17.4 % for every scheme |
| E101 | does the offline score predict the online effect? | **no**: +4.45 pp frame-wise vs -1.40 pp online |

**The structural statement**: with one candidate set and one commit per frame, a missed detection and a
false report are the same event, so identity work is a Pareto surface, not an improvement; and a selector
cannot be validated by a per-frame GT join because a wrong commit persists through the state machine.

## The remaining legitimate headroom

- **The pool is there, the selector is the problem**: E097 joined the pre-veto pool to GT and found the
  target present on **65.84 %** of visible frames against the shipped tracker's **22.02 %**.
- **The only structural fix left**: a second decision channel that can *withhold* a commitment -- report
  "unassigned" for a frame without terminating the track, so a wrong identity stops being a commitment
  the state machine must follow. Pre-registered outline in `docs/NEXT_ROUND.md` round 44-45.
- **Criterion 1** needs a detector-side move (DetA), not a tracker-side one.

## Reproduce the headline numbers

```powershell
# shipped DanceTrack arm on the 22-sequence protocol
python scripts\vote_sweep.py --arms vote_off,shipped --limit-seqs 22
# partial-view guard (criterion 3)
python _x\e089_guard.py
# E094-style release verification (22/22 bit-identical)
python scripts\vote_sweep.py --arms vote_off,shipped
```

`python -m pytest -q` is green (600+ tests); `_x/publish.py` rebuilds `dist/tgtrack_handoff` and pushes it
to `github.com/DarknessLED/tgtrack` (message from `_x/handoff_msg.txt`).
