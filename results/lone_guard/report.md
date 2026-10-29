# E086-r — the lone-candidate guard on real data: a controlled pair at last

**Question.** Does arming `CostCfg.lone_candidate_accept_cost` (shipped default `None`)
remove false acquisitions on real DanceTrack sequences, and at what price? The
previous claim in this repo ("3.6 pp of target success on the upper-body mask") was
**retracted** because it compared `results/target_obs19_shipped` — which contains no
armed variant — against an armed grid. This report is the controlled replacement.

Pre-registration: `docs/E086_PLAN.md` (revision section, written before any armed
summary existed). Protocol: `scripts/vote_sweep.py`, DanceTrack val, 300 frames,
subsample 2, mask `none`, GT detector, per-sequence target ids identical to E079–E084.
Arms: `vote_off` (shipped, `None`) vs `vote_lone_0.30 / 0.40 / 0.50`.
Sequences were fixed in advance: four where the shipped tracker actually reports
strangers (`0010` FAR 0.680, `0018` 0.587, `0073` 0.583, `0077` 0.410) and two with
low shipped FAR (`0034` 0.155, `0079` 0.023).
Runner: `_x/e086_run.py`; raw summaries under `results/vote_sweep/runs/vote_lone_*/`.

## Results

| seq | TSR off | TSR 0.30 | TSR 0.40 | TSR 0.50 | FAR off | FAR 0.30 | FAR 0.40 | FAR 0.50 |
|---|---|---|---|---|---|---|---|---|
| 0010 | 0.2831 | 0.2831 | 0.2831 | **0.2904** | 0.6800 | 0.5867 | 0.5867 | **0.3500** |
| 0018 | 0.2579 | 0.1706 | 0.1706 | 0.1706 | 0.5873 | 0.1825 | 0.1825 | 0.1825 |
| 0073 | 0.2280 | 0.1240 | 0.1240 | 0.1080 | 0.5833 | 0.5633 | 0.5600 | 0.5567 |
| 0077 | 0.2333 | 0.0767 | 0.0767 | 0.0767 | 0.4100 | 0.0300 | 0.0300 | 0.0300 |
| 0034 | 0.1815 | **0.2944** | 0.2419 | 0.2419 | 0.1546 | **0.0790** | 0.1546 | 0.1546 |
| 0079 | 0.9133 | 0.9067 | 0.9067 | 0.9067 | 0.0233 | 0.0233 | 0.0233 | 0.0233 |
| **mean** | **0.3495** | 0.3092 | 0.3005 | 0.2991 | **0.4064** | 0.2442 | 0.2562 | **0.2162** |

Pooled deltas vs shipped: `0.30` **-4.03 pp TSR / -16.23 pp FAR**; `0.40`
-4.90 / -15.02; `0.50` -5.05 / **-19.02**. `id_consistency` rises on all three arms:
**0.4397 → 0.5612 / 0.5281 / 0.5463** (+12.2 pp at 0.30).

## Verdict: NO CHANGE (keep `None` shipped)

Pre-registered checks, all three arms: P1 (pooled dFAR <= -2 pp) **passes**; P2
(pooled dTSR >= -1 pp) **fails** (-4.0 to -5.1 pp); worst-case (no sequence below
-3 pp) **fails** (-15.67 pp on `0077`); P3 (low-FAR sequences bit-identical)
**fails**. The rule is a real FAR/TSR trade, not a free win.

Decisive secondary comparison — same six sequences, already measured in E084b:

| arm | mean TSR | mean FAR |
|---|---|---|
| shipped (`vote_off`) | 0.3495 | 0.4064 |
| `lone_0.30` | 0.3092 | 0.2442 |
| `anchor_vote="any"` (k=1) | 0.5406 | 0.2287 |
| **`anchor_vote="contrast"` δ=0.10** | **0.5926** | **0.1282** |

`contrast` **dominates** every lone arm on both axes: it buys a larger FAR cut
(-27.8 pp) while *gaining* +24.3 pp TSR. The lone guard is therefore the same trade
the vote family already offers, at a worse exchange rate, and it needs its own
mechanism (a global refusal that changes search behaviour) where the vote needs a
reference comparison.

## What the pair does establish (three findings)

1. **The guard is not inert on real data.** This is the opposite of its sibling knob:
   `missing_appearance_cost` is *bit-identical at every price* from 0.0 to 1.0
   (`results/motion_ablation/unid_cost.md`) because the handover happens with
   `n_dets = 1`, where `select_candidate` leaves `margin = inf` and price never enters
   the decision. The lone guard does not compare against a runner-up — it refuses on
   absolute cost — so it can and does fire in that regime.
2. **The direction of the retracted claim was right; only its control was missing.**
   The guard really does stop strangers being latched: FAR falls 16–19 pp pooled and
   `id_consistency` gains 12 pp. What the retracted comparison got wrong was the
   *price*, and even the correctly measured price (-4 to -5 pp TSR) is not the
   binding objection — domination by `contrast` is.
3. **The falsifier P3 was mis-specified, and its failure is informative.** I assumed
   a low shipped FAR made a sequence a control, so any movement there meant the guard
   was binding outside its regime. `dancetrack0034` moved *favourably* (+11.3 pp TSR,
   -7.6 pp FAR): its shipped run was silently latched onto strangers in a crowd
   (FAR 0.155 is not a control), and refusing lone candidates also kept the gallery
   clean, which raised later correct matches. So "controls must not move" is the wrong
   shape of falsifier — the right one is "movement must be explicable", and a
   pre-registered falsifier that fires in the *good* direction still kills the arm by
   rule, which is what happened here.

## Reproduce

```powershell
python _x/e086_run.py --table          # read out (no runs)
python _x/e086_run.py                  # ~85 min on the 4060, 17 new pairs
```

Raw: `results/vote_sweep/runs/{vote_off,vote_lone_0.30,vote_lone_0.40,vote_lone_0.50}/<seq>/summary.json`.
