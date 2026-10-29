# E099 -- the pre-veto pool as a second candidate source

**Status**: pre-registered 2026-10-29, not measured. **Motivation**: E097.

## What E097 established

On five informative held-out DanceTrack sequences the pre-veto candidate pool contains the true target
on 65.84 % of visible frames while the shipped tracker is correct on 22.02 % (+43.82 pp), and a 40-line
offline window selector over that pool converts 44 % of the headroom into +19.25 pp TSR (0.2202 ->
0.4127, TSR only, weights fit on the same five sequences).

The mechanism is not "more recall": the candidates are already produced and already scored. They are
*scored away* by `anchor_ceiling` before the vote is counted, which is why every identity-layer lever
measured so far (E082/E089/E092/E092b/E093/E096) exchanged TSR for FAR at ~1:1: they all re-weight the
one decision the ceiling already made.

## Mechanism (to implement)

1. `TargetTracker` keeps a bounded history of the pre-veto pool: `(frame, boxes, appearance_costs)` for
   the last `W` frames (`state.preveto_window`, default 0 = off).
2. When the committed identity has been coasting inside that window -- `frames_since_update >
   state.lost_short_after` on any frame in the window -- run the E097 DP over the window.
3. Re-attribute only if the winning chain's current-frame candidate is a different person from the
   committed one: IoU < 0.3 against every committed box in the window, **and** only while the tracker is
   in a blind state. Locked frames are never re-attributed.
4. On re-attribution, the committed box becomes the chain's current candidate and the state returns to
   `tracking` with `frames_since_update = 0`; otherwise nothing changes.

## Pre-registered bar (do not relax after seeing the result)

- Control `preveto_L0` must be **bit-identical** to the shipped default (22/22).
- Ship only if the paired per-sequence dTSR bootstrap CI (22 held-out sequences, 300 frames, subsample 2,
  the E089 protocol) excludes 0, **and**
  - P2: `id_consistency` may not drop more than 0.05 on any sequence where it was >= 0.90;
  - P3: no sequence drops more than 3.0 pp TSR;
  - P4: pooled FAR may rise by at most 3.0 pp.
- Predictions: P1 pooled TSR >= +5 pp; P2 gains concentrate on 0081 / 0097 / 0077 (the wide-pool, low-TSR
  sequences).

## Why it can be one-sided when the knobs were not

The knobs move the threshold on a decision the tracker already makes. This changes *when* the tracker is
allowed to look at candidates it already has: it acts only on frames where the shipped tracker is
provably blind (coasting >= `lost_short_after`, correct rate 0 % on E088's numbers) and only when the
window evidence names a different person outright. A frame the shipped tracker gets right cannot move,
which is exactly the guard the identity-layer levers could not offer.

## Risks

- The offline selector used GT only for scoring, not for selection, so the +19.25 pp should transfer --
  but the offline DP saw the *entire* 300-frame trajectory, while the online version sees only `W = 8`.
- FAR is unmeasured offline; the ~1:1 exchange of the identity layer is the prior that it will rise.
