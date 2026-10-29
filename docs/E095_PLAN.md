# E095 — the second decision channel: bounded-lag refutation

Status: **measured -- KILL (inert)**. Result in docs/EXPERIMENTS.md E095; the pass is bit-identical
in TSR on 5/5 screened sequences because the post-veto buffer holds no rival candidate. E095b
(pre-veto shadow candidates) is pre-registered in docs/NEXT_ROUND.md round 44. Written after E093 (docs/EXPERIMENTS.md) established that
every identity-layer lever lies on one Pareto surface with a ~1:1 TSR/FAR exchange rate.

## The problem the identity layer cannot solve

Four measured levers (E082 contrast vote, E089 widened lost ceiling, E092/E092b temporal hold, E093
stack) all move TSR and FAR in opposite directions at roughly one point of TSR per point of FAR. E093
showed the two recall-side levers are *substitutes*, not complements: stacking them scores lower than
either alone (+5.83 pp versus +6.83 pp against the same reference).

The cause is structural, not a missing knob. `TargetTracker` commits one identity per frame from one
candidate set, and the commit is final:

- if it commits and is wrong → a **false acquisition**;
- if it refuses → a **miss** (the frame is `blind` while the target is visible, which the protocol
  records as `blind & present`).

One decision, two metrics, opposite directions. No re-weighting of that decision can improve both.

## The mechanism: give the decision a bounded future

E095 adds a **second, independent channel** that can revise an already-made decision using evidence
that arrives a few frames *after* it, before the report is delivered. This is legitimate offline
tracking: every top MOT17 entry is a round/offline method, and a bounded lag `L` (default candidate:
8 frames) is a delivery latency, not a memory of the answer key.

Two pieces:

1. **Decision buffer.** `TargetTracker.step` keeps producing decisions as today, but the per-frame
   report is written to a ring buffer of depth `L` instead of being emitted immediately. At frame `t`
   the entry for `t - L` is released, possibly revised. With `L = 0` the shipped path is
   bit-identical (the buffer is a pass-through); this is the mandatory control.
2. **Refutation pass.** When an entry is about to be released, it is re-examined against the `L`
   buffered frames around it using evidence the online decision did not have:
   - **Trajectory smoothness**: the reported target trajectory over the buffer is fitted with a
     constant-velocity model; an entry that requires a physically implausible jump (beyond what the
     existing `cost.size` / `cost.direction` model already tolerates *within* a frame) is suspect.
   - **Cross-frame identity competition**: for a suspect entry, the candidate pool at that frame is
     re-read from the buffer's stored candidates and the assignment is redone for the whole buffer
     window as a short-tracklet matching problem (Hungarian over the window with a smoothness prior),
     rather than one frame at a time.
   - Only **re-attribution** is allowed, never "delete the report": deleting a report is exactly the
     metric gaming the protocol's `blind & present` share exists to expose, and E095's guard below
     forbids it.

## Why this can be one-sided where the knobs are not

The knobs are all *within-frame* evidence weightings, so they trade. The refutation pass has evidence
of a different kind: **the future of the trajectory**. On a wrong latch, the tracker typically
oscillates (E092 measured the wrong latch as a short run; E088 measured 55 % of wrong frames at
`frames_since_update >= 60`). A short-window global assignment can see that the two candidate tracks
are being followed alternately and can commit the whole window to one of them, whereas the online
decision had to choose per frame with no view of the switch.

## Pre-registration

**Arms.** `lag_refute_L8` (`state.decision_lag = 8`, refutation on) against the shipped default, plus
the control `lag_refute_L0` (must be bit-identical to shipped on every sequence).

**Protocol.** The standard 22 held-out DanceTrack sequences, 300 frames, subsample 2, mask none.

**Predictions.**
- P1: pooled TSR rises by >= 1.0 pp **and** pooled FAR falls by >= 1.0 pp (the one-sided result that
  no within-frame lever has produced). If the two still trade at 1:1, the coupling is not "one
  decision per frame" but something deeper, and the architectural hypothesis is wrong.
- P2: `id_consistency` rises (fewer switches is the direct consequence of window-level assignment).
- P3: the `blind & present` share does **not** rise by more than 1.0 pp. This is the anti-gaming
  guard: if the refutation pass is buying FAR by deleting reports, b&p will move up.
- P4: gains concentrate on the sequences where the shipped tracker oscillates between two candidates
  (`0010` -23.53 pp under the stack, `0077` +34.00, `0081` -11.42). A gain concentrated on sequences
  that were already stable would mean the pass is leeching from the smoothness prior, not fixing
  switches.

**Ship rule.** Same as the campaign: paired per-sequence dTSR bootstrap CI excludes 0, no sequence
worse than -2.0 pp, P2 (idcons drop <= 0.05 where reference >= 0.90) and P3 as hard constraints.

**Kill criteria.** (a) TSR falls on the screen; (b) b&p rises while FAR falls (gaming); (c) the
control `lag_refute_L0` is not bit-identical.

## Implementation sketch (where the hooks go)

- `src/tgtrack/config.py`: `StateCfg.decision_lag: int = 0`, `StateCfg.refute_window: int = 8`,
  `StateCfg.refute_smoothness: float = 0.5` (all inert at defaults).
- `src/tgtrack/target_tracker.py`: `step()` returns the decision as today, and additionally appends
  `(frame, box, candidate_boxes, candidate_costs, state)` to `self._decision_buffer`; a new
  `TargetTracker.release_ready()` yields the entries whose lag has elapsed, applying
  `_refute_window()` first. The buffer stores **candidate** information, not the answer, so the
  refutation cannot see the ground truth.
- `scripts/run_target.py`: the per-frame writer consumes `release_ready()` instead of `step()`'s
  return value; with `decision_lag = 0` this is a pass-through, so bit-identity holds by construction.
- `tests/test_decision_lag.py`: (1) lag 0 bit-identical, (2) buffer releases exactly once per frame in
  order, (3) a synthetic oscillation is re-attributed, (4) a clean trajectory is untouched, (5) b&p
  cannot improve through deletion (the refutation API has no "drop" operation).

## Cost estimate

No new model, no new detector: the pass is O(L^2) per frame on already-computed candidate data, i.e.
seconds per sequence. The whole 22-sequence measurement should cost less than a single E093 arm
(~100 min GPU), which is why it is the recommended next experiment rather than another sweep of the
identity knobs.
