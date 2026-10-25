# E071 plan — consensus-before-commit (a decision-rule change, not a threshold change)

Status: **design only.** Deliberately not implemented while the E070 sweep is running, because
`scripts/gate_sweep.py` imports `src/tgtrack` once per child run and editing the tracker
mid-sweep would silently mix code versions across the 110 pairs.

## The diagnosis this answers

E070 asks whether moving `cost.search_margin` can buy TSR without paying FAR. It can only
**slide along** a trade-off: the same scalar gate decides both "accept this person" and
"refuse this person", so every point on the axis trades one criterion for the other. The
measurements behind the current failure mode say something more specific:

- E063/E062: 113 wrong frames on dancetrack0007, **all in `tracking` state**, target detected
  in 108 of them, median detection IoU 0.93 — we are not failing to see the person, we are
  accepting the wrong box while the right one is available.
- E061: the swaps are **one-or-two-frame tie-breaks** (frame 59/103/162 transitions, smooth
  0.19/0.14/0.10 body-height jumps) — not a jump to a distant stranger.
- E064: the second witness (recent EMA) is the incumbent, so it cannot outvote the incumbent.

A single-frame decision rule is exactly the wrong instrument for a one-frame tie. The rule
should be **commit only on consensus over a few frames**, which is orthogonal to where the
gate sits.

## Mechanism

Two independent switches, both off by default, so each can be priced alone.

### M1 — `commit_consensus` (integer, default 0 = off)

When a *non-incumbent* candidate wins the association for `m` consecutive frames **and** those
`m` frames agree on the same person (mutual IoU of their boxes above `consensus_iou`,
default 0.30), only then switch identity. Until then the incumbent keeps the identity and
`state` stays `tracking` with a `pending_challenge` diag entry.

Why this need not cost TSR: a genuine re-acquisition (target reappears elsewhere) persists for
many frames, so it passes 2-of-2 or 3-of-3 trivially. A one-frame lookalike tie cannot. The
cost is **latency** (`m` frames), which is measurable and must be reported, not hidden.

### M2 — `consensus_pool` (bool, default False)

Before the m-of-m vote, pool the challenger's appearance over the pending window
(mean of L2-normalised descriptors) and re-test it against `stm`/`ltm`. A real person's pooled
descriptor is stable; a chimeric track assembled from two different people is not. This is the
"temporal pooling before decision" idea from video face recognition, transplanted to a
re-identification gate.

## Instrumentation (must exist before any run)

- `_pending`: list of `(frame_idx, det_index, box, feat)` for the current challenge;
- diag keys: `pending_n`, `pending_id_agree` (how many of the pending frames are the same
  person by IoU chaining), `pending_pool_dist`;
- event kinds: `challenge_open`, `challenge_abandoned`, `challenge_committed` with the frame
  span — an abandoned challenge is the *direct* measure of "a one-frame tie we refused",
  which is the behaviour E063 showed we need.

## Ablation points (register in `run_target.GROUPS["consensus"]`)

| variant | overrides |
|---|---|
| `consensus_off` | defaults (control) |
| `c2_iou30` | `commit_consensus=2, consensus_iou=0.30` |
| `c3_iou30` | `commit_consensus=3, consensus_iou=0.30` |
| `c3_iou15` | `commit_consensus=3, consensus_iou=0.15` (looser chaining) |
| `c2_pool` | `commit_consensus=2, consensus_pool=True` |

## Acceptance rule (decided before the numbers)

1. Run on the 22 held-out sequences (`scripts/gate_sweep.py --points consensus_*` mechanism or
   `run_target.py --ablate consensus` on the same protocol), then analyse with the
   select-on-one-half script. **No dev sequences.**
2. Ship only if the **paired** TSR difference vs the shipped reference has a CI excluding 0
   *and* the FAR paired difference does not significantly worsen *and* added latency per frame
   is under 15 %.
3. If TSR rises but FAR rises with it (same σ as the E070 axis), record it as another point on
   the same trade-off curve and do **not** ship — that is the outcome that would prove the
   gate framing wrong.
4. Report `challenge_abandoned` counts as the mechanism evidence: a mechanism that never
   abandons a challenge is inert (E023/E024/E025 class), whatever the metric says.

## Known traps to avoid (from this campaign's own ledger)

- **Latch confirmation must not extend the blind window**: while a challenge is pending, the
  incumbent's box is still the reported one, so `frames_since_update` semantics must be
  clarified (a pending challenge is *not* a miss — otherwise the state machine will cascade
  into `lost_short` and undo the point).
- **Consensus must key on identity, not on box stability**: two consecutive boxes of two
  different people at the same place would otherwise look like agreement.
- **Do not let `consensus_iou` become the new hidden margin**: the chaining threshold gates
  *agreement*, not *acceptance* — keep it low (0.15-0.30) and measure its effect separately.
- **First hair-cut rule**: if `challenge_committed` never fires across the sweep, the
  mechanism is mis-wired, not useless; check the counter before reading the metric.
