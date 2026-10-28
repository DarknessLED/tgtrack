# E077 — channel-conditional fusion + bounded-aspect coasting (pre-registered)

Status: PRE-REGISTERED, not implemented. Full reasoning and numbers: `docs/EXPERIMENTS.md` E077-PREP
and E077 sections. Ship rule is written there and must not be edited after the run.

## Why these two and not a new weight
- E076: the neural appearance term denies a candidate whose detection has IoU 0.968 with the target
  (`scripts/probe_recovery7.py`), and `--no-embed` keeps `tracking` for 17 more frames.
- E013: DanceTrack appearance AUC 0.59-0.67, i.e. barely informative. A fixed `w_appearance = 1.6`
  in an additive cost must over-refuse on this dataset.
- `scripts/probe_recoverable.py`: 29.9 % of frames are blind while the target is present; only
  3.8 pp are `lost_long`. The prize is short-term recovery.
- `scripts/probe_box_convention.py`: the IoU threshold and box convention are NOT the issue
  (+2.4 pp at IoU 0.3, -0.2 pp when widening the box), so the metric does measure identity.

## Arm B — channel-conditional acceptance
Inside `src/tgtrack/association.py`, the additive `total` remains the ranking score, but acceptance
gains an OR-branch:

    accept if (total <= bound) or (geometry_conclusive) or (appearance_conclusive)
      geometry_conclusive  : norm_centre_dist <= g_tau  AND  |log(h_ratio)| <= 0.25
      appearance_conclusive: app_cost <= a_tau     AND  margin >= distractor_margin
    refusal conditions (ambiguous / lone / lost margin) still apply on top.

`g_tau` / `a_tau` are the only two new numbers and both come from already-measured distributions
(median correct-frame geometry cost, median correct-frame appearance cost), not from a sweep.

## Arm C — bounded-aspect coast
In `src/tgtrack/motion.py::coast`, after the predict step, clamp the state's aspect `a = w/h` to
`[aspect_min_ratio, aspect_max_ratio] * a_last_observed` and zero the aspect velocity if it was
clamped. This reuses the shape-prior bookkeeping added in E014b (which was rejected *as an
acceptance-time prior*, a different use).

## Arms and protocol
A = reference (shipped). B = fusion. C = aspect clamp. D = both.
22 held-out DanceTrack-val sequences (dev 0004/0005/0007 excluded and asserted absent), 300 frames,
subsample 2, `--mask none`, `--jobs 1`, bootstrap 2000 reps seed 0, paired deltas vs A.

## Ship rule (fixed now)
Ship an arm only if pooled TSR >= +3 pp with the paired 95 % CI excluding 0, the paired dFAR CI
upper bound < +3 pp, and `blind AND present` share falls (mechanism engaged). Kill on bit-identical
or on a TSR drop.
