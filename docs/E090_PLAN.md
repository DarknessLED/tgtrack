# E090 - relax the ambiguity margin while the target is lost (preregistered)

Status: preregistered before any measurement. Follows E089 (shipped: the identity ceiling is scaled
while the target is lost).

## Motivation, from measurement

E088 attributed the shipped target loss frame by frame (3 of the worst held-out sequences, shipped
regime, reference summaries reproduced exactly). The refusal reasons on the frames where a candidate
existed and the target was visible:

- 41 % `all_gated:anchor_ceiling` -> addressed and shipped by E089 (band x1.25 while lost),
- most of the remainder `ambiguous: margin X < 0.060` -> **not addressed by anything yet**.

The same argument that justified E089 applies verbatim to the ambiguity gate: while the target is lost
there is no identity left to protect, yet the gate keeps demanding a 0.06 margin over the runner-up.
The knob already exists and is inert:

`cost.distractor_margin_lost: float = 0.0` (`src/tgtrack/config.py:422`), consulted in
`src/tgtrack/association.py:707` and `:799`:

```python
margin_thr = float(cfg.distractor_margin)          # 0.06
if lost:
    override = float(getattr(cfg, "distractor_margin_lost", 0.0) or 0.0)
    if override > 0:
        margin_thr = override
```

`0.0` means "disabled, fall back to 0.06". A **value below 0.06 relaxes** the gate while lost; the
existing field semantics allow it and nothing in the tree ships a value below 0.06.

## Design (pre-registered)

- Epoch: the **shipped** epoch (E089: `anchor_band_lost_scale = 1.25`). Baseline = the already measured
  `ceil_lost_1.25` records (22 sequences, `results/vote_sweep/runs/ceil_lost_1.25/`), which *are* the
  shipped configuration - no re-run of the baseline is needed.
- Arms (measured on the same 22 held-out sequences, 300 frames, subsample 2, mask none):
  - `margin_lost_0.03` - `cost.distractor_margin_lost = 0.03`
  - `margin_lost_0.01` - `cost.distractor_margin_lost = 0.01`
- The sweep harness still needs `vote_off` in `--arms` (it pins target ids); `vote_off` records are
  already complete and are not re-measured.

## Step 0 - wiring check BEFORE interpreting anything

On `dancetrack0097` and `dancetrack0034` (the two sequences with the heaviest `ambiguous:` refusal
counts in E088), run the shipped configuration and `0.01` with `--debug-csv` and count
`refuse_reason`:

- **required**: the number of `ambiguous:` refusals falls by **>= 30 %** at `0.01` on at least one of
  the two sequences, and
- **required**: on a sequence where no `ambiguous:` refusal ever occurs, the run is bit-identical.

If the bound is not met, the arms are inert and E090 stops here (same discipline as E089 step 0).

## Predictions (scored honestly, before looking)

- P1: pooled TSR rises **>= +2.0 pp** vs the shipped baseline.
- P2: pooled FAR rises by **<= +0.02** (relaxing an ambiguity gate can admit the runner-up).
- P3: `id_consistency` drops by **<= 0.05** on sequences where the baseline was >= 0.90.
- P4: the gain concentrates on sequences whose E088 `ambiguous:` share is high (`0034`, `0073`, `0077`,
  `0097`), i.e. this is a different population from E089's gain set (`0018/0034/0081/0077`).

## Ship rule (pre-committed)

Ship the largest scale that satisfies **all** of:

1. paired per-sequence dTSR vs the shipped baseline, 22 sequences, 95 % cluster-bootstrap CI excludes 0
   (the campaign rule);
2. P2 and P3 hold;
3. the criterion-3 guard is bit-identical (`MOT17-02-FRCNN`, `--mask upper`, target 19, 300 frames:
   TSR 0.8545 / FAR 0.0000 / idcons 1.0000);
4. no held-out sequence loses more than 2.0 pp of TSR.

Otherwise every arm is KILLed and the default stays at `distractor_margin_lost = 0.0`.

## Cost

44 pairs, ~96 s/pair -> ~70 min on the 4060.
