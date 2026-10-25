# Unidentifiable target: `missing_appearance_cost` x similarity

The `unidentifiable_target` family makes the target's descriptor unusable
for a stretch of frames while an unoccluded bystander rides the stale
prediction.  `CostCfg.missing_appearance_cost` is the price charged to a
candidate whose appearance could not be read at all; the shipped 0.5 makes
it *cheaper* than a genuine mismatch (cosine 1.0 -> 1.6), so "we could not
read this person" can win the selection.  This grid sweeps that price.

- runs: 100 (10 scenarios x 10 configs), errors: 0
- handover = first frame whose *accepted* report sits closer (by IoU) to the
  bystander than to the target, starting from the frame the target stops
  being observed

## Where the handover happens

`n_dets=1` means the bystander was the only finite candidate, so
`association.select_candidate` left `margin = inf` and the ambiguity test
could not fire at any `distractor_margin`; `n_dets=2` means the runner-up
was present and priced.

| mode | config | n_dets=1 | n_dets=2 | never | in the gap | in the blind window |
|---|---|---|---|---|---|---|
| missing | imm-c000 | 3 | 0 | 1 | 3 | 0 |
| missing | imm-c010 | 3 | 0 | 1 | 3 | 0 |
| missing | imm-c025 | 3 | 0 | 1 | 3 | 0 |
| missing | imm-c050 | 3 | 0 | 1 | 3 | 0 |
| missing | imm-c100 | 3 | 0 | 1 | 3 | 0 |
| missing | kf-c000 | 3 | 0 | 1 | 3 | 0 |
| missing | kf-c010 | 3 | 0 | 1 | 3 | 0 |
| missing | kf-c025 | 3 | 0 | 1 | 3 | 0 |
| missing | kf-c050 | 3 | 0 | 1 | 3 | 0 |
| missing | kf-c100 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | imm-c000 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | imm-c010 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | imm-c025 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | imm-c050 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | imm-c100 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | kf-c000 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | kf-c010 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | kf-c025 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | kf-c050 | 3 | 0 | 1 | 3 | 0 |
| orthogonal | kf-c100 | 3 | 0 | 1 | 3 | 0 |

## model `imm`

### missing: target success rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.60 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |

### missing: false acquisition rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.60 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |

### missing: false-acquisition frames

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.60 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |

### missing: id consistency

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.60 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |

### missing: handover frame (first frame on the bystander)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.60 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.85 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |

### missing: n_dets at the handover

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### missing: median accepted margin inside the blind window

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 0.31 | 0.47 | 0.71 | 1.11 | 1.91 |
| 0.60 | 0.08 | 0.24 | 0.48 | 0.88 | 1.68 |
| 0.85 | 0.24 | 0.40 | 0.64 | 1.04 | 1.84 |

### orthogonal: target success rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 52.6 | 52.6 | 52.6 | 52.6 | 52.6 |
| 0.60 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |

### orthogonal: false acquisition rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 53.8 | 53.8 | 53.8 | 53.8 | 53.8 |
| 0.60 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |

### orthogonal: false-acquisition frames

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 42.00 | 42.00 | 42.00 | 42.00 | 42.00 |
| 0.60 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |

### orthogonal: id consistency

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 |
| 0.60 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |

### orthogonal: handover frame (first frame on the bystander)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.60 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.85 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |

### orthogonal: n_dets at the handover

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### orthogonal: median accepted margin inside the blind window

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

## model `kf`

### missing: target success rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.60 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |

### missing: false acquisition rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.60 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |

### missing: false-acquisition frames

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.60 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |

### missing: id consistency

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.60 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |

### missing: handover frame (first frame on the bystander)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.60 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.85 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |

### missing: n_dets at the handover

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### missing: median accepted margin inside the blind window

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 0.31 | 0.47 | 0.71 | 1.11 | 1.91 |
| 0.60 | 0.08 | 0.24 | 0.48 | 0.88 | 1.68 |
| 0.85 | 0.24 | 0.40 | 0.64 | 1.04 | 1.84 |

### orthogonal: target success rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 55.1 | 55.1 | 55.1 | 55.1 | 55.1 |
| 0.40 | 52.6 | 52.6 | 52.6 | 52.6 | 52.6 |
| 0.60 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |
| 0.85 | 29.5 | 29.5 | 29.5 | 29.5 | 29.5 |

### orthogonal: false acquisition rate (%)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 30.8 | 30.8 | 30.8 | 30.8 | 30.8 |
| 0.40 | 53.8 | 53.8 | 53.8 | 53.8 | 53.8 |
| 0.60 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |
| 0.85 | 76.9 | 76.9 | 76.9 | 76.9 | 76.9 |

### orthogonal: false-acquisition frames

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 24.00 | 24.00 | 24.00 | 24.00 | 24.00 |
| 0.40 | 42.00 | 42.00 | 42.00 | 42.00 | 42.00 |
| 0.60 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |
| 0.85 | 60.00 | 60.00 | 60.00 | 60.00 | 60.00 |

### orthogonal: id consistency

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.61 | 0.61 | 0.61 | 0.61 |
| 0.40 | 0.46 | 0.46 | 0.46 | 0.46 | 0.46 |
| 0.60 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| 0.85 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |

### orthogonal: handover frame (first frame on the bystander)

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.60 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |
| 0.85 | 18.00 | 18.00 | 18.00 | 18.00 | 18.00 |

### orthogonal: n_dets at the handover

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### orthogonal: median accepted margin inside the blind window

| similarity | c000 | c010 | c025 | c050 | c100 |
|---|---|---|---|---|---|
| 0.00 | - | - | - | - | - |
| 0.40 | 1.03 | 1.03 | 1.03 | 1.03 | 1.03 |
| 0.60 | 1.27 | 1.27 | 1.27 | 1.27 | 1.27 |
| 0.85 | 1.34 | 1.34 | 1.34 | 1.34 | 1.34 |

## Findings

- handovers split by n_dets: n=1 60 run(s) (lone finite candidate, margin=inf so the ambiguity test is vacuous), n=2 0 run(s) (runner-up present and priced), never 20 run(s)
- missing/imm at similarity 0.85 - c000: TSR 29.5% / FAR 76.9%; c010: TSR 29.5% / FAR 76.9%; c025: TSR 29.5% / FAR 76.9%; c050: TSR 29.5% / FAR 76.9%; c100: TSR 29.5% / FAR 76.9%
- missing/imm: best false-acquisition rate over the whole price axis = 76.9% -> pricing alone does NOT close the handover
- orthogonal/imm at similarity 0.85 - c000: TSR 29.5% / FAR 76.9%; c010: TSR 29.5% / FAR 76.9%; c025: TSR 29.5% / FAR 76.9%; c050: TSR 29.5% / FAR 76.9%; c100: TSR 29.5% / FAR 76.9%
- orthogonal/imm: best false-acquisition rate over the whole price axis = 76.9% -> pricing alone does NOT close the handover
- missing/kf at similarity 0.85 - c000: TSR 29.5% / FAR 76.9%; c010: TSR 29.5% / FAR 76.9%; c025: TSR 29.5% / FAR 76.9%; c050: TSR 29.5% / FAR 76.9%; c100: TSR 29.5% / FAR 76.9%
- missing/kf: best false-acquisition rate over the whole price axis = 76.9% -> pricing alone does NOT close the handover
- orthogonal/kf at similarity 0.85 - c000: TSR 29.5% / FAR 76.9%; c010: TSR 29.5% / FAR 76.9%; c025: TSR 29.5% / FAR 76.9%; c050: TSR 29.5% / FAR 76.9%; c100: TSR 29.5% / FAR 76.9%
- orthogonal/kf: best false-acquisition rate over the whole price axis = 76.9% -> pricing alone does NOT close the handover
