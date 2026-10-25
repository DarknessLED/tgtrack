# Partial-view appearance experiment

Gallery identities: **3**.  Mask = the region painted out of the query crop.
The tracker is *not* told which mask was applied; `quality_w`/`visible_gate` estimate
the still-visible regions from image structure alone (Laplacian energy).

## Control: does each part carry identity evidence?

Same-identity vs different-identity mean cosine on **clean** crops. A gap near
zero means that part is zero-evidence and must be kept out of any fused score.

| part | same-id cos | diff-id cos | gap | n |
|---|---|---|---|---|
| full | +0.9110 | +0.7172 | +0.1938 | 44 |
| upper | +0.9332 | +0.7647 | +0.1684 | 44 |
| lower | +0.9205 | +0.7467 | +0.1738 | 44 |
| head | +0.8651 | +0.5679 | +0.2971 | 44 |

## Measured region activity (0 = flat/occluded, 1 = fully structured)

| mask | full | upper | lower | head |
|---|---|---|---|---|
| none | 0.867 | 0.861 | 0.893 | 0.819 |
| lower | 0.427 | 0.714 | 0.093 | 0.819 |
| upper | 0.448 | 0.160 | 0.814 | 0.000 |
| lower60 | 0.337 | 0.564 | 0.000 | 0.819 |
| upper60 | 0.357 | 0.000 | 0.648 | 0.000 |
| left | 0.447 | 0.440 | 0.462 | 0.418 |
| right | 0.436 | 0.436 | 0.447 | 0.427 |

## Rank-1 identity accuracy

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | 0.886 | 1.000 | 1.000 | 1.000 | 1.000 |
| lower | 0.545 | 1.000 | 0.977 | 1.000 | 1.000 |
| upper | 0.886 | 1.000 | 0.886 | 0.886 | 0.886 |
| lower60 | 0.409 | 1.000 | 0.977 | 0.977 | 0.977 |
| upper60 | 0.909 | 1.000 | 0.977 | 0.955 | 0.955 |
| left | 0.182 | 0.705 | 0.727 | 0.705 | 0.705 |
| right | 1.000 | 1.000 | 0.841 | 1.000 | 1.000 |

## Mean margin (correct minus best wrong)

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | +0.1938 | +0.2097 | +0.1640 | +0.2080 | +0.2097 |
| lower | +0.0552 | +0.1299 | +0.2960 | +0.1702 | +0.1550 |
| upper | +0.1473 | +0.0902 | +0.1466 | +0.1393 | +0.1492 |
| lower60 | +0.0286 | +0.1027 | +0.2969 | +0.1747 | +0.1408 |
| upper60 | +0.0755 | +0.0520 | +0.1161 | +0.1023 | +0.0963 |
| left | -0.0261 | +0.0444 | +0.0323 | +0.0419 | +0.0444 |
| right | +0.1996 | +0.1639 | +0.1402 | +0.1631 | +0.1639 |
