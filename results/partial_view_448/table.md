# Partial-view appearance experiment

Gallery identities: **3**.  Mask = the region painted out of the query crop.
The tracker is *not* told which mask was applied; `quality_w`/`visible_gate` estimate
the still-visible regions from image structure alone (Laplacian energy).

## Control: does each part carry identity evidence?

Same-identity vs different-identity mean cosine on **clean** crops. A gap near
zero means that part is zero-evidence and must be kept out of any fused score.

| part | same-id cos | diff-id cos | gap | n |
|---|---|---|---|---|
| full | +0.9286 | +0.7613 | +0.1673 | 44 |
| upper | +0.8942 | +0.7358 | +0.1584 | 44 |
| lower | +0.9331 | +0.7617 | +0.1714 | 44 |
| head | +0.8857 | +0.6118 | +0.2738 | 44 |

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
| none | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lower | 0.614 | 1.000 | 1.000 | 1.000 | 1.000 |
| upper | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| lower60 | 0.545 | 1.000 | 1.000 | 1.000 | 1.000 |
| upper60 | 0.909 | 1.000 | 1.000 | 1.000 | 1.000 |
| left | 0.659 | 0.932 | 0.932 | 0.932 | 0.932 |
| right | 0.955 | 1.000 | 0.818 | 0.977 | 1.000 |

## Mean margin (correct minus best wrong)

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | +0.1673 | +0.1933 | +0.1610 | +0.1917 | +0.1933 |
| lower | +0.0571 | +0.1251 | +0.2597 | +0.1648 | +0.1539 |
| upper | +0.1033 | +0.0761 | +0.1567 | +0.1270 | +0.1302 |
| lower60 | +0.0448 | +0.1038 | +0.2654 | +0.1663 | +0.1336 |
| upper60 | +0.0778 | +0.0570 | +0.1310 | +0.1129 | +0.1051 |
| left | +0.0785 | +0.1064 | +0.1088 | +0.1059 | +0.1064 |
| right | +0.0883 | +0.1031 | +0.0991 | +0.1027 | +0.1031 |
