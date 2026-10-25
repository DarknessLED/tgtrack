# Partial-view appearance experiment

Gallery identities: **25**.  Mask = the region painted out of the query crop.
The tracker is *not* told which mask was applied; `quality_w`/`visible_gate` estimate
the still-visible regions from image structure alone (Laplacian energy).

## Control: does each part carry identity evidence?

Same-identity vs different-identity mean cosine on **clean** crops. A gap near
zero means that part is zero-evidence and must be kept out of any fused score.

| part | same-id cos | diff-id cos | gap | n |
|---|---|---|---|---|
| full | +0.6664 | +0.4632 | +0.2032 | 414 |
| upper | +0.6402 | +0.4249 | +0.2153 | 414 |
| lower | +0.6073 | +0.4088 | +0.1985 | 414 |
| head | +0.5860 | +0.4097 | +0.1763 | 414 |

## Measured region activity (0 = flat/occluded, 1 = fully structured)

| mask | full | upper | lower | head |
|---|---|---|---|---|
| none | 0.678 | 0.693 | 0.650 | 0.733 |
| lower | 0.359 | 0.600 | 0.069 | 0.733 |
| upper | 0.330 | 0.111 | 0.600 | 0.000 |
| lower60 | 0.293 | 0.490 | 0.000 | 0.733 |
| upper60 | 0.270 | 0.000 | 0.490 | 0.000 |
| left | 0.354 | 0.361 | 0.339 | 0.383 |
| right | 0.346 | 0.353 | 0.332 | 0.380 |

## Rank-1 identity accuracy

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | 0.903 | 0.983 | 0.928 | 0.983 | 0.983 |
| lower | 0.792 | 0.932 | 0.884 | 0.957 | 0.923 |
| upper | 0.681 | 0.807 | 0.845 | 0.870 | 0.855 |
| lower60 | 0.669 | 0.889 | 0.920 | 0.942 | 0.930 |
| upper60 | 0.483 | 0.524 | 0.606 | 0.674 | 0.664 |
| left | 0.710 | 0.833 | 0.742 | 0.831 | 0.831 |
| right | 0.696 | 0.867 | 0.819 | 0.860 | 0.836 |

## Mean margin (correct minus best wrong)

| mask | full | mean_parts | max_part | quality_w | visible_gate |
|---|---|---|---|---|---|
| none | +0.2032 | +0.2244 | +0.1991 | +0.2239 | +0.2244 |
| lower | +0.0762 | +0.1358 | +0.1573 | +0.1664 | +0.1592 |
| upper | +0.0432 | +0.0480 | +0.1132 | +0.1027 | +0.1033 |
| lower60 | +0.0429 | +0.1023 | +0.1594 | +0.1555 | +0.1388 |
| upper60 | -0.0044 | +0.0032 | +0.0302 | +0.0383 | +0.0390 |
| left | +0.0547 | +0.0862 | +0.0724 | +0.0861 | +0.0807 |
| right | +0.0747 | +0.1178 | +0.1019 | +0.1174 | +0.1138 |
