# E017 — does the ReID *metric* need fixing? (training-free transforms)

Verification on real YOLO26m detections matched to GT (IoU >= 0.5). `win` = fraction of queries whose best same-id crop is strictly closer than their best different-id crop (the quantity that actually drives FAR).

## dancetrack0005  (469 crops, 4 ids, 120 frames)

| transform | AUC | win | same mean | diff mean | margin p50 |
|---|---|---|---|---|---|
| `raw_l2` | **0.7499** | 0.996 | 0.489 | 0.602 | +0.2161 |
| `pnorm_l2` | **0.7444** | 0.994 | 0.504 | 0.612 | +0.2083 |
| `whiten` | **0.5467** | 0.962 | 0.995 | 1.004 | +0.1290 |
| `pnorm_whiten` | **0.6889** | 0.987 | 0.965 | 1.014 | +0.2124 |
| `whiten_s001` | **0.5274** | 0.881 | 1.000 | 1.003 | +0.0468 |
| `whiten_s050` | **0.5934** | 0.981 | 0.983 | 1.008 | +0.2552 |

## MOT17-02-FRCNN  (1565 crops, 36 ids, 120 frames)

| transform | AUC | win | same mean | diff mean | margin p50 |
|---|---|---|---|---|---|
| `raw_l2` | **0.9654** | 0.973 | 0.463 | 0.752 | +0.2256 |
| `pnorm_l2` | **0.9637** | 0.972 | 0.484 | 0.762 | +0.2168 |
| `whiten` | **0.6937** | 0.971 | 0.952 | 1.003 | +0.2183 |
| `pnorm_whiten` | **0.8847** | 0.965 | 0.885 | 1.006 | +0.2155 |
| `whiten_s001` | **0.6478** | 0.969 | 0.968 | 1.002 | +0.1821 |
| `whiten_s050` | **0.7991** | 0.974 | 0.908 | 1.005 | +0.2793 |
