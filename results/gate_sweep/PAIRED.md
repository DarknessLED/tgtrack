# Paired per-sequence analysis: `margin_0.00` vs `margin_0.20`

Sequences compared: **22**

| sequence | TSR ref | TSR cmp | dTSR (pp) | FAR ref | FAR cmp | dFAR (pp) | class |
|---|---|---|---|---|---|---|---|
| dancetrack0010 | 0.283 | 0.301 | +1.8 | 0.680 | 0.680 | +0.0 | FREE WIN |
| dancetrack0014 | 0.069 | 0.074 | +0.5 | 0.657 | 0.760 | +10.3 | trade-off up |
| dancetrack0018 | 0.258 | 0.266 | +0.8 | 0.587 | 0.532 | -5.6 | FREE WIN |
| dancetrack0019 | 0.219 | 0.330 | +11.1 | 0.037 | 0.273 | +23.7 | trade-off up |
| dancetrack0025 | 0.015 | 0.015 | +0.0 | 0.060 | 0.570 | +51.0 | free loss |
| dancetrack0026 | 0.081 | 0.137 | +5.6 | 0.053 | 0.457 | +40.4 | trade-off up |
| dancetrack0030 | 0.370 | 0.403 | +3.3 | 0.017 | 0.357 | +34.0 | trade-off up |
| dancetrack0034 | 0.181 | 0.222 | +4.0 | 0.155 | 0.186 | +3.1 | trade-off up |
| dancetrack0035 | 0.022 | 0.022 | +0.0 | 0.057 | 0.471 | +41.4 | free loss |
| dancetrack0041 | 0.174 | 0.462 | +28.7 | 0.047 | 0.250 | +20.3 | trade-off up |
| dancetrack0043 | 0.111 | 0.111 | +0.0 | 0.043 | 0.163 | +12.0 | free loss |
| dancetrack0047 | 0.146 | 0.192 | +4.6 | 0.470 | 0.453 | -1.7 | FREE WIN |
| dancetrack0058 | 0.047 | 0.044 | -0.3 | 0.340 | 0.763 | +42.3 | free loss |
| dancetrack0063 | 0.030 | 0.110 | +8.0 | 0.157 | 0.333 | +17.7 | trade-off up |
| dancetrack0065 | 0.047 | 0.077 | +3.0 | 0.027 | 0.460 | +43.3 | trade-off up |
| dancetrack0073 | 0.228 | 0.228 | +0.0 | 0.583 | 0.607 | +2.3 | free loss |
| dancetrack0077 | 0.233 | 0.263 | +3.0 | 0.410 | 0.403 | -0.7 | FREE WIN |
| dancetrack0079 | 0.913 | 0.913 | +0.0 | 0.023 | 0.023 | +0.0 | flat |
| dancetrack0081 | 0.027 | 0.023 | -0.5 | 0.200 | 0.517 | +31.7 | free loss |
| dancetrack0090 | 0.004 | 0.030 | +2.6 | 0.554 | 0.668 | +11.4 | trade-off up |
| dancetrack0094 | 0.880 | 0.880 | +0.0 | 0.097 | 0.097 | +0.0 | flat |
| dancetrack0097 | 0.160 | 0.160 | +0.0 | 0.000 | 0.000 | +0.0 | flat |

## Classification of the 22 sequences

| class | meaning | count |
|---|---|---|
| **FREE WIN** | TSR up, FAR not worse | **4** |
| free loss | TSR not better, FAR worse | 6 |
| trade-off up | both up (the gate trade-off) | 9 |
| trade-off down | both down | 0 |
| flat | both within 1 pp | 3 |

## Pooled (frame-weighted over the compared sequences)

| metric | ref | cmp | delta |
|---|---|---|---|
| TSR | 0.2104 | 0.2460 | **+3.56 pp** |
| FAR | 0.2472 | 0.4166 | **+16.94 pp** |
| frames behind the pooled number | 6181 | | |

Rank correlation between dTSR and dFAR across sequences: **rho = -0.029**

**Reading: 4 sequence(s) gain TSR at no FAR cost**, so the trade-off is NOT uniform and a per-sequence predictor is a candidate cheaper-than-E071 win. Inspect those rows before building anything: with 22 sequences a couple of free wins can be frame-count noise.
