# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 2 (MOT17-02-FRCNN, MOT17-04-FRCNN)
- workers: jobs=2 mode=process-pool wall=46.1s
- measured scaling: sum(pair wall)=80.5s / wall=46.1s -> speedup 1.74x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=0.05, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytetrack.yaml | 2 | 0.5072 | 0.4478 | 0.5789 | 0.8385 | 0.4563 | 0.6053 | 0.7178 | 0.5232 | 2000 | 6097 | 119.0000 | 247.0000 | 37.0000 | 31.0000 | 34.79 | 28.75 |
| botsort.yaml | 2 | 0.5042 | 0.4487 | 0.5707 | 0.8383 | 0.4571 | 0.6047 | 0.7157 | 0.5235 | 2021 | 6079 | 104.0000 | 238.0000 | 38.0000 | 30.0000 | 48.96 | 20.42 |
| tracktrack.yaml | 2 | 0.5022 | 0.4537 | 0.5602 | 0.8343 | 0.4449 | 0.6080 | 0.6969 | 0.5392 | 2431 | 5850 | 108.0000 | 277.0000 | 40.0000 | 27.0000 | 53.31 | 18.76 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytetrack.yaml | MOT17-04-FRCNN | 0.5857 | 0.5331 | 0.6469 | 0.8429 | 0.5745 | 0.6986 | 1073 | 2684 | 40.0000 | 92.0000 | 35.53 | 28.14 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.5849 | 0.5433 | 0.6332 | 0.8388 | 0.5768 | 0.7086 | 1222 | 2530 | 24.0000 | 103.0000 | 52.81 | 18.94 |
| botsort.yaml | MOT17-04-FRCNN | 0.5787 | 0.5341 | 0.6304 | 0.8423 | 0.5749 | 0.6895 | 1096 | 2664 | 33.0000 | 86.0000 | 51.45 | 19.44 |
| botsort.yaml | MOT17-02-FRCNN | 0.3745 | 0.3283 | 0.4342 | 0.8293 | 0.2873 | 0.4650 | 925 | 3415 | 71.0000 | 152.0000 | 46.48 | 21.52 |
| bytetrack.yaml | MOT17-02-FRCNN | 0.3693 | 0.3277 | 0.4238 | 0.8287 | 0.2860 | 0.4521 | 927 | 3413 | 79.0000 | 155.0000 | 34.04 | 29.37 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3587 | 0.3306 | 0.3951 | 0.8244 | 0.2546 | 0.4461 | 1209 | 3320 | 84.0000 | 174.0000 | 53.81 | 18.58 |
