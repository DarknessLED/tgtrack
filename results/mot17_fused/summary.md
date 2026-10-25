# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=88.9s
- measured scaling: sum(pair wall)=159.5s / wall=88.9s -> speedup 1.79x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4944 | 0.4706 | 0.5264 | 0.8013 | 0.4954 | 0.6268 | 0.6794 | 0.5817 | 4826 | 8877 | 520.0000 | 981.0000 | 146.0000 | 76.0000 | 89.19 | 11.21 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.5864 | 0.5614 | 0.6190 | 0.8050 | 0.6410 | 0.7517 | 1252 | 1888 | 63.0000 | 126.0000 | 101.46 | 9.86 |
| tracktrack.yaml | MOT17-11-FRCNN | 0.5523 | 0.5089 | 0.6030 | 0.8461 | 0.4852 | 0.6414 | 592 | 486 | 16.0000 | 38.0000 | 97.67 | 10.24 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4957 | 0.4925 | 0.5032 | 0.7862 | 0.5520 | 0.6583 | 510 | 1205 | 80.0000 | 220.0000 | 102.84 | 9.72 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.4845 | 0.5207 | 0.4527 | 0.8267 | 0.5477 | 0.5928 | 327 | 425 | 50.0000 | 58.0000 | 85.53 | 11.69 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4316 | 0.4202 | 0.4508 | 0.7850 | 0.4200 | 0.5658 | 674 | 1276 | 120.0000 | 166.0000 | 72.10 | 13.87 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3839 | 0.3620 | 0.4147 | 0.7992 | 0.3438 | 0.4849 | 967 | 2959 | 135.0000 | 266.0000 | 92.59 | 10.80 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3543 | 0.3652 | 0.3473 | 0.7483 | 0.2512 | 0.4788 | 504 | 638 | 56.0000 | 107.0000 | 71.65 | 13.96 |
