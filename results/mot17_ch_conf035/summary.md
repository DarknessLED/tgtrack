# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=43.0s
- measured scaling: sum(pair wall)=77.7s / wall=43.0s -> speedup 1.81x on 32 logical CPUs
- pipeline: weights=data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=0.35, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4985 | 0.4718 | 0.5320 | 0.8014 | 0.5713 | 0.6486 | 0.7940 | 0.5483 | 1548 | 10270 | 266.0000 | 912.0000 | 125.0000 | 107.0000 | 38.46 | 26.00 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6411 | 0.6005 | 0.6860 | 0.8617 | 0.6889 | 0.7620 | 104 | 551 | 6.0000 | 38.0000 | 40.08 | 24.95 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.6003 | 0.5693 | 0.6368 | 0.7990 | 0.7203 | 0.8050 | 481 | 1984 | 31.0000 | 130.0000 | 43.69 | 22.89 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5423 | 0.5792 | 0.5087 | 0.8224 | 0.6994 | 0.6975 | 91 | 417 | 25.0000 | 56.0000 | 40.15 | 24.91 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4495 | 0.4693 | 0.4325 | 0.8013 | 0.5780 | 0.5987 | 158 | 1488 | 45.0000 | 206.0000 | 42.74 | 23.40 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4159 | 0.3866 | 0.4541 | 0.7773 | 0.4559 | 0.5523 | 253 | 1640 | 49.0000 | 164.0000 | 36.12 | 27.68 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3485 | 0.3478 | 0.3515 | 0.7935 | 0.4112 | 0.4594 | 261 | 3314 | 69.0000 | 224.0000 | 39.48 | 25.33 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3290 | 0.3258 | 0.3394 | 0.7471 | 0.3019 | 0.4540 | 200 | 876 | 41.0000 | 94.0000 | 27.20 | 36.76 |
