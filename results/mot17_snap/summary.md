# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=75.2s
- measured scaling: sum(pair wall)=136.7s / wall=75.2s -> speedup 1.82x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4966 | 0.4748 | 0.5258 | 0.8204 | 0.5093 | 0.6095 | 0.7159 | 0.5306 | 3054 | 10350 | 428.0000 | 1063.0000 | 122.0000 | 90.0000 | 76.22 | 13.12 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6183 | 0.5487 | 0.7019 | 0.8485 | 0.5567 | 0.7220 | 264 | 669 | 9.0000 | 30.0000 | 75.56 | 13.24 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.5974 | 0.5715 | 0.6285 | 0.8357 | 0.6361 | 0.7188 | 869 | 2331 | 47.0000 | 169.0000 | 87.97 | 11.37 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.4795 | 0.5569 | 0.4151 | 0.8333 | 0.6283 | 0.5694 | 173 | 449 | 37.0000 | 70.0000 | 83.07 | 12.04 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4729 | 0.4716 | 0.4769 | 0.7927 | 0.5371 | 0.6301 | 341 | 1455 | 59.0000 | 237.0000 | 77.20 | 12.95 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4268 | 0.4223 | 0.4389 | 0.7965 | 0.4584 | 0.5518 | 449 | 1404 | 80.0000 | 175.0000 | 56.28 | 17.77 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3574 | 0.3556 | 0.3667 | 0.8227 | 0.3324 | 0.4485 | 723 | 3247 | 162.0000 | 288.0000 | 82.06 | 12.19 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3555 | 0.3568 | 0.3575 | 0.7562 | 0.3350 | 0.4992 | 235 | 795 | 34.0000 | 94.0000 | 72.30 | 13.83 |
