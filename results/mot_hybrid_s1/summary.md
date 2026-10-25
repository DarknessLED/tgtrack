# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 3 (MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-13-FRCNN)
- workers: jobs=3 mode=process-pool wall=39.8s
- measured scaling: sum(pair wall)=103.1s / wall=39.8s -> speedup 2.59x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 3 | 0.4473 | 0.4285 | 0.4718 | 0.8017 | 0.4558 | 0.5837 | 0.7073 | 0.4968 | 796 | 2862 | 120.0000 | 266.0000 | 42.0000 | 52.0000 | 57.04 | 17.53 |
| hybridsort.yaml | 3 | 0.4162 | 0.4285 | 0.4100 | 0.7999 | 0.4326 | 0.5272 | 0.6275 | 0.4545 | 919 | 2833 | 187.0000 | 281.0000 | 42.0000 | 50.0000 | 81.29 | 12.30 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-09-FRCNN | 0.5217 | 0.5378 | 0.5077 | 0.8344 | 0.5719 | 0.6628 | 232 | 498 | 29.0000 | 54.0000 | 60.53 | 16.52 |
| hybridsort.yaml | MOT17-09-FRCNN | 0.4544 | 0.5325 | 0.3898 | 0.8309 | 0.5539 | 0.5393 | 257 | 491 | 43.0000 | 54.0000 | 89.58 | 11.16 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4404 | 0.4037 | 0.4853 | 0.8051 | 0.4794 | 0.5810 | 180 | 1635 | 43.0000 | 119.0000 | 53.35 | 18.74 |
| hybridsort.yaml | MOT17-13-FRCNN | 0.4265 | 0.4048 | 0.4561 | 0.8051 | 0.4576 | 0.5500 | 214 | 1643 | 79.0000 | 128.0000 | 74.05 | 13.51 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3655 | 0.3687 | 0.3663 | 0.7478 | 0.2744 | 0.4981 | 384 | 729 | 48.0000 | 93.0000 | 57.68 | 17.34 |
| hybridsort.yaml | MOT17-05-FRCNN | 0.3454 | 0.3726 | 0.3252 | 0.7467 | 0.2425 | 0.4693 | 448 | 699 | 65.0000 | 99.0000 | 81.27 | 12.30 |
