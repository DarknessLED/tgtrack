# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 3 (MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-13-FRCNN)
- workers: jobs=3 mode=process-pool wall=39.0s
- measured scaling: sum(pair wall)=101.0s / wall=39.0s -> speedup 2.59x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 3 | 0.4473 | 0.4285 | 0.4718 | 0.8017 | 0.4558 | 0.5837 | 0.7073 | 0.4968 | 796 | 2862 | 120.0000 | 266.0000 | 42.0000 | 52.0000 | 56.23 | 17.79 |
| hybridsort.yaml | 3 | 0.4377 | 0.4438 | 0.4380 | 0.7980 | 0.4401 | 0.5526 | 0.6305 | 0.4918 | 1077 | 2604 | 206.0000 | 312.0000 | 57.0000 | 36.0000 | 80.54 | 12.42 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-09-FRCNN | 0.5217 | 0.5378 | 0.5077 | 0.8344 | 0.5719 | 0.6628 | 232 | 498 | 29.0000 | 54.0000 | 59.61 | 16.78 |
| hybridsort.yaml | MOT17-09-FRCNN | 0.4979 | 0.5373 | 0.4632 | 0.8286 | 0.5561 | 0.6056 | 288 | 459 | 40.0000 | 56.0000 | 89.19 | 11.21 |
| hybridsort.yaml | MOT17-13-FRCNN | 0.4409 | 0.4256 | 0.4640 | 0.8029 | 0.4732 | 0.5671 | 273 | 1518 | 89.0000 | 148.0000 | 73.54 | 13.60 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4404 | 0.4037 | 0.4853 | 0.8051 | 0.4794 | 0.5810 | 180 | 1635 | 43.0000 | 119.0000 | 51.55 | 19.40 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3655 | 0.3687 | 0.3663 | 0.7478 | 0.2744 | 0.4981 | 384 | 729 | 48.0000 | 93.0000 | 57.94 | 17.26 |
| hybridsort.yaml | MOT17-05-FRCNN | 0.3566 | 0.3889 | 0.3314 | 0.7478 | 0.2375 | 0.4668 | 516 | 627 | 77.0000 | 108.0000 | 79.98 | 12.50 |
