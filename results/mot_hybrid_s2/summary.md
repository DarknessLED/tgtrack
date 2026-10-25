# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 3 (MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-13-FRCNN)
- workers: jobs=3 mode=process-pool wall=39.4s
- measured scaling: sum(pair wall)=101.8s / wall=39.4s -> speedup 2.58x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 3 | 0.4473 | 0.4285 | 0.4718 | 0.8017 | 0.4558 | 0.5837 | 0.7073 | 0.4968 | 796 | 2862 | 120.0000 | 266.0000 | 42.0000 | 52.0000 | 55.21 | 18.11 |
| hybridsort.yaml | 3 | 0.4347 | 0.4425 | 0.4334 | 0.7982 | 0.4412 | 0.5507 | 0.6315 | 0.4882 | 1048 | 2623 | 208.0000 | 315.0000 | 53.0000 | 37.0000 | 80.52 | 12.42 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-09-FRCNN | 0.5217 | 0.5378 | 0.5077 | 0.8344 | 0.5719 | 0.6628 | 232 | 498 | 29.0000 | 54.0000 | 58.15 | 17.20 |
| hybridsort.yaml | MOT17-09-FRCNN | 0.4888 | 0.5382 | 0.4458 | 0.8288 | 0.5606 | 0.5900 | 279 | 459 | 41.0000 | 58.0000 | 89.59 | 11.16 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4404 | 0.4037 | 0.4853 | 0.8051 | 0.4794 | 0.5810 | 180 | 1635 | 43.0000 | 119.0000 | 50.49 | 19.80 |
| hybridsort.yaml | MOT17-13-FRCNN | 0.4396 | 0.4237 | 0.4633 | 0.8036 | 0.4716 | 0.5667 | 264 | 1529 | 93.0000 | 146.0000 | 74.37 | 13.45 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3655 | 0.3687 | 0.3663 | 0.7478 | 0.2744 | 0.4981 | 384 | 729 | 48.0000 | 93.0000 | 57.34 | 17.44 |
| hybridsort.yaml | MOT17-05-FRCNN | 0.3579 | 0.3866 | 0.3369 | 0.7468 | 0.2412 | 0.4769 | 505 | 635 | 74.0000 | 111.0000 | 78.74 | 12.70 |
