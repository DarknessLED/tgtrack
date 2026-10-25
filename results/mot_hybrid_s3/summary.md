# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 3 (MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-13-FRCNN)
- workers: jobs=3 mode=process-pool wall=40.3s
- measured scaling: sum(pair wall)=104.1s / wall=40.3s -> speedup 2.58x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 3 | 0.4473 | 0.4285 | 0.4718 | 0.8017 | 0.4558 | 0.5837 | 0.7073 | 0.4968 | 796 | 2862 | 120.0000 | 266.0000 | 42.0000 | 52.0000 | 56.22 | 17.79 |
| hybridsort.yaml | 3 | 0.4268 | 0.4423 | 0.4180 | 0.7990 | 0.4448 | 0.5373 | 0.6206 | 0.4738 | 999 | 2641 | 214.0000 | 303.0000 | 52.0000 | 37.0000 | 81.94 | 12.20 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-09-FRCNN | 0.5217 | 0.5378 | 0.5077 | 0.8344 | 0.5719 | 0.6628 | 232 | 498 | 29.0000 | 54.0000 | 59.24 | 16.88 |
| hybridsort.yaml | MOT17-09-FRCNN | 0.4706 | 0.5375 | 0.4145 | 0.8295 | 0.5606 | 0.5596 | 272 | 462 | 45.0000 | 56.0000 | 90.18 | 11.09 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4404 | 0.4037 | 0.4853 | 0.8051 | 0.4794 | 0.5810 | 180 | 1635 | 43.0000 | 119.0000 | 51.74 | 19.33 |
| hybridsort.yaml | MOT17-13-FRCNN | 0.4315 | 0.4222 | 0.4478 | 0.8043 | 0.4727 | 0.5530 | 246 | 1540 | 96.0000 | 142.0000 | 74.49 | 13.43 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3655 | 0.3687 | 0.3663 | 0.7478 | 0.2744 | 0.4981 | 384 | 729 | 48.0000 | 93.0000 | 58.06 | 17.22 |
| hybridsort.yaml | MOT17-05-FRCNN | 0.3629 | 0.3888 | 0.3446 | 0.7480 | 0.2544 | 0.4826 | 481 | 639 | 73.0000 | 105.0000 | 82.19 | 12.17 |
