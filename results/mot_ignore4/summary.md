# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=50.3s
- measured scaling: sum(pair wall)=90.3s / wall=50.3s -> speedup 1.80x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4980 | 0.4609 | 0.5431 | 0.8246 | 0.4950 | 0.6125 | 0.7513 | 0.5170 | 2572 | 11363 | 299.0000 | 840.0000 | 115.0000 | 103.0000 | 49.68 | 20.13 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.6006 | 0.5490 | 0.6597 | 0.8449 | 0.6138 | 0.7249 | 581 | 2842 | 23.0000 | 151.0000 | 53.15 | 18.81 |
| tracktrack.yaml | MOT17-11-FRCNN | 0.5607 | 0.5152 | 0.6144 | 0.8415 | 0.4461 | 0.6061 | 465 | 706 | 6.0000 | 18.0000 | 54.58 | 18.32 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5217 | 0.5378 | 0.5077 | 0.8344 | 0.5719 | 0.6628 | 232 | 498 | 29.0000 | 54.0000 | 49.66 | 20.14 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4613 | 0.4705 | 0.4557 | 0.7902 | 0.5186 | 0.6091 | 417 | 1454 | 58.0000 | 207.0000 | 57.46 | 17.40 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4404 | 0.4037 | 0.4853 | 0.8051 | 0.4794 | 0.5810 | 180 | 1635 | 43.0000 | 119.0000 | 40.25 | 24.85 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3672 | 0.3469 | 0.3937 | 0.8345 | 0.3692 | 0.4641 | 313 | 3499 | 92.0000 | 198.0000 | 50.25 | 19.90 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3655 | 0.3687 | 0.3663 | 0.7478 | 0.2744 | 0.4981 | 384 | 729 | 48.0000 | 93.0000 | 42.38 | 23.60 |
