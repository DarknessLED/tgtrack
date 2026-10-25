# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=1 mode=serial wall=68.0s
- measured scaling: sum(pair wall)=68.0s / wall=68.0s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4856 | 0.4398 | 0.5411 | 0.8241 | 0.4398 | 0.5931 | 0.6955 | 0.5170 | 4128 | 11363 | 299.0000 | 840.0000 | 115.0000 | 103.0000 | 37.50 | 26.67 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.5862 | 0.5260 | 0.6558 | 0.8448 | 0.5665 | 0.7058 | 1003 | 2842 | 23.0000 | 151.0000 | 40.77 | 24.53 |
| tracktrack.yaml | MOT17-11-FRCNN | 0.5533 | 0.5017 | 0.6144 | 0.8415 | 0.4141 | 0.5960 | 533 | 706 | 6.0000 | 18.0000 | 36.55 | 27.36 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.4815 | 0.4580 | 0.5075 | 0.8343 | 0.3728 | 0.5984 | 585 | 498 | 29.0000 | 54.0000 | 36.18 | 27.64 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4517 | 0.4510 | 0.4557 | 0.7902 | 0.4712 | 0.5930 | 607 | 1454 | 58.0000 | 207.0000 | 37.52 | 26.65 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4389 | 0.4010 | 0.4854 | 0.8049 | 0.4710 | 0.5780 | 210 | 1635 | 43.0000 | 119.0000 | 39.22 | 25.50 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3653 | 0.3679 | 0.3665 | 0.7477 | 0.2706 | 0.4970 | 390 | 729 | 48.0000 | 93.0000 | 32.29 | 30.97 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3548 | 0.3264 | 0.3909 | 0.8315 | 0.2905 | 0.4407 | 800 | 3499 | 92.0000 | 198.0000 | 39.79 | 25.13 |
