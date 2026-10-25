# MOT evaluation

- pairs: 14 (14 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=22.8s
- measured scaling: sum(pair wall)=43.0s / wall=22.8s -> speedup 1.88x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=30

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybridsort.yaml | 7 | 0.5690 | 0.4688 | 0.7026 | 0.8228 | 0.4748 | 0.6669 | 0.7787 | 0.5831 | 547 | 1584 | 37.0000 | 107.0000 | 78.0000 | 48.0000 | 59.74 | 16.74 |
| tracktrack.yaml | 7 | 0.5682 | 0.4585 | 0.7150 | 0.8260 | 0.4772 | 0.6679 | 0.8103 | 0.5681 | 449 | 1683 | 26.0000 | 84.0000 | 74.0000 | 57.0000 | 46.92 | 21.31 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-09-FRCNN | 0.6901 | 0.6341 | 0.7537 | 0.8466 | 0.6941 | 0.8241 | 22 | 45 | 0.0000 | 1.0000 | 48.82 | 20.48 |
| hybridsort.yaml | MOT17-09-FRCNN | 0.6599 | 0.6199 | 0.7064 | 0.8437 | 0.6575 | 0.7804 | 32 | 42 | 1.0000 | 1.0000 | 52.55 | 19.03 |
| hybridsort.yaml | MOT17-04-FRCNN | 0.6588 | 0.5643 | 0.7773 | 0.8443 | 0.6252 | 0.7708 | 92 | 368 | 7.0000 | 18.0000 | 72.48 | 13.80 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.6528 | 0.5596 | 0.7695 | 0.8459 | 0.6268 | 0.7664 | 84 | 376 | 5.0000 | 16.0000 | 43.53 | 22.97 |
| tracktrack.yaml | MOT17-11-FRCNN | 0.5694 | 0.4266 | 0.7827 | 0.8261 | 0.2158 | 0.5557 | 128 | 209 | 1.0000 | 1.0000 | 44.24 | 22.61 |
| hybridsort.yaml | MOT17-11-FRCNN | 0.5654 | 0.4203 | 0.7840 | 0.8236 | 0.1787 | 0.5468 | 146 | 207 | 1.0000 | 1.0000 | 52.37 | 19.09 |
| hybridsort.yaml | MOT17-10-FRCNN | 0.5517 | 0.5210 | 0.5872 | 0.8029 | 0.5884 | 0.7107 | 70 | 148 | 10.0000 | 37.0000 | 58.65 | 17.05 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.5345 | 0.4967 | 0.5787 | 0.8050 | 0.5650 | 0.7072 | 55 | 176 | 10.0000 | 28.0000 | 49.78 | 20.09 |
| hybridsort.yaml | MOT17-13-FRCNN | 0.5215 | 0.4321 | 0.6336 | 0.8279 | 0.4924 | 0.6548 | 46 | 312 | 8.0000 | 29.0000 | 60.26 | 16.59 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.5188 | 0.4082 | 0.6626 | 0.8313 | 0.4743 | 0.6365 | 31 | 345 | 3.0000 | 21.0000 | 44.18 | 22.63 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.4922 | 0.3472 | 0.7137 | 0.8079 | 0.3490 | 0.5671 | 67 | 400 | 3.0000 | 9.0000 | 50.61 | 19.76 |
| hybridsort.yaml | MOT17-02-FRCNN | 0.4840 | 0.3601 | 0.6713 | 0.8032 | 0.3449 | 0.5511 | 80 | 388 | 5.0000 | 10.0000 | 63.11 | 15.84 |
| hybridsort.yaml | MOT17-05-FRCNN | 0.3644 | 0.3056 | 0.4370 | 0.7515 | 0.1277 | 0.4769 | 81 | 119 | 5.0000 | 11.0000 | 58.73 | 17.03 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3564 | 0.2818 | 0.4525 | 0.7539 | 0.1574 | 0.4550 | 62 | 132 | 4.0000 | 8.0000 | 47.30 | 21.14 |
