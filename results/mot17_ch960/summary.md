# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=47.2s
- measured scaling: sum(pair wall)=83.9s / wall=47.2s -> speedup 1.78x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4847 | 0.4464 | 0.5380 | 0.7753 | 0.5187 | 0.6391 | 0.7431 | 0.5606 | 3159 | 10081 | 327.0000 | 969.0000 | 132.0000 | 94.0000 | 40.59 | 24.63 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6374 | 0.5963 | 0.6847 | 0.8557 | 0.6805 | 0.7599 | 175 | 498 | 6.0000 | 34.0000 | 41.21 | 24.26 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.5498 | 0.5125 | 0.6012 | 0.7499 | 0.6310 | 0.7513 | 1041 | 2214 | 38.0000 | 164.0000 | 50.77 | 19.70 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5025 | 0.5570 | 0.4541 | 0.8188 | 0.6441 | 0.6356 | 175 | 414 | 42.0000 | 61.0000 | 37.21 | 26.87 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4890 | 0.4632 | 0.5218 | 0.7879 | 0.5620 | 0.6636 | 281 | 1423 | 51.0000 | 228.0000 | 41.05 | 24.36 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.4148 | 0.4309 | 0.4030 | 0.7833 | 0.4963 | 0.5861 | 160 | 606 | 40.0000 | 88.0000 | 29.34 | 34.08 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4062 | 0.3692 | 0.4547 | 0.7612 | 0.3816 | 0.5368 | 519 | 1610 | 78.0000 | 168.0000 | 36.91 | 27.09 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3424 | 0.3166 | 0.3734 | 0.7562 | 0.3220 | 0.4630 | 808 | 3316 | 72.0000 | 226.0000 | 47.24 | 21.17 |
