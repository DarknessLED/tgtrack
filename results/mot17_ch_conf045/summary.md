# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=41.3s
- measured scaling: sum(pair wall)=76.3s / wall=41.3s -> speedup 1.85x on 32 logical CPUs
- pipeline: weights=data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=0.45, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.4968 | 0.4499 | 0.5530 | 0.8065 | 0.5534 | 0.6495 | 0.8453 | 0.5274 | 890 | 11491 | 206.0000 | 887.0000 | 112.0000 | 136.0000 | 38.44 | 26.02 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6178 | 0.5882 | 0.6503 | 0.8655 | 0.6753 | 0.7227 | 60 | 624 | 6.0000 | 46.0000 | 40.00 | 25.00 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.5967 | 0.5569 | 0.6430 | 0.8025 | 0.7059 | 0.7983 | 322 | 2274 | 28.0000 | 159.0000 | 48.29 | 20.71 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5590 | 0.5817 | 0.5378 | 0.8239 | 0.7061 | 0.7245 | 59 | 436 | 26.0000 | 53.0000 | 37.69 | 26.53 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4626 | 0.4454 | 0.4820 | 0.8059 | 0.5548 | 0.6245 | 94 | 1658 | 32.0000 | 207.0000 | 40.01 | 25.00 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.3976 | 0.3400 | 0.4693 | 0.7856 | 0.4152 | 0.5341 | 127 | 1929 | 31.0000 | 159.0000 | 33.77 | 29.61 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3592 | 0.3178 | 0.4080 | 0.7983 | 0.3900 | 0.4732 | 119 | 3604 | 52.0000 | 176.0000 | 43.75 | 22.85 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.2989 | 0.2916 | 0.3143 | 0.7547 | 0.3088 | 0.4251 | 109 | 966 | 31.0000 | 87.0000 | 25.44 | 39.31 |
