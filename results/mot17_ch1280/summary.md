# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=48.2s
- measured scaling: sum(pair wall)=87.9s / wall=48.2s -> speedup 1.82x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.5050 | 0.4829 | 0.5341 | 0.7962 | 0.5586 | 0.6531 | 0.7430 | 0.5826 | 3000 | 9085 | 355.0000 | 978.0000 | 141.0000 | 85.0000 | 42.69 | 23.43 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6196 | 0.6008 | 0.6414 | 0.8561 | 0.6791 | 0.7275 | 187 | 484 | 11.0000 | 36.0000 | 42.16 | 23.72 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.6029 | 0.5729 | 0.6389 | 0.7964 | 0.7103 | 0.7963 | 762 | 1778 | 45.0000 | 119.0000 | 54.64 | 18.30 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5537 | 0.5847 | 0.5253 | 0.8251 | 0.6734 | 0.7113 | 170 | 387 | 22.0000 | 58.0000 | 40.52 | 24.68 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4735 | 0.4968 | 0.4535 | 0.7947 | 0.6004 | 0.6271 | 267 | 1284 | 50.0000 | 228.0000 | 43.59 | 22.94 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4265 | 0.4096 | 0.4514 | 0.7708 | 0.4393 | 0.5743 | 546 | 1374 | 81.0000 | 182.0000 | 35.56 | 28.12 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3756 | 0.3612 | 0.3940 | 0.7839 | 0.3765 | 0.4921 | 761 | 2999 | 99.0000 | 261.0000 | 50.53 | 19.79 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3245 | 0.3546 | 0.3071 | 0.7439 | 0.2919 | 0.4377 | 307 | 779 | 47.0000 | 94.0000 | 31.53 | 31.72 |
