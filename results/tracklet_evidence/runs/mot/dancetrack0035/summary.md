# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0035)
- workers: jobs=1 mode=serial wall=38.0s
- measured scaling: sum(pair wall)=38.0s / wall=38.0s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.3557 | 0.6611 | 0.1932 | 0.8681 | 0.7193 | 0.3604 | 0.3645 | 0.3563 | 245 | 296 | 94.0000 | 89.0000 | 5.0000 | 0.0000 | 64.32 | 15.55 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0035 | 0.3557 | 0.6611 | 0.1932 | 0.8681 | 0.7193 | 0.3604 | 245 | 296 | 94.0000 | 89.0000 | 64.32 | 15.55 |
