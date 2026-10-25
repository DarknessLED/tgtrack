# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0030)
- workers: jobs=1 mode=serial wall=50.3s
- measured scaling: sum(pair wall)=50.3s / wall=50.3s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.6739 | 0.8479 | 0.5368 | 0.9134 | 0.9339 | 0.6613 | 0.6620 | 0.6606 | 46 | 50 | 23.0000 | 24.0000 | 6.0000 | 0.0000 | 119.26 | 8.39 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0030 | 0.6739 | 0.8479 | 0.5368 | 0.9134 | 0.9339 | 0.6613 | 46 | 50 | 23.0000 | 24.0000 | 119.26 | 8.39 |
