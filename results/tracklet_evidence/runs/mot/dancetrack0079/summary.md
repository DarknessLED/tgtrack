# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0079)
- workers: jobs=1 mode=serial wall=33.5s
- measured scaling: sum(pair wall)=33.5s / wall=33.5s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4996 | 0.6428 | 0.3948 | 0.8705 | 0.6840 | 0.4946 | 0.5309 | 0.4629 | 291 | 757 | 101.0000 | 143.0000 | 6.0000 | 0.0000 | 63.79 | 15.68 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0079 | 0.4996 | 0.6428 | 0.3948 | 0.8705 | 0.6840 | 0.4946 | 291 | 757 | 101.0000 | 143.0000 | 63.79 | 15.68 |
