# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0079)
- workers: jobs=1 mode=serial wall=58.5s
- measured scaling: sum(pair wall)=58.5s / wall=58.5s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.5058 | 0.6508 | 0.4001 | 0.8693 | 0.6727 | 0.5102 | 0.5394 | 0.4840 | 344 | 717 | 129.0000 | 142.0000 | 6.0000 | 0.0000 | 137.31 | 7.28 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0079 | 0.5058 | 0.6508 | 0.4001 | 0.8693 | 0.6727 | 0.5102 | 344 | 717 | 129.0000 | 142.0000 | 137.31 | 7.28 |
