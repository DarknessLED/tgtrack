# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0094)
- workers: jobs=1 mode=serial wall=39.5s
- measured scaling: sum(pair wall)=39.5s / wall=39.5s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.5251 | 0.6954 | 0.4006 | 0.8996 | 0.7633 | 0.5422 | 0.5848 | 0.5054 | 211 | 1005 | 168.0000 | 204.0000 | 14.0000 | 0.0000 | 76.85 | 13.01 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0094 | 0.5251 | 0.6954 | 0.4006 | 0.8996 | 0.7633 | 0.5422 | 211 | 1005 | 168.0000 | 204.0000 | 76.85 | 13.01 |
