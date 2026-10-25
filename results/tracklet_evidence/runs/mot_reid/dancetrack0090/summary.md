# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0090)
- workers: jobs=1 mode=serial wall=62.2s
- measured scaling: sum(pair wall)=62.2s / wall=62.2s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4582 | 0.6600 | 0.3224 | 0.8799 | 0.7068 | 0.4756 | 0.5014 | 0.4522 | 314 | 699 | 138.0000 | 146.0000 | 8.0000 | 0.0000 | 154.08 | 6.49 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0090 | 0.4582 | 0.6600 | 0.3224 | 0.8799 | 0.7068 | 0.4756 | 314 | 699 | 138.0000 | 146.0000 | 154.08 | 6.49 |
