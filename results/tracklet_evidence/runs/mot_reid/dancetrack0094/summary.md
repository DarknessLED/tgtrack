# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0094)
- workers: jobs=1 mode=serial wall=78.6s
- measured scaling: sum(pair wall)=78.6s / wall=78.6s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.5057 | 0.6926 | 0.3745 | 0.8968 | 0.7534 | 0.5315 | 0.5660 | 0.5009 | 284 | 956 | 202.0000 | 216.0000 | 16.0000 | 0.0000 | 202.69 | 4.93 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0094 | 0.5057 | 0.6926 | 0.3745 | 0.8968 | 0.7534 | 0.5315 | 284 | 956 | 202.0000 | 216.0000 | 202.69 | 4.93 |
