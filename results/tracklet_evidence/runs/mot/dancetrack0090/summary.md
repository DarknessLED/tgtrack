# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0090)
- workers: jobs=1 mode=serial wall=34.2s
- measured scaling: sum(pair wall)=34.2s / wall=34.2s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4639 | 0.6720 | 0.3252 | 0.8859 | 0.7243 | 0.4493 | 0.4772 | 0.4245 | 266 | 700 | 116.0000 | 135.0000 | 9.0000 | 0.0000 | 54.69 | 18.29 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0090 | 0.4639 | 0.6720 | 0.3252 | 0.8859 | 0.7243 | 0.4493 | 266 | 700 | 116.0000 | 135.0000 | 54.69 | 18.29 |
