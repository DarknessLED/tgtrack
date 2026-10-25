# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0025)
- workers: jobs=1 mode=serial wall=25.9s
- measured scaling: sum(pair wall)=25.9s / wall=25.9s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.6713 | 0.7689 | 0.5893 | 0.8987 | 0.8838 | 0.6943 | 0.7252 | 0.6659 | 33 | 249 | 25.0000 | 59.0000 | 7.0000 | 0.0000 | 44.02 | 22.72 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0025 | 0.6713 | 0.7689 | 0.5893 | 0.8987 | 0.8838 | 0.6943 | 33 | 249 | 25.0000 | 59.0000 | 44.02 | 22.72 |
