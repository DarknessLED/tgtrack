# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0025)
- workers: jobs=1 mode=serial wall=47.7s
- measured scaling: sum(pair wall)=47.7s / wall=47.7s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.7444 | 0.7729 | 0.7196 | 0.8981 | 0.8838 | 0.8117 | 0.8419 | 0.7836 | 53 | 236 | 18.0000 | 59.0000 | 7.0000 | 0.0000 | 115.27 | 8.68 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0025 | 0.7444 | 0.7729 | 0.7196 | 0.8981 | 0.8838 | 0.8117 | 53 | 236 | 18.0000 | 59.0000 | 115.27 | 8.68 |
