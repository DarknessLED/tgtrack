# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0035)
- workers: jobs=1 mode=serial wall=60.9s
- measured scaling: sum(pair wall)=60.8s / wall=60.9s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.3462 | 0.6443 | 0.1881 | 0.8648 | 0.6923 | 0.3440 | 0.3403 | 0.3479 | 321 | 270 | 105.0000 | 101.0000 | 5.0000 | 0.0000 | 145.73 | 6.86 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0035 | 0.3462 | 0.6443 | 0.1881 | 0.8648 | 0.6923 | 0.3440 | 321 | 270 | 105.0000 | 101.0000 | 145.73 | 6.86 |
