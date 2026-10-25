# MOT evaluation

- pairs: 1 (1 ok, 0 failed)
- sequences: 1 (dancetrack0030)
- workers: jobs=1 mode=serial wall=40.1s
- measured scaling: sum(pair wall)=40.1s / wall=40.1s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=2, max_frames=300

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.6821 | 0.8537 | 0.5456 | 0.9142 | 0.9478 | 0.6936 | 0.7019 | 0.6856 | 21 | 63 | 10.0000 | 26.0000 | 6.0000 | 0.0000 | 73.57 | 13.59 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | dancetrack0030 | 0.6821 | 0.8537 | 0.5456 | 0.9142 | 0.9478 | 0.6936 | 21 | 63 | 10.0000 | 26.0000 | 73.57 | 13.59 |
