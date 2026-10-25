# MOT evaluation

- pairs: 2 (2 ok, 0 failed)
- sequences: 1 (MOT-SYNTH-01)
- workers: jobs=1 mode=serial wall=10.0s
- measured scaling: sum(pair wall)=10.0s / wall=10.0s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=True, subsample=1, max_frames=None

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4932 | 0.4052 | 0.9196 | 0.7407 | 0.3833 | 0.6574 | 0.7396 | 0.5917 | 25 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 54.04 | 18.50 |
| deepocsort.yaml | 1 | 0.4932 | 0.4052 | 0.9196 | 0.7407 | 0.3833 | 0.6574 | 0.7396 | 0.5917 | 25 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 55.09 | 18.15 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | MOT-SYNTH-01 | 0.4932 | 0.4052 | 0.6039 | 0.7407 | 0.3833 | 0.6574 | 25 | 49 | 0.0000 | 0.0000 | 54.04 | 18.50 |
| deepocsort.yaml | MOT-SYNTH-01 | 0.4932 | 0.4052 | 0.6039 | 0.7407 | 0.3833 | 0.6574 | 25 | 49 | 0.0000 | 0.0000 | 55.09 | 18.15 |
