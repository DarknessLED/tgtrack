# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 1 (MOT-SYNTH-01)
- workers: jobs=4 mode=process-pool wall=14.3s
- measured scaling: sum(pair wall)=45.0s / wall=14.3s -> speedup 3.15x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=1, max_frames=None

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 44.34 | 22.55 |
| ocsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 32.42 | 30.84 |
| deepocsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 44.67 | 22.39 |
| fasttrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 33.57 | 29.79 |
| tracktrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 28.35 | 35.27 |
| bytetrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 18.49 | 54.07 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 44.34 | 22.55 |
| bytetrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 18.49 | 54.07 |
| deepocsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 44.67 | 22.39 |
| fasttrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 33.57 | 29.79 |
| ocsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 32.42 | 30.84 |
| tracktrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 28.35 | 35.27 |

## Parallel scaling (same plan, different worker counts)

| jobs | pairs | wall_s | sum_pair_wall_s | speedup |
|---|---|---|---|---|
| 1 | 6 | 13.4 | 13.4 | 1.00 |
| 2 | 6 | 10.9 | 18.5 | 1.70 |
| 4 | 6 | 14.3 | 45.0 | 3.15 |
