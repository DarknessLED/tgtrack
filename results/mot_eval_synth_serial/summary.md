# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 1 (MOT-SYNTH-01)
- workers: jobs=1 mode=serial wall=14.3s
- measured scaling: sum(pair wall)=14.3s / wall=14.3s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=1, max_frames=None

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 32.62 | 30.66 |
| ocsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 20.37 | 49.10 |
| deepocsort.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 30.30 | 33.00 |
| fasttrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 17.93 | 55.76 |
| tracktrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 29.88 | 33.47 |
| bytetrack.yaml | 1 | 0.4988 | 0.4145 | 0.9196 | 0.7407 | 0.4083 | 0.6667 | 0.7634 | 0.5917 | 22 | 49 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 19.97 | 50.08 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 32.62 | 30.66 |
| bytetrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 19.97 | 50.08 |
| deepocsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 30.30 | 33.00 |
| fasttrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 17.93 | 55.76 |
| ocsort.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 20.37 | 49.10 |
| tracktrack.yaml | MOT-SYNTH-01 | 0.4988 | 0.4145 | 0.6039 | 0.7407 | 0.4083 | 0.6667 | 22 | 49 | 0.0000 | 0.0000 | 29.88 | 33.47 |
