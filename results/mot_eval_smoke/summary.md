# MOT evaluation

- pairs: 2 (2 ok, 0 failed)
- sequences: 1 (MOT-SYNTH-01)
- workers: jobs=2 mode=process-pool wall=7.1s
- measured scaling: sum(pair wall)=10.8s / wall=7.1s -> speedup 1.51x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=default, half=True, embed=False, subsample=1, max_frames=12

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | 1 | 0.5599 | 0.4931 | 0.9638 | 0.7355 | 0.4167 | 0.6957 | 0.7273 | 0.6667 | 9 | 12 | 0.0000 | 0.0000 | 2.0000 | 1.0000 | 43.93 | 22.76 |
| bytetrack.yaml | 1 | 0.5599 | 0.4931 | 0.9638 | 0.7355 | 0.4167 | 0.6957 | 0.7273 | 0.6667 | 9 | 12 | 0.0000 | 0.0000 | 2.0000 | 1.0000 | 32.12 | 31.14 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| botsort.yaml | MOT-SYNTH-01 | 0.5599 | 0.4931 | 0.6480 | 0.7355 | 0.4167 | 0.6957 | 9 | 12 | 0.0000 | 0.0000 | 43.93 | 22.76 |
| bytetrack.yaml | MOT-SYNTH-01 | 0.5599 | 0.4931 | 0.6480 | 0.7355 | 0.4167 | 0.6957 | 9 | 12 | 0.0000 | 0.0000 | 32.12 | 31.14 |
