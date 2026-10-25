# MOT evaluation

- pairs: 2 (2 ok, 0 failed)
- sequences: 2 (MOT17-02-FRCNN, MOT17-04-FRCNN)
- workers: jobs=1 mode=serial wall=44.5s
- measured scaling: sum(pair wall)=44.5s / wall=44.5s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=C:\Projects\track\data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 2 | 0.5208 | 0.4851 | 0.5635 | 0.7925 | 0.5736 | 0.6819 | 0.7755 | 0.6085 | 1523 | 4777 | 144.0000 | 380.0000 | 48.0000 | 24.0000 | 68.72 | 14.55 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.6029 | 0.5729 | 0.6389 | 0.7964 | 0.7103 | 0.7963 | 762 | 1778 | 45.0000 | 119.0000 | 71.75 | 13.94 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3756 | 0.3612 | 0.3940 | 0.7839 | 0.3765 | 0.4921 | 761 | 2999 | 99.0000 | 261.0000 | 65.69 | 15.22 |
