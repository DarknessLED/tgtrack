# MOT evaluation

- pairs: 2 (2 ok, 0 failed)
- sequences: 2 (MOT17-02-FRCNN, MOT17-04-FRCNN)
- workers: jobs=1 mode=serial wall=352.6s
- measured scaling: sum(pair wall)=352.6s / wall=352.6s -> speedup 1.00x on 32 logical CPUs
- pipeline: weights=C:\Projects\track\data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=default, half=True, embed=True, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 2 | 0.5205 | 0.4847 | 0.5628 | 0.7915 | 0.5719 | 0.6816 | 0.7725 | 0.6098 | 1569 | 4752 | 148.0000 | 385.0000 | 47.0000 | 23.0000 | 826.97 | 1.21 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.6049 | 0.5747 | 0.6407 | 0.7957 | 0.7144 | 0.7994 | 759 | 1759 | 30.0000 | 120.0000 | 1195.02 | 0.84 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3711 | 0.3588 | 0.3875 | 0.7821 | 0.3665 | 0.4869 | 810 | 2993 | 118.0000 | 265.0000 | 458.92 | 2.18 |
