# MOT evaluation

- pairs: 7 (7 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=46.9s
- measured scaling: sum(pair wall)=84.5s / wall=46.9s -> speedup 1.80x on 32 logical CPUs
- pipeline: weights=data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=0.3, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 7 | 0.5021 | 0.4789 | 0.5324 | 0.7988 | 0.5711 | 0.6546 | 0.7758 | 0.5661 | 2085 | 9706 | 297.0000 | 923.0000 | 131.0000 | 96.0000 | 41.76 | 23.95 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-11-FRCNN | 0.6352 | 0.6020 | 0.6722 | 0.8594 | 0.6885 | 0.7623 | 137 | 517 | 8.0000 | 35.0000 | 44.92 | 22.26 |
| tracktrack.yaml | MOT17-04-FRCNN | 0.5927 | 0.5719 | 0.6186 | 0.7974 | 0.7203 | 0.7910 | 575 | 1880 | 41.0000 | 115.0000 | 48.11 | 20.79 |
| tracktrack.yaml | MOT17-09-FRCNN | 0.5447 | 0.5732 | 0.5187 | 0.8216 | 0.6842 | 0.6962 | 122 | 413 | 25.0000 | 54.0000 | 39.91 | 25.05 |
| tracktrack.yaml | MOT17-10-FRCNN | 0.4850 | 0.4859 | 0.4873 | 0.7977 | 0.5970 | 0.6492 | 201 | 1373 | 41.0000 | 221.0000 | 44.72 | 22.36 |
| tracktrack.yaml | MOT17-13-FRCNN | 0.4242 | 0.4028 | 0.4535 | 0.7742 | 0.4609 | 0.5729 | 350 | 1518 | 56.0000 | 165.0000 | 39.25 | 25.48 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3615 | 0.3555 | 0.3704 | 0.7901 | 0.4001 | 0.4766 | 444 | 3187 | 82.0000 | 228.0000 | 43.91 | 22.77 |
| tracktrack.yaml | MOT17-05-FRCNN | 0.3276 | 0.3454 | 0.3186 | 0.7462 | 0.3013 | 0.4382 | 256 | 818 | 44.0000 | 105.0000 | 31.25 | 32.00 |
