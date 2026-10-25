# MOT evaluation

- pairs: 21 (21 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=121.9s
- measured scaling: sum(pair wall)=235.1s / wall=121.9s -> speedup 1.93x on 32 logical CPUs
- pipeline: weights=data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| configs\ch_loose.yaml | 7 | 0.4989 | 0.4826 | 0.5221 | 0.7952 | 0.5546 | 0.6432 | 0.7254 | 0.5777 | 3219 | 8960 | 375.0000 | 995.0000 | 144.0000 | 81.0000 | 42.66 | 23.44 |
| configs\ch_loose_buf.yaml | 7 | 0.4989 | 0.4826 | 0.5221 | 0.7952 | 0.5546 | 0.6432 | 0.7254 | 0.5777 | 3219 | 8960 | 375.0000 | 995.0000 | 144.0000 | 81.0000 | 42.93 | 23.30 |
| configs\ch_loose_reid.yaml | 7 | 0.4989 | 0.4826 | 0.5221 | 0.7952 | 0.5546 | 0.6432 | 0.7254 | 0.5777 | 3219 | 8960 | 375.0000 | 995.0000 | 144.0000 | 81.0000 | 40.07 | 24.96 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| configs\ch_loose.yaml | MOT17-11-FRCNN | 0.6352 | 0.5980 | 0.6769 | 0.8557 | 0.6744 | 0.7594 | 212 | 472 | 8.0000 | 38.0000 | 40.93 | 24.43 |
| configs\ch_loose_buf.yaml | MOT17-11-FRCNN | 0.6352 | 0.5980 | 0.6769 | 0.8557 | 0.6744 | 0.7594 | 212 | 472 | 8.0000 | 38.0000 | 39.53 | 25.30 |
| configs\ch_loose_reid.yaml | MOT17-11-FRCNN | 0.6352 | 0.5980 | 0.6769 | 0.8557 | 0.6744 | 0.7594 | 212 | 472 | 8.0000 | 38.0000 | 39.19 | 25.52 |
| configs\ch_loose.yaml | MOT17-04-FRCNN | 0.6018 | 0.5730 | 0.6365 | 0.7961 | 0.7104 | 0.7973 | 765 | 1769 | 50.0000 | 119.0000 | 46.38 | 21.56 |
| configs\ch_loose_buf.yaml | MOT17-04-FRCNN | 0.6018 | 0.5730 | 0.6365 | 0.7961 | 0.7104 | 0.7973 | 765 | 1769 | 50.0000 | 119.0000 | 46.85 | 21.34 |
| configs\ch_loose_reid.yaml | MOT17-04-FRCNN | 0.6018 | 0.5730 | 0.6365 | 0.7961 | 0.7104 | 0.7973 | 765 | 1769 | 50.0000 | 119.0000 | 46.83 | 21.35 |
| configs\ch_loose.yaml | MOT17-09-FRCNN | 0.5187 | 0.5733 | 0.4704 | 0.8226 | 0.6650 | 0.6373 | 188 | 379 | 27.0000 | 59.0000 | 39.40 | 25.38 |
| configs\ch_loose_buf.yaml | MOT17-09-FRCNN | 0.5187 | 0.5733 | 0.4704 | 0.8226 | 0.6650 | 0.6373 | 188 | 379 | 27.0000 | 59.0000 | 43.30 | 23.09 |
| configs\ch_loose_reid.yaml | MOT17-09-FRCNN | 0.5187 | 0.5733 | 0.4704 | 0.8226 | 0.6650 | 0.6373 | 188 | 379 | 27.0000 | 59.0000 | 43.65 | 22.91 |
| configs\ch_loose.yaml | MOT17-10-FRCNN | 0.4729 | 0.4981 | 0.4514 | 0.7942 | 0.5977 | 0.6306 | 297 | 1267 | 48.0000 | 235.0000 | 45.58 | 21.94 |
| configs\ch_loose_buf.yaml | MOT17-10-FRCNN | 0.4729 | 0.4981 | 0.4514 | 0.7942 | 0.5977 | 0.6306 | 297 | 1267 | 48.0000 | 235.0000 | 46.04 | 21.72 |
| configs\ch_loose_reid.yaml | MOT17-10-FRCNN | 0.4729 | 0.4981 | 0.4514 | 0.7942 | 0.5977 | 0.6306 | 297 | 1267 | 48.0000 | 235.0000 | 45.94 | 21.77 |
| configs\ch_loose.yaml | MOT17-13-FRCNN | 0.4239 | 0.4129 | 0.4428 | 0.7698 | 0.4427 | 0.5663 | 566 | 1347 | 76.0000 | 183.0000 | 44.36 | 22.54 |
| configs\ch_loose_buf.yaml | MOT17-13-FRCNN | 0.4239 | 0.4129 | 0.4428 | 0.7698 | 0.4427 | 0.5663 | 566 | 1347 | 76.0000 | 183.0000 | 42.40 | 23.58 |
| configs\ch_loose_reid.yaml | MOT17-13-FRCNN | 0.4239 | 0.4129 | 0.4428 | 0.7698 | 0.4427 | 0.5663 | 566 | 1347 | 76.0000 | 183.0000 | 33.79 | 29.59 |
| configs\ch_loose.yaml | MOT17-02-FRCNN | 0.3517 | 0.3608 | 0.3470 | 0.7819 | 0.3650 | 0.4621 | 842 | 2967 | 121.0000 | 264.0000 | 53.05 | 18.85 |
| configs\ch_loose_buf.yaml | MOT17-02-FRCNN | 0.3517 | 0.3608 | 0.3470 | 0.7819 | 0.3650 | 0.4621 | 842 | 2967 | 121.0000 | 264.0000 | 54.37 | 18.39 |
| configs\ch_loose_reid.yaml | MOT17-02-FRCNN | 0.3517 | 0.3608 | 0.3470 | 0.7819 | 0.3650 | 0.4621 | 842 | 2967 | 121.0000 | 264.0000 | 44.23 | 22.61 |
| configs\ch_loose.yaml | MOT17-05-FRCNN | 0.3199 | 0.3600 | 0.2904 | 0.7437 | 0.2794 | 0.4258 | 349 | 759 | 45.0000 | 97.0000 | 28.50 | 35.09 |
| configs\ch_loose_buf.yaml | MOT17-05-FRCNN | 0.3199 | 0.3600 | 0.2904 | 0.7437 | 0.2794 | 0.4258 | 349 | 759 | 45.0000 | 97.0000 | 28.03 | 35.68 |
| configs\ch_loose_reid.yaml | MOT17-05-FRCNN | 0.3199 | 0.3600 | 0.2904 | 0.7437 | 0.2794 | 0.4258 | 349 | 759 | 45.0000 | 97.0000 | 27.31 | 36.62 |
