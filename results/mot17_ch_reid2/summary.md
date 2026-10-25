# MOT evaluation

- pairs: 14 (14 ok, 0 failed)
- sequences: 7 (MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN)
- workers: jobs=2 mode=process-pool wall=101.0s
- measured scaling: sum(pair wall)=197.2s / wall=101.0s -> speedup 1.95x on 32 logical CPUs
- pipeline: weights=data\weights\crowdhuman_yolov8n_best.pt, imgsz=1280, conf=default, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| configs\bs_reid.yaml | 7 | 0.5009 | 0.4856 | 0.5226 | 0.7980 | 0.5676 | 0.6475 | 0.7449 | 0.5727 | 2643 | 9159 | 385.0000 | 929.0000 | 142.0000 | 89.0000 | 50.03 | 19.99 |
| configs\dos_reid.yaml | 7 | 0.4910 | 0.4819 | 0.5065 | 0.7976 | 0.5591 | 0.6282 | 0.7228 | 0.5555 | 2687 | 9213 | 526.0000 | 952.0000 | 143.0000 | 85.0000 | 51.92 | 19.26 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| configs\dos_reid.yaml | MOT17-11-FRCNN | 0.6465 | 0.6052 | 0.6924 | 0.8589 | 0.6894 | 0.7729 | 163 | 490 | 7.0000 | 29.0000 | 49.03 | 20.40 |
| configs\bs_reid.yaml | MOT17-11-FRCNN | 0.6399 | 0.6032 | 0.6805 | 0.8588 | 0.6847 | 0.7657 | 168 | 492 | 10.0000 | 28.0000 | 47.84 | 20.90 |
| configs\bs_reid.yaml | MOT17-04-FRCNN | 0.5992 | 0.5761 | 0.6274 | 0.7971 | 0.7132 | 0.7921 | 736 | 1779 | 44.0000 | 120.0000 | 52.38 | 19.09 |
| configs\dos_reid.yaml | MOT17-04-FRCNN | 0.5798 | 0.5710 | 0.5937 | 0.7958 | 0.7062 | 0.7573 | 753 | 1795 | 74.0000 | 128.0000 | 55.94 | 17.88 |
| configs\dos_reid.yaml | MOT17-09-FRCNN | 0.5156 | 0.5900 | 0.4509 | 0.8255 | 0.6875 | 0.6482 | 147 | 373 | 34.0000 | 58.0000 | 56.19 | 17.80 |
| configs\bs_reid.yaml | MOT17-09-FRCNN | 0.5119 | 0.5859 | 0.4480 | 0.8236 | 0.6881 | 0.6198 | 151 | 370 | 32.0000 | 58.0000 | 54.28 | 18.42 |
| configs\dos_reid.yaml | MOT17-10-FRCNN | 0.4709 | 0.4914 | 0.4540 | 0.7965 | 0.5970 | 0.6247 | 226 | 1318 | 71.0000 | 219.0000 | 55.03 | 18.17 |
| configs\bs_reid.yaml | MOT17-10-FRCNN | 0.4704 | 0.4958 | 0.4493 | 0.7974 | 0.6000 | 0.6182 | 236 | 1314 | 53.0000 | 221.0000 | 53.42 | 18.72 |
| configs\bs_reid.yaml | MOT17-13-FRCNN | 0.4329 | 0.4138 | 0.4597 | 0.7727 | 0.4696 | 0.5819 | 412 | 1394 | 87.0000 | 169.0000 | 48.82 | 20.48 |
| configs\dos_reid.yaml | MOT17-13-FRCNN | 0.4010 | 0.4088 | 0.4003 | 0.7739 | 0.4424 | 0.5175 | 425 | 1426 | 139.0000 | 173.0000 | 49.30 | 20.28 |
| configs\dos_reid.yaml | MOT17-02-FRCNN | 0.3629 | 0.3562 | 0.3728 | 0.7870 | 0.3739 | 0.4743 | 675 | 3060 | 140.0000 | 243.0000 | 62.24 | 16.07 |
| configs\bs_reid.yaml | MOT17-02-FRCNN | 0.3560 | 0.3598 | 0.3553 | 0.7873 | 0.3816 | 0.4769 | 654 | 3063 | 110.0000 | 231.0000 | 58.70 | 17.04 |
| configs\bs_reid.yaml | MOT17-05-FRCNN | 0.3406 | 0.3675 | 0.3225 | 0.7476 | 0.3237 | 0.4681 | 286 | 747 | 49.0000 | 102.0000 | 35.26 | 28.36 |
| configs\dos_reid.yaml | MOT17-05-FRCNN | 0.3137 | 0.3627 | 0.2743 | 0.7463 | 0.3063 | 0.4230 | 298 | 751 | 61.0000 | 102.0000 | 36.27 | 27.57 |
