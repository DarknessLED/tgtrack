# MOT evaluation

- pairs: 6 (6 ok, 0 failed)
- sequences: 2 (MOT17-02-FRCNN, MOT17-04-FRCNN)
- workers: jobs=2 mode=process-pool wall=44.1s
- measured scaling: sum(pair wall)=77.7s / wall=44.1s -> speedup 1.76x on 32 logical CPUs
- pipeline: weights=config default, imgsz=default, conf=0.25, half=True, embed=False, subsample=3, max_frames=200

## By tracker (combine_mot over sequences, sorted by HOTA)

| tracker | num_sequences | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDP | IDR | FP | FN | IDSW | Frag | MT | ML | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | 2 | 0.5038 | 0.4437 | 0.5753 | 0.8407 | 0.4535 | 0.6059 | 0.7360 | 0.5150 | 1803 | 6341 | 115.0000 | 349.0000 | 36.0000 | 31.0000 | 54.78 | 18.25 |
| bytetrack.yaml | 2 | 0.5037 | 0.4392 | 0.5812 | 0.8432 | 0.4528 | 0.5989 | 0.7378 | 0.5040 | 1675 | 6463 | 132.0000 | 322.0000 | 35.0000 | 34.0000 | 33.93 | 29.47 |
| botsort.yaml | 2 | 0.5015 | 0.4403 | 0.5747 | 0.8429 | 0.4540 | 0.6000 | 0.7371 | 0.5058 | 1694 | 6436 | 121.0000 | 323.0000 | 34.0000 | 33.0000 | 46.00 | 21.74 |

## Per (sequence, tracker), sorted by HOTA

| tracker | sequence | HOTA | DetA | AssA | LocA | MOTA | IDF1 | FP | FN | IDSW | Frag | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tracktrack.yaml | MOT17-04-FRCNN | 0.5862 | 0.5260 | 0.6558 | 0.8448 | 0.5665 | 0.7058 | 1003 | 2842 | 23.0000 | 151.0000 | 55.24 | 18.10 |
| bytetrack.yaml | MOT17-04-FRCNN | 0.5832 | 0.5217 | 0.6544 | 0.8468 | 0.5632 | 0.6979 | 950 | 2906 | 42.0000 | 139.0000 | 33.79 | 29.59 |
| botsort.yaml | MOT17-04-FRCNN | 0.5777 | 0.5226 | 0.6413 | 0.8463 | 0.5642 | 0.6873 | 963 | 2888 | 38.0000 | 139.0000 | 43.69 | 22.89 |
| botsort.yaml | MOT17-02-FRCNN | 0.3655 | 0.3225 | 0.4207 | 0.8350 | 0.2952 | 0.4546 | 731 | 3548 | 83.0000 | 184.0000 | 48.31 | 20.70 |
| bytetrack.yaml | MOT17-02-FRCNN | 0.3606 | 0.3211 | 0.4117 | 0.8350 | 0.2936 | 0.4341 | 725 | 3557 | 90.0000 | 183.0000 | 34.07 | 29.35 |
| tracktrack.yaml | MOT17-02-FRCNN | 0.3548 | 0.3264 | 0.3909 | 0.8315 | 0.2905 | 0.4407 | 800 | 3499 | 92.0000 | 198.0000 | 54.33 | 18.41 |
