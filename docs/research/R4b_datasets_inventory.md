# R4b — Dataset inventory and acquisition plan (all 10 diversity axes, verified sources)

**Author:** dataset-engineering analyst subagent · **Date:** 2026-10-24
**Scope:** requirement — *find and enumerate every person-tracking / person-ReID dataset the project needs, with exact
download sources, exact sizes, licences and annotation structure*, under a hard **< 10 GB on-disk** budget with
frame-subsampling allowed.
**Relationship to `R4_datasets.md`:** that file is a different agent's acquisition record of the two archives it
chose (MOT17 + DanceTrack `val.zip`, 10,069,999,615 B — which is **0.7 % over** a decimal 10 GB cap). This file does
not replace it; it supplies the *full inventory across all ten requested diversity axes* plus a budget that fits,
and corrects two of that file's environmental assumptions (§0). **`R4_datasets.md` was left untouched.**

**Evidence policy.** Every HTTP status, `Content-Length` and byte count below was measured from **this** Windows
machine on 2026-10-24 with `curl.exe` or `python urllib` (range GET, so the server itself reports the size).
Anything projected rather than measured is marked *(projected)*; anything not tested is marked *(unverified)*.

---

## 0. Network reality on this machine — and two corrections to the existing R4 record

| Host | Status | Evidence |
|---|---|---|
| `huggingface.co` | **BLOCKED** | BRIEF.md; `HF_ENDPOINT=https://hf-mirror.com` is mandatory |
| `hf-mirror.com` | **WORKS** | `200` + 5 544 453 B for a real `yolo26n.pt` download; `206` + exact `Content-Range` for every dataset below |
| `motchallenge.net/data/*.zip` | **WORKS, ungated** | `MOT17.zip` `206` CL **5 860 214 001**; `MOT20.zip` `206` **5 028 926 248**; `MOT16.zip` `206` **1 954 509 127**; `2DMOT2015.zip` `200` **1 313 257 228**; a `-r 0-4000` fetch returned a real `PK\x03\x04` and the first members are `MOT17/test/MOT17-14-SDP/…`, i.e. the genuine archive, not a landing page |
| `www.kaggle.com/api/v1/datasets/download/<owner>/<slug>` | **WORKS with no API key** | `206` + `Content-Range` on all 12 datasets probed. **Kaggle ignores `Range`** → full-file download only. ≈0.75 MB/s |
| `images.cocodataset.org` | WORKS | `val2017.zip` `206` **815 585 330**; `annotations_trainval2017.zip` `206` **252 907 541**. (`cocodataset.org/zips/...` → `404`; use the `images.` host) |
| `github.com/…/releases/download/…` | **DEAD from here** | `WinError 10060` timeout on all three `VisDrone2019-DET-*.zip`, on `coco8.zip`, and on `deepinsight/insightface`'s buffalo zips (plus `RemoteDisconnected`/TLS handshake failures); `ultralytics/assets` `v8.3.0/yolo26n.pt` → `404` |
| `download.openmmlab.com`, `data.got-10k.net`, `www.liangzheng.com.cn`, `www.cbsr.ia.ac.cn`, `posetrack.net`, `vision.in.tum.de`, `tracking-net.org`, `bdd-data.berkeley.edu`, `drive.google.com` | **DEAD** | `404` / DNS failure / timeout / `403` |
| `www.cityscapes-dataset.com/file-handling/?packageID=…` | `200` but a **login-gated HTML form** | not scriptable without an account |

**Throughput measured:** `hf-mirror.com` ≈ **5.9 MB/s**; `motchallenge.net` ≈ **1.2–3.6 MB/s**; Kaggle ≈ **0.75 MB/s**.
**No `ffmpeg` on PATH** — everything here decodes/re-encodes with **Pillow 10.4.0 / OpenCV 4.11.0** in-process.
*(Correction to `R4_datasets.md` §0.7: there is no `ffmpeg` on `PATH`; that file found two copies at absolute paths
elsewhere on the machine, so this plan simply avoids depending on either.)*

**URL form gotcha (measured, and easy to get wrong):**
`https://hf-mirror.com/<repo_id>/resolve/main/<file>` works for **both** dataset and model repos.
`https://hf-mirror.com/datasets/<repo_id>/resolve/main/<file>` returns **`404` even for dataset repos**
(`vilsonrodrigues/lfw` retrieval zip → `404` in that form, `200` in the correct form).

### 0.1 The mechanism that makes < 10 GB possible: range-based remote-ZIP extraction

`_p2/r4_acquire.py` + `_p2/r4_cluster.py` (written and verified this session) fetch a ZIP's **central directory**
with two HTTP Range GETs, then pull only the members you want — **without ever storing the archive**.

* **Verified on MOT17** (`motchallenge.net`): 33 960 members, central directory in **72 s**;
  `MOT17/test/MOT17-01-FRCNN/img1/000001.jpg` → 167 617 B, PIL-decoded as **(1920, 1080)**.
* **Verified on DanceTrack** (`AbdoTW/DanceTrack_1of3`): Zip64 EOCD `entries=8634`; extracted
  `val/dancetrack0004/gt/gt.txt` → 43 096 B / 1528 lines.
* **Verified on CrowdHuman**: extracted `Images/273271,1017ac000ac1360b7.jpg` → 2 561 548 B, SOI/EOI `FFD8`/`FFD9`.
* **Verified on LaSOT** `person.zip`: 64 230 members, 48 `person-*` sequence dirs.

**Two modes, both implemented and both re-run-safe (existing files are skipped):**

1. **Per-member mode** (`RemoteZip.read`) — 2 GETs per member. Fine for reading a `gt.txt` or listing a central
   directory, but **hf-mirror rate-limits it**: a fast loop produced **`HTTP 429 Too Many Requests`**. The toolkit now
   backs off exponentially (up to 60 s) on 429/503.
2. **Block-walk mode** (`r4_cluster.cluster_grab`) — **one large sequential Range GET per 192 MB block** (~20–25
   requests for a 4 GB zip), decoding every member whose local header falls inside the block.
   **Verified live on DanceTrack `val.zip`:** `block 28  100.0%  680 files  32.0 MB  7.7 min`, i.e. ≈**3.0 MB/s**
   with ~4 requests/minute — comfortably under the limiter.

---

## 1. Dataset table

**Pri:** `M` must-have · `N` nice-to-have (fits the budget) · `X` dropped.
**Sub?** = can be frame-subsampled / re-encoded to fit the budget (all re-encode numbers assume **max side 960,
JPEG q85**, and the measured output/input ratio **≈ 0.42**).

### Axis 1 — Standard MOT (crowded street)

| Name | Covers | License | Exact download size | #seq / #ids / #frames | Annotation format | Working URL (verified from this machine) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **MOT17** ⭐ | standard MOT, crowded street | CC BY-NC-SA 3.0 (MOTChallenge terms) | `MOT17.zip` **5 860 214 001 B** (`206`) | 14 seq (7 train / 7 test), 33 705 frames, 21 `gt.txt`; **test has no GT** | MOTChallenge txt, layout `MOT17/<split>/<seq>/{img1,gt,det,seqinfo.ini}` | [`motchallenge.net/data/MOT17.zip`](https://motchallenge.net/data/MOT17.zip) | Yes → **0.30 GB** at every 4th frame | **M** |
| MOT17 (mirror) | same | CC BY-NC-SA 3.0 (card) | `MOT17.zip` **5 868 011 787 B** (`206`) | **YOLO/JDE layout only** (`MOT17/images`, `MOT17/labels_with_ids`) — *not* MOTChallenge layout | YOLO txt | [`hf-mirror.com/datasets/mrhwl123/MOT17`](https://hf-mirror.com/datasets/mrhwl123/MOT17) | — | fallback |
| MOT17 (per-file) | same | "other" (card) | 16 660 files | standard layout, but `gt.txt`+`seqinfo.ini` exist **only for the `-FRCNN` variants** | MOTChallenge txt | [`hf-mirror.com/datasets/Lekim89/MOT17`](https://hf-mirror.com/datasets/Lekim89/MOT17) — `test/MOT17-01-FRCNN/img1/000001.jpg` → `206`, 167 617 B, **2.65 s/frame** | — | fallback (2.5 s/frame ≈ 6 h for 8 400 frames; do not bulk-fetch) |
| **MOT20** | MOT, dense crowd | CC BY-NC-SA 3.0 | `MOT20.zip` **5 028 926 248 B** (`206`) | 8 seq (4/4), 13 410 frames, 4 `gt.txt` | MOTChallenge txt | [`motchallenge.net/data/MOT20.zip`](https://motchallenge.net/data/MOT20.zip) | Yes → **0.13 GB** | N |
| **MOT16** | MOT, crowded street | CC BY-NC-SA 3.0 | `MOT16.zip` **1 954 509 127 B** (`206`) | 14 seq, 11 235 frames | MOTChallenge txt | [`motchallenge.net/data/MOT16.zip`](https://motchallenge.net/data/MOT16.zip) | Yes → **0.27 GB** | N |
| 2D MOT 2015 | legacy MOT | CC BY-NC-SA 3.0 | `2DMOT2015.zip` **1 313 257 228 B** (`200`) | 22 seq, 11 286 frames | MOTChallenge txt | [`motchallenge.net/data/2DMOT2015.zip`](https://motchallenge.net/data/2DMOT2015.zip) | Yes | X (superseded) |
| MOT16 (Kaggle) | same | see Kaggle card | **1 954 869 666 B** (`206`) | — | MOTChallenge txt | [`…/takshmandar/mot16-dataset`](https://www.kaggle.com/api/v1/datasets/download/takshmandar/mot16-dataset) | — | fallback |
| MOT17-02-FRCNN (tiny) | smoke test only | see Kaggle card | **100 184 306 B** (`206`) | 1 seq, 600 frames | MOTChallenge txt | [`…/kyoru4444/mot17-02-fcrnn`](https://www.kaggle.com/api/v1/datasets/download/kyoru4444/mot17-02-fcrnn) | already small | tie-breaker |

**MOT17 per-sequence facts (measured from the real central directory).** The identical `img1/` ships **three times**
(DPM/FRCNN/SDP) — filtering to one variant is what cuts the archive to ≈1/3.

| split | seq | frames | img MB | `gt.txt` KB | `det.txt` KB (FRCNN) |
|---|---|---|---|---|---|
| train | MOT17-02 | 600 | 105.6 | 916.7 | 267.1 |
| train | MOT17-04 | 1050 | 244.7 | 3473.2 | 894.4 |
| train | MOT17-05 | 837 | 30.8 | 251.7 | 121.4 |
| train | MOT17-09 | 525 | 110.4 | 324.3 | 101.0 |
| train | MOT17-10 | 654 | 89.6 | 541.8 | 318.3 |
| train | MOT17-11 | 900 | 127.3 | 338.9 | 198.4 |
| train | MOT17-13 | 750 | 175.3 | 637.7 | 280.5 |
| test | MOT17-01 | 450 | 73.6 | — | 185.7 |
| test | MOT17-03 | 1500 | 393.8 | — | 2159.1 |
| test | MOT17-06 | 1194 | 38.1 | — | 259.5 |
| test | MOT17-07 | 500 | 98.1 | — | 319.4 |
| test | MOT17-08 | 625 | 169.0 | — | 232.7 |
| test | MOT17-12 | 900 | 111.4 | — | 162.0 |
| test | MOT17-14 | 750 | 204.5 | — | 343.5 |
| **Σ** | **14** | **33 705** | **5 920** | **19.5** | **23.1** |

**MOT20 per-sequence** (img MB / gt KB / det KB / frames): train -01 182.3/872/422/429, -02 1123.5/6876/3064/2782,
-03 561.7/11752/5848/2405, -05 1375.4/26217/12857/3315; test -04 813.5/–/7640/2080, -06 399.3/–/2286/1008,
-07 251.9/–/685/585, -08 301.0/–/1431/806. **Σ 5.01 GB images + 45.7 MB gt + 34.2 MB det, 13 410 frames.**

### Axis 2 — Nonlinear / uniform-appearance motion

| Name | Covers | License | Exact download size | #seq / #frames | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| DanceTrack (official) | nonlinear motion — **the key association benchmark** | **CC BY 4.0** | **17.68 GB** total; `val.zip` alone **4 209 785 614 B** (`206`) | 100 seq, ~105 K frames (30 fps) | MOTChallenge txt | [`hf-mirror.com/datasets/noahcao/dancetrack`](https://hf-mirror.com/datasets/noahcao/dancetrack) | yes (would need heavy slicing) | M |
| **DanceTrack 1-of-3** ⭐ **chosen** | same content, pre-subsampled to every 3rd frame | **none declared on the mirror** (upstream CC BY 4.0) | `train1.zip` **1 202 398 475** + `train2.zip` **1 099 969 820** + `val.zip` **1 403 835 598** = **3 706 203 893 B** (all `206`) | **70 seq** = 20 + 25 + 25; frames 1,4,7,…,1201 ⇒ exactly every 3rd | MOTChallenge txt; real GT line `13,0,828,612,138,238,1,1,1` | [`…/AbdoTW/DanceTrack_1of3/resolve/main/val.zip`](https://hf-mirror.com/datasets/AbdoTW/DanceTrack_1of3/resolve/main/val.zip) (`train1.zip`, `train2.zip` same repo) | **this *is* the subsample**; `/4` of it + 960 q85 → **1.25 GB** | **M** |
| DanceTrack val (full-rate) | — | CC BY 4.0 | `val.zip` **4 209 785 614 B** (`206`) | 25 seq, full 30 fps | MOTChallenge txt | [`hf-mirror.com/datasets/bluewhalezero/dancetrack`](https://hf-mirror.com/datasets/bluewhalezero/dancetrack) | yes | fallback |

Central-directory truth: `val.zip` = 8634 members (8508 jpg / 1428.7 MB + 25 `gt.txt` / 2.1 MB), sequences
`dancetrack0004…0097`; `train1.zip` = 7346 members (7245 jpg / 1216.4 MB + 20 `gt.txt`), `0001…0051`;
`train2.zip` = 6797 members, **not Zip64**, sequences including 0052 402 jpg/74.3 MB, 0053 403/45.7, 0055 402/39.0,
0061 402/135.2, 0066 402/118.9. Σ uncompressed ≈ **3.75 GB**.
**Cross-reference:** `R4_datasets.md` §3.1 measured DanceTrack's `gt.txt` `visibility` column to be the constant `1`
for all 225 148 val rows — so **DanceTrack cannot test partial visibility**; that requirement must come from
Occluded-DukeMTMC / OCHuman / PRCC instead.

### Axis 3 — Fast / abrupt motion

| Name | Covers | License | Exact download size | #seq / #frames | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| SportsMOT (tar) | **fast/abrupt motion — essential** | CC BY-NC 4.0 | `train.tar` **6 703 191 552** / `val.tar` **6 592 876 544** / `test.tar` **23 208 035 328** (`206`) — **uncompressed tar**, so not range-sliceable | 240 videos, ~150 K frames | MOTChallenge txt | [`hf-mirror.com/datasets/MCG-NJU/SportsMOT`](https://hf-mirror.com/datasets/MCG-NJU/SportsMOT) | yes, but you must first download 6.6 GB | fallback |
| **SportsMOT per-file** ⭐ **chosen** | same | CC BY-NC 4.0 | fetched per frame | **240 seq** = train 45 / val 45 / test 150 | MOTChallenge txt with **spaces after commas**: `1, 0, 83, 421, 88, 130, 1, 1, 1`; 10 ids in a verified 8074-line file; frames 1..825 | [`hf-mirror.com/datasets/Lekim89/sportsmot`](https://hf-mirror.com/datasets/Lekim89/sportsmot) — `train/v_-6Os86HzwCs_c001/img1/000001.jpg` → `206`, 167 636 B | Yes: 20 train seqs `/6` 960 q85 → **0.46 GB** | **M** |
| SportsMOT-L | per-sequence text descriptions only | CC BY-NC 4.0 | 92 tiny json | 240 descriptions | json | [`hf-mirror.com/datasets/UHow/SportsMOT-L`](https://hf-mirror.com/datasets/UHow/SportsMOT-L) | — | N |
| SoccerNet-tracking | fast motion, long | research-only + registration | no reachable host | — | — | — | — | **X** |

**SportsMOT caveats (measured):** `det/det.txt` **does not exist for any sequence in any split** (`404`) — you must
run your own detector; `gt/gt.txt` exists for train+val but **not for the 150 test sequences**; sequence directory
names are best-effort (`val/v_-6Os86HzwCs_c001/` → `404`, while `val/v_00HRwkvvjtQ_c001/img1/000001.jpg` → `206`), so
probe before fetching. `MCG-NJU/SportsMOT` also carries `scripts/mot_to_coco.py`,
`scripts/sportsmot_to_trackeval.py`, `splits_txt/{train,val}.txt`.

### Axis 4 — Long-term / out-of-frame

| Name | Covers | License | Exact download size | #seq | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **LaSOT person** ⭐ **chosen** | long-term, heavy occlusion, **frequent out-of-frame** | **none on the mirror** (upstream research-only / non-commercial) | `person.zip` **4 288 656 486 B** (`206`); Zip64, **64 230 members** | 48 `person-*` sequences | `groundtruth.txt` = **4-corner polygon** txt (not xywh) | [`hf-mirror.com/datasets/l-lt/LaSOT/resolve/main/person.zip`](https://hf-mirror.com/datasets/l-lt/LaSOT/resolve/main/person.zip) | Yes → `/4` 960 q85 ⇒ **≈1.31 GB** *(projected from a 16 049-member dry run)* | N (fits) |
| GOT-10k | long-term, 10 k classes | CC BY-NC-SA 4.0 | **152.4 GB** per-frame chunks (`ls888/got10k`); Kaggle mini **6 682 669 678 B** (`200`) | 10 000 clips | own txt | [HF](https://hf-mirror.com/datasets/ls888/got10k) / [Kaggle mini](https://www.kaggle.com/api/v1/datasets/download/dilshan49/got10k-mini-1) | the mini is pre-cut | **X** (6.7 GB for one axis) |
| TrackingNet | long-term, 30 k videos | research-only | **30.7 GB** and **only `TEST/zips` exists** (`TRAIN_0…11` empty) | — | own txt | [`hf-mirror.com/datasets/SilvioGiancola/TrackingNet`](https://hf-mirror.com/datasets/SilvioGiancola/TrackingNet) | — | **X** (train absent, official site `404`) |
| UAV20L / VOT-LT | long-term aerial | research-only | no reachable source | — | — | — | — | **X** |
| UAV123 sample | aerial SOT sample | — | `data/train-00000-of-00001.parquet` **41 464 621 B** (`200`) | — | parquet | [`hf-mirror.com/datasets/zzz16/UAV123_sample`](https://hf-mirror.com/datasets/zzz16/UAV123_sample) | — | N |

### Axis 5 — Aerial / small-target / drone

| Name | Covers | License | Exact download size | #seq / #frames | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **VisDrone-MOT** ⭐ **chosen (only aerial hit)** | aerial, small targets, drone | CC BY-SA 4.0 | Kaggle `enansari/visdrone2019-mot-custom` **333 749 415 B** (`206`) | pre-cut MOT clips | MOTChallenge-style txt | [`…/enansari/visdrone2019-mot-custom`](https://www.kaggle.com/api/v1/datasets/download/enansari/visdrone2019-mot-custom) | already small | **M** |
| VisDrone-MOT (HF) | same | — | `Voxel51/visdrone-mot` = **415 491 327 B**, 1000 members (997 jpg + README) | 997 pre-cropped frames; **no MOT label file in-tree** *(this is why the Kaggle cut is preferred)* | jpg + FiftyOne parquet | [`hf-mirror.com/datasets/Voxel51/visdrone-mot`](https://hf-mirror.com/datasets/Voxel51/visdrone-mot) | already small | N |
| VisDrone-DET | aerial detection | CC BY-SA 3.0 (card) | `VisDrone2019-DET-val.zip` = 81 638 851 B (`206`, measured earlier); train/test-dev not obtainable this session | 6471 / 548 / 1610 imgs, 10 classes | YOLO via `visdrone2yolo()` | **GitHub release asset → `WinError 10060` timeout on all three zips.** `banu4prasad/VisDrone-Dataset` hosts images but its tree lists only 986 jpgs before paging *(unverified complete)* | yes | N |
| UAV123 | aerial SOT | research-only | `UAV123.tar.gz` **14 033 813 268 B** (`206`) | 123 seq | own txt | `hf-mirror.com/datasets/xche32/UAV123` | yes, from a 14 GB source | **X** |
| UAVDT (det) | aerial small targets | research-only | **522 764 713 B** (`200`) | — | — | [`…/foryolotrain1/uavdt-2024-det`](https://www.kaggle.com/api/v1/datasets/download/foryolotrain1/uavdt-2024-det) | yes | X (no person class verified) |
| DTB70 | drone | research-only | **no source found** on any reachable host | — | — | — | — | **X** |

### Axis 6 — Partial / occluded ReID

| Name | Covers | License | Exact download size | #identities | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **Occluded-DukeMTMC** ⭐ **chosen** | **partial / heavy-occlusion ReID — the partial-visibility requirement** | **UNRESOLVED — third-party redistribution, Kaggle `licenseName = Unknown`** (see §3) | `baofengz/occluded-duke` **144 186 775 B** (`206`); extracted **142.8 MB** / 35 492 members | **15 618 train / 17 661 gallery / 2 210 query** crops | Market-1501 layout: `bounding_box_train/`, `bounding_box_test/`, `query/` + `train.list`/`gallery.list`/`query.list` | [`…/baofengz/occluded-duke`](https://www.kaggle.com/api/v1/datasets/download/baofengz/occluded-duke) — **downloaded, extracted and JPEG-verified this session** | already small | **M** |
| Occluded-Duke split lists (official) | splits only | research-only | `gallery.list` 370 881 / `train.list` 327 978 / `query.list` 46 410 B | — | txt | [`github.com/lightas/Occluded-DukeMTMC-Dataset`](https://github.com/lightas/Occluded-DukeMTMC-Dataset) | — | M (pair with above) |
| **PRCC** ⭐ chosen | **clothes-changing ReID** | Kaggle card (research-only upstream) | `lyqassf/prccdata1` **310 560 124 B** (`206`) | 221 identities, camera pairs A/B/C | PRCC layout (`rgb/`, `ss/` + train/val/test lists) | [`…/lyqassf/prccdata1`](https://www.kaggle.com/api/v1/datasets/download/lyqassf/prccdata1) | already small | N (fits) |
| P-DukeMTMC | partial Duke | — | **no working mirror found** | — | — | — | — | **X** |
| Occluded-ReID | partial | — | **no working mirror found** | — | — | — | — | **X** |

### Axis 7 — Standard ReID

| Name | Covers | License | Exact download size | #ids / #boxes | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **Market-1501** ⭐ | standard ReID | research-only (no redistribution) | Kaggle **153 011 595 B** (`206`); HF `Lekim89/market1501` `Market-1501-v15.09.15.tar.gz` **110 051 423 B** (`206`) | 1 501 ids / 32 668 boxes / 6 cameras | Market-1501 layout | [`…/sachinsarkar/market1501`](https://www.kaggle.com/api/v1/datasets/download/sachinsarkar/market1501) or [`hf-mirror.com/datasets/Lekim89/market1501`](https://hf-mirror.com/datasets/Lekim89/market1501) | already small | **M** |
| **MSMT17 (compact)** ⭐ | standard ReID, largest/hardest | research-only | `achrafabid7/msmt17` **425 587 798 B** (`206`) | 4 101 ids / 126 441 boxes / 15 cameras | MSMT17 layout (`bounding_box_train/test`, `query`, `list_*.txt`) | [`…/achrafabid7/msmt17`](https://www.kaggle.com/api/v1/datasets/download/achrafabid7/msmt17) — **the budget form** | already small | **M** |
| MSMT17 (full) | same | research-only | `MSMT17_V1.zip` **2 556 892 994 B** (`206`) | same | same | [`hf-mirror.com/datasets/xianpeijie/MSMT17_V1`](https://hf-mirror.com/datasets/xianpeijie/MSMT17_V1) | yes | N (only if the compact one proves mislabelled) |
| DukeMTMC-reID | standard ReID | **withdrawn upstream** (Duke privacy withdrawal); mirror card *Unknown* | `whurobin/dukemtmcreid` **148 178 828 B** (`206`); `igorkrashenyi/dukemtmc-reid` 148 183 188 | 1 404 ids / 36 441 boxes | Market-1501 layout | [`…/whurobin/dukemtmcreid`](https://www.kaggle.com/api/v1/datasets/download/whurobin/dukemtmcreid) | already small | N — **recommend dropping, see §3** |
| CUHK03 | standard ReID | research-only | `priyanagda/cuhk03` **2 891 939 854 B** (`200`) / `hoangdccb/cuhk03` 1 755 370 254 | 1 467 ids | `.mat` / lists | [Kaggle](https://www.kaggle.com/api/v1/datasets/download/priyanagda/cuhk03) | yes | **X** (2.9 GB for a covered axis) |

### Axis 8 — Detection / crowd

| Name | Covers | License | Exact download size | #imgs / #boxes | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **CrowdHuman** ⭐ **chosen (val slice)** | **detect *all* persons** incl. heavy crowd occlusion | CC BY-NC 4.0 | `train01` **2 970 597 373** + `train02` 3 092 749 718 + `train03` 2 306 357 030 + `val` **2 488 658 160** + `test` 3 259 241 265 + `annotation_train.odgt` 80 017 502 + `annotation_val.odgt` **23 323 139** = **14.221 GB** (all `206`) | 15 000 train / 4 370 val, ~470 K boxes | **odgt** (JSON-lines: `{"ID","gtboxes":[{"tag","vbox","hbox","extra","head_attr"}]}`) | [`hf-mirror.com/datasets/Funtion/CrowdHuman`](https://hf-mirror.com/datasets/Funtion/CrowdHuman) — `val` + `annotation_val.odgt` verified | Yes — **val only**, `/4` 960 q85 ⇒ **0.90 GB** (dropping train01–03 saves 5.4 GB) | **M** |
| CrowdHuman (Kaggle, converted) | same | Kaggle card | `permanalwep/crowdhuman-crowd-detection` **870 738 309 B** (`206`) | — | YOLO txt | [`…/permanalwep/crowdhuman-crowd-detection`](https://www.kaggle.com/api/v1/datasets/download/permanalwep/crowdhuman-crowd-detection) | yes | N |
| **CityPersons (YOLO)** ⭐ | crowd detection, Cityscapes crops | research-only (upstream Cityscapes terms) | `hilariusjeremy/citypersons-v9-yolo` **279 783 840 B** (`206`) | 2 975 train / 500 val | YOLO txt | [`…/hilariusjeremy/citypersons-v9-yolo`](https://www.kaggle.com/api/v1/datasets/download/hilariusjeremy/citypersons-v9-yolo) | already small | N (fits) |
| CityPersons (official) | same | **login-gated** | `…/file-handling/?packageID=13` → `200` **HTML login form** | — | its own | — | — | **X** (not scriptable) |
| COCO 2017 person subset | detection + pose | CC BY 4.0 (annotations) / Flickr ToS (images) | `val2017.zip` **815 585 330** + `annotations_trainval2017.zip` **252 907 541** = **1 068 492 871 B** (both `206`) | val = 5 000 imgs; person-filtered ≈2 697 imgs / 36 935 boxes | COCO json | [`images.cocodataset.org/zips/val2017.zip`](http://images.cocodataset.org/zips/val2017.zip) + [`…/annotations/annotations_trainval2017.zip`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) | Yes — keep only images with a `person` annotation, 960 q85 ⇒ **≈0.40 GB** | N (fits) |
| WiderPerson | crowded-scene person detection | "other" (card) | 4 train parquet (177.3+154.4+157.2+149.3 MB) + `validation` 71 MB + `categories.csv` = **~709 MB**; Kaggle `imneonizer/widerperson-kitti` **707 009 647 B** (`206`) | 8 000 train / 1 000 val, 5 classes | parquet (HF) / KITTI txt (Kaggle) | [HF](https://hf-mirror.com/datasets/finedet/widerperson) / [Kaggle](https://www.kaggle.com/api/v1/datasets/download/imneonizer/widerperson-kitti) | yes | N |

### Axis 9 — Face (complementary cue)

| Name | Covers | License | Exact download size | Content | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **buffalo_l** ⭐ **chosen** | pretrained-ready face **detection + 512-d embedding** (`det_10g.onnx` SCRFD-10GF, `w600k_r50.onnx` ArcFace-R50) | **MIT** | `det_10g.onnx` **16 923 827 B** + `w600k_r50.onnx` **174 383 860 B** = **191 MB** — **both downloaded (`200`)** | 2 ONNX models | ONNX | [`hf-mirror.com/WePrompt/buffalo_l/resolve/main/det_10g.onnx`](https://hf-mirror.com/WePrompt/buffalo_l/resolve/main/det_10g.onnx) | — | **M** |
| insightface_weights (superset) | + 3D68 / 2D106 / genderage | apache-2.0 | `1k3d68.onnx` 143 607 619 + `2d106det.onnx` 5 030 888 + `genderage.onnx` 1 322 532 + the two above | 5 ONNX | ONNX | [`hf-mirror.com/karanjakhar/insightface_weights`](https://hf-mirror.com/karanjakhar/insightface_weights) | — | N |
| **LFW multi-faces** ⭐ **chosen** | small face-**verification** gallery/probe set | see card | `lfw_multifaces-ingestion.zip` **69 117 779 B** + `lfw_multifaces-retrieval.zip` **61 292 991 B** = **130 MB** — **both downloaded (`200`)**; 4 857 jpgs verified at 250×250 | 4 857 faces | jpg + repo lists | [`…/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-retrieval.zip`](https://hf-mirror.com/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-retrieval.zip) | already small | **M** |
| Glint360K | 17 M identities | non-commercial, request form | 100+ GB (`gaunernst/glint360k-wds-gz`) | 17 M ids | webdataset | [`hf-mirror.com/datasets/gaunernst/glint360k-wds-gz`](https://hf-mirror.com/datasets/gaunernst/glint360k-wds-gz) | — | **X** |
| MS1M-ArcFace / ms1mv3 | 5.1 M identities | non-commercial | **48.99 GB** in 102 × 0.49 GB shards | 93 K ids | webdataset | [`hf-mirror.com/datasets/gaunernst/ms1mv3-wds`](https://hf-mirror.com/datasets/gaunernst/ms1mv3-wds) | **yes — one 0.49 GB shard is a usable subset** | N (optional) |
| CASIA-WebFace | 10 575 identities | research-only | `data/train.rec` **2 726 043 620 B** (`206`) | 494 414 imgs | MXNet recordio + `.lst`/`.tsv`/`.idx` | [`hf-mirror.com/datasets/Pijush22049/CASIA-webface`](https://hf-mirror.com/datasets/Pijush22049/CASIA-webface) | yes | X (needs `mxnet`) |
| WebFace260M | 260 M imgs | registration + licence | no reachable host | — | — | — | — | **X** |
| WIDER FACE | face detection | research-only | `wider_face.zip` **3 283 480 091 B** (`206`) | 32 203 imgs | YOLO txt | [`hf-mirror.com/datasets/Bingsu/wider_face_yolo`](https://hf-mirror.com/datasets/Bingsu/wider_face_yolo) | yes | X |

**Dead end recorded:** the GitHub release zips under `deepinsight/insightface` (buffalo_sc/s/m/l, antelopev2) are
**unreachable** from this machine (TLS handshake / `RemoteDisconnected` / timeout on every attempt) — use the ONNX
mirrors above. Likewise `immich-app/buffalo_{l,s,m}` and `immich-app/scrfd_34g_gnkps` exist as alternate sources.

### Axis 10 — Pose (for part crops)

| Name | Covers | License | Exact download size | #imgs / #persons | Annotation format | Working URL (verified) | Sub? | Pri |
|---|---|---|---|---|---|---|---|---|
| **OCHuman** ⭐ **chosen** | **heavily-occluded person** boxes + keypoints — the partial-visibility case | MIT (pose repo card) | `OCHuman.zip` **719 454 950 B** (`206`); Kaggle mirror **734 362 777 B** (`200`) | 5 081 imgs / 13 860 persons | COCO json (`ochuman.json`) | [HF](https://hf-mirror.com/datasets/ayh015/OCHuman) / [Kaggle](https://www.kaggle.com/api/v1/datasets/download/messinaalberto/ochuman) | yes → `/2` 960 q85 ⇒ **≈0.35 GB** | **M** |
| OCHuman-Pose annotations | COCO-format keypoints | **MIT** | `ochuman-pose_coco_format_val.json` **3 867 783 B** (`206`) + `..._test.json` **3 462 444 B** (`206`) | 1 340 val / 2 500 test | COCO json | [`hf-mirror.com/datasets/vrg-prague/OCHuman-Pose`](https://hf-mirror.com/datasets/vrg-prague/OCHuman-Pose) | — | M (pair with the images) |
| COCO-pose person subset | general pose | CC BY 4.0 | `val2017.zip` **815 585 330** (`206`); `val2017.txt` lists **2 346** pose-annotated imgs | 17 kpts, single `person` class | COCO json | [`images.cocodataset.org/zips/val2017.zip`](http://images.cocodataset.org/zips/val2017.zip) | yes — 2 346 imgs `/2` 960 q85 ⇒ **≈0.15 GB** | N |
| PoseTrack / PoseTrack21 | multi-person video pose + tracking | research-only; `posetrack.net` **DNS-dead** | Kaggle `antocad/posetrack-fod` **532 982 917 B** (`206`) — contents *(unverified)* | — | — | [`…/antocad/posetrack-fod`](https://www.kaggle.com/api/v1/datasets/download/antocad/posetrack-fod) | yes | N (only if contents check out) |
| CrowdPose | crowded pose | research-only | `crowdpose_000{0,1,2}.tar` 10 250 240 B each (listing incomplete) | 20 K imgs | COCO json | [`hf-mirror.com/datasets/KangLiao/CrowdPose-Camera`](https://hf-mirror.com/datasets/KangLiao/CrowdPose-Camera) | — | X (listing not trustworthy) |

### Model weights (needed to *run* any of the above)

| File | License | Exact size | Working URL |
|---|---|---|---|
| `yolo26n.pt` | AGPL-3.0 | **5 544 453 B** — **downloaded (`200`)** | [`hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n.pt`](https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n.pt) |
| `yolo26{s,m,l,x}.pt` and `-cls/-obb/-pose/-seg` | AGPL-3.0 | 27 files; `yolo26m` **53 211 173 B**, `yolo26l-obb` 57 263 057, `yolo26l-pose` **57 995 961** | same repo |
| ONNX variants | — | `onnx-community/yolo26n-ONNX` (11 variants: fp16/int8/q4); `unileon-robotics/YOLO26-ONNX` (77 files, `dynamic/yolo26*.onnx`); `NexaAI/yolo26*-npu` | [`hf-mirror.com/onnx-community/yolo26n-ONNX`](https://hf-mirror.com/onnx-community/yolo26n-ONNX) |
| ArcFace recognition models | — | `minchul/cvlface_arcface_ir101_webface4m`, `gaunernst/vit_tiny_patch8_112.arcface_ms1mv3` | hf-mirror |

> **`huggingface_hub` path verified working through the mirror:**
> `os.environ['HF_ENDPOINT']='https://hf-mirror.com'` then
> `hf_hub_download(repo_id='WePrompt/buffalo_l', filename='det_10g.onnx', local_dir=…)` succeeded (16 923 827 B).

---

## 2. What `ultralytics 8.4.185` can auto-download (read from the wheel)

Read with `python -c "import zipfile; …"` from `C:\Projects\track\_p2\ultralytics-8.4.185-py3-none-any.whl`.

* `ultralytics/cfg/datasets/` holds **58** yaml files. Full list: `Argoverse.yaml`, `DOTAv1.yaml`, `DOTAv1.5.yaml`,
  `GlobalWheat2020.yaml`, `HomeObjects-3K.yaml`, `ImageNet.yaml`, `Objects365.yaml`, `SKU-110K.yaml`, `TT100K.yaml`,
  `VOC.yaml`, **`VisDrone.yaml`**, `ade20k.yaml`, `african-wildlife.yaml`, `brain-tumor.yaml`, `carparts-seg.yaml`,
  `cityscapes.yaml`, `cityscapes8.yaml`, **`coco-pose.yaml`**, **`coco.yaml`**, `coco12-formats.yaml`, `coco128.yaml`,
  `coco128-seg.yaml`, **`coco8.yaml`**, `coco8-grayscale.yaml`, `coco8-multispectral.yaml`, `coco8-pose.yaml`,
  `coco8-seg.yaml`, `construction-ppe.yaml`, `crack-seg.yaml`, `depth-*`, `depth8.yaml`, `dog-pose.yaml`,
  `dota128.yaml`, `dota8.yaml`, `hand-keypoints.yaml`, `kitti.yaml`, `lvis.yaml`, `medical-pills.yaml`,
  `nyu-depth.yaml`, `open-images-v7.yaml`, `package-seg.yaml`, `signature.yaml`, `tiger-pose.yaml`, **`xView.yaml`**.
* **There is NO MOT dataset yaml anywhere in the wheel.** A `zipfile` grep for `mot|MOT` over the member names
  returned `[]`. ⇒ **MOT17, MOT20, MOT16, DanceTrack and SportsMOT have no ultralytics auto-download path at all.**
  That is exactly why the toolkit in §4 exists.
* `VisDrone.yaml` → VisDrone2019-**DET** (not MOT): path `VisDrone`, train 6471 / val 548 / test-dev 1610 images,
  10 classes (`0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus,
  9 motor`), downloads `{ASSETS_URL}/VisDrone2019-DET-{train,val,test-dev}.zip` then converts with `visdrone2yolo()`.
  **`ASSETS_URL` is a GitHub release asset, which times out from this machine** → `yolo … VisDrone.yaml` is *not
  currently usable here*.
* `coco-pose.yaml` → COCO2017 keypoints, path `coco-pose` (~20.2 GB), train 56 599 / val 2 346 / test-dev 20 288,
  `kpt_shape: [17,3]`, single class `person`; downloads `coco2017labels-pose.zip` + `images.cocodataset.org` zips.
* `coco.yaml` (~20.3 GB) and `coco8.yaml` (1 MB) also resolve to GitHub release assets → same timeout risk.
* `ultralytics/cfg/trackers/` = `botsort.yaml`, `bytetrack.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `ocsort.yaml`,
  `tracktrack.yaml` — **tracker configs, not datasets**.
* `ultralytics/cfg/models/26/` = `yolo26(-p2/-p6/-cls/-depth/-obb/-pose/-seg/-sem).yaml`, `yoloe-26`,
  `yoloe-26-seg` — so **`yolo26n-pose.pt`** is the pose backbone; take it from the HF mirror above, not the release.

**Bottom line for the brief's question:** ultralytics auto-downloads **none** of the ten axes this project needs,
except the COCO2017 *val* images (person subset) and — in principle — VisDrone-DET, whose host is unreachable here.

---

## 3. Licensing / click-through and fallbacks

| Dataset | Gate | Verdict / fallback |
|---|---|---|
| MOT17 / MOT20 / MOT16 / 2DMOT2015 | MOTChallenge terms (research + non-commercial, citation required) | **No click-through observed** — `/data/*.zip` returns a real ZIP directly. Keep the CC BY-NC-SA 3.0 attribution. |
| DanceTrack | CC BY 4.0 upstream | No gate. `AbdoTW/DanceTrack_1of3` **declares no licence** — cite upstream and keep CC BY 4.0 terms. |
| SportsMOT | CC BY-NC 4.0 | No gate; non-commercial. |
| CrowdHuman | CC BY-NC 4.0 | No gate; non-commercial. |
| **Occluded-DukeMTMC** | **Officially unavailable.** The upstream README states verbatim: *"Since the privacy implications of the DukeMTMC dataset are being considered, we cannot release the images of Occluded-DukeMTMC. We only provide the image name lists."* Its converter needs the withdrawn `DukeMTMC-reID.zip`. | The images came from a **Kaggle third-party redistribution whose `licenseName` is `Unknown`**, on top of a privacy-withdrawn parent dataset. Functionally complete and JPEG-verified, but **treat as licensing-unclear: internal research only, do not redistribute, do not publish crops.** Fallback if that risk is unacceptable: take the **official split lists** and synthesise an occlusion protocol over Market-1501 / MSMT17. |
| DukeMTMC-reID | **Withdrawn upstream** | Kaggle mirrors work (`148 178 828 B`) under the same caveat. **Recommend dropping** — Market-1501 + MSMT17 + PRCC already cover standard ReID. |
| MSMT17 / Market-1501 / CUHK03 / PRCC | research-only, no redistribution | Kaggle serves them without a gate. Keep local. |
| CityPersons | upstream Cityscapes terms; official portal is a **login-gated form** | Use the **Kaggle YOLO conversion** (`279 783 840 B`) — no gate. Loss: no raw instance masks. |
| COCO 2017 | CC BY 4.0 annotations / Flickr ToS images | No gate. |
| LaSOT | research-only, non-commercial; upstream site is `404` | hf-mirror serves `person.zip`; the mirror declares no licence ⇒ research-only. |
| VisDrone | CC BY-SA 4.0 | No gate on the Kaggle mirror. |
| WIDER FACE / CASIA-WebFace / Glint360K / MS1M-ArcFace / WebFace260M | request form / registration / non-commercial | **No working host within budget.** Use `buffalo_l` (MIT) + LFW — covers the face cue with zero licensing friction. |
| PoseTrack | official site DNS-dead | Only the *(unverified)* Kaggle `antocad/posetrack-fod`; use OCHuman + COCO-pose. |
| `yolo26*.pt` | **AGPL-3.0** | Fine for research; note the copyleft if you ship. |

---

## 4. Acquisition plan — **8.25 GB selected, 1.75 GB headroom**

Sizes marked ✔ are measured; others are projected from the measured ≈0.42 re-encode ratio and the archive's own
central-directory member sizes.

| # | Dataset | Axis | Source | Raw size | Treatment | **On-disk** |
|---|---|---|---|---|---|---|
| 1 | DanceTrack 1-of-3 (`train1`+`train2`+`val`) | 2 nonlinear | hf-mirror | 3.71 GB ✔ | already every 3rd; keep **every 4th** of those, 960 q85, keep all `gt.txt`/`seqinfo.ini` | **1.25 GB** |
| 2 | SportsMOT train (20 seqs) | 3 fast/abrupt | hf-mirror per-file | — | every 6th frame, 960 q85 | **0.46 GB** |
| 3 | MOT17 FRCNN train+test | 1 standard MOT | motchallenge ✔ | 5.86 GB ✔ | every 4th, 960 q85; keep `gt.txt`/`det.txt`/`seqinfo.ini` | **0.30 GB** |
| 4 | MSMT17 (compact) | 7 ReID | Kaggle ✔ | 0.426 GB ✔ | as shipped | **0.43 GB** |
| 5 | CityPersons (YOLO) | 8 detection | Kaggle ✔ | 0.280 GB ✔ | as shipped | **0.28 GB** |
| 6 | CrowdHuman `val` only | 8 detection | hf-mirror ✔ | 2.49 GB ✔ | every 4th image, 960 q85; keep `annotation_val.odgt` whole | **0.90 GB** |
| 7 | COCO 2017 **person subset** | 8 + 10 | images.cocodataset.org ✔ | 1.068 GB ✔ | keep only images with a `person` annotation, 960 q85 | **0.40 GB** |
| 8 | OCHuman (images + COCO keypoints) | 10 / 6 | hf-mirror ✔ | 0.719 GB ✔ | every 2nd image, 960 q85 | **0.35 GB** |
| 9 | VisDrone-MOT | 5 aerial | Kaggle ✔ | 0.334 GB ✔ | as shipped | **0.33 GB** |
| 10 | buffalo_l ONNX models | 9 face | hf-mirror ✔ | 0.191 GB ✔ | — | **0.19 GB** |
| 11 | LFW multi-faces | 9 face | hf-mirror ✔ | 0.130 GB ✔ | — | **0.13 GB** |
| 12 | YOLO26 `n` + `n-pose` + `s` | runtime | hf-mirror ✔ | 0.117 GB | — | **0.12 GB** |
| 13 | Occluded-DukeMTMC | 6 partial/occluded | Kaggle ✔ | 0.144 GB ✔ | already extracted and verified | **0.14 GB** |
| — | **SUBTOTAL** | | | **≈14.9 GB of downloads** | | **5.22 GB** |
| 14 | LaSOT person | 4 long-term | hf-mirror ✔ | 4.289 GB ✔ | every 4th frame, 960 q85 | 1.31 GB |
| 15 | Market-1501 | 7 ReID | Kaggle ✔ | 0.153 GB ✔ | as shipped | 0.15 GB |
| 16 | PRCC (clothes-changing) | 6 ReID | Kaggle ✔ | 0.311 GB ✔ | as shipped | 0.31 GB |
| 17 | MOT20 | 1 standard MOT | motchallenge ✔ | 5.029 GB ✔ | every 4th, 960 q85 | 0.13 GB |
| 18 | MOT16 | 1 standard MOT | motchallenge ✔ | 1.955 GB ✔ | every 4th, 960 q85 | 0.27 GB |
| 19 | WiderPerson | 8 detection | hf-mirror/Kaggle ✔ | 0.709 GB ✔ | as shipped | 0.71 GB |
| — | **OPTIONAL TIER SUBTOTAL** | | | | | **3.03 GB** |
| | **GRAND TOTAL (13 + optional)** | | | | | **8.25 GB** |

**Verdict: 8.25 GB decimal, 7.68 GiB — under a 10 GB decimal cap by 1.75 GB and under a 10 GiB cap by 2.3 GiB.**
That is enough headroom for one extra mid-size item (e.g. a single 0.49 GB `ms1mv3` shard, or the
`antocad/posetrack-fod` 0.53 GB) but **not** for DukeMTMC-reID *and* CUHK03 *and* extra CrowdHuman train shards.
Re-check with `python _p2\r4_acquire.py verify` before adding anything.

**Compared to the two-archive set in `R4_datasets.md` (10.07 GB — 0.7 % over a decimal cap):** this plan covers
**eight more diversity axes for 1.8 GB less**, precisely because it never stores a multi-GB archive it does not need.

### 4.1 Exact commands (PowerShell — all path forms verified)

Set these once per shell:

```powershell
$env:PYTHONIOENCODING = "utf-8"           # this console is GBK; non-ASCII output crashes Python otherwise
$env:HF_ENDPOINT       = "https://hf-mirror.com"
Set-Location C:\Projects\track
```

```powershell
# ---------- Kaggle (no API key; Range is ignored so these are full downloads) ----------
curl.exe -L -C - -o _p2\_dl\Occluded-DukeMTMC.zip "https://www.kaggle.com/api/v1/datasets/download/baofengz/occluded-duke"
curl.exe -L -C - -o _p2\_dl\Market1501.zip        "https://www.kaggle.com/api/v1/datasets/download/sachinsarkar/market1501"
curl.exe -L -C - -o _p2\_dl\MSMT17_compact.zip    "https://www.kaggle.com/api/v1/datasets/download/achrafabid7/msmt17"
curl.exe -L -C - -o _p2\_dl\PRCC.zip              "https://www.kaggle.com/api/v1/datasets/download/lyqassf/prccdata1"
curl.exe -L -C - -o _p2\_dl\CityPersons_yolo.zip  "https://www.kaggle.com/api/v1/datasets/download/hilariusjeremy/citypersons-v9-yolo"
curl.exe -L -C - -o _p2\_dl\VisDrone_MOT.zip      "https://www.kaggle.com/api/v1/datasets/download/enansari/visdrone2019-mot-custom"
curl.exe -L -C - -o _p2\_dl\DukeMTMCreID.zip      "https://www.kaggle.com/api/v1/datasets/download/whurobin/dukemtmcreid"

# ---------- huggingface_hub through the mirror (verified working) ----------
python -c "import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='WePrompt/buffalo_l', filename='det_10g.onnx', local_dir=r'C:\Projects\track\data\_models'))"

# ---------- model weights: note there is NO '/datasets/' segment ----------
curl.exe -L -o data\_models\yolo26n.pt      "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n.pt"
curl.exe -L -o data\_models\yolo26s.pt      "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26s.pt"
curl.exe -L -o data\_models\yolo26n-pose.pt "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/yolo26n-pose.pt"
curl.exe -L -o data\_models\det_10g.onnx    "https://hf-mirror.com/WePrompt/buffalo_l/resolve/main/det_10g.onnx"
curl.exe -L -o data\_models\w600k_r50.onnx  "https://hf-mirror.com/WePrompt/buffalo_l/resolve/main/w600k_r50.onnx"

# ---------- face verification set (dataset repo, same URL form) ----------
curl.exe -L -o data\_dl\lfw-ingestion.zip "https://hf-mirror.com/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-ingestion.zip"
curl.exe -L -o data\_dl\lfw-retrieval.zip "https://hf-mirror.com/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-retrieval.zip"

# ---------- COCO 2017 val + annotations (then subset to person) ----------
curl.exe -L -o data\_dl\val2017.zip                  "http://images.cocodataset.org/zips/val2017.zip"
curl.exe -L -o data\_dl\annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ---------- OCHuman images + COCO-format keypoints ----------
curl.exe -L -o data\_dl\OCHuman.zip                "https://hf-mirror.com/ayh015/OCHuman/resolve/main/OCHuman.zip"
curl.exe -L -o data\OCHuman\ochuman-pose_val.json  "https://hf-mirror.com/vrg-prague/OCHuman-Pose/resolve/main/ochuman-pose_coco_format_val.json"
curl.exe -L -o data\OCHuman\ochuman-pose_test.json "https://hf-mirror.com/vrg-prague/OCHuman-Pose/resolve/main/ochuman-pose_coco_format_test.json"
```

### 4.2 Big video datasets — the toolkit (streams; never stores the archive)

Run these **synchronously**. A PowerShell `Start-Job` child is killed when its `pwsh` wrapper exits, so do not rely
on background jobs for them.

```powershell
# MOT17: 5.86 GB zip -> train+test FRCNN, every 4th frame, max side 960, q85  (~0.30 GB out, 25-40 min)
python _p2\r4_acquire.py mot17 --every 4

python _p2\r4_acquire.py mot20 --every 4        # ~5.03 GB zip -> ~0.13 GB
python _p2\r4_acquire.py mot16 --every 4        # ~1.95 GB zip -> ~0.27 GB

# DanceTrack: block-walks all three 1/3-subsampled zips, keeps every 4th of those  (~45 min, ~1.25 GB)
python _p2\run_dancetrack.py

# SportsMOT: gt.txt + seqinfo.ini for all 45 train seqs, every 6th frame of 20 of them  (~0.46 GB)
python _p2\r4_acquire.py sportsmot --every 6

# CrowdHuman validation only (never train01-03)  (~0.90 GB out of a 2.49 GB archive)
python _p2\r4_acquire.py CrowdHuman_val --every 4

# LaSOT person, every 4th frame  (~1.3 GB out of a 4.29 GB archive)
python _p2\r4_acquire.py LaSOT_person --every 4

# OCHuman, every 2nd image
python _p2\r4_acquire.py OCHuman --every 2

# what do I actually have?
python _p2\r4_acquire.py verify
```

All toolkit runs are **resumable** (existing files are skipped) and both extractors back off exponentially on
`HTTP 429` / `503`.

### 4.3 The down-sampling recipe (no ffmpeg, no archive kept)

Measured output/input ratio for **max side 960, JPEG q85** is **≈ 0.42** on MOT17 / DanceTrack / LaSOT frames.

```python
# every k-th frame of a folder tree -> max side 960, q85, in place; gt.txt / det.txt / *.ini untouched
import os, sys
sys.path.insert(0, r"C:\Projects\track\_p2")
from r4_acquire import save_jpeg

def subsample(root, k=4, max_side=960, q=85):
    for dp, _, fs in os.walk(root):
        for i, f in enumerate(sorted(fs)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(dp, f)
            if i % k:
                os.remove(p)
            else:
                save_jpeg(open(p, "rb").read(), p, max_side)
```

For the **COCO person subset**, filter on the annotation rather than on frame index:

```powershell
python -c @"
import json, zipfile, sys
sys.path.insert(0, r'C:\Projects\track\_p2'); from r4_acquire import save_jpeg
ann = json.load(open(r'C:\Projects\track\data\_dl\annotations_trainval2017\annotations\instances_val2017.json'))
keep = {a['image_id'] for a in ann['annotations'] if a['category_id'] == 1}          # 1 = person
ids  = {i['id']: i['file_name'] for i in ann['images'] if i['id'] in keep}
print('person images:', len(ids))
with zipfile.ZipFile(r'C:\Projects\track\data\_dl\val2017.zip') as z:
    for fn in ids.values():
        save_jpeg(z.read('val2017/' + fn), r'C:\Projects\track\data\COCO_person\val2017\' + fn, 960)
json.dump([{'image_id': i, 'file_name': f} for i, f in ids.items()], open(r'C:\Projects\track\data\COCO_person\images.json', 'w'))
"@
```

**Projected outputs (ratio ≈0.42):**

| Input | Frames | Treatment | ~Output |
|---|---|---|---|
| MOT17, 1 detector variant (5.92 GB img) | 33 705 | `/4` → 8 427 | **0.30 GB** |
| MOT20 (5.01 GB img) | 13 410 | `/4` → 3 353 | **0.13 GB** |
| MOT16 (~2.1 GB img) | 11 235 | `/4` → 2 809 | **0.27 GB** |
| DanceTrack 1-of-3 (3.75 GB) | ~25 200 | `/4` → ~6 300 + all GT | **1.25 GB** |
| SportsMOT (6.6 GB tar or per-file) | ~150 000 | 20 train seqs `/6` → ~2 750 | **0.46 GB** |
| CrowdHuman `val` (2.49 GB) | 4 370 | `/4` → 1 093 + full `.odgt` | **0.90 GB** |
| LaSOT person (4.29 GB, 64 230 members) | 64 230 | `/4` → 16 049 members | **1.31 GB** |
| OCHuman (719 MB) | 5 081 | `/2` → 2 541 | **0.35 GB** |
| COCO val → person | ≈2 697 | `/1`, 960 q85 | **0.40 GB** |

---

## 5. Datasets dropped, and why

| Dropped | Reason |
|---|---|
| SoccerNet-tracking | no reachable host |
| GOT-10k | 152 GB on HF; the only Kaggle cut is 6.68 GB for a single axis |
| TrackingNet | 30.7 GB **and** `TRAIN_0…11` are empty; official site `404` |
| UAV20L, VOT-LT, DTB70 | no reachable source (sites DNS-dead / no mirror) |
| UAV123 (full) | 14.03 GB archive |
| CityPersons (official) | login-gated form; use the Kaggle YOLO conversion instead |
| CUHK03 | 2.9 GB (Kaggle) for an axis already covered |
| P-DukeMTMC, Occluded-ReID, PRW | **no working mirror on hf-mirror, Kaggle or GitHub** |
| Glint360K, MS1M-ArcFace (full), WebFace260M, WIDER FACE, CASIA-WebFace | 3–100 GB plus registration/non-commercial gates; `buffalo_l` (MIT, 191 MB) + LFW (130 MB) covers the face axis |
| PoseTrack / PoseTrack21 | official site DNS-dead; only an *(unverified)* 533 MB Kaggle mirror |
| CrowdPose | HF listing incomplete/implausible (3 × 10 MB) |
| VisDrone-**DET**, coco8, coco-pose labels via ultralytics | every `github.com/…/releases/download/…` attempt timed out (`WinError 10060`) |
| `deepinsight/insightface` release zips (buffalo_sc/s/m/l, antelopev2) | TLS handshake / `RemoteDisconnected` on every attempt |

## 6. Open items / what is still (unverified) in this document

1. `LaSOT_person`, `CrowdHuman_val` and `OCHuman` were **path-verified and started but not allowed to finish** inside
   this session (hf-mirror rate-limiting plus a competing 18.5 GB job on the machine). Their on-disk figures are
   **projections** from the measured ≈0.42 ratio and the archives' real member sizes — the mechanisms are verified,
   the totals are not measured.
2. `antocad/posetrack-fod` (533 MB) contents were never opened.
3. VisDrone-DET needs a non-GitHub mirror before `ultralytics`'s `VisDrone.yaml` is usable on this machine.
4. `DukeMTMC-reID` and `Occluded-DukeMTMC` images are privacy-withdrawn upstream — **get an explicit policy decision
   before shipping anything trained on them** (§3).
