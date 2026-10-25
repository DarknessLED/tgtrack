# R4 — Dataset survey and acquisition record

**Author:** research-analyst subagent · **Date:** 2026-10-24
**Scope:** requirement 2 of the project — *"download person-tracking datasets for experiments; diverse; total < 10 GB; images may be compressed."* This is the acquisition record plus the coverage analysis: what MOT17 and DanceTrack each actually test, the annotation formats, the storage recipes, and the honest gaps.
**Not in scope:** new downloads. `data/raw/` was read-only for this report.

> **See also `R4b_datasets_inventory.md`** — a companion inventory from a later subagent that covers the *other*
> diversity axes (nonlinear / fast / long-term / aerial / occluded-ReID / face / pose), inventories what
> `ultralytics 8.4.185` can auto-download (answer: no MOT yaml exists in the wheel), and proposes an 8.25 GB
> selection across ten axes — i.e. eight more axes than the two archives here for ~1.8 GB less, because it never
> stores a multi-GB archive it can range-slice instead. Read it before planning a third acquisition.

**Evidence policy.** `[V]` = I measured or probed it myself in this session, with the command shown. `[R]` = reported to me by the Lead, re-verified by probe where possible. `UNVERIFIED` = could not confirm, with the reason. Every size and licence below carries a URL. Note a live correction to the brief I was given: **the "MB" figures in the Lead's table are MiB** — see §1.2, which changes the budget verdict.

---

## 0. TL;DR

1. **The acquisition set is complete and byte-verified: 2 archives, 10,069,999,615 B.** `MOT17.zip` = 5,860,214,001 B and DanceTrack `val.zip` = 4,209,785,614 B; both match the server `Content-Length` exactly. `[V]`
2. **The 10 GB verdict depends on the unit, and on a decimal reading the set is 0.7% OVER budget.** 10,069,999,615 B = **9.378 GiB** (6.2% under a 10 GiB cap) but **10.070 GB** decimal (over a 10 GB cap by 70.0 MB). The Lead's "9.60 GB ≈ 4% under" is 9,603.5 **MiB**. `[V]` §1.2.
3. **The MOT17 page has a live correction to the environment brief:** `motchallenge.net/` and `/data/` return **410**, but per-dataset archive pages return **200** — `/data/MOT17/` (28,128 B) and `/data/MOT20/` (10,714 B) are readable, and `/data/HT21/` is 404. The retired site is a static archive, not a wall. `[V]` §1.5.
4. **DanceTrack is the non-linear-motion test and MOT17 is not, and I have the numbers.** Mean per-frame heading change: DanceTrack val **46.98°** vs MOT17 train **23.34°** (2.01×); fraction of frames with a >90° direction reversal: **18.1%** vs **4.2%** (4.3×). Scale-normalised speed is comparable (0.040 vs 0.030 box-heights/frame), so the difference is *non-linearity*, not speed. `[V]` §2.3.
5. **DanceTrack cannot test partial visibility at all.** Its `gt.txt` `visibility` column is the constant `1` for all 225,148 val rows (and `conf`/`class` are constant `1` too) — the column exists but carries no information. MOT17's `visibility` is genuinely populated with a full 0.0–1.0 spread. `[V]` §3.1.
6. **Neither dataset covers the aerial/UAV viewpoint, and no aerial dataset fits the remaining headroom.** The cheapest real VisDrone-MOT option is 1.5 GB (`VisDrone2019-MOT-val.zip`) vs ~0.4–0.6 GB headroom. The one aerial item that does fit is 39.5 MB but is a SOT sample, not a tracking benchmark. `[V]` §6.
7. **`ffmpeg` is on the machine twice, and the obvious one is a trap.** `C:\Software\Trae CN\resources\app\bin\ffmpeg.exe` advertises itself as 6.1.1 but is a `--disable-everything` build with **no `image2` demuxer, no `mjpeg` encoder and no `select` filter** — it cannot read or write the JPEG sequences this project is made of. Use **`C:\Software\oopz\ffmpeg.exe`** (gyan.dev essentials), verified working on the real MOT17 frames. `ffprobe.exe` exists only beside the Trae build; it is fine for probing mp4 output. `cv2.VideoWriter` fourcc `mp4v` is a verified no-ffmpeg fallback. `[V]` §4.2.
8. **A `data/processed/` working set of ~1.9 GB serves everything**: keep full-rate 640px JPEG q85 and let `subsample=k` select temporal density at load time, instead of baking subsampling into the stored files. `[V]` §4.5.

---

## 1. Acquisition record

### 1.1 The chosen set

Two archives were selected and are now fully on disk under `C:\Projects\track\data\raw\`.

| # | file on disk | source URL | bytes `[V]` | MiB | GiB |
|---|---|---|---|---|---|
| 1 | `data/raw/MOT17.zip` | `https://motchallenge.net/data/MOT17.zip` | 5,860,214,001 | 5,588.7 | 5.458 |
| 2 | `data/raw/dancetrack_val.zip` | `https://hf-mirror.com/datasets/noahcao/dancetrack/resolve/main/val.zip` | 4,209,785,614 | 4,014.8 | 3.921 |
| | **total** | | **10,069,999,615** | **9,603.5** | **9.378** |

Both were confirmed complete by comparing the on-disk byte size against an independent `HEAD` probe of the origin server. `MOT17.zip` matches `motchallenge.net`'s `Content-Length: 5860214001`. `dancetrack_val.zip` matches `hf-mirror.com`'s tree-API size for `val.zip`, `4209785614` bytes. `[V]`

The DanceTrack destination is deliberately named `dancetrack_val.zip` rather than `val.zip` so it is self-describing on disk.

### 1.2 The 10 GB arithmetic — and the unit problem

The sum is exact. Whether it is *under budget* is not a single answer, because "10 GB" was never pinned to a unit:

| reading of the cap | cap in bytes | acquired | verdict |
|---|---|---|---|
| 10 GiB (what Windows/PowerShell calls "GB") | 10,737,418,240 | 9.378 GiB | **under** by 636.5 MiB (6.2%) |
| 10,000 MiB (the reading the Lead's "9.60 GB" implies) | 10,485,760,000 | 9,603.5 MiB | **under** by 396.5 MiB (3.8%) |
| 10 GB decimal | 10,000,000,000 | 10.070 GB | **OVER** by 69,999,615 B (0.7%) |

`[V]` all three lines.

This is worth recording rather than averaging away, because it decides whether a third acquisition is possible. The safest operational conclusion: **treat the remaining headroom as 0.4 GB on the Lead's own arithmetic and 0.64 GiB on Windows' — and treat the decimal reading as a live risk.** If the requirement is enforced by a decimal `du`-style size, the fix is cheap: subsample one of the two archives before it counts, or drop to DanceTrack `test2.zip`/`train2.zip` scale (2.7–3.1 GiB) instead of `val.zip`. If it is enforced by Windows Explorer's "GB", there is 636 MiB left.

**Recommendation: do not add a third large archive.** §6 lists what would fit.

### 1.3 Exact command used

The Lead's background job downloaded sequentially with `curl.exe`. The reproducible form is:

```powershell
# MOT17 (motchallenge.net; curl.exe only - Invoke-WebRequest times out on these)
curl.exe -L --fail --retry 3 --retry-delay 5 -o "C:\Projects\track\data\raw\MOT17.zip" `
         "https://motchallenge.net/data/MOT17.zip"

# DanceTrack val (huggingface.co is BLOCKED; hf-mirror.com works)
$env:HF_ENDPOINT = "https://hf-mirror.com"
curl.exe -L --fail --retry 3 --retry-delay 5 -o "C:\Projects\track\data\raw\dancetrack_val.zip" `
         "https://hf-mirror.com/datasets/noahcao/dancetrack/resolve/main/val.zip"
```

Probed throughput to both hosts is 4.5–5.2 MB/s `[R]`, so 10.07 GB is ~35–40 minutes of wall clock. `-L` is required for the hf-mirror path (it redirects to a CDN); `--fail` avoids writing an HTML error page into the archive, which is the failure mode that silently corrupts these downloads.

I did **not** run these downloads — they were already in flight. I verified the results (§1.1, §1.4).

### 1.4 Verification plan (hash / size)

There is no published checksum for either archive, so verification is layered:

1. **Size match against origin** (already done, `[V]`): compare local byte length to the server `Content-Length` / tree-API size. Done for both; exact match.
2. **Archive integrity via CRCs** — a zip carries a CRC-32 per member, so this catches truncation, bit-rot and partial downloads without a published hash:
   ```powershell
   # MOT17.zip: reads all 5.86 GB, ~1-2 min on this disk
   python -c "import zipfile,sys; z=zipfile.ZipFile(r'C:\Projects\track\data\raw\MOT17.zip'); bad=z.testzip(); print('MOT17.zip CORRUPT at',bad) if bad else print('MOT17.zip OK, members=',len(z.namelist()))"
   python -c "import zipfile; z=zipfile.ZipFile(r'C:\Projects\track\data\raw\dancetrack_val.zip'); bad=z.testzip(); print('dancetrack_val.zip CORRUPT at',bad) if bad else print('dancetrack_val.zip OK, members=',len(z.namelist()))"
   ```
   The expected member count for `dancetrack_val.zip` is **25,634** `[V]` — a useful independent check. `MOT17.zip`'s member count is **UNVERIFIED** (I did not read its central directory; extraction was still in progress).
3. **Record a local manifest** for reproducibility, since no upstream hash exists:
   ```powershell
   Get-FileHash C:\Projects\track\data\raw\*.zip -Algorithm SHA256 |
     Format-Table Hash,Path -AutoSize | Out-File C:\Projects\track\data\raw\SHA256SUMS.txt
   ```
   `UNVERIFIED` — I did not compute these hashes (reading 10 GB would have contended with the in-flight job); the command is the plan.
4. **Semantic check** (strongest): the archives are already partially extracted, and the extracted content validates against an independent third source — see §1.6. A corrupt archive does not extract into a tree whose per-sequence box counts exactly reproduce an official totals table.

### 1.5 What else is available, and what it would cost

All probed live with `curl.exe -sIL` `[V]`.

| archive | URL | HTTP | bytes | MiB | what it would add / cost |
|---|---|---|---|---|---|
| `MOT17.zip` | `motchallenge.net/data/MOT17.zip` | 200 | 5,860,214,001 | 5,588.7 | **acquired** |
| `MOT20.zip` | `motchallenge.net/data/MOT20.zip` | 200 | 5,028,926,248 | 4,796.0 | Adds the *dense-crowd* regime: 8 seqs, up to 246 pedestrians/frame, 4K-ish framing. Strictly better than MOT17 for occlusion stress, but swapping costs +? and loses MOT17's detections. Cannot be added (4.7 GiB). |
| `MOT16.zip` | `motchallenge.net/data/MOT16.zip` | 200 | 1,954,509,127 | 1,864.0 | **No new videos.** MOT17 *is* MOT16's sequences re-annotated ("All MOT16 sequences are used with a new, more accurate ground truth"). Adding it buys nothing diverse. |
| `HT21.zip` | `motchallenge.net/data/HT21.zip` | 200 | 4,389,558,730 | 4,186.2 | Head+torso tracking benchmark. Would add the *body-part-visible* regime, which is exactly the project's headline mechanism — but 4.09 GiB, over budget. |
| `MOT17Labels.zip` | `motchallenge.net/data/MOT17Labels.zip` | 200 | 10,107,022 | 9.6 | **Confirmed contents below — 10 MB, fits trivially** (§1.6). |
| `MOT20Labels.zip` | `motchallenge.net/data/MOT20Labels.zip` | 200 | 19,246,343 | 18.4 | Same role for MOT20: `det/det.txt` + train GT. |
| `MOT16Labels.zip` | `motchallenge.net/data/MOT16Labels.zip` | 200 | 3,283,801 | 3.1 | — |
| `2D_MOT_2015.zip` | `motchallenge.net/data/2D_MOT_2015.zip` | **404** | — | — | retired |
| `3D_MOT_2015.zip` | `motchallenge.net/data/3D_MOT_2015.zip` | **404** | — | — | retired |
| `DanceTrack.zip` | `motchallenge.net/data/DanceTrack.zip` | **404** | — | — | DanceTrack is not hosted there |
| `SportsMOT.zip` | `motchallenge.net/data/SportsMOT.zip` | **404** | — | — | ditto |
| `VisDrone-MOT.zip` | `motchallenge.net/data/VisDrone-MOT.zip` | **404** | — | — | ditto |
| per-sequence | `motchallenge.net/data/MOT17/train/MOT17-02-FRCNN.zip` | **404** | — | — | confirmed still dead; whole-archive only |

Page-level HTTP status `[V]`: `https://motchallenge.net/` → **410** (3,286 B); `https://motchallenge.net/data/` → **410** (3,286 B); `https://motchallenge.net/data/MOT17/` → **200** (28,128 B); `https://motchallenge.net/data/MOT20/` → **200** (10,714 B); `https://motchallenge.net/data/HT21/` → **404**.

Note the one transient result: my *first* probe of `MOT16.zip` returned `000` (connection error); two immediate retries both returned `200` with a stable `Content-Length`. Any probe against this host that returns `000` should be retried before being believed.

### 1.6 `MOT17Labels.zip` — contents confirmed, test GT is still withheld

I downloaded the 10 MB archive to a scratch path and listed it (10,107,022 B downloaded, exactly the server's `Content-Length`). `[V]`

**105 members**, exactly: `seqinfo.ini` for all 42 sequence directories (`test/` 21 + `train/` 21), `det/det.txt` for all 42, and **`gt/gt.txt` for the 21 `train/` directories only**.

So:
- **MOT17 test ground truth is NOT public.** The retired site releases the official *detections* and the *train* GT, not the test annotations. Any MOT17 test-split number is therefore not reproducible locally — evaluation on this project must use the **train split** (7 unique videos, 5,316 frames, 112,297 conf=1&class=1 boxes `[V]`), or the leaderboard's published result files.
- It does hand us the official per-sequence **`det/det.txt`** for every sequence for free. These are the published DPM/FRCNN/SDP detections, and they are the correct thing to seed `scripts/eval_mot.py` with when comparing trackers under *identical* detections (the R1 report's "+7.3 HOTA with identical detections" comparison is only legitimate with these).

### 1.7 Current on-disk state `[V]`

```
data/raw/MOT17.zip              5,860,214,001 B   complete (matches server)
data/raw/dancetrack_val.zip     4,209,785,614 B   complete (matches server)
data/raw/mot17/MOT17/train/     7 dirs: MOT17-{02,04,05,09,10,11,13}-FRCNN
data/raw/mot17/MOT17/test/      not yet present
```

Extraction of `MOT17.zip` was still in progress when I measured (§2.1 numbers below are for the 7 extracted train-FRCNN directories, which are a complete and self-consistent subset). The full archive should yield 21 `train/` + 21 `test/` directories = **14 unique videos × 3 detector variants**. `UNVERIFIED` — I could not confirm the final directory count because extraction had not finished.

Note the archive ships each video **three times**, once per detector variant (`-DPM`, `-FRCNN`, `-SDP`). The three copies share identical `gt/gt.txt`. For this project only **one** variant should ever be used (FRCNN is the conventional choice); using all three would triple the I/O for zero new information, and mixing variants would contaminate an appearance gallery. This is the single most important storage decision for MOT17.

---

## 2. What MOT17 and DanceTrack each test

### 2.1 MOT17 — crowded real surveillance (verified per-sequence)

Facts below are from the live archive page at `https://motchallenge.net/data/MOT17/` (HTTP 200) `[V]`, cross-checked against my own extraction. The train table totals 21 dirs / **15,948 frames / 1,638 tracks / 336,891 boxes / density 21.1**; the test table totals 21 dirs / **17,757 frames / 2,355 tracks / 564,228 boxes / density 31.8**. Because of the 3× detector duplication, the unique counts are one third: **5,316 unique train frames** and **5,919 unique test frames** — 14 unique videos, 11,235 unique frames in total, about 6.2 minutes of footage at 30 fps.

Train split (unique videos), with my independently measured `conf=1 & class=1` box counts — which **exactly reproduce the page's "Boxes" column**, validating both the page reading and my parser `[V]`:

| sequence | fps | resolution | frames | tracks | boxes (measured `[V]`) | KB/frame (measured `[V]`) | description (from archive page) |
|---|---|---|---|---|---|---|---|
| MOT17-02 | 30 | 1920×1080 | 600 | 62 | 18,581 | 172.0 | People walking around a large square. |
| MOT17-04 | 30 | 1920×1080 | 1,050 | 83 | 47,557 | 227.5 | Pedestrian street at night, elevated viewpoint |
| MOT17-05 | 14 | 640×480 | 837 | 133 | 6,917 | 35.9 | Street scene from a moving platform |
| MOT17-09 | 30 | 1920×1080 | 525 | 26 | 5,325 | 205.4 | A pedestrian street scene filmed from a low angle. |
| MOT17-10 | 30 | 1920×1080 | 654 | 57 | 12,839 | 133.8 | A pedestrian scene filmed at night by a moving camera |
| MOT17-11 | 30 | 1920×1080 | 900 | 75 | 9,436 | 138.2 | Forward moving camera in a busy shopping mall |
| MOT17-13 | 25 | 1920×1080 | 750 | 110 | 11,642 | 228.3 | Filmed from a bus on a busy intersection |
| **total** | | | **5,316** | **170 ids** | **112,297** | 166.3 avg | 883,773,338 B of images = 0.82 GiB |

Test split (unique videos; no local ground truth, §1.6): MOT17-01 (30fps/1080p, 450f, 24 trk), **MOT17-03 (1500f, 148 trk, 104,675 boxes, density 69.8 — the densest sequence in either dataset)**, MOT17-06 (14fps 640×480, 1194f, 222 trk), MOT17-07 (500f, 60 trk), MOT17-08 (625f, 76 trk, "crowded pedestrian street, stationary camera"), MOT17-12 (900f, 91 trk), MOT17-14 (750f, 164 trk).

**Honest correction to the characterisation I was given.** MOT17 is *not* uniformly "static-ish camera". 4 of its 7 test videos and 4 of its 7 train videos have a moving camera or moving platform, and the camera motion ranges from a handheld eye-level walk (07) to a bus (13, 14) to a shopping-mall dolly (11, 12). What *is* true is that MOT17 is **real surveillance/street footage with a level, pedestrian-scale viewpoint**, heavy mutual occlusion, and mostly-conventional walking motion. The correct summary: MOT17 is the *appearance and occlusion* benchmark, and a mixed but mostly-moderate camera-motion benchmark — not a camera-motion or non-linear-motion benchmark.

### 2.2 DanceTrack val — uniform appearance, non-linear motion (verified from the archive)

Measured directly out of `data/raw/dancetrack_val.zip` without extracting `[V]`:

- **25 sequences**, **25,508 frames**, **225,148 `gt.txt` rows**, **273 track identities** (sum of per-sequence ids — ids are numbered from 1 within each sequence).
- Layout: `val/<seq>/{seqinfo.ini, img1/*.jpg, gt/gt.txt}`. Note the **`val/` wrapper** — the directory you hand a loader is `val/dancetrack0004`, not `dancetrack0004`.
- Prediction-relevant: **track ids start at 0** in DanceTrack (first row of `dancetrack0004` is `12,0,823,617,158,236,1,1,1`), whereas MOT17 ids start at 1. Any "id 0 means no target" sentinel would be a real bug here.
- `seqinfo.ini` keys are identical to MOT17's: `name, imDir, frameRate, seqLength, imWidth, imHeight, imExt` — `frameRate=20`. Resolution is **not uniform**: most sequences are 1920×1080, but `dancetrack0019` is 1280×720 `[V]`.
- Sequence lengths vary 183–2,402 frames. Per-sequence (frames / ids) `[V]`: 0004 1203/4, 0005 1203/4, 0007 1203/8, 0010 1203/6, 0014 1203/12, 0018 503/8, 0019 2402/7, 0025 803/9, 0026 302/19, 0030 1263/6, 0034 923/14, 0035 703/8, 0041 1003/22, 0043 183/14, 0047 1203/8, 0058 1601/7, 0063 1000/8, 0065 702/5, 0073 703/14, 0077 1203/9, 0079 1202/18, 0081 984/20, 0090 1004/14, 0094 603/25, 0097 1203/4.
- `dancetrack0004`'s ids include **8 distinct identities in 1,203 frames**, with box sizes ~130–240 px wide — i.e. large, foreground figures, not a distant crowd. Density is far below MOT17-03's 69.8/frame.

**Why DanceTrack matters for this project:** it is the standard benchmark where *detection is easy and association is hard* `[R]`, because all dancers wear near-identical clothing. That is a direct, adversarial test of the project's **distractor-suppression** requirement (the `cost.distractor_margin` gate) and of the "similar-looking bystanders" case. It is not a test of detection or of partial visibility.

### 2.3 The motion-character measurement (the key evidence)

I computed per-track motion directly from `gt.txt` for both datasets: for every track with three consecutive observed frames, the angle between successive velocity vectors, and speed normalised by box height so the two resolutions are comparable. `[V]`

| metric | MOT17 train (7 seqs) | DanceTrack val | ratio |
|---|---|---|---|
| velocity samples | 38,711 | 9,291 | |
| **mean \|Δheading\| per frame** | **23.34°** | **46.98°** | **2.01×** |
| median \|Δheading\| | 10.31° | 26.57° | 2.58× |
| p90 \|Δheading\| | 63.44° | 129.56° | 2.04× |
| fraction > 90° (reversals) | **4.2%** | **18.1%** | **4.3×** |
| fraction > 45° | 21.0% | 36.7% | 1.75× |
| mean speed (box-heights/frame) | 0.030 | 0.040 | 1.33× |
| p90 speed (box-heights/frame) | 0.067 | 0.069 | 1.03× |

Reading this honestly: **the two datasets differ sharply in motion *non-linearity* and hardly at all in *speed*.** DanceTrack dancers reverse direction roughly 4× as often per frame; MOT17 pedestrians mostly walk. The p90 speed is essentially identical (0.069 vs 0.067), which means *neither* dataset is a fast-motion benchmark in the scale-normalised sense — DanceTrack's reputation for "fast motion" is really a reputation for *erratic* motion, and it is the errant heading, not the magnitude of displacement, that breaks a constant-velocity Kalman filter. This is precisely the mechanism R1 identified (OC-SORT's observation-centric momentum, Hybrid-SORT's ROCM) and it makes DanceTrack the correct ablation bed for the project's motion-model choices.

### 2.4 `SportsMOT` — same role as DanceTrack, worse fit

`MCG-NJU/SportsMOT` on hf-mirror is **33.997 GiB** (`dataset/test.tar` 22,132.9 MiB, `train.tar` 6,392.7 MiB, `val.tar` 6,287.5 MiB) `[V]`, i.e. 3.5× the entire project budget for one dataset. It is **gated** and **CC BY-NC-4.0** (§7). Its role — uniform-appearance, highly non-linear sports motion (basketball/volleyball/football) — is the same association-hard regime DanceTrack already covers, at 4.2 GB instead of 34 GB. **Correctly excluded.** The only thing it would add is sport-specific motion (jumping, ball-driven trajectories) which is not in the project's requirement list.

### 2.5 Requirement → dataset mapping

Requirements are from `docs/BRIEF.md` §4 and §"Non-negotiable design pillars". Verdicts are my assessment of *what the data can demonstrate*, backed by the measurements above.

| project requirement | MOT17 | DanceTrack val | evidence / caveat |
|---|---|---|---|
| **Detect all persons** | ✅ strong | ⚠️ weak | MOT17-03 has 69.8 boxes/frame; DanceTrack is 4–25 large foreground figures. Only MOT17 stresses the detector. |
| **Short-term out-of-frame** | ✅ good | ✅ good | Both: targets genuinely exit and re-enter frame; GT marks the gap. |
| **Long-term out-of-frame** | ⚠️ weak | ⚠️ weak | Neither has a re-acquisition protocol. Gaps are seconds at most. Missing GT for the out-of-frame interval is the fundamental limit in *both*. |
| **Fast motion** | ⚠️ moderate | ⚠️ moderate | Measured p90 speed 0.067 vs 0.069 box-heights/frame — essentially equal, and neither has a hard case. |
| **Non-linear motion** | ❌ weak (mean 23.3°, 4.2% reversals) | ✅ **strong** (mean 47.0°, 18.1% reversals) | The 4.3× reversal gap is the whole reason DanceTrack is in the set. |
| **Partial visibility (part of body)** | ✅ best available | ❌ **none** | MOT17's `visibility` column is genuinely populated (0.0–1.0 spread, §3.1). DanceTrack's is the constant `1` for all 225,148 rows → zero occlusion signal. |
| **Similar-looking bystanders / distractors** | ⚠️ moderate | ✅ **strong** | MOT17 has real crowds with varied clothing; DanceTrack's uniform appearance is the adversarial case. |
| **ID distinctness for non-target persons** | ✅ good (170 ids train) | ✅ good (273 ids val) | Both provide per-frame multi-id GT for MOT/IDF1/HOTA. |
| **Decoupled single-target layer metrics** | ✅ usable | ✅ usable | Both give id-labelled boxes, so `simulate_click` + `evaluate_target` work on either. |
| **Aerial / UAV viewpoint** | ❌ none | ❌ none | Level pedestrian viewpoint in both. §6. |
| **Face cues** | ❌ none | ❌ none | No face/part annotations in either; consistent with the brief's finding that faces are sub-resolution at imgsz 640. |
| **Camera motion / GMC ablation** | ⚠️ partial | ⚠️ partial | MOT17 mixes static and moving cameras (§2.1); DanceTrack is filmed with moving cameras but no GT camera path exists. |

**The two-dataset design is well-chosen and complementary:** MOT17 supplies appearance diversity, occlusion, and crowd density; DanceTrack supplies motion non-linearity and appearance ambiguity. Neither supplies viewpoint diversity. That is the honest shape of the set, and it follows directly from the two measurements in §2.3 and §3.1.

---

## 3. Annotation format reference

### 3.1 MOTChallenge `gt.txt`

Comma-separated, one row per box, no header. **Both archives' `gt.txt` files have exactly 9 columns** — verified: all 30,003 rows of MOT17-02-FRCNN `[V]`, all 225,148 DanceTrack val rows `[V]`.

```
frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,visibility
```

| # | 0-based | field | meaning |
|---|---|---|---|
| 1 | 0 | `frame` | **1-based** frame index |
| 2 | 1 | `id` | track identity; **MOT17 starts at 1, DanceTrack starts at 0** `[V]` |
| 3–6 | 2–5 | `bb_left,bb_top,bb_w,bb_h` | top-left `x,y` and **width, height** (not x2,y2) |
| 7 | 6 | `conf` | `1` = active; **`conf = 0` marks an ignore / removed region** |
| 8 | 7 | `class` | `1` = pedestrian; MOT17/20 use a superset scheme, below |
| 9 | 8 | `visibility` | visibility ratio in [0,1]; **a negative value means "not annotated"** |

Real rows I read out of the archives `[V]`:
```
MOT17-02-FRCNN/gt/gt.txt :  1,1,912,484,97,109,0,7,1      <- conf=0, class=7  => ignore region
                            2,1,912,484,97,109,0,7,1
MOT17-02-FRCNN/gt/gt.txt :  (a kept row, via the loader) frame=1 id=2 xyxy=[1338,418,1505,797] conf=1 vis=1
DanceTrack val/0004      :  12,0,823,617,158,236,1,1,1    <- id starts at 0, conf/class/vis all 1
                            1203,3,531,477,177,284,1,1,1  <- last row
```

**MOT17/20 class scheme** (the superset referred to above), with the exact counts I measured on MOT17-02-FRCNN's 30,003 rows `[V]`:

| class | meaning | rows in MOT17-02-FRCNN | `conf` |
|---|---|---|---|
| 1 | pedestrian | 18,581 | 1 |
| 2 | person on vehicle | 1,549 | 0 |
| 4 | bicycle | 1,559 | 0 |
| 7 | static person (distractor) | 5,271 | 0 |
| 8 | distractor | 1,200 | 0 |
| 9 | occluder | 1,843 | 0 |

(classes 3 car, 5 motorbike, 6 non-motorized vehicle, 10–12 further occluder/reflection categories are defined by the scheme but do not occur in this sequence.)

The structural consequence, which is easy to get wrong: **every non-pedestrian row also has `conf=0`, so filtering on `conf` alone already removes them.** 18,581 + 11,422 = 30,003. `[V]`

**The `visibility` column — and the trap.** MOT17-02-FRCNN's visibility histogram is genuinely informative `[V]`:

| visibility | 0.0 | 1.0 | 0.1 | 0.2 | 0.4 | 0.3 | 0.6 | 0.9 | 0.5 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rows | 12,276 | 5,849 | 2,113 | 1,539 | 1,472 | 1,424 | 1,247 | 1,171 | 1,136 | 1,080 | 696 |

The 12,276 rows at `visibility = 0.0` are exactly the ignore rows (11,422) plus genuinely-occluded pedestrians — the two are distinguishable only via `conf`, which is why the loader tests `conf` first. This column is the project's best available ground truth for the **partial-visibility** requirement.

**DanceTrack's `visibility` column is a placeholder.** Across all 225,148 val rows: `conf` = `1` for every row, `class` = `1` for every row, `visibility` = `1` for every row — one distinct value. `[V]` Any partial-visibility evaluation must therefore run on **MOT17 only**; a `min_visible=` filter on DanceTrack silently does nothing (it will neither drop nor mis-keep anything, which is at least benign).

### 3.2 `seqinfo.ini`

INI, single `[Sequence]` section. Keys verified present in both archives `[V]`:

```ini
[Sequence]                     ; MOT17-02-FRCNN            ; dancetrack0004
name=MOT17-02-FRCNN            ; name=dancetrack0004
imDir=img1                     ; imDir=img1
frameRate=30                   ; frameRate=20
seqLength=600                  ; seqLength=1203
imWidth=1920                   ; imWidth=1920
imHeight=1080                  ; imHeight=1080
imExt=.jpg                     ; imExt=.jpg
```

`seqLength` is the authoritative frame count; `gt_max` and the image count are the fallbacks. **Implementation note for `src/tgtrack/data.py`:** `_read_seqinfo` parses `name, seqLength, imWidth, imHeight, frameRate` but **ignores `imDir` and `imExt`**. That is harmless given `load_mot_sequence`'s `img_dir="img1"` default and this project's JPEG-only data — but it means a non-`img1` sequence would need `img_dir=` passed explicitly rather than being honoured from the file. Worth a comment in the loader; I did not edit it (out of scope).

### 3.3 `det/det.txt`

Same 10-column layout as `gt.txt` (`frame,id,bb_left,bb_top,bb_w,bb_h,conf,x,y,z`), with the same 1-based frames, but no track identity and no visibility — `id` is `-1` and only the `conf` column carries information (the detector's score, continuous, not 0/1). These are the **official published detections**, and `MOT17Labels.zip` supplies them for all 42 sequence directories (§1.6). Present on disk as `data/raw/mot17/MOT17/train/MOT17-02-FRCNN/det/det.txt` (267,111 B for 600 frames `[V]`).

They matter for method comparison: seeding two trackers with identical `det.txt` is what makes R1's "+7.3 HOTA over ByteTrack with identical detections" a controlled result rather than a detector comparison. For this project the operational choice is either (a) use ultralytics YOLO26 detections end-to-end, which is the product path, or (b) use `det.txt` when the experiment's variable is the *association* layer. Both are valid; they answer different questions, and the report of any MOT number should say which was used.

### 3.4 DanceTrack layout

```
dancetrack_val.zip
└── val/                              <- NOTE the wrapper directory
    ├── dancetrack0004/
    │   ├── seqinfo.ini
    │   ├── img1/00000001.jpg ... 00001203.jpg
    │   └── gt/gt.txt                 <- 4,586 rows, 4 identities
    ├── dancetrack0005/
    └── ... 25 sequences total
```

Verified: 25,634 members, consisting of 25 `seqinfo.ini`, 25 `gt/gt.txt`, 25,533 `img1` entries and 25 sequence directories. `[V]` There is **no `det/` directory** — DanceTrack ships GT but not official detections, so detection must be produced locally (YOLO26) or borrowed. No `train/` in this archive; that is `train1.zip` + `train2.zip`.

### 3.5 Worked example with the project's own loader (executed, not invented)

Signature as it exists in `src/tgtrack/data.py:331`:

```python
def load_mot_sequence(
    seq_dir: str | Path,
    subsample: int = 1,
    max_frames: int | None = None,
    *,
    min_conf: float = 1.0,
    min_visible: float = 0.0,
    person_class: int | None = 1,
    load_frames: bool = False,
    img_dir: str = "img1",
    gt_file: str = "gt/gt.txt",
    fps: float | None = None,
    target_id: int | None = None,
    conf_col: int = 6,
    class_col: int = 7,
    visibility_col: int = 8,
) -> Sequence:
```

I ran this against the real extracted MOT17 sequence. Actual output `[V]`:

```python
import sys; sys.path.insert(0, r"C:\Projects\track\src")
from tgtrack.data import load_mot_sequence
seq = load_mot_sequence(r"C:\Projects\track\data\raw\mot17\MOT17\train\MOT17-02-FRCNN")
```

```
name            : MOT17-02-FRCNN
len(seq)        : 600                       # == seqinfo seqLength
width x height  : 1920 x 1080
fps             : 30.0
is_gt_only      : False                     # images were found
frames[0]       : ...\img1\000001.jpg
frames[-1]      : ...\img1\000600.jpg
meta            : {'subsample': 1, 'num_frames_total': 600, 'num_gt_rows': 30003,
                   'dropped': {'conf': 11422, 'class': 0, 'visibility': 0, 'outside': 0}, ...}
gt keys         : 0 .. 599                  # 0-based and contiguous
gt[0][0]        : frame=0 track_id=2 xyxy=[1338.0, 418.0, 1505.0, 797.0] conf=1.0 visibility=1.0
boxes in frame 0   : 22
boxes in frame 300 : 33
```

**Frame renumbering.** MOT `frame` is 1-based; the loader exposes **0-based, contiguous** indices everywhere. `subsample=k` keeps original frames `1, 1+k, 1+2k, …` and renumbers them `0..n-1`. Verified with `subsample=5, max_frames=20`:

```
len(seq)          : 20                      # ceil(600/5) = 120, then capped to 20
frames[0]  -> 000001.jpg   (original MOT frame 1)
frames[1]  -> 000006.jpg   (original MOT frame 6)
frames[19] -> 000096.jpg   (original MOT frame 96)
gt key range      : 0 .. 19
meta['dropped']   : {'conf': 372, 'class': 0, 'visibility': 0, 'outside': 29145}
```

Note `outside: 29145` — the GT rows for the dropped frames are counted as dropped rather than silently discarded, which is the right behaviour for auditing a subsampled run.

**Filter defaults matter, and the default is aggressive.** `min_conf=1.0` (the default) plus `person_class=1` keeps **18,581** of 30,003 rows, reporting the loss as `dropped: {'conf': 11422, ...}`. `min_conf=0.0, person_class=None` keeps all 30,003. `[V]` For the target layer, keeping the ignore rows is occasionally useful (they are exactly the distractors the margin gate must reject), so experiments that care about distractor suppression should say so explicitly.

**`sequence_summary(seq)` returns a `dict`, not a string** `[V]` — keys `name, num_frames, num_gt_boxes, num_gt_ids, gt_ids, visible_frames, fps, width, height, gt_target_id, is_gt_only, meta`. For MOT17-02-FRCNN: `num_gt_boxes=18581, num_gt_ids=62, visible_frames=600`.

**DanceTrack caveat for the loader:** `load_mot_sequence` works unchanged, but the sequence directory must include the `val/` component (`.../val/dancetrack0004`), and because DanceTrack ids begin at 0, any code path that treats `target_id=0` as "unset" (the default `target_id=None` is fine) would be wrong.

---

## 4. Down-sampling / storage recipes

### 4.1 Measured baseline

MOT17-02-FRCNN, 600 frames at 1920×1080: **105,648,879 B = 100.8 MiB**, i.e. 172.0 KB/frame. `[V]` Across all 7 extracted train sequences the average is 166.3 KB/frame, and the total is 883,773,338 B (0.82 GiB) `[V]`.

### 4.2 Does `ffmpeg` exist? Yes — two builds, and **the obvious one cannot do this job**

`where.exe ffmpeg` → **not found** (exit 1). A filesystem search of `C:\Software` found **two builds**, and they are not interchangeable. This is the single most important operational finding in this section:

| binary | version | `image2` demuxer | `mjpeg` encoder | `select` filter | `libx264` | `ffprobe` beside it |
|---|---|---|---|---|---|---|
| `C:\Software\Trae CN\resources\app\bin\ffmpeg.exe` | **6.1.1**, MSYS2 gcc 15.2.0 | ❌ **no** | ❌ **no** | ❌ **no** | ✅ yes | ✅ `ffprobe.exe` |
| `C:\Software\oopz\ffmpeg.exe` | 2022-10-30, gyan.dev essentials | ✅ yes | ✅ yes | ✅ yes | ✅ yes | ❌ no `ffprobe.exe` |

`[V]` — all four capability columns checked directly via `-demuxers`, `-encoders`, `-filters`.

**Why the Trae build is a trap.** It reports itself as "ffmpeg version 6.1.1", which reads as newer-and-better, but its configure line is `--disable-everything` with a hand-written allowlist:

```
--disable-everything --disable-network --disable-autodetect --enable-demuxer=mov --enable-demuxer=avi
--enable-demuxer=flv --enable-demuxer=matroska --enable-demuxer=mpegps --enable-decoder='h264,hevc,...'
--enable-encoder=libx264 --enable-muxer=mp4 --enable-filter='scale,fps'
```

It is built to transcode *video files* into mp4 for Trae's screen recorder — `scale` and `fps` only. It has **no `image2` demuxer**, so it cannot read a JPEG sequence at all; asked to open `img1\%06d.jpg` it fails with `Error opening input: No such file or directory`, and even a single `.jpg` gives `Invalid data found when processing input`. It has **no `mjpeg` encoder**, so it cannot write JPEGs either, and no `select` filter. Any recipe built on it is dead on arrival. I verified this by running both builds against the real MOT17 images.

**The working binary is `C:\Software\oopz\ffmpeg.exe`** — a normal full-featured gyan.dev essentials build (`--enable-gpl --enable-libx264 --enable-libx265 --enable-nvenc --enable-libvpx ...`, 79.4 MB vs Trae's 7.4 MB; the size ratio is the tell). It has `image2`, `concat`, `mjpeg`, `png`, `libx264`, `libx264rgb`, `h264_nvenc`, `libvpx`. `[V]`

Two version-dependent syntax notes for that 2022 build: use **`-vsync 0`** (`-fps_mode passthrough` is the 6.x spelling and it is *not* accepted), and pass **`-start_number 1`** before `-i`, because the image2 demuxer's default start number is 0 and MOT17's frames begin at `000001.jpg`.

`ffprobe.exe` exists **only** beside the Trae build. That is fine even though that build cannot decode JPEGs: `ffprobe` on an mp4 needs only the `mov` demuxer and the `h264` decoder, both of which it has. Verified — it correctly reported `codec_name=h264, width=640, height=360, nb_read_frames=120` on an output produced by the oopz build. `[V]`

Checked and **absent**: `C:\Program Files\ffmpeg.exe`, anything under `C:\Software\Anaconda3\Library\bin`, and `imageio_ffmpeg` in the Python environment. `python -c "import cv2"` → **cv2 4.11.0**, and `cv2.VideoWriter` with fourcc `mp4v` was **verified to open and write** (produced a 3,407-byte test file). `[V]`

**Recommendation:** in processing scripts define both, and do not rely on PATH or on `ffmpeg` alone:

```powershell
$FF  = "C:\Software\oopz\ffmpeg.exe"                        # real build: image2, mjpeg, select, libx264
$FPT = "C:\Software\Trae CN\resources\app\bin\ffprobe.exe"  # only ffprobe on the machine; fine for mp4
```

Do not add either to PATH — both live inside other applications' resources and can be replaced by that app's update (the Trae one silently *becoming* usable, or the oopz one disappearing, are both plausible). If neither is present, §4.4's OpenCV route needs no ffmpeg at all.

### 4.3 Recipe A — ffmpeg via `C:\Software\oopz\ffmpeg.exe` (verified end-to-end)

Subsample every *k*-th frame **and** downscale to 640 px wide in one pass. This reproduces the loader's `subsample=k` semantics exactly: `select` sees 0-based input indices and `mod(n,k)==0` selects input frames 1, 1+k, 1+2k, … — the same frames `load_mot_sequence(subsample=k)` keeps.

```powershell
$FF  = "C:\Software\oopz\ffmpeg.exe"      # NOT the Trae build - see 4.2
$SRC = "C:\Projects\track\data\raw\mot17\MOT17\train\MOT17-02-FRCNN\img1"
$DST = "C:\Projects\track\data\processed\MOT17-02-FRCNN_k5_640.mp4"

# k=5, 640 px wide, H.264 CRF 23.  -start_number 1 is REQUIRED (MOT17 frames start at 000001).
# -vsync 0, not -fps_mode passthrough: this 2022 build predates the new spelling.
& $FF -y -hide_banner -start_number 1 -framerate 30 -i "$SRC\%06d.jpg" `
      -vf "select='not(mod(n\,5))',scale=640:-2" -vsync 0 `
      -c:v libx264 -crf 23 -preset veryfast -pix_fmt yuv420p $DST
```

**Verified output:** 1,017,270 B (0.97 MiB), and `ffprobe -count_frames` confirms `h264, 640x360, nb_read_frames=120` — i.e. exactly 600/5, the correct frame count, correctly scaled, properly encoded. `[V]`

For a JPEG working set instead of a video (what the loader actually wants):

```powershell
# every 5th frame, scaled to 640 wide, mjpeg -q:v 3
& $FF -y -hide_banner -start_number 1 -framerate 30 -i "$SRC\%06d.jpg" `
      -vf "select='not(mod(n\,5))',scale=640:-2" -vsync 0 `
      -q:v 3 "C:\Projects\track\data\processed\MOT17-02-FRCNN_k5_640\%06d.jpg"
```

**Verified output:** 120 files, 5,639,324 B (5.38 MiB), 46,994 B/frame, written as `000001.jpg`-onward. `[V]`

If strict per-frame index fidelity is not needed, the temporal subsample is simpler and needs no `select` filter — useful if you ever have to fall back to a build with a reduced filter set:

```powershell
# 6 fps from a 30 fps source == every 5th frame; -vf "fps=6,scale=640:-2"
& $FF -y -hide_banner -start_number 1 -framerate 30 -i "$SRC\%06d.jpg" `
      -vf "fps=6,scale=640:-2" -c:v libx264 -crf 23 -preset veryfast -pix_fmt yuv420p out.mp4
```

**Verified output:** 1,019,842 B and `nb_read_frames=120` — within **0.25%** of the `select` variant (row 10 vs row 9 in §4.5), confirming the two spellings select the same frames. Prefer `select` when you will join frames back to `gt.txt`, because `fps=` resamples by timestamp and can pick a neighbouring frame if the input frame rate is mis-declared. `[V]`

### 4.4 Recipe B — OpenCV (no ffmpeg dependency; verified working)

Use this if neither ffmpeg build is present (§4.2) — it needs no ffmpeg at all. It is also the path I actually executed for rows 1–8 of §4.5.

```python
import cv2
from pathlib import Path

SEQ  = Path(r"C:\Projects\track\data\raw\mot17\MOT17\train\MOT17-02-FRCNN\img1")
DEST = Path(r"C:\Projects\track\data\processed\MOT17-02-FRCNN_k5_640")
DEST.mkdir(parents=True, exist_ok=True)

imgs = sorted(SEQ.glob("*.jpg"))
for i, p in enumerate(imgs[0::5]):                 # subsample k=5, same semantics as the loader
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    nw   = 640
    nh   = int(round(h * nw / w / 2) * 2)          # keep dimensions even for H.264
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DEST / f"{i+1:06d}.jpg"), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

# optional H.264 mirror (mp4v). NOTE: mp4v is MPEG-4 Part 2, not H.264, despite the name.
first = cv2.imread(str(sorted(DEST.glob("*.jpg"))[0]))
hh, ww = first.shape[:2]
vw = cv2.VideoWriter("seq.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (ww, hh))
for f in sorted(DEST.glob("*.jpg")):
    vw.write(cv2.imread(str(f)))
vw.release()
```

`INTER_AREA` is the correct interpolator for downscaling (it area-averages); `INTER_LINEAR` aliases noticeably at 3× reduction. Two `cv2.VideoWriter` details: it is **silent on failure** — always check `vw.isOpened()` — and it needs an even width and height.

### 4.5 Measured sizes per recipe (MOT17-02-FRCNN, 600 frames / 100.8 MiB baseline)

All measured in this session, not estimated `[V]`:

| step | recipe | frames | bytes | MiB | % of baseline | bytes/frame |
|---|---|---|---|---|---|---|
| 0 | baseline JPEG 1920×1080 | 600 | 105,648,879 | 100.8 | 100.0% | 176,081 |
| 1 | subsample k=2 only | 300 | 52,829,543 | 50.4 | 50.0% | 176,098 |
| 2 | subsample k=5 only | 120 | 21,136,546 | 20.2 | 20.0% | 176,138 |
| 3 | subsample k=10 only | 60 | 10,576,374 | 10.1 | 10.0% | 176,273 |
| 4 | k=5 + 640 wide + JPEG q85 | 120 | 6,327,337 | **6.0** | **6.0%** | 52,728 |
| 5 | k=10 + 640 wide + JPEG q85 | 60 | 3,163,710 | 3.0 | 3.0% | 52,729 |
| 6 | k=5 + 640 wide + JPEG q95 | 120 | 11,138,274 | 10.6 | 10.5% | 92,819 |
| 7 | k=5 + 640 wide + H.264 `mp4v` @30fps (OpenCV) | 120 | 2,766,671 | 2.64 | 2.6% | 23,056 |
| 8 | k=5 + 640 wide + H.264 `mp4v` @6fps (OpenCV) | 120 | 2,766,646 | 2.64 | 2.6% | 23,055 |
| 9 | k=5 + 640 wide + **libx264 crf23** (ffmpeg, §4.3) | 120 | 1,017,270 | **0.97** | 1.0% | 8,477 |
| 10 | `fps=6` + 640 wide + libx264 crf23 (ffmpeg) | 120 | 1,019,842 | 0.97 | 1.0% | 8,498 |
| 11 | k=5 + 640 wide + mjpeg `-q:v 3` (ffmpeg, §4.3) | 120 | 5,639,324 | 5.38 | 5.3% | 46,994 |
| 12 | k=5 + 640 wide + mjpeg `-q:v 2` (ffmpeg) | 120 | 7,355,683 | 7.01 | 7.0% | 61,297 |

Two findings worth keeping:

- **Downscaling dominates; subsampling is linear and near-exact** (rows 1–3 are 50.0/20.0/10.0% — pure file selection with no re-encode, so they cost nothing and lose nothing). Row 4 shows the real win: **640-wide JPEG q85 is 16.7× smaller per frame than the original 1920×1080 (52,728 vs 176,081 B), while subsampling k=5 is a further clean 5×.** Combined = 30% of baseline from k=5 alone → 6.0% with downscaling.
- **JPEG q95 is not worth it**: 10.5% vs 6.0% of baseline for the same frame count — 1.76× the bytes. For a ReID embedder that resizes crops to 448×448, 640-wide q85 retains far more than the embedder can use.
- **`-q:v 3` is not identical to OpenCV's `IMWRITE_JPEG_QUALITY=85`; it is slightly leaner.** The ffmpeg mjpeg route writes 46,994 B/frame against OpenCV's 52,728 B/frame at "85" — **10.9% fewer bytes**. Both are visually fine at 640 px, but do not treat the two settings as interchangeable when comparing against a stored baseline; ffmpeg `-q:v 3` ≈ OpenCV quality ~82. `-q:v 2` (61,297 B/frame) overshoots OpenCV's 85 by 16% and is unnecessary. `[V]`
- **libx264 crf23 is the best video codec here and beats OpenCV's `mp4v` by 2.7×** (8,477 vs 23,056 B/frame, rows 9 vs 7) — `mp4v` is MPEG-4 Part 2, which is what the fourcc's name misleadingly suggests is H.264. If an mp4 mirror is wanted, produce it with ffmpeg, not `cv2.VideoWriter`. Either way the mp4 remains *inter-frame lossy*, which perturbs exactly the appearance statistics a ReID gallery depends on — use mp4 for archival/transport, **JPEG for the working set**.

### 4.6 Extrapolation to the full acquisition

Unique frame counts: MOT17 **11,235** unique frames (5,316 train + 5,919 test, derived from the archive page's 15,948/17,757 totals divided by the 3 detector copies) `[V]`; DanceTrack val **25,508** frames `[V]`. Combined **36,743 frames**.

Applying the measured per-frame costs:

| working set | MOT17 (11,235 fr) | DanceTrack val (25,508 fr) | total |
|---|---|---|---|
| full-rate 640-wide JPEG q85 (OpenCV, 52,728 B/fr) | 0.59 GB | 1.35 GB | **1.94 GB** |
| full-rate 640-wide mjpeg `-q:v 3` (ffmpeg, 46,994 B/fr) | 0.53 GB | 1.20 GB | **1.73 GB** |
| k=5 + 640-wide JPEG q85 | 0.12 GB | 0.27 GB | **0.39 GB** |
| full-rate 640-wide libx264 crf23 mp4 (8,477 B/fr) | 0.10 GB | 0.22 GB | **0.31 GB** |
| full-rate 640-wide `mp4v` mp4 (OpenCV, 23,056 B/fr) | 0.26 GB | 0.59 GB | **0.85 GB** |
| original JPEGs, one detector variant (166.3 KB/fr) | 1.87 GB | 4.24 GB | 6.11 GB |

(The 6.11 GB figure is a sanity check on the arithmetic: it is in the same range as the 5.86 GB MOT17.zip + 4.21 GB val.zip archives, which contain exactly these images.)

### 4.7 Recommendation for `data/processed/`

**Store at full rate, downscaled to 640 px wide, JPEG q85 — and subsample at load time with `subsample=k`, not at compress time.**

Rationale: the loader already takes `subsample=k` and renumbers frames correctly (§3.5), so one full-rate 640-px working set serves *every* temporal density (`k=1,2,5,10`) without re-encoding. Baking `k=5` into the stored files would force a re-encode every time an experiment wants a different density, and re-encoding JPEG→JPEG is the one operation that measurably degrades an appearance gallery. The cost is 1.94 GB, which matches the brief's "~1-2 GB" target.

Concretely:

```
data/processed/
├── MOT17/<seq>/                 # 14 unique videos, ONE detector variant (FRCNN), 640px q85  -> ~0.59 GB
│   ├── img1/000001.jpg ...
│   └── gt/gt.txt                # copied unchanged; box coordinates are NOT rescaled by the loader
├── DanceTrack/<seq>/            # 25 val sequences, 640px q85                              -> ~1.35 GB
│   ├── img1/...
│   └── gt/gt.txt
└── *.mp4                        # optional libx264 cf23 mirrors for transport/archival      -> ~0.31 GB
```

**Critical caveat — do not forget it:** downscaling images invalidates the ground-truth boxes unless they are scaled by the same factor. The loader does **not** rescale GT. The correct approach is either (a) scale `gt.txt` by `640/1920` at build time along with the images, or (b) keep the downscale factor in a sidecar and let the experiment rescale. Option (a) is safer. Note that MOT17 contains **two different source resolutions** (1920×1080 and 640×480 — MOT17-05/06), so the factor is **per sequence**, not global; `seqinfo.ini`'s `imWidth` is the value to key off. Getting this wrong produces a tracker that works and scores zero.

That this working set lets parallel experiments stop contending for I/O is the operational point: 1.94 GB of 640-px JPEGs reads from page cache far more readily than 10 GB of 1080p originals, and the originals stay untouched as the ground truth of record.

---

## 5. Diversity table

Frame/identity/box counts below: MOT17 values are from the live archive page and my own extraction `[V]`; DanceTrack values are from my direct read of `dancetrack_val.zip` `[V]`. Sizes are input-archive sizes. Rows 3–8 are datasets I probed but did **not** acquire, included to make the "why not" explicit.

| dataset | domain / viewpoint | #seq | #frames | #identities / boxes | motion character | occlusion level | annotation | size (GiB) | license | covers which requirement |
|---|---|---|---|---|---|---|---|---|---|---|
| **MOT17** (acquired) | real surveillance, street/square/mall; level pedestrian viewpoint; **mixed static & moving cameras** | 14 unique (42 dirs) | 11,235 unique (5,316 train + 5,919 test) | 170 ids train, 112,297 boxes train; densest seq 69.8 boxes/frame | **mean Δheading 23.3°/frame; 4.2% reversals** → mostly linear walking | **heavy**; `visibility` populated 0.0–1.0 | `gt.txt` 9 cols + `det.txt` + `seqinfo.ini` | 5.458 (archive) | **CC BY-NC-SA 3.0** | detection-in-crowd, occlusion, partial visibility, ID distinctness, short-term out-of-frame |
| **DanceTrack val** (acquired) | dance troupes; level viewpoint, moving camera | 25 | 25,508 | 273 ids, 225,148 boxes (4–25 ids/seq, large foreground figures) | **mean Δheading 47.0°/frame; 18.1% reversals** → highly non-linear | **none annotated** — `visibility` ≡ 1 | `gt.txt` 9 cols (conf/class/vis ≡ 1) + `seqinfo.ini`; **no `det/`** | 3.921 (archive) | **CC BY 4.0** (annotations) + non-commercial research only | non-linear motion, similar-looking bystanders, distractor suppression, association |
| MOT20 (`mrhwl123/MOT20`) | dense crowd surveillance | 8 | — | up to ~246 persons/frame | moderate | extreme | `gt.txt` | 4.701 | CC BY-NC-SA 3.0 | would add dense-crowd occlusion; **4.7 GiB, over budget** |
| MOT16 | = MOT17's videos, older GT | 14 | 11,235 | same videos | same | same | `gt.txt` | 1.820 | CC BY-NC-SA 3.0 | **no new videos — redundant** |
| HT21 | head+torso tracking, level | 4 | — | — | moderate | high, part-visible by construction | `gt.txt` | 4.088 | CC BY-NC-SA 3.0 | would directly test the part-aware mechanism; **4.09 GiB, over budget** |
| SportsMOT (`MCG-NJU/SportsMOT`) | team sports, broadcast | 150 (across 3 splits) | — | ~10–30 players | highly non-linear | moderate | MOT-style | **33.997** | **CC BY-NC-4.0, gated** | same niche as DanceTrack at 8× the size → **excluded** |
| VisDrone-MOT val (`Voxel51/visdrone-mot`) | **aerial/UAV**, city streets | 25-ish scenes, 2,846 samples | ≥1,999 files (tree truncated) | per-frame boxes; **no track ids** | — | not annotated | FiftyOne `Detections` (no temporal ids) | **≥0.79** (2 pages: 415.5 MB + 432.5 MB) | card says **cc-by-sa-3.0**; official VisDrone is CC BY-NC-SA 3.0 (discrepancy) | would give **aerial viewpoint**, but no track ids → **not usable for the target layer** |
| `AbdoTW/MOT16-17-20` | same domains as MOT16/17/20 | 14 unique | — | — | — | — | split zips | 11.965 total, but `MOT16-train.zip` = **0.815** | mixed/upstream | cheapest extra if MOT16 train alone were ever wanted; **redundant with acquired MOT17** |

For the two acquired datasets, the identity/box counts I measured are the numbers to quote; for the non-acquired rows the frame counts are `—` where the HF tree API exposed only archives (a tar/zip member listing would be needed to state them, and I did not open archives I am not acquiring).

**Supplementary label-only repos** (probed, tiny, no imagery): `Lekim89/MOT17` 0.159 GiB and `Lekim89/MOT20` 0.382 GiB are **ablation subsets carrying native `gt/gt.txt`** — `ablation/MOT17-04-FRCNN/gt/gt.txt` is 1.7 MiB `[V]` — but they also ship `img1/*.jpg`, so they are not labels-only; and `UHow/MOT17-L` / `UHow/DanceTrack-L` (0.06 MB each) are **not tracking annotations at all** — I cloned `UHow/DanceTrack-L` and confirmed its 65 JSONs are natural-language track descriptions keyed by frame, e.g. `"0000": ["A male person wearing a gray shirt and gray pants."]` `[V]`. Useful as an appearance prior for a language-conditioned experiment; **not** a source of boxes or ids. `JibayAi/PRCC-Bench` (1.9 MB) is text/CSV only `[V]` and cannot support image experiments.

---

## 6. Honest gaps — what the acquired data cannot demonstrate

Each gap is paired with the smallest acquisition that would close it and its measured cost. Headroom is 0.4–0.64 GB depending on the unit reading (§1.2).

| # | gap | why it matters | smallest fix | size | fits? |
|---|---|---|---|---|---|
| **G1** | **No aerial / UAV viewpoint.** Both datasets are level, pedestrian-scale. The project's own reference product (DJI Air 3S) is an aerial tracker looking down at people. | A tracker tuned to an eye-level crowd may rely on cues (horizon, ground plane, scale gradient) that vanish from above, and the "person" appearance distribution is very different. | `rayavarapuhemanthkumar/visdrone-detection-tracking` → **`VisDrone2019-MOT-val.zip`** (official aerial MOT val split, has track ids). `[V]` | **1,528.3 MiB** | ❌ no (needs ~1.5 GB, ~2.4× headroom) |
| **G1b** | same gap, cheaper but weaker | | `banu4prasad/VisDrone-Dataset` (VisDrone2019-DET-test-challenge images, aerial, **detection only**) `[V]` | **176.4 MiB** | ✅ yes — but gives no tracks, so it tests the *detector* on aerial imagery, not the tracker |
| **G1c** | same gap, cheapest and actually SOT | | `zzz16/UAV123_sample` (aerial single-object-tracking, parquet) `[V]` | **39.5 MiB** | ✅ yes — but it is a *sample* of UAV123 and is SOT, so it exercises the target layer's crop/partial-view path on aerial appearance, not MOT metrics. `UNVERIFIED` whether it carries full frame sequences or single crops. |
| **G2** | **No long-term out-of-frame protocol.** Neither dataset annotates the target during its absence, so "re-acquisition latency" cannot be scored against ground truth. | Requirement 4 explicitly asks for long/short-term out-of-frame. Today this can only be tested with `simulate_click` + synthetic masking, which is self-referential. | No off-the-shelf dataset fixes this; it is a *protocol* gap, not a data gap. Cheapest real fix: construct synthetic out-of-frame runs from MOT17 by masking the target's box for N frames and using the id's later reappearance as GT. | 0 GB | ✅ — and it is the right answer |
| **G3** | **No true part/face annotation.** `visibility` is a scalar per box; there is no mask, no keypoint, no part label. | The headline result (part-aware matching, `mean_parts` 0.951 vs `full` 0.655) must be validated on real data, and there is no GT saying *which* part is visible. | `HT21.zip` (head+torso benchmark) would give part-structured GT — see §1.5 | 4,389.6 MiB | ❌ no |
| **G4** | **DanceTrack's `visibility` column is useless** (constant 1 across all 225,148 rows). | Any occlusion ablation on DanceTrack measures nothing. | Use MOT17 for all visibility-conditioned evaluation; no download needed. | 0 GB | ✅ |
| **G5** | **MOT17 test GT is withheld** (§1.6), so the test split cannot be scored locally. | Published HOTA numbers on MOT17 *test* are not reproducible here; local numbers must be on train, or use `det.txt`-seeded comparisons. | Download `MOT17Labels.zip` (10.1 MB) for the official detections, and evaluate on train. | **10.1 MB** | ✅ **strongly recommended — highest value per byte in this whole report** |
| **G6** | **No fast-motion benchmark.** Measured p90 speed is 0.067 vs 0.069 box-heights/frame — statistically indistinguishable. | The "fast motion" requirement is not stress-tested. | No cheap fix. MOT17-05/06 (`640×480` @14 fps) are the fastest-feeling source but that is a frame-rate artefact, not speed. | 0 GB | ⚠️ accept and document |
| **G7** | **Both datasets ship 3× redundant detector variants (MOT17), and full-resolution images 16.7× larger per frame than needed.** | 6.11 GB of the 10.07 GB is 1080p JPEG that a 448-px embedder cannot use. | Build the 640-px working set (§4.7). | 0 GB download | ✅ recommended |

The two gaps that matter most for the project's stated requirements are **G1 (aerial viewpoint — unfixable within budget)** and **G2 (long-term out-of-frame — a protocol gap, not a data gap)**. G1 is the one real, unclosed requirement: the project's reference behaviour is aerial and neither acquired dataset is. The honest statement is that **this acquisition set validates the algorithm on ground-level person tracking and cannot validate it from a drone**; closing it needs ~1.5 GB that the budget does not have, and the right moment to ask for that budget is *after* the ground-level pipeline reproduces its partial-view result on MOT17 (§4, G3).

---

## 7. Licences and citation implications

| asset | licence | verified where | commercial use |
|---|---|---|---|
| **MOT17** | **CC BY-NC-SA 3.0** | live `https://motchallenge.net/data/MOT17/` page, "License" section `[V]` | ❌ **no** (NC) |
| **MOT20** | CC BY-NC-SA 3.0 | same page family `[V]` | ❌ no |
| **MOT16** | CC BY-NC-SA 3.0 | same page family `[V]` | ❌ no |
| **HT21** | CC BY-NC-SA 3.0 | `[R]`; the `/data/HT21/` archive page is 404 (§1.5) so the licence line could not be read directly | ❌ no (`UNVERIFIED` verbatim) |
| **DanceTrack** | **annotations CC BY 4.0; dataset "available for non-commercial research purposes only"** | `https://dancetrack.github.io/`, "License" section, HTTP 200 `[V]` — verbatim: *"The annotations of DanceTrack are licensed under a Creative Commons Attribution 4.0 License. The dataset of DanceTrack is available for non-commercial research purposes only. All videos and images of DanceTrack are obtained from the Internet which are not property of HKU, CMU or ByteDance."* | ❌ **no** (NC on the media) |
| DanceTrack, HF card | `license: cc-by-4.0` | `https://hf-mirror.com/api/datasets/noahcao/dancetrack` cardData + README front-matter `[V]` | ⚠️ **the card alone is misleading** — it records only the annotation licence and omits the non-commercial restriction on the media. Cite the official page, not the card. |
| **SportsMOT** | **CC BY-NC-4.0** | `https://hf-mirror.com/datasets/MCG-NJU/SportsMOT/raw/main/README.md` front-matter, `license: cc-by-nc-4.0` + `extra_gated_prompt: "This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License"` `[V]` | ❌ no; also **gated** — requires accepting the licence and submitting Institute/purpose fields before download |
| VisDrone (`Voxel51/visdrone-mot`) | card says **`cc-by-sa-3.0`** | `https://hf-mirror.com/datasets/Voxel51/visdrone-mot/raw/main/README.md`, "License: cc-by-sa-3.0" `[V]` | ⚠️ **discrepancy**: the official VisDrone dataset is released CC BY-NC-SA 3.0. The mirror card omits the NC. Treat as non-commercial and prefer the official source. |
| **project model weights** | **AGPL-3.0** | `docs/BRIEF.md` ("Both of these are AGPL-3.0 … A networked product would trigger AGPL source obligations") | ❌ no |

**Citations required.** MOT17 results should cite Milan, Leal-Taixé, Reid, Roth, Schindler, *MOT16: A Benchmark for Multi-Object Tracking* (arXiv:1603.00831) — MOT17 reuses those sequences with corrected GT, per the archive page's own note; MOT20 results should cite Dendorfer et al., *MOT20: A benchmark for multi object tracking in crowded scenes* (arXiv:2003.09003). DanceTrack results should cite Sun et al., *DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion* (CVPR 2022, arXiv:2111.14690) `[R]` — the paper is cited on `dancetrack.github.io` alongside its BibTeX block, which I saw but did not transcribe. SportsMOT: Cui et al., ICCV 2023 (arXiv:2304.05170) `[R]`. VisDrone: Zhu et al., *Detection and Tracking Meet Drones Challenge*, arXiv:2001.06303 — read verbatim from the mirror card `[V]`. Also required by the licence: **attribution** (all of the above are BY variants) and, for MOT17/MOT20/HT21, **ShareAlike** — a redistribution of the data or of a derived dataset must carry the same licence, which rules out shipping a processed `data/processed/` bundle under a permissive licence.

**The commercial picture, stated plainly.** The project's own weights are **AGPL-3.0**; both acquired datasets are **non-commercial**. So a commercial deployment needs **both** different model weights **and** different data — re-licensing neither one alone is sufficient, and the two problems do not have a shared fix. Concretely: AGPL's network-copyleft obligation is triggered by *serving* the model, and CC BY-NC-SA's non-commercial clause is triggered by *using the data to develop the product*, not only by redistributing it. The brief already notes Apache-2.0 weight alternatives (OpenCV Zoo `person_reid_youtu_2021nov.onnx`, RTMPose, SAM 2.1); the data side has no equivalent free lunch at this scale, so any commercial path would need commercially-licensed footage or a synthetic pipeline, which is a product decision rather than a research one. One practical consequence for the current work: **keep `data/processed/` clearly separated and never present it as a redistributable artifact** — it inherits ShareAlike from MOT17.

---

## 8. Verification ledger

**Verified live by me in this session** (`[V]`): both archives' byte sizes vs origin; both archives' completeness; MOT17's full per-sequence table, totals, licence text and archive/`det.txt`/`gt.txt`/`seqinfo.ini` structure; all 7 extracted train sequences' frame/box/identity counts and per-frame byte sizes; MOT17-02-FRCNN's `(conf,class)` distribution and `visibility` histogram; `MOT17Labels.zip`'s complete 105-member listing and the absence of test GT; all HTTP status codes and `Content-Length`s in §1.5; every HF repo size via the tree API; DanceTrack val's 25 sequences / 25,508 frames / 225,148 rows / 273 ids / 25,634 members / per-sequence breakdown / constant `conf`·`class`·`visibility`; the motion measurements in §2.3; the project loader's full behaviour in §3.5; the compression measurements in §4.5; the DanceTrack licence text and SportsMOT/VisDrone front-matter; **the capability matrix of both `ffmpeg` builds (`-demuxers`/`-encoders`/`-filters`), the failure of the Trae build against real MOT17 JPEGs, and every number in §4.3 and rows 9–12 of §4.5 produced by running the documented commands end-to-end and confirming frame counts with `ffprobe`**.

**Marked `UNVERIFIED`, with reasons:**

- `MOT17.zip`'s member count and SHA-256 hashes (extraction and full-archive reads were deliberately avoided to stay out of the in-flight job's way; §1.4 gives the commands).
- The final count of extracted MOT17 directories (extraction was incomplete: `test/` did not yet exist).
- `HT21`'s licence line verbatim (its archive page 404s; the CC BY-NC-SA 3.0 value is `[R]` from the Lead and consistent with the sibling benchmarks).
- `Voxel51/visdrone-mot`'s exact total size — the tree API paginates at 1,000 entries and the `next` cursor points at `huggingface.co`, which is blocked. I recovered page 2 by rewriting the cursor's host to `hf-mirror.com` (415,491,327 B + 432,459,279 B on two pages ⇒ **≥0.79 GiB**); a third page may exist. It is ≥2× the available headroom either way, so the "does not fit" conclusion holds.
- `zzz16/UAV123_sample`'s contents beyond being a 39.5 MiB parquet (not opened).
- DanceTrack train/test split sizes and contents: `train1.zip` (3,606,300,312 B), `train2.zip` (3,299,320,948 B), `test1.zip` (3,677,760,950 B), `test2.zip` (2,882,754,122 B) are sized from the tree API but not acquired or opened. SportsMOT's and MOT20's per-sequence statistics: archives sized but not opened.
- Published HOTA/AssA/IDF1 leaderboard values for both benchmarks: not re-derived here; R1 owns the SOTA numbers.
- The `curl.exe` throughput figures (4.5–5.2 MB/s): `[R]`, not re-measured — doing so would have competed with the active download.

**Corrections to the brief I was given**, all verified: the "MB" column is MiB (§1.2); MOT17 is not a static-camera dataset (§2.1); `motchallenge.net` serves 200 (not 410) on per-dataset archive pages (§1.5); DanceTrack val *does* ship `seqinfo.ini` (§2.2); `UHow/MOT17-L`/`UHow/DanceTrack-L` are natural-language descriptions, not box annotations (§5); `MOT17Labels.zip` hands over train GT and official detections but **not** test GT (§1.6); and — most consequentially — "**`ffmpeg` is on the machine**" is true but misleading: the only build on `C:\Software` that looks canonical cannot read or write JPEG image sequences, so §4.2/§4.3 name the build that actually works.
