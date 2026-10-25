"""Tests for :mod:`tgtrack.data` — dataset loaders and the DJI "select" helpers.

Everything here runs against **synthetic sequences written into a tmp dir**: no
dataset download, no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tgtrack.data import (
    Sequence,
    discover_sequences,
    iter_frames,
    load_mot_sequence,
    load_yolo_or_coco_person_gt,
    make_target_gt,
    read_mot_gt,
    sequence_summary,
    simulate_click,
)
from tgtrack.evaluate import GtBox

# --------------------------------------------------------------------------- #
# Synthetic MOTChallenge fixture
# --------------------------------------------------------------------------- #
SEQINFO = """[Sequence]
name=MOTSEQ
imDir=img1
frameRate=25
seqLength=6
imWidth=640
imHeight=480
imExt=.jpg
"""

# frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,y,visibility
GT_ROWS = [
    "1,1,10,10,20,40,1,1,1.0",     # orig0  id1  kept
    "1,2,100,100,20,40,1,1,0.8",   # orig0  id2  kept (vis 0.8)
    "1,3,0,0,60,80,1,1,1.0",       # orig0  id3  big crowd box for simulate_click
    "2,1,12,10,20,40,1,1,1.0",     # orig1  id1  kept
    "2,2,102,100,20,40,0,1,0.9",   # orig1  id2  conf=0 -> ignore, dropped
    "3,1,14,10,20,40,1,2,1.0",     # orig2  id1  class 2 (car) -> dropped
    "3,2,104,100,20,40,1,1,0.2",   # orig2  id2  low visibility
    "4,1,16,10,20,40,1,1,-1",      # orig3  id1  visibility unknown -> treated as 1.0
    "5,1,18,10,20,40,1,1,1.0",     # orig4  id1
    "6,1,20,10,20,40,1,1,1.0",     # orig5  id1
]


def _write_images(img_dir: Path, count: int, size: tuple[int, int] = (80, 60)) -> None:
    """Write `count` real JPEGs so lazy decoding can be exercised."""
    import cv2

    img_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    for i in range(1, count + 1):
        img = np.full((h, w, 3), i * 10, dtype=np.uint8)
        ok = cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), img)
        assert ok, f"cv2.imwrite failed for {img_dir / f'{i:06d}.jpg'}"


@pytest.fixture()
def mot_dir(tmp_path: Path) -> Path:
    seq = tmp_path / "MOTSEQ"
    (seq / "gt").mkdir(parents=True)
    (seq / "gt" / "gt.txt").write_text("\n".join(GT_ROWS) + "\n", encoding="utf-8")
    (seq / "seqinfo.ini").write_text(SEQINFO, encoding="utf-8")
    _write_images(seq / "img1", 6)
    return seq


# --------------------------------------------------------------------------- #
# read_mot_gt
# --------------------------------------------------------------------------- #
def test_read_mot_gt_columns_and_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text(
        "# comment\n"
        "\n"
        "3,7,10,20,30,40,1,1,0.5\n"          # 9 columns
        "4,8,1,2,3,4,1,1,-1\n"               # unknown visibility
        "5,9,1,2,3,4\n"                      # short row -> defaults
        "not,a,row\n"                        # malformed -> skipped
        "6;10;1;2;3;4;1;1;0.25\n",           # semicolon separated
        encoding="utf-8",
    )
    rows = read_mot_gt(path)
    assert len(rows) == 4
    first = rows[0]
    assert first["frame"] == 3 and first["track_id"] == 7
    assert first["xyxy"] == [10.0, 20.0, 40.0, 60.0]  # xywh -> xyxy
    assert first["conf"] == 1.0 and first["cls"] == 1 and first["visibility"] == 0.5
    assert first["visibility_known"] is True

    unknown = rows[1]
    assert unknown["visibility"] == 1.0 and unknown["visibility_known"] is False

    short = rows[2]
    assert short["conf"] == 1.0 and short["cls"] == 1 and short["visibility"] == 1.0

    semi = rows[3]
    assert semi["visibility"] == 0.25 and semi["frame"] == 6


def test_read_mot_gt_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_mot_gt(tmp_path / "nope.txt")


# --------------------------------------------------------------------------- #
# load_mot_sequence
# --------------------------------------------------------------------------- #
def test_load_mot_sequence_basic(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir)

    assert isinstance(seq, Sequence)
    assert len(seq) == 6 and seq.num_frames == 6
    assert seq.name == "MOTSEQ"
    assert seq.fps == 25.0
    assert seq.width == 640 and seq.height == 480
    assert seq.is_gt_only is False
    assert seq.gt_ids() == [1, 2, 3]
    assert seq.meta["num_frames_total"] == 6
    assert seq.meta["num_gt_rows"] == len(GT_ROWS)
    assert seq.meta["img_dir"] == "img1"

    # conf=0 (orig frame 1, id2) and class=2 (orig frame 2, id1) were dropped.
    assert seq.meta["dropped"] == {"conf": 1, "class": 1, "visibility": 0, "outside": 0}
    assert [b.track_id for b in seq.gt_at(0)] == [1, 2, 3]
    assert [b.track_id for b in seq.gt_at(2)] == [2]  # id1 dropped, id2 kept at vis 0.2
    assert seq.num_gt_boxes == 8

    # 1-based -> 0-based and xywh -> xyxy.
    box = seq.gt_at(0)[0]
    assert box.frame == 0 and box.track_id == 1
    np.testing.assert_allclose(box.xyxy, [10.0, 10.0, 30.0, 50.0])
    assert box.area == pytest.approx(20 * 40)
    # visibility -1 means "not annotated" and is normalized to 1.0.
    assert seq.gt_at(3)[0].visibility == 1.0


def test_load_mot_sequence_filters(mot_dir: Path) -> None:
    keep_ignore = load_mot_sequence(mot_dir, min_conf=0.0)
    assert [b.track_id for b in keep_ignore.gt_at(1)] == [1, 2]  # conf=0 row kept

    any_class = load_mot_sequence(mot_dir, person_class=None)
    assert sorted(b.track_id for b in any_class.gt_at(2)) == [1, 2]

    strict_vis = load_mot_sequence(mot_dir, min_visible=0.5)
    assert [b.track_id for b in strict_vis.gt_at(0)] == [1, 2, 3]  # 0.8 >= 0.5 kept
    assert strict_vis.gt_at(2) == []                              # 0.2 < 0.5 dropped
    assert strict_vis.visible_frames() == [0, 1, 3, 4, 5]
    assert strict_vis.meta["dropped"]["visibility"] == 1
    # An unknown (-1) visibility is never used as a rejection criterion.
    assert len(strict_vis.gt_at(3)) == 1


def test_load_mot_sequence_subsample_renumbers(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir, subsample=2)
    # Keeps original 1-based frames 1, 3, 5 -> 0-based 0, 2, 4 -> renumbered 0, 1, 2.
    assert len(seq) == 3
    assert seq.meta["num_frames_total"] == 6
    assert seq.meta["dropped"]["outside"] == 4  # orig frames 1, 3 (x1), 5
    assert [b.track_id for b in seq.gt_at(0)] == [1, 2, 3]
    assert [b.track_id for b in seq.gt_at(1)] == [2]   # orig frame 2 (class filter)
    assert [b.track_id for b in seq.gt_at(2)] == [1]   # orig frame 4
    assert seq.num_gt_boxes == 5
    # Frames stay aligned with their GT after renumbering.
    assert seq.gt_at(2)[0].frame == 2
    assert Path(seq.frames[1]).name == "000003.jpg"


def test_load_mot_sequence_max_frames(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir, max_frames=2)
    assert len(seq) == 2
    assert sorted(seq.gt) == [0, 1]
    assert seq.meta["dropped"]["outside"] == 5  # orig frames 2..5
    assert seq.num_gt_boxes == 4


def test_load_mot_sequence_bad_args(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mot_sequence(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_mot_sequence(tmp_path, subsample=0)


def test_load_mot_sequence_gt_only(tmp_path: Path) -> None:
    """No img1/ and no seqinfo: the frame count comes from gt.txt alone."""
    seq_dir = tmp_path / "GTONLY"
    (seq_dir / "gt").mkdir(parents=True)
    (seq_dir / "gt" / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n3,1,0,0,10,10,1,1,1\n", encoding="utf-8")
    seq = load_mot_sequence(seq_dir)
    assert len(seq) == 3
    assert seq.is_gt_only is True
    assert seq.frames == ["", "", ""]
    assert seq.frame(0) is None
    assert seq.frame_path(0) is None
    assert seq.fps == 30.0  # documented default when seqinfo is absent
    assert seq.visible_frames() == [0, 2]


def test_load_mot_sequence_without_gt_file(mot_dir: Path) -> None:
    (mot_dir / "gt" / "gt.txt").unlink()
    seq = load_mot_sequence(mot_dir)
    assert seq.gt == {}
    assert len(seq) == 6  # still driven by seqLength / the images
    assert seq.meta["gt_file"] is None


# --------------------------------------------------------------------------- #
# Lazy frame decoding
# --------------------------------------------------------------------------- #
def test_frame_decoding_is_lazy_and_cached(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir)
    assert all(isinstance(f, str) for f in seq.frames)  # nothing decoded yet
    assert seq._cache == {}
    assert Path(seq.frame_path(0)).name == "000001.jpg"

    first = seq.frame(0)
    assert isinstance(first, np.ndarray)
    assert first.shape == (60, 80, 3)
    assert seq.frame(0) is first  # cached: same object, no second decode
    assert seq.frame(0, decode=False) is None
    assert set(seq._cache) == {0}

    seq.clear_cache()
    assert seq._cache == {}
    with pytest.raises(IndexError):
        seq.frame(99)


def test_load_frames_eager(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir, load_frames=True)
    assert all(isinstance(f, np.ndarray) for f in seq.frames)
    assert seq.frame(3) is seq.frames[3]


# --------------------------------------------------------------------------- #
# make_target_gt / simulate_click
# --------------------------------------------------------------------------- #
def test_make_target_gt(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir)
    target = make_target_gt(seq, 1)
    assert sorted(target) == [0, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(target[0], [10.0, 10.0, 30.0, 50.0])
    np.testing.assert_allclose(target[1], [12.0, 10.0, 32.0, 50.0])
    # id 1's orig-frame-2 box was filtered out (class 2) -> the target is absent.
    assert target[2] is None
    np.testing.assert_allclose(target[3], [16.0, 10.0, 36.0, 50.0])
    assert all(v is None for v in make_target_gt(seq, 99).values())


def test_simulate_click_picks_smallest_containing_box(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir)
    # (20, 20) is inside id1 (10,10)-(30,50) and inside the big crowd box id3 (0,0)-(60,80):
    # the click must select the *most specific* (smallest) box.
    assert simulate_click(seq, (20, 20), 0) == 1
    assert simulate_click(seq, (5, 5), 0) == 3    # only the big box contains this point
    assert simulate_click(seq, (110, 110), 0) == 2
    assert simulate_click(seq, (500, 400), 0) is None
    assert simulate_click(seq, (12, 15), 1) == 1
    assert simulate_click(seq, (22, 15), 5) == 1


# --------------------------------------------------------------------------- #
# discover_sequences
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dataset_root(tmp_path: Path, mot_dir: Path) -> Path:
    yolo = tmp_path / "YOLOSEQ"
    (yolo / "labels").mkdir(parents=True)
    (yolo / "labels" / "0001.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    coco = tmp_path / "COCOSEQ"
    (coco / "annotations").mkdir(parents=True)
    (coco / "annotations" / "inst.json").write_text('{"images": [], "annotations": []}', encoding="utf-8")
    (tmp_path / "README.md").write_text("not a sequence", encoding="utf-8")
    return tmp_path


def test_discover_sequences_layouts(dataset_root: Path) -> None:
    root = dataset_root
    assert [Path(p).name for p in discover_sequences(root, "mot")] == ["MOTSEQ"]
    assert [Path(p).name for p in discover_sequences(root, "yolo")] == ["YOLOSEQ"]
    assert [Path(p).name for p in discover_sequences(root, "coco")] == ["COCOSEQ"]
    assert [Path(p).name for p in discover_sequences(root)] == ["COCOSEQ", "MOTSEQ", "YOLOSEQ"]
    # Idempotent when handed a single sequence.
    assert discover_sequences(root / "MOTSEQ") == [str(root / "MOTSEQ")]
    with pytest.raises(NotADirectoryError):
        discover_sequences(root / "README.md")
    with pytest.raises(ValueError):
        discover_sequences(root, "nope")


# --------------------------------------------------------------------------- #
# YOLO / COCO
# --------------------------------------------------------------------------- #
def test_load_yolo_person_gt(tmp_path: Path) -> None:
    root = tmp_path / "YOLO"
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "0001.txt").write_text("0 0.5 0.5 0.5 0.5\n2 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "0002.txt").write_text("2 0.1 0.1 0.2 0.2\n", encoding="utf-8")

    gt = load_yolo_or_coco_person_gt(root, width=640, height=480)
    assert sorted(gt) == [0]  # frame 1 has no person -> absent, not an empty list
    box = gt[0][0]
    assert box.track_id == -1
    np.testing.assert_allclose(box.xyxy, [160.0, 120.0, 480.0, 360.0])
    assert box.frame == 0

    cars = load_yolo_or_coco_person_gt(root, width=640, height=480, person_class=2)
    assert sorted(cars) == [0, 1]
    np.testing.assert_allclose(cars[0][0].xyxy, [0.0, 0.0, 128.0, 96.0])

    with pytest.raises(ValueError):
        load_yolo_or_coco_person_gt(root)  # no images and no width/height

    first_only = load_yolo_or_coco_person_gt(root, width=640, height=480, max_frames=1)
    assert sorted(first_only) == [0] and len(first_only[0]) == 1


def test_load_yolo_uses_image_size_when_available(tmp_path: Path) -> None:
    root = tmp_path / "YOLO2"
    (root / "labels").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    _write_images(root / "images", 1)  # 80x60
    (root / "labels" / "000001.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    gt = load_yolo_or_coco_person_gt(root)
    np.testing.assert_allclose(gt[0][0].xyxy, [0.0, 0.0, 80.0, 60.0])


def test_load_coco_person_gt(tmp_path: Path) -> None:
    root = tmp_path / "COCO"
    (root / "annotations").mkdir(parents=True)
    payload = {
        "images": [
            {"id": 11, "file_name": "img_0002.jpg", "width": 640, "height": 480},
            {"id": 10, "file_name": "img_0001.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 10, "category_id": 3, "bbox": [0, 0, 5, 5]},          # car
            {"id": 3, "image_id": 11, "category_id": 1, "bbox": [1, 2, 3, 4], "track_id": 7},
        ],
    }
    (root / "annotations" / "inst.json").write_text(json.dumps(payload), encoding="utf-8")

    gt = load_yolo_or_coco_person_gt(root)
    assert sorted(gt) == [0, 1]  # ordered by file_name, not by image id
    assert gt[0][0].track_id == -1
    np.testing.assert_allclose(gt[0][0].xyxy, [10.0, 20.0, 40.0, 60.0])
    assert gt[1][0].track_id == 7
    np.testing.assert_allclose(gt[1][0].xyxy, [1.0, 2.0, 4.0, 6.0])

    # Every class when person_class is None.
    everything = load_yolo_or_coco_person_gt(root, person_class=None)
    assert len(everything[0]) == 1  # None -> auto-picks class 1 for COCO
    assert load_yolo_or_coco_person_gt(root / "annotations" / "inst.json")[1][0].track_id == 7
    with pytest.raises(ValueError):
        load_yolo_or_coco_person_gt(root, layout="nope")


# --------------------------------------------------------------------------- #
# iter_frames / sequence_summary
# --------------------------------------------------------------------------- #
def test_iter_frames_and_summary(mot_dir: Path) -> None:
    seq = load_mot_sequence(mot_dir)
    walked = list(iter_frames(seq))
    assert len(walked) == 6
    assert all(img is None for _, img, _ in walked)  # no decoding by default
    assert walked[0][0] == 0 and len(walked[0][2]) == 3
    assert len(list(iter_frames(seq, decode=True))[0][1].shape) == 3
    assert len(list(iter_frames(seq, limit=2))) == 2

    summary = sequence_summary(seq)
    assert summary["name"] == "MOTSEQ"
    assert summary["num_frames"] == 6
    assert summary["num_gt_boxes"] == 8
    assert summary["num_gt_ids"] == 3
    assert summary["visible_frames"] == 6
    assert summary["is_gt_only"] is False
    assert summary["gt_target_id"] is None
    json.dumps(summary)  # must stay JSON friendly


def test_sequence_dataclass_contract() -> None:
    seq = Sequence(name="x", frames=["a.jpg"], gt={0: [GtBox.from_xywh(0, 5, [1, 2, 3, 4])]})
    assert len(seq) == 1 and seq.num_frames == 1
    assert seq.gt_ids() == [5]
    assert seq.frame_indices == [0]
    assert seq.gt_at(7) == []
    assert seq.visible_frames() == [0]
    assert seq.frame_path(0) == "a.jpg"
    assert seq.num_gt_boxes == 1
