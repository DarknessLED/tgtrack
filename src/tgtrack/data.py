"""Dataset loading and sequence plumbing for ``tgtrack``.

Layouts supported
-----------------
* **MOTChallenge** — ``<seq>/img1/000001.jpg`` … plus ``<seq>/gt/gt.txt`` and an
  optional ``<seq>/seqinfo.ini``::

      # gt.txt — one row per box
      frame,id,bb_left,bb_top,bb_w,bb_h,conf,class,y,visibility

  Column 7 (0-based ``6``) is ``conf`` (``0`` marks an *ignore* / removed region),
  column 8 (0-based ``7``) is the class id (``1`` = pedestrian) and column 9
  (0-based ``8``) is the visibility ratio in ``[0, 1]`` — the layout used by
  MOT16/17/20 ground truth and by TrackEval.  Column 10 (0-based ``9``) is
  reserved and ignored.  The indices are overridable through ``conf_col`` /
  ``class_col`` / ``visibility_col``.  A **negative** visibility means "not
  annotated" and is therefore treated as ``1.0`` rather than as a hard zero.
* **YOLO** — ``<root>/images/*.jpg`` beside ``<root>/labels/*.txt`` whose rows are
  ``class cx cy w h`` normalized to ``[0, 1]`` (a trailing confidence column is
  ignored).
* **COCO** — ``<root>/annotations/*.json`` plus the images it references.

Frame indices
-------------
MOTChallenge frame numbers are **1-based**; this module exposes **0-based,
contiguous** frame indices everywhere (``GtBox.frame`` and every dict key), so a
box and its ``Sequence.frames`` entry always share one index — which is what
:func:`tgtrack.evaluate.mot_frame_data` and :func:`tgtrack.evaluate.evaluate_target`
require.  ``subsample=k`` keeps the original frames ``1, 1+k, 1+2k, …`` and
renumbers them ``0..n-1``.

Frames are **not** decoded eagerly: ``Sequence.frames`` holds paths and
``Sequence.frame(i)`` decodes on demand into a per-sequence cache, so a
20 000-frame sequence costs paths only until something asks for pixels.  Pass
``load_frames=True`` to decode everything up front.  ``cv2`` and ``Pillow`` are
imported lazily, so the loaders work in a GT-only environment.

Typical use::

    seq = load_mot_sequence(r"data/MOT17/train/MOT17-02-FRCNN", subsample=2)
    target = make_target_gt(seq, simulate_click(seq, (640, 360), 0))
"""

from __future__ import annotations

import configparser
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence as TSequence

import numpy as np

from .evaluate import GtBox

__all__ = [
    "IMAGE_EXTS",
    "Sequence",
    "discover_sequences",
    "iter_frames",
    "load_mot_sequence",
    "load_yolo_or_coco_person_gt",
    "make_target_gt",
    "read_mot_gt",
    "sequence_summary",
    "simulate_click",
]

#: Image extensions considered by :func:`_list_images`.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def _num_key(path: Path) -> tuple[int, str]:
    """Sort key: numeric stems sort numerically, everything else lexically."""
    stem = path.stem
    try:
        return int(stem), ""
    except ValueError:
        return (1 << 62), stem


def _list_images(directory: Path) -> list[Path]:
    """Sorted image paths in ``directory`` (empty when it does not exist)."""
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=_num_key)


def _decode_image(path: str | Path) -> np.ndarray:
    """Decode one image with cv2 (preferred, BGR) or Pillow (RGB) as a fallback."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    try:  # pragma: no cover - depends on the environment
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:  # pragma: no cover - fall through to Pillow
        pass
    try:  # pragma: no cover - depends on the environment
        from PIL import Image

        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cannot decode {path}: neither cv2 nor Pillow could read it") from exc


def image_size(path: str | Path) -> tuple[int, int] | None:
    """``(width, height)`` of an image without decoding its pixels, or ``None``."""
    try:  # pragma: no cover - depends on the environment
        from PIL import Image

        with Image.open(path) as handle:
            return int(handle.width), int(handle.height)
    except Exception:
        pass
    try:  # pragma: no cover - depends on the environment
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
        if img is not None:
            h, w = img.shape[:2]
            return int(w * 8), int(h * 8)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #
@dataclass
class Sequence:
    """One video sequence: its frames, its all-person ground truth and metadata.

    Attributes:
        name: sequence name (usually the directory name).
        frames: per-frame entry — a path (``str``) that is decoded on demand, a
            decoded ``np.ndarray``, or ``""`` for a GT-only sequence with no
            images on disk.
        gt: ground-truth boxes keyed by **0-based** frame index.
        fps: frame rate (``seqinfo.ini`` when available).
        width: frame width in pixels (0 when unknown).
        height: frame height in pixels (0 when unknown).
        gt_target_id: identity of the DJI-style clicked target, when known.
        meta: free-form provenance (source directory, subsample, total frames…).
    """

    name: str
    frames: list[Any]
    gt: dict[int, list[GtBox]]
    fps: float = 30.0
    width: int = 0
    height: int = 0
    gt_target_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.gt = {int(k): list(v) for k, v in dict(self.gt).items()}
        self.name = str(self.name)
        self.fps = float(self.fps)

    # -- shape ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.frames)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_indices(self) -> list[int]:
        return list(range(len(self.frames)))

    @property
    def num_gt_boxes(self) -> int:
        return sum(len(v) for v in self.gt.values())

    @property
    def is_gt_only(self) -> bool:
        """True when the sequence has no decodable images (GT-only evaluation)."""
        return not any(self.frames)

    def gt_ids(self) -> list[int]:
        """Every ground-truth identity in the sequence, sorted."""
        return sorted({int(b.track_id) for boxes in self.gt.values() for b in boxes})

    def visible_frames(self) -> list[int]:
        """Frame indices that carry at least one GT box."""
        return sorted(self.gt)

    # -- access ------------------------------------------------------------ #
    def gt_at(self, frame_idx: int) -> list[GtBox]:
        """GT boxes for one 0-based frame index (empty list when none)."""
        return list(self.gt.get(int(frame_idx), ()))

    def frame_path(self, frame_idx: int) -> str | None:
        """Path of a frame, or ``None`` when it is already decoded / unavailable."""
        item = self.frames[int(frame_idx)]
        return str(item) if isinstance(item, (str, Path)) and str(item) else None

    def frame(self, frame_idx: int, *, decode: bool = True) -> np.ndarray | None:
        """The image for one frame, decoded once and cached.

        Returns ``None`` for a GT-only entry.  Channel order follows the decoder
        (cv2 → BGR, Pillow → RGB); evaluation never needs the pixels, so this is
        only for visualisation and detector smoke tests.
        """
        idx = int(frame_idx)
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"frame {idx} out of range for sequence {self.name!r} ({len(self.frames)} frames)")
        item = self.frames[idx]
        if isinstance(item, np.ndarray):
            return item
        if not decode:
            return None
        if idx in self._cache:
            return self._cache[idx]
        if not item:
            return None
        img = _decode_image(item)
        self._cache[idx] = img
        return img

    def clear_cache(self) -> None:
        """Drop every decoded frame (frees RAM; they are re-decoded on demand)."""
        self._cache.clear()


# --------------------------------------------------------------------------- #
# MOTChallenge
# --------------------------------------------------------------------------- #
def read_mot_gt(
    path: str | Path,
    *,
    conf_col: int = 6,
    class_col: int = 7,
    visibility_col: int = 8,
) -> list[dict[str, Any]]:
    """Parse a MOTChallenge ``gt.txt`` (or ``det.txt``) into raw row dicts.

    No filtering happens here — that is :func:`load_mot_sequence`'s job — so the
    raw rows stay available for inspection and for custom filter policies.

    Args:
        path: the ``gt.txt``/``det.txt`` file.
        conf_col: 0-based index of the confidence / validity column.
        class_col: 0-based index of the class column (``1`` = pedestrian).
        visibility_col: 0-based index of the visibility column.

    Returns:
        One dict per row with ``frame`` (**1-based**, as written in the file),
        ``track_id``, ``xyxy``, ``conf``, ``cls``, ``visibility`` (negatives
        normalized to ``1.0``) and ``visibility_known``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such annotation file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [f.strip() for f in line.split(";")] if ";" in line else next(csv.reader([line]))
            if len(fields) < 6:
                continue
            try:
                frame = int(float(fields[0]))
                track_id = int(float(fields[1]))
                x, y, w, h = (float(v) for v in fields[2:6])
                conf = float(fields[conf_col]) if len(fields) > conf_col else 1.0
                cls = int(float(fields[class_col])) if len(fields) > class_col else 1
                vis = float(fields[visibility_col]) if len(fields) > visibility_col else 1.0
            except ValueError:
                continue  # malformed row — a real MOT file can carry stray text
            known = vis >= 0.0
            rows.append({
                "frame": frame,
                "track_id": track_id,
                "xyxy": [x, y, x + w, y + h],
                "conf": conf,
                "cls": cls,
                "visibility": vis if known else 1.0,
                "visibility_known": known,
            })
    return rows


def _read_seqinfo(seq_dir: Path) -> dict[str, Any]:
    """Read the ``[Sequence]`` section of ``seqinfo.ini`` (empty dict if absent)."""
    path = seq_dir / "seqinfo.ini"
    if not path.is_file():
        return {}
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return {}
    if not parser.has_section("Sequence"):
        return {}
    section = parser["Sequence"]
    info: dict[str, Any] = {}
    if section.get("name"):
        info["name"] = section.get("name")
    for key in ("seqLength", "imWidth", "imHeight"):
        raw = section.get(key)
        if raw:
            try:
                info[key] = int(float(raw))
            except ValueError:
                pass
    raw_fps = section.get("frameRate")
    if raw_fps:
        try:
            info["frameRate"] = float(raw_fps)
        except ValueError:
            pass
    return info


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
    """Load a MOTChallenge-format sequence.

    Args:
        seq_dir: ``<seq>`` directory (contains ``img1/`` and ``gt/gt.txt``).
        subsample: keep every ``subsample``-th original frame (1-based frames
            ``1, 1+k, 1+2k, …``), renumbered to ``0..n-1``.
        max_frames: keep at most this many frames (after subsampling).
        min_conf: drop rows with ``conf < min_conf`` (the default drops the
            ``conf == 0`` "ignore" rows).
        min_visible: drop rows with a *known* visibility below this value.
        person_class: keep only this class id; ``None`` keeps every class.
        load_frames: decode every image up front instead of lazily.
        img_dir: image sub-directory name (``img1``; ``img``/``images`` are tried
            as fallbacks).
        gt_file: annotation file relative to ``seq_dir`` (``gt/gt.txt``).  A
            missing file yields an empty ``gt`` rather than an error, so image-only
            sequences still load.
        fps: override the frame rate from ``seqinfo.ini``.
        target_id: set :attr:`Sequence.gt_target_id` up front.
        conf_col, class_col, visibility_col: see :func:`read_mot_gt`.

    Returns:
        A :class:`Sequence` with 0-based, contiguous frame indices.

    Raises:
        FileNotFoundError: if ``seq_dir`` is not a directory.
        ValueError: if ``subsample < 1``.
    """
    seq_dir = Path(seq_dir)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"not a sequence directory: {seq_dir}")
    if subsample < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")

    info = _read_seqinfo(seq_dir)
    images = _list_images(seq_dir / img_dir)
    if not images:
        for alt in ("img", "images"):
            if alt != img_dir:
                images = _list_images(seq_dir / alt)
                if images:
                    img_dir = alt
                    break

    gt_path = seq_dir / gt_file
    raw = read_mot_gt(gt_path, conf_col=conf_col, class_col=class_col, visibility_col=visibility_col) if gt_path.is_file() else []

    # Total number of original frames: seqinfo wins, but never truncate GT or images.
    gt_max = max((row["frame"] for row in raw), default=0)
    n_total = max(int(info.get("seqLength", 0) or 0), gt_max, len(images))
    kept = list(range(0, n_total, subsample))
    if max_frames is not None:
        kept = kept[: max(0, int(max_frames))]
    remap = {orig: new for new, orig in enumerate(kept)}

    if images:
        frames: list[Any] = [str(images[orig]) if orig < len(images) else "" for orig in kept]
    else:
        frames = [""] * len(kept)

    gt: dict[int, list[GtBox]] = {}
    dropped = {"conf": 0, "class": 0, "visibility": 0, "outside": 0}
    for row in raw:
        orig = int(row["frame"]) - 1  # MOTChallenge frames are 1-based
        new = remap.get(orig)
        if new is None:
            dropped["outside"] += 1
            continue
        if float(row["conf"]) < min_conf:
            dropped["conf"] += 1
            continue
        if person_class is not None and int(row["cls"]) != int(person_class):
            dropped["class"] += 1
            continue
        if min_visible > 0.0 and row["visibility_known"] and float(row["visibility"]) < min_visible:
            dropped["visibility"] += 1
            continue
        gt.setdefault(new, []).append(
            GtBox(
                frame=new,
                track_id=int(row["track_id"]),
                xyxy=row["xyxy"],
                conf=float(row["conf"]),
                visibility=float(row["visibility"]),
            )
        )

    width = int(info.get("imWidth", 0) or 0)
    height = int(info.get("imHeight", 0) or 0)
    if images and (width <= 0 or height <= 0):
        size = image_size(images[0])
        if size is not None:
            width, height = size

    seq = Sequence(
        name=str(info.get("name") or seq_dir.name),
        frames=frames,
        gt=gt,
        fps=float(fps if fps is not None else info.get("frameRate", 30.0)),
        width=width,
        height=height,
        gt_target_id=target_id,
        meta={
            "source": str(seq_dir),
            "img_dir": img_dir if images else None,
            "gt_file": str(gt_path) if gt_path.is_file() else None,
            "subsample": int(subsample),
            "num_frames_total": int(n_total),
            "num_gt_rows": len(raw),
            "dropped": dropped,
        },
    )
    if load_frames and not seq.is_gt_only:
        for idx in range(len(seq)):
            img = seq.frame(idx)
            if img is None:
                continue
            seq.frames[idx] = img
    return seq


# --------------------------------------------------------------------------- #
# YOLO / COCO
# --------------------------------------------------------------------------- #
def _yolo_label_boxes(
    label_path: Path,
    frame: int,
    width: int,
    height: int,
    person_class: int,
) -> list[GtBox]:
    """Parse one YOLO ``.txt`` label file into GT boxes (normalized → pixels)."""
    boxes: list[GtBox] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if cls != person_class:
            continue
        bw, bh = w * width, h * height
        boxes.append(
            GtBox.from_xywh(frame, -1, [(cx * width) - bw / 2, (cy * height) - bh / 2, bw, bh])
        )
    return boxes


def _load_coco(
    ann_path: Path,
    *,
    person_class: int | None,
    track_id_field: str,
) -> dict[int, list[GtBox]]:
    """Load COCO-format annotations (``bbox`` is absolute ``x, y, w, h``)."""
    data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    images = sorted(data.get("images", []), key=lambda im: _num_key(Path(str(im.get("file_name", "")))))
    frame_of = {im.get("id"): idx for idx, im in enumerate(images)}
    gt: dict[int, list[GtBox]] = {}
    for ann in data.get("annotations", []):
        frame = frame_of.get(ann.get("image_id"))
        if frame is None:
            continue
        if person_class is not None and int(ann.get("category_id", -1)) != int(person_class):
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        track_id = ann.get(track_id_field, -1)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            track_id = -1
        gt.setdefault(frame, []).append(GtBox.from_xywh(frame, track_id, [float(v) for v in bbox[:4]]))
    return gt


def load_yolo_or_coco_person_gt(
    source: str | Path,
    *,
    layout: str = "auto",
    images_dir: str | Path | None = None,
    labels_dir: str | Path | None = None,
    ann_file: str | Path | None = None,
    person_class: int | None = None,
    width: int | None = None,
    height: int | None = None,
    track_id_field: str = "track_id",
    max_frames: int | None = None,
) -> dict[int, list[GtBox]]:
    """Load person ground truth from a YOLO label tree or COCO annotations.

    Args:
        source: a directory, a ``.json`` (COCO) or a labels directory (YOLO).
        layout: ``"auto"`` | ``"yolo"`` | ``"coco"``.
        images_dir, labels_dir, ann_file: explicit overrides for the conventional
            ``images/``, ``labels/`` and ``annotations/*.json`` locations.
        person_class: class to keep.  ``None`` means *auto*: ``1`` for COCO
            (``category_id`` of person) and ``0`` for YOLO (class index).
        width, height: frame size used to denormalize YOLO labels when the images
            cannot be read.
        track_id_field: COCO annotation key holding a temporal identity.  COCO
            detection annotations carry no track ids, so every box falls back to
            ``-1`` — ID-based metrics are then meaningless and only
            detection-quality metrics apply.
        max_frames: keep at most this many frames.

    Returns:
        ``{0-based frame index: [GtBox, …]}`` with frames ordered by file name.

    Raises:
        ValueError: when the layout cannot be determined or YOLO labels need a
            frame size that is unavailable.
    """
    source = Path(source)
    if layout == "auto":
        if source.suffix.lower() == ".json" or (source.is_dir() and (source / "annotations").is_dir()):
            layout = "coco"
        elif source.is_dir() and (source / "labels").is_dir():
            layout = "yolo"
        elif source.is_dir() and any(source.glob("*.txt")):
            layout = "yolo"
        else:
            raise ValueError(f"cannot infer layout for {source!r}; pass layout='yolo' or layout='coco'")

    if layout == "coco":
        if ann_file is None:
            if source.suffix.lower() == ".json":
                ann_file = source
            else:
                candidates = sorted((source / "annotations").glob("*.json"))
                if not candidates:
                    raise FileNotFoundError(f"no COCO json under {source / 'annotations'}")
                ann_file = candidates[0]
        cls = 1 if person_class is None else int(person_class)
        gt = _load_coco(Path(ann_file), person_class=cls, track_id_field=track_id_field)
    elif layout == "yolo":
        label_root = Path(labels_dir) if labels_dir is not None else (
            source if source.is_dir() and source.name == "labels" else source / "labels"
        )
        if not label_root.is_dir():
            raise FileNotFoundError(f"no YOLO labels directory: {label_root}")
        image_root = None
        if images_dir is not None:
            image_root = Path(images_dir)
        elif (source / "images").is_dir():
            image_root = source / "images"
        cls = 0 if person_class is None else int(person_class)
        label_files = sorted([p for p in label_root.glob("*.txt")], key=_num_key)
        if max_frames is not None:
            label_files = label_files[: max(0, int(max_frames))]
        gt = {}
        for frame, label_path in enumerate(label_files):
            w, h = width, height
            if (w is None or h is None) and image_root is not None:
                for ext in IMAGE_EXTS:
                    candidate = image_root / f"{label_path.stem}{ext}"
                    if candidate.is_file():
                        size = image_size(candidate)
                        if size is not None:
                            w, h = size
                        break
            if w is None or h is None:
                raise ValueError(
                    f"YOLO labels need a frame size: pass width/height or provide readable images "
                    f"(missing for {label_path})"
                )
            boxes = _yolo_label_boxes(label_path, frame, int(w), int(h), cls)
            if boxes:
                gt[frame] = boxes
    else:
        raise ValueError(f"unknown layout {layout!r}; expected 'auto', 'yolo' or 'coco'")
    return gt


def discover_sequences(root: str | Path, layout: str = "auto") -> list[str]:
    """Find sequence directories under ``root``.

    Args:
        root: a dataset root, or a single sequence directory.
        layout: ``"auto"`` (any known layout), ``"mot"``, ``"yolo"`` or ``"coco"``.

    Returns:
        Sorted absolute-ish paths (as strings) of the sequences found.  ``root``
        itself is returned when it already is one sequence, so the function is
        idempotent for callers that accept either.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    wanted = ("mot", "coco", "yolo") if layout == "auto" else (layout,)
    if layout not in ("auto", "mot", "yolo", "coco"):
        raise ValueError(f"unknown layout {layout!r}")

    def matches(path: Path, kind: str) -> bool:
        if kind == "mot":
            return (path / "gt" / "gt.txt").is_file() or (path / "img1").is_dir()
        if kind == "coco":
            return (path / "annotations").is_dir() and any((path / "annotations").glob("*.json"))
        return (path / "labels").is_dir() or (path / "images").is_dir()

    if any(matches(root, kind) for kind in wanted):
        return [str(root)]
    found = [p for p in sorted(root.iterdir()) if p.is_dir() and any(matches(p, kind) for kind in wanted)]
    return [str(p) for p in found]


# --------------------------------------------------------------------------- #
# Target helpers (the DJI "select" interaction)
# --------------------------------------------------------------------------- #
def make_target_gt(seq: Sequence, target_id: int) -> dict[int, np.ndarray | None]:
    """Per-frame target box for one identity — ``{frame: xyxy or None}``.

    Every frame of the sequence is keyed (``None`` when the target is not
    visible), which is exactly what
    :func:`tgtrack.evaluate.evaluate_target` expects.  When a malformed GT puts
    several boxes with the same id in one frame, the largest one is used.
    """
    out: dict[int, np.ndarray | None] = {}
    for idx in range(len(seq)):
        best: GtBox | None = None
        for box in seq.gt_at(idx):
            if int(box.track_id) != int(target_id):
                continue
            if best is None or box.area > best.area:
                best = box
        out[idx] = None if best is None else np.asarray(best.xyxy, dtype=np.float64).copy()
    return out


def simulate_click(seq: Sequence, click_xy: tuple[int, int] | TSequence[int], frame_idx: int) -> int | None:
    """Which GT identity is under a click — the DJI "select this person" action.

    Args:
        seq: the sequence being inspected.
        click_xy: click position ``(x, y)`` in pixels.
        frame_idx: 0-based frame index of the click.

    Returns:
        The ``track_id`` of the box containing the click, choosing the **smallest**
        (most specific) box when several contain it — clicking a face inside a
        person box must select that person, not the crowd.  ``None`` when the
        click misses every box, which is the UI's "no target selected" case.
    """
    x, y = float(click_xy[0]), float(click_xy[1])
    candidates = []
    for box in seq.gt_at(int(frame_idx)):
        x0, y0, x1, y1 = (float(v) for v in box.xyxy)
        if x0 <= x <= x1 and y0 <= y <= y1:
            candidates.append(box)
    if not candidates:
        return None
    return int(min(candidates, key=lambda b: b.area).track_id)


def iter_frames(
    seq: Sequence,
    *,
    decode: bool = False,
    limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray | None, list[GtBox]]]:
    """Yield ``(frame_idx, image_or_None, gt_boxes)`` for the whole sequence.

    ``decode=False`` (the default) yields ``None`` for the image and touches no
    pixels — use it to walk annotations cheaply.  ``decode=True`` decodes every
    frame (cached by :class:`Sequence`).
    """
    stop = len(seq) if limit is None else min(len(seq), max(0, int(limit)))
    for idx in range(stop):
        yield idx, (seq.frame(idx) if decode else None), seq.gt_at(idx)


def sequence_summary(seq: Sequence) -> dict[str, Any]:
    """Compact JSON-friendly description of a sequence (for reports/logs)."""
    ids = seq.gt_ids()
    counts = {int(k): len(v) for k, v in sorted(seq.gt.items())}
    return {
        "name": seq.name,
        "num_frames": len(seq),
        "num_gt_boxes": seq.num_gt_boxes,
        "num_gt_ids": len(ids),
        "gt_ids": ids,
        "visible_frames": len(counts),
        "fps": seq.fps,
        "width": seq.width,
        "height": seq.height,
        "gt_target_id": seq.gt_target_id,
        "is_gt_only": seq.is_gt_only,
        "meta": dict(seq.meta),
    }
