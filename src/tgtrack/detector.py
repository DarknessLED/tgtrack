"""YOLO26 person detector — the "all persons" front end.

Verified facts about ultralytics 8.4.185 that shape this module (read from the
wheel ``_p2/ultralytics-8.4.185-py3-none-any.whl``):

* ``YOLO(weights).predict(...)`` returns ``Results`` whose ``boxes`` are already in
  ORIGINAL-image pixel coordinates (the predictor letterboxes on the way in and
  calls ``scale_boxes`` on the way out). We therefore never rescale here.
* ``Boxes.data`` layout is ``[x1, y1, x2, y2, (track_id,) conf, cls]`` — see
  ``ultralytics/engine/results.py:976-985`` (``assert n in {6, 7}``).
* ``half=True`` is only valid on CUDA; ultralytics raises/falls back on CPU, so we
  force ``half=False`` unless the resolved device is a CUDA device.

The post-processing is factored into the pure function :func:`boxes_to_detections`
so it can be unit-tested on synthetic tensors without a GPU, a checkpoint or even
ultralytics being importable.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DetectorCfg
from .types import Detection

__all__ = [
    "MIRROR_TEMPLATES",
    "PersonDetector",
    "boxes_to_detections",
    "download_weights",
    "visible_ratio",
    "compare_models",
]

# ---------------------------------------------------------------------------
# parts.py is owned by appearance-dev and may not exist yet: guard the import.
# ---------------------------------------------------------------------------
_PARTS_VISIBLE_RATIO: Any = None
_PARTS_VISIBLE_RATIO_SIG = "local"
try:  # pragma: no cover - depends on a parallel agent's file
    from .parts import visible_ratio as _parts_visible_ratio  # type: ignore

    _PARTS_VISIBLE_RATIO = _parts_visible_ratio
    _PARTS_VISIBLE_RATIO_SIG = "parts.visible_ratio"
except Exception:  # ImportError, or a partially written module
    _PARTS_VISIBLE_RATIO = None


def visible_ratio(xyxy: np.ndarray, width: int, height: int) -> float:
    """Fraction of the box AREA that lies inside the frame (1.0 = fully inside).

    Matches the contract documented on ``types.Detection.visible_ratio``:
    "Fraction of the crop that is inside the frame (1.0 = fully inside)."

    Args:
        xyxy: Box in ``[x1, y1, x2, y2]`` pixel coordinates.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Float in ``[0, 1]``; ``0.0`` for a degenerate (zero-area) box.
    """
    x0, y0, x1, y1 = (float(v) for v in np.asarray(xyxy, dtype=np.float64).reshape(4))
    box_area = (x1 - x0) * (y1 - y0)
    if box_area <= 0 or width <= 0 or height <= 0:
        return 0.0
    ix = max(0.0, min(x1, float(width)) - max(x0, 0.0))
    iy = max(0.0, min(y1, float(height)) - max(y0, 0.0))
    return float(min(1.0, max(0.0, (ix * iy) / box_area)))


def _compute_visible_ratio(xyxy: np.ndarray, width: int, height: int) -> float:
    """Use ``parts.visible_ratio`` when its signature fits, else the local geometry."""
    if _PARTS_VISIBLE_RATIO is not None:
        for args in ((xyxy, width, height), (xyxy, (height, width))):
            try:
                return float(_PARTS_VISIBLE_RATIO(*args))
            except TypeError:
                continue
            except Exception:
                break
    return visible_ratio(xyxy, width, height)


def boxes_to_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    width: int,
    height: int,
    conf_thres: float = 0.25,
    person_class: int | None = 0,
    max_det: int = 300,
) -> list[Detection]:
    """Pure post-processing: raw detector rows -> filtered ``Detection`` list.

    Order of operations (each step independently unit-tested):

    1. flatten to ``(N, 4)`` / ``(N,)`` arrays;
    2. keep only ``cls == person_class`` when ``person_class`` is not None;
    3. keep only ``conf >= conf_thres``;
    4. drop non-finite and degenerate (``w <= 0`` or ``h <= 0``) boxes;
    5. sort by score descending, truncate to ``max_det``;
    6. build ``Detection`` with ``quality = score`` and ``visible_ratio`` from geometry.

    Args:
        xyxy: ``(N, 4)`` raw boxes in original-image pixels.
        conf: ``(N,)`` scores.
        cls: ``(N,)`` class indices.
        width: Frame width (for ``visible_ratio``).
        height: Frame height (for ``visible_ratio``).
        conf_thres: Minimum score to keep.
        person_class: COCO person index (0); None keeps every class.
        max_det: Maximum number of detections to return.

    Returns:
        List of ``Detection`` sorted by descending score.
    """
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(conf, dtype=np.float32).reshape(-1)
    classes = np.asarray(cls).reshape(-1)
    n = min(len(boxes), len(scores), len(classes))
    boxes, scores, classes = boxes[:n], scores[:n], classes[:n]

    keep = np.ones(n, dtype=bool)
    if person_class is not None:
        keep &= classes.astype(np.int64) == int(person_class)
    keep &= np.isfinite(scores) & (scores >= float(conf_thres))
    keep &= np.isfinite(boxes).all(axis=1)
    wh = boxes[:, 2:] - boxes[:, :2]
    keep &= (wh[:, 0] > 0) & (wh[:, 1] > 0)

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return []
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    if max_det is not None and max_det > 0:
        idx = idx[: int(max_det)]

    out: list[Detection] = []
    for i in idx:
        box = boxes[i].astype(np.float32)
        score = float(scores[i])
        out.append(
            Detection(
                xyxy=box,
                score=score,
                cls=int(classes[i]),
                quality=score,
                visible_ratio=_compute_visible_ratio(box, width, height),
            )
        )
    return out


#: github.com is unreachable from this machine, so ``yolo26*.pt`` is fetched from
#: mirrors instead. ``{}`` is the weights filename (e.g. ``yolo26n.pt``).
MIRROR_TEMPLATES: tuple[str, ...] = (
    "https://hf-mirror.com/Ultralytics/YOLO26/resolve/main/{}",
    "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.4.0/{}",
)
_MIRROR_MIN_BYTES = 1_000_000

_QUANTIZE_SUPPORTED: bool | None = None


def _supports_quantize() -> bool:
    """True when the installed ultralytics uses the unified ``quantize`` arg."""
    global _QUANTIZE_SUPPORTED
    if _QUANTIZE_SUPPORTED is None:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            _QUANTIZE_SUPPORTED = "quantize" in DEFAULT_CFG_DICT
        except Exception:
            _QUANTIZE_SUPPORTED = False
    return _QUANTIZE_SUPPORTED


def download_weights(name: str, dest_dir: str | Path | None = None, timeout: float = 120.0) -> Path | None:
    """Fetch ``yolo26*.pt`` from a mirror when GitHub is blocked.

    Returns the local path on success (or the already-present file), else ``None``.
    Existing files are never overwritten, and a short download is rejected.
    """
    import urllib.request

    dest_dir = Path(dest_dir) if dest_dir is not None else Path.cwd()
    dest = dest_dir / Path(name).name
    if dest.exists() and dest.stat().st_size >= _MIRROR_MIN_BYTES:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for tpl in MIRROR_TEMPLATES:
        url = tpl.format(Path(name).name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tgtrack/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if tmp.stat().st_size < _MIRROR_MIN_BYTES:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dest)
            warnings.warn(f"downloaded {name} from mirror {url}", RuntimeWarning, stacklevel=2)
            return dest
        except Exception:
            tmp.unlink(missing_ok=True)
    return None


def _resolve_device(requested: str | None) -> str:
    """Return a usable torch device string, falling back to 'cpu' with a warning."""
    want = (requested or "cuda").strip()
    if want.lower().startswith("cuda") or want.isdigit():
        try:
            import torch

            if torch.cuda.is_available():
                return want
        except Exception:
            pass
        warnings.warn(
            f"PersonDetector: device {requested!r} requested but CUDA is unavailable "
            "(torch may still be the CPU build) -> falling back to 'cpu'",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return want


def fuse_detections(groups: list[list[Detection]], iou_thr: float = 0.55) -> list[Detection]:
    """Merge person boxes from several detectors with greedy NMS (E055).

    Why fuse at the box level instead of ensembling scores: the two detectors fail on
    *different* people.  The general model anchors tight boxes on the people it sees; a
    CrowdHuman specialist finds the occluded ones with looser boxes.  Measured on the E003
    protocol the union recalls 0.830 against 0.731 (general) and 0.803 (crowd), at 25.9
    detections/frame against 19.1 / 22.8 (``results/e054_union.json``).

    Args:
        groups: one list of ``Detection`` per detector, already filtered to persons.
        iou_thr: suppress a lower-scoring box whose IoU with a kept box is ``>= iou_thr``.

    Returns:
        Fused detections sorted by descending score. Higher score wins a suppressed pair, so a
        confident tight box from the general model keeps its geometry over a loose duplicate.
    """
    flat: list[Detection] = [d for group in groups for d in (group or [])]
    if len(flat) <= 1:
        return flat
    order = sorted(range(len(flat)), key=lambda i: -float(flat[i].score))
    keep: list[int] = []
    for i in order:
        di = flat[i]
        ai = max(0.0, di.xyxy[2] - di.xyxy[0]) * max(0.0, di.xyxy[3] - di.xyxy[1])
        suppressed = False
        for j in keep:
            dj = flat[j]
            ix = max(0.0, min(di.xyxy[2], dj.xyxy[2]) - max(di.xyxy[0], dj.xyxy[0]))
            iy = max(0.0, min(di.xyxy[3], dj.xyxy[3]) - max(di.xyxy[1], dj.xyxy[1]))
            inter = ix * iy
            if inter <= 0.0:
                continue
            aj = max(0.0, dj.xyxy[2] - dj.xyxy[0]) * max(0.0, dj.xyxy[3] - dj.xyxy[1])
            if inter / max(ai + aj - inter, 1e-9) >= float(iou_thr):
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return [flat[i] for i in keep]


def snap_geometry(
    primary: list[Detection],
    snappers: list[Detection],
    snap_iou: float = 0.30,
) -> list[Detection]:
    """Replace each primary box's geometry with a better-anchored overlapping box (E056).

    The premise is measured, not assumed.  E047 showed a CrowdHuman specialist finds 7.2 pp more
    people than the general model but its boxes are loose, and the looseness costs association
    (AssA 0.5431 -> 0.5341).  E055 then showed that simply ADDING the general model's boxes makes
    HOTA worse (0.4944), because extra boxes are extra false positives and duplicates.  Snapping
    is the third option: keep the specialist's box COUNT and scores (its recall), but where the
    general model has a tighter box on the same person, adopt the tighter geometry.

    Args:
        primary: the detections whose count and scores are preserved.
        snappers: candidate geometries, typically from the tighter general model.
        snap_iou: minimum IoU between the two boxes to consider them the same person.

    Returns:
        A new list of ``Detection``-shaped boxes (score/cls/feature fields of the primary item
        are preserved; only ``xyxy`` changes, so callers must re-derive crops if they cached them).
    """
    if not primary or not snappers:
        return primary
    snap = np.asarray([d.xyxy for d in snappers], dtype=np.float64).reshape(-1, 4)
    p = np.asarray([d.xyxy for d in primary], dtype=np.float64).reshape(-1, 4)
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_s = np.clip(snap[:, 2] - snap[:, 0], 0, None) * np.clip(snap[:, 3] - snap[:, 1], 0, None)
    ix = np.clip(np.minimum(p[:, None, 2], snap[None, :, 2]) - np.maximum(p[:, None, 0], snap[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(p[:, None, 3], snap[None, :, 3]) - np.maximum(p[:, None, 1], snap[None, :, 1]), 0, None)
    inter = ix * iy
    union = np.maximum(area_p[:, None] + area_s[None, :] - inter, 1e-9)
    iou = inter / union
    best = np.argmax(iou, axis=1)
    best_iou = iou[np.arange(len(p)), best]
    out = list(primary)
    for i in np.nonzero(best_iou >= float(snap_iou))[0]:
        d = out[int(i)]
        out[int(i)] = Detection(
            xyxy=snap[int(best[i])].astype(np.float32),
            score=d.score,
            cls=d.cls,
            feat=d.feat,
            part_feats=d.part_feats,
            track_id=d.track_id,
            quality=d.quality,
            sharpness=d.sharpness,
            visible_ratio=d.visible_ratio,
        )
    return out


class PersonDetector:
    """YOLO26 person-only detector.

    Example:
        >>> det = PersonDetector(DetectorCfg(weights="yolo26n.pt", device="cuda"))
        >>> dets = det.detect(img_bgr)          # boxes already in original pixels
        >>> det.model_info()["n_params"]
    """

    def __init__(self, cfg: DetectorCfg | None = None) -> None:
        self.cfg = cfg or DetectorCfg()
        self.device = _resolve_device(self.cfg.device)
        # ultralytics refuses half precision off CUDA.
        self.half = bool(self.cfg.half and self.device.lower().startswith("cuda"))
        self._warmed = False
        self.model = self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> Any:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "ultralytics is not importable yet "
                f"({type(exc).__name__}: {exc}). The environment owner (lead) is installing "
                "torch cu126 + ultralytics; retry once that finishes."
            ) from exc
        weights = self.cfg.weights
        path = Path(weights)
        if not path.exists() and path.suffix == "":
            weights = str(path.with_suffix(".pt"))
        if not Path(weights).exists():
            # github.com is blocked on this machine, so skip ultralytics' 3x30 s
            # retry loop and go straight to the mirror for a missing checkpoint.
            local = download_weights(Path(weights).name)
            if local is not None:
                weights = str(local)
        try:
            return YOLO(weights)
        except Exception as exc:  # pragma: no cover - network dependent
            local = download_weights(Path(weights).name)
            if local is not None:
                try:
                    return YOLO(str(local))
                except Exception as exc2:  # pragma: no cover - corrupt download
                    raise RuntimeError(f"mirror weights {local} failed to load: {exc2}") from exc2
            raise RuntimeError(
                f"failed to load weights {weights!r}: {type(exc).__name__}: {exc}. "
                "yolo26*.pt auto-downloads from https://github.com/ultralytics/assets; "
                "github.com is unreachable here, and no configured mirror served it either "
                f"(tried: {', '.join(MIRROR_TEMPLATES)}). Place the .pt next to the project manually."
            ) from exc

    # --------------------------------------------------------------- predict
    @property
    def predict_kwargs(self) -> dict[str, Any]:
        """Keyword arguments handed to ``ultralytics`` predict (single source of truth).

        ultralytics 8.4.185 unifies INT8/FP16/FP32 under ``quantize`` and maps the
        legacy ``half`` flag onto it (``ultralytics/cfg/__init__.py:617-621``,
        ``deprecation_warn("half", "quantize")``; ``engine/predictor.py:471`` reads
        ``fp16=self.args.quantize == 16``). We therefore send ``quantize=16`` for
        FP16 and ``quantize=32`` for FP32, and only fall back to ``half=`` on an
        older ultralytics that has no ``quantize`` key.
        """
        kwargs: dict[str, Any] = {
            "imgsz": int(self.cfg.imgsz),
            "conf": float(self.cfg.conf),
            "iou": float(self.cfg.iou),
            "max_det": int(self.cfg.max_det),
            "device": self.device,
            "verbose": False,
        }
        if _supports_quantize():
            kwargs["quantize"] = 16 if self.half else 32
        else:  # pragma: no cover - older ultralytics only
            kwargs["half"] = self.half
        if self.cfg.person_class is not None:
            kwargs["classes"] = [int(self.cfg.person_class)]
        return kwargs

    def warmup(self) -> None:
        """Run one dummy forward pass so the first real frame is not timing noise."""
        dummy = np.zeros((max(64, int(self.cfg.imgsz) // 2), int(self.cfg.imgsz), 3), dtype=np.uint8)
        self.detect(dummy)
        self._warmed = True

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Detect persons in one BGR frame.

        Args:
            img_bgr: ``(H, W, 3)`` uint8 BGR frame (the OpenCV convention).

        Returns:
            Person-only ``Detection`` objects, boxes in ORIGINAL pixel xyxy,
            sorted by descending score.
        """
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
            raise ValueError(f"detect() expects an (H, W, 3) BGR ndarray, got {type(img_bgr)}")
        res = self.model.predict(source=img_bgr, **self.predict_kwargs)[0]
        height, width = int(res.orig_shape[0]), int(res.orig_shape[1])
        boxes = getattr(res, "boxes", None)
        dets = []
        if boxes is not None and len(boxes) != 0:
            dets = boxes_to_detections(
                self._to_numpy(boxes.xyxy),
                self._to_numpy(boxes.conf),
                self._to_numpy(boxes.cls),
                width,
                height,
                conf_thres=float(self.cfg.conf),
                person_class=self.cfg.person_class,
                max_det=int(self.cfg.max_det),
            )
        extras = self._extra_detections(img_bgr, width, height, tuple(getattr(self.cfg, "extra_weights", ()) or ()))
        if extras:
            dets = fuse_detections([dets, *extras], float(self.cfg.fusion_iou))
        snappers = self._extra_detections(img_bgr, width, height, tuple(getattr(self.cfg, "snap_weights", ()) or ()))
        if snappers:
            dets = snap_geometry(dets, [d for group in snappers for d in group], float(self.cfg.snap_iou))
        return dets

    def _extra_detections(
        self,
        img_bgr: np.ndarray,
        width: int,
        height: int,
        wanted: tuple[str, ...],
    ) -> list[list[Detection]]:
        """Run each checkpoint in ``wanted`` and return its person boxes.

        Used for both roles: E055 box fusion (`cfg.extra_weights`) and E056 geometry snapping
        (`cfg.snap_weights`). Lazily loads each checkpoint on first use — nothing is imported
        or downloaded unless the caller configured one — and reuses the primary detector's
        predict settings so resolution/conf/device stay consistent. A checkpoint with a single
        class is treated as person-only, so ``person_class=0`` is correct for it.
        """
        if not wanted:
            return []
        cache = getattr(self, "_extra_models", None)
        if cache is None:
            cache = {}
            self._extra_models = cache
        for name in wanted:
            key = str(name)
            if key in cache:
                continue
            from ultralytics import YOLO  # local import: keeps `import tgtrack` cheap

            model = YOLO(key)
            n_classes = len(getattr(model, "names", {}) or {})
            cache[key] = (model, 0 if n_classes <= 1 else self.cfg.person_class)
        out: list[list[Detection]] = []
        for name in wanted:
            model, person_class = cache[str(name)]
            kwargs = dict(self.predict_kwargs)
            kwargs["classes"] = None if person_class is None else [int(person_class)]
            res = model.predict(source=img_bgr, **kwargs)[0]
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            out.append(
                boxes_to_detections(
                    self._to_numpy(boxes.xyxy),
                    self._to_numpy(boxes.conf),
                    self._to_numpy(boxes.cls),
                    width,
                    height,
                    conf_thres=float(self.cfg.conf),
                    person_class=person_class,
                    max_det=int(self.cfg.max_det),
                )
            )
        return out

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """torch.Tensor / np.ndarray -> np.ndarray (no-op for numpy)."""
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x)

    # ------------------------------------------------------------------ info
    @property
    def names(self) -> dict:
        """Class index -> name mapping of the loaded model."""
        names = getattr(self.model, "names", None) or {}
        return dict(names)

    def model_info(self) -> dict[str, Any]:
        """Honest description of what is actually loaded (params, file, device)."""
        info: dict[str, Any] = {
            "weights": self.cfg.weights,
            "device": self.device,
            "half": self.half,
            "imgsz": int(self.cfg.imgsz),
            "conf": float(self.cfg.conf),
            "iou": float(self.cfg.iou),
            "max_det": int(self.cfg.max_det),
            "person_class": self.cfg.person_class,
            "n_classes": len(self.names),
        }
        weights_path = None
        try:
            ckpt = getattr(self.model, "ckpt_path", None)
            if ckpt:
                weights_path = Path(str(ckpt))
            else:
                p = Path(self.cfg.weights)
                weights_path = p if p.exists() else None
        except Exception:
            weights_path = None
        if weights_path is not None and weights_path.exists():
            info["weights_path"] = str(weights_path.resolve())
            info["weights_bytes"] = int(weights_path.stat().st_size)
        else:
            info["weights_path"] = None
            info["weights_bytes"] = None
        n_params = None
        try:
            module = getattr(self.model, "model", None)
            if module is not None and hasattr(module, "parameters"):
                n_params = int(sum(p.numel() for p in module.parameters()))
        except Exception:
            n_params = None
        info["n_params"] = n_params
        info["n_params_M"] = None if n_params is None else round(n_params / 1e6, 4)
        return info

    def benchmark(self, img_bgr: np.ndarray, n: int = 50) -> dict[str, Any]:
        """Time ``detect`` on one frame.

        Args:
            img_bgr: Frame to run repeatedly.
            n: Number of timed iterations (excludes a preceding warmup pass).

        Returns:
            ``{'fps', 'ms_mean', 'ms_p95', 'ms_min', 'device', 'n', 'half',
            'imgsz', 'n_persons', 'n_params'}``.
        """
        n = max(1, int(n))
        if not self._warmed:
            self.warmup()
        self.detect(img_bgr)  # untimed
        times = np.empty(n, dtype=np.float64)
        n_persons = 0
        for i in range(n):
            t0 = time.perf_counter()
            dets = self.detect(img_bgr)
            times[i] = (time.perf_counter() - t0) * 1000.0
            n_persons = len(dets)
        return {
            "fps": float(1000.0 / max(times.mean(), 1e-9)),
            "ms_mean": float(times.mean()),
            "ms_p95": float(np.percentile(times, 95)),
            "ms_min": float(times.min()),
            "device": self.device,
            "half": self.half,
            "imgsz": int(self.cfg.imgsz),
            "n": n,
            "n_persons": int(n_persons),
            "n_params": self.model_info()["n_params"],
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PersonDetector(weights={self.cfg.weights!r}, device={self.device!r}, "
            f"half={self.half}, imgsz={self.cfg.imgsz}, conf={self.cfg.conf})"
        )


def compare_models(
    weights: Sequence[str],
    img_bgr: np.ndarray | None = None,
    n: int = 30,
    imgsz: int = 640,
    device: str = "cuda",
    conf: float = 0.25,
) -> list[dict[str, Any]]:
    """Benchmark several checkpoints on one frame — evidence for the model choice.

    Args:
        weights: e.g. ``["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]``.
        img_bgr: Frame to benchmark on; a synthetic 720p frame when None.
        n: Timed iterations per model.
        imgsz: Inference size.
        device: ``"cuda"`` / ``"cpu"``.
        conf: Confidence threshold.

    Returns:
        One dict per model: ``model_info()`` fields plus ``benchmark()`` numbers.
    """
    if img_bgr is None:
        img_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for w in weights:
        try:
            det = PersonDetector(DetectorCfg(weights=w, imgsz=imgsz, device=device, conf=conf))
        except Exception as exc:  # pragma: no cover - network dependent
            rows.append({"weights": w, "error": f"{type(exc).__name__}: {exc}"})
            continue
        info = det.model_info()
        bench = det.benchmark(img_bgr, n=n)
        row = {"weights": w, **{k: info[k] for k in ("n_params", "n_params_M", "weights_bytes", "device")}, **bench}
        rows.append(row)
        print(
            f"{w:16s} params={info['n_params_M']}M  device={bench['device']}  "
            f"ms_mean={bench['ms_mean']:.2f}  ms_p95={bench['ms_p95']:.2f}  fps={bench['fps']:.2f}  "
            f"persons={bench['n_persons']}"
        )
        del det
    return rows


if __name__ == "__main__":  # pragma: no cover - manual evidence run
    # `python src/tgtrack/detector.py --compare yolo26n.pt yolo26s.pt yolo26m.pt`
    import argparse

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser(description="PersonDetector evidence run")
    ap.add_argument("--compare", nargs="*", default=["yolo26n.pt"])
    ap.add_argument("--source", default=None, help="image to benchmark on (default: synthetic 720p)")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    frame = None
    if a.source:
        import cv2

        frame = cv2.imread(a.source)
    rows = compare_models(a.compare, frame, n=a.n, imgsz=a.imgsz, device=a.device)
    import json

    print(json.dumps(rows, indent=2))
