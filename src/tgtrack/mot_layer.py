"""All-person MOT layer — "other persons only need an ID".

Route chosen: **DIRECT CLASS REUSE** (`BYTETracker` / `BOTSORT` / `OCSORT` /
`DeepOCSORT` / `FASTTracker` / `TRACKTRACK`), NOT `model.track()`.

Source evidence (ultralytics 8.4.185, wheel `_p2/ultralytics-8.4.185-py3-none-any.whl`):

* ``ultralytics/engine/model.py:570-611`` — ``Model.track()`` is only
  ``register_tracker(self, persist)`` + ``self.predict(mode="track")``; the
  predictor therefore runs ITS OWN detector every frame. Feeding our
  ``PersonDetector`` output through ``model.track()`` would detect twice.
* ``ultralytics/trackers/track.py:128-132`` — the official call is
  ``tracker.update(det, result.orig_img, **kwargs)`` where
  ``det = result.boxes.cpu().numpy()``: ultralytics itself hands the tracker an
  already-computed detection set, so reusing the class is exactly the internal path.
* ``ultralytics/trackers/track.py:19-26`` — ``TRACKER_MAP`` is the real registry
  (the six classes re-exported by ``ultralytics/trackers/__init__.py:3-9``).
* ``ultralytics/trackers/byte_tracker.py:274`` — signature
  ``update(self, results, img=None, feats=None, **kwargs) -> np.ndarray``.
* ``ultralytics/trackers/byte_tracker.py:285-286`` + ``512-514`` + the ``STrack.result``
  property (``byte_tracker.py:207-211``) — the return is ``(N, 8)``
  ``[x1, y1, x2, y2, track_id, score, cls, idx]`` where ``idx`` indexes back into the
  detection array we passed in. (OBB would be ``(N, 9)``; we always send xyxy.)
* ``ultralytics/trackers/byte_tracker.py:323-339`` (``_split_detections``) and
  ``ultralytics/trackers/utils/stracks.py:59-70`` (``parse_bboxes``) — the tracker does
  NOT take a raw ``(N, 6)`` array: it needs an object exposing ``.conf``, ``.cls`` and
  ``.xywh``/``.xywhr`` that supports boolean indexing. ``Boxes`` is that object
  (``ultralytics/engine/results.py:968-986``), and its data layout is
  ``[x1, y1, x2, y2, (track_id,) conf, cls]``.
* ``ultralytics/trackers/byte_tracker.py:341-361`` (``_input_for``) and
  ``ultralytics/trackers/utils/reid.py:139-161`` (``build_encoder``) — with
  ``model="auto"`` the "encoder" is ``_auto_encoder``, which consumes the per-detection
  features passed as the ``feats=`` kwarg; with an explicit path it runs that model on
  detection crops of ``img``. So ReID backends can use OUR embeddings (no second
  detector, no second encoder).

``reset()`` maps to ``byte_tracker.py:542-549`` (clears tracked/lost/removed, frame_id,
Kalman filter and the per-tracker ID iterator) — required between sequences so IDs do
not leak across videos.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import MotCfg
from .detector import PersonDetector
from .types import Detection

__all__ = [
    "AllPersonTracker",
    "TRACKER_YAMLS",
    "REID_CAPABLE",
    "dets_to_box_array",
    "apply_track_ids",
    "make_det_container",
    "resolve_tracker_yaml",
    "build_tracker_args",
]

#: The six backends shipped as ``ultralytics/cfg/trackers/*.yaml``.
TRACKER_YAMLS: tuple[str, ...] = (
    "bytetrack.yaml",
    "botsort.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
    "fasttrack.yaml",
    "tracktrack.yaml",
)
#: Backends that actually consume appearance features (``with_reid``).
REID_CAPABLE: tuple[str, ...] = ("botsort", "deepocsort", "tracktrack")


# ---------------------------------------------------------------------------
# pure helpers (unit-testable without ultralytics / torch)
# ---------------------------------------------------------------------------
def dets_to_box_array(dets: Iterable[Detection]) -> np.ndarray:
    """``Detection`` list -> ``(N, 6)`` float32 ``[x1, y1, x2, y2, conf, cls]``.

    That is exactly the ``Boxes`` data layout
    (``ultralytics/engine/results.py:976-985``): xyxy, then ``conf``, then ``cls`` —
    NOT ``[xyxy, cls, conf]``.
    """
    rows = [(*np.asarray(d.xyxy, dtype=np.float32).reshape(4).tolist(), float(d.score), float(d.cls)) for d in dets]
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def apply_track_ids(dets: list[Detection], tracks: np.ndarray | None) -> list[Detection]:
    """Write ``track_id`` back onto ``dets`` in place, using the tracker's ``idx`` column.

    Args:
        dets: Detections in the exact order they were handed to the tracker.
        tracks: The tracker's ``(N, 8)`` output ``[x1,y1,x2,y2,track_id,score,cls,idx]``
            (or ``(N, 9)`` for OBB). Column positions are read from the right
            (``-4`` = track id, ``-1`` = detection index) so both layouts work.

    Returns:
        The same list object, mutated in place.
    """
    if tracks is None:
        return dets
    arr = np.asarray(tracks)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return dets
    if arr.shape[1] < 8:
        raise ValueError(f"unexpected tracker output shape {arr.shape}, expected (N, 8)")
    n = len(dets)
    for row in arr:
        idx = int(round(float(row[-1])))
        if 0 <= idx < n:
            dets[idx].track_id = int(round(float(row[-4])))
    return dets


class _BoxesShim:
    """Minimal numpy stand-in for ``ultralytics.engine.results.Boxes``.

    Implements only the surface the trackers touch: ``__len__``, ``conf``, ``cls``,
    ``xyxy``, ``xywh`` and boolean-mask ``__getitem__``. Used if ``Boxes`` cannot be
    imported/constructed, so the layer degrades instead of crashing.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, 6)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: Any) -> "_BoxesShim":
        return _BoxesShim(np.asarray(self.data)[idx])

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def make_det_container(data: np.ndarray, orig_shape: tuple[int, int]) -> Any:
    """Wrap ``(N, 6)`` detection rows in the container the trackers expect.

    Prefers the real ultralytics ``Boxes`` (identical to the predictor's own input);
    falls back to :class:`_BoxesShim` if ultralytics is missing or the constructor fails.

    Args:
        data: ``(N, 6)`` ``[x1, y1, x2, y2, conf, cls]`` float32 array.
        orig_shape: ``(height, width)`` of the frame.
    """
    data = np.asarray(data, dtype=np.float32).reshape(-1, 6)
    try:
        from ultralytics.engine.results import Boxes

        return Boxes(data, orig_shape)
    except Exception:
        return _BoxesShim(data)


def resolve_tracker_yaml(name: str) -> Path:
    """Resolve a tracker yaml by path, by package name, or via ultralytics' search.

    Deterministic first (``<pkg>/cfg/trackers/<name>``) so we never hit
    ``check_file``'s "Multiple files match" error.
    """
    raw = Path(str(name))
    if raw.exists():
        return raw
    stem = raw.stem
    if stem not in {Path(t).stem for t in TRACKER_YAMLS}:
        # Project-local backends (e.g. configs/trackers/hybridsort.yaml, E012) live
        # outside ultralytics' package, so look next to the repo before giving up.
        for cand in (
            Path.cwd() / "configs" / "trackers" / f"{stem}.yaml",
            Path(__file__).resolve().parents[2] / "configs" / "trackers" / f"{stem}.yaml",
        ):
            if cand.exists():
                return cand
        raise ValueError(f"unknown tracker {name!r}; expected one of {list(TRACKER_YAMLS)}")
    try:
        import ultralytics

        cand = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / f"{stem}.yaml"
        if cand.exists():
            return cand
    except Exception:
        pass
    from ultralytics.utils.checks import check_yaml

    found = check_yaml(f"{stem}.yaml")
    if not found:
        raise FileNotFoundError(f"tracker yaml {stem}.yaml not found")
    return Path(str(found))


def build_tracker_args(cfg: MotCfg, device: str | None = None) -> tuple[Any, Path]:
    """Parse the yaml + apply ``MotCfg`` overrides -> ``(IterableSimpleNamespace, yaml_path)``.

    Only the ``MotCfg`` fields that differ from "do not touch" are written; everything
    else comes from ultralytics' own yaml, and ``MotCfg.extra`` is applied last (it can
    override any key, e.g. ``{"model": "yolo26n-reid.onnx", "fuse_score": False}``).
    """
    from ultralytics.utils import YAML, IterableSimpleNamespace

    path = resolve_tracker_yaml(cfg.tracker)
    data: dict[str, Any] = dict(YAML.load(path) or {})
    data.setdefault("tracker_type", Path(path).stem)
    overrides: dict[str, Any] = {
        "track_high_thresh": cfg.track_high_thresh,
        "track_low_thresh": cfg.track_low_thresh,
        "new_track_thresh": cfg.new_track_thresh,
        "track_buffer": cfg.track_buffer,
        "match_thresh": cfg.match_thresh,
        "with_reid": cfg.with_reid,
        "gmc_method": cfg.gmc_method,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    data.update(dict(cfg.extra or {}))
    if not data.get("model"):
        data["model"] = "auto"
    args = IterableSimpleNamespace(**data)
    args.device = device or getattr(args, "device", None) or _default_device()
    return args, path


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _tracker_map() -> dict[str, type]:
    """``{name: class}`` from ultralytics (``track.py:19-26``), with an import fallback."""
    try:
        from ultralytics.trackers.track import TRACKER_MAP

        return dict(TRACKER_MAP)
    except Exception:
        from ultralytics.trackers import BOTSORT, BYTETracker, DeepOCSORT, FASTTracker, OCSORT, TRACKTRACK

        return {
            "bytetrack": BYTETracker,
            "botsort": BOTSORT,
            "ocsort": OCSORT,
            "deepocsort": DeepOCSORT,
            "fasttrack": FASTTracker,
            "tracktrack": TRACKTRACK,
        }


class AllPersonTracker:
    """All-person multi-object tracker wrapping an ultralytics backend.

    Example:
        >>> mot = AllPersonTracker(MotCfg(tracker="deepocsort.yaml"), detector=det)
        >>> dets = det.detect(frame)
        >>> mot.update(frame, dets)          # det.track_id filled in place
        >>> mot.reset()                      # between videos / sequences
    """

    def __init__(self, cfg: MotCfg | None = None, detector: PersonDetector | None = None) -> None:
        self.cfg = cfg or MotCfg()
        self.detector = detector
        self.tracker_yaml = resolve_tracker_yaml(self.cfg.tracker)
        self.tracker_type = Path(self.tracker_yaml).stem
        if self.cfg.with_reid and self.tracker_type not in REID_CAPABLE:
            warnings.warn(
                f"tracker {self.tracker_type!r} does not consume appearance features; "
                f"with_reid=True is ignored (ReID-capable: {list(REID_CAPABLE)})",
                RuntimeWarning,
                stacklevel=2,
            )
        device = getattr(detector, "device", None) if detector is not None else None
        self.args, _ = build_tracker_args(self.cfg, device=device)
        tracker_cls = _tracker_map()[self.args.tracker_type]
        # same construction call the official callback uses (track.py:90)
        self._tracker = tracker_cls(args=self.args)
        self.n_updates = 0
        self._warned_empty = False
        self._warned_feats = False

    # ------------------------------------------------------------------ util
    def _resolve_feats(self, dets: list[Detection], feats: np.ndarray | None) -> np.ndarray | None:
        """Per-detection appearance features for the tracker's native ReID hook.

        Returns an ``(N, D)`` array aligned 1:1 with ``dets``, or None. None is the safe
        value: ``_input_for`` (``byte_tracker.py:359-360``) turns it into "no ReID this
        frame" for ``model="auto"`` instead of feeding a BGR frame into the auto encoder.
        """
        if feats is not None:
            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim == 1 and len(dets) == 1:
                arr = arr[None, :]
            if arr.ndim == 2 and arr.shape[0] == len(dets) and np.isfinite(arr).all():
                return arr
            if not self._warned_feats:
                self._warned_feats = True
                warnings.warn(
                    f"ignoring feats with shape {getattr(feats, 'shape', None)} for {len(dets)} detections",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return None
        if not dets or any(d.feat is None for d in dets):
            return None
        dims = {int(np.asarray(d.feat).size) for d in dets}
        if len(dims) != 1:
            return None
        return np.stack([np.asarray(d.feat, dtype=np.float32).reshape(-1) for d in dets])

    def _advance_empty(self, img_bgr: np.ndarray) -> None:
        """Step the tracker on a frame with no detections (keeps track ages honest)."""
        try:
            container = make_det_container(np.zeros((0, 6), dtype=np.float32), img_bgr.shape[:2])
            self._tracker.update(container, img_bgr)
            self.n_updates += 1
        except Exception as exc:  # never let an empty frame kill a run
            if not self._warned_empty:
                self._warned_empty = True
                warnings.warn(f"empty-frame tracker update failed ({type(exc).__name__}: {exc})", RuntimeWarning)

    # ---------------------------------------------------------------- public
    def update(
        self,
        img_bgr: np.ndarray,
        dets: list[Detection],
        feats: np.ndarray | None = None,
    ) -> list[Detection]:
        """Assign stable ids to ``dets`` in place and return the same list.

        Args:
            img_bgr: Current BGR frame (used by GMC and by an external ReID encoder).
            dets: Detections from ``PersonDetector`` (same frame).
            feats: Optional ``(N, D)`` appearance features aligned with ``dets``. When
                omitted, ``det.feat`` is used if EVERY detection has one.

        Returns:
            ``dets``, with ``track_id`` set (None for detections the tracker did not
            activate this frame, e.g. low-score second-stage misses).
        """
        dets = list(dets)
        for d in dets:
            d.track_id = None
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
            raise ValueError(f"update() expects an (H, W, 3) BGR ndarray, got {type(img_bgr)}")
        if not dets:
            self._advance_empty(img_bgr)
            return dets
        container = make_det_container(dets_to_box_array(dets), img_bgr.shape[:2])
        per_det_feats = self._resolve_feats(dets, feats)
        if per_det_feats is not None:
            tracks = self._tracker.update(container, img_bgr, feats=per_det_feats)
        else:
            tracks = self._tracker.update(container, img_bgr)
        self.n_updates += 1
        apply_track_ids(dets, tracks)
        return dets

    def reset(self) -> None:
        """Clear all tracker state (tracks, Kalman filter, ID counter).

        Must be called between sequences: leaked state corrupts MOTA/IDF1/HOTA.
        """
        self._tracker.reset()
        self.n_updates = 0

    # ------------------------------------------------------------------ info
    @property
    def names(self) -> dict:
        """Tracker namespace as a plain dict (the effective per-key config)."""
        return dict(vars(self.args))

    def describe(self) -> dict[str, Any]:
        """Reproducibility record of the effective tracker configuration."""
        keys = (
            "tracker_type",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "track_buffer",
            "match_thresh",
            "fuse_score",
            "with_reid",
            "model",
            "gmc_method",
            "proximity_thresh",
            "appearance_thresh",
        )
        return {k: getattr(self.args, k, None) for k in keys} | {
            "tracker_yaml": str(self.tracker_yaml),
            "device": getattr(self.args, "device", None),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AllPersonTracker({self.tracker_type!r}, reid={self.cfg.with_reid})"
