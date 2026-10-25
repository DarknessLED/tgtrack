"""Appearance embedding for detections and part crops.

Wraps ultralytics' own YOLO26 ReID encoder (``yolo26{k}-reid.onnx``) so the
target tracker and the all-person MOT layer share one embedding space and one
preprocessing path. The encoder is part-aware: for every (box, part) pair it
produces an L2-normalized embedding, in ONE batched forward pass per chunk.

Two modes, never blurred together:

``mode="onnx"`` / ``mode="pt"``
    The real YOLO26-ReID network via ``ultralytics.trackers.utils.reid.ReID``.
``mode="fallback"``
    A deterministic handcrafted descriptor (HSV histogram + HOG-lite spatial
    gradient histograms) so the whole pipeline still runs, and is testable,
    before ``onnxruntime``/CUDA/the weights exist. It is a *different space* and
    must never be reported as neural-encoder output: ``is_real`` is False and
    ``fallback_reason`` says exactly why the real path was unavailable.

The fallback descriptor is not merely a stub: it is position-invariant, dim-
fixed and L2-normalized, so identical crops map to identical embeddings and the
memory banks can be exercised end to end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

import cv2
import numpy as np

from .config import EmbedderCfg
from .parts import crop as crop_box
from .parts import part_boxes
from .types import PART_KEYS

try:  # ORT/CUDA shim (Lead-owned). Optional so this module stays importable alone.
    from .runtime import ensure_ort_cuda, ort_providers
except Exception:  # pragma: no cover - standalone use without the shim
    def ensure_ort_cuda(verbose: bool = False) -> dict:  # type: ignore[misc]
        return {"applied": False}

    def ort_providers(device: str = "auto", use_tensorrt: bool = False) -> list[str]:  # type: ignore[misc]
        return ["CPUExecutionProvider"]

__all__ = ["FALLBACK_DIM", "Embedder", "fallback_descriptor"]

# --- fallback descriptor geometry -------------------------------------------
#: Dimensions of the fixed-size patch the fallback descriptor is computed on.
_FB_W, _FB_H = 32, 64
#: HSV histograms: hue / saturation / value bins (12 + 8 + 8 = 28 features).
_FB_H_BINS, _FB_S_BINS, _FB_V_BINS = 12, 8, 8
#: HOG-lite: 2x4 spatial cells, 8 orientation bins (64 features) + 8 energy.
_FB_CELLS_X, _FB_CELLS_Y, _FB_ORI = 2, 4, 8
#: Total dimensionality of the fallback descriptor.
FALLBACK_DIM = (
    _FB_H_BINS
    + _FB_S_BINS
    + _FB_V_BINS
    + _FB_CELLS_X * _FB_CELLS_Y * _FB_ORI
    + _FB_CELLS_X * _FB_CELLS_Y
)


def fallback_descriptor(crop_bgr: np.ndarray) -> np.ndarray:
    """Deterministic L2-normalized handcrafted descriptor of one BGR crop.

    Composition (``FALLBACK_DIM`` = 100):
        * 12-bin hue, 8-bin saturation, 8-bin value histograms — coarse colour
          identity (clothing colour is the single most useful cue available
          without a network);
        * 2x4 cells x 8 orientations of gradient-orientation histograms —
          silhouette/structure identity, roughly where the HOG "lite" budget
          lands;
        * 8 per-cell mean gradient magnitudes — "is there structure here at
          all", which tells a flat/blank crop apart from a textured one.

    The crop is resized with ``INTER_AREA`` when shrinking so downscaling does
    not alias. Fully degenerate input (empty array) yields a zero vector; every
    non-empty image yields a unit-norm vector.

    Args:
        crop_bgr: Cropped person/part image (BGR or grayscale).

    Returns:
        ``(FALLBACK_DIM,)`` float32 with (almost always) unit L2 norm.
    """
    img = np.asarray(crop_bgr)
    if img.size == 0:
        return np.zeros((FALLBACK_DIM,), dtype=np.float32)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w > _FB_W or h > _FB_H) else cv2.INTER_LINEAR
    small = cv2.resize(img, (_FB_W, _FB_H), interpolation=interp)

    # --- colour ---
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    px = float(_FB_W * _FB_H)
    hist_h, _ = np.histogram(hsv[:, :, 0], bins=_FB_H_BINS, range=(0.0, 180.0))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=_FB_S_BINS, range=(0.0, 256.0))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=_FB_V_BINS, range=(0.0, 256.0))
    colour = np.concatenate(
        [hist_h / px, hist_s / px, hist_v / px]
    ).astype(np.float32)

    # --- HOG-lite ---
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)  # unsigned orientation in [0, pi)

    ys, xs = np.mgrid[0:_FB_H, 0:_FB_W]
    cell = (ys * _FB_CELLS_Y // _FB_H) * _FB_CELLS_X + (xs * _FB_CELLS_X // _FB_W)
    ori = np.clip((ang / np.pi * _FB_ORI).astype(np.int32), 0, _FB_ORI - 1)
    flat = (cell * _FB_ORI + ori).ravel()
    hog = np.bincount(flat, weights=mag.ravel(), minlength=_FB_CELLS_X * _FB_CELLS_Y * _FB_ORI)
    hog = hog.reshape(_FB_CELLS_X * _FB_CELLS_Y, _FB_ORI).astype(np.float32)
    hog /= np.maximum(hog.sum(axis=1, keepdims=True), 1e-6)

    cell_mag = np.bincount(
        cell.ravel(), weights=mag.ravel(), minlength=_FB_CELLS_X * _FB_CELLS_Y
    ).astype(np.float32)
    energy = (cell_mag / max(float(_FB_W * _FB_H), 1.0)).astype(np.float32).reshape(-1, 1)

    desc = np.concatenate([colour, hog.ravel(), energy.ravel()]).astype(np.float32)
    if desc.size != FALLBACK_DIM:  # pragma: no cover - geometry guard
        raise RuntimeError(f"fallback descriptor size {desc.size} != FALLBACK_DIM {FALLBACK_DIM}")
    if not np.all(np.isfinite(desc)):
        return np.zeros((FALLBACK_DIM,), dtype=np.float32)
    n = float(np.linalg.norm(desc))
    if n < 1e-12:
        return np.zeros((FALLBACK_DIM,), dtype=np.float32)
    return np.ascontiguousarray(desc / n, dtype=np.float32)


def _as_bgr_uint8(img: Any) -> np.ndarray | None:
    """Coerce an image/crop to a contiguous uint8 BGR array, or None."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        return None
    if img.dtype != np.uint8:
        img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _unit(feat: Any) -> np.ndarray | None:
    """L2-normalized finite float32 row vector, or None when unusable."""
    if feat is None:
        return None
    f = np.asarray(feat, dtype=np.float32).reshape(-1)
    if f.size == 0 or not np.all(np.isfinite(f)):
        return None
    n = float(np.linalg.norm(f))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return np.ascontiguousarray(f / n, dtype=np.float32)


class Embedder:
    """Part-aware appearance encoder over a YOLO26 ReID model.

    Args:
        cfg: ``EmbedderCfg`` — model path/name, crop size, fp16, part sets,
            ``max_batch``.
        device: ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` / ``"mps"``. Requesting
            CUDA without a CUDA build falls back to CPU instead of raising.
        force_fallback: Skip the real model entirely (used by tests and by
            callers that want a dependency-free run).
        warmup: Run one probe forward at construction so ``dim``, the real
            input size and the true mode are known immediately. A probe failure
            demotes the embedder to the fallback path with a recorded reason.

    Attributes:
        mode: ``"onnx"`` | ``"pt"`` | ``"fallback"``.
        dim: Embedding dimensionality (None until known).
        fallback_reason: Why the real model was not used ("" when it was).
    """

    def __init__(
        self,
        cfg: EmbedderCfg,
        device: str = "cuda",
        force_fallback: bool = False,
        warmup: bool = True,
    ) -> None:
        self.cfg = cfg
        #: The literal ``device`` argument, before any downgrade. Kept separately
        #: because ``requested_device`` is the *resolved* one: without it, a CUDA
        #: request that ``_resolve_device`` downgrades to CPU would be
        #: indistinguishable from a deliberate CPU request and the silent
        #: fallback would go unflagged.
        self.raw_device: str = str(device or "cpu")
        #: What the caller asked for, and what actually runs (they can differ:
        #: AutoBackend silently degrades an unusable CUDA EP to the CPU one).
        self.requested_device: str = self._resolve_device(device)
        self.device: str = self.requested_device
        #: Real execution provider, e.g. "CPUExecutionProvider" / "CUDAExecutionProvider".
        self.provider: str | None = None
        #: Human-readable explanation when the runtime differs from the request.
        self.device_note: str = ""
        if self.raw_device != self.requested_device and self.raw_device.startswith("cuda"):
            self.device_note = (
                f"{self.raw_device} requested but no usable CUDA execution provider is available "
                "(onnxruntime CUDA EP and/or torch CUDA missing); running on cpu"
            )
        #: True when CUDA was asked for and the real work is not on CUDA.
        self.gpu_fallback: bool = False
        self.mode: str = "fallback"
        self.dim: int | None = None
        self.fallback_reason: str = ""
        self.model_name: str = str(getattr(cfg, "model", "") or "")
        self.input_size: tuple[int, int] | None = None  # (H, W) as the model wants
        self.batch_size: int | None = None
        self.max_batch: int = max(1, int(getattr(cfg, "max_batch", 128) or 128))
        #: Adaptive forward chunk ceiling. Starts at ``max_batch`` and is only
        #: ever reduced, by an allocation failure (see ``_note_shrink``). It is
        #: deliberately persisted across calls: the point is not to re-pay the
        #: same OOM on every frame of a parallel grid.
        self._step: int = self.max_batch
        #: Size of the most recent successful forward (the batch actually used).
        self.last_batch_size: int = 0
        #: Cumulative allocation-failure retries, and the last failure's text.
        self.oom_retries: int = 0
        self.last_oom_reason: str = ""
        #: Wall-clock ms of the last forward, and how many forwards ran.
        self.last_latency_ms: float = 0.0
        self.forward_calls: int = 0

        self._reid: Any = None
        self._torch: Any = None
        #: E067 -- direct onnxruntime handle (see ``_wire_direct_session``).
        #: ``None`` means the forward goes through ultralytics' ``AutoBackend``.
        self._ort: Any = None
        self._ort_input: str | None = None
        self._ort_outputs: list[str] | None = None
        #: Which forward path actually runs: "session.run" or "autobackend".
        self.forward_path: str = "autobackend"
        #: Why the direct path was not wired (empty when it was).
        self.direct_session_note: str = "not attempted"
        #: Declared ONNX input type, e.g. "tensor(float)"; decides ``fp16``.
        self.input_dtype: str | None = None

        if force_fallback:
            self.fallback_reason = "force_fallback=True"
        else:
            self._try_load_real(device)

        if self.mode == "fallback":
            self.dim = FALLBACK_DIM
            self.forward_path = "fallback"
            self.direct_session_note = "non-neural fallback: no onnx session"
        elif warmup:
            self._warmup()

        self._flag_gpu_fallback()

    @property
    def _wants_cuda(self) -> bool:
        """True when the caller asked for CUDA, before or after resolution."""
        return self.raw_device.startswith("cuda") or self.requested_device.startswith("cuda")

    @property
    def effective_max_batch(self) -> int:
        """Chunk ceiling actually in force, after any allocation-failure shrink."""
        return int(self._step)

    @property
    def on_cuda(self) -> bool:
        """True only when the real forward demonstrably runs on a CUDA EP.

        Not the same question as ``not gpu_fallback``: an encoder built for CPU
        is neither on CUDA nor fallen back from it.
        """
        if self.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            return True
        return self.provider == "torch" and str(self.device).startswith("cuda")

    @property
    def fp16_effective(self) -> bool:
        """True only when the forward really is half precision.

        ``EmbedderCfg.fp16`` is a *request*. ultralytics resolves
        ``AutoBackend.fp16`` from the graph's own input type, and the shipped
        ReID ONNX declares ``tensor(float)`` -- so the request is a no-op there.
        Repeating the request as if it were the truth claims a precision that
        never ran (R_perf defect (b)).
        """
        backend = getattr(self._reid, "model", None) if self._reid is not None else None
        resolved = getattr(backend, "fp16", None)
        if resolved is not None:
            return bool(resolved)
        return "float16" in (self.input_dtype or "")

    @property
    def fp16_note(self) -> str:
        """One honest sentence about fp16 for logs and result files."""
        if self.fp16_effective:
            return "fp16 active: the model's input is float16"
        if bool(getattr(self.cfg, "fp16", False)):
            dtype = self.input_dtype or "float32"
            return (
                f"EmbedderCfg.fp16=True is a no-op: the model input is {dtype}, "
                "so the forward runs fp32 (this is not a speedup)"
            )
        return ""

    def _flag_gpu_fallback(self) -> None:
        """Loudly record a CUDA request that is not actually running on CUDA.

        A silent downgrade to the CPU execution provider costs 4-8x and turns a
        GPU experiment into an unlabelled CPU one, which is exactly the class of
        bug that ruins a benchmark. Three separate ways to land on CPU are
        covered: the *resolved* device already being cpu (``_resolve_device``
        found no usable CUDA EP), the session reporting a non-CUDA provider, and
        the torch backend reporting a cpu device on the PT path.
        """
        if not self._wants_cuda or self.on_cuda:
            return
        self.gpu_fallback = True
        if self.mode == "fallback":
            why = f"the encoder fell back to the non-neural descriptor ({self.fallback_reason})"
        elif self.provider == "torch":
            why = f"the torch backend reports device {self.device!r}"
        else:
            why = f"the session actually runs on {self.provider or 'an unknown provider'}"
        warnings.warn(
            f"Embedder: device={self.raw_device!r} was requested but {why}; "
            "appearance features are NOT coming from the GPU",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Downgrade a CUDA request to CPU when no *usable* CUDA path exists.

        ``torch.cuda.is_available()`` is not the right question on its own:
        onnxruntime's CUDA execution provider is a lazily-loaded plugin DLL that
        also needs torch's bundled CUDA/cuDNN runtime. Asking "can ORT actually
        run on the GPU" only answers correctly *after* the shim has put those
        DLLs on the search path, so the shim runs first and the provider list
        -- not the torch build -- decides.
        """
        dev = str(device or "cpu")
        if not dev.startswith("cuda"):
            return dev
        # ultralytics forwards ``device.index`` as the CUDA provider's
        # ``device_id`` option. A bare "cuda" has ``index is None``, so the
        # provider option arrives as ``{'device_id': None}`` and onnxruntime
        # rejects the whole provider with "Failed to parse provider option"
        # before quietly running on CPU. An explicit index avoids that.
        if dev in ("cuda", "cuda:"):
            dev = "cuda:0"
        ensure_ort_cuda()
        if "CUDAExecutionProvider" not in ort_providers("cuda"):
            return "cpu"
        try:
            import torch  # noqa: PLC0415 - optional heavy dependency

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
        return dev

    def _try_load_real(self, device: str) -> None:
        """Attempt to build the ultralytics ReID encoder; record the failure."""
        if self.model_name == "auto":
            self.fallback_reason = (
                "cfg.model='auto' means detector-backbone features and needs the detector; "
                "Embedder requires a standalone ReID model (e.g. 'yolo26n-reid.onnx')"
            )
            return
        try:
            import torch  # noqa: PLC0415 - optional heavy dependency

            self._torch = torch
            # Must run before the InferenceSession is built: it puts torch's
            # bundled CUDA 12 / cuDNN 9 DLLs on the loader path, without which
            # ORT's CUDA provider plugin fails to load *silently* and the
            # encoder quietly runs on CPU. Idempotent and never raises.
            ensure_ort_cuda()
            from ultralytics.trackers.utils.reid import ReID  # noqa: PLC0415

            self._reid = ReID(
                self.model_name,
                imgsz=int(getattr(self.cfg, "imgsz", 224) or 224),
                device=self.device,
                fp16=bool(getattr(self.cfg, "fp16", False)),
            )
            self.mode = "pt" if getattr(self._reid, "is_pt", False) else "onnx"
            self.batch_size = getattr(self._reid, "batch_size", None)
            size = getattr(self._reid, "imgsz", None)
            if isinstance(size, (tuple, list)) and len(size) == 2:
                self.input_size = (int(size[0]), int(size[1]))
            self._detect_provider()
            if self.provider == "CPUExecutionProvider" and self.requested_device.startswith("cuda"):
                self._retry_gpu_session()
            self._wire_direct_session()
        except Exception as exc:  # pragma: no cover - depends on the environment
            self._reid = None
            self.mode = "fallback"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self._drop_direct_session()

    def _detect_provider(self) -> None:
        """Record where the real forward actually runs, and never lie about it.

        ``AutoBackend`` silently falls back to the CPU execution provider when
        the CUDA one cannot load (e.g. an onnxruntime built against CUDA 13 on a
        CUDA 12.6 machine). Reporting ``device="cuda:0"`` in that case would
        poison every benchmark built on ``info``, so the *actual* provider wins.
        """
        reid = self._reid
        if reid is None:
            return
        backend = getattr(reid, "model", None)
        session = getattr(reid, "session", None) or getattr(backend, "session", None)
        if session is not None:
            try:
                providers = list(session.get_providers())
            except Exception:  # pragma: no cover - exotic backends
                providers = []
            if providers:
                self.provider = str(providers[0])
                if self.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
                    self.device = "cuda:0"
                elif self.provider == "CPUExecutionProvider":
                    self.device = "cpu"
                    if self.requested_device.startswith("cuda"):
                        self.device_note = (
                            f"{self.requested_device} requested but the runtime uses "
                            "CPUExecutionProvider (CUDA execution provider unavailable)"
                        )
                return
        dev = getattr(backend, "device", None)
        if dev is not None:
            self.provider = "torch"
            self.device = str(dev)

    def _retry_gpu_session(self) -> None:
        """Rebuild the ONNX session on the CUDA provider if it landed on CPU.

        ``ultralytics`` builds its provider list itself and forwards
        ``{"device_id": device.index}``. That normally works (``_resolve_device``
        makes sure of it), but it is a silent failure mode when it does not:
        ORT rejects the option, logs, and runs on CPU. The shim's provider list
        carries no options at all, so it cannot be rejected that way -- so the
        session is rebuilt with it and swapped in, preserving ultralytics' own
        session options. Any failure here is recorded, never raised, and the
        CPU session stays in place as the working fallback.
        """
        reid = self._reid
        backend = getattr(reid, "model", None)
        session = getattr(reid, "session", None) or getattr(backend, "session", None)
        if backend is None or session is None:
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415 - optional dependency

            opts = getattr(backend, "session_options", None)
            if not isinstance(opts, ort.SessionOptions):
                opts = ort.SessionOptions()
            new = ort.InferenceSession(
                str(self.model_name), sess_options=opts, providers=ort_providers(self.requested_device)
            )
        except Exception as exc:  # pragma: no cover - depends on the environment
            self.device_note = f"{self.device_note}; GPU session retry failed: {type(exc).__name__}: {exc}".lstrip("; ")
            return
        try:
            backend.session = new
            if getattr(reid, "session", None) is not None:
                reid.session = new
        except Exception:  # pragma: no cover - exotic backends
            return
        self.device_note = ""
        self._detect_provider()
        self._wire_direct_session()

    def _wire_direct_session(self) -> None:
        """Point the forward straight at onnxruntime, bypassing ``AutoBackend``.

        ``AutoBackend`` is a format dispatcher, and this export is a
        *static-batch* graph (``backend.batch == 1``) with a dynamic ``batch``
        axis: any chunk wider than one crop is split into that many single-image
        ``session.run`` calls -- so a 64-crop chunk pays 64 forwards and gets no
        batching at all -- and every output is then copied back through torch
        (``from_numpy`` -> ``.cpu().numpy()``). Handing the *same preprocessed
        tensor* to the *same session* keeps one true batched forward; measured
        bit-identical (``max_abs_diff == 0.0``) and 2.06x on the forward at 64
        crops (235.0 -> 114.1 ms, ``results/perf_impl/forward_parity.json``).

        Nothing about the call changes: same session object (so the provider and
        session options survive ``_retry_gpu_session``), same
        ``_crops_to_tensor`` preprocessing, same chunking and OOM halving. The
        CPU fallback and the non-neural descriptor path never reach this code.
        """
        self._ort = None
        self._ort_input = None
        self._ort_outputs = None
        reid = self._reid
        backend = getattr(reid, "model", None)
        session = getattr(reid, "session", None) or getattr(backend, "session", None)
        if session is None:
            self.forward_path = "autobackend"
            self.direct_session_note = "no onnxruntime session on the ReID model"
            return
        try:
            inputs = list(session.get_inputs())
            outputs = list(session.get_outputs())
        except Exception as exc:  # pragma: no cover - exotic backends
            self.forward_path = "autobackend"
            self.direct_session_note = f"{type(exc).__name__}: {exc}"
            return
        self.input_dtype = str(getattr(inputs[0], "type", "") or "") if inputs else None
        if len(inputs) != 1:
            self.forward_path = "autobackend"
            self.direct_session_note = f"{len(inputs)} graph inputs; only single-input graphs are wired directly"
            return
        if getattr(reid, "is_pt", False) or str(getattr(backend, "format", "")) != "onnx":
            self.forward_path = "autobackend"
            self.direct_session_note = f"backend format {getattr(backend, 'format', None)!r} is not onnx"
            return
        if "float16" in (self.input_dtype or ""):
            # An fp16 graph needs the fp16 cast + io-binding path. Leave that to
            # AutoBackend instead of re-implementing precision handling here.
            self.forward_path = "autobackend"
            self.direct_session_note = "fp16 graph: AutoBackend keeps the half-precision cast"
            return
        self._ort = session
        self._ort_input = str(inputs[0].name)
        self._ort_outputs = [str(o.name) for o in outputs]
        self.forward_path = "session.run"
        self.direct_session_note = ""

    def _effective_forward_path(self) -> str:
        """The path a crop forward would take *right now* (E067).

        ``self.forward_path`` records what ``_wire_direct_session`` decided; the
        handles can change afterwards (a GPU retry, a dropped session, a torch
        backend). ``info`` must describe reality, so it asks here instead: the
        only way ``session.run`` is claimed is if the direct handle is live and
        ``_forward_chunk`` would really use it.
        """
        reid = self._reid
        if reid is None or not self.is_real:
            return self.forward_path
        if getattr(reid, "is_pt", False):
            return "autobackend"
        if self._ort is not None and self._ort_input is not None:
            return "session.run"
        return "autobackend"

    def _drop_direct_session(self) -> None:
        """Forget the direct session handle after the real encoder is gone."""
        self._ort = None
        self._ort_input = None
        self._ort_outputs = None
        self.forward_path = "fallback"

    def _warmup(self) -> None:
        """One probe forward: fixes ``dim``, validates the path, warms the GPU."""
        probe = np.full((256, 128, 3), 127, dtype=np.uint8)
        try:
            feats = self._encode_crops_real([probe])
        except Exception as exc:  # pragma: no cover - depends on the environment
            self._reid = None
            self.mode = "fallback"
            self.dim = FALLBACK_DIM
            self.fallback_reason = f"warmup failed: {type(exc).__name__}: {exc}"
            self._drop_direct_session()
            return
        f = feats[0] if feats else None
        if f is None:
            self._reid = None
            self.mode = "fallback"
            self.dim = FALLBACK_DIM
            self.fallback_reason = "warmup produced no embedding"
            self._drop_direct_session()
            return
        self.dim = int(f.size)

    # -------------------------------------------------------------- properties
    @property
    def is_real(self) -> bool:
        """True when embeddings come from the neural ReID network."""
        return self.mode != "fallback" and self._reid is not None

    @property
    def info(self) -> dict[str, Any]:
        """Compact description for logs and result files."""
        return {
            "mode": self.mode,
            "model": self.model_name,
            "device": self.device,
            "requested_device": self.requested_device,
            "raw_device": self.raw_device,
            "provider": self.provider,
            "device_note": self.device_note,
            "gpu_fallback": self.gpu_fallback,
            "on_cuda": self.on_cuda,
            "dim": self.dim,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "max_batch": self.max_batch,
            "effective_max_batch": self.effective_max_batch,
            "last_batch_size": self.last_batch_size,
            "oom_retries": self.oom_retries,
            # E067: report the precision that actually runs, not the one that was
            # requested -- `EmbedderCfg.fp16=True` is a no-op against the shipped
            # fp32 ReID ONNX (session input type `tensor(float)`).
            "fp16": self.fp16_effective,
            "fp16_requested": bool(getattr(self.cfg, "fp16", False)),
            "fp16_note": self.fp16_note,
            "input_dtype": self.input_dtype,
            # E067: which forward path ran (AutoBackend vs the session directly).
            # Asked of the live handles, not of the flag set at wiring time, so a
            # dropped/rewired session can never be reported as "session.run"
            # while the crops actually go through AutoBackend.
            "forward_path": self._effective_forward_path(),
            "direct_session_note": self.direct_session_note,
            "is_real": self.is_real,
            "fallback_reason": self.fallback_reason,
        }

    # ------------------------------------------------------------ encode: boxes
    def encode(
        self,
        img_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        parts: Sequence[str] = ("full",),
    ) -> list[dict[str, np.ndarray]]:
        """Encode every requested part of every box in one frame.

        Args:
            img_bgr: ``(H, W, 3)`` BGR frame.
            boxes_xyxy: ``(N, 4)`` absolute-pixel person boxes.
            parts: Part keys to compute (see ``PART_KEYS``). Cropping a part
                that falls outside the frame is handled by zero-padding, so a
                part box is never silently squeezed.

        Returns:
            ``list`` of N dicts; ``result[i][part]`` is the L2-normalized
            ``float32`` embedding for box ``i``. A part is simply *absent* from
            the dict when its crop was degenerate — callers must use
            ``.get(part)``, never index blindly.
        """
        parts = tuple(parts)
        bad = [p for p in parts if p not in PART_KEYS]
        if bad:
            raise KeyError(f"unknown part key(s) {bad}; valid keys: {list(PART_KEYS)}")

        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes.size == 0:
            return []
        boxes = boxes.reshape(-1, 4)
        out: list[dict[str, np.ndarray]] = [{} for _ in range(boxes.shape[0])]
        if img_bgr is None or not parts:
            return out

        crops: list[np.ndarray] = []
        index: list[tuple[int, str]] = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            if not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            for part, pbox in part_boxes(box, parts).items():
                c = crop_box(img_bgr, pbox, out_size=None)
                if c is None:
                    continue
                crops.append(c)
                index.append((i, part))

        feats = self.encode_crops(crops)
        for (i, part), f in zip(index, feats):
            if f is not None:
                out[i][part] = f
        return out

    # ------------------------------------------------------------ encode: crops
    def encode_crops(self, crops: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        """Batch-encode pre-cropped images (the target's part crops).

        Args:
            crops: BGR crops of any size; ``None``/empty entries are tolerated.

        Returns:
            One L2-normalized ``(dim,)`` float32 embedding per input, or None
            for an entry that could not be encoded.
        """
        crops = list(crops)
        if not crops:
            return []
        prepared: list[np.ndarray] = []
        slots: list[int] = []
        for i, c in enumerate(crops):
            img = _as_bgr_uint8(c)
            if img is None:
                continue
            prepared.append(img)
            slots.append(i)
        out: list[np.ndarray | None] = [None] * len(crops)
        if not prepared:
            return out
        if self.is_real:
            feats = self._encode_crops_real(prepared)
        else:
            feats = [fallback_descriptor(c) for c in prepared]
        for slot, f in zip(slots, feats):
            f = _unit(f)
            if f is None:
                continue
            if self.mode != "fallback" and self.dim is None:
                self.dim = int(f.size)
            if self.dim is not None and f.size != self.dim:
                continue
            out[slot] = f
        return out

    # ------------------------------------------------------------ real forward
    #: Substrings that mark a *resource* failure rather than a logic bug. A
    #: genuine shape/dtype error must still raise: retrying it 64->32->...->1
    #: would only waste six forwards and hide the real message.
    _ALLOC_MARKERS: tuple[str, ...] = (
        "bfcarena",
        "allocaterawinternal",
        "failed to allocate",
        "out of memory",
        "outofmemory",
        "cudamemoryallocation",
        "cuda_error_out_of_memory",
        "cudnn_status_alloc_failed",
        "bad_alloc",
        "memory allocation",
    )

    def _is_alloc_error(self, exc: BaseException) -> bool:
        """True when ``exc`` looks like a GPU/host allocation failure.

        Deliberately broad on one point: any ``[ONNXRuntimeError]`` out of
        ``session.run`` counts. Inside a 4-worker grid sharing an 8 GB card the
        arena fails in ways that do not spell "out of memory" (the reported one
        was ``BFCArena::AllocateRawInternal`` under a Conv node), and the cost of
        retrying a non-allocation runtime error at a smaller batch is one wasted
        forward versus losing a 120-run grid.
        """
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(m in text for m in self._ALLOC_MARKERS):
            return True
        return "onnxruntimeerror" in text

    def _note_shrink(self, failed: int, new: int, exc: BaseException) -> None:
        """Record a batch shrink loudly, then try to hand memory back.

        Silent degradation is worse than a loud one here: the caller is choosing
        ``max_batch`` against a speculative VRAM budget, so it needs to know the
        number it asked for was not the number that ran.
        """
        self.oom_retries += 1
        self.last_oom_reason = f"{type(exc).__name__}: {exc}"[:400]
        self._step = min(self._step, max(1, int(new)))
        self._empty_cuda_cache()
        warnings.warn(
            f"Embedder: ONNX forward failed at batch {failed} "
            f"({self.last_oom_reason}); halving to {new} and retrying "
            f"(effective max_batch {self._step}/{self.max_batch})",
            RuntimeWarning,
            stacklevel=3,
        )

    def _empty_cuda_cache(self) -> None:
        """Best-effort ``torch.cuda.empty_cache()`` after an allocation failure.

        Only frees torch's own cached blocks -- onnxruntime's arena is its own --
        but that is often enough to make room for the next, smaller forward.
        Never raises: this runs on an error path.
        """
        torch = self._torch
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - depends on the environment
            pass

    @staticmethod
    def _row_array(x: Any) -> np.ndarray:
        """One embedding as a private float32 copy.

        The copy is load-bearing: with IO binding / a reused output buffer the
        returned array can alias memory that the *next* forward overwrites.
        """
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.array(x, dtype=np.float32, copy=True)

    @classmethod
    def _to_rows(cls, out: Any) -> list[np.ndarray]:
        """Normalize a forward result (tensor, ndarray or list) to per-crop rows."""
        if isinstance(out, (list, tuple)):
            return [cls._row_array(r) for r in out]
        arr = cls._row_array(out)
        if arr.ndim <= 1:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    def _chunk_step(self, n: int) -> int:
        """The chunk size to start from for ``n`` items, after any shrink."""
        step = self._step
        if self.batch_size:  # a static ONNX batch must be fed exactly that many
            step = min(step, int(self.batch_size))
        return max(1, min(int(step), max(1, int(n))))

    def _chunks(self, n: int) -> list[slice]:
        """Split ``n`` items into forward-pass chunks honoring model limits."""
        step = self._chunk_step(n)
        return [slice(s, min(s + step, n)) for s in range(0, n, step)]

    def _pad_chunk(self, chunk: Any, fixed: int) -> Any:
        """Pad a short final chunk up to a static ONNX batch with its last row."""
        torch = self._torch
        if torch is None or chunk.shape[0] >= fixed:
            return chunk
        return torch.cat([chunk, chunk[-1:].expand(fixed - chunk.shape[0], *chunk.shape[1:])], 0)

    def _forward_chunk(self, reid: Any, chunk: Any) -> Any:
        """One forward of one already-preprocessed chunk.

        Routes through ``session.run`` when ``_wire_direct_session`` could build
        the handle, else through ``reid.model`` (ultralytics' ``AutoBackend``).
        Both paths receive the *same* tensor, so the numbers are identical by
        construction; only the dispatch layer differs.
        """
        if self._ort is None or self._ort_input is None:
            return reid.model(chunk)
        arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        outs = self._ort.run(self._ort_outputs, {self._ort_input: np.ascontiguousarray(arr)})
        # `InferenceSession.run` always hands back a *list of outputs*, while
        # `AutoBackend.__call__` hands back the embedding tensor itself. Unwrap
        # here so both paths give `_to_rows` the same object; otherwise the
        # outer list is read as "one row per output" and every crop collapses
        # into a single (N, dim) row.
        if isinstance(outs, (list, tuple)):
            return outs[0] if outs else np.zeros((0, 0), np.float32)
        return outs

    def _run_onnx(self, reid: Any, batch: Any) -> list[np.ndarray]:
        """Forward ``batch`` in chunks, halving the chunk on allocation failure.

        Returns one row per input crop because the chunk size may shrink
        mid-call. Shrinking is remembered in ``_step`` so the next frame starts
        at the size that actually fit instead of rediscovering the OOM. A static
        (fixed-batch) model cannot be shrunk -- it must be fed exactly
        ``batch_size`` rows -- so there the original error is raised.
        """
        n = int(batch.shape[0])
        fixed = int(reid.batch_size) if getattr(reid, "batch_size", None) else 0
        rows: list[np.ndarray] = []
        step = self._chunk_step(n)
        pos = 0
        while pos < n:
            size = min(step, n - pos)
            chunk = batch[pos : pos + size]
            padded = bool(fixed and size < fixed)
            if padded:
                chunk = self._pad_chunk(chunk, fixed)
            try:
                out = self._forward_chunk(reid, chunk)
            except Exception as exc:
                if size <= 1 or padded or not self._is_alloc_error(exc):
                    raise
                step = max(1, size // 2)
                self._note_shrink(size, step, exc)
                continue
            rows.extend(self._to_rows(out)[:size])
            self.last_batch_size = size
            pos += size
        return rows

    def _run_pt(self, reid: Any, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Torch path: same adaptive halving, sliced over the crop list."""
        rows: list[np.ndarray] = []
        step = self._chunk_step(len(crops))
        pos = 0
        while pos < len(crops):
            size = min(step, len(crops) - pos)
            batch = crops[pos : pos + size]
            try:
                raw = reid.model.predictor(batch)
            except Exception as exc:
                if size <= 1 or not self._is_alloc_error(exc):
                    raise
                step = max(1, size // 2)
                self._note_shrink(size, step, exc)
                continue
            if not isinstance(raw, (list, tuple)) and len(raw) != len(batch):
                first = raw[0] if len(raw) else None
                if getattr(first, "shape", (0,))[0] == len(batch):
                    raw = raw[0]
            rows.extend(self._to_rows(raw)[:size])
            self.last_batch_size = size
            pos += size
        return rows

    def _encode_crops_real(self, crops: list[np.ndarray]) -> list[np.ndarray | None]:
        """Run the ultralytics ReID model over all crops, chunked adaptively.

        Never lets a *resource* failure become fatal: an allocation error halves
        the forward batch and retries (down to 1). A part whose crop is
        degenerate is decided before this point, so a retry can never turn a
        legitimately-absent part into an error -- see ``encode_crops``.
        """
        torch = self._torch
        reid = self._reid
        if torch is None or reid is None:
            return [None] * len(crops)
        import time  # noqa: PLC0415 - local, keeps import surface small

        t0 = time.perf_counter()
        self.last_batch_size = 0
        with torch.no_grad():
            if getattr(reid, "is_pt", False):
                rows = self._run_pt(reid, crops)
            else:
                batch = reid._crops_to_tensor(crops)  # noqa: SLF001 - ultralytics' own preprocessing
                rows = self._run_onnx(reid, batch)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.forward_calls += 1
        rows = rows[: len(crops)]
        if len(rows) < len(crops):
            rows += [None] * (len(crops) - len(rows))
        return rows

    # ---------------------------------------------------------------- fallback
    def encode_fallback(self, crops: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        """Force the handcrafted descriptor (ignores the loaded model)."""
        out: list[np.ndarray | None] = []
        for c in crops:
            img = _as_bgr_uint8(c)
            out.append(None if img is None else fallback_descriptor(img))
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"Embedder(mode={self.mode!r}, model={self.model_name!r}, device={self.device!r}, "
            f"dim={self.dim}, input_size={self.input_size}, is_real={self.is_real})"
        )
