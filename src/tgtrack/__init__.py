"""tgtrack — DJI-ActiveTrack-style human target tracking.

Detect all persons, track only the clicked target with long-term + short-term
ReID, robust to partial visibility, out-of-frame gaps, fast and nonlinear motion.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config
from .data import Sequence, discover_sequences, load_mot_sequence, make_target_gt, simulate_click
from .evaluate import (
    GtBox,
    PredBox,
    combine_mot,
    combine_target,
    evaluate_mot,
    evaluate_target,
)
from .runtime import ensure_ort_cuda, ort_providers, ort_status, torch_lib_dir
from .types import (
    PART_KEYS,
    PART_X_RANGES,
    PART_Y_RANGES,
    Detection,
    FrameMeta,
    TargetEvent,
    TargetStatus,
    clip_xyxy,
    cosine_distance_matrix,
    iou_matrix,
    l2norm,
)

__all__ = [
    "PART_KEYS",
    "PART_X_RANGES",
    "PART_Y_RANGES",
    "Config",
    "Detection",
    "Embedder",
    "FrameMeta",
    "GtBox",
    "LongTermGallery",
    "MotionModel",
    "PredBox",
    "Sequence",
    "ShortTermMemory",
    "TargetEvent",
    "TargetState",
    "TargetStatus",
    "TargetTracker",
    "__version__",
    "clip_xyxy",
    "combine_mot",
    "combine_target",
    "cosine_distance_matrix",
    "discover_sequences",
    "ensure_ort_cuda",
    "evaluate_mot",
    "evaluate_target",
    "iou_matrix",
    "l2norm",
    "load_config",
    "load_mot_sequence",
    "make_target_gt",
    "ort_providers",
    "ort_status",
    "simulate_click",
    "torch_lib_dir",
]

# The heavy components are exposed lazily. Importing them eagerly would drag in
# ultralytics/torch/onnxruntime and turn a 0.27 s `import tgtrack` into a multi
# second one -- painful for every test and smoke script -- while `Config` and the
# metric/loader helpers above stay free to import. Access still works normally:
# `tgtrack.TargetTracker` resolves on first touch (PEP 562).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Embedder": ("appearance", "Embedder"),
    "LongTermGallery": ("memory", "LongTermGallery"),
    "MotionModel": ("motion", "MotionModel"),
    "ShortTermMemory": ("memory", "ShortTermMemory"),
    "TargetState": ("target_tracker", "TargetState"),
    "TargetTracker": ("target_tracker", "TargetTracker"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = target
    return getattr(importlib.import_module(f".{module_name}", __name__), attr)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
