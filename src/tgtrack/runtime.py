"""Runtime shims that make the shared Windows environment behave consistently.

The problem this solves
-----------------------
``onnxruntime-gpu``'s CUDA execution provider is not linked into the core
library: it is a *plugin* DLL (``onnxruntime_providers_cuda.dll``) that ORT
loads lazily the first time it is asked for.  That plugin links against the
CUDA 12 runtime -- ``cublasLt64_12.dll``, ``cublas64_12.dll``, ``cudart64_12.dll``
and the cuDNN 9 family.

Those DLLs are not installed system-wide here.  They ship *inside the torch
wheel*, in ``<site-packages>/torch/lib``, and that directory is not on ``PATH``
by default.  The consequence is nasty because it is silent: ORT fails to load
the plugin, drops ``CUDAExecutionProvider`` from its available provider list,
and quietly runs on CPU.  Nothing errors; the encoder is just ~8x slower.

Measured on this machine for ``yolo26n-reid.onnx`` at 224x224:

===========  ==========  ===========
batch        CPU (ms)    CUDA (ms)
===========  ==========  ===========
1            4.25        2.74
40           108.0       12.28
===========  ==========  ===========

i.e. 2.7 ms -> 0.31 ms per crop.  For the real pipeline shape (10 persons x 1
part + the target's 4 parts = 14 crops per frame) that is ~38 ms/frame versus
~4.3 ms/frame, which is the difference between ~18 FPS and ~45 FPS.

What this module does
---------------------
:func:`ensure_ort_cuda` prepends torch's ``lib`` directory to ``PATH`` and
registers it as a DLL directory.  Call it *before the first*
``onnxruntime.InferenceSession`` is constructed.

It is deliberately best-effort and never raises: on a non-Windows host, with no
torch installed, or with an ORT build that genuinely has no CUDA provider, the
call is a cheap no-op and the caller keeps working on CPU.  A silent fallback to
CPU is an acceptable *outcome*; a silent fallback that nobody can explain is not,
which is why :func:`ort_status` exists to report the truth on demand.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ensure_ort_cuda", "ort_status", "ort_providers", "torch_lib_dir"]

# Cache the torch lib lookup: it costs a torch import the first time.
_TORCH_LIB: str | None = None
# os.add_dll_directory returns a handle that must stay referenced, otherwise the
# directory is removed from the search path again.
_DLL_HANDLES: list[object] = []
_APPLIED = False


def torch_lib_dir() -> str | None:
    """Return ``<site-packages>/torch/lib`` if torch is importable, else None."""
    global _TORCH_LIB
    if _TORCH_LIB is None:
        try:
            import torch  # noqa: PLC0415  (deliberately lazy: heavy import)

            cand = Path(torch.__file__).resolve().parent / "lib"
        except Exception:
            cand = None
        _TORCH_LIB = str(cand) if cand is not None and cand.is_dir() else ""
    return _TORCH_LIB or None


def ensure_ort_cuda(verbose: bool = False) -> dict:
    """Make ORT's CUDA execution provider loadable. Safe to call repeatedly.

    Returns a small dict describing what was done, which is handy in smoke
    tests and reports.
    """
    global _APPLIED

    info: dict = {
        "torch_lib": None,
        "path_added": False,
        "add_dll_directory": False,
        "applied": False,
    }

    lib = torch_lib_dir()
    info["torch_lib"] = lib

    if lib:
        entries = os.environ.get("PATH", "").split(os.pathsep)
        if lib not in entries:
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
            info["path_added"] = True
        # Belt and braces: AddDllDirectory covers loads that pass an explicit
        # search flag, PATH covers the plain LoadLibrary path the plugin uses.
        if hasattr(os, "add_dll_directory") and not _APPLIED:
            try:
                _DLL_HANDLES.append(os.add_dll_directory(lib))
                info["add_dll_directory"] = True
            except OSError:
                pass

    _APPLIED = True
    info["applied"] = True

    if verbose:
        status = ort_status()
        info["available"] = status["available"]
        info["preferred"] = status["preferred"]

    return info


def ort_status() -> dict:
    """Report what onnxruntime would actually use right now."""
    import onnxruntime as ort  # noqa: PLC0415

    return {
        "version": ort.__version__,
        "available": list(ort.get_available_providers()),
        "preferred": ort_providers(),
    }


def ort_providers(device: str = "auto", use_tensorrt: bool = False) -> list[str]:
    """Provider preference list for ``InferenceSession(..., providers=...)``.

    Always ends with ``CPUExecutionProvider`` so ORT has a guaranteed fallback
    and the session cannot fail merely because the GPU is busy.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]

    if device == "cpu":
        return ["CPUExecutionProvider"]

    providers: list[str] = []
    # TensorRT can beat CUDA on paper, but it builds engines at load time
    # (minutes) and is fragile across driver updates, so it is opt-in.
    if use_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
