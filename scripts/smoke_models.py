"""Smoke-test the external model dependencies (Lead-owned).

Answers the only questions that matter before the real experiments:
  1. Do the YOLO26 detector weights actually download in this network?
  2. Does the yolo26*-reid ONNX encoder download and run on the GPU?
  3. What do they cost per frame, and can both fit in 8 GB together?

Run:  python scripts/smoke_models.py
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402


def main() -> int:
    out: dict = {"steps": []}

    def step(name: str, fn):
        t0 = time.perf_counter()
        try:
            val = fn()
            out["steps"].append({"name": name, "ok": True, "sec": round(time.perf_counter() - t0, 2), "value": val})
            print(f"[ok]   {name}  ({time.perf_counter() - t0:.2f}s)  {val}")
            return val
        except Exception as exc:  # noqa: BLE001 - smoke test reports everything
            import traceback

            out["steps"].append({
                "name": name, "ok": False, "sec": round(time.perf_counter() - t0, 2),
                "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:],
            })
            print(f"[FAIL] {name}  {type(exc).__name__}: {exc}")
            return None

    # ---------------------------------------------------------------- detector
    from tgtrack.config import DetectorCfg, EmbedderCfg
    from tgtrack.detector import PersonDetector

    frame = np.full((720, 1280, 3), 96, dtype=np.uint8)
    # A crude person-ish blob so we can tell "ran fine, found nobody" apart from
    # "the model never loaded".
    frame[200:700, 500:640] = 190

    det = PersonDetector(DetectorCfg())
    step("detector.load", lambda: det.model_info())
    step("detector.warmup", lambda: det.warmup())
    dets = step("detector.detect", lambda: [d.xyxy.tolist() for d in det.detect(frame)])
    step("detector.benchmark", lambda: det.benchmark(frame, n=30))

    # ---------------------------------------------------------------- embedder
    from tgtrack.appearance import Embedder

    emb = Embedder(EmbedderCfg(), device="cuda", force_fallback=False, warmup=True)
    step("embedder.backend", lambda: getattr(emb, "backend", None) or repr(emb.__class__))
    boxes = np.array([[500, 200, 640, 700], [100, 100, 200, 400]], dtype=np.float32)

    def _encode():
        enc = emb.encode(frame, boxes, parts=("full", "upper", "lower", "head"))
        return [{p: (None if v is None else list(np.asarray(v).shape)) for p, v in d.items()} for d in enc]

    feats = step("embedder.encode", _encode)
    if feats:
        out["features"] = feats
        out["embedder_mode"] = getattr(emb, "mode", None)
        out["embedder_dim"] = getattr(emb, "dim", None)

    # ---------------------------------------------------------------- together
    def both():
        import torch

        for _ in range(10):
            det.detect(frame)
            emb.encode(frame, boxes, parts=("full",))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return {
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            }
        return {"cuda": False}

    step("detector+embedder.10_frames", both)

    dest = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "smoke_models.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    bad = [s["name"] for s in out["steps"] if not s["ok"]]
    print("\n" + ("ALL OK" if not bad else "FAILED: " + ", ".join(bad)))
    print(f"wrote {dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
