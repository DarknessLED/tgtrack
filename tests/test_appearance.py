"""Tests for :mod:`tgtrack.parts` and :mod:`tgtrack.appearance`.

Part geometry is checked against hand-computed numbers from ``PART_Y_RANGES`` /
``PART_X_RANGES``; the encoder is checked on the **fallback** path (always
available) for shape, normalization, determinism and position invariance. The
real ONNX network is exercised only when ``TGTRACK_TEST_REAL_REID=1`` is set, so
the suite never touches the network by accident.
"""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np
import pytest

from tgtrack.appearance import FALLBACK_DIM, Embedder, fallback_descriptor
from tgtrack.config import EmbedderCfg
from tgtrack.parts import (
    crop,
    crop_quality,
    estimate_partial_state,
    part_boxes,
    visible_ratio,
)
from tgtrack.types import PART_KEYS

ALL_PARTS = ("full", "upper", "lower", "head", "torso")


def rng_image(seed: int = 0, h: int = 200, w: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def fallback_embedder() -> Embedder:
    emb = Embedder(EmbedderCfg(), device="cpu", force_fallback=True)
    assert emb.mode == "fallback"
    assert emb.is_real is False
    return emb


# --------------------------------------------------------------------------- parts
def test_part_boxes_math_on_a_known_box():
    boxes = part_boxes(np.array([0, 0, 100, 200], dtype=np.float32), ALL_PARTS)
    assert set(boxes) == set(ALL_PARTS)
    assert np.allclose(boxes["full"], [0, 0, 100, 200])
    assert np.allclose(boxes["upper"], [0, 0, 100, 120])  # y 0.00 -> 0.60
    assert np.allclose(boxes["lower"], [0, 90, 100, 200])  # y 0.45 -> 1.00
    assert np.allclose(boxes["head"], [15, 0, 85, 56])  # x 0.15-0.85, y 0.00-0.28
    assert np.allclose(boxes["torso"], [5, 30, 95, 110])  # x 0.05-0.95, y 0.15-0.55
    for b in boxes.values():
        assert b.dtype == np.float32 and b.shape == (4,)


def test_part_boxes_offsets_scale_with_the_person_box():
    boxes = part_boxes(np.array([50, 400, 150, 1000], dtype=np.float32), ("head",))
    assert np.allclose(boxes["head"], [50 + 0.15 * 100, 400, 50 + 0.85 * 100, 400 + 0.28 * 600])


def test_part_boxes_clips_to_the_frame_when_size_is_given():
    boxes = part_boxes(np.array([-10, -10, 100, 100], dtype=np.float32), ("full",), frame_size=(60, 60))
    assert np.allclose(boxes["full"], [0, 0, 59, 59])


def test_part_boxes_rejects_unknown_and_empty_part_sets():
    with pytest.raises(KeyError):
        part_boxes(np.array([0, 0, 10, 20], dtype=np.float32), ("left_arm",))
    with pytest.raises(ValueError):
        part_boxes(np.array([0, 0, 10, 20], dtype=np.float32), ())
    with pytest.raises(ValueError):
        part_boxes(np.array([np.nan, 0, 10, 20]), ("full",))


def test_crop_keeps_geometry_when_the_box_leaves_the_frame():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = 50
    img[10:20, 10:20] = 255
    c = crop(img, np.array([-10, -10, 30, 30], dtype=np.float32))
    assert c is not None
    assert c.shape == (40, 40, 3), "box extent must be preserved, not squeezed"
    assert np.all(c[0, 0] == 0)  # outside the frame -> zero padding
    assert np.all(c[5, 5] == 0)
    assert np.all(c[15, 15] == 50)  # img[0, 0]
    assert np.all(c[25, 25] == 255)  # img[10, 10]
    # No padding at all when the box is fully inside.
    inside = crop(img, np.array([10, 10, 30, 30], dtype=np.float32))
    assert inside is not None and inside.shape == (20, 20, 3)


def test_crop_out_size_is_width_height_and_pad_expands():
    img = rng_image(3)
    c = crop(img, np.array([20, 30, 80, 150], dtype=np.float32), out_size=(32, 64))
    assert c is not None and c.shape == (64, 32, 3)
    plain = crop(img, np.array([50, 50, 100, 150], dtype=np.float32))
    padded = crop(img, np.array([50, 50, 100, 150], dtype=np.float32), pad=0.2)
    assert plain is not None and padded is not None
    assert plain.shape == (100, 50, 3)
    assert padded.shape == (140, 70, 3)  # +20 % of w (50) and h (100) on every side


def test_crop_returns_none_for_degenerate_and_outside_boxes():
    img = rng_image(4)
    assert crop(img, np.array([10, 10, 10, 20], dtype=np.float32)) is None
    assert crop(img, np.array([10, 20, 30, 20], dtype=np.float32)) is None
    assert crop(img, np.array([500, 500, 600, 600], dtype=np.float32)) is None
    assert crop(img, np.array([np.nan, 0, 10, 20], dtype=np.float32)) is None
    assert crop(np.zeros((0, 0, 3), dtype=np.uint8), np.array([0, 0, 5, 5], dtype=np.float32)) is None


def test_visible_ratio_counts_the_inside_part():
    # `visible_ratio` is box-relative: the fraction of the *box* inside the frame.
    assert visible_ratio(np.array([10, 10, 40, 60], dtype=np.float32), 100, 100) == pytest.approx(1.0)
    assert visible_ratio(np.array([-50, 0, 50, 100], dtype=np.float32), 100, 100) == pytest.approx(0.5)
    assert visible_ratio(np.array([50, 0, 150, 100], dtype=np.float32), 100, 100) == pytest.approx(0.5)
    assert visible_ratio(np.array([200, 200, 300, 300], dtype=np.float32), 100, 100) == pytest.approx(0.0)
    assert visible_ratio(np.array([10, 10, 10, 10], dtype=np.float32), 100, 100) == 0.0
    assert visible_ratio(np.array([np.nan, 0, 10, 10], dtype=np.float32), 100, 100) == 0.0


def test_crop_quality_scores_area_and_sharpness():
    noisy = rng_image(7, h=100, w=100)
    q, sharp = crop_quality(noisy, np.array([0, 0, 100, 100], dtype=np.float32))
    assert sharp > 10.0
    assert 0.0 < q <= 1.0
    flat = np.zeros((100, 100, 3), dtype=np.uint8)
    q_flat, sharp_flat = crop_quality(flat, np.array([0, 0, 100, 100], dtype=np.float32))
    assert sharp_flat == pytest.approx(0.0, abs=1e-6)
    assert q_flat == 0.0
    # A small crop is worth less than a big one at equal sharpness.
    small = noisy[:20, :20]
    q_small, _ = crop_quality(small, np.array([0, 0, 20, 20], dtype=np.float32))
    assert q_small < q
    assert crop_quality(noisy, np.array([200, 200, 300, 300], dtype=np.float32)) == (0.0, 0.0)


def test_estimate_partial_state_fully_visible_box():
    st = estimate_partial_state(np.array([100, 100, 200, 370], dtype=np.float32), 640, 480)
    assert st["truncated"] is False
    assert st["sides"] == []
    assert st["kind"] == "none"
    assert st["est_visible_frac"] == pytest.approx(1.0)
    assert st["upper_only"] is False and st["lower_only"] is False


def test_estimate_partial_state_box_crossing_the_top_edge():
    st = estimate_partial_state(np.array([100, -30, 200, 240], dtype=np.float32), 640, 480)
    assert st["truncated"] is True
    assert st["sides"] == ["top"]
    assert st["kind"] == "top"
    assert st["lower_only"] is True
    assert st["upper_only"] is False
    assert st["est_visible_frac"] == pytest.approx(240 / 270, abs=1e-3)
    assert st["visible"][1] == pytest.approx(0.0)  # clipped to the frame


def test_estimate_partial_state_other_edges_and_corners():
    bottom = estimate_partial_state(np.array([100, 300, 200, 481], dtype=np.float32), 640, 480)
    assert bottom["sides"] == ["bottom"] and bottom["kind"] == "bottom"
    assert bottom["upper_only"] is True and bottom["lower_only"] is False
    left = estimate_partial_state(np.array([0, 100, 120, 400], dtype=np.float32), 640, 480)
    assert left["sides"] == ["left"] and left["kind"] == "left"
    corner = estimate_partial_state(np.array([-20, -10, 100, 200], dtype=np.float32), 640, 480)
    assert corner["sides"] == ["top", "left"] and corner["kind"] == "corner"
    # A short box high in the frame reads as an upper-body-only detection.
    partial = estimate_partial_state(np.array([300, 100, 400, 220], dtype=np.float32), 640, 480)
    assert partial["truncated"] is False and partial["upper_only"] is True
    assert partial["est_visible_frac"] < 0.6


# ----------------------------------------------------------------------- fallback
def test_fallback_descriptor_is_unit_norm_and_deterministic():
    img = rng_image(11, h=120, w=60)
    d1 = fallback_descriptor(img)
    d2 = fallback_descriptor(img.copy())
    assert d1.shape == (FALLBACK_DIM,)
    assert d1.dtype == np.float32
    assert float(np.linalg.norm(d1)) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(d1, d2)
    # A grayscale crop takes the same path.
    gray = np.random.default_rng(2).integers(0, 255, (120, 60), dtype=np.uint8)
    assert fallback_descriptor(gray).shape == (FALLBACK_DIM,)
    assert fallback_descriptor(np.zeros((0, 0, 3), dtype=np.uint8)).shape == (FALLBACK_DIM,)


def test_fallback_descriptor_is_position_invariant():
    img = rng_image(12, h=200, w=300)
    box = np.array([30, 40, 110, 180], dtype=np.float32)
    c1 = crop(img, box)
    big = np.zeros((400, 500, 3), dtype=np.uint8)
    big[60:260, 80:380] = img
    c2 = crop(big, box + np.array([80, 60, 80, 60], dtype=np.float32))
    assert c1 is not None and c2 is not None and np.array_equal(c1, c2)
    assert np.allclose(fallback_descriptor(c1), fallback_descriptor(c2))
    # Different content must give a different descriptor.
    assert not np.allclose(fallback_descriptor(c1), fallback_descriptor(crop(img, box + 60)))


# ------------------------------------------------------------------------ encode
def test_fallback_encode_shapes_and_normalization():
    emb = fallback_embedder()
    assert emb.dim == FALLBACK_DIM
    img = rng_image(21, h=300, w=400)
    boxes = np.array([[20, 30, 90, 250], [200, 60, 260, 260]], dtype=np.float32)
    out = emb.encode(img, boxes, parts=ALL_PARTS)
    assert len(out) == 2
    for d in out:
        assert set(d) == set(ALL_PARTS)
        for part, f in d.items():
            assert f.dtype == np.float32 and f.shape == (FALLBACK_DIM,)
            assert float(np.linalg.norm(f)) == pytest.approx(1.0, abs=1e-5)
    # A subset of parts is honored exactly.
    only_full = emb.encode(img, boxes, parts=("full",))
    assert set(only_full[0]) == {"full"}
    # Different people must not collapse to the same embedding.
    assert not np.allclose(out[0]["full"], out[1]["full"])


def test_fallback_encode_is_deterministic_and_position_invariant():
    img = rng_image(22, h=300, w=400)
    boxes = np.array([[30, 40, 110, 180]], dtype=np.float32)
    a = fallback_embedder().encode(img, boxes, parts=ALL_PARTS)[0]
    b = fallback_embedder().encode(img, boxes, parts=ALL_PARTS)[0]
    for part in ALL_PARTS:
        assert np.array_equal(a[part], b[part]), "same crop must give the same embedding"

    big = np.zeros((600, 700, 3), dtype=np.uint8)
    big[100:400, 150:550] = img
    c = fallback_embedder().encode(big, boxes + np.array([150, 100, 150, 100], dtype=np.float32), parts=ALL_PARTS)[0]
    for part in ALL_PARTS:
        assert np.allclose(a[part], c[part]), f"{part} must not depend on absolute position"


def test_encode_handles_edge_clipped_boxes_and_empty_input():
    emb = fallback_embedder()
    img = rng_image(23, h=300, w=300)
    out = emb.encode(img, np.array([[-40, 20, 60, 220]], dtype=np.float32), parts=ALL_PARTS)[0]
    assert set(out) == set(ALL_PARTS), "a partly out-of-frame person still yields part crops"
    assert emb.encode(img, np.zeros((0, 4), dtype=np.float32), parts=ALL_PARTS) == []
    # A degenerate box contributes an empty dict rather than raising.
    assert emb.encode(img, np.array([[10, 10, 10, 50]], dtype=np.float32), parts=ALL_PARTS) == [{}]


def test_encode_crops_batches_and_tolerates_bad_entries():
    emb = fallback_embedder()
    crops = [rng_image(31, h=64, w=32), None, np.zeros((10, 10, 3), dtype=np.uint8), rng_image(32, h=64, w=32)]
    out = emb.encode_crops(crops)
    assert len(out) == 4
    assert out[1] is None
    for f in (out[0], out[2], out[3]):
        assert f is not None and f.shape == (FALLBACK_DIM,)
        assert float(np.linalg.norm(f)) == pytest.approx(1.0, abs=1e-5)
    assert emb.encode_crops([]) == []
    # Grayscale and float crops are normalized into the same pipeline.
    assert emb.encode_crops([np.full((64, 32), 128, dtype=np.uint8)])[0] is not None
    assert emb.encode_crops([np.full((64, 32, 3), 1.5, dtype=np.float32)])[0] is not None


def test_encode_rejects_unknown_parts():
    emb = fallback_embedder()
    with pytest.raises(KeyError):
        emb.encode(rng_image(41), np.array([[0, 0, 30, 60]], dtype=np.float32), parts=("left_arm",))


def test_auto_model_reports_why_it_fell_back():
    emb = Embedder(EmbedderCfg(model="auto"), device="cpu")
    assert emb.mode == "fallback"
    assert emb.is_real is False
    assert "auto" in emb.fallback_reason
    assert emb.info["is_real"] is False


def test_force_fallback_records_the_reason_and_info():
    emb = Embedder(EmbedderCfg(), device="cuda", force_fallback=True)
    assert emb.mode == "fallback"
    assert emb.fallback_reason == "force_fallback=True"
    assert emb.input_size is None
    assert emb.info["mode"] == "fallback"
    assert emb.info["dim"] == FALLBACK_DIM
    assert "Embedder" in repr(emb)


def _real_embedder() -> Embedder:
    """Real encoder built on the *resolved* weights path.

    Never pass a bare ``"yolo26n-reid.onnx"``: ultralytics sees that basename in
    ``REID_ASSETS`` and tries to download it from GitHub, which is unreachable
    here -- the test then spends minutes in retries instead of failing. A
    default-constructed ``EmbedderCfg`` resolves the local file for us.
    """
    model = os.environ.get("TGTRACK_REID_MODEL") or EmbedderCfg().model
    return Embedder(EmbedderCfg(model=model), device="cpu")


@pytest.mark.skipif(
    os.environ.get("TGTRACK_TEST_REAL_REID") != "1",
    reason="set TGTRACK_TEST_REAL_REID=1 to exercise the ONNX/PT ReID path",
)
def test_real_reid_path_optional():
    emb = _real_embedder()
    if not emb.is_real:
        pytest.skip(f"real ReID unavailable: {emb.fallback_reason}")
    assert emb.dim and emb.dim > 0
    img = rng_image(51, h=300, w=400)
    out = emb.encode(img, np.array([[20, 30, 90, 250]], dtype=np.float32), parts=("full", "upper"))
    assert set(out[0]) == {"full", "upper"}
    for f in out[0].values():
        assert f.shape == (emb.dim,)
        assert float(np.linalg.norm(f)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(
    os.environ.get("TGTRACK_TEST_REAL_REID") != "1",
    reason="set TGTRACK_TEST_REAL_REID=1 to exercise the ONNX/PT ReID path",
)
def test_real_part_aware_survives_occlusion():
    """The novelty claim, measured on the real network.

    With the lower body occluded, the whole-body embedding must degrade more
    than the still-intact part embeddings — that is exactly why the gallery is
    part-aware instead of storing one full-body vector per target.
    """
    emb = _real_embedder()
    if not emb.is_real:
        pytest.skip(f"real ReID unavailable: {emb.fallback_reason}")
    parts = ("full", "upper", "lower", "head", "torso")
    person = np.full((400, 140, 3), 30, np.uint8)
    person[:200] = (40, 40, 220)
    person[:200:8, :] = 255
    person[200:] = (200, 120, 30)
    person[200::10, ::10] = 255
    frame = np.full((600, 600, 3), 90, np.uint8)
    frame[100:500, 200:340] = person
    box = np.array([[200, 100, 340, 500]], dtype=np.float32)

    clean = emb.encode(frame, box, parts=parts)[0]
    occluded_frame = frame.copy()
    occluded_frame[300:500, 200:340] = 0  # big occluder over the lower body only
    occluded = emb.encode(occluded_frame, box, parts=parts)[0]

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    sims = {p: cos(clean[p], occluded[p]) for p in parts}
    assert sims["full"] < 0.9, f"whole-body embedding should degrade: {sims}"
    assert sims["upper"] > sims["full"] + 0.1, f"the intact part must hold up better: {sims}"
    assert sims["lower"] < 0.8, f"the occluded part should be the unreliable one: {sims}"


# ------------------------------------------- self-healing batch + honest flags
#: The real failure that killed a 120-run grid: one Conv buffer of 257 MiB while
#: four workers shared an 8 GB card. Reproducing it for real would mean actually
#: exhausting the GPU, so the session is stubbed to raise exactly this text.
_OOM_MSG = (
    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running "
    "Conv node. Name:'/model.2/cv2/conv/Conv' Status Message: "
    "...bfc_arena.cc:376 onnxruntime::BFCArena::AllocateRawInternal "
    "Failed to allocate memory for requested buffer of size 269746176"
)


class _ChunkBatch:
    """Stand-in for the ``(B, 3, H, W)`` tensor, carrying one id per crop.

    The ids let a test prove that every crop comes back exactly once, in order,
    even when the forward gets split into differently-sized chunks on retry.
    """

    def __init__(self, ids: list[float]) -> None:
        self.ids = list(ids)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.ids), 3, 8, 8)

    def __getitem__(self, sl: slice) -> _ChunkBatch:
        return _ChunkBatch(self.ids[sl])


class _FakeSession:
    """A session whose ``run`` fails like ORT's arena above a given size."""

    def __init__(
        self,
        dim: int = 4,
        fail_above: int | None = None,
        fail_times: int | None = 1,
        message: str = _OOM_MSG,
    ) -> None:
        self.dim = dim
        self.fail_above = fail_above
        self.fail_times = fail_times
        self.message = message
        self.sizes: list[int] = []

    def run(self, chunk: _ChunkBatch) -> np.ndarray:
        n = int(chunk.shape[0])
        self.sizes.append(n)
        failing = self.fail_above is not None and n > self.fail_above
        if failing and (self.fail_times is None or self.fail_times > 0):
            if self.fail_times is not None:
                self.fail_times -= 1
            raise RuntimeError(self.message)
        # component 1 is monotonic in the crop id, so the original crop order is
        # still recoverable from the L2-normalized embedding.
        return np.array([[1.0, float(c), 0.0, 0.0][: self.dim] for c in chunk.ids], dtype=np.float32)

    def __call__(self, chunk: _ChunkBatch) -> np.ndarray:
        """``reid.model(chunk)`` is the ORT wrapper around this same ``run``."""
        return self.run(chunk)


class _FakeTorch:
    """Only what the ONNX path touches: ``no_grad`` and the cuda-cache hooks."""

    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @property
    def cuda(self) -> _FakeTorch:
        return self

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def no_grad(self):  # noqa: ANN201 - mimics torch.no_grad()
        return contextlib.nullcontext()


class _FakeReID:
    """ReID stand-in: ``reid.model(chunk)`` delegates to the session's ``run``."""

    is_pt = False
    batch_size = None

    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.model = session

    def _crops_to_tensor(self, crops: list[np.ndarray]) -> _ChunkBatch:
        return _ChunkBatch([float(c.reshape(-1)[0]) for c in crops])


def stub_embedder(session: _FakeSession, max_batch: int = 8) -> tuple[Embedder, _FakeTorch]:
    """An Embedder wired to a fake ONNX session; no model is ever loaded."""
    emb = Embedder(EmbedderCfg(max_batch=max_batch), device="cpu", force_fallback=True)
    torch = _FakeTorch()
    emb.mode = "onnx"
    emb.dim = session.dim
    emb.provider = "CPUExecutionProvider"
    emb.batch_size = None
    emb._torch = torch
    emb._reid = _FakeReID(session)
    assert emb.is_real is True
    return emb, torch


def id_crops(n: int) -> list[np.ndarray]:
    """``n`` distinguishable constant crops; crop ``k`` carries the id ``k + 1``."""
    return [np.full((16, 16, 3), k + 1, np.uint8) for k in range(n)]


def test_allocation_failure_halves_the_batch_and_still_returns_embeddings():
    session = _FakeSession(fail_above=4)
    emb, torch = stub_embedder(session, max_batch=8)

    with pytest.warns(RuntimeWarning, match="halving to 4"):
        out = emb.encode_crops(id_crops(8))

    assert session.sizes == [8, 4, 4], "the 8-crop forward must fail, then two 4-crop ones run"
    assert emb.oom_retries == 1
    assert emb.last_batch_size == 4
    assert emb.effective_max_batch == 4, "the shrink must be remembered for the next call"
    assert emb.info["effective_max_batch"] == 4
    assert torch.empty_cache_calls >= 1, "cached blocks are handed back before retrying"
    assert all(f is not None and f.shape == (4,) for f in out)
    assert float(np.linalg.norm(out[0])) == pytest.approx(1.0, abs=1e-5)
    key = [float(f[1]) for f in out]
    assert key == sorted(key) and len(set(key)) == len(key), "every crop once, in order"
    assert "269746176" in emb.last_oom_reason


def test_a_shrunk_batch_is_reused_on_the_next_call():
    session = _FakeSession(fail_above=4, fail_times=1)
    emb, _ = stub_embedder(session, max_batch=8)
    with pytest.warns(RuntimeWarning):
        emb.encode_crops(id_crops(8))

    session.sizes.clear()
    out = emb.encode_crops(id_crops(8))  # no warning this time
    assert session.sizes == [4, 4], "must not re-pay the OOM every frame"
    assert all(f is not None for f in out)


def test_allocation_failure_at_batch_one_still_raises():
    session = _FakeSession(fail_above=0, fail_times=None)  # nothing fits, not even one crop
    emb, _ = stub_embedder(session, max_batch=8)

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="BFCArena"):
            emb.encode_crops(id_crops(8))

    assert session.sizes == [8, 4, 2, 1], "halve down to 1, then give up loudly"
    assert emb.effective_max_batch == 1
    assert emb.oom_retries == 3


def test_a_logic_error_is_never_retried():
    session = _FakeSession(fail_above=0, message="shape mismatch: expected 3 channels, got 4")
    emb, _ = stub_embedder(session, max_batch=8)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        emb.encode_crops(id_crops(8))

    assert session.sizes == [8], "only resource failures may retry"
    assert emb.oom_retries == 0


@pytest.mark.parametrize(
    "message",
    [
        _OOM_MSG,
        "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : something went wrong in session.run",
    ],
)
def test_onnx_runtime_errors_are_treated_as_retryable(message):
    session = _FakeSession(fail_above=2, message=message)
    emb, _ = stub_embedder(session, max_batch=4)

    with pytest.warns(RuntimeWarning):
        out = emb.encode_crops(id_crops(4))

    assert session.sizes == [4, 2, 2]
    assert all(f is not None for f in out)


def test_a_retry_does_not_turn_a_missing_part_into_an_error():
    """Contract: a degenerate crop stays absent, before and after a retry."""
    session = _FakeSession(fail_above=2)
    emb, _ = stub_embedder(session, max_batch=8)
    frame = rng_image(7, h=200, w=300)
    boxes = np.array([[10, 10, 60, 150], [0, 0, 0, 0], [150, 20, 210, 180]], dtype=np.float32)

    with pytest.warns(RuntimeWarning, match="halving"):
        out = emb.encode(frame, boxes, parts=("full", "upper"))

    assert len(out) == 3
    assert out[1] == {}, "the degenerate box has no parts, and the retry must not invent any"
    assert set(out[0]) == {"full", "upper"}
    assert set(out[2]) == {"full", "upper"}


def test_cuda_request_downgraded_before_the_session_is_flagged(monkeypatch):
    """The silent-fallback hole: ``_resolve_device`` finding no usable CUDA.

    ``requested_device`` is the *resolved* device, so a cuda request downgraded
    to cpu used to look like a deliberate cpu request and went unflagged.
    """
    monkeypatch.setattr(Embedder, "_resolve_device", staticmethod(lambda device: "cpu"))

    with pytest.warns(RuntimeWarning, match="NOT coming from the GPU"):
        emb = Embedder(EmbedderCfg(), device="cuda", force_fallback=True)

    assert emb.raw_device == "cuda"
    assert emb.requested_device == "cpu"
    assert emb.device == "cpu"
    assert emb.gpu_fallback is True
    assert emb.info["gpu_fallback"] is True
    assert emb.on_cuda is False
    assert "cuda" in emb.device_note


def test_a_deliberate_cpu_encoder_is_not_reported_as_a_gpu_fallback():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        emb = Embedder(EmbedderCfg(), device="cpu", force_fallback=True)

    assert emb.raw_device == "cpu"
    assert emb.gpu_fallback is False
    assert emb.on_cuda is False
    assert emb.info["on_cuda"] is False
