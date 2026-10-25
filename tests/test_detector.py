"""Tests for the perception layer: ``PersonDetector`` + ``AllPersonTracker``.

Run:  python -m pytest tests/test_detector.py -q

The unit tests need neither torch-CUDA, nor a checkpoint, nor even ultralytics:
the box post-processing (:func:`tgtrack.detector.boxes_to_detections`) and the
detection<->tracker glue (:mod:`tgtrack.mot_layer`) are pure numpy. The GPU test is
skipped when CUDA is unavailable or the weights cannot be loaded.
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tgtrack.config import DetectorCfg, MotCfg  # noqa: E402
from tgtrack.detector import PersonDetector, boxes_to_detections, visible_ratio  # noqa: E402
from tgtrack.mot_layer import (  # noqa: E402
    TRACKER_YAMLS,
    _BoxesShim,
    apply_track_ids,
    dets_to_box_array,
)
from tgtrack.types import Detection  # noqa: E402

BUS_URL = "https://ultralytics.com/images/bus.jpg"
PERSON = 0


def _det(box, score=0.9, cls=PERSON):
    return Detection(xyxy=np.asarray(box, dtype=np.float32), score=score, cls=cls)


# ---------------------------------------------------------------------------
# A) pure post-processing: person-class / conf filtering on synthetic tensors
# ---------------------------------------------------------------------------
RAW_XYXY = np.array(
    [
        [10.0, 20.0, 60.0, 180.0],  # person, conf .90  -> keep
        [100.0, 20.0, 150.0, 180.0],  # class 2, conf .80 -> keep only when person_class=None
        [200.0, 20.0, 250.0, 180.0],  # person, conf .05  -> drop (below threshold)
        [300.0, 20.0, 400.0, 180.0],  # person, conf .99  -> keep
        [500.0, 20.0, 500.0, 180.0],  # person, conf .70, ZERO WIDTH -> drop
        [10.0, 30.0, 60.0, 190.0],  # person, conf NaN   -> drop
    ],
    dtype=np.float32,
)
RAW_CONF = np.array([0.90, 0.80, 0.05, 0.99, 0.70, np.nan], dtype=np.float32)
RAW_CLS = np.array([0, 2, 0, 0, 0, 0])


def test_person_class_and_conf_filtering():
    dets = boxes_to_detections(RAW_XYXY, RAW_CONF, RAW_CLS, 640, 480, conf_thres=0.25, person_class=0)
    assert [round(d.score, 2) for d in dets] == [0.99, 0.90]  # sorted by descending score
    assert {d.cls for d in dets} == {0}


def test_person_class_none_keeps_other_classes():
    dets = boxes_to_detections(RAW_XYXY, RAW_CONF, RAW_CLS, 640, 480, conf_thres=0.25, person_class=None)
    assert [round(d.score, 2) for d in dets] == [0.99, 0.90, 0.80]
    assert 2 in {d.cls for d in dets}


def test_conf_threshold_edges_and_degenerate_boxes():
    dets = boxes_to_detections(RAW_XYXY, RAW_CONF, RAW_CLS, 640, 480, conf_thres=0.95, person_class=0)
    assert len(dets) == 1 and round(dets[0].score, 2) == 0.99
    # conf == threshold is inclusive; zero-width and NaN boxes never survive
    dets = boxes_to_detections(RAW_XYXY, RAW_CONF, RAW_CLS, 640, 480, conf_thres=0.90, person_class=0)
    assert [round(d.score, 2) for d in dets] == [0.99, 0.90]
    assert all(d.wh[0] > 0 and d.wh[1] > 0 for d in dets)


def test_max_det_and_quality_and_no_input_mutation():
    xyxy = RAW_XYXY.copy()
    conf = RAW_CONF.copy()
    dets = boxes_to_detections(xyxy, conf, RAW_CLS, 640, 480, conf_thres=0.25, person_class=0, max_det=1)
    assert len(dets) == 1 and round(dets[0].score, 2) == 0.99
    assert dets[0].quality == dets[0].score  # quality comes from the score
    assert np.array_equal(xyxy, RAW_XYXY)  # inputs are never mutated in place
    assert np.array_equal(conf[np.isfinite(RAW_CONF)], RAW_CONF[np.isfinite(RAW_CONF)])  # NaN-aware
    assert np.isnan(conf[5])


def test_empty_input_returns_empty_list():
    assert boxes_to_detections(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), 640, 480) == []


def test_torch_tensor_input_matches_numpy():
    torch = pytest.importorskip("torch")
    from_np = boxes_to_detections(RAW_XYXY, RAW_CONF, RAW_CLS, 640, 480)
    from_pt = boxes_to_detections(
        torch.as_tensor(RAW_XYXY), torch.as_tensor(RAW_CONF), torch.as_tensor(RAW_CLS), 640, 480
    )
    assert [round(d.score, 3) for d in from_pt] == [round(d.score, 3) for d in from_np]
    assert np.allclose(from_pt[0].xyxy, from_np[0].xyxy)


def test_to_numpy_helper_handles_tensors():
    torch = pytest.importorskip("torch")
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert isinstance(PersonDetector._to_numpy(t), np.ndarray)
    assert isinstance(PersonDetector._to_numpy(np.zeros(3, dtype=np.float32)), np.ndarray)


def test_visible_ratio_geometry():
    assert visible_ratio([10, 10, 50, 90], 640, 480) == 1.0
    # right half outside the 640-wide frame: 40 of 100 px wide inside
    assert visible_ratio([600, 100, 700, 200], 640, 480) == pytest.approx(0.4)
    assert visible_ratio([-50, 100, 50, 200], 640, 480) == pytest.approx(0.5)
    assert visible_ratio([700, 100, 800, 200], 640, 480) == 0.0
    assert visible_ratio([10, 10, 10, 90], 640, 480) == 0.0  # degenerate
    dets = boxes_to_detections(np.array([[600.0, 100.0, 700.0, 200.0]]), np.array([0.9]), np.array([0]), 640, 480)
    assert dets[0].visible_ratio == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# B) detection -> tracker glue (pure numpy, no ultralytics needed)
# ---------------------------------------------------------------------------
def test_detection_array_layout_is_xyxy_conf_cls():
    dets = [_det([1, 2, 3, 4], 0.5), _det([5, 6, 7, 8], 0.75)]
    arr = dets_to_box_array(dets)
    assert arr.shape == (2, 6) and arr.dtype == np.float32
    assert arr[0].tolist() == [1.0, 2.0, 3.0, 4.0, 0.5, 0.0]  # conf BEFORE cls (Boxes layout)
    assert arr[1].tolist() == [5.0, 6.0, 7.0, 8.0, 0.75, 0.0]
    assert dets_to_box_array([]).shape == (0, 6)


def test_apply_track_ids_uses_idx_column():
    dets = [_det([1, 1, 2, 2]), _det([3, 3, 4, 4]), _det([5, 5, 6, 6])]
    # (N, 8) [x1,y1,x2,y2,track_id,score,cls,idx]; row order deliberately shuffled
    tracks = np.array(
        [
            [5, 5, 6, 6, 42, 0.8, 0, 2],
            [1, 1, 2, 2, 7, 0.9, 0, 0],
        ],
        dtype=np.float32,
    )
    out = apply_track_ids(dets, tracks)
    assert out is dets
    assert [d.track_id for d in dets] == [7, None, 42]


def test_apply_track_ids_tolerates_empty_and_obb_layout():
    dets = [_det([1, 1, 2, 2])]
    assert apply_track_ids(dets, np.zeros((0,), dtype=np.float32))[0].track_id is None
    assert apply_track_ids(dets, np.zeros((0, 8), dtype=np.float32))[0].track_id is None
    assert apply_track_ids(dets, None)[0].track_id is None
    # OBB layout (N, 9) [x,y,w,h,angle,track_id,score,cls,idx] -> -4 / -1 still work
    obb = np.array([[1, 1, 2, 2, 0.0, 5, 0.9, 0, 0]], dtype=np.float32)
    assert apply_track_ids(dets, obb)[0].track_id == 5
    with pytest.raises(ValueError):
        apply_track_ids(dets, np.zeros((1, 7), dtype=np.float32))


def test_boxes_shim_satisfies_tracker_contract():
    data = np.array([[0, 0, 10, 20, 0.9, 0], [10, 0, 30, 20, 0.5, 1]], dtype=np.float32)
    shim = _BoxesShim(data)
    assert len(shim) == 2
    assert np.allclose(shim.xyxy, data[:, :4])
    assert np.allclose(shim.conf, [0.9, 0.5]) and np.allclose(shim.cls, [0, 1])
    assert np.allclose(shim.xywh[0], [5, 10, 10, 20])  # [cx, cy, w, h]
    sub = shim[np.array([True, False])]
    assert len(sub) == 1 and np.allclose(sub.xyxy[0], data[0, :4])


# ---------------------------------------------------------------------------
# C) real backends: direct-class route, stable ids, reset, empty frames
# ---------------------------------------------------------------------------
def _synthetic_frame(h=480, w=640, shift=0):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[100:300, 50 + shift : 90 + shift] = 200
    return frame


def _moving_persons(shift):
    return [
        _det([50 + shift, 100, 90 + shift, 300], 0.9),
        _det([250 + shift, 120, 290 + shift, 320], 0.8),
        _det([450 + shift, 90, 490 + shift, 290], 0.7),
    ]


@pytest.mark.parametrize("tracker", list(TRACKER_YAMLS))
def test_all_backends_assign_stable_ids_and_reset(tracker):
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import AllPersonTracker

    mot = AllPersonTracker(MotCfg(tracker=tracker, with_reid=False, gmc_method="none"))
    ids_f1 = sorted(d.track_id for d in mot.update(_synthetic_frame(), _moving_persons(0)))
    assert all(i is not None for i in ids_f1), f"{tracker}: unconfirmed ids on frame 1"
    assert len(set(ids_f1)) == 3
    ids_f2 = sorted(d.track_id for d in mot.update(_synthetic_frame(shift=5), _moving_persons(5)))
    assert ids_f2 == ids_f1, f"{tracker}: ids not stable across a 5 px shift ({ids_f1} -> {ids_f2})"
    mot.reset()
    ids_after = sorted(d.track_id for d in mot.update(_synthetic_frame(), _moving_persons(0)))
    assert ids_after == [1, 2, 3], f"{tracker}: reset() did not restart the ID counter ({ids_after})"


@pytest.mark.parametrize("tracker", ["botsort.yaml", "deepocsort.yaml", "tracktrack.yaml"])
def test_reid_backends_consume_our_embedding_features(tracker):
    """model='auto' + feats= uses OUR embeddings; no ReID download, no second detector."""
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import AllPersonTracker

    rng = np.random.default_rng(0)
    mot = AllPersonTracker(MotCfg(tracker=tracker, with_reid=True, gmc_method="none"))
    assert mot.describe()["model"] == "auto"
    dets = _moving_persons(0)
    for i, d in enumerate(dets):
        d.feat = rng.normal(size=128).astype(np.float32) + i  # distinct appearance per person
    dets = mot.update(_synthetic_frame(), dets)
    assert len({d.track_id for d in dets}) == 3


def test_empty_frame_does_not_crash():
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import AllPersonTracker

    mot = AllPersonTracker(MotCfg(tracker="botsort.yaml", with_reid=False, gmc_method="none"))
    mot.update(_synthetic_frame(), _moving_persons(0))
    for _ in range(3):
        assert mot.update(_synthetic_frame(), []) == []  # no detections: advances, never raises
    assert mot.update(_synthetic_frame(shift=5), _moving_persons(5))[0].track_id is not None


def test_gmc_sparse_optflow_path_runs():
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import AllPersonTracker

    mot = AllPersonTracker(MotCfg(tracker="botsort.yaml", with_reid=False, gmc_method="sparseOptFlow"))
    mot.update(_synthetic_frame(), _moving_persons(0))
    dets = mot.update(_synthetic_frame(shift=3), _moving_persons(3))
    assert any(d.track_id is not None for d in dets)


def test_tracker_yaml_resolution_and_override_merge():
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import build_tracker_args, resolve_tracker_yaml

    for name in TRACKER_YAMLS:
        path = resolve_tracker_yaml(name)
        assert path.exists() and path.name == name
    with pytest.raises(ValueError):
        resolve_tracker_yaml("definitely_not_a_tracker.yaml")
    cfg = MotCfg(tracker="deepocsort.yaml", track_buffer=7, match_thresh=0.55, extra={"use_byte": True})
    args, path = build_tracker_args(cfg, device="cpu")
    assert args.tracker_type == "deepocsort" and path.name == "deepocsort.yaml"
    assert args.track_buffer == 7 and args.match_thresh == pytest.approx(0.55)
    assert args.use_byte is True and args.device == "cpu"


def test_mot_layer_describe_reports_effective_config():
    pytest.importorskip("ultralytics")
    from tgtrack.mot_layer import AllPersonTracker

    d = AllPersonTracker(MotCfg(tracker="bytetrack.yaml", with_reid=False)).describe()
    assert d["tracker_type"] == "bytetrack" and d["tracker_yaml"].endswith("bytetrack.yaml")


def test_predict_kwargs_uses_unified_quantize_arg():
    """ultralytics 8.4.185 deprecated `half` in favour of `quantize` (16=FP16, 32=FP32)."""
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import DEFAULT_CFG_DICT

    det = PersonDetector.__new__(PersonDetector)  # no checkpoint needed for kwargs
    det.cfg = DetectorCfg(device="cpu", half=False, imgsz=512, conf=0.4, iou=0.6, max_det=42)
    det.device = "cpu"
    det.half = False
    kw = det.predict_kwargs
    assert kw["classes"] == [PERSON]
    assert (kw["imgsz"], kw["conf"], kw["iou"], kw["max_det"]) == (512, 0.4, 0.6, 42)
    assert kw["quantize"] == 32 and "half" not in kw
    assert "quantize" in DEFAULT_CFG_DICT
    det.half = True
    assert det.predict_kwargs["quantize"] == 16


# ---------------------------------------------------------------------------
# D) end-to-end smoke test: real YOLO26n weights (SKIPPED without CUDA/weights)
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _weights_path() -> Path | None:
    for cand in (ROOT / "yolo26n.pt", Path.cwd() / "yolo26n.pt", ROOT / "weights" / "yolo26n.pt"):
        if cand.exists():
            return cand
    return None


def _bus_image() -> np.ndarray | None:
    import cv2

    for cand in (ROOT / "data" / "bus.jpg", ROOT / "results" / "bus.jpg", Path(tempfile.gettempdir()) / "bus.jpg"):
        if cand.exists():
            img = cv2.imread(str(cand))
            if img is not None:
                return img
    try:
        dst = Path(tempfile.gettempdir()) / "tgtrack_assets" / "bus.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            urllib.request.urlretrieve(BUS_URL, dst)  # noqa: S310 - fixed ultralytics asset URL
        return cv2.imread(str(dst))
    except Exception:
        return None


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available (torch may still be the CPU build)")
def test_yolo26n_gpu_smoke_detects_persons_and_benchmarks():
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    weights = _weights_path()
    if weights is None:
        if _bus_image() is None:
            pytest.skip("no yolo26n.pt weights available")
        pytest.skip("no yolo26n.pt weights available (auto-download happens on first PersonDetector use)")
    det = PersonDetector(DetectorCfg(weights=str(weights), device="cuda", half=True, conf=0.25))
    assert det.device.startswith("cuda") and det.half is True
    img = _bus_image()
    if img is None:
        pytest.skip("no test image available")
    dets = det.detect(img)
    assert isinstance(dets, list)
    assert len(dets) >= 1, "expected at least one person in the bus.jpg smoke image"
    assert all(d.cls == PERSON and 0.0 <= d.visible_ratio <= 1.0 for d in dets)
    info = det.model_info()
    assert info["n_params"] and info["n_params"] > 1e6
    bench = det.benchmark(img, n=10)
    assert bench["fps"] > 0 and bench["ms_mean"] > 0 and bench["device"].startswith("cuda")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_detector_falls_back_to_cpu_when_forced():
    pytest.importorskip("ultralytics")
    weights = _weights_path()
    if weights is None:
        pytest.skip("no yolo26n.pt weights available")
    det = PersonDetector(DetectorCfg(weights=str(weights), device="cpu", half=True))
    assert det.device == "cpu" and det.half is False  # half is force-disabled off CUDA
    assert det.names.get(0) == "person" or len(det.names) == 80
