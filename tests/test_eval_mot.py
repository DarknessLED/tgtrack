#!/usr/bin/env python3
"""Unit tests for ``scripts/eval_mot.py`` (the parallel MOT evaluation harness).

Everything here runs on tiny synthetic MOT sequences written into ``tmp_path`` and
on injected fake detector/tracker/embedder factories, so the whole harness --
pair enumeration, the frame loop, GT/Pred accumulation, ``evaluate_mot`` scoring,
parallel bookkeeping and report writing -- is covered **without torch,
ultralytics, ONNX, a GPU or a real dataset**.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_mot.py"


@pytest.fixture(scope="module")
def harness():
    """Load ``scripts/eval_mot.py`` as a module (it is not an importable package)."""
    spec = importlib.util.spec_from_file_location("eval_mot_harness", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def gt_rows(n_frames=8, ids=(1, 2)):
    """The GT table written by :func:`write_sequence` (frame, id, x, y, w, h, ...).

    Track ``i`` moves right by 3 px/frame and sits 30 px right of the previous one,
    so the left-to-right order of the ids never changes.
    """
    rows = []
    for t in range(n_frames):
        for i, tid in enumerate(ids):
            rows.append((t + 1, tid, 4 + 3 * t + 30 * i, 6 + 4 * i, 12, 20, 1, 1, 1.0))
    return rows


def write_sequence(root: Path, name="MOT-01", n_frames=8, ids=(1, 2)) -> Path:
    """Write ``<root>/<name>/{img1/*.jpg, gt/gt.txt}`` and return the sequence dir."""
    import cv2

    seq = root / name
    (seq / "img1").mkdir(parents=True, exist_ok=True)
    (seq / "gt").mkdir(parents=True, exist_ok=True)
    for t in range(n_frames):
        img = np.full((64, 80, 3), 40, np.uint8)
        for i, _tid in enumerate(ids):
            x, y = 4 + 3 * t + 30 * i, 6 + 4 * i
            cv2.rectangle(img, (x, y), (x + 12, y + 20), (200, 200, 200), -1)
        cv2.imwrite(str(seq / "img1" / f"{t + 1:06d}.jpg"), img)
    lines = [",".join(str(v) for v in row) for row in gt_rows(n_frames, ids)]
    (seq / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return seq


def perfect_boxes(seq_dir: Path):
    """Per-frame GT boxes read back through the loader (a 'perfect detector')."""
    from tgtrack.data import load_mot_sequence

    seq = load_mot_sequence(str(seq_dir))
    return [[np.asarray(g.xyxy, dtype=np.float64) for g in seq.gt_at(i)] for i in range(len(seq))]


class FakeDetector:
    """Replays precomputed boxes in frame order; marks itself CPU-only."""

    def __init__(self, boxes_by_frame):
        self.by_frame = [list(b) for b in boxes_by_frame]
        self.device = "cpu"
        self.calls = 0
        self.warmups = 0

    def warmup(self):
        # Mirrors PersonDetector.warmup() without consuming a replay slot.
        self.warmups += 1

    def detect(self, img):
        from tgtrack.types import Detection

        boxes = self.by_frame[self.calls] if self.calls < len(self.by_frame) else []
        self.calls += 1
        return [Detection(xyxy=np.asarray(b, dtype=np.float64), score=0.9) for b in boxes]


class FakeTracker:
    """Slot-order id assignment (order == GT order here, so tracking is perfect)."""

    def __init__(self):
        self.resets = 0
        self.updates = 0
        self.last_feats = None
        self.last_dets = 0

    def update(self, img, dets, feats=None):
        self.last_feats = feats
        self.last_dets = len(dets)
        for i, det in enumerate(dets):
            det.track_id = i + 1
        self.updates += 1
        return dets

    def reset(self):
        self.resets += 1


class FakeEmbedder:
    """Part-aware embedder stub: returns a ``dim``-vector for all but ``skip`` boxes."""

    def __init__(self, dim=8, skip=()):
        self.dim = dim
        self.skip = set(skip)
        self.info = {"mode": "fake", "dim": dim, "is_real": False, "provider": "None"}
        self.calls = 0

    def encode(self, img, boxes, parts=("full",)):
        self.calls += 1
        out = []
        for i in range(len(boxes)):
            if i in self.skip:
                out.append({})
            else:
                vec = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.float32)
                out.append({"full": vec})
        return out


def fake_record(tracker, result, *, ms=10.0, wall=1.0, n_frames=8):
    """A record shaped like :func:`eval_mot.run_pair`'s return value."""
    return {
        "ok": True,
        "pair_id": f"{Path(tracker).stem}__MOT-01",
        "sequence": "MOT-01",
        "tracker": tracker,
        "num_frames": n_frames,
        "timing": {
            "setup_ms": 5.0,
            "wall_s": wall,
            "frame_ms": {"n": n_frames, "mean": ms, "p50": ms, "p95": ms, "min": ms, "max": ms, "sum": ms * n_frames},
            "detector_ms": {"mean": ms * 0.6},
            "embedder_ms": {"mean": 0.0},
            "tracker_ms": {"mean": ms * 0.4},
            "pipeline_fps": 1000.0 / ms,
            "wall_fps": n_frames / wall,
        },
        "metrics": {k: result.get(k) for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR", "FP", "FN", "IDSW", "Frag", "MT", "ML")},
        "result": result,
        "config": {"tracker": tracker},
        "embedder": None,
        "warnings": [],
    }


def simple_gt_pred(*, drop_frames=()):
    """GT/pred dicts for one id over 6 frames; ``drop_frames`` removes pred boxes."""
    from tgtrack.evaluate import GtBox, PredBox

    gt, pred = {}, {}
    for t in range(6):
        xyxy = np.array([4 + 3 * t, 6, 16 + 3 * t, 26], dtype=np.float64)
        gt[t] = [GtBox(frame=t, track_id=1, xyxy=xyxy, conf=1.0, visibility=1.0)]
        if t not in drop_frames:
            pred[t] = [PredBox(frame=t, track_id=1, xyxy=xyxy, score=0.9)]
    return gt, pred


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_parse_trackers_normalizes_and_dedupes(harness):
    assert harness.parse_trackers("botsort, ocsort.yaml ,botsort") == ["botsort.yaml", "ocsort.yaml"]
    assert harness.parse_trackers("bytetrack.yaml;tracktrack") == ["bytetrack.yaml", "tracktrack.yaml"]


def test_parse_trackers_blank_returns_defaults(harness):
    assert harness.parse_trackers(None) == list(harness.DEFAULT_TRACKERS)
    assert harness.parse_trackers("   ") == list(harness.DEFAULT_TRACKERS)
    assert len(harness.DEFAULT_TRACKERS) == 6


def test_enumerate_pairs_is_cartesian_and_ordered(harness):
    pairs = harness.enumerate_pairs(["/data/S01", "/data/S02"], ["botsort.yaml", "bytetrack.yaml"])
    assert [(p.sequence, p.tracker) for p in pairs] == [
        ("S01", "botsort.yaml"),
        ("S01", "bytetrack.yaml"),
        ("S02", "botsort.yaml"),
        ("S02", "bytetrack.yaml"),
    ]
    assert pairs[0].pair_id == "botsort__S01"
    # ids must stay filesystem safe for report files
    assert harness.safe_name("MOT17-02-FRCNN:1") == "MOT17-02-FRCNN_1"


def test_clamp_jobs(harness):
    cpu = os.cpu_count() or 1
    assert harness.clamp_jobs(0, 3) == min(cpu, 3)
    assert harness.clamp_jobs(99, 2) == 2
    assert harness.clamp_jobs(2, 5) == 2
    assert harness.clamp_jobs(4, 0) == 0


def test_filter_sequences_substring_and_limit(harness):
    dirs = ["/d/MOT17-02-FRCNN", "/d/MOT17-04-DPM", "/d/dancetrack0001"]
    assert harness.filter_sequences(dirs, only="mot17") == dirs[:2]
    assert harness.filter_sequences(dirs, only="dpm,dance") == [dirs[1], dirs[2]]
    assert harness.filter_sequences(dirs, limit=1) == dirs[:1]
    assert harness.filter_sequences(dirs, only="", limit=0) == []


def test_discover_in_missing_root_is_a_warning_not_a_crash(harness, tmp_path):
    found, notes = harness.discover_in(tmp_path / "nope")
    assert found == [] and notes and "does not exist" in notes[0]


def test_discover_in_finds_synthetic_sequence(harness, tmp_path):
    seq = write_sequence(tmp_path, "MOT-01")
    found, notes = harness.discover_in(tmp_path)
    assert notes == []
    assert [Path(p).name for p in found] == [seq.name]


def test_write_mot_tracks_uses_one_based_frames(harness, tmp_path):
    from tgtrack.evaluate import PredBox

    pred = {0: [PredBox(frame=0, track_id=7, xyxy=np.array([1.0, 2.0, 11.0, 22.0]), score=0.5)]}
    out = harness.write_mot_tracks(tmp_path / "tracks" / "x.txt", pred)
    line = out.read_text(encoding="utf-8").strip()
    assert line == "1,7,1.00,2.00,10.00,20.00,0.5000,-1,-1,-1"


def test_pred_boxes_from_dets_skips_untracked_and_coerces(harness):
    from tgtrack.types import Detection

    dets = [
        Detection(xyxy=np.array([0, 0, 10, 20]), score=0.8, track_id=3),
        Detection(xyxy=np.array([1, 1, 9, 19]), score=0.2, track_id=None),
        Detection(xyxy=np.array([2, 2, 8, 18]), score=0.6, track_id=np.int64(4)),
    ]
    boxes = harness.pred_boxes_from_dets(5, dets)
    assert [b.track_id for b in boxes] == [3, 4]
    assert all(b.frame == 5 and isinstance(b.track_id, int) for b in boxes)
    assert boxes[0].xyxy.dtype == np.float64


def test_feats_from_embeddings_fills_missing_rows(harness):
    embs = [{"full": np.array([1.0, 0.0, 0.0, 0.0], np.float32)}, {}, {"full": np.array([0.0, 1.0, 0.0, 0.0], np.float32)}]
    arr, missing = harness.feats_from_embeddings(embs, 4, 3)
    assert missing == 1
    assert arr.shape == (3, 4) and np.isfinite(arr).all()
    assert arr[1].tolist() == pytest.approx(np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2))


def test_feats_from_embeddings_all_missing_or_unknown_dim(harness):
    arr, missing = harness.feats_from_embeddings([{}, {}], 4, 2)
    assert arr is None and missing == 2
    arr, missing = harness.feats_from_embeddings([], None, 3)
    assert arr is None and missing == 3
    # a wrong-length vector counts as missing rather than corrupting the matrix
    arr, missing = harness.feats_from_embeddings([{"full": np.zeros(3, np.float32)}], 4, 1)
    assert arr is None and missing == 1


def test_latency_stats(harness):
    stats = harness.latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["n"] == 4 and stats["mean"] == pytest.approx(25.0) and stats["min"] == 10.0 and stats["max"] == 40.0
    assert harness.latency_stats([])["n"] == 0


def test_build_config_applies_cli_overrides(harness):
    opts = harness.RunOptions(weights="/tmp/w.pt", imgsz=512, conf=0.4, iou=0.6, max_det=50, device="cpu", half=False, embed=True)
    cfg = harness.build_config(opts, "deepocsort.yaml")
    assert cfg.mot.tracker == "deepocsort.yaml"
    assert cfg.mot.with_reid is True  # --embed implies the appearance term
    assert cfg.detector.imgsz == 512 and cfg.detector.conf == pytest.approx(0.4)
    assert cfg.detector.iou == pytest.approx(0.6) and cfg.detector.max_det == 50
    assert cfg.detector.device == "cpu" and cfg.detector.half is False
    plain = harness.build_config(harness.RunOptions(), "bytetrack.yaml")
    assert plain.mot.with_reid is False


def test_options_from_args_types(harness):
    args = harness.build_parser().parse_args(["--data-root", "d", "--max-frames", "0", "--jobs", "3", "--no-half", "--embed"])
    opts = harness.options_from_args(args)
    assert opts.max_frames is None  # 0 means "no cap"
    assert opts.half is False and opts.embed is True and opts.iou_thresh == pytest.approx(0.5)
    assert isinstance(opts.subsample, int) and isinstance(opts.cache_flush, int)


# --------------------------------------------------------------------------- #
# the frame loop
# --------------------------------------------------------------------------- #
def test_run_pair_perfect_tracking_scores_one(harness, tmp_path):
    seq_dir = write_sequence(tmp_path, "MOT-01", n_frames=8)
    boxes = perfect_boxes(seq_dir)
    trackers = []
    detectors = []

    def tracker_factory(cfg, det):
        trk = FakeTracker()
        trackers.append(trk)
        return trk

    def detector_factory(cfg):
        det = FakeDetector(boxes)
        detectors.append(det)
        return det

    rec = harness.run_pair(
        harness.PairSpec(sequence="MOT-01", seq_dir=str(seq_dir), tracker="botsort.yaml"),
        harness.RunOptions(out_dir=str(tmp_path / "out"), cache_flush=4),
        detector_factory=detector_factory,
        tracker_factory=tracker_factory,
    )
    assert rec["ok"] is True
    assert rec["num_frames"] == 8
    assert rec["num_orphan_detections"] == 0
    assert rec["num_missing_features"] == 0
    assert rec["result"]["num_gt_dets"] == 16 and rec["result"]["num_pred_dets"] == 16
    assert rec["metrics"]["HOTA"] == pytest.approx(1.0)
    assert rec["metrics"]["MOTA"] == pytest.approx(1.0)
    assert rec["metrics"]["IDF1"] == pytest.approx(1.0)
    assert rec["metrics"]["IDSW"] == 0 and rec["metrics"]["FP"] == 0 and rec["metrics"]["FN"] == 0
    # timing bookkeeping is populated and internally consistent
    timing = rec["timing"]
    assert timing["frame_ms"]["n"] == 8 and timing["frame_ms"]["mean"] > 0
    assert timing["pipeline_fps"] == pytest.approx(1000.0 / timing["frame_ms"]["mean"])
    assert timing["wall_s"] > 0 and timing["setup_ms"] >= 0
    # the detector is warmed inside setup, so a cold CUDA/JIT pass is not billed to frame 1
    assert timing["warmed_up"] is True and detectors[0].warmups == 1
    # tracker is reset after construction and again at teardown
    assert trackers[0].resets >= 2
    assert trackers[0].updates == 8


def test_warmup_detector_is_best_effort(harness):
    """A missing or exploding warmup must never fail a pair (only the timing gets noisier)."""

    class NoWarmup:
        pass

    class Boom:
        def warmup(self):
            raise RuntimeError("simulated cuda failure")

    class Fine:
        def __init__(self):
            self.n = 0

        def warmup(self):
            self.n += 1

    assert harness._warmup_detector(NoWarmup()) is False
    assert harness._warmup_detector(Boom()) is False
    fine = Fine()
    assert harness._warmup_detector(fine) is True and fine.n == 1


def test_gpu_memory_mb_is_optional_and_never_raises(harness, monkeypatch):
    """CUDA diagnostics must not force a torch import and must never fail a pair."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert harness.gpu_memory_mb() is None

    class Boom:
        class cuda:  # noqa: N801 - mimics the torch.cuda namespace
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated():
                raise RuntimeError("no CUDA context")

    monkeypatch.setitem(sys.modules, "torch", Boom)
    assert harness.gpu_memory_mb() is None


def test_run_pair_feeds_embedding_rows_to_tracker(harness, tmp_path):
    seq_dir = write_sequence(tmp_path, "MOT-01", n_frames=3)
    boxes = perfect_boxes(seq_dir)
    trackers = []
    embedder = FakeEmbedder(dim=8, skip=(1,))

    def tracker_factory(cfg, det):
        trk = FakeTracker()
        trackers.append(trk)
        return trk

    rec = harness.run_pair(
        harness.PairSpec(sequence="MOT-01", seq_dir=str(seq_dir), tracker="deepocsort.yaml"),
        harness.RunOptions(out_dir=str(tmp_path / "out"), embed=True),
        detector_factory=lambda cfg: FakeDetector(boxes),
        tracker_factory=tracker_factory,
        embedder_factory=lambda cfg, opts: embedder,
    )
    assert rec["ok"] is True and rec["embedder"]["mode"] == "fake"
    # 3 frames x 2 boxes, one box per frame has no feature -> filled, counted as missing
    assert rec["num_missing_features"] == 3
    assert embedder.calls == 3
    feats = trackers[0].last_feats
    assert feats is not None and feats.shape == (2, 8) and np.isfinite(feats).all()
    assert rec["timing"]["embedder_ms"]["n"] == 3


def test_run_pair_max_frames_and_subsample(harness, tmp_path):
    seq_dir = write_sequence(tmp_path, "MOT-01", n_frames=10)
    rec = harness.run_pair(
        harness.PairSpec(sequence="MOT-01", seq_dir=str(seq_dir), tracker="ocsort.yaml"),
        harness.RunOptions(out_dir=str(tmp_path / "out"), max_frames=4, subsample=2),
        detector_factory=lambda cfg: FakeDetector(perfect_boxes(seq_dir)),
        tracker_factory=lambda cfg, det: FakeTracker(),
    )
    # 10 frames, keep every 2nd -> 5, then cap at 4
    assert rec["num_frames"] == 4 and rec["result"]["num_frames"] == 4


def test_run_pair_saves_mot_format_tracks(harness, tmp_path):
    seq_dir = write_sequence(tmp_path, "MOT-01", n_frames=2, ids=(1,))
    out = tmp_path / "out"
    harness.run_pair(
        harness.PairSpec(sequence="MOT-01", seq_dir=str(seq_dir), tracker="bytetrack.yaml"),
        harness.RunOptions(out_dir=str(out), save_tracks=True),
        detector_factory=lambda cfg: FakeDetector(perfect_boxes(seq_dir)),
        tracker_factory=lambda cfg, det: FakeTracker(),
    )
    dump = out / "tracks" / "bytetrack__MOT-01.txt"
    assert dump.exists()
    lines = dump.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 and lines[0].split(",")[0] == "1"


def test_pair_worker_never_raises(harness, tmp_path):
    rec = harness.pair_worker(
        (harness.PairSpec(sequence="X", seq_dir=str(tmp_path / "missing"), tracker="botsort.yaml"), harness.RunOptions())
    )
    assert rec["ok"] is False and rec["pair_id"] == "botsort__X"
    assert "FileNotFoundError" in rec["error"] and "Traceback" in rec["traceback"]


# --------------------------------------------------------------------------- #
# parallel dispatch
# --------------------------------------------------------------------------- #
def stub_worker(payload):
    """Module-level (picklable) stand-in for :func:`eval_mot.pair_worker`."""
    spec, _opts = payload
    return {
        "ok": True,
        "pair_id": spec.pair_id,
        "tracker": spec.tracker,
        "sequence": spec.sequence,
        "timing": {"wall_s": 2.0},
        "metrics": {"HOTA": 0.5},
    }


def test_run_pairs_serial_meta_and_ordering(harness):
    specs = harness.enumerate_pairs(["/d/B", "/d/A"], ["botsort.yaml", "bytetrack.yaml"])

    records, meta = harness.run_pairs(specs, harness.RunOptions(), jobs=1, worker=stub_worker, quiet=True)
    assert meta["mode"] == "serial" and meta["jobs"] == 1
    assert meta["num_pairs"] == 4 and len(records) == 4
    assert meta["sum_pair_wall_s"] == pytest.approx(8.0)
    assert meta["speedup"] == pytest.approx(meta["sum_pair_wall_s"] / meta["wall_s"])
    # records come back in a deterministic (tracker, sequence) order regardless of completion order
    assert [(r["tracker"], r["sequence"]) for r in records] == sorted((r["tracker"], r["sequence"]) for r in records)


def test_run_pairs_parallel_pool_in_a_child_process(harness, tmp_path):
    """The spawn pool path end to end: real child processes, real pickling.

    ``PairSpec``/``RunOptions`` are pickled by reference, so the child must be able
    to import the harness under the same module name. Running a helper script that
    registers it before submitting proves the production ``python scripts/eval_mot.py``
    layout (where the module is ``__main__``/``__mp_main__``) really can fan out.
    """
    seq = write_sequence(tmp_path, "MOT-01", n_frames=3)
    helper = tmp_path / "pool_helper.py"
    helper.write_text(
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('eval_mot_harness', r'{SCRIPT}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['eval_mot_harness'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "\n"
        "def worker(payload):\n"
        "    spec_, _opts = payload\n"
        "    return {'ok': True, 'pair_id': spec_.pair_id, 'tracker': spec_.tracker,\n"
        "            'sequence': spec_.sequence, 'timing': {'wall_s': 1.0}, 'metrics': {'HOTA': 0.25}}\n"
        "\n"
        "def main():\n"
        f"    specs = mod.enumerate_pairs([r'{seq}'], ['botsort.yaml', 'bytetrack.yaml', 'ocsort.yaml'])\n"
        "    records, meta = mod.run_pairs(specs, mod.RunOptions(), jobs=2, worker=worker, quiet=True)\n"
        "    print('META', json.dumps(meta, sort_keys=True))\n"
        "    print('OK', sum(1 for r in records if r['ok']), 'OF', len(records))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(helper)], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert '"mode": "process-pool"' in proc.stdout and '"jobs": 2' in proc.stdout
    assert "OK 3 OF 3" in proc.stdout


# --------------------------------------------------------------------------- #
# aggregation + reports
# --------------------------------------------------------------------------- #
def test_aggregate_writes_reports_sorted_by_hota(harness, tmp_path):
    from tgtrack.evaluate import evaluate_mot

    gt, pred_perfect = simple_gt_pred()
    _gt2, pred_degraded = simple_gt_pred(drop_frames=(2, 3))
    rec_a = fake_record("aaa.yaml", evaluate_mot(gt, pred_perfect, return_state=True), ms=10.0, wall=2.0)
    rec_b = fake_record("bbb.yaml", evaluate_mot(gt, pred_degraded, return_state=True), ms=20.0, wall=1.0)
    bad = {"ok": False, "pair_id": "ccc__MOT-01", "tracker": "ccc.yaml", "sequence": "MOT-01", "error": "boom"}

    out = tmp_path / "eval"
    run_meta = harness.aggregate(
        [rec_b, rec_a, bad],
        out,
        opts=harness.RunOptions(),
        parallel={"jobs": 2, "mode": "process-pool", "wall_s": 3.0, "sum_pair_wall_s": 3.0, "speedup": 1.0, "cpu_count": 8},
        trackers=["aaa.yaml", "bbb.yaml"],
        sequences=["/d/MOT-01"],
        scaling=[{"jobs": 1, "pairs": 2, "wall_s": 3.0, "sum_pair_wall_s": 3.0, "speedup": 1.0}],
        argv=["eval_mot.py"],
    )
    # sorted by HOTA descending: perfect aaa first
    assert [r["tracker"] for r in run_meta["summary"]] == ["aaa.yaml", "bbb.yaml"]
    assert run_meta["summary"][0]["HOTA"] == pytest.approx(1.0)

    rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    assert rows[0]["tracker"] == "aaa.yaml" and float(rows[1]["HOTA"]) < 1.0
    pair_rows = list(csv.DictReader((out / "pairs.csv").open(encoding="utf-8")))
    assert len(pair_rows) == 2 and pair_rows[0]["tracker"] == "aaa.yaml"
    assert "ms_per_frame" in pair_rows[0] and float(pair_rows[0]["pipeline_fps"]) == pytest.approx(100.0)

    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "| tracker |" in md and "HOTA" in md
    assert "Parallel scaling" in md and "Failures" in md and "ccc__MOT-01: boom" in md

    assert (out / "pairs" / "aaa__MOT-01.json").exists()
    plan = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert plan["parallel"]["jobs"] == 2 and plan["env"]["cpu_count"] == os.cpu_count()
    assert [f["pair_id"] for f in plan["failures"]] == ["ccc__MOT-01"]
    # the per-pair JSON must keep the raw state (so combine_mot can be re-derived offline)
    payload = json.loads((out / "pairs" / "aaa__MOT-01.json").read_text(encoding="utf-8"))
    assert "state" in payload["result"] and "hota_tp" in payload["result"]["state"]


def test_aggregate_handles_empty_records(harness, tmp_path):
    out = tmp_path / "empty"
    run_meta = harness.aggregate([], out, opts=harness.RunOptions())
    assert run_meta["summary"] == [] and run_meta["pairs"] == []
    assert (out / "summary.md").exists() and (out / "summary.csv").exists()


def test_scaling_table(harness):
    rows = harness.scaling_table(
        [
            {"jobs": 1, "meta": {"jobs": 1, "num_pairs": 6, "wall_s": 12.0, "sum_pair_wall_s": 12.0, "speedup": 1.0}},
            {"jobs": 4, "meta": {"num_pairs": 6, "wall_s": 4.0, "sum_pair_wall_s": 12.0, "speedup": 3.0}},
        ]
    )
    assert [r["jobs"] for r in rows] == [1, 4]
    assert rows[1]["speedup"] == pytest.approx(3.0) and rows[1]["pairs"] == 6


# --------------------------------------------------------------------------- #
# CLI end to end
# --------------------------------------------------------------------------- #
def test_main_dry_run_writes_plan_without_data(harness, tmp_path):
    out = tmp_path / "plan"
    rc = harness.main(["--data-root", str(tmp_path / "absent"), "--out", str(out), "--trackers", "botsort.yaml,bytetrack.yaml", "--dry-run", "--quiet"])
    assert rc == 0
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["num_sequences"] == 0 and plan["num_pairs"] == 0
    assert plan["trackers"] == ["botsort.yaml", "bytetrack.yaml"]
    assert plan["warnings"] and "does not exist" in plan["warnings"][0]


def test_main_dry_run_enumerates_synthetic_pairs(harness, tmp_path):
    write_sequence(tmp_path / "seqs", "MOT-01", n_frames=4)
    write_sequence(tmp_path / "seqs", "MOT-02", n_frames=4)
    out = tmp_path / "plan"
    rc = harness.main(
        ["--data-root", str(tmp_path / "seqs"), "--out", str(out), "--trackers", "botsort,ocsort", "--limit-sequences", "1", "--dry-run", "--quiet"]
    )
    assert rc == 0
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["num_sequences"] == 1 and plan["num_pairs"] == 2
    assert [p["tracker"] for p in plan["pairs"]] == ["botsort.yaml", "ocsort.yaml"]
    assert plan["sequences"][0]["num_frames"] == 4 and plan["sequences"][0]["num_gt_ids"] == 2
    assert not (out / "pairs.csv").exists()


def test_main_without_sequences_reports_and_exits_nonzero(harness, tmp_path):
    out = tmp_path / "run"
    rc = harness.main(["--data-root", str(tmp_path / "absent"), "--out", str(out), "--trackers", "botsort.yaml", "--quiet"])
    assert rc == 1
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["summary"] == [] and meta["pairs"] == []


def test_dry_run_subprocess_does_not_import_torch(tmp_path):
    """A dry run must stay installable-and-runnable with no torch/ultralytics at all."""
    absent = tmp_path / "absent"
    out = tmp_path / "out"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{SCRIPT}')\n"
        "mod = importlib.util.module_from_spec(spec); sys.modules['m'] = mod; spec.loader.exec_module(mod)\n"
        f"rc = mod.main(['--data-root', r'{absent}', '--out', r'{out}', '--dry-run', '--quiet'])\n"
        "print('RC', rc)\n"
        "print('TORCH', 'torch' in sys.modules)\n"
        "print('ULTRALYTICS', 'ultralytics' in sys.modules)\n"
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "RC 0" in proc.stdout
    assert "TORCH False" in proc.stdout
    assert "ULTRALYTICS False" in proc.stdout
