"""Tests for :mod:`tgtrack.motion` — single-target motion model (task-1).

Covers the required behaviours: linear extrapolation, constant-acceleration
tracking, the *measured* nonlinear claim (IMM beats a plain KF on curved
paths), NSA score-adaptive measurement noise, OC-SORT direction cost,
Mahalanobis gating, rollback, NaN-safety and coast idempotence.

Run from the repo root::

    python -m pytest tests/test_motion.py -q

Thresholds below are not guesses: each one is derived from a measurement and
carries ~2x margin, so the assertions are meaningful but not flaky.  The
reference numbers are quoted in the assertion messages / comments.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # pragma: no cover - bootstrap
    sys.path.insert(0, str(_SRC))

from tgtrack.config import CostCfg, MotionCfg  # noqa: E402
from tgtrack.motion import MotionModel, gate_radius  # noqa: E402

BOX_W = 120.0
BOX_H = 240.0
CX0, CY0 = 400.0, 300.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_box(cx: float, cy: float, w: float = BOX_W, h: float = BOX_H) -> np.ndarray:
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32
    )


def centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


def line_track(n: int, speed: float) -> np.ndarray:
    """Constant-velocity walk, ``speed`` px/frame along +x."""
    return np.stack([make_box(100.0 + speed * t, CY0) for t in range(n)]).astype(np.float32)


def accel_track(n: int, v0: float, a: float) -> np.ndarray:
    """Nonlinear (parabolic) path: position integrates v0 + a*t."""
    xs, x, v = [], 100.0, v0
    for _ in range(n):
        xs.append(x)
        x += v
        v += a
    return np.stack([make_box(x, CY0) for x in xs]).astype(np.float32)


def circle_track(n: int, r: float, speed: float) -> np.ndarray:
    """Constant-speed circular path — constant curvature, the CT model's home turf."""
    w = speed / r
    return np.stack(
        [make_box(CX0 + r * math.cos(w * t), CY0 + r * math.sin(w * t)) for t in range(n)]
    ).astype(np.float32)


def figure8_track(n: int, r: float, speed: float) -> np.ndarray:
    """Figure-eight — curvature changes sign, so no single CT rate fits it."""
    w = speed / r
    return np.stack(
        [
            make_box(CX0 + r * math.sin(w * t), CY0 + r * math.sin(2.0 * w * t))
            for t in range(n)
        ]
    ).astype(np.float32)


def centre_errors(boxes: np.ndarray, model: str, score: float = 1.0) -> np.ndarray:
    """One-step-ahead centre error at every frame after initiation."""
    model_obj = MotionModel(MotionCfg(model=model))
    model_obj.initiate(boxes[0], score)
    errs = []
    for i in range(1, len(boxes)):
        pred = model_obj.predict()
        errs.append(float(np.linalg.norm(centre(pred) - centre(boxes[i]))))
        model_obj.update(boxes[i], score)
    return np.asarray(errs, dtype=np.float64)


def warm_model(model: str, n: int = 30, speed: float = 4.0) -> MotionModel:
    """A model that has settled onto a clean constant-velocity track."""
    model_obj = MotionModel(MotionCfg(model=model))
    model_obj.initiate(make_box(100.0, CY0), 1.0)
    for i in range(1, n):
        model_obj.predict()
        model_obj.update(make_box(100.0 + speed * i, CY0), 1.0)
    return model_obj


# --------------------------------------------------------------------------- #
# 0. interface contract
# --------------------------------------------------------------------------- #
def test_interface_contract():
    m = MotionModel(MotionCfg(model="imm"), CostCfg())
    m.initiate(make_box(CX0, CY0), 1.0)

    pred = m.predict()
    assert isinstance(pred, np.ndarray) and pred.shape == (4,)
    assert pred.dtype == np.float32
    assert np.all(np.isfinite(pred))
    assert pred[2] > pred[0] and pred[3] > pred[1]

    vel = m.velocity()
    assert vel.shape == (4,) and np.all(np.isfinite(vel))

    hyps = m.multi_hypothesis()
    assert isinstance(hyps, list) and len(hyps) >= 1
    for box, weight in hyps:
        assert isinstance(box, np.ndarray) and box.shape == (4,)
        assert np.all(np.isfinite(box))
        assert isinstance(weight, float) and 0.0 <= weight <= 1.0

    hist = m.history()
    assert isinstance(hist, list) and len(hist) >= 1
    assert all(isinstance(b, np.ndarray) and b.shape == (4,) for b in hist)

    for name in ("initiate", "predict", "update", "coast", "rollback", "reset",
                 "direction_cost", "mahalanobis", "velocity", "history",
                 "multi_hypothesis"):
        assert callable(getattr(m, name))

    m.reset(None)
    assert m.history() == []
    fresh = m.predict()
    assert fresh.shape == (4,) and np.all(np.isfinite(fresh))

    m.reset(make_box(10.0, 10.0))
    assert len(m.history()) == 1

    # never-initialised model must still answer public queries safely
    bare = MotionModel(MotionCfg(model="kf"))
    assert bare.history() == []
    assert np.all(np.isfinite(bare.velocity()))
    assert bare.direction_cost(np.stack([make_box(CX0, CY0)])).shape == (1,)
    assert np.all(np.isfinite(bare.mahalanobis(np.stack([make_box(CX0, CY0)]))))
    assert np.all(np.isfinite(bare.coast()))


def test_helper_gate_radius_grows_with_speed():
    slow = gate_radius(np.array([0.0, 0.0, 0.0, 0.0]))
    fast = gate_radius(np.array([8.0, 0.0, 0.0, 0.0]))
    faster = gate_radius(np.array([40.0, 0.0, 0.0, 0.0]))
    assert slow < fast < faster
    assert slow == pytest.approx(CostCfg().gate_base)
    assert np.isfinite(gate_radius(np.array([np.inf, 0.0, 0.0, 0.0])))

    m = warm_model("kf")
    assert m.gate_radius() > 0.0 and np.isfinite(m.gate_radius())
    assert m.gate_radius(2.0) > m.gate_radius(1.0)


# --------------------------------------------------------------------------- #
# 1. linear extrapolation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["kf", "nsa_kf", "imm"])
def test_linear_extrapolation_is_accurate_after_warmup(model):
    """On clean CV motion the filter must lock on, not lag.  Measured: mean
    error 0.0000 px over the last 50 frames, velocity exactly (4, 0, 0, 0)."""
    boxes = line_track(120, 4.0)
    m = MotionModel(MotionCfg(model=model))
    m.initiate(boxes[0], 1.0)
    errs = []
    for i in range(1, len(boxes)):
        errs.append(float(np.linalg.norm(centre(m.predict()) - centre(boxes[i]))))
        m.update(boxes[i], 1.0)
    errs = np.asarray(errs)

    assert errs[-50:].mean() < 0.05, f"{model}: steady-state error {errs[-50:].mean()}"
    assert errs[-50:].max() < 0.2, f"{model}: worst steady-state error {errs[-50:].max()}"

    vel = m.velocity()
    assert vel[0] == pytest.approx(4.0, abs=0.05), f"{model} vx={vel[0]}"
    assert abs(vel[1]) < 0.05 and abs(vel[2]) < 1e-3 and abs(vel[3]) < 1e-3

    # a fast target (40 px/frame, i.e. 1/6 box width per frame) must also settle
    fast = line_track(200, 40.0)
    m_fast = MotionModel(MotionCfg(model=model))
    m_fast.initiate(fast[0], 1.0)
    e_fast = []
    for i in range(1, len(fast)):
        e_fast.append(float(np.linalg.norm(centre(m_fast.predict()) - centre(fast[i]))))
        m_fast.update(fast[i], 1.0)
    e_fast = np.asarray(e_fast)
    assert e_fast[-50:].mean() < 0.5, f"{model}: fast-track error {e_fast[-50:].mean()}"
    assert m_fast.velocity()[0] == pytest.approx(40.0, abs=0.5)


def test_prediction_is_ahead_not_behind_for_constant_velocity():
    """The box handed to the associator is for the NEXT frame, so at a steady
    state the predicted centre must sit one step ahead of the last observation."""
    boxes = line_track(60, 6.0)
    m = MotionModel(MotionCfg(model="kf"))
    m.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        m.predict()
        m.update(boxes[i], 1.0)
    pred = m.predict()
    lead = float(centre(pred)[0] - centre(boxes[-1])[0])
    assert lead == pytest.approx(6.0, abs=0.3), f"lead={lead}"


# --------------------------------------------------------------------------- #
# 2. constant acceleration
# --------------------------------------------------------------------------- #
def test_constant_acceleration_imm_beats_kf():
    """Parabolic path (0.5 px/frame^2).  Measured mean error: kf 6.12 px,
    imm 2.36 px (ratio 0.39); imm last-50 2.53 px."""
    boxes = accel_track(160, 0.0, 0.5)
    e_kf = centre_errors(boxes, "kf")
    e_imm = centre_errors(boxes, "imm")

    assert e_imm.mean() < 0.6 * e_kf.mean(), f"imm {e_imm.mean():.3f} vs kf {e_kf.mean():.3f}"
    assert e_imm[-50:].mean() < 3.5, f"imm steady-state {e_imm[-50:].mean():.3f}"
    assert np.all(np.isfinite(e_imm))


# --------------------------------------------------------------------------- #
# 3. the nonlinear claim: IMM beats a plain KF on curved motion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "radius,speed,kf_mean,imm_mean",
    [
        (40.0, 4.0, 4.085, 3.062),
        (80.0, 8.0, 8.169, 6.145),
    ],
)
def test_imm_beats_kf_on_circular_motion(radius, speed, kf_mean, imm_mean):
    """Measured headline result on 160-frame circles (radius / px-per-frame):

        r=40 v=4 : kf 4.085  nsa_kf 4.085  imm 3.062  -> imm/kf = 0.750
        r=80 v=8 : kf 8.169  nsa_kf 8.169  imm 6.145  -> imm/kf = 0.752

    The <=0.9 bound is deliberately looser than the measurement so the test
    guards the *phenomenon* (IMM's manoeuvre model helps) rather than a
    brittle exact value.
    """
    boxes = circle_track(160, radius, speed)
    e_kf = centre_errors(boxes, "kf")
    e_nsa = centre_errors(boxes, "nsa_kf")
    e_imm = centre_errors(boxes, "imm")

    assert e_kf.mean() == pytest.approx(kf_mean, rel=0.25)
    assert e_imm.mean() == pytest.approx(imm_mean, rel=0.25)
    assert e_imm.mean() < 0.9 * e_kf.mean(), (
        f"r={radius} v={speed}: imm {e_imm.mean():.3f} not < 0.9*kf {e_kf.mean():.3f}"
    )
    assert e_imm.mean() < e_kf.mean()

    # score 1.0 must make the NSA variant identical to the plain KF
    assert e_nsa.mean() == pytest.approx(e_kf.mean(), rel=1e-6)

    # stability guard: a model that alternates between good and bad frames can
    # still show a low mean, so every 40-frame segment must beat the KF.
    for start in range(0, len(e_imm) - 39, 40):
        seg_imm = e_imm[start : start + 40].mean()
        seg_kf = e_kf[start : start + 40].mean()
        assert seg_imm < seg_kf, f"segment @{start}: imm {seg_imm:.3f} >= kf {seg_kf:.3f}"


def test_imm_beats_kf_on_figure_eight():
    """Curvature flips sign — measured imm/kf = 0.905, i.e. still strictly better."""
    boxes = figure8_track(160, 60.0, 5.0)
    e_kf = centre_errors(boxes, "kf")
    e_imm = centre_errors(boxes, "imm")
    assert e_imm.mean() < e_kf.mean(), f"fig8 imm {e_imm.mean():.3f} vs kf {e_kf.mean():.3f}"
    assert e_imm.mean() < 0.97 * e_kf.mean()


def test_imm_weights_are_normalised_and_favour_a_single_manoeuvre_model():
    boxes = circle_track(160, 40.0, 4.0)
    m = MotionModel(MotionCfg(model="imm"))
    m.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        m.predict()
        m.update(boxes[i], 1.0)
    hyps = m.multi_hypothesis()
    assert len(hyps) == 3
    weights = np.array([w for _, w in hyps])
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0.0).all()
    # on a constant-curvature path the turn model must not be the loser
    assert weights.max() > 1.0 / 3.0


# --------------------------------------------------------------------------- #
# 4. NSA: score-adaptive measurement noise
# --------------------------------------------------------------------------- #
def _filtered_box(m: MotionModel) -> np.ndarray:
    """Post-update blended box without advancing the filter.

    Uses the per-component ``box`` (set by ``update``/``predict``) so it works
    for both single-model and IMM configurations.  Note ``_blend()`` returns
    ``(mean, cov)``.
    """
    if len(m._comps) == 1:
        return np.asarray(m._comps[0].box, dtype=np.float64)
    w = m._mode_weights()
    return np.asarray(
        (w[:, None] * np.stack([c.box for c in m._comps])).sum(axis=0), dtype=np.float64
    )


def _outlier_shift(score: float, model: str = "nsa_kf") -> float:
    m = MotionModel(MotionCfg(model=model))
    m.initiate(make_box(100.0, CY0), 1.0)
    for i in range(1, 30):
        m.predict()
        m.update(make_box(100.0 + 4.0 * i, CY0), 1.0)
    m.predict()
    before = centre(_filtered_box(m))
    m.update(make_box(400.0, CY0), score)  # implausible 300 px jump
    return float(np.linalg.norm(centre(_filtered_box(m)) - before))


def test_nsa_kf_low_score_resists_outliers():
    """Measured centre shift after a 300 px outlier jump:
    score 1.0 -> 118.96 px, 0.5 -> 58.97 px, 0.15 -> 7.56 px (monotone)."""
    s1 = _outlier_shift(1.0)
    s05 = _outlier_shift(0.5)
    s015 = _outlier_shift(0.15)

    assert s015 < s05 < s1, f"shifts not monotone: {s015:.2f} {s05:.2f} {s1:.2f}"
    assert s015 < 0.25 * s1, f"low-score shift {s015:.2f} vs full-score {s1:.2f}"


def test_nsa_kf_inflates_covariance_for_low_score():
    """Low confidence -> larger measurement noise -> larger posterior covariance."""

    def prepared() -> MotionModel:
        m = MotionModel(MotionCfg(model="nsa_kf"))
        m.initiate(make_box(100.0, CY0), 1.0)
        for i in range(1, 30):
            m.predict()
            m.update(make_box(100.0 + 4.0 * i, CY0), 1.0)
        m.predict()
        return m

    high, low = prepared(), prepared()
    box = make_box(232.0, CY0 + 6.0)
    high.update(box, 1.0)
    low.update(box, 0.15)
    assert np.trace(low._comps[0].cov) > np.trace(high._comps[0].cov)


def test_plain_kf_ignores_score():
    """The non-adaptive models must be completely insensitive to `score`."""

    def prepared() -> MotionModel:
        m = MotionModel(MotionCfg(model="kf"))
        m.initiate(make_box(100.0, CY0), 1.0)
        for i in range(1, 30):
            m.predict()
            m.update(make_box(100.0 + 4.0 * i, CY0), 1.0)
        m.predict()
        return m

    a, b = prepared(), prepared()
    box = make_box(400.0, CY0)
    a.update(box, 1.0)
    b.update(box, 0.15)
    assert np.allclose(_filtered_box(a), _filtered_box(b), atol=1e-4)


# --------------------------------------------------------------------------- #
# 5. direction cost (OC-SORT OCM)
# --------------------------------------------------------------------------- #
def test_direction_cost_penalises_reversal():
    """Measured: forward 0.0, perpendicular 0.1, reversal 0.2 (= direction_weight),
    and all zeros while history is too short."""
    m = warm_model("kf", n=12, speed=4.0)
    last = centre(m.history()[-1])
    dets = np.stack(
        [
            make_box(last[0] + 4.0, last[1]),  # forward
            make_box(last[0] - 4.0, last[1]),  # reversal
            make_box(last[0], last[1] + 4.0),  # perpendicular
        ]
    ).astype(np.float32)

    cost = m.direction_cost(dets)
    assert cost.shape == (3,)
    assert np.all(np.isfinite(cost)) and np.all(cost >= 0.0)
    assert cost[0] == pytest.approx(0.0, abs=1e-6)
    assert cost[1] == pytest.approx(MotionCfg().direction_weight, abs=1e-6)
    assert cost[0] < cost[2] < cost[1]

    # too little history -> no opinion
    fresh = MotionModel(MotionCfg(model="kf"))
    assert np.allclose(fresh.direction_cost(dets[:1]), 0.0)
    one = MotionModel(MotionCfg(model="kf"))
    one.initiate(make_box(100.0, CY0), 1.0)
    assert np.allclose(one.direction_cost(dets), 0.0)

    # empty input
    assert m.direction_cost(np.zeros((0, 4), dtype=np.float32)).shape == (0,)


def test_direction_cost_is_scaled_by_direction_weight():
    """``MotionCfg.direction_weight`` must scale the reversal penalty linearly."""
    for w in (0.0, 0.3):
        m = MotionModel(MotionCfg(model="kf", direction_weight=w))
        m.initiate(make_box(100.0, CY0), 1.0)
        for i in range(1, 12):
            m.predict()
            m.update(make_box(100.0 + 4.0 * i, CY0), 1.0)
        last = centre(m.history()[-1])
        dets = np.stack([make_box(last[0] - 4.0, last[1])]).astype(np.float32)
        assert m.direction_cost(dets)[0] == pytest.approx(w, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. Mahalanobis gating
# --------------------------------------------------------------------------- #
def test_mahalanobis_gates_far_detection():
    """Measured on a settled CV track: near (2 px off) ~0.026, far (300 px off)
    ~161, against motion_gate 9.4877 (chi2, 4 dof, 0.95)."""
    m = warm_model("kf", n=30, speed=4.0)
    pred = m.predict()
    c = centre(pred)

    near = m.mahalanobis(np.stack([make_box(c[0] + 2.0, c[1] + 2.0)]).astype(np.float32))
    far = m.mahalanobis(np.stack([make_box(c[0] + 300.0, c[1] + 40.0)]).astype(np.float32))

    gate = CostCfg().motion_gate
    assert near.shape == (1,) and far.shape == (1,)
    assert near[0] < gate < far[0], f"near {near[0]:.3f} gate {gate} far {far[0]:.3f}"

    multi = m.mahalanobis(
        np.stack(
            [
                make_box(c[0] + 2.0, c[1]),
                make_box(c[0] - 2.0, c[1]),
                make_box(c[0] + 400.0, c[1]),
            ]
        ).astype(np.float32)
    )
    assert multi.shape == (3,) and np.all(np.isfinite(multi))
    assert (multi[:2] < gate).all() and multi[2] > gate


def test_mahalanobis_handles_empty_and_degenerate_input():
    m = warm_model("kf", n=20)
    assert m.mahalanobis(np.zeros((0, 4), dtype=np.float32)).shape == (0,)
    bad = np.array(
        [[10.0, 10.0, 10.0, 10.0], [np.nan, 0.0, 100.0, 200.0]], dtype=np.float32
    )
    out = m.mahalanobis(bad)
    assert out.shape == (2,) and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# 7. rollback
# --------------------------------------------------------------------------- #
def test_rollback_restores_pre_occlusion_state():
    """Measured: 5 frames of garbage drag the prediction 865 px off; after
    ``rollback(5)`` the restored state matches the pre-occlusion prediction to
    0.000 px (rolling back 5 *predicted frames* = the last 5 history entries).

    Note the accounting: components push history once per ``predict()``, so
    ``rollback(d)`` restores ``hist_mean[-1-d]``.  We therefore do NOT call
    ``predict()`` after the loop — that extra push would move the clean state
    one entry further back.
    """
    m = warm_model("kf", n=40, speed=4.0)
    good = m.predict().copy()

    for _ in range(5):  # 5 frames of garbage "detections"
        m.predict()
        m.update(make_box(900.0, 700.0, w=200.0, h=400.0), 0.4)
    poisoned = m._last_pred_box.copy()
    assert np.linalg.norm(centre(poisoned) - centre(good)) > 100.0, "not poisoned?"

    m.rollback(5)
    restored = m._last_pred_box.copy()
    err = float(np.linalg.norm(centre(restored) - centre(good)))
    assert err < 1.0, f"rollback err {err:.3f}"
    assert np.all(np.isfinite(restored)) and restored[3] > restored[1]

    # the next prediction is then exactly one constant-velocity step on
    assert np.linalg.norm(centre(m.predict()) - centre(good)) < 8.0

    hyp = m.multi_hypothesis()
    assert len(hyp) >= 1 and np.all(np.isfinite(hyp[0][0]))


def test_rollback_is_bounded_and_safe():
    m = warm_model("kf", n=20)
    m.predict()
    before = m._last_pred_box.copy()
    m.rollback(0)  # no-op: must leave the state completely untouched
    assert np.allclose(m._last_pred_box, before)
    assert np.all(np.isfinite(m.predict()))

    m.rollback(10**6)  # deeper than history: must clamp, not raise
    after = m.predict()
    assert after.shape == (4,) and np.all(np.isfinite(after))
    assert after[2] > after[0] and after[3] > after[1]

    fresh = MotionModel(MotionCfg(model="imm"))
    fresh.rollback(3)  # nothing to roll back
    assert np.all(np.isfinite(fresh.predict()))


def test_rollback_depth_controls_how_far_back_we_go():
    """Rolling back further must land earlier in the track (monotone), and each
    level must stay finite.  Verified: x-centre decreases with depth."""

    def tracked() -> MotionModel:
        m = MotionModel(MotionCfg(model="kf"))
        m.initiate(make_box(100.0, CY0), 1.0)
        for i in range(1, 40):
            m.predict()
            m.update(make_box(100.0 + 4.0 * i, CY0), 1.0)
        m.predict()
        return m

    xs = {}
    for depth in (0, 3, 8):
        m = tracked()
        m.rollback(depth)
        pred = m.predict()
        assert np.all(np.isfinite(pred))
        xs[depth] = float(centre(pred)[0])

    assert xs[0] > xs[3] > xs[8], f"depth did not move state back: {xs}"
    # a 3-frame rollback should sit ~3 CV steps (~12 px) behind the no-op
    assert xs[0] - xs[3] == pytest.approx(12.0, abs=4.0)


# --------------------------------------------------------------------------- #
# 8. NaN / degenerate-input safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["kf", "nsa_kf", "imm"])
def test_degenerate_boxes_never_produce_nan(model):
    m = MotionModel(MotionCfg(model=model))
    m.initiate(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32), 0.0)
    assert np.all(np.isfinite(m.predict()))

    for bad in (
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # zero area
        np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float32),  # inverted
        np.array([np.inf, 0.0, 10.0, 10.0], dtype=np.float32),
        # very large coordinates (still float32-representable widths)
        np.array([1e6, 1e6, 1e6 + 200.0, 1e6 + 400.0], dtype=np.float32),
        np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
    ):
        m.update(bad, 0.9)
        pred = m.predict()
        assert np.all(np.isfinite(pred)), f"{model}: non-finite after {bad}"
        assert pred[2] > pred[0] and pred[3] > pred[1]
        coasted = m.coast()
        assert np.all(np.isfinite(coasted))
        assert coasted[3] > coasted[1]
        assert np.all(np.isfinite(m.velocity()))
        assert np.all(np.isfinite(m.mahalanobis(np.stack([bad]))))


def test_extreme_scores_are_safe():
    m = MotionModel(MotionCfg(model="nsa_kf"))
    m.initiate(make_box(CX0, CY0), 1.0)
    for score in (0.0, -1.0, 100.0, float("nan")):
        m.update(make_box(CX0 + 50.0, CY0), score)
        assert np.all(np.isfinite(m.predict()))
        assert np.all(np.isfinite(m.velocity()))


# --------------------------------------------------------------------------- #
# 9. coast
# --------------------------------------------------------------------------- #
def test_coast_is_idempotent_within_a_frame():
    """Repeated coast() calls in the same frame must be a no-op after the first,
    and must damp velocity rather than integrate it twice."""
    m = warm_model("imm", n=30, speed=4.0)
    m.predict()
    c1 = m.coast()
    v1 = m.speed()
    c2 = m.coast()
    c3 = m.coast()

    assert np.allclose(c1, c2, atol=1e-5), f"coast not idempotent: {c1} vs {c2}"
    assert np.allclose(c2, c3, atol=1e-5)
    assert m.speed() <= v1 + 1e-6
    for c in (c1, c2, c3):
        assert np.all(np.isfinite(c)) and c[3] > c[1] and c[2] > c[0]


def test_coast_damps_velocity_and_advances_without_detections():
    boxes = line_track(40, 8.0)
    m = MotionModel(MotionCfg(model="kf"))
    m.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        m.predict()
        m.update(boxes[i], 1.0)

    v_before = m.speed()
    assert v_before > 1.0
    last = m.predict()
    coasted = m.coast()
    v_after = m.speed()
    assert v_after < v_before, f"coast did not damp: {v_before} -> {v_after}"
    # damping factor is coast_velocity_damp per coast
    assert v_after == pytest.approx(v_before * MotionCfg().coast_velocity_damp, rel=0.05)
    # the box advances forward but only modestly, and stays a valid box
    assert centre(coasted)[0] >= centre(last)[0] - 1e-3
    assert coasted[2] > coasted[0] and coasted[3] > coasted[1]

    # a long coast must stay bounded and finite
    for _ in range(200):
        c = m.coast()
    assert np.all(np.isfinite(c))
    assert c[3] > c[1] > 0 and c[2] > c[0]


def test_coast_without_prior_predict_still_moves_forward():
    m = warm_model("kf", n=20, speed=5.0)
    before = _filtered_box(m)
    c = m.coast()
    assert np.all(np.isfinite(c))
    assert centre(c)[0] >= centre(before)[0] - 1e-3


# --------------------------------------------------------------------------- #
# 10. multi-hypothesis + history bookkeeping
# --------------------------------------------------------------------------- #
def test_multi_hypothesis_shape_and_weights():
    kf = warm_model("kf", n=20)
    hyps_kf = kf.multi_hypothesis()
    assert len(hyps_kf) == 1
    assert hyps_kf[0][1] == pytest.approx(1.0, abs=1e-6)

    imm = warm_model("imm", n=20)
    hyps = imm.multi_hypothesis()
    assert len(hyps) == 3
    weights = np.array([w for _, w in hyps])
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0.0).all()
    boxes = np.stack([b for b, _ in hyps])
    assert np.all(np.isfinite(boxes))
    assert (boxes[:, 3] > boxes[:, 1]).all() and (boxes[:, 2] > boxes[:, 0]).all()


def test_history_is_oldest_to_newest_and_copied():
    boxes = line_track(10, 4.0)
    m = MotionModel(MotionCfg(model="kf"))
    m.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        m.predict()
        m.update(boxes[i], 1.0)
    hist = m.history()
    assert len(hist) == len(boxes)
    assert np.allclose(hist[0], boxes[0]) and np.allclose(hist[-1], boxes[-1])
    assert all(np.all(np.isfinite(b)) for b in hist)

    hist[0][:] = -1.0  # mutating the returned list must not touch the model
    assert np.allclose(m.history()[0], boxes[0])


def test_state_dict_is_finite_for_every_model():
    """``state_dict()`` goes through the blended state, so it must work for the
    single-model filters too (their mode weights fall back to 1.0)."""
    for model in ("kf", "nsa_kf", "imm"):
        boxes = line_track(15, 4.0)
        m = MotionModel(MotionCfg(model=model))
        m.initiate(boxes[0], 1.0)
        for i in range(1, len(boxes)):
            m.predict()
            m.update(boxes[i], 0.8)
        m.predict()
        d = m.state_dict()
        assert isinstance(d, dict) and d["model"] == model
        assert np.all(np.isfinite(np.asarray(d["box"], dtype=np.float64)))
        assert np.all(np.isfinite(np.asarray(d["velocity"], dtype=np.float64)))
        assert np.isfinite(float(d["gate_radius"]))

    for model in ("kf", "nsa_kf", "imm"):
        bare = MotionModel(MotionCfg(model=model))
        assert np.all(np.isfinite(np.asarray(bare.state_dict()["box"], dtype=np.float64)))


def test_predict_advances_exactly_one_frame_per_call():
    """``predict()`` is the per-frame advance (idempotence is ``coast()``'s job):
    two calls in a row predict two frames ahead."""
    m = warm_model("kf", n=30, speed=4.0)
    first = m.predict()
    second = m.predict()
    assert centre(second)[0] - centre(first)[0] == pytest.approx(4.0, abs=0.5)

    # ...and coast() is the idempotent one
    c1 = m.coast()
    c2 = m.coast()
    assert np.allclose(c1, c2, atol=1e-5)


def test_history_is_bounded_by_max_history():
    m = MotionModel(MotionCfg(model="imm"), max_history=16)
    m.initiate(make_box(100.0, CY0), 1.0)
    for i in range(1, 80):
        m.predict()
        m.update(make_box(100.0 + 2.0 * i, CY0), 1.0)
    assert len(m.history()) <= 16
    assert np.all(np.isfinite(m.predict()))
    m.rollback(5)
    assert np.all(np.isfinite(m.predict()))

# --------------------------------------------------------------------------- #
# 9. shape prior (E014): a partial view must not telescope the predicted box
# --------------------------------------------------------------------------- #
def _shrinking_track(n: int = 8, w0: float = 170.0, shrink: float = 0.93) -> np.ndarray:
    """Boxes that get progressively NARROWER, like half-occluded detections.

    This is the trace shape that broke the filter on MOT17-02 under the
    upper-body mask: measured width 176 -> 165 -> 150 px while the filter's
    aspect velocity integrated into a predicted width of ~0 px.
    """
    return np.stack([make_box(100.0 + 6.0 * t, CY0, w=w0 * shrink**t) for t in range(n)]).astype(np.float32)


def test_shape_prior_stops_the_predicted_width_from_collapsing():
    boxes = _shrinking_track()
    on = MotionModel(MotionCfg(model="kf", shape_prior=True))
    on.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        on.predict()
        on.update(boxes[i], 1.0)
    ref_aspect = (boxes[-1][2] - boxes[-1][0]) / (boxes[-1][3] - boxes[-1][1])
    for _ in range(12):
        pred = on.predict()
    w = float(pred[2] - pred[0]); h = float(pred[3] - pred[1])
    assert w > 1.0, f"predicted width collapsed to {w:.3f} px"
    assert w / h >= 0.6 * ref_aspect - 1e-3, f"aspect {w / h:.3f} below the 0.6x prior of {ref_aspect:.3f}"

    # the prior is what does the work: without it the same trace telescopes further
    off = MotionModel(MotionCfg(model="kf", shape_prior=False))
    off.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        off.predict()
        off.update(boxes[i], 1.0)
    for _ in range(12):
        pred_off = off.predict()
    w_off = float(pred_off[2] - pred_off[0])
    assert w_off < w, f"prior should widen the prediction, got off={w_off:.2f} on={w:.2f}"


def test_shape_prior_allows_a_genuine_approach():
    """A person walking toward the camera grows every frame; the prior must not
    clip that (the reference updates on every accepted observation)."""
    m = MotionModel(MotionCfg(model="kf", shape_prior=True))
    boxes = [make_box(100.0 + 4.0 * t, CY0, w=60.0 * 1.06**t, h=BOX_H * 1.06**t) for t in range(24)]
    m.initiate(boxes[0], 1.0)
    for i in range(1, len(boxes)):
        m.predict()
        m.update(boxes[i], 1.0)
    pred = m.predict()
    last_w = float(boxes[-1][2] - boxes[-1][0])
    got_w = float(pred[2] - pred[0])
    assert abs(got_w - last_w) / last_w < 0.25, f"approach clipped: {got_w:.1f} vs {last_w:.1f}"


def test_aspect_velocity_is_clamped_and_aspect_stays_positive():
    cfg = MotionCfg(model="kf", shape_prior=True, aspect_vel_max=0.02)
    m = MotionModel(cfg)
    m.initiate(make_box(100.0, CY0), 1.0)
    for i in range(1, 10):
        m.predict()
        m.update(make_box(100.0 + 3.0 * i, CY0, w=BOX_W * 0.85**i), 1.0)
    for c in m._comps:
        assert abs(float(c.mean[6])) <= 0.02 + 1e-9
        assert float(c.mean[2]) > 0.0
    assert np.all(np.isfinite(m.predict()))
