"""Motion / prediction model for the single clicked target.

This is the motion half of the *target layer* (the novel part of the system).
It is deliberately decoupled from the all-person MOT layer and is pure NumPy
(no torch), so it can run inside the association inner loop thousands of times
per frame budget.

Three selectable models (``MotionCfg.model``):

``kf``
    One constant-velocity Kalman filter on the SORT/DeepSORT 8-D state
    ``[cx, cy, a, h, vcx, vcy, va, vh]`` with
    ``std_weight_position`` / ``std_weight_velocity`` process noise.

``nsa_kf``
    Same filter, but the **measurement** noise is scaled by the detection
    confidence (NSA Kalman, "Noise-Scaled Adaptive"): a low-confidence box is
    trusted less, which widens the innovation covariance and therefore the
    Kalman gain's covariance reduction.  With ``nsa_adaptive`` the scale is
    ``1 / clip(score, eps, 1)`` so ``score -> 0`` blows the measurement noise
    up; with it disabled the classic ``1 / (1 - score)`` form is available.

``imm``
    Interacting Multiple Model estimator over ``MotionCfg.imm_models``.  Every
    model owns its own Kalman state:

    ``cv``  8-D constant velocity,
    ``ca``  12-D constant acceleration (augmented with ``[acx, acy, aa, ah]``),
    ``ct``  9-D constant turn-rate (augmented with yaw rate ``w``, image plane);
            propagated through the exact nonlinear turn transition and
            linearised by finite differences (an EKF).

    The models are mixed per ``imm_stay_prob`` (a sticky diagonal Markov
    chain), each is updated with the same measurement, and the posterior model
    probabilities are the normalised measurement likelihoods.
    ``multi_hypothesis()`` exposes the per-model boxes + weights (for a
    multi-hypothesis association stage); ``predict()`` returns the
    weight-averaged box.

Beyond prediction the module supplies the pieces the target tracker's
association stage needs:

* :func:`gate_radius` — adaptive spatial gate from the current speed,
* :meth:`MotionModel.mahalanobis` — gating distances,
* :meth:`MotionModel.direction_cost` — OC-SORT OCM direction consistency,
* :meth:`MotionModel.rollback` — FastTracker-style Kalman rollback.

Numerical safety: every public output is ``float32`` and finite, degenerate or
zero-size boxes never raise, sizes are clamped positive.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .config import CostCfg, MotionCfg

__all__ = ["MotionModel", "gate_radius", "box_xyxy_to_state", "state_to_xyxy"]

#: Kalman state vector layout (SORT / DeepSORT convention).
#: ``cx``/``cy`` are CENTRE coordinates NORMALISED BY THE BOX HEIGHT (that is
#: what SORT/DeepSORT store: ``h`` is the only absolute length in the state,
#: everything else is relative to it), ``a = w / h`` is the aspect ratio and
#: ``h`` is the height in px.  Velocities are per frame in the same units.
STATE_NAMES = ("cx", "cy", "a", "h", "vcx", "vcy", "va", "vh")
_NDIM_BASE = 8
#: Absolute std used for the aspect-ratio component (dimensionless, so it is
#: not scaled by the box height like the centre/height components).
_ASPECT_STD = 1e-2
#: Degenerate boxes are clamped to this side length (px) so ratios stay finite.
_MIN_SIZE = 1e-3
_EPS = 1e-9
#: Covariance floor for position-like state components.
_POS_FLOOR = 1e-8
#: Covariance floor for velocity-like state components.
_VEL_FLOOR = 1e-10
#: Hard bound on a single coasted step's centre displacement (px).
_MAX_STEP = 1e4


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _finite(x: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
    """Return ``x`` as a finite float array (NaN/inf -> 0)."""
    a = np.asarray(x, dtype=np.float64)
    if a.size:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.astype(dtype)


def box_xyxy_to_state(box: np.ndarray) -> np.ndarray:
    """``xyxy`` -> ``[cx/h, cy/h, a, h]`` with positive, finite ``h``/``a``.

    The returned 4-vector is exactly the measurement vector (see
    :func:`_measure`), which is what keeps the filter's state space and
    measurement space consistent.
    """
    b = np.asarray(box, dtype=np.float64).reshape(4)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    x0, y0, x1, y1 = b
    w = max(abs(float(x1 - x0)), _MIN_SIZE)
    h = max(abs(float(y1 - y0)), _MIN_SIZE)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return np.array([cx / h, cy / h, w / h, h], dtype=np.float32)


def state_to_xyxy(cx: float, cy: float, a: float, h: float) -> np.ndarray:
    """``[cx/h, cy/h, a, h]`` -> absolute ``xyxy`` px (sizes clamped positive).

    ``cx``/``cy`` are height-normalised centres; ``a``/``h`` are absolute.
    """
    if not np.isfinite(cx):
        cx = 0.0
    if not np.isfinite(cy):
        cy = 0.0
    if not np.isfinite(a) or a <= 0.0:
        a = _MIN_SIZE
    if not np.isfinite(h) or h <= 0.0:
        h = _MIN_SIZE
    cx = float(cx) * h
    cy = float(cy) * h
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def _state_to_meas(s: np.ndarray) -> np.ndarray:
    """State -> measurement vector (identity on the first four components)."""
    v = np.asarray(s, dtype=np.float64).reshape(-1)[:4].copy()
    v[3] = max(abs(float(v[3])), _MIN_SIZE)
    if not np.isfinite(v[2]) or v[2] <= 0.0:
        v[2] = _MIN_SIZE
    return v


def _px_velocity(s: np.ndarray) -> tuple[float, float, float, float]:
    """Convert state velocities to px/frame ``(vcx, vcy, vw, vh)``."""
    v = np.asarray(s, dtype=np.float64).reshape(-1)
    vcx_n, vcy_n = float(v[4]), float(v[5])
    va = float(v[6]) if v.size > 6 else 0.0
    vh = float(v[7]) if v.size > 7 else 0.0
    a = max(abs(float(v[2])), _MIN_SIZE)
    h = max(abs(float(v[3])), _MIN_SIZE)
    return vcx_n * h, vcy_n * h, va * h + a * vh, vh


def _measure(xyxy: np.ndarray) -> np.ndarray:
    """``xyxy`` (N,4) -> normalised measurements (N,4): ``[cx/h, cy/h, a, h]``.

    Identical to :func:`box_xyxy_to_state` by construction, so the Kalman
    filter's ``H`` is the identity on the first four state components and the
    gate threshold is scale-invariant (a person far from the camera gets the
    same tolerance in box-height units as one close to it).
    """
    b = np.asarray(xyxy, dtype=np.float64)
    squeeze = b.ndim == 1
    b = b.reshape(-1, 4)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(np.abs(b[:, 2] - b[:, 0]), _MIN_SIZE)
    h = np.maximum(np.abs(b[:, 3] - b[:, 1]), _MIN_SIZE)
    cx = (b[:, 0] + b[:, 2]) / 2.0
    cy = (b[:, 1] + b[:, 3]) / 2.0
    out = np.stack([cx / h, cy / h, w / h, h], axis=1)
    return out[0] if squeeze else out


def gate_radius(velocity: np.ndarray, dt: float = 1.0, cost: CostCfg | None = None) -> float:
    """Adaptive spatial gate radius.

    ``radius = gate_base + gate_speed_gain * |v| * dt`` where ``|v|`` is the
    centre speed in px/frame (rotation/scale components are ignored: they do
    not move the box).  A fast target therefore gets a proportionally wider
    search window, which is what keeps fast-motion association alive.
    """
    c = cost if cost is not None else CostCfg()
    base = float(getattr(c, "gate_base", 0.10))
    gain = float(getattr(c, "gate_speed_gain", 2.5))
    v = _finite(velocity).reshape(-1)
    speed = 0.0
    if v.size >= 2:
        speed = float(np.hypot(v[0], v[1]))
    dt = 1.0 if not np.isfinite(dt) else float(dt)
    return float(max(1e-6, base + gain * speed * max(dt, 0.0)))


def _process_noise_std(cfg: MotionCfg, mean: np.ndarray, ndim: int) -> np.ndarray:
    """Per-component process-noise std, in *normalised state* units.

    The state centres are stored as ``cx / h`` / ``cy / h`` (see
    :func:`box_xyxy_to_state`), so DeepSORT's absolute convention
    ``std = weight * h`` only survives on the height component.  Scaling the
    normalised centre components by ``h`` as well inflates the innovation
    covariance by ``h**2`` and silently disables Mahalanobis gating.
    """
    wp = float(getattr(cfg, "std_weight_position", 1.0 / 20.0))
    wv = float(getattr(cfg, "std_weight_velocity", 1.0 / 160.0))
    h = max(abs(float(mean[3])), 1.0)
    std = np.zeros(int(ndim), dtype=np.float64)
    std[0] = wp
    std[1] = wp
    std[2] = _ASPECT_STD
    std[3] = wp * h
    if ndim > 4:
        std[4] = wv
        std[5] = wv
        std[6] = _ASPECT_STD
        std[7] = wv * h
    if ndim > _NDIM_BASE:
        # Acceleration / yaw-rate states, in normalised-position units per
        # frame^2.  This must stay on the same footing as the velocity noise
        # (``wv``): DeepSORT's ``wv * h`` scaling would make the acceleration
        # 240x more agile than the velocity, so the CA state winds up and the
        # IMM mixing step injects that divergence into every component.
        std[_NDIM_BASE:] = wv
    return std


def _nsa_noise_scale(score: float, adaptive: bool) -> float:
    """Measurement-noise multiplier for a detection confidence.

    Lower confidence -> larger noise.  ``adaptive`` (NSA) uses
    ``1 / clip(score, eps, 1)``; otherwise the classic ``1 / (1 - score)``
    style scale is used.  Always >= 1 so an honest detection never *shrinks*
    the modelled measurement noise below the base value.
    """
    s = float(score)
    if not np.isfinite(s):
        s = 1.0
    s = min(max(s, 0.0), 1.0)
    if adaptive:
        return float(1.0 / max(s, 1e-2))
    return float(1.0 / max(1.0 - s, 1e-2))


# --------------------------------------------------------------------------- #
# individual motion models (IMM components)
# --------------------------------------------------------------------------- #
class _MotionComponent:
    """One Kalman component: state layout + transition + process noise.

    Subclasses define ``dim``, ``transition_matrix``, ``process_noise`` and
    ``predict_components`` (fast scalar propagation used by :meth:`predict`).
    """

    name = "base"
    dim = _NDIM_BASE
    #: Heuristic process-noise multiplier on the base weights.
    noise_mult = 1.0

    def __init__(self, cfg: MotionCfg, cost: CostCfg | None = None) -> None:
        self.cfg = cfg
        self.cost = cost if cost is not None else CostCfg()
        n = self.dim
        self.mean = np.zeros(n, dtype=np.float64)
        self.cov = np.eye(n, dtype=np.float64) * 1e3
        self.hist_mean: list[np.ndarray] = []
        self.hist_cov: list[np.ndarray] = []
        self.hist_box: list[np.ndarray] = []
        self.box = np.zeros(4, dtype=np.float32)

    # -- geometry ---------------------------------------------------------- #
    @property
    def box_state(self) -> np.ndarray:
        """``[cx/h, cy/h, a, h]`` from the mean."""
        return np.array(self.mean[:4], dtype=np.float64)

    def set_box_state(self, nu_cx: float, nu_cy: float, a: float, h: float) -> None:
        """Initialise from a height-normalised centre plus ``a``/``h`` (px)."""
        m = np.zeros(self.dim, dtype=np.float64)
        m[0], m[1], m[2], m[3] = nu_cx, nu_cy, a, h
        self.mean = m
        self.cov = np.eye(self.dim, dtype=np.float64)
        self.cov[:4, :4] *= (h * h)

    def current_xyxy(self) -> np.ndarray:
        b = self.box_state
        return state_to_xyxy(b[0], b[1], b[2], b[3])

    # -- IMM mixing -------------------------------------------------------- #
    def mix(
        self,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
        default_cov: float,
    ) -> None:
        """Set ``mean``/``cov`` to a weighted mixture of the other models."""
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0.0:
            w = np.full(len(w), 1.0 / max(len(w), 1))
        else:
            w = w / s
        means = np.asarray(means, dtype=np.float64)
        covs = np.asarray(covs, dtype=np.float64)
        mixed = w @ means
        # Second moment about the mixture mean.
        second = np.zeros_like(mixed)
        for wi, mu, cv in zip(w, means, covs):
            if wi <= 0.0:
                continue
            cv = np.asarray(cv, dtype=np.float64)
            second += wi * (np.diag(cv) + (mu - mixed) ** 2)
        if self.dim > means.shape[1]:
            second = np.concatenate(
                [second, np.full(self.dim - means.shape[1], default_cov, dtype=np.float64)]
            )
        elif second.size > self.dim:
            second = second[: self.dim]
        self.mean = mixed[: self.dim].copy()
        self.cov = np.diag(np.maximum(second, 1e-12))

    # -- prediction -------------------------------------------------------- #
    def transition_matrix(self, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        f = np.eye(self.dim, dtype=np.float64)
        f[0, 4] = dt * damp
        f[1, 5] = dt * damp
        f[2, 6] = dt * damp
        f[3, 7] = dt * damp
        return f

    def jacobian(self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        """State-transition Jacobian (identity for the linear models)."""
        return self.transition_matrix(damp=damp, dt=dt)

    def process_noise(self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        del damp  # velocity damping is applied to the state, not to the noise
        std = _process_noise_std(self.cfg, mean, self.dim)
        q = (std * dt * self.noise_mult) ** 2
        return np.diag(np.maximum(q, 1e-14))

    def predict_components(
        self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Return ``(abs_center_px, vel_center_px, vel_w_px, aspect, height)``."""
        cx, cy, a, h, vcx, vcy, va, vh = mean[:8]
        cx = float(cx + dt * damp * vcx)
        cy = float(cy + dt * damp * vcy)
        a = float(a + dt * damp * va)
        h = float(h + dt * damp * vh)
        hh = max(abs(h), _MIN_SIZE)
        aa = max(abs(a), _MIN_SIZE)
        return (
            np.array([cx * hh, cy * hh], dtype=np.float64),
            np.array([vcx * hh, vcy * hh], dtype=np.float64) * damp * dt,
            float(va * hh + aa * vh) * damp * dt,
            a,
            h,
        )

    def predict(self, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        f = self.jacobian(self.mean, damp=damp, dt=dt)
        mean = np.asarray(f @ self.mean, dtype=np.float64)
        if not np.all(np.isfinite(mean)):
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.process_noise(mean, damp=damp, dt=dt)
        cov = f @ self.cov @ f.T + q
        self.mean = mean
        self.cov = 0.5 * (cov + cov.T)
        self._push_history()
        self.box = self.current_xyxy()
        return self.box

    def _push_history(self) -> None:
        self.hist_mean.append(self.mean.copy())
        self.hist_cov.append(self.cov.copy())
        self.hist_box.append(self.current_xyxy())

    def rollback(self, depth: int) -> bool:
        d = int(depth)
        if d <= 0 or len(self.hist_mean) <= d:
            return False
        self.mean = self.hist_mean[-1 - d].copy()
        self.cov = self.hist_cov[-1 - d].copy()
        if len(self.hist_box) > d:
            self.box = self.hist_box[-1 - d].copy()
        del self.hist_mean[-d:]
        del self.hist_cov[-d:]
        del self.hist_box[-d:]
        return True

    def trim(self, max_len: int) -> None:
        if max_len > 0 and len(self.hist_mean) > max_len:
            self.hist_mean = self.hist_mean[-max_len:]
            self.hist_cov = self.hist_cov[-max_len:]
            self.hist_box = self.hist_box[-max_len:]


class _ConstantVelocity(_MotionComponent):
    name = "cv"
    dim = 8
    noise_mult = 1.0


class _ConstantAcceleration(_MotionComponent):
    """12-D: ``[cx, cy, a, h, vcx, vcy, va, vh, acx, acy, aa, ah]``."""

    name = "ca"
    dim = 12
    # Uniform base noise across IMM components: a per-model multiplier inflates
    # the higher-dimensional models' innovation covariance, so they lose the
    # model-probability comparison on dimensionality rather than on fit.
    noise_mult = 1.0

    def transition_matrix(self, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        f = super().transition_matrix(damp=damp, dt=dt)
        f[0, 8] = 0.5 * dt * dt
        f[1, 9] = 0.5 * dt * dt
        f[2, 10] = 0.5 * dt * dt
        f[3, 11] = 0.5 * dt * dt
        f[4, 8] = dt
        f[5, 9] = dt
        f[6, 10] = dt
        f[7, 11] = dt
        return f

    def predict_components(
        self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        cx, cy, a, h, vcx, vcy, va, vh, acx, acy, aa, ah = mean[:12]
        cx = float(cx + dt * damp * vcx + 0.5 * dt * dt * acx)
        cy = float(cy + dt * damp * vcy + 0.5 * dt * dt * acy)
        a = float(a + dt * damp * va + 0.5 * dt * dt * aa)
        h = float(h + dt * damp * vh + 0.5 * dt * dt * ah)
        hh = max(abs(h), _MIN_SIZE)
        aa_ = max(abs(a), _MIN_SIZE)
        return (
            np.array([cx * hh, cy * hh], dtype=np.float64),
            np.array([vcx * hh, vcy * hh], dtype=np.float64) * damp * dt,
            float(va * hh + aa_ * vh) * damp * dt,
            a,
            h,
        )


class _ConstantTurn(_MotionComponent):
    """9-D: ``[cx, cy, a, h, vcx, vcy, va, vh, w]`` with image-plane yaw rate ``w``.

    The centre propagates through the exact constant-turn transition
    (rotation of the velocity vector by ``w * dt``); ``a``/``h`` keep the
    linear model.  Linearised by finite differences -> an EKF, which is exact
    for the (linear) centre dynamics in the limit of small ``w * dt`` and is
    stable for the whole practical range.
    """

    name = "ct"
    dim = 9
    noise_mult = 1.0
    _W_LIMIT = np.pi / 2.0
    _FD_EPS = 1e-4
    #: Process-noise std of the yaw-rate state itself (rad/frame).  A large
    #: value makes ``w`` an unconstrained random walk, which shows up as
    #: oscillation in the blended IMM prediction.
    _W_NOISE_STD = 0.05

    def transition_matrix(self, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        """Local linearisation of the constant-turn step (not a bare identity).

        ``predict()`` uses the exact nonlinear step; this returns its Jacobian so
        that any caller propagating a covariance stays consistent with it.
        """
        return self._fd_jacobian(self.mean, damp=damp, dt=dt)

    def _fd_jacobian(self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        m = np.asarray(mean, dtype=np.float64).reshape(-1)
        if m.size < self.dim:
            m = np.concatenate([m, np.zeros(self.dim - m.size)])
        base = self._step(m, damp, dt)
        f = np.eye(self.dim, dtype=np.float64)
        for j in range(self.dim):
            probe = m.copy()
            probe[j] += self._FD_EPS
            f[:, j] = (self._step(probe, damp, dt) - base) / self._FD_EPS
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _step(self, mean: np.ndarray, damp: float, dt: float) -> np.ndarray:
        cx, cy, a, h, vcx, vcy, va, vh, w = mean[:9]
        theta = float(w) * dt
        theta = float(np.clip(theta, -self._W_LIMIT, self._W_LIMIT))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        vx = float(vcx) * damp
        vy = float(vcy) * damp
        m = np.array(mean, dtype=np.float64).copy()
        if abs(theta) < 1e-6:
            # Small-angle limit: dx = vx*dt, dy = vy*dt (second order in theta).
            dx = vx * dt
            dy = vy * dt
        else:
            s = sin_t / theta
            c = (1.0 - cos_t) / theta
            dx = dt * (vx * s + vy * c)
            dy = dt * (-vx * c + vy * s)
        m[0] = cx + dx
        m[1] = cy + dy
        m[2] = a + dt * damp * va
        m[3] = h + dt * damp * vh
        m[4] = vx * cos_t - vy * sin_t
        m[5] = vx * sin_t + vy * cos_t
        m[6] = va * damp
        m[7] = vh * damp
        m[8] = w
        return m

    def jacobian(self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0) -> np.ndarray:
        return self._fd_jacobian(mean, damp=damp, dt=dt)

    def predict(
        self, damp: float = 1.0, dt: float = 1.0
    ) -> np.ndarray:  # noqa: D102 - documented on the base class
        f = self.jacobian(self.mean, damp=damp, dt=dt)
        mean = self._step(self.mean, damp, dt)
        mean[8] = float(np.clip(mean[8], -self._W_LIMIT, self._W_LIMIT))
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.process_noise(mean, damp=damp, dt=dt)
        # Extra uncertainty on the estimated turn rate itself.
        q[8, 8] = max(q[8, 8], (self._W_NOISE_STD * dt) ** 2)
        cov = f @ self.cov @ f.T + q
        self.mean = mean
        self.cov = 0.5 * (cov + cov.T)
        self._push_history()
        self.box = self.current_xyxy()
        return self.box

    def predict_components(
        self, mean: np.ndarray, damp: float = 1.0, dt: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        m = np.asarray(mean, dtype=np.float64).reshape(-1)
        if m.size < self.dim:
            m = np.concatenate([m, np.zeros(self.dim - m.size)])
        after = self._step(m, damp, dt)
        cx, cy, a, h, vcx, vcy, va, vh = after[:8]
        hh = max(abs(h), _MIN_SIZE)
        aa_ = max(abs(a), _MIN_SIZE)
        return (
            np.array([cx * hh, cy * hh], dtype=np.float64),
            np.array([vcx * hh, vcy * hh], dtype=np.float64),
            float(va * hh + aa_ * vh),
            float(a),
            float(h),
        )


_MODEL_FACTORY: dict[str, Callable[[MotionCfg, CostCfg | None], _MotionComponent]] = {
    "cv": _ConstantVelocity,
    "ca": _ConstantAcceleration,
    "ct": _ConstantTurn,
}


# --------------------------------------------------------------------------- #
# the Kalman filter over the shared measurement
# --------------------------------------------------------------------------- #
_H = np.zeros((4, 12), dtype=np.float64)
for _i in range(4):
    _H[_i, _i] = 1.0


def _kalman_update(
    mean: np.ndarray,
    cov: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    h_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Standard KF update.  Returns ``(mean, cov, log_likelihood)``."""
    n = mean.shape[0]
    h = h_mat[:, :n]
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    y = z - h @ mean
    s = h @ cov @ h.T + r
    s = 0.5 * (s + s.T) + np.eye(s.shape[0]) * 1e-9
    try:
        si = np.linalg.inv(s)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the jitter above
        si = np.linalg.pinv(s)
    k = cov @ h.T @ si
    mean = mean + k @ y
    cov = cov - k @ h @ cov
    cov = 0.5 * (cov + cov.T)
    sign, logdet = np.linalg.slogdet(s)
    if not np.isfinite(logdet) or sign <= 0:
        logdet = float(np.log(max(np.linalg.det(s), 1e-12)))
    quad = float(y @ si @ y)
    log_lik = -0.5 * (quad + logdet + len(z) * np.log(2.0 * np.pi))
    return mean, cov, float(log_lik)


# --------------------------------------------------------------------------- #
# public model
# --------------------------------------------------------------------------- #
class MotionModel:
    """Motion model for the clicked target.  See the module docstring."""

    def __init__(
        self,
        cfg: MotionCfg,
        cost: CostCfg | None = None,
        max_history: int = 256,
    ) -> None:
        self.cfg = cfg
        self.cost = cost if cost is not None else CostCfg()
        self.max_history = max(int(max_history), 8)
        self.hist_box: list[np.ndarray] = []
        self.hist_t: list[int] = []
        self._started = False
        self._last_box = np.zeros(4, dtype=np.float32)
        self._last_vel = np.zeros(4, dtype=np.float32)
        self._last_pred_box = np.zeros(4, dtype=np.float32)
        self._last_innov_cov = np.eye(4, dtype=np.float64) * 1e-6
        self._predicted = False
        self._logged_low_score = False

        names = list(cfg.imm_models) if cfg.model == "imm" else []
        names = [n for n in names if n in _MODEL_FACTORY]
        if cfg.model == "imm" and not names:
            names = ["cv"]
        self._imm_names = names
        self._comps: list[_MotionComponent] = []
        self._mu = np.zeros(0, dtype=np.float64)
        self._trans = np.zeros((0, 0), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def is_imm(self) -> bool:
        return self.cfg.model == "imm" and len(self._comps) > 1

    @property
    def started(self) -> bool:
        return self._started

    @property
    def model_names(self) -> list[str]:
        if self._comps:
            return [c.name for c in self._comps]
        return ["kf"]

    def _imm_setup(self) -> None:
        self._comps = [_MODEL_FACTORY[n](self.cfg, self.cost) for n in self._imm_names]
        k = len(self._comps)
        self._mu = np.full(k, 1.0 / k, dtype=np.float64)
        stay = float(np.clip(getattr(self.cfg, "imm_stay_prob", 0.85), 1e-3, 0.9999))
        off = (1.0 - stay) / max(k - 1, 1)
        t = np.full((k, k), off, dtype=np.float64)
        np.fill_diagonal(t, stay)
        self._trans = t

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def initiate(self, bbox_xyxy: np.ndarray, score: float = 1.0) -> None:
        """Start (or restart) the filter from one observed box."""
        del score  # kept for interface parity; only ``update`` scales noise
        b = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)
        if b.size < 4:
            b = np.concatenate([b, np.zeros(4 - b.size)])
        bx = _finite(b[:4])
        s = box_xyxy_to_state(bx)
        if self.cfg.model == "imm":
            # Always rebuild: `_imm_setup` sizes each component's covariance to
            # its own dimension, so it must not be reused with mutated shapes.
            self._imm_setup()
        else:
            self._comps = []
        if self._comps:
            for c in self._comps:
                c.set_box_state(float(s[0]), float(s[1]), float(s[2]), float(s[3]))
        else:
            self._comps = [_ConstantVelocity(self.cfg, self.cost)]
            self._comps[0].set_box_state(float(s[0]), float(s[1]), float(s[2]), float(s[3]))
            self._imm_names = ["cv"]
        self.hist_box = [bx.astype(np.float32)]
        self.hist_t = [0]
        self._started = True
        self._predicted = False
        self._last_box = bx.astype(np.float32)
        self._last_vel = np.zeros(4, dtype=np.float32)
        self._last_pred_box = bx.astype(np.float32)
        self._last_innov_cov = np.eye(4, dtype=np.float64)
        self._ref_state = np.asarray(s, dtype=np.float64).copy()

    def reset(self, bbox_xyxy: np.ndarray | None = None) -> None:
        """Forget everything; optionally re-initiate on ``bbox_xyxy``."""
        self.hist_box = []
        self.hist_t = []
        self._started = False
        self._predicted = False
        self._last_box = np.zeros(4, dtype=np.float32)
        self._last_vel = np.zeros(4, dtype=np.float32)
        self._last_pred_box = np.zeros(4, dtype=np.float32)
        self._last_innov_cov = np.eye(4, dtype=np.float64) * 1e-6
        self._ref_state = None
        if self.cfg.model == "imm":
            self._imm_setup()
        else:
            self._comps = [_ConstantVelocity(self.cfg, self.cost)]
            self._imm_names = ["cv"]
        if bbox_xyxy is not None:
            self.initiate(bbox_xyxy)

    def _ensure(self) -> None:
        """Make sure components exist and the filter has been initiated."""
        if self.cfg.model == "imm":
            if not self._comps or len(self._comps) != len(self._imm_names):
                self._imm_setup()
        elif not self._comps:
            self._comps = [_ConstantVelocity(self.cfg, self.cost)]
            self._imm_names = ["cv"]
        if not self._started:
            self.initiate(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def _blend(self) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean/covariance of the per-model box states."""
        mus = []
        covs = []
        for c in self._comps:
            m = np.zeros(12, dtype=np.float64)
            m[:4] = c.mean[:4]
            nv = min(4, max(0, c.dim - 4))
            m[4 : 4 + nv] = c.mean[4 : 4 + nv]
            if c.dim >= 12:
                m[8:12] = c.mean[8:12]
            cm = np.eye(12, dtype=np.float64) * 1e-9
            n = min(c.dim, 12)
            cm[:n, :n] = c.cov[:n, :n]
            mus.append(m)
            covs.append(cm)
        w = self._mode_weights()
        mixed = np.tensordot(w, np.stack(mus), axes=(0, 0))
        cov = np.zeros((12, 12), dtype=np.float64)
        for wi, m, cv in zip(w, mus, covs):
            d = (m - mixed).reshape(-1, 1)
            cov += wi * (cv + d @ d.T)
        return mixed, cov

    def _mode_weights(self) -> np.ndarray:
        """Normalised model probabilities.

        Single-model filters (``kf``/``nsa_kf``) never populate ``_mu``, so the
        weights must fall back to "all mass on the only component" instead of
        dividing an empty vector.
        """
        k = len(self._comps)
        if k <= 0:
            return np.zeros(0, dtype=np.float64)
        mu = np.asarray(self._mu, dtype=np.float64).reshape(-1)
        if mu.size != k or not np.all(np.isfinite(mu)):
            return np.full(k, 1.0 / k, dtype=np.float64)
        s = float(mu.sum())
        if not np.isfinite(s) or s <= _EPS:
            return np.full(k, 1.0 / k, dtype=np.float64)
        return mu / s

    def _mix_components(self) -> None:
        """IMM interaction step: re-mix each model from the others' priors."""
        if not self._comps:
            return
        k = len(self._comps)
        means = np.stack([np.pad(c.mean, (0, max(0, 12 - c.dim)))[:12] for c in self._comps])
        covs = np.stack(
            [
                np.pad(c.cov, ((0, max(0, 12 - c.dim)), (0, max(0, 12 - c.dim))))[:12, :12]
                for c in self._comps
            ]
        )
        mu = self._mode_weights()
        cbar = self._trans.T @ mu  # (k,) predicted mode probability
        cbar = np.maximum(cbar, 1e-12)
        default_cov = max((self.cfg.std_weight_position or 0.05) ** 2, 1e-6)
        for j in range(k):
            wj = self._trans[:, j] * mu / cbar[j]
            self._comps[j].mix(means, covs, wj, default_cov)
        self._mu = cbar / max(float(cbar.sum()), _EPS)

    def _set_push_box(self, box: np.ndarray) -> np.ndarray:
        b = _finite(box)
        self._last_pred_box = b.astype(np.float32)
        return self._last_pred_box

    def _apply_shape_prior(self, box: np.ndarray) -> np.ndarray:
        """Bound a PREDICTED box to a multiple of the last OBSERVED shape.

        See :attr:`tgtrack.config.MotionCfg.shape_prior` for the trace that
        motivated this.  Only the predicted extent is bounded -- the centre is
        left to the motion model, and the reference updates on every accepted
        observation so a real approach/recede (which changes the box gradually)
        is unaffected.
        """
        if not bool(getattr(self.cfg, "shape_prior", True)):
            return box
        ref = getattr(self, "_ref_state", None)
        if ref is None:
            return box
        ref_a = max(abs(float(ref[2])), _MIN_SIZE)
        ref_h = max(abs(float(ref[3])), _MIN_SIZE)
        b = np.asarray(box, dtype=np.float64).reshape(4)
        cx = float(b[0] + b[2]) / 2.0
        cy = float(b[1] + b[3]) / 2.0
        w = max(abs(float(b[2] - b[0])), _MIN_SIZE)
        h = max(abs(float(b[3] - b[1])), _MIN_SIZE)
        amin = max(float(getattr(self.cfg, "aspect_min_ratio", 0.6)), 1e-2)
        amax = max(float(getattr(self.cfg, "aspect_max_ratio", 1.7)), amin)
        hmin = max(float(getattr(self.cfg, "size_min_ratio", 0.6)), 1e-2)
        hmax = max(float(getattr(self.cfg, "size_max_ratio", 1.9)), hmin)
        a = float(np.clip(w / h, ref_a * amin, ref_a * amax))
        h = float(np.clip(h, ref_h * hmin, ref_h * hmax))
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)

    def _observed_speed(self) -> float:
        """Median centre displacement per frame over the last few OBSERVED boxes."""
        hist = self.hist_box
        if len(hist) < 2:
            return 0.0
        k = min(5, len(hist) - 1)
        ds = []
        for a, b in zip(hist[-k - 1 : -1], hist[-k:]):
            ds.append(float(np.hypot((b[0] + b[2]) / 2.0 - (a[0] + a[2]) / 2.0,
                                     (b[1] + b[3]) / 2.0 - (a[1] + a[3]) / 2.0)))
        return float(np.median(ds)) if ds else 0.0

    def _clamp_velocities(self) -> None:
        """Keep the aspect state positive and the velocities physically plausible.

        Two independent runtime guards live here, both motivated by the measured
        divergence trace in `results/e014_shapeprio`:

        * ``aspect_vel_max`` bounds the aspect velocity so a partial view cannot
          integrate into a telescoping box;
        * ``velocity_clamp_ratio`` bounds each component's translation velocity to
          a multiple of the recent OBSERVED speed -- after one poisoned update the
          filter predicted -40 px/frame for 12 frames while the person walked
          +8 px/frame, which is what pushed every correct detection out of the
          motion gate.
        """
        vmax = float(getattr(self.cfg, "aspect_vel_max", 0.05))
        ratio = getattr(self.cfg, "velocity_clamp_ratio", None)
        speed = 0.0
        if ratio is not None:
            unit = float(getattr(self.cfg, "velocity_clamp_floor", 2.0))
            speed = max(self._observed_speed(), unit) * max(float(ratio), 1e-3)
        for c in self._comps:
            mean = getattr(c, "mean", None)
            if mean is None or mean.size < 7:
                continue
            mean[2] = max(float(mean[2]), 1e-2)
            if np.isfinite(vmax) and vmax > 0.0:
                mean[6] = float(np.clip(mean[6], -vmax, vmax))
            if speed > 0.0:
                h = max(abs(float(mean[3])), _MIN_SIZE)
                vpx = float(mean[4]) * h
                vpy = float(mean[5]) * h
                norm = float(np.hypot(vpx, vpy))
                if norm > speed and norm > 0.0:
                    k = speed / norm
                    mean[4] = float(mean[4]) * k
                    mean[5] = float(mean[5]) * k
            c.box = c.current_xyxy()

    def _do_predict(self, damp: float = 1.0) -> np.ndarray:
        self._ensure()
        comps = self._comps
        if self.is_imm:
            self._mix_components()
            boxes = np.stack([c.predict(damp=damp) for c in comps])
            w = self._mode_weights()
            box = (w[:, None] * boxes).sum(axis=0)
        else:
            box = comps[0].predict(damp=damp)
        box = _finite(box)
        # Guard against pathological extrapolation.
        h = max(abs(float(box[3] - box[1])), _MIN_SIZE)
        w_ = max(abs(float(box[2] - box[0])), _MIN_SIZE)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        if self.hist_box:
            last = self.hist_box[-1]
            lcx = float(last[0] + last[2]) / 2.0
            lcy = float(last[1] + last[3]) / 2.0
            maxd = _MAX_STEP
            ddx, ddy = cx - lcx, cy - lcy
            dist = float(np.hypot(ddx, ddy))
            if dist > maxd:
                cx = lcx + ddx / dist * maxd
                cy = lcy + ddy / dist * maxd
        box = np.array([cx - w_ / 2.0, cy - h / 2.0, cx + w_ / 2.0, cy + h / 2.0], dtype=np.float64)
        box = self._apply_shape_prior(box)
        self._clamp_velocities()
        self._predicted = True
        return self._set_push_box(box)

    def predict(self) -> np.ndarray:
        """Advance one frame and return the predicted ``(4,)`` xyxy box."""
        return self._do_predict(damp=1.0).copy()

    def multi_hypothesis(self) -> list[tuple[np.ndarray, float]]:
        """Per-model predictions with their IMM weights (IMM only).

        Returns a single hypothesis for ``kf``/``nsa_kf``.  The boxes are the
        current (already predicted) per-model boxes, so this is meant to be
        called after :meth:`predict` when building the association candidate
        set.
        """
        self._ensure()
        if not self.is_imm:
            return [(self._last_pred_box.copy(), 1.0)]
        w = self._mode_weights()
        out: list[tuple[np.ndarray, float]] = []
        for wi, c in zip(w, self._comps):
            out.append((c.box.astype(np.float32).copy(), float(wi)))
        return out

    def coast(self) -> np.ndarray:
        """Advance without an observation, damping the velocity.

        Idempotent within a frame: if :meth:`predict` already advanced the
        state, this only applies the damping/inflation and returns the same
        box.  Never produces NaN/inf and keeps ``h``/``a`` positive.
        """
        self._ensure()
        damp = float(getattr(self.cfg, "coast_velocity_damp", 1.0))
        if not np.isfinite(damp) or damp <= 0.0 or damp > 1.0:
            damp = float(np.clip(damp, 1e-3, 1.0)) if np.isfinite(damp) else 1.0
        if not self._predicted:
            return self._do_predict(damp=damp).copy()
        # Already advanced this frame: only damp the velocity of each model.
        for c in self._comps:
            if c.dim >= 8:
                c.mean[4] *= damp
                c.mean[5] *= damp
                c.mean[6] *= damp
                c.mean[7] *= damp
                if c.dim >= 9:
                    c.mean[8] *= damp
            n = c.dim
            scale = np.ones(n, dtype=np.float64)
            scale[4:8] = 1.0 / max(damp * damp, 1e-6)
            c.cov = c.cov * scale[None, :] * scale[:, None]
            c.cov = 0.5 * (c.cov + c.cov.T)
            c.box = c.current_xyxy()
        if self.is_imm:
            w = self._mode_weights()
            box = (w[:, None] * np.stack([c.box for c in self._comps])).sum(axis=0)
            box = _finite(box)
        else:
            box = _finite(self._comps[0].box)
        return self._set_push_box(box).copy()

    # ------------------------------------------------------------------ #
    # measurement update
    # ------------------------------------------------------------------ #
    def update(self, bbox_xyxy: np.ndarray, score: float = 1.0) -> None:
        """Correct the filter with one accepted observation."""
        self._ensure()
        b = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)
        if b.size < 4:
            b = np.concatenate([b, np.zeros(4 - b.size)])
        bx = _finite(b[:4])
        z = _measure(bx)
        # Measurement noise in normalised measurement units: the centre and the
        # height-component follow DeepSORT's ``weight * h`` scaling, but the
        # centre is already divided by ``h``, so only the height keeps the
        # factor.  The aspect ratio uses a small constant.
        h = max(abs(float(bx[3] - bx[1])), _MIN_SIZE)
        wp = float(self.cfg.std_weight_position)
        if str(self.cfg.model) == "nsa_kf":
            scale = _nsa_noise_scale(score, bool(getattr(self.cfg, "nsa_adaptive", True)))
        else:
            scale = 1.0
        std = np.array([wp, wp, _ASPECT_STD, wp * h], dtype=np.float64) * scale
        r = np.diag(np.maximum(std**2, 1e-12))

        comps = self._comps
        logs = np.zeros(len(comps), dtype=np.float64)
        for i, c in enumerate(comps):
            n = c.dim
            mean, cov, ll = _kalman_update(c.mean, c.cov, z, r, _H[:, :n])
            c.mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            c.cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            # Keep the covariance from collapsing to zero.
            c.cov[np.diag_indices(n)] += 1e-12
            c.box = c.current_xyxy()
            logs[i] = ll
        if len(comps) > 1:
            self._update_weights(logs)
        self._clamp_velocities()
        self._last_box = bx.astype(np.float32)
        self._ref_state = np.asarray(z, dtype=np.float64).copy()
        self._last_vel = self.velocity()
        self.hist_box.append(bx.astype(np.float32))
        self.hist_t.append(len(self.hist_t))
        if len(self.hist_box) > self.max_history:
            self.hist_box = self.hist_box[-self.max_history :]
            self.hist_t = self.hist_t[-self.max_history :]
            for c in self._comps:
                c.trim(self.max_history)
        self._predicted = False

    def _update_weights(self, logs: np.ndarray) -> None:
        logs = np.nan_to_num(logs, nan=-1e9, posinf=0.0, neginf=-1e9)
        logs = logs - float(np.max(logs))
        lik = np.exp(np.clip(logs, -50.0, 50.0))
        post = lik * self._mu
        s = float(post.sum())
        if not np.isfinite(s) or s <= 0.0:
            post = np.full_like(self._mu, 1.0 / len(self._mu))
        else:
            post = post / s
        # Regularise so a singleton model can never dominate forever.
        floor = 0.02 / len(post)
        post = np.maximum(post, floor)
        self._mu = post / float(post.sum())

    # ------------------------------------------------------------------ #
    # readouts used by association / the state machine
    # ------------------------------------------------------------------ #
    def velocity(self) -> np.ndarray:
        """``(4,)`` ``[vcx, vcy, vw, vh]`` in px/frame."""
        self._ensure()
        if self.is_imm:
            w = self._mode_weights()
            acc = np.zeros(4, dtype=np.float64)
            for wi, c in zip(w, self._comps):
                acc += wi * np.asarray(_px_velocity(c.mean), dtype=np.float64)
        else:
            acc = np.asarray(_px_velocity(self._comps[0].mean), dtype=np.float64)
        return _finite(acc).astype(np.float32)

    def history(self) -> list[np.ndarray]:
        """Recent accepted boxes, oldest -> newest."""
        return [b.copy() for b in self.hist_box]

    def _pred_of(self, idx: int) -> np.ndarray:
        """Absolute box predicted for ``idx + 1`` from stored state ``idx``."""
        out = []
        for c in self._comps:
            if 0 <= idx < len(c.hist_mean):
                m = np.asarray(c.hist_mean[idx], dtype=np.float64)
                pc, _, _, a, h = c.predict_components(m)
                out.append(state_to_xyxy(float(pc[0]) / max(abs(h), _MIN_SIZE), float(pc[1]) / max(abs(h), _MIN_SIZE), a, h))
        if out:
            return np.mean(np.stack(out).astype(np.float64), axis=0)
        if 0 <= idx < len(self.hist_box):
            return self.hist_box[idx]
        return self._last_box

    def direction_cost(self, dets_xyxy: np.ndarray) -> np.ndarray:
        """OC-SORT OCM-style direction-consistency cost, ``(N,)``.

        Compares the direction from the last observation to each candidate
        against the track's momentum direction (the predicted displacement
        over the last steps).  Forward-continuing candidates cost ~0, a
        reversal costs ~``direction_weight``.  Returns zeros when the history
        is shorter than 2 accepted boxes (nothing to be consistent with).
        """
        dets = np.asarray(dets_xyxy, dtype=np.float64)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        n = int(dets.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        w_dir = float(getattr(self.cfg, "direction_weight", 0.2))
        window = max(int(getattr(self.cfg, "direction_window", 3)), 1)
        if len(self.hist_box) < 2:
            return np.zeros((n,), dtype=np.float32)
        past = self.hist_box[-1]
        momentum = None
        for k in range(1, window + 1):
            idx = len(self.hist_box) - 1 - k
            if idx < 0:
                break
            d = self._pred_of(idx)[:2] - self.hist_box[idx][:2]
            momentum = d if momentum is None else momentum + d
        if momentum is None:
            return np.zeros((n,), dtype=np.float32)
        norm = float(np.linalg.norm(momentum))
        if norm < 1e-6:
            return np.zeros((n,), dtype=np.float32)
        unit = momentum / norm
        centers = np.stack([(dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0], axis=1)
        obs = centers - np.array([(past[0] + past[2]) / 2.0, (past[1] + past[3]) / 2.0])
        norms = np.linalg.norm(obs, axis=1)
        ok = norms > 1e-6
        cos = np.zeros(n, dtype=np.float64)
        cos[ok] = (obs[ok] / norms[ok, None]) @ unit
        cos = np.clip(np.nan_to_num(cos, nan=0.0), -1.0, 1.0)
        cost = w_dir * (1.0 - cos) / 2.0
        return np.clip(cost, 0.0, 1.0).astype(np.float32)

    def _innov_cov(self) -> np.ndarray:
        """Innovation covariance S of the current predicted state (4x4)."""
        if not self._comps:
            return np.eye(4, dtype=np.float64) * 1e-6
        if self.is_imm:
            _, cov = self._blend()
        else:
            n = self._comps[0].dim
            cov = np.eye(12, dtype=np.float64) * 1e-9
            cov[:n, :n] = self._comps[0].cov
        # Measurement covariance is unknown for a probe; use the same normalised
        # base std as update() so the gate is comparable to the chi-square
        # threshold in CostCfg.motion_gate.
        h = max(abs(float(self._last_pred_box[3] - self._last_pred_box[1])), 1.0)
        wp = float(self.cfg.std_weight_position)
        std = np.array([wp, wp, _ASPECT_STD, wp * h], dtype=np.float64)
        r = np.diag(np.maximum(std**2, 1e-12))
        h_mat = _H[:, :12]
        s = h_mat @ cov @ h_mat.T + r
        return 0.5 * (s + s.T) + np.eye(4) * 1e-9

    def mahalanobis(self, dets_xyxy: np.ndarray) -> np.ndarray:
        """``(N,)`` squared Mahalanobis gating distances of candidate boxes.

        Distances are in the same normalised measurement space as
        ``update()`` (centres divided by box height), so they can be compared
        directly against ``CostCfg.motion_gate`` (chi-square, 4 dof).
        """
        dets = np.asarray(dets_xyxy, dtype=np.float64)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        n = int(dets.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        self._ensure()
        s = self._innov_cov()
        try:
            si = np.linalg.inv(s)
        except np.linalg.LinAlgError:  # pragma: no cover
            si = np.linalg.pinv(s)
        z = _measure(dets)
        if self.is_imm:
            mixed, _ = self._blend()
        else:
            mixed = np.zeros(12, dtype=np.float64)
            mixed[: self._comps[0].dim] = self._comps[0].mean
        pred = mixed[:4]
        d = z - pred[None, :]
        out = np.einsum("ni,ij,nj->n", d, si, d)
        out = np.nan_to_num(out, nan=1e9, posinf=1e9, neginf=1e9)
        return np.clip(out, 0.0, 1e12).astype(np.float32)

    # ------------------------------------------------------------------ #
    # rollback
    # ------------------------------------------------------------------ #
    def rollback(self, depth: int) -> None:
        """Restore the KF state from ``depth`` predicted frames ago.

        This is the FastTracker occlusion-rollback idea: when an occlusion
        starts, undo the Kalman updates that were polluted by the occluder so
        the motion estimate is not dragged off course.  Bounded history means
        an over-deep request is clamped to what is available and never raises.
        """
        d = int(depth)
        if d <= 0:
            return
        self._ensure()
        ok = False
        maxd = min(d, self.max_history - 1)
        for c in self._comps:
            ok = c.rollback(maxd) or ok
        if not ok and self.is_imm:
            # Try shallower depths per component (dims may differ).
            for k in range(maxd - 1, 0, -1):
                if any(c.rollback(k) for c in self._comps):
                    ok = True
                    break
        if ok and len(self.hist_box) > min(d, len(self.hist_box) - 1):
            cut = min(d, len(self.hist_box) - 1)
            if cut > 0:
                del self.hist_box[-cut:]
                del self.hist_t[-cut:]
        if self._comps:
            if self.is_imm:
                w = self._mode_weights()
                box = (w[:, None] * np.stack([c.box for c in self._comps])).sum(axis=0)
            else:
                box = self._comps[0].box
            self._last_pred_box = _finite(box).astype(np.float32)
            self._last_box = self.hist_box[-1].copy() if self.hist_box else self._last_pred_box.copy()
            self._last_vel = self.velocity()
        self._predicted = False

    # ------------------------------------------------------------------ #
    # gates
    # ------------------------------------------------------------------ #
    def gate_radius(self, dt: float = 1.0) -> float:
        """Adaptive spatial gate radius (px) for the current speed."""
        return gate_radius(self.velocity(), dt=dt, cost=self.cost)

    def speed(self) -> float:
        """Centre speed in px/frame."""
        v = self.velocity()
        return float(np.hypot(float(v[0]), float(v[1])))

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        """Debug snapshot (used by the evaluator / reports)."""
        return {
            "model": str(self.cfg.model),
            "is_imm": bool(self.is_imm),
            "models": self.model_names,
            "weights": [float(x) for x in self._mu] if len(self._mu) else [1.0],
            "box": self._last_pred_box.astype(float).tolist(),
            "velocity": self.velocity().astype(float).tolist(),
            "history": len(self.hist_box),
            "gate_radius": float(self.gate_radius()),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MotionModel(model={self.cfg.model!r}, models={self.model_names}, "
            f"started={self._started}, history={len(self.hist_box)})"
        )
