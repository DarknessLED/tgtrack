"""E089 -- widen (or drop) the identity ceiling while the target is LOST.

E088's per-frame audit (`docs/EXPERIMENTS.md`, E088) measured where the shipped target loss
lives: 91 % of it is a reported box that is neither the target nor locked to anyone, 55 % of
those frames carry `frames_since_update >= 60`, and on the missed frames a candidate existed
100 % of the time -- refused 41 % of the time by `all_gated:anchor_ceiling` while the tracker
was already in `lost_short` / `lost_long`.

So E089 scales the E027/E029 ceiling against the frozen click crop once the target is lost:
`cost.anchor_band_lost_scale` is a multiplier applied to the band limit only after
`lost_short_after` frames without an accepted update.  What must remain true:

* the default 1.0 is the shipped behaviour and is bit-identical (the limit is returned
  untouched, so no arithmetic changes anywhere);
* 1.0 must stay inert even when the target is lost for a long time;
* > 1 widens the band exactly by the factor, and only once the target is lost;
* <= 0 drops the ceiling (returns None) only once the target is lost, never while fresh.

These are unit tests of the gate arithmetic on a tracker built the way the other ceiling tests
build one, plus one contract test on the ablation arms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tgtrack.config import Config  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
BOX = [500.0, 200.0, 600.0, 500.0]
BAND = 0.40


class _FixedEmbedder:
    """Encoder that answers the same vector for every crop."""

    def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        vec = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        return [{p: vec.copy() for p in parts} for _ in range(int(boxes.shape[0]))]


def _tracker(**overrides) -> TargetTracker:
    cfg = Config().clone(
        **{"cost.anchor_adaptive": False, "cost.anchor_max_dist": BAND, **overrides}
    )
    tt = TargetTracker(cfg, _FixedEmbedder())
    tt.initialize(FRAME, BOX, frame_idx=0)
    return tt


# --------------------------------------------------------------------------- #
# the knob
# --------------------------------------------------------------------------- #
def test_default_scale_is_the_shipped_1_25() -> None:
    # E089 shipped 1.25 (docs/EXPERIMENTS.md); 1.0 is the pre-E089 control and must stay
    # inert, which is what keeps the historical 22-sequence records comparable.
    assert Config().cost.anchor_band_lost_scale == pytest.approx(1.25)


def test_scale_one_is_inert_even_when_lost() -> None:
    tt = _tracker(**{"cost.anchor_band_lost_scale": 1.0})
    tt.frames_since_update = 500
    assert tt._lost_scaled_limit(tt.cfg, BAND) == pytest.approx(BAND)
    # and no diagnostic field appears, so the historical path's run record is unchanged
    assert "anchor_band_lost_scale" not in tt._last_diag


def test_widening_only_starts_after_lost_short_after() -> None:
    tt = _tracker(**{"cost.anchor_band_lost_scale": 2.0})
    after = int(tt.cfg.state.lost_short_after)
    tt.frames_since_update = after
    assert tt._lost_scaled_limit(tt.cfg, BAND) == pytest.approx(BAND)
    tt.frames_since_update = after + 1
    assert tt._lost_scaled_limit(tt.cfg, BAND) == pytest.approx(BAND * 2.0)


def test_scale_multiplies_the_band() -> None:
    for scale in (1.25, 1.5, 2.0):
        tt = _tracker(**{"cost.anchor_band_lost_scale": scale})
        tt.frames_since_update = 10_000
        assert tt._lost_scaled_limit(tt.cfg, BAND) == pytest.approx(BAND * scale)


def test_zero_scale_drops_the_ceiling_only_while_lost() -> None:
    tt = _tracker(**{"cost.anchor_band_lost_scale": 0.0})
    tt.frames_since_update = 0
    assert tt._lost_scaled_limit(tt.cfg, BAND) == pytest.approx(BAND)
    tt.frames_since_update = 10_000
    assert tt._lost_scaled_limit(tt.cfg, BAND) is None


def test_no_ceiling_stays_no_ceiling() -> None:
    tt = _tracker(**{"cost.anchor_band_lost_scale": 2.0})
    tt.frames_since_update = 10_000
    assert tt._lost_scaled_limit(tt.cfg, None) is None


def test_applied_scale_is_recorded_in_diagnostics() -> None:
    tt = _tracker(**{"cost.anchor_band_lost_scale": 1.5})
    tt.frames_since_update = 10_000
    tt._lost_scaled_limit(tt.cfg, BAND)
    assert tt._last_diag["anchor_band_lost_scale"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# the ablation arms
# --------------------------------------------------------------------------- #
def test_run_target_exposes_the_ceiling_lost_arms() -> None:
    import run_target as rt  # noqa: E402  (needs the path insert above)

    variants = rt.build_variants("ceiling_lost")
    names = {v.name for v in variants}
    # `baseline` always rides along: every group is measured against the shipped model.
    assert names == {"baseline", "ceil_lost_1.25", "ceil_lost_1.50", "ceil_lost_2.00",
                     "ceil_lost_off", "ceil_lost_1.00", "shipped_e089"}
    control = [v for v in variants if v.name == "ceil_lost_1.00"][0]
    assert control.overrides["cost.anchor_band_lost_scale"] == pytest.approx(1.0)
    # the shipped arm spells out no scale, so it tracks the config default (E089: 1.25)
    shipped = [v for v in variants if v.name == "shipped_e089"][0]
    assert "cost.anchor_band_lost_scale" not in shipped.overrides
    assert Config().cost.anchor_band_lost_scale == pytest.approx(1.25)
    # every non-baseline arm must pin the shipped vote mode, otherwise it changes two things
    arms = [v for v in variants if v.name != "baseline"]
    assert all(v.overrides["cost.anchor_vote"] == "off" for v in arms)
