"""E099 -- re-attribute over the PRE-veto candidate pool with a bounded window DP.

E097 measured why this is the last mechanism with a reason to be one-sided: over five
informative held-out sequences the pre-veto pool contains the true target on 65.84 % of
visible frames while the shipped tracker is correct on 22.02 % (+43.82 pp), because the
identity ceiling vetoes the true target *before* the vote counts (E084: 41 % of refusals
are `all_gated:anchor_ceiling`).  An offline copy of this selector captured 44 % of that
headroom (+19.25 pp TSR).

What must remain true in the shipped code:

* ``state.preveto_window = 0`` (the default) records nothing and changes nothing;
* the retained pool is pruned to the window;
* the DP needs the current frame and at least two frames, else it declines;
* the DP prefers an appearance-cheap, motion-continuous chain;
* re-attribution fires only on an *outright* person replacement (IoU < 0.3 against the
  committed box), never on a near variant -- that is the E092b lesson;
* the ablation group ships the two windows plus a bit-identical control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tgtrack.config import Config  # noqa: E402
from tgtrack.target_tracker import TargetTracker  # noqa: E402
from tgtrack.types import Detection  # noqa: E402

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


def _det(box, score=0.9) -> Detection:
    return Detection(xyxy=np.asarray(box, np.float32), score=float(score))


def _push(tt: TargetTracker, frame: int, boxes, app, shipped: int = -1, chosen=None) -> None:
    """Feed one frame's pre-veto pool through the same two calls ``step()`` uses."""
    tt.frame_idx = int(frame)
    tt._preveto_pending = ([_det(b) for b in boxes], np.asarray(app, float))
    tt._preveto_record(chosen)


# --------------------------------------------------------------------------- #
# the shipped default is untouched
# --------------------------------------------------------------------------- #
def test_default_window_is_off_and_records_nothing():
    tt = _tracker()
    assert int(tt.cfg.state.preveto_window) == 0
    assert tt._preveto_on is False
    tt.step(FRAME, [], frame_idx=1)
    assert tt._preveto_hist == {}
    assert not any(k.startswith("preveto_") for k in tt._last_diag)


def test_armed_tracker_prunes_the_window():
    tt = _tracker(**{"state.preveto_window": 3})
    assert tt._preveto_on is True
    boxes = [[100.0, 100.0, 200.0, 300.0], [400.0, 100.0, 500.0, 300.0]]
    for f in range(1, 8):
        _push(tt, f, boxes, [0.1, 0.2])
    assert sorted(tt._preveto_hist) == [5, 6, 7]


def test_shipped_choice_is_remembered_as_the_switching_prior():
    tt = _tracker(**{"state.preveto_window": 4})
    boxes = [[100.0, 100.0, 200.0, 300.0], [400.0, 100.0, 500.0, 300.0]]
    chosen = _det(boxes[1])
    _push(tt, 1, boxes, [0.5, 0.2], chosen=chosen)
    assert tt._preveto_hist[1]["shipped"] == 1
    _push(tt, 2, boxes, [0.5, 0.2], chosen=_det([9999.0, 9999.0, 9998.0, 9998.0]))
    assert tt._preveto_hist[2]["shipped"] == -1  # no pool candidate matches the choice


# --------------------------------------------------------------------------- #
# the DP
# --------------------------------------------------------------------------- #
def test_pick_declines_without_the_current_frame():
    tt = _tracker(**{"state.preveto_window": 4})
    _push(tt, 1, [[0.0, 0.0, 10.0, 20.0]], [0.1])
    _push(tt, 2, [[0.0, 0.0, 10.0, 20.0]], [0.1])
    tt.frame_idx = 3  # nothing recorded for frame 3
    assert tt._preveto_pick() is None


def test_pick_declines_with_a_single_frame():
    tt = _tracker(**{"state.preveto_window": 4})
    _push(tt, 1, [[0.0, 0.0, 10.0, 20.0]], [0.1])
    assert tt._preveto_pick() is None


def test_pick_prefers_the_appearance_cheap_chain():
    tt = _tracker(**{"state.preveto_window": 4})
    near = [[100.0, 100.0, 200.0, 300.0], [104.0, 100.0, 204.0, 300.0]]
    far = [[900.0, 100.0, 1000.0, 300.0], [904.0, 100.0, 1004.0, 300.0]]
    _push(tt, 1, near, [0.10, 0.90])
    _push(tt, 2, near, [0.10, 0.90])
    det = tt._preveto_pick()
    assert det is not None
    assert np.allclose(np.asarray(det.xyxy, float), near[0], atol=1e-3)
    assert far  # (kept for readability of the two-candidate layout)


def test_pick_prefers_motion_continuity_at_equal_appearance():
    tt = _tracker(**{"state.preveto_window": 4})
    # both candidates carry the same appearance cost; only one continues the track
    _push(tt, 1, [[100.0, 100.0, 200.0, 300.0]], [0.30])
    _push(tt, 2, [[104.0, 100.0, 204.0, 300.0], [800.0, 100.0, 900.0, 300.0]], [0.30, 0.30])
    det = tt._preveto_pick()
    assert det is not None
    assert np.allclose(np.asarray(det.xyxy, float), [104.0, 100.0, 204.0, 300.0], atol=1e-3)


# --------------------------------------------------------------------------- #
# re-attribution: outright replacement only
# --------------------------------------------------------------------------- #
def _stub_match(tt: TargetTracker) -> list[tuple]:
    calls: list[tuple] = []

    def _fake(frame_bgr, det, result, h, w):
        calls.append((np.asarray(det.xyxy, float), result.reason))

    tt._on_match = _fake  # type: ignore[assignment]
    tt._check_drift = lambda *a, **k: None  # type: ignore[assignment]
    return calls


def test_reattribute_refuses_a_near_variant_of_the_committed_box():
    tt = _tracker(**{"state.preveto_window": 4})
    calls = _stub_match(tt)
    tt._box = np.asarray(BOX, np.float32)
    _push(tt, 1, [BOX], [0.10])
    _push(tt, 2, [[502.0, 202.0, 602.0, 502.0]], [0.10])
    tt._preveto_try_reattribute(FRAME, _mr(), 720, 1280)
    assert calls == []
    assert tt._preveto_switches == 0


def test_reattribute_accepts_a_different_person():
    tt = _tracker(**{"state.preveto_window": 4})
    calls = _stub_match(tt)
    tt._box = np.asarray(BOX, np.float32)
    other = [900.0, 300.0, 1000.0, 600.0]
    _push(tt, 1, [other], [0.10])
    _push(tt, 2, [other], [0.10])
    tt._preveto_try_reattribute(FRAME, _mr(), 720, 1280)
    assert len(calls) == 1
    assert calls[0][1] == "preveto:reattributed"
    assert tt._preveto_switches == 1
    assert tt._last_diag.get("preveto_switched") == 1


def _mr():
    from tgtrack.association import MatchResult

    return MatchResult(index=None, cost=0.42, accepted=False, reason="ambiguous: margin 0.01 < 0.060")


# --------------------------------------------------------------------------- #
# ablation contract
# --------------------------------------------------------------------------- #
def test_preveto_ablation_arms():
    from run_target import build_variants

    variants = build_variants("preveto")
    names = [v.name for v in variants]
    assert names[-1] == "preveto_L0", names
    assert {"preveto_L8", "preveto_L4", "preveto_L0"}.issubset(set(names))
    by = {v.name: v for v in variants}
    assert int(by["preveto_L8"].overrides["state.preveto_window"]) == 8
    assert int(by["preveto_L4"].overrides["state.preveto_window"]) == 4
    assert int(by["preveto_L0"].overrides["state.preveto_window"]) == 0
    for name in ("preveto_L8", "preveto_L4", "preveto_L0"):
        assert by[name].overrides.get("cost.anchor_vote") == "off"
