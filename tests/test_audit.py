"""E064 -- the in-track rival audit needs TWO references to agree.

E063's audit compared every non-overlapping detection with the frozen click
crop ALONE.  That crop is the only descriptor the impostor cannot edit
(E036/E039), but it also cannot move, so on a target whose look legitimately
changes (dancetrack0005, the sequence E024 showed needs fast forgetting) it
prefers a stranger to the true target and the audit vetoes a CORRECT track.
E064 adds ``stm.current()`` -- the EMA of the recent confident crops -- as a
second witness: switch only when BOTH prefer the rival.

These tests drive ``_audit_switch`` directly with hand-set features, so no GPU
encoder is involved.
"""

from __future__ import annotations

import numpy as np

from tgtrack.config import Config
from tgtrack.target_tracker import TargetState, TargetTracker
from tgtrack.types import Detection

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
CUR_BOX = np.array([200.0, 200.0, 300.0, 500.0])
RIVAL_BOX = np.array([700.0, 200.0, 800.0, 500.0])

#: The frozen click crop and the recent-appearance EMA are two different looks.
CLICK = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
RECENT = np.array([0.0, 1.0, 0.0, 0.0], np.float32)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


#: The box we follow: a mediocre match to both references (cos 0.50 each),
#: because it spends part of its norm on a third direction.
CUR_FEAT = _unit([0.5, 0.5, 0.7071, 0.0])
#: Rival both references prefer: cos 0.745 to the click, 0.667 to the recent EMA.
RIVAL_AGREE = _unit([0.95, 0.85, 0.0, 0.0])
#: Rival that is clearly better against the click (0.978) and clearly WORSE
#: against the recent appearance (-0.206) -- the E063 dancetrack0005 failure.
RIVAL_DISAGREE = _unit([0.95, -0.20, 0.0, 0.0])


def _tracker(**cost_kw) -> TargetTracker:
    cfg = Config()
    cost_kw.setdefault("audit_interval", 1)
    for k, v in cost_kw.items():
        setattr(cfg.cost, k, v)
    tr = TargetTracker(cfg)
    tr.state = TargetState.TRACKING
    tr.frames_since_update = 0
    tr._box = CUR_BOX.copy()
    tr._anchor_feat = CLICK.copy()
    tr._last_diag = {}
    tr.stm.reset()
    tr.stm.update(RECENT, quality=1.0, frame_idx=0)
    return tr


def _audit(tr: TargetTracker, rival_feat, *, cur_feat=CUR_FEAT, frames: int = 2) -> None:
    """Run ``frames`` audits with a fixed current/rival pair (persist = 2)."""
    for k in range(frames):
        cur = Detection(xyxy=CUR_BOX, score=0.9, feat=cur_feat)
        rival = Detection(xyxy=RIVAL_BOX, score=0.9, feat=rival_feat)
        tr.frame_idx = k
        tr._audit_switch(FRAME, [cur, rival], cur, 720, 1280)


def test_audit_switches_when_both_references_prefer_the_rival():
    tr = _tracker()
    _audit(tr, RIVAL_AGREE)
    assert tr._last_diag.get("audit_switch") == 1
    assert np.allclose(tr._box, RIVAL_BOX)
    assert tr._last_diag["audit_margin_click"] > 0.05
    assert tr._last_diag["audit_margin_recent"] > 0.05
    # a switch re-seeds the appearance memory with the rival's own crop
    assert np.allclose(_unit(tr.stm.current()), _unit(RIVAL_AGREE))


def test_audit_refuses_when_the_two_references_disagree():
    tr = _tracker()
    _audit(tr, RIVAL_DISAGREE, frames=4)
    assert tr._last_diag.get("audit_switch") is None
    assert np.allclose(tr._box, CUR_BOX)
    assert tr._audit_n == 0
    # the click crop ALONE would switch; the recent reference is the veto
    assert tr._last_diag["audit_margin_click"] > 0.05
    assert tr._last_diag["audit_margin_recent"] < 0.05


def test_click_anchor_alone_decides_when_audit_use_recent_is_off():
    tr = _tracker(audit_use_recent=False)
    _audit(tr, RIVAL_DISAGREE)
    assert tr._last_diag.get("audit_switch") == 1
    assert np.allclose(tr._box, RIVAL_BOX)
    # no recent distance is computed at all, so no stale margin can be read
    assert tr._last_diag["audit_margin_recent"] is None


def test_audit_falls_back_to_the_click_anchor_without_a_recent_reference():
    tr = _tracker()
    tr.stm.reset()
    assert tr.stm.current() is None
    _audit(tr, RIVAL_DISAGREE)
    assert tr._last_diag.get("audit_switch") == 1
    assert tr._last_diag["audit_margin_recent"] is None


def test_click_anchor_alone_still_refuses_a_rival_that_wins_nothing():
    """The control's silence is not vacuous: the click test can still say no."""
    tr = _tracker(audit_use_recent=False)
    _audit(tr, CUR_FEAT)
    assert tr._last_diag.get("audit_switch") is None
    assert tr._last_diag["audit_margin_click"] <= 0.05
