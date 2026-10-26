"""Tests for the clicked-target state machine (Lead-owned).

These are the tests that matter most: they encode the *behavioural* promises of
the system — that it escalates through occlusion states, that it refuses to
gamble on identity, that it re-acquires from appearance rather than geometry,
and that it never absorbs the occluder into the target's memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import Config
from tgtrack.target_tracker import BLIND_STATES, TargetState, TargetTracker
from tgtrack.types import Detection

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
#: x-centre that separates "the target" (left) from "a bystander" (right).
SPLIT = 800.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


IDENT_A = _unit([1.0, 0.0, 0.0, 0.0])
IDENT_B = _unit([0.0, 1.0, 0.0, 0.0])


class StubEmbedder:
    """Deterministic encoder: identity is decided by the box's x-centre.

    A real ReID net is not needed to test the state machine — what matters is
    that *the same person always gets the same vector* and a different person
    gets an orthogonal one, which this guarantees exactly.
    """

    def __init__(self, split: float = SPLIT) -> None:
        self.split = split
        self.calls = 0
        self.seen: list[tuple[float, ...]] = []

    def feat_for(self, box) -> np.ndarray:
        cx = (float(box[0]) + float(box[2])) / 2.0
        return IDENT_A if cx < self.split else IDENT_B

    def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        self.calls += 1
        out = []
        for b in boxes:
            self.seen.append(tuple(np.round(b, 1)))
            f = self.feat_for(b)
            out.append({p: f.copy() for p in parts})
        return out


def _det(box, score=0.9, **kw) -> Detection:
    return Detection(xyxy=np.asarray(box, np.float32), score=score, **kw)


def _tracker_at(box, **cost_kw) -> tuple[TargetTracker, StubEmbedder]:
    """Build a tracker whose click is ``box`` (some tests need the target to
    start somewhere other than the default spot)."""
    cfg = Config()
    for k, v in cost_kw.items():
        setattr(cfg.cost, k, v)
    emb = StubEmbedder()
    tt = TargetTracker(cfg, emb)
    tt.initialize(FRAME, box, frame_idx=0)
    return tt, emb


def _tracker(**cost_kw) -> tuple[TargetTracker, StubEmbedder]:
    return _tracker_at([500, 200, 600, 500], **cost_kw)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_initialize_seeds_appearance_memory_with_all_parts():
    tt, emb = _tracker()
    assert tt.state is TargetState.TRACKING
    assert tt.status().is_reporting
    assert len(tt.ltm) > 0, "the click must seed the long-term gallery"
    # The clicked crop is the one frame we know is the right person, so it
    # should have populated every configured part bank.
    assert set(tt.ltm.parts()) >= {"full", "upper", "lower", "head"}
    assert tt.stm.current() is not None


def test_tracks_a_smoothly_moving_target():
    tt, _ = _tracker()
    for i in range(1, 31):
        d = _det([500 + 5 * i, 200, 600 + 5 * i, 500])
        s = tt.step(FRAME, [d], frame_idx=i)
        assert s.state == TargetState.TRACKING, (i, s.state, s.diag)
        assert s.frames_since_update == 0
    assert tt.hits == 31
    assert np.allclose(tt.status().xyxy, [650, 200, 750, 500])


def test_reports_no_location_after_giving_up():
    tt, _ = _tracker()
    tt.cfg.state.give_up_after = 10
    for i in range(1, 20):
        s = tt.step(FRAME, [], frame_idx=i)
    assert s.state == TargetState.GAVE_UP


# --------------------------------------------------------------------------- #
# occlusion escalation
# --------------------------------------------------------------------------- #


def test_state_escalates_on_the_configured_schedule():
    """Each threshold must fire on the exact frame the config promises."""
    tt, _ = _tracker()
    tt.step(FRAME, [_det([500, 200, 600, 500])], frame_idx=1)  # one good match

    seen: dict[int, str] = {}
    for i in range(2, 80):
        s = tt.step(FRAME, [], frame_idx=i)
        seen[s.frames_since_update] = s.state

    # soft_occlusion_after=2, lost_short_after=12, lost_long_after=60
    assert seen[1] == TargetState.TRACKING
    assert seen[2] == TargetState.SOFT_OCCLUSION
    assert seen[12] == TargetState.SOFT_OCCLUSION
    assert seen[13] == TargetState.LOST_SHORT
    assert seen[60] == TargetState.LOST_SHORT
    assert seen[61] == TargetState.LOST_LONG
    assert seen[78] == TargetState.LOST_LONG

    assert any(e.kind == "occlusion_rollback" for e in tt.events)
    assert sum(1 for e in tt.events if e.kind == "occlusion_rollback") == 1, "rollback once per episode"


def test_coasts_the_box_while_occluded_so_the_operator_still_sees_a_position():
    tt, _ = _tracker()
    # The filter needs ~20 observations before its velocity estimate is real
    # (Kalman warm-up: see the motion notes in docs/), so warm it up first --
    # otherwise this test measures the transient, not the coasting behaviour.
    for i in range(1, 21):
        tt.step(FRAME, [_det([500 + 10 * i, 200, 600 + 10 * i, 500])], frame_idx=i)
    s = tt.step(FRAME, [], frame_idx=21)
    assert s.xyxy is not None
    # Last seen at x=700; coasting must carry it past that, not freeze on it.
    assert s.xyxy[0] > 700, f"coasting stalled at {s.xyxy[0]}"


def test_appearance_memory_is_frozen_while_blind():
    """The occluder must never be learned as the target."""
    tt, _ = _tracker()
    for i in range(1, 4):
        tt.step(FRAME, [_det([500, 200, 600, 500])], frame_idx=i)
    before = len(tt.ltm)
    # A large occluder (identity B -- a genuinely different person) covers where
    # the target was. If the gallery were not frozen, these crops would be
    # absorbed as the target and the tracker would become confident about the
    # wrong person.
    for i in range(4, 30):
        tt.step(FRAME, [_det([900, 100, 1150, 700])], frame_idx=i)
        if i >= 5:
            assert tt.state in BLIND_STATES, (i, tt.state)
    assert tt.memory_frozen
    assert len(tt.ltm) == before, "blind states must not absorb appearance"


# --------------------------------------------------------------------------- #
# identity discipline
# --------------------------------------------------------------------------- #


def test_refuses_to_reacquire_on_geometry_alone():
    """Without appearance evidence a rebind is a guess, and guesses are banned."""
    cfg = Config()
    tt = TargetTracker(cfg, embedder=None)  # no encoder at all
    tt.initialize(FRAME, [500, 200, 600, 500], frame_idx=0)
    for i in range(1, 80):
        tt.step(FRAME, [], frame_idx=i)
    assert tt.state == TargetState.LOST_LONG
    # A person appears far away. With no features we cannot know who it is.
    s = tt.step(FRAME, [_det([100, 100, 200, 400])], frame_idx=80)
    assert s.state == TargetState.LOST_LONG
    assert not s.reacquired


def test_reacquires_from_appearance_after_a_long_loss():
    tt, _ = _tracker()
    for i in range(1, 4):
        tt.step(FRAME, [_det([500, 200, 600, 500])], frame_idx=i)
    # Gone for much longer than lost_long_after -> LOST_LONG, gate disabled.
    for i in range(4, 80):
        tt.step(FRAME, [], frame_idx=i)
    assert tt.state == TargetState.LOST_LONG

    # Reappears at a completely different place; only appearance can find it.
    s = tt.step(FRAME, [_det([100, 100, 200, 400])], frame_idx=80)
    assert s.reacquired, s.diag
    assert s.state == TargetState.REACQUIRING
    assert np.allclose(s.xyxy, [100, 100, 200, 400])
    assert any(e.kind == "reacquire" for e in tt.events)

    # It must hold before being called TRACKING again.
    s2 = tt.step(FRAME, [_det([100, 100, 200, 400])], frame_idx=81)
    assert s2.state == TargetState.TRACKING
    assert any(e.kind == "confirm" for e in tt.events)


def test_does_not_reacquire_a_different_person():
    """The distractor looks like *a* person but not like THIS person."""
    tt, _ = _tracker()
    for i in range(1, 4):
        tt.step(FRAME, [_det([500, 200, 600, 500])], frame_idx=i)
    for i in range(4, 80):
        tt.step(FRAME, [], frame_idx=i)
    # A bystander (identity B) appears where the target vanished.
    s = tt.step(FRAME, [_det([900, 200, 1000, 500])], frame_idx=80)
    assert not s.reacquired
    assert s.state == TargetState.LOST_LONG


def test_ambiguity_is_refused_and_the_distractor_is_remembered():
    """Two indistinguishable candidates must not be resolved by a coin flip."""
    tt, _ = _tracker(distractor_margin=0.5)
    # Two boxes, both identity A, both essentially on the prediction.
    dets = [_det([500, 200, 600, 500]), _det([502, 201, 602, 501])]
    s = tt.step(FRAME, dets, frame_idx=1)
    assert s.state != TargetState.TRACKING or s.frames_since_update > 0
    assert any(e.kind == "swap_risk" for e in tt.events), tt.events
    assert tt.ltm.num_hard_negatives >= 1, "the competitor must be memorized"


def test_lookalike_bystander_is_stored_as_a_hard_negative():
    """After a confident match, a near-identical OTHER person is a distractor."""
    tt, _ = _tracker()
    # Target (identity A) on the left AND a look-alike (also A) further away.
    s = tt.step(
        FRAME,
        [_det([500, 200, 600, 500]), _det([640, 210, 740, 510])],
        frame_idx=1,
    )
    assert s.state == TargetState.TRACKING
    assert tt.ltm.num_hard_negatives >= 1


# --------------------------------------------------------------------------- #
# out of frame
# --------------------------------------------------------------------------- #


def test_out_of_frame_is_detected_and_announced():
    # The target must stay identity A (x-centre < 800) for the whole run, so it
    # starts at x~750 and walks left.
    tt, _ = _tracker_at([700, 200, 800, 500])
    # Warm the filter up so the velocity is real, then walk the target off the
    # left edge; the coasted box must keep going rather than freeze at x~0.
    for i in range(1, 26):
        tt.step(FRAME, [_det([700 - 20 * i, 200, 800 - 20 * i, 500])], frame_idx=i)
    out = False
    for i in range(26, 90):
        s = tt.step(FRAME, [], frame_idx=i)
        if s.state == TargetState.OUT_OF_FRAME:
            out = True
            break
    assert out, "the box must eventually leave the frame and be reported"
    assert any(e.kind == "out_of_frame" for e in tt.events)


def test_reentering_the_frame_is_announced():
    tt, _ = _tracker_at([700, 200, 800, 500])
    for i in range(1, 26):
        tt.step(FRAME, [_det([700 - 20 * i, 200, 800 - 20 * i, 500])], frame_idx=i)
    for i in range(26, 70):
        tt.step(FRAME, [], frame_idx=i)
    assert tt._was_out_of_frame
    # Comes back in from the left edge (identity A, the target): because it is
    # off-frame the spatial gate is disabled and appearance alone decides.
    s = tt.step(FRAME, [_det([60, 200, 160, 500])], frame_idx=70)
    assert s.reacquired, s.diag
    assert any(e.kind == "reentered" for e in tt.events)


# --------------------------------------------------------------------------- #
# memory + serialization
# --------------------------------------------------------------------------- #


def test_snapshot_restore_round_trip_is_faithful():
    tt, _ = _tracker()
    for i in range(1, 7):
        tt.step(FRAME, [_det([500 + 5 * i, 200, 600 + 5 * i, 500])], frame_idx=i)
    snap = tt.snapshot()
    box, state, hits = tt.box, tt.state, tt.hits

    tt.restore(snap)
    assert tt.state is state
    assert tt.hits == hits
    assert np.allclose(tt.box, box)
    assert tt.stm.current() is not None
    assert len(tt.ltm) == len(snap["ltm"]["entries"])


def test_stats_is_serializable_and_complete():
    import json

    tt, _ = _tracker()
    for i in range(1, 5):
        tt.step(FRAME, [_det([500 + 5 * i, 200, 600 + 5 * i, 500])], frame_idx=i)
    st = tt.stats()
    json.dumps(st)  # must not raise
    assert st["state"] == TargetState.TRACKING.value
    assert st["is_imm"] is True
    assert set(st["ltm_parts"]) >= {"full"}


def test_events_are_recorded_for_the_evaluator():
    tt, _ = _tracker()
    kinds = [e.kind for e in tt.events]
    assert kinds[0] == "init"
    for i in range(1, 25):
        tt.step(FRAME, [], frame_idx=i)
    kinds = [e.kind for e in tt.events]
    assert "lost_short" in kinds


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #


def test_step_before_initialize_raises():
    tt = TargetTracker(Config(), StubEmbedder())
    with pytest.raises(RuntimeError):
        tt.step(FRAME, [])


def test_degenerate_click_is_rejected():
    tt = TargetTracker(Config(), StubEmbedder())
    with pytest.raises(ValueError):
        tt.initialize(FRAME, [100, 100, 100, 100])


def test_encoder_failure_does_not_kill_tracking():
    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    tt = TargetTracker(Config(), Broken())
    tt.initialize(FRAME, [500, 200, 600, 500], frame_idx=0)
    for i in range(1, 6):
        s = tt.step(FRAME, [_det([500 + 5 * i, 200, 600 + 5 * i, 500])], frame_idx=i)
        assert s.state == TargetState.TRACKING


def test_broken_encoder_is_recorded_not_swallowed():
    """E067: a raising encoder still degrades to geometry-only -- but loudly.

    The swallow that motivated E067 left one diag key that `initialize` wipes and
    nothing reads, so a run where the encoder never worked looked exactly like a
    healthy geometry-only tracker (that is how the first profile reported
    "encoder = 2% of wall"). Both signals must survive now.
    """

    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    tt = TargetTracker(Config(), Broken())
    tt.initialize(FRAME, [500, 200, 600, 500], frame_idx=0)
    for i in range(1, 4):
        s = tt.step(FRAME, [_det([500 + 5 * i, 200, 600 + 5 * i, 500])], frame_idx=i)
        assert s.state == TargetState.TRACKING, "the fallback must still track"

    st = tt.stats()
    assert st["encoder_errors"] >= 1, "an encoder that never once succeeded must be counted"
    assert "CUDA out of memory" in (st["encoder_error"] or ""), st["encoder_error"]
    assert st["encoder_error_frame"] >= 0
    assert "encoder_error" in tt._last_diag, "the per-frame diag must carry it too"
    import json

    json.dumps(st)  # the counter must not break the JSONL telemetry


def test_accepted_detection_is_encoded_exactly_once_per_frame():
    """E067: no second forward of the winning box.

    The geometry pass encodes every surviving detection; the appearance pass used
    to call the encoder again for the winner, i.e. the same pixels through the
    same model a second time (94.0 ms/frame on dancetrack0007). Counting encoder
    calls with a stub catches any future refactor that removes the reuse, and the
    per-box count catches one that reuses the *wrong* box.
    """
    tt, emb = _tracker()
    for i in range(1, 6):
        box = [500 + 5 * i, 200, 600 + 5 * i, 500]
        far = [1200 + 5 * i, 200, 1300 + 5 * i, 500]
        emb.calls = 0
        emb.seen.clear()
        s = tt.step(FRAME, [_det(box), _det(far, score=0.8)], frame_idx=i)
        assert s.state == TargetState.TRACKING, (i, s.state, s.diag)
        assert emb.calls == 1, f"frame {i}: {emb.calls} encoder calls for one batch of detections"
        wanted = tuple(np.round(np.asarray(box, np.float32), 1))
        assert emb.seen.count(wanted) == 1, f"frame {i}: accepted box encoded {emb.seen.count(wanted)}x"


def test_empty_and_none_detections_are_tolerated():
    tt, _ = _tracker()
    assert tt.step(FRAME, [], frame_idx=1).frames_since_update == 1
    assert tt.step(FRAME, None, frame_idx=2).frames_since_update == 2


def test_metadata_only_replay_without_pixels():
    """The evaluator replays from JSONL, so no frame is available."""
    tt = TargetTracker(Config(), None)
    tt.initialize(FRAME, [500, 200, 600, 500], frame_idx=0)
    s = tt.step(None, [_det([510, 200, 610, 500])], frame_idx=1, frame_shape=(720, 1280))
    assert s.state == TargetState.TRACKING


# --------------------------------------------------------------------------- #
# search radius while blind  (regression: the gate used to be a bare pixel
# count and stopped growing, so a target that reappeared a few hundred pixels
# away was rejected as `outside_radius` until lost_long_after -- ~2 s at 30 fps)
# --------------------------------------------------------------------------- #

DIAG = float(np.hypot(1280.0, 720.0))  # 1469.3 px, what _associate passes


def _blind_tracker(warm: int = 12, step: float = -8.0):
    """A tracker that has been following a target moving at ``step`` px/frame
    and has then lost it."""
    tt, emb = _tracker_at([700, 200, 800, 500])
    for i in range(1, warm + 1):
        box = [700 + step * i, 200, 800 + step * i, 500]
        tt.step(FRAME, [_det(box)], frame_idx=i)
    assert tt.state is TargetState.TRACKING, "warm-up must keep the target"
    return tt, emb


def test_gate_is_open_while_tracking_and_scales_with_the_frame_when_blind():
    tt, _ = _blind_tracker()
    assert tt._gate_override(DIAG) is None, "a tracked target needs no widened gate"

    tt.step(FRAME, [], frame_idx=100)
    r = tt._gate_override(DIAG)
    # soft_occlusion_after=2, so the gate only opens on the third missed frame.
    for i in range(101, 104):
        tt.step(FRAME, [], frame_idx=i)
    r = tt._gate_override(DIAG)
    # gate_base is a FRACTION of the frame, so on 1080p the base alone is
    # 0.10 * 1469 ~= 147 px, plus 2.5 * 8 px/frame of speed.  The old override
    # returned ~20 px here, which gated out nearly everything.
    assert r > 100.0, r


def test_gate_radius_is_monotone_and_never_saturates_while_blind():
    tt, _ = _blind_tracker()
    radii = []
    for i in range(13, 75):
        tt.step(FRAME, [], frame_idx=i)
        r = tt._gate_override(DIAG)
        if r is not None:  # None for the first two frames (still TRACKING)
            radii.append(r)
    assert all(b >= a - 1e-9 for a, b in zip(radii, radii[1:])), radii
    # The coast-damped speed made this peak (~264 px at fsu=49) and then decay.
    assert radii[-1] > radii[0] + 150.0, (radii[0], radii[-1])


def test_search_is_global_once_the_target_is_only_short_term_lost():
    """A gap of 20 frames must already be searched globally.

    Before the fix ``_global_search`` was armed only past ``lost_long_after``
    (60 frames), i.e. ~2 s of latency on a gap that is visually trivial.
    """
    tt, _ = _blind_tracker()
    for i in range(13, 33):
        tt.step(FRAME, [], frame_idx=i)
    assert tt.state is TargetState.LOST_SHORT
    assert tt.frames_since_update == 20
    assert tt._global_search() is True


def test_reacquires_a_far_reappearance_within_a_few_frames():
    """The user-visible consequence: gap of 20 frames, reappearance ~300 px
    from the prediction. Before the fix this waited for lost_long_after."""
    tt, _ = _blind_tracker()
    for i in range(13, 33):
        tt.step(FRAME, [], frame_idx=i)

    # Identity is keyed on the box's x-centre (< 800 == the target), so the
    # reappearance is still "the same person", just somewhere else.
    far = [130, 200, 230, 500]
    accepted = None
    for i in range(33, 40):
        s = tt.step(FRAME, [_det(far)], frame_idx=i)
        if s.frames_since_update == 0:
            accepted = i - 33
            break
    assert accepted is not None, "the far reappearance was never re-acquired"
    assert accepted <= 2, f"re-acquisition took {accepted} frames after reappearance"
    # One lucky frame is not a re-acquisition: the rebind is held for
    # reacquire_hold frames before the target is promoted back to TRACKING.
    assert tt.state is TargetState.REACQUIRING
    for i in range(40, 43):
        tt.step(FRAME, [_det(far)], frame_idx=i)
    assert tt.state is TargetState.TRACKING

def test_drift_monitor_forces_a_reanchor_when_ltm_distance_climbs() -> None:
    """E021: a confident-and-wrong track must fall back to searching the click.

    The long-term bank is the one memory a stranger cannot be absorbed into, so a
    rising distance to it is the swap signature.  The rule is inert until both
    knobs are set, which is what keeps the shipped default unchanged.
    """

    class _StubLtm:
        def __init__(self, dist: float) -> None:
            self.dist = dist
            self.calls = 0

        def entries(self) -> list:
            return [object()]

        def best(self, feat, part=None, frame_idx=None, touch=False):
            self.calls += 1
            return (self.dist, None)

    def _tracker(window: int, thr) -> TargetTracker:
        tr, _emb = _tracker_at([500, 200, 600, 500])
        tr.cfg = tr.cfg.clone(**{"memory.drift_window": window, "memory.drift_threshold": thr})
        return tr

    det = Detection(xyxy=np.array([500, 200, 600, 500], np.float32), score=0.9, cls=0)
    det.feat = np.array([1, 0, 0, 0], np.float32)

    # inert: window 0
    tr = _tracker(0, 0.3)
    tr.ltm = _StubLtm(0.99)
    for _ in range(10):
        tr._check_drift(det)
    assert tr.state is TargetState.TRACKING
    assert tr.ltm.calls == 0

    # inert: threshold None
    tr = _tracker(4, None)
    tr.ltm = _StubLtm(0.99)
    for _ in range(10):
        tr._check_drift(det)
    assert tr.state is TargetState.TRACKING

    # fires after `window` frames above the threshold, and re-anchors
    tr = _tracker(3, 0.30)
    tr.ltm = _StubLtm(0.55)
    tr._check_drift(det)
    tr._check_drift(det)
    assert tr.state is TargetState.TRACKING, "must not fire before the window is full"
    tr._check_drift(det)
    assert tr.state is TargetState.LOST_SHORT
    assert any(e.kind == "drift" for e in tr.events)
    assert tr._drift_hist == []

    # staying below the threshold never fires
    tr = _tracker(2, 0.60)
    tr.ltm = _StubLtm(0.40)
    for _ in range(20):
        tr._check_drift(det)
    assert tr.state is TargetState.TRACKING

def test_search_guard_requires_a_winning_margin_while_searching(monkeypatch) -> None:
    """E022: while blind, a re-acquisition must WIN by `cost.search_margin`.

    A crowd hands the search many candidates, and a stranger routinely finishes a
    hair ahead of a dozen lookalikes.  Winning by a hair is a coin flip; refusing
    is the honest answer.  The guard lives in `TargetTracker._associate` because
    only it knows the search is a search.
    """
    import tgtrack.target_tracker as tt_mod
    from tgtrack.association import CostBreakdown

    def fake_costs(dets, pred, cfg, **kw):
        n = len(dets)
        total = np.linspace(0.10, 0.14, n).astype(np.float32) if n else np.zeros(0, np.float32)
        return CostBreakdown(
            motion=np.zeros(n, np.float32),
            appearance=total.copy(),
            direction=np.zeros(n, np.float32),
            size=np.zeros(n, np.float32),
            total=total,
            appearance_valid=np.ones(n, bool),
            gate=np.ones(n, bool),
            iou=np.ones(n, np.float32),
            maha=np.zeros(n, np.float32),
            hard_negative=np.zeros(n, np.float32),
            winning_parts=("full",) * n,
            excluded=("",) * n,
        )

    monkeypatch.setattr(tt_mod, "compute_costs", fake_costs)
    dets = [
        Detection(xyxy=np.array([100, 100, 160, 340], np.float32), score=0.9, cls=0),
        Detection(xyxy=np.array([300, 100, 360, 340], np.float32), score=0.9, cls=0),
    ]
    for d, feat in zip(dets, (np.array([1, 0, 0, 0], np.float32), np.array([0.98, 0.2, 0, 0], np.float32))):
        d.feat = feat

    def run(**cost_kw) -> str:
        tr, _emb = _tracker_at([100, 100, 160, 340], distractor_margin=0.0, **cost_kw)
        # force a search state: only the search path drops the motion gate
        tr.state = TargetState.LOST_SHORT
        tr.frames_since_update = 30
        tr.step(FRAME, dets)
        return str(tr._last_diag.get("reason", ""))

    # no guard: a 0.02 lead wins (the old behaviour, and FAR is born here)
    assert not run(search_margin=None).startswith("search_margin")
    # guard: the same photo-finish is refused instead of gambled on
    assert run(search_margin=0.20).startswith("search_margin")

# --------------------------------------------------------------------------- #
# E058 -- anchor bootstrap: averaging the click anchor over confident matches
# --------------------------------------------------------------------------- #
def _bootstrap_tracker(k: int = 0):
    """A bare tracker whose anchor machinery is set up by hand (no harness needed)."""
    import numpy as np

    from tgtrack import Config
    from tgtrack.target_tracker import TargetTracker

    tr = TargetTracker(Config().clone(**{"cost.anchor_bootstrap_frames": int(k)}))
    tr._anchor_seed = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    tr._anchor_feat = tr._anchor_seed.copy()
    tr._last_diag = {}
    tr._bootstrap_feats = []
    tr._bootstrap_left = max(0, int(k))
    return tr


def test_bootstrap_anchor_is_inert_when_the_knob_is_zero():
    import numpy as np

    tr = _bootstrap_tracker(0)
    before = tr._anchor_feat.copy()
    tr._bootstrap_anchor(np.array([0.0, 1.0, 0.0, 0.0], np.float32))
    assert tr._bootstrap_feats == []
    assert np.allclose(tr._anchor_feat, before)


def test_bootstrap_anchor_averages_toward_the_accepted_crops():
    import numpy as np

    tr = _bootstrap_tracker(2)
    tr._bootstrap_anchor(np.array([1.0, 0.2, 0.0, 0.0], np.float32))
    assert tr._bootstrap_left == 1
    assert len(tr._bootstrap_feats) == 1
    # unit norm, and tilted toward the observed crop but still dominated by the click
    assert np.isclose(float(np.linalg.norm(tr._anchor_feat)), 1.0, atol=1e-6)
    assert 0.0 < float(tr._anchor_feat[1]) < 0.2


def test_bootstrap_anchor_stops_after_the_budget_and_ignores_missing_features():
    import numpy as np

    tr = _bootstrap_tracker(1)
    tr._bootstrap_anchor(None)
    assert tr._bootstrap_left == 1  # a missing crop does not spend the budget
    tr._bootstrap_anchor(np.array([0.9, 0.1, 0.0, 0.0], np.float32))
    assert tr._bootstrap_left == 0
    tr._bootstrap_anchor(np.array([0.0, 0.0, 1.0, 0.0], np.float32))
    assert len(tr._bootstrap_feats) == 1  # budget exhausted -> frozen anchor


# --------------------------------------------------------------------------- #
# E071 -- temporal consensus on the handback (docs/E071_PLAN.md)
# --------------------------------------------------------------------------- #
#: The identity cross-over used by these tests: the stub encoder assigns IDENT_A
#: left of this box centre and IDENT_B right of it, so walking a person across it
#: changes WHO they are without changing where they are -- the crossing that E061
#: lost, in its smallest form.
CROSS_X = 700.0

#: Knobs that leave the consensus vote as the ONLY gate between the crossing
#: candidate and the identity.  Both shipped vetoes would otherwise refuse the
#: challenger before the vote ever sees it: an orthogonal descriptor costs
#: 1.0 x w_appearance 1.6 = 1.6 > accept_cost 1.0, and the E029 adaptive anchor
#: band (floor 0.35) vetoes the same candidate on appearance.  That is fine for
#: the *measurement* (the sweep runs the shipped config, not this one) but it
#: would make these tests measure the vetoes instead of the vote.
_CONSENSUS_ONLY = {
    "cost.accept_cost": 2.0,
    "cost.accept_cost_lost": 2.5,
    "cost.anchor_adaptive": False,
    "cost.anchor_max_dist": None,
}


def _crossing_tracker(**overrides):
    """Track a person walking right, across the identity cross-over."""
    cfg = Config().clone(**{**_CONSENSUS_ONLY, **overrides})
    emb = StubEmbedder(split=CROSS_X)
    tt = TargetTracker(cfg, emb)
    tt.initialize(FRAME, [500, 200, 600, 500], frame_idx=0)
    return tt


def _walk(tt, n, *, start=500.0, dx=20.0, first=1):
    """Step ``n`` frames of the walk; returns ``(frame_idx, x, status)`` rows."""
    rows = []
    x = start
    for i in range(first, first + n):
        x += dx
        rows.append((i, x, tt.step(FRAME, [_det([x, 200, x + 100, 500])], frame_idx=i)))
    return rows


def test_consensus_off_is_indistinguishable_from_the_shipped_path():
    """The master switch must be inert when off.

    The reference is the module's own ``frames_required=1`` control, which
    commits on the first proposal -- i.e. exactly the pre-E071 rule.  Every
    frame must agree on state, acceptance and reported box, and the mechanism
    must leave no trace in the diagnostics or the counters.
    """
    shipped = _crossing_tracker()
    control = _crossing_tracker(**{"cost.commit_consensus": True, "cost.consensus_frames": 1})
    for a, b in zip(_walk(shipped, 12), _walk(control, 12)):
        assert a[2].state == b[2].state, a[0]
        assert a[2].diag["accepted"] == b[2].diag["accepted"], a[0]
        assert np.allclose(a[2].xyxy, b[2].xyxy, atol=1e-4), a[0]

    st = shipped.stats()
    assert st["challenge_enabled"] is False
    assert st["challenge_opened"] == 0
    assert st["challenge_committed"] == 0
    assert st["challenge_pending_frames"] == 0
    assert [e.kind for e in shipped.events if e.kind.startswith("consensus")] == []
    assert "challenge_state" not in shipped.status().diag


def test_consensus_refuses_a_challenger_until_it_commits():
    """A non-incumbent may only take the identity once the vote commits."""
    tt = _crossing_tracker(**{"cost.commit_consensus": True, "cost.consensus_frames": 2})
    rows = _walk(tt, 8)  # frame 8 is where the descriptor flips
    frame, _, pending = rows[-1]
    assert pending.diag["accepted"] is False
    assert pending.diag["reason"] == "consensus_pending"
    assert pending.diag["challenge_state"] == "pending"
    assert pending.state == TargetState.TRACKING
    assert pending.frames_since_update == 0, "a pending challenge is not a miss"
    hits_before = tt.hits
    assert tt.stats()["challenge_opened"] == 1
    assert tt.stats()["challenge_committed"] == 0

    _, _, committed = _walk(tt, 1, start=640.0, first=frame + 1)[0]
    assert committed.diag["accepted"] is True
    assert tt.hits == hits_before + 1, "only the committed frame is a hit"
    st = tt.stats()
    assert st["challenge_committed"] == 1
    assert st["challenge_pending_frames"] == 1
    kinds = [e.kind for e in tt.events]
    assert kinds.count("consensus_pending") == 1
    assert kinds.count("consensus_commit") == 1


def test_consensus_pending_frames_do_not_escalate_the_state_machine():
    """Three frames without a commit, still TRACKING with fsu 0.

    Counting a pending challenge as a miss would walk the ladder
    TRACKING -> SOFT_OCCLUSION after ``soft_occlusion_after`` frames and freeze
    appearance memory exactly when the challenger's crops are the evidence the
    vote needs (docs/E071_PLAN.md, trap 1).
    """
    tt = _crossing_tracker(**{"cost.commit_consensus": True, "cost.consensus_frames": 3})
    _walk(tt, 7)  # everything before the cross-over
    x = 640.0
    for frame in (8, 9):
        x += 20.0
        s = tt.step(FRAME, [_det([x, 200, x + 100, 500])], frame_idx=frame)
        assert s.diag["reason"] == "consensus_pending"
        assert s.frames_since_update == 0
        assert s.state == TargetState.TRACKING
    assert tt.stats()["challenge_pending_frames"] == 2
    assert tt.stats()["challenge_committed"] == 0


def test_snapshot_restore_preserves_an_open_challenge():
    """The half-built vote is state: a round-trip must not lose it."""
    tt = _crossing_tracker(**{"cost.commit_consensus": True, "cost.consensus_frames": 2})
    _walk(tt, 8)
    assert tt.stats()["challenge_pending_frames"] == 1
    snap = tt.snapshot()
    assert snap["challenge"]["chain_frames"] == [8]
    assert snap["challenge"]["counters"]["n_opened"] == 1

    # run past the commit, then rewind to the pending frame
    _, _, committed = _walk(tt, 1, start=640.0, first=9)[0]
    assert committed.diag["accepted"] is True
    assert tt.stats()["challenge_committed"] == 1
    tt.restore(snap)
    st = tt.stats()
    assert st["challenge_opened"] == 1
    assert st["challenge_committed"] == 0
    assert st["challenge_pending_frames"] == 1

    # ...and the restored chain still commits, instead of opening a fresh one
    _, _, again = _walk(tt, 1, start=640.0, first=9)[0]
    assert again.diag["accepted"] is True
    assert tt.stats()["challenge_committed"] == 1
    assert tt.stats()["challenge_opened"] == 1, "restore must not double-count openings"


def test_consensus_classifies_a_candidate_without_evidence_as_incumbent():
    """No descriptor (encoder failure) or no reference (first frame) -> no vote.

    Refusing on absent evidence is E067's failure mode wearing a new hat, and
    "no incumbent to keep" is the documented fallback to the shipped path.
    """
    tt = _crossing_tracker(**{"cost.commit_consensus": True, "cost.consensus_frames": 2})
    _walk(tt, 3)
    assert tt._consensus_is_incumbent(IDENT_A) is True
    assert tt._consensus_is_incumbent(IDENT_B) is False
    assert tt._consensus_is_incumbent(None) is True

