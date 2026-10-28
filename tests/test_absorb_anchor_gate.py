"""E078 -- the absorb-ratio guard on long-term gallery admission.

The mechanism under test (docs/E078_PLAN in EXPERIMENTS.md): once the tracker locks
onto the wrong person, the long-term gallery absorbs the impostor and every later
frame "confirms" it.  The guard admits a crop into the gallery only while its
descriptor sits inside a *tight* band around the frozen click anchor, so a drift
cannot poison the memory that would otherwise detect it.

What must remain true while the guard is on:

* the click crop itself is always recorded (it IS the anchor);
* short-term memory and the part EMA keep working (a legitimately changed look must
  still be re-acquirable);
* hard negatives keep their own path -- an impostor is still remembered as a
  *negative*, which is the half of the memory that must not be muted;
* with the flag off (the shipped default) nothing changes at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import Config
from tgtrack.target_tracker import TargetTracker
from tgtrack.types import Detection

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
BOX = [500.0, 200.0, 600.0, 500.0]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / float(np.linalg.norm(v))


ANCHOR = _unit([1.0, 0.0, 0.0, 0.0])


def _vec_at(dist: float) -> np.ndarray:
    """A unit 4-vector whose *cosine distance* from ``ANCHOR`` is ``dist``."""
    d = float(np.clip(dist, 0.0, 2.0))
    theta = float(np.arccos(np.clip(1.0 - d, -1.0, 1.0)))
    return _unit([np.cos(theta), np.sin(theta), 0.0, 0.0])


class RampEmbedder:
    """Deterministic encoder whose descriptor distance to the anchor is box-keyed.

    ``table`` maps a box x-centre to the cosine distance the crop should have from
    the click anchor, so a test can drive one crop to a distance it names instead
    of hoping a real ReID net cooperates.
    """

    def __init__(self, table: dict[float, float]) -> None:
        self.table = dict(table)
        self.calls = 0

    def dist_for(self, box) -> float:
        cx = (float(box[0]) + float(box[2])) / 2.0
        key = min(self.table, key=lambda k: abs(float(k) - cx))
        return float(self.table[key])

    def encode(self, img_bgr, boxes_xyxy, parts=("full",)):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        self.calls += 1
        out = []
        for b in boxes:
            f = _vec_at(self.dist_for(b))
            out.append({p: f.copy() for p in parts})
        return out


def _tracker(embedder: RampEmbedder | None = None, **overrides) -> TargetTracker:
    """Tracker clicked on ``BOX`` with a FIXED 0.40 anchor band unless overridden."""
    cfg = Config()
    cfg = cfg.clone(
        **{
            "cost.anchor_adaptive": False,
            "cost.anchor_max_dist": 0.40,
            **overrides,
        }
    )
    emb = embedder or RampEmbedder({550.0: 0.0})
    tt = TargetTracker(cfg, emb)
    tt.initialize(FRAME, BOX, frame_idx=0)
    return tt


def _absorb(tt: TargetTracker, cx: float) -> None:
    box = np.array([cx - 50.0, 200.0, cx + 50.0, 500.0], np.float32)
    tt.frame_idx += 1
    tt._absorb(FRAME, box, score=0.9, quality=1.0, force=False)


# --------------------------------------------------------------------------- #
# inert by default
# --------------------------------------------------------------------------- #
def test_absorb_guard_is_inert_by_default() -> None:
    """The shipped default must not change what the gallery absorbs."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 1.0})
    tt = _tracker(emb)
    assert tt.cfg.memory.ltm_absorb_anchor_only is False
    before = len(tt.ltm)
    _absorb(tt, 900.0)  # a crop the guard WOULD refuse (distance 1.0 >> band)
    assert len(tt.ltm) > before, "with the guard off the far crop must still be absorbed"
    assert "absorb_anchor_blocked" not in tt._last_diag or not tt._last_diag.get(
        "absorb_anchor_blocked"
    )


def test_absorb_guard_shipped_defaults_are_off_and_1_25() -> None:
    mem = Config().memory
    assert mem.ltm_absorb_anchor_only is False
    assert mem.ltm_absorb_anchor_ratio == pytest.approx(1.25)


# --------------------------------------------------------------------------- #
# the guard itself
# --------------------------------------------------------------------------- #
def test_absorb_guard_refuses_a_crop_outside_the_band() -> None:
    # band 0.40 * ratio 1.25 = 0.50
    emb = RampEmbedder({550.0: 0.0, 700.0: 0.30, 900.0: 1.00})
    tt = _tracker(emb, **{"memory.ltm_absorb_anchor_only": True})
    before = len(tt.ltm)
    _absorb(tt, 700.0)  # distance 0.30: inside the band, outside the diversity gate
    inside = len(tt.ltm)
    assert inside > before
    assert tt._last_diag["absorb_anchor_limit"] == pytest.approx(0.50, abs=1e-6)
    _absorb(tt, 900.0)  # distance 1.00: outside the band
    assert len(tt.ltm) == inside, "an out-of-band crop must not enter the gallery"
    assert tt._last_diag["absorb_anchor_blocked"] == 1
    assert tt._last_diag["absorb_anchor_dist"] > tt._last_diag["absorb_anchor_limit"]


def test_absorb_guard_ratio_is_monotone() -> None:
    """The same crop is refused by the tight ratio and admitted by the loose one."""
    emb = RampEmbedder({550.0: 0.0, 850.0: 0.70})
    # 0.40 * 1.25 = 0.50 < 0.70 -> blocked
    tight = _tracker(emb, **{"memory.ltm_absorb_anchor_only": True})
    before = len(tight.ltm)
    _absorb(tight, 850.0)
    assert len(tight.ltm) == before
    # 0.40 * 2.50 = 1.00 > 0.70 -> admitted
    loose = _tracker(
        emb,
        **{"memory.ltm_absorb_anchor_only": True, "memory.ltm_absorb_anchor_ratio": 2.50},
    )
    before = len(loose.ltm)
    _absorb(loose, 850.0)
    assert len(loose.ltm) > before


def test_absorb_guard_still_records_the_click_anchor() -> None:
    """``initialize`` absorbs with force=True: the anchor must always be stored."""
    tt = _tracker(**{"memory.ltm_absorb_anchor_only": True, "memory.ltm_absorb_anchor_ratio": 0.0})
    assert len(tt.ltm) > 0
    assert set(tt.ltm.parts()) >= {"full", "upper", "lower", "head"}


def test_blocked_absorb_still_feeds_short_term_memory_and_part_ema() -> None:
    """Only the gallery is protected; the re-acquisition machinery keeps its evidence."""
    emb = RampEmbedder({550.0: 0.0, 900.0: 1.00})
    tt = _tracker(emb, **{"memory.ltm_absorb_anchor_only": True})
    before_gallery = len(tt.ltm)
    before_stm = np.asarray(tt.stm.current(), np.float32).copy()
    _absorb(tt, 900.0)
    assert len(tt.ltm) == before_gallery, "the gallery is the only thing gated"
    stm = np.asarray(tt.stm.current(), np.float32)
    assert not np.allclose(stm, before_stm), "STM must follow the new look"
    assert "full" in tt._part_ema, "the part EMA must still update"


# --------------------------------------------------------------------------- #
# the negative bank is NOT gated
# --------------------------------------------------------------------------- #
def test_impostor_is_still_remembered_as_a_hard_negative() -> None:
    """An impostor must go into the negative bank even while the gallery refuses it.

    This is the half of the mechanism that keeps a swap detectable: the guard stops
    the gallery from calling the impostor the target, it must never stop the tracker
    from remembering him as *somebody else*.
    """
    emb = RampEmbedder({550.0: 0.0, 900.0: 1.00})
    tt = _tracker(emb, **{"memory.ltm_absorb_anchor_only": True})
    gallery_before = len(tt.ltm)
    _absorb(tt, 900.0)
    assert len(tt.ltm) == gallery_before

    # Drive the look-alike path directly: the impostor is a *look-alike* of the crop we
    # accepted (that is what puts him in the bank), and both are far outside the band,
    # so the gallery refuses them while the negative bank keeps learning.
    chosen = Detection(xyxy=np.array([500.0, 200.0, 600.0, 500.0], np.float32), score=0.9, cls=0)
    chosen.feat = _vec_at(0.90)
    impostor = Detection(xyxy=np.array([850.0, 200.0, 950.0, 500.0], np.float32), score=0.9, cls=0)
    impostor.feat = _vec_at(0.95)
    assert 1.0 - float(np.dot(chosen.feat, impostor.feat)) <= 0.35, "must be confusable"
    tt._frame_dets = [chosen, impostor]
    tt.cfg.memory.hard_negative_confident_only = False
    assert tt._absorb_allowed(chosen.feat) is False, "the guard must refuse this crop"
    before_negatives = tt.ltm.num_hard_negatives
    tt._memorize_lookalikes(chosen)
    assert tt.ltm.num_hard_negatives > before_negatives, (
        "the negative bank must keep learning while the gallery is gated"
    )


# --------------------------------------------------------------------------- #
# band resolution
# --------------------------------------------------------------------------- #
def test_absorb_band_uses_the_adaptive_band_when_observable() -> None:
    tt = _tracker(**{"cost.anchor_adaptive": True, "cost.anchor_adaptive_ceil": 0.75})
    for d in (0.10, 0.12, 0.11, 0.13, 0.10, 0.12):
        tt._anchor_dist_hist.append(d)
    # median 0.115, MAD ~0.01 -> med + 4*1.4826*MAD stays under the 0.75 ceiling
    band = tt._absorb_band()
    assert band is not None and 0.10 <= band <= 0.75


def test_absorb_band_falls_back_to_the_configured_ceiling_in_bootstrap() -> None:
    """While the adaptive band is unobservable, the guard still has one meaning.

    ``_anchor_limit`` already prefers the fixed ``anchor_max_dist`` during bootstrap
    (E027), so the ceiling only has to cover the case where no fixed bound is set.
    """
    tt = _tracker(
        **{
            "cost.anchor_adaptive": True,
            "cost.anchor_adaptive_ceil": 0.75,
            "cost.anchor_max_dist": None,
        }
    )
    assert tt._anchor_dist_hist == []
    assert tt._absorb_band() == pytest.approx(0.75)

    fixed = _tracker(**{"cost.anchor_adaptive": False, "cost.anchor_max_dist": None})
    assert fixed._absorb_band() is None


def test_absorb_guard_admits_everything_without_an_anchor() -> None:
    """No embedder -> no anchor -> no evidence -> no veto (never a blind refusal)."""
    cfg = Config().clone(**{"memory.ltm_absorb_anchor_only": True})
    tt = TargetTracker(cfg, None)
    tt.initialize(FRAME, np.asarray(BOX, np.float32), frame_idx=0)
    assert tt._anchor_feat is None
    assert tt._absorb_allowed(_vec_at(1.0)) is True


# --------------------------------------------------------------------------- #
# engagement counters (E071 precedent: a rule that never fires is not a result)
# --------------------------------------------------------------------------- #
def test_absorb_guard_exposes_engagement_counters_in_stats() -> None:
    emb = RampEmbedder({550.0: 0.0, 700.0: 0.30, 900.0: 1.00})
    tt = _tracker(emb, **{"memory.ltm_absorb_anchor_only": True})
    _absorb(tt, 700.0)  # allowed
    _absorb(tt, 900.0)  # blocked
    stats = tt.stats()
    assert stats["absorb_anchor_only"] is True
    assert stats["absorb_anchor_ratio"] == pytest.approx(1.25)
    assert stats["absorb_anchor_gated"] == 2
    assert stats["absorb_anchor_blocked"] == 1

    # With the flag off the counter stays 0 and reports that, rather than a blank.
    off = _tracker(RampEmbedder({550.0: 0.0, 900.0: 1.00}))
    _absorb(off, 900.0)
    stats = off.stats()
    assert stats["absorb_anchor_only"] is False
    assert stats["absorb_anchor_gated"] == 0
    assert stats["absorb_anchor_blocked"] == 0
