"""Tests for :mod:`tgtrack.memory` — the short- and long-term appearance banks.

These tests pin the *policies*, not just the plumbing: the EMA, the staleness
clock, the diversity gate, the quality*recency eviction, the freeze guard, the
part-aware query, and the hard-negative penalty. Those policies are what the
experiments tune, so they need regression protection.
"""

from __future__ import annotations

import numpy as np
import pytest

from tgtrack.config import MemoryCfg
from tgtrack.memory import GalleryEntry, LongTermGallery, ShortTermMemory


def cfg(**kw) -> MemoryCfg:
    c = MemoryCfg()
    for k, v in kw.items():
        assert hasattr(c, k), f"unknown MemoryCfg field {k}"
        setattr(c, k, v)
    return c


def basis(i: int, dim: int = 8) -> np.ndarray:
    """Unit basis vector — mutually orthogonal, so cosine distance is 1.0."""
    v = np.zeros((dim,), dtype=np.float32)
    v[i] = 1.0
    return v


# --------------------------------------------------------------------------- STM
def test_stm_ema_converges_toward_constant_stream():
    stm = ShortTermMemory(cfg(stm_alpha=0.9, stm_len=16))
    assert stm.current() is None
    assert stm.update(basis(0), quality=1.0, frame_idx=0) is True
    target = basis(1)
    for i in range(1, 60):
        stm.update(target, quality=1.0, frame_idx=i)
    cur = stm.current()
    assert cur is not None
    assert cur.shape == (8,)
    assert float(np.linalg.norm(cur)) == pytest.approx(1.0, abs=1e-5)
    # Cosine similarity to the dominant stream must be ~1 after 60 updates.
    assert float(cur @ target) > 0.999
    assert float(cur @ basis(0)) < 0.05


def test_stm_alpha_one_keeps_the_first_observation():
    stm = ShortTermMemory(cfg(stm_alpha=1.0))
    stm.update(basis(0), quality=1.0, frame_idx=0)
    for i in range(1, 10):
        stm.update(basis(3), quality=1.0, frame_idx=i)
    assert np.allclose(stm.current(), basis(0))


def test_stm_ring_buffer_is_bounded_and_chronological():
    stm = ShortTermMemory(cfg(stm_len=4))
    for i in range(7):
        stm.update(basis(i % 8), quality=1.0, frame_idx=i)
    assert len(stm) == 4
    recent = stm.recent(k=8)
    assert recent.shape == (4, 8)
    # Oldest first: frames 3,4,5,6 survived in a maxlen=4 deque.
    assert np.allclose(recent[0], basis(3))
    assert np.allclose(recent[-1], basis(6))
    assert stm.recent(k=2).shape == (2, 8)
    weighted = stm.mean_recent(k=4)
    assert weighted is not None and float(np.linalg.norm(weighted)) == pytest.approx(1.0, abs=1e-5)


def test_stm_stale_uses_stm_stale_frames():
    stm = ShortTermMemory(cfg(stm_stale_frames=90))
    assert stm.is_stale(0) is True  # never updated -> stale
    stm.update(basis(0), quality=1.0, frame_idx=100)
    assert stm.is_stale(100) is False
    assert stm.is_stale(190) is False  # exactly at the limit
    assert stm.is_stale(191) is True
    assert stm.age(120) == 20
    stm.reset()
    assert stm.current() is None and len(stm) == 0 and stm.is_stale(1000) is True


def test_stm_rejects_zero_norm_nan_and_bad_quality():
    stm = ShortTermMemory(cfg())
    assert stm.update(np.zeros(8), 1.0, 0) is False
    assert stm.update(np.full(8, np.nan), 1.0, 0) is False
    assert stm.update(basis(0), 0.0, 0) is False  # unusable crop
    assert stm.update(basis(0), float("nan"), 0) is False
    assert len(stm) == 0 and stm.current() is None and stm.last_frame is None
    # A rejected update must not advance the staleness clock.
    assert stm.update(basis(0), 1.0, 10) is True
    assert stm.update(np.zeros(8), 1.0, 500) is False
    assert stm.is_stale(60) is False


def test_stm_dim_is_locked_after_first_update():
    stm = ShortTermMemory(cfg())
    assert stm.update(basis(0, dim=16), 1.0, 0) is True
    assert stm.dim == 16
    assert stm.update(basis(0, dim=8), 1.0, 1) is False  # wrong dim rejected


# --------------------------------------------------------------------------- LTM
def test_ltm_diversity_gate_rejects_near_duplicate_accepts_orthogonal():
    g = LongTermGallery(cfg(ltm_diversity_thresh=0.15, ltm_min_quality=0.0, ltm_half_life=None))
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert g.add(v, part="full", quality=1.0, frame_idx=0) is True
    assert g.add(v, part="full", quality=1.0, frame_idx=1) is False  # distance 0 -> refused
    near = np.array([0.99, 0.05, 0.0, 0.0], dtype=np.float32)  # distance ~0.001 -> refused
    assert g.add(near, part="full", quality=1.0, frame_idx=2) is False
    assert len(g) == 1
    orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # distance 1.0 -> accepted
    assert g.add(orth, part="full", quality=1.0, frame_idx=3) is True
    assert len(g) == 2


def test_ltm_diversity_gate_is_per_part():
    g = LongTermGallery(cfg(ltm_diversity_thresh=0.15, ltm_min_quality=0.0))
    v = basis(0)
    assert g.add(v, part="full", quality=1.0, frame_idx=0) is True
    # Same vector, different part key: a different bank, so not a duplicate.
    assert g.add(v, part="upper", quality=1.0, frame_idx=1) is True
    assert sorted(g.stats()["per_part"].items()) == [
        ("full", 1),
        ("head", 0),
        ("lower", 0),
        ("torso", 0),
        ("upper", 1),
    ]


def test_ltm_min_quality_gate():
    g = LongTermGallery(cfg(ltm_min_quality=0.35))
    assert g.add(basis(0), part="full", quality=0.2, frame_idx=0) is False
    assert g.add(basis(0), part="full", quality=0.9, frame_idx=0) is True
    assert len(g) == 1


def test_ltm_capacity_eviction_removes_lowest_quality_recency():
    g = LongTermGallery(
        cfg(ltm_capacity=2, ltm_diversity_thresh=0.15, ltm_min_quality=0.0, ltm_half_life=None)
    )
    assert g.add(basis(0), "full", quality=0.9, frame_idx=0) is True
    assert g.add(basis(1), "full", quality=0.4, frame_idx=0) is True
    assert g.add(basis(2), "full", quality=0.7, frame_idx=0) is True
    assert len(g) == 2
    kept = [e.feat for e in g.entries()]
    assert not any(np.allclose(f, basis(1)) for f in kept), "lowest quality entry should be evicted"
    assert any(np.allclose(f, basis(0)) for f in kept)
    assert any(np.allclose(f, basis(2)) for f in kept)


def test_ltm_eviction_weighs_recency_with_half_life():
    g = LongTermGallery(
        cfg(ltm_capacity=2, ltm_diversity_thresh=0.15, ltm_min_quality=0.0, ltm_half_life=300.0)
    )
    g.add(basis(0), "full", quality=0.8, frame_idx=0)  # stale but higher quality
    g.add(basis(1), "full", quality=0.9, frame_idx=1900)
    g.add(basis(2), "full", quality=0.1, frame_idx=2000)  # forces an eviction at frame 2000
    kept = [e.feat for e in g.entries()]
    assert not any(np.allclose(f, basis(0)) for f in kept), "stale entry should lose on quality*recency"


def test_ltm_fifo_replace_policy():
    g = LongTermGallery(
        cfg(ltm_capacity=2, ltm_diversity_thresh=0.15, ltm_min_quality=0.0, ltm_replace_policy="fifo")
    )
    g.add(basis(0), "full", quality=1.0, frame_idx=10)
    g.add(basis(1), "full", quality=1.0, frame_idx=20)
    g.add(basis(2), "full", quality=1.0, frame_idx=30)
    kept = [e.feat for e in g.entries()]
    assert not any(np.allclose(f, basis(0)) for f in kept), "FIFO must drop the oldest frame_idx"
    assert len(g) == 2


def test_ltm_freeze_blocks_inserts():
    g = LongTermGallery(cfg(ltm_min_quality=0.0))
    g.freeze(True)
    assert g.frozen is True
    assert g.add(basis(0), "full", quality=1.0, frame_idx=0) is False
    assert len(g) == 0
    assert g.freeze(False) is None
    assert g.add(basis(0), "full", quality=1.0, frame_idx=0) is True
    assert len(g) == 1


def test_ltm_part_aware_query_only_matches_the_same_part():
    g = LongTermGallery(cfg(part_aware=True, ltm_diversity_thresh=0.0, ltm_min_quality=0.0))
    full_feat = basis(0)
    upper_feat = basis(1)
    g.add(full_feat, "full", quality=1.0, frame_idx=0)
    g.add(upper_feat, "upper", quality=1.0, frame_idx=1)

    # An 'upper' query must ignore the (closer!) full-body entry entirely.
    hits = g.query(full_feat, part="upper", k=5)
    assert len(hits) == 1 and hits[0][1].part == "upper"
    d, entry = g.best(full_feat, part="upper")
    assert entry is not None and entry.part == "upper"
    assert d == pytest.approx(1.0, abs=1e-6)  # orthogonal, not the 0.0 of the full entry

    # No bank for the requested part -> fall back to the full-body bank.
    hits = g.query(upper_feat, part="head", k=5)
    assert [e.part for _, e in hits] == ["full"]

    # part=None searches everything.
    assert len(g.query(full_feat, part=None, k=5)) == 2
    # part_aware=False searches everything as well.
    g.cfg.part_aware = False
    assert len(g.query(full_feat, part="upper", k=5)) == 2


def test_ltm_hard_negative_penalty_is_smooth_and_bounded():
    g = LongTermGallery(cfg(hard_negative_capacity=8, part_aware=True, ltm_min_quality=0.0))
    assert g.hard_negative_penalty(basis(0)) == 0.0  # empty bank -> no penalty
    assert g.add_hard_negative(basis(0), frame_idx=0) is True
    assert g.num_hard_negatives == 1
    # The stored distractor direction is maximally penalized.
    assert g.hard_negative_penalty(basis(0)) == pytest.approx(1.0, abs=1e-6)
    # An orthogonal probe must score exactly 0 (below the smoothstep's lower knee).
    assert g.hard_negative_penalty(basis(1)) == 0.0
    # Monotone in between: a 45-degree probe sits strictly inside (0, 1).
    mid = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mid_pen = g.hard_negative_penalty(mid)
    assert 0.0 < mid_pen < 1.0
    # Invalid probes are safe.
    assert g.hard_negative_penalty(np.full(8, np.nan)) == 0.0
    assert g.add_hard_negative(np.zeros(8), frame_idx=1) is False
    # Duplicates are not stored twice; the bank respects its capacity (FIFO).
    assert g.add_hard_negative(basis(0), frame_idx=2) is False
    for i in range(20):
        g.add_hard_negative(basis(i % 8), frame_idx=i)
    assert g.num_hard_negatives <= 8


def test_ltm_hard_negative_bank_can_be_disabled():
    g = LongTermGallery(cfg(hard_negative_bank=False))
    assert g.add_hard_negative(basis(0), frame_idx=0) is False
    assert g.hard_negative_penalty(basis(0)) == 0.0


def test_ltm_best_on_empty_gallery_is_infinite():
    g = LongTermGallery(cfg())
    d, entry = g.best(basis(0))
    assert entry is None and d == float("inf")
    assert g.query(basis(0)) == []


def test_ltm_rejects_bad_features_and_dims():
    g = LongTermGallery(cfg(ltm_min_quality=0.0))
    assert g.add(np.zeros(8), "full", 1.0, 0) is False
    assert g.add(np.full(8, np.nan), "full", 1.0, 0) is False
    assert g.add(np.array([], dtype=np.float32), "full", 1.0, 0) is False
    assert g.add(basis(0, dim=8), "full", 1.0, 0) is True
    assert g.add(basis(0, dim=4), "full", 1.0, 1) is False  # dim mismatch
    assert len(g) == 1
    # A query with a mismatched dim simply cannot match.
    assert g.query(basis(0, dim=4)) == []


def test_ltm_unknown_part_key_is_rejected():
    g = LongTermGallery(cfg())
    with pytest.raises(KeyError):
        g.add(basis(0), "left_arm", 1.0, 0)
    with pytest.raises(KeyError):
        g.query(basis(0), part="left_arm")


def test_ltm_query_touch_updates_use_count_and_recency():
    g = LongTermGallery(cfg(ltm_min_quality=0.0, ltm_half_life=100.0))
    g.add(basis(0), "full", quality=1.0, frame_idx=0)
    hits = g.query(basis(0), k=1, frame_idx=50, touch=True)
    assert len(hits) == 1
    entry = hits[0][1]
    assert entry.use_count == 1
    assert entry.recency == pytest.approx(0.5 ** 0.5, rel=1e-6)
    g.query(basis(0), k=1, frame_idx=50, touch=False)
    assert entry.use_count == 1, "touch=False must not mutate the gallery"


def test_ltm_stats_and_snapshot_round_trip():
    g = LongTermGallery(cfg(ltm_capacity=8, ltm_min_quality=0.0, hard_negative_capacity=4))
    g.add(basis(0), "full", quality=0.8, frame_idx=0, meta={"score": 0.9})
    g.add(basis(1), "upper", quality=0.6, frame_idx=5)
    g.add_hard_negative(basis(2), frame_idx=6)
    st = g.stats()
    assert st["size"] == 2 and st["capacity"] == 8 and st["dim"] == 8
    assert st["per_part"]["full"] == 1 and st["per_part"]["upper"] == 1
    assert st["mean_quality"] == pytest.approx(0.7)
    assert st["hard_negatives"] == 1
    assert st["frozen"] is False

    snap = g.snapshot()
    probe = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    before = g.best(probe, part="full")[0]
    g.freeze(True)
    g.clear()
    assert len(g) == 0 and g.num_hard_negatives == 0
    g.restore(snap)
    assert len(g) == 2
    assert g.num_hard_negatives == 1
    assert g.frozen is False, "restore must reinstate the snapshotted freeze flag"
    assert g.entries()[0].meta == {"score": 0.9}
    assert g.best(probe, part="full")[0] == pytest.approx(before, abs=1e-6)
    assert g.hard_negative_penalty(basis(2)) == pytest.approx(1.0, abs=1e-6)


def test_gallery_entry_copy_is_independent():
    e = GalleryEntry(feat=basis(0), part="full", quality=0.5, frame_idx=3, meta={"a": 1})
    c = e.copy()
    c.feat[0] = 99.0
    c.meta["a"] = 2
    assert e.feat[0] == 1.0 and e.meta["a"] == 1


def test_ltm_diversity_gate_refresh_upgrades_the_stored_entry():
    """A refused near-duplicate must REFRESH the bank, not vanish.

    Regression for a real defect: ``ltm_diversity_thresh`` is a cosine
    *distance*, so ``dist <= 0.15`` rejects everything with similarity > 0.85 --
    and our own measured same-person part similarities are upper 0.920, torso
    0.941, head 1.000, i.e. all inside the reject zone. Dropping refused inserts
    meant the best views of the target were thrown away and every part bank
    stayed pinned to whatever frame 0 looked like, with no upgrade path.
    """
    g = LongTermGallery(cfg(ltm_diversity_thresh=0.15, ltm_min_quality=0.0, ltm_half_life=None))
    v = basis(0)
    assert g.add(v, part="upper", quality=0.40, frame_idx=0, meta={"src": "frame0"}) is True
    assert g.last_add_action == "added"

    # cosine 0.90 with v -> distance 0.10, inside the 0.15 reject zone
    better = (0.90 * v + 0.436 * basis(1)).astype(np.float32)
    assert g.add(better, part="upper", quality=0.95, frame_idx=7, meta={"src": "frame7"}) is False

    assert len(g) == 1, "a near-duplicate must not create a second slot"
    e = g.entries()[0]
    assert e.quality == pytest.approx(0.95), "the better view must upgrade the stored quality"
    assert e.frame_idx == 7, "a refresh counts as seeing the target again"
    assert e.meta["src"] == "frame7"
    assert g.last_add_action == "refreshed"
    assert float(np.linalg.norm(e.feat)) == pytest.approx(1.0, abs=1e-5)
    # the descriptor genuinely moved towards the better view
    assert float(np.dot(e.feat, better)) > float(np.dot(v, better)) + 1e-3

    # a genuinely novel view still earns its own slot (the gate keeps its purpose)
    assert g.add(basis(1), part="upper", quality=0.50, frame_idx=8) is True
    assert len(g) == 2

    # a *worse* near-duplicate must not drag the stored quality back down
    assert g.add(basis(1) * 1.0, part="upper", quality=0.10, frame_idx=9) is False
    assert max(x.quality for x in g.entries()) == pytest.approx(0.95)


def test_ltm_freeze_stops_recency_decaying():
    """Frozen frames must not age the bank.

    Regression for a real defect: ``_recency`` decayed from the ABSOLUTE frame
    index, so while the target was occluded (gallery correctly frozen) the
    entries kept ageing. After ``~ltm_half_life`` frames of occlusion the
    ``quality * recency`` weighting silently degenerates to quality-only and the
    eviction policy changes behaviour with no parameter changed.
    """
    g = LongTermGallery(
        cfg(ltm_capacity=8, ltm_diversity_thresh=0.0, ltm_min_quality=0.0, ltm_half_life=100.0)
    )
    quals = (0.90, 0.50, 0.70)
    for i, q in enumerate(quals):
        assert g.add(basis(i), part="full", quality=q, frame_idx=0)

    def eviction_scores(now: int) -> list[float]:
        """Exactly the expression ``_eviction_index`` ranks entries by."""
        return [float(e.quality) * g._recency(e, now) for e in g.entries()]

    eviction_scores(0)  # anchor the gallery clock at frame 0
    live = eviction_scores(50)  # 50 live frames -> half of one half-life
    assert live == pytest.approx([q * 0.5**0.5 for q in quals])

    g.freeze(True)
    frozen = eviction_scores(5050)  # 5000 frames the gallery never observed
    assert frozen == pytest.approx(live), "a frozen gallery must not age"
    assert g._eviction_index(5050) == g._eviction_index(50), "eviction order must not drift"

    # and ageing resumes - only for frames the gallery is actually live for
    g.freeze(False)
    eviction_scores(5060)  # re-anchors the clock, no gap is charged
    resumed = eviction_scores(5070)  # 10 genuinely live frames
    assert resumed == pytest.approx([s * 0.5**0.1 for s in live], rel=1e-6)
