"""Short- and long-term appearance memory for the selected target.

This is the novelty core of the project. The all-person MOT layer only needs
enough appearance to keep IDs apart; the *selected* target needs a memory that
survives occlusion, out-of-frame gaps and partial visibility without drifting
onto a look-alike. Two banks do that:

* :class:`ShortTermMemory` — an EMA of the last few accepted embeddings plus a
  ring buffer. Fast, aggressive, and exactly what
  ``ultralytics.trackers.utils.reid.smooth_feature`` produces. This is what
  keeps the box locked on during continuous tracking.
* :class:`LongTermGallery` — a capacity-bounded, diversity-gated,
  quality-and-recency-weighted bank of *distinct appearance modes*, optionally
  split per body part. This is what re-identifies the target after a long gap
  and what lets an upper-body-only query match an upper-body gallery entry.
* a hard-negative bank of confirmed distractors, turned into a smooth penalty
  that suppresses ID swaps between similar-looking people.

Policies are declarative (see ``MemoryCfg``) because the strategy, not the
model, is what the experiments tune.

Invariants enforced everywhere:
    * features are finite, L2-normalized ``float32`` of a single fixed dim;
    * zero-norm / NaN / wrong-dim features never enter a bank;
    * ``freeze(True)`` makes inserts no-ops (used while the target is occluded
      or lost, so a distractor cannot poison the gallery).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MemoryCfg
from .types import PART_KEYS, cosine_distance_matrix

__all__ = ["GalleryEntry", "LongTermGallery", "ShortTermMemory"]

#: Cosine similarity to the closest hard negative below which the penalty is 0.
_HN_SIM_LO = 0.5
#: Cosine similarity to the closest hard negative at or above which it is 1.
_HN_SIM_HI = 0.9
#: Hard negatives closer than this to an existing one are treated as duplicates.
_HN_DEDUP_DIST = 0.01


def _unit(feat: Any, dim: int | None = None) -> np.ndarray | None:
    """Return an L2-normalized finite ``float32`` copy, or None if unusable."""
    try:
        f = np.asarray(feat, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError):
        return None
    if f.size == 0 or not np.all(np.isfinite(f)):
        return None
    if dim is not None and f.size != dim:
        return None
    n = float(np.linalg.norm(f))
    if not np.isfinite(n) or n < 1e-12:
        return None
    return np.ascontiguousarray(f / n, dtype=np.float32)


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float, else ``default``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


@dataclass
class GalleryEntry:
    """One stored appearance mode of the target.

    Attributes:
        feat: L2-normalized ``float32`` embedding.
        part: Part key this embedding was computed from (see ``PART_KEYS``).
        quality: Crop quality at insert time, in [0, 1].
        frame_idx: Frame the embedding was observed at (drives recency decay).
        clock: Gallery clock value at the entry's last insert/refresh. Unlike
            ``frame_idx`` this only advances while the gallery is *live*, so a
            frozen gallery does not age its contents (see
            :meth:`LongTermGallery._tick`).
        recency: Recency weight cached at the entry's last insert/refresh
            (1.0 at insert). Live decisions recompute it from ``frame_idx``.
        use_count: How many times this entry was returned by a query.
        meta: Free-form diagnostics (det score, box, state, ...).
    """

    feat: np.ndarray
    part: str = "full"
    quality: float = 1.0
    frame_idx: int = 0
    clock: float = 0.0
    recency: float = 1.0
    use_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> GalleryEntry:
        """Deep-enough copy: the feature array and meta dict are duplicated."""
        return GalleryEntry(
            feat=np.array(self.feat, dtype=np.float32, copy=True),
            part=self.part,
            quality=float(self.quality),
            frame_idx=int(self.frame_idx),
            clock=float(self.clock),
            recency=float(self.recency),
            use_count=int(self.use_count),
            meta=dict(self.meta),
        )


class ShortTermMemory:
    """EMA + ring buffer of the target's most recent accepted embeddings.

    The EMA follows ``smooth_feature`` exactly: L2-normalize the new feature,
    then ``smooth = alpha * smooth + (1 - alpha) * feat``, re-normalized. The
    ring buffer keeps the raw recent observations for the cost layer (e.g. a
    min-over-recent appearance cost beats a single EMA when the target turns).
    """

    def __init__(self, cfg: MemoryCfg, dim: int | None = None) -> None:
        self.cfg = cfg
        self.dim = int(dim) if dim else None
        self._ema: np.ndarray | None = None
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.stm_len)))
        self._last_frame: int | None = None
        self._last_quality: float = 0.0
        self._updates: int = 0

    # ------------------------------------------------------------------ update
    def update(self, feat: np.ndarray, quality: float = 1.0, frame_idx: int = 0) -> bool:
        """Feed one observation.

        Args:
            feat: Appearance embedding (any norm; normalized here).
            quality: Crop quality in [0, 1]. Non-finite or <= 0 is treated as an
                unusable crop and the update is ignored entirely (the EMA and
                the staleness clock both stay put).
            frame_idx: Frame index of the observation.

        Returns:
            True when the observation was accepted.
        """
        q = _finite_float(quality, default=-1.0)
        if q <= 0.0:
            return False
        f = _unit(feat, self.dim)
        if f is None:
            return False
        if self.dim is None:
            self.dim = int(f.size)
        alpha = float(np.clip(_finite_float(self.cfg.stm_alpha, 0.9), 0.0, 1.0))
        if self._ema is None:
            self._ema = f.copy()
        else:
            smooth = alpha * self._ema + (1.0 - alpha) * f
            n = float(np.linalg.norm(smooth))
            if not np.isfinite(n) or n < 1e-12:
                return False
            self._ema = np.ascontiguousarray(smooth / n, dtype=np.float32)
        self._buf.append(f.copy())
        self._last_frame = int(frame_idx)
        self._last_quality = q
        self._updates += 1
        return True

    # ------------------------------------------------------------------- reads
    def current(self) -> np.ndarray | None:
        """EMA of the recent features (L2-normalized), or None if empty."""
        return None if self._ema is None else self._ema.copy()

    def recent(self, k: int = 8) -> np.ndarray:
        """The last ``k`` observations, oldest first, shape ``(n, D)``.

        Returns an empty ``(0, 0)`` array when nothing was stored.
        """
        k = int(k)
        if k <= 0 or not self._buf:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        items = list(self._buf)[-k:]
        return np.stack(items, axis=0).astype(np.float32)

    def mean_recent(self, k: int = 8) -> np.ndarray | None:
        """L2-normalized mean of the last ``k`` observations, or None."""
        r = self.recent(k)
        if r.shape[0] == 0:
            return None
        return _unit(r.mean(axis=0))

    def is_stale(self, frame_idx: int) -> bool:
        """True when no observation was accepted within ``stm_stale_frames``.

        An empty bank is always stale. A never-updated bank counts as stale so
        the state machine can trigger a feature re-init.
        """
        if self._last_frame is None:
            return True
        limit = max(0, int(self.cfg.stm_stale_frames))
        return (int(frame_idx) - self._last_frame) > limit

    def age(self, frame_idx: int) -> int:
        """Frames since the last accepted observation (``-1`` if never)."""
        if self._last_frame is None:
            return -1
        return int(frame_idx) - self._last_frame

    @property
    def last_frame(self) -> int | None:
        return self._last_frame

    @property
    def last_quality(self) -> float:
        return self._last_quality

    def reset(self) -> None:
        """Forget everything (target re-initialization)."""
        self._ema = None
        self._buf.clear()
        self._last_frame = None
        self._last_quality = 0.0
        self._updates = 0

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"ShortTermMemory(n={len(self._buf)}, dim={self.dim}, "
            f"last_frame={self._last_frame}, updates={self._updates})"
        )


class LongTermGallery:
    """Quality/diversity-weighted, part-aware long-term appearance gallery.

    Insert policy (``add``), in order:
        1. reject when frozen or when the feature is unusable;
        2. reject when ``quality < ltm_min_quality``;
        3. reject when the cosine distance to the nearest *same-part* entry is
           <= ``ltm_diversity_thresh`` (a near-duplicate adds no new mode);
        4. if at capacity, evict per ``ltm_replace_policy``
           (``"quality_recency"``: the entry minimizing ``quality * recency``;
           ``"fifo"``: the oldest ``frame_idx``) and insert anyway.
    Rejected duplicates are NOT merged into the stored entry: the gallery is
    deliberately a set of distinct modes, and drift while tracking is handled by
    the short-term EMA.
    """

    def __init__(self, cfg: MemoryCfg, dim: int | None = None) -> None:
        self.cfg = cfg
        self.dim = int(dim) if dim else None
        self._entries: list[GalleryEntry] = []
        self._negatives: deque[tuple[np.ndarray, int]] = deque(
            maxlen=max(1, int(cfg.hard_negative_capacity))
        )
        self._frozen: bool = False
        #: Gallery clock: elapsed frames during which the gallery was *live*.
        #: Recency is measured against this, not against absolute frame index,
        #: so a frozen gallery stops ageing its entries (see ``_tick``).
        self._clock: float = 0.0
        self._last_frame: int | None = None
        #: Outcome of the most recent ``add``: "added" | "refreshed" | "rejected"
        #: | "frozen". Diagnostics for tests and logs.
        self.last_add_action: str = ""

    # ------------------------------------------------------------------ helpers
    def _tick(self, frame_idx: int | None) -> None:
        """Advance the gallery clock to ``frame_idx`` if the gallery is live.

        The clock counts frames the gallery actually observed. While frozen the
        tracker is asserting "the target is not visible right now", and a bank
        that is not being updated must not decay as if it were stale: otherwise
        ``quality * recency`` silently collapses to quality-only after
        ``~ltm_half_life`` frames of occlusion and the eviction policy changes
        behaviour without anyone changing a parameter.
        """
        if self._frozen or frame_idx is None:
            return
        frame = int(frame_idx)
        if self._last_frame is None:
            self._last_frame = frame
            return
        if frame > self._last_frame:
            self._clock += float(frame - self._last_frame)
            self._last_frame = frame
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def capacity(self) -> int:
        return max(1, int(self.cfg.ltm_capacity))

    def entries(self) -> list[GalleryEntry]:
        """Shallow copy of the entry list (entries themselves are live)."""
        return list(self._entries)

    def parts(self) -> list[str]:
        """Part keys currently present in the gallery."""
        return sorted({e.part for e in self._entries})

    def _recency(self, entry: GalleryEntry, now: int | None) -> float:
        """Exponential recency weight of ``entry`` at frame ``now``.

        Ages against the *gallery clock* rather than the absolute frame index,
        so time spent frozen does not decay the bank. Passing ``now`` is
        therefore not purely a read: it advances the clock (see ``_tick``).
        """
        # E025: an adaptive caller may override the configured time-scale frame by
        # frame; None means "use the config value".
        half_life = getattr(self, "_half_life_override", None)
        if half_life is None:
            half_life = self.cfg.ltm_half_life
        hl = -1.0 if half_life is None else _finite_float(half_life, -1.0)
        if hl <= 0.0:  # recency decay disabled
            return 1.0
        if now is None:
            return _finite_float(entry.recency, 1.0)
        self._tick(now)
        age = max(0.0, float(self._clock - float(entry.clock)))
        return float(0.5 ** (age / hl))

    def _candidates(self, part: str | None) -> list[GalleryEntry]:
        """Entries eligible for a query of ``part`` under the part-aware policy."""
        if part is None or not self.cfg.part_aware:
            return self._entries
        same = [e for e in self._entries if e.part == part]
        if same:
            return same
        # No bank for this part yet -> fall back to the full-body bank.
        return [e for e in self._entries if e.part == "full"]

    # --------------------------------------------------------------------- add
    def add(
        self,
        feat: np.ndarray,
        part: str = "full",
        quality: float = 1.0,
        frame_idx: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Try to insert one appearance mode.

        Returns True when a new entry was stored (which may have evicted an
        older one), False when the insert was refused.
        """
        if self._frozen:
            self.last_add_action = "frozen"
            return False
        if part not in PART_KEYS:
            raise KeyError(f"unknown part key {part!r}; valid keys: {list(PART_KEYS)}")
        f = _unit(feat, self.dim)
        if f is None:
            self.last_add_action = "rejected"
            return False
        if self.dim is None:
            self.dim = int(f.size)
        q = _finite_float(quality, default=-1.0)
        if q < float(self.cfg.ltm_min_quality):
            self.last_add_action = "rejected"
            return False

        # Diversity gate: a near-duplicate of an existing same-part mode must not
        # create a SECOND entry -- but it must not be discarded either. Measured
        # same-person part similarities live *inside* the reject zone (upper
        # 0.920, torso 0.941, head 1.000 at a 0.15 threshold), so dropping them
        # would refuse the best views of the target and pin the bank to whatever
        # frame 0 looked like forever. A refused near-duplicate is therefore a
        # REFRESH of the entry it duplicates; only a genuinely novel view earns
        # a new slot.
        same_idx = [i for i, e in enumerate(self._entries) if e.part == part]
        if same_idx and float(self.cfg.ltm_diversity_thresh) > 0.0:
            bank = np.stack([self._entries[i].feat for i in same_idx], axis=0)
            dists = np.asarray(cosine_distance_matrix(f, bank), dtype=np.float64).reshape(-1)
            nearest = int(np.argmin(dists))
            if float(dists[nearest]) <= float(self.cfg.ltm_diversity_thresh):
                self._refresh(self._entries[same_idx[nearest]], f, q, frame_idx, meta)
                self.last_add_action = "refreshed"
                return False

        self._tick(frame_idx)
        entry = GalleryEntry(
            feat=f,
            part=part,
            quality=float(np.clip(q, 0.0, 1.0)),
            frame_idx=int(frame_idx),
            clock=float(self._clock),
            recency=1.0,
            use_count=0,
            meta=dict(meta or {}),
        )

        if len(self._entries) >= self.capacity:
            victim = self._eviction_index(frame_idx)
            if victim is None:  # cannot happen while capacity >= 1
                self.last_add_action = "rejected"
                return False
            self._entries.pop(victim)

        self._entries.append(entry)
        self.last_add_action = "added"
        return True

    def _refresh(
        self,
        entry: GalleryEntry,
        feat: np.ndarray,
        quality: float,
        frame_idx: int,
        meta: dict[str, Any] | None,
    ) -> None:
        """Fold a near-duplicate view into ``entry`` instead of adding a slot.

        The feature is a quality-weighted EMA: a sharp view pulls the stored
        descriptor towards the truth, a blurry one barely moves it. Quality
        itself is a running maximum -- a bank that has seen a good view keeps
        that credit -- and recency restarts because the target was just seen.
        """
        q_new = float(np.clip(_finite_float(quality, 0.0), 0.0, 1.0))
        q_old = float(np.clip(_finite_float(entry.quality, 0.0), 0.0, 1.0))
        total = q_old + q_new
        w_new = 0.5 if total <= 0.0 else (q_new / total)
        blended = (1.0 - w_new) * np.asarray(entry.feat, dtype=np.float32) + w_new * np.asarray(feat, np.float32)
        merged = _unit(blended, self.dim)
        if merged is not None:
            entry.feat = merged
        entry.quality = float(np.clip(max(q_old, q_new), 0.0, 1.0))
        self._tick(frame_idx)
        entry.frame_idx = int(frame_idx)
        entry.clock = float(self._clock)
        entry.recency = 1.0
        if meta:
            entry.meta.update(meta)

    def _eviction_index(self, now: int | None) -> int | None:
        """Index of the entry to drop when the gallery is full."""
        if not self._entries:
            return None
        n = len(self._entries)
        # E023: the first `ltm_anchor_count` admissions are pinned -- they are the
        # crops taken at the click, the only direct evidence of who was selected.
        # A gallery that can evict them drifts onto whoever is being followed.
        pinned = min(max(int(getattr(self.cfg, "ltm_anchor_count", 0) or 0), 0), n)
        cand = list(range(pinned, n))
        if not cand:
            # everything is an anchor: refuse the insert rather than evict the
            # evidence that defines the target
            return None
        policy = str(self.cfg.ltm_replace_policy).lower()
        if policy == "fifo":
            return min(cand, key=lambda i: (self._entries[i].frame_idx, i))
        scores = [
            _finite_float(self._entries[i].quality, 0.0) * self._recency(self._entries[i], now)
            for i in cand
        ]
        return int(cand[int(np.argmin(np.asarray(scores, dtype=np.float64)))])

    def set_half_life(self, value: float | None) -> None:
        """Override the recency half-life at runtime (E025 adaptive forgetting).

        ``None`` restores ``cfg.ltm_half_life``.  Kept as a setter rather than a
        config mutation so a run can adapt per frame without cloning the config.
        """
        self._half_life_override = None if value is None else float(value)

    # ------------------------------------------------------------------- query
    def query(
        self,
        feat: np.ndarray,
        part: str | None = None,
        k: int = 5,
        frame_idx: int | None = None,
        touch: bool = False,
    ) -> list[tuple[float, GalleryEntry]]:
        """Nearest gallery entries by cosine distance, closest first.

        Args:
            feat: Query embedding (any norm).
            part: Query part key. With ``cfg.part_aware`` the search is
                restricted to that part's bank, falling back to the ``"full"``
                bank when it does not exist yet. ``None`` searches everything.
            k: Maximum number of results.
            frame_idx: Current frame, used only when ``touch`` is True.
            touch: When True, bump ``use_count`` and refresh ``recency`` on the
                returned entries (LRU-style reinforcement).

        Returns:
            ``[(cosine_distance, entry), ...]`` with distances in [0, 2].
        """
        if part is not None and part not in PART_KEYS:
            raise KeyError(f"unknown part key {part!r}; valid keys: {list(PART_KEYS)}")
        f = _unit(feat, self.dim)
        if f is None or not self._entries or int(k) <= 0:
            return []
        cands = self._candidates(part)
        if not cands:
            return []
        bank = np.stack([e.feat for e in cands], axis=0)
        dists = cosine_distance_matrix(f, bank)
        order = np.argsort(dists, kind="stable")[: int(k)]
        out: list[tuple[float, GalleryEntry]] = []
        for i in order:
            entry = cands[int(i)]
            if touch:
                entry.use_count += 1
                entry.recency = self._recency(entry, frame_idx)
            out.append((float(dists[int(i)]), entry))
        return out

    def best(
        self,
        feat: np.ndarray,
        part: str | None = None,
        frame_idx: int | None = None,
        touch: bool = False,
    ) -> tuple[float, GalleryEntry | None]:
        """Closest entry as ``(cosine_distance, entry)``.

        Returns ``(inf, None)`` when the gallery cannot answer — an infinite
        distance makes an ungated cost layer reject the candidate, which is the
        safe failure mode.
        """
        hits = self.query(feat, part=part, k=1, frame_idx=frame_idx, touch=touch)
        if not hits:
            return float("inf"), None
        return hits[0]

    # ---------------------------------------------------------- hard negatives
    def add_hard_negative(self, feat: np.ndarray, frame_idx: int = 0) -> bool:
        """Store a confirmed distractor embedding (no-op if the bank is off)."""
        if not self.cfg.hard_negative_bank:
            return False
        f = _unit(feat, self.dim)
        if f is None:
            return False
        if self.dim is None:
            self.dim = int(f.size)
        if self._negatives:
            bank = np.stack([n for n, _ in self._negatives], axis=0)
            if float(cosine_distance_matrix(f, bank).min()) <= _HN_DEDUP_DIST:
                return False
        self._negatives.append((f, int(frame_idx)))  # deque(maxlen) evicts FIFO
        return True

    def hard_negative_penalty(self, feat: np.ndarray) -> float:
        """Smooth 0..1 penalty for matching a stored distractor, else 0.

        Uses the cosine similarity to the closest hard negative, mapped through
        a smoothstep between ``_HN_SIM_LO`` and ``_HN_SIM_HI``: an orthogonal
        probe scores 0.0, a stored distractor scores 1.0.
        """
        if not self._negatives:
            return 0.0
        f = _unit(feat, self.dim)
        if f is None:
            return 0.0
        bank = np.stack([n for n, _ in self._negatives], axis=0)
        sim = 1.0 - float(cosine_distance_matrix(f, bank).min())
        t = (sim - _HN_SIM_LO) / (_HN_SIM_HI - _HN_SIM_LO)
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    @property
    def num_hard_negatives(self) -> int:
        return len(self._negatives)

    # ------------------------------------------------------------------ policy
    def freeze(self, frozen: bool) -> None:
        """Freeze/unfreeze inserts (used while the target is occluded or lost).

        Unfreezing re-anchors the gallery clock: the frames spent frozen must
        not count as ageing (see :meth:`_tick`), otherwise the first live frame
        after a long occlusion would advance the clock by the entire gap and
        every entry would age at once.
        """
        was_frozen = self._frozen
        self._frozen = bool(frozen)
        if was_frozen and not self._frozen:
            self._last_frame = None

    def clear(self) -> None:
        """Drop all entries and hard negatives (target re-initialization)."""
        self._entries.clear()
        self._negatives.clear()
        self._clock = 0.0
        self._last_frame = None
        self.last_add_action = ""

    def stats(self) -> dict[str, Any]:
        """Sizes, per-part counts and mean quality — for logging/experiments."""
        per_part: dict[str, int] = {p: 0 for p in PART_KEYS}
        for e in self._entries:
            per_part[e.part] = per_part.get(e.part, 0) + 1
        quals = [float(e.quality) for e in self._entries]
        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "per_part": per_part,
            "mean_quality": float(np.mean(quals)) if quals else 0.0,
            "frozen": self._frozen,
            "dim": self.dim,
            "hard_negatives": len(self._negatives),
            "hard_negative_capacity": int(self.cfg.hard_negative_capacity),
            "mean_use_count": float(np.mean([e.use_count for e in self._entries]))
            if self._entries
            else 0.0,
            "clock": float(self._clock),
            "last_add_action": self.last_add_action,
        }

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        """Deep copy of the whole bank state (for rollback / checkpointing)."""
        return {
            "dim": self.dim,
            "frozen": self._frozen,
            "clock": float(self._clock),
            "last_frame": self._last_frame,
            "entries": [
                {
                    "feat": np.array(e.feat, dtype=np.float32, copy=True),
                    "part": e.part,
                    "quality": float(e.quality),
                    "frame_idx": int(e.frame_idx),
                    "clock": float(e.clock),
                    "recency": float(e.recency),
                    "use_count": int(e.use_count),
                    "meta": dict(e.meta),
                }
                for e in self._entries
            ],
            "negatives": [
                (np.array(f, dtype=np.float32, copy=True), int(fi)) for f, fi in self._negatives
            ],
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore a :meth:`snapshot` result (replaces all current state)."""
        entries: list[GalleryEntry] = []
        for d in snap.get("entries", []):
            entries.append(
                GalleryEntry(
                    feat=np.array(d["feat"], dtype=np.float32, copy=True),
                    part=str(d["part"]),
                    quality=float(d.get("quality", 1.0)),
                    frame_idx=int(d.get("frame_idx", 0)),
                    clock=float(d.get("clock", 0.0)),
                    recency=float(d.get("recency", 1.0)),
                    use_count=int(d.get("use_count", 0)),
                    meta=dict(d.get("meta", {})),
                )
            )
        self._entries = entries
        self._negatives.clear()
        for f, fi in snap.get("negatives", []):
            self._negatives.append((np.array(f, dtype=np.float32, copy=True), int(fi)))
        self.dim = snap.get("dim", self.dim)
        self._frozen = bool(snap.get("frozen", self._frozen))
        self._clock = float(snap.get("clock", 0.0))
        self._last_frame = snap.get("last_frame", None)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"LongTermGallery(n={len(self._entries)}/{self.capacity}, dim={self.dim}, "
            f"parts={self.parts()}, negatives={len(self._negatives)}, frozen={self._frozen})"
        )
