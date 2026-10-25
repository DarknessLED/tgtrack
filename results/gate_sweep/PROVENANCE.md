# PROVENANCE — which code state defines the numbers in `results/gate_sweep/`

Written 2026-10-26 01:5x while the E070 sweep was completing `margin_0.30`. Purpose: the sweep
re-imports `tgtrack` for **every (point, sequence) child**, so a mid-sweep source edit would make
pairs non-comparable. This file records the check that was done and the rules that follow from it.

## The question raised

The sweep owner observed `src/tgtrack/consensus.py` with mtime `01:15:51`, i.e. inside the sweep
window (started 01:08), and asked whether pairs measured after that moment ran different code than
pairs before it.

## The check (all verifiable, re-runnable)

| claim | how it was checked | result |
|---|---|---|
| `consensus.py` is **new**, not a modification | it did not exist before this session's E071 work; it is the E071 mechanism module | new file |
| **nothing on the measurement path imports it** | grep for `consensus` in `src/tgtrack/__init__.py`, `src/tgtrack/target_tracker.py`, `scripts/run_target.py`, `scripts/gate_sweep.py` | **zero matches** |
| the executed code is unchanged since before the sweep | mtimes | `target_tracker.py` 10/25 18:40:11, `appearance.py` 18:54:48, `run_target.py` 16:48:14, `config.py` 16:44:59, `association.py` 10/25 00:49:26, `memory.py` 05:58:27, `motion.py` 04:28:57 — all **before** 01:08 |
| the only measurement-path file with a 01:xx mtime | mtimes | `scripts/gate_sweep.py` 01:08:16 — written by the sweep owner, not by the Lead |
| the Lead never wrote to `runs/` | the Lead's scripts read `runs/` and write only `SPLIT*.md` / `PAIRED.md` | confirmed |

**Conclusion: a module that is never imported cannot change a decision.** Pairs before and after
01:15:51 are behaviourally identical. The sweep is not measuring two code states.

## Rules that follow

1. **`shipped` means E068** — TSR 0.2102 / FAR 0.2364 on the 22 held-out sequences, as recorded in
   `docs/EXPERIMENTS.md`. That is the criterion number and it does not move because of a sweep.
2. **Within-sweep comparisons use the sweep's own `margin_0.20` pairs**, measured in the same
   session with the same provenance stamp. Never compare a stamped pair against an unstamped one.
3. **`PAIRED.md` and `SPLIT3.md` are PROVISIONAL.** They were generated from `runs/` *before* the
   owner's purge-and-rerun, so they must be regenerated from the fully stamped tree before being
   quoted. Their qualitative reading (monotone trade-off; dFAR anti-correlated with reference TSR)
   is unlikely to change, but the point estimates will.
4. **Any unstamped pair is void.** Pairs written during the earlier duplicate-sweep window
   (00:45–00:56, when two workers shared `runs/` and one `summary.json` was corrupted into
   `JSONDecodeError: Extra data`) carry no authority.
5. `dancetrack0041` and `dancetrack0047` disagree with E068 at `margin_0.20`
   (FAR 0.030 → 0.047 and 0.263 → 0.470). Since no measurement-path code changed, these are
   **sweep-side** discrepancies: re-run both from scratch with a stamp and report whether the delta
   reproduces. If reproducible, the cause (target id, purge/rerun mix, CUDA nondeterminism) must be
   named before any per-axis number is quoted; if not, the earlier pair is void.

## Freeze

`src/`, `scripts/run_target.py` and `scripts/gate_sweep.py` are **frozen** for the remainder of the
E070 sweep. E071's tracker wiring is deliberately deferred for exactly this reason; only its
standalone module and tests exist (`src/tgtrack/consensus.py`, `tests/test_consensus.py`), and they
are inert until wired.
