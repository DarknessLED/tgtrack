# METHOD.md -- how experiments are run in this repository

This is the methodology companion to `docs/EXPERIMENTS.md` (the ledger, 36 numbered
experiments). The ledger records *what happened*; this file records *how to work so that what
happened can be believed*. It is written from the mistakes the programme actually made.

Audience: whoever runs the next experiment, including a future maintainer with no memory of
this session. Every rule below exists because violating it produced a wrong conclusion here.

---

## 1. The loop

```
bottleneck (measured) -> question -> cheapest decisive measurement -> single-variable change
    -> verdict (accept / reject / null / bug-fix) -> ledger entry -> ship only if Pareto
```

A round is not "try ideas until a number gets better". A round ends with a *verdict*: either a
shipped change with a measured delta and no regression on the other hard criteria, or a named
mechanism proved useless (which is knowledge, and is cheaper than shipping it).

**Find the bottleneck before touching the strategy.** E006/E007 proved that offline association
repair could not matter, and E010a proved that the tracker's output gating costs only 1-3 % of
recall, because the *detector* caps DetA. Two full experiments were saved by that measurement.

## 2. Measurement discipline

**Same command, same data, same seed.** A delta is only a delta if everything except the one
variable is identical. Diff the exact CLI between baseline and treatment and print both.

**Name the epoch.** Defaults change during a programme. After E029 the appearance band became
the default and changed both TSR and FAR on every sequence; numbers measured before and after
are not comparable without saying which epoch they belong to. The ledger prefixes each row with
the epoch, and any table that mixes epochs is wrong.

**Fresh baseline rule.** Never trust a baseline artifact from an earlier default-set. E019
found `results/dancetrack_target` was 35 min older than the association code it claimed to test
(00:14 summary vs 00:49 `association.py`). The rule: record source mtimes next to the artifact;
re-run the baseline whenever any file it depends on is newer. A cheap `Get-ChildItem` check
before quoting a number is what caught it.

**Nondiscrimination check.** Before reading a grid, ask whether the control moved. E035's tiling
grid produced `transcript_ok 1` on a fake video and `-1` on a real one; the "failure" was the
scenario never bending the transcript. E023's pinned-anchor grid was *inert* because the gallery
had not evicted anything in 200 frames -- the mechanism was never exercised. A grid where every
row is identical is not a null result, it is an unexercised mechanism; find out which.

**Sample size.** 200 frames x subsample 2 on one sequence is enough to reject a mechanism that
moves 5 pp, and not enough to rank two mechanisms that differ by 1 pp. Say which you are doing.

## 3. Controls

- **Off-control must be genuinely off.** E021's first sweep compared "shipped default" against
  live values; the moment the default went live the control *became* a treatment. Re-point the
  control at `None`/0 explicitly.
- **Positive control.** When a mechanism is supposed to fire, assert it fired. E025's adaptive
  half-life was INERT because the drift statistic was computed from a gallery that had already
  absorbed the impostor -- the measurement was self-referential. E026 replaced the reference
  with the frozen click crop and the axis became real. If a knob does nothing, prove *why*
  before concluding the idea is bad.
- **Negative control.** E022's `search_uses_stm` was a hypothesis-disentangling control: with
  the guard on and the STM reference off, 0004 collapsed (0.265 -> 0.040), which is the only
  reason we can say the *band* rather than the *reference* is doing the work.
- **Solo control for swap experiments.** E031's FAR regressions were partially measurement
  artifacts: a "wrong" box with no distractor present is not a swap. Always compare against a
  run whose scenario contains no distractor.

## 4. Verdict taxonomy

| verdict | meaning | what happens to the code |
|---|---|---|
| **accept** | measured delta, Pareto-safe on all hard criteria | shipped as default, tests added |
| **reject** | mechanism active but the trade is bad | knob kept with default off, ledger entry |
| **null** | mechanism never fired / no effect | knob kept or removed; ledger records *why* it was inert |
| **bug-fix** | the measurement revealed a defect in the harness or metric | fix landed, tests added, often the largest real gain |
| **detached** | result not comparable (epoch/artifact/config drift) | re-measured, never quoted |

Two are easy to confuse and must not be: **null** ("the idea does nothing") and **reject** ("the
idea does something harmful"). E023 was null-then-reject; E032b was pure null and was the most
informative result of the round.

## 5. Shipping rules

1. **Pareto only.** A change ships if it improves at least one hard criterion without regressing
   any other beyond noise. E027 had a huge win on one sequence and lost 11.9 pp on another; it
   did not ship.
2. **Hard criterion first.** The programme has four acceptance numbers. A mechanism that moves a
   *secondary* metric while a hard one regresses is a reject, no matter how elegant. This is why
   E030's band (pooled FAR 0.387, TSR 0.437) was rejected even though it looked like progress.
3. **Conservative-margin re-examination.** A tie on the training subset must be re-examined at a
   conservative value before shipping. E022's `search_margin = 0.20` tied with 0.10 on the
   sample and then beat 0.10 on the wider grid; the tie was the signal, not the noise.
4. **Sequence-dependent levers do not ship.** E024 proved gallery half-life helps 0004 and hurts
   0007 in a way no single constant can satisfy; E025 then showed the adaptive repair cannot see
   the difference because both look like drifting appearance *to the gallery*. A lever whose
   correct setting depends on which failure mode the sequence has cannot be tuned by a constant.
5. **Real data arbitrates over synthetic.** E-unid's lone-candidate guard won on the synthetic
   family and lost 3.6 pp of upper-body TSR on MOT17 with zero real FAR benefit; the default
   reverted to `None` and the negative result was kept.
6. **Keep the losing branch.** Every rejected knob stays in the config with `default off` and a
   comment citing the ledger entry. Deleting it deletes the knowledge.

## 6. Anti-pattern catalogue (all of these happened here)

1. **Silent no-op patch.** E028's edit applied to a duplicated config block and did nothing for
   three runs. Always assert the string you replaced existed (`assert old in src`) and re-read
   the file after writing.
2. **Silent flag drop.** E009's `--ignore-regions` did nothing because `RunOptions` copies flags
   field-by-field; an argparse flag that is not also a `RunOptions` field is silently inert.
3. **Off-by-one in annotation frames.** MOTChallenge `gt.txt` is 1-based while our frames are
   0-based; the first ignore-region probe reported 33.5 % instead of 38.4 % because of it. E008's
   fix moved HOTA +1.24 pp.
4. **Metric sign inversion.** The first pinning probe reported `auc(perfect) = 0.000` -- it
   measured a distance as if it were a similarity. Always include a perfect-match and a
   random-match fixture in a new metric's test.
5. **Separate normalisation in a fusion.** The first pose+appearance fusion scored *below both
   inputs* because each block was normalised to its own band before combining.
6. **Non-discriminating scenario.** A fake video whose transcript never bends (E035) grades every
   strategy as perfect. Validate a scenario can fail before using it to rank.
7. **Unexercised mechanism.** A grid over an eviction policy when nothing was ever evicted.
8. **Dead knob kept in a "measurement".** A `--max-frames 40` run cannot show a 60-frame
   mechanism; check the mechanism's own time-scale against the run length.
9. **Wrong field name in a patch that only runs under a flag.** E032b's conjunction raised
   `AttributeError` on three variants whose rows vanished from the table while `exit code 0`
   still printed; the tell was `errors=3` next to `ok=1` in `[result]`. Always read the
   `[result]` line, never just the rows.

## 7. A domain finding worth carrying forward

Three separate families were tried to separate "the target changed" from "this is someone else":
adaptive appearance band (E029), capped appearance band (E030), adaptive motion band (E032), and
their conjunction (E032b). All fail the same way, and E032b proves why: on `dancetrack0007` the
impostor is *closer to the target's own past* than the target is (appearance cost 0.07-0.17 vs
0.22-0.30; the true target ranked 2nd or worse in 74 of 111 wrong frames).

**Conclusion:** any discriminator defined as a distance to the target's own history is exhausted.
Further gains require an *independent* identity signal -- pose/limb geometry, local patch
verification, or co-visibility structure with the surrounding people. Record this here so the
next round does not re-derive it by sweeping another threshold.

## 8. Adding an experiment

1. Pick the bottleneck and write the question as a falsifiable sentence.
2. Check the mechanism is reachable: does the config path execute, is the run long enough, is
   the control genuinely off?
3. Run baseline and treatment with an identical command; read `[result]` (ok/errors) first.
4. Compare on the hard criteria; apply the shipping rules above.
5. Append to `docs/EXPERIMENTS.md` with: hypothesis, exact command, table, verdict, and the
   *reason* for the verdict. A table without a reason is not an experiment.
6. If accepted, add a regression test that fails without the change.
7. Update this file if the work exposed a new way to be wrong.

## 8. A reference cannot certify the thing it was built from (the contamination rule)

Five independent experiments failed for one shared reason, and it is now a rule rather than a surprise:

| experiment | reference it trusted | outcome |
|---|---|---|
| E025 adaptive forgetting | the gallery's own drift | inert (self-referential) |
| E029 adaptive distance band | the target's own distance history | shipped, but only via a *frozen click* anchor |
| E032/E032b extra motion axis | the target's own motion history | 0.000 joint movement |
| **E063 click-anchor audit** | **the crop captured at the click** | **+20.3 pp TSR on one sequence** |
| **E064 two-witness audit** | click anchor **+ `stm.current()`** | **bit-exact no-op: recent vetoes 31/31, 23/23, 16/17** |

The mechanism: `stm.current()` is the EMA of *accepted* crops. Once the tracker latches onto the wrong
person, the short-term memory **is** the impostor, so any signal computed "distance from my own recent
appearance" is measuring agreement with the failure. It can suppress switches, never rescue them —
measured, not argued: the recent witness vetoed every audit where the click crop preferred the rival,
with median margins -0.1445 / -0.1986 / -0.0613 and zero switches.

**Rule for this project:** an identity reference is only admissible if it was captured **before** the
failure it is meant to detect (the click anchor) or comes from **outside** the tracked person's own
history (the all-person layer's tracklet id, or another tracker's evidence). Self-referential
references must be re-asked as "would this have been written before the failure?" — if not, the
experiment is void, however many knobs it has.

## 9. Harness traps that silently change what a number means

Each of these produced, or nearly produced, a wrong conclusion. They are listed because the CLI
defaults are the kind of thing that a future reader will not check.

- **`scripts/run_target.py --ablate` defaults to `"all"`.** Omitting the flag runs the entire
  built-in variant grid (84 variants) on every sequence: measured at 450 s for 10 frames of one
  sequence, 85 headline rows, `meta.ablate == "all"`. `build_variants("")` also maps to `"all"`, so
  the current CLI **cannot express "baseline only"** — a held-out run has to pin
  `run_target.build_variants` in a child process and assert `meta.variants == ["baseline"]`. Until
  fixed, never run `run_target.py` without `--ablate <group>` and never trust a summary whose
  `meta.variants` is not the expected list.
- `--trackers a.yaml b.yaml` fails; it is one comma-separated value (`a.yaml,b.yaml`).
- `detector.imgsz` must match the ONNX/weights it was trained at or the resolution claim is void
  (E002/E003); the ReID spec stamp exists for the same reason.
- `--target-id` is mandatory for any masked or comparative target run; without it the harness picks
  the *observable* target, and `dancetrack0005` has been picked as both id 0 and id 2 (see the audit
  corrections in `README.md`). A comparison across epochs without a pinned id is comparing people.
