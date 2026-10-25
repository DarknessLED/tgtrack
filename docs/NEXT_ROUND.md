# Next round: gate calibration on the HELD-OUT split

Written at the end of the review round. Everything below is derived from measurements already on disk;
no new model, no new dataset, no fine-tuning is needed.

## The diagnosis this round produced

1. **Criterion 2's real value is TSR 0.2104 / FAR 0.2472 over the 22 held-out sequences**
   (E070's shipped-configuration re-run on today's code; E068's 0.2102 / 0.2364 is superseded — it predates `target_tracker.py` 18:40:11) 
   (`results/heldout_dancetrack/report.json`, E068), not the dev-split 0.512/0.280. The dev TSR lies
   outside the held-out CI, so the dev split certifies nothing (its intervals were +-0.394 TSR and
   +-0.708 FAR on n=3).
2. On held-out data the failure mode is **losing the target**, not locking onto the wrong person
   (FAR is statistically unchanged between dev and held-out).
3. The target is **detected in ~84 %** of the frames where we lose it
   (`results/tracklet_evidence/analysis_reid.json`, E069): a MOT box with IoU >= 0.5 to the target's GT
   exists on 84.4 % of 969 measurable loss frames.
4. The all-person MOT layer's **tracklet id is not usable as an external anchor**: only 9.7 % of frames
   inside >=5-frame loss runs are single-id anchored and all five >=50-frame runs are unanchored (E069).
5. Two independent families are therefore closed: self-referential appearance evidence (E025, E032,
   E032b, E064 -> METHOD.md section 8) and external MOT tracklet identity (E069).

**Conclusion: the remaining gap is gate calibration, not missing evidence.** The same acceptance gate
that prevents a wrong-identity lock (E022 `cost.search_margin = 0.20`, E029 adaptive band) is the one
refusing a correct re-acquisition.

## The one experiment to run

Sweep the search-side gate **on the 22 held-out sequences only** (never on dancetrack0004/0005/0007):

- primary axis: `cost.search_margin` in {0.0, 0.10, 0.15, 0.20 (shipped), 0.30};
- secondary: `cost.anchor_adaptive_k` in {3.0, 4.0 (shipped)}, only if the primary shows a monotone trade;
- protocol: `scripts/heldout_report.py` (it refuses sweep flags by design — add a narrow, explicit
  exception for these two knobs, or drive it from a child process that pins `build_variants`), 300
  frames, subsample 2, `--jobs 1`, `--target-id` from each run's own `meta.target_ids`;
- report per axis point: pooled TSR and FAR with sequence-level cluster-bootstrap 95 % CI (2000 reps),
  plus the count of sequences whose TSR moves by > 5 pp, so a bimodal shift cannot hide inside a mean;
- acceptance rule, fixed BEFORE the sweep: ship a value only if pooled TSR rises by >= +5 pp **and** the
  upper bound of its FAR CI does not exceed the shipped FAR's upper bound. Anything else is recorded as
  a negative and the default is left alone.

## Why not the alternatives

- More detector recall: six negatives (E001/E002/E003/E011/E055/E056) and an active detection problem
  would show low coverage, but coverage is 84.4 %.
- New identity evidence: both admissible families are closed by measurement (points 4-5).
- More knobs in the same family: E032b showed the family is bit-identical at its own end state.
