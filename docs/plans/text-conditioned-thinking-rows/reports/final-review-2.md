# Final review, iteration 2 (omp/reviewer, high): a80122f..HEAD, docs/plans excluded

Verdict: correct. No P0–P2 findings. Confidence 0.9.

## DoD and regression checks (CPU)
- `report roleinfo`, `role-primary`, `role-curve`, `role-reliability`, `followon-primary` and `role10k-primary` all exit 0, as do `followon-training`, `role-training` and `role-training --results results/13-role-training-10k`.
- `pytest test_role_tokens.py test_parity.py test_label_isolation.py`: 37 passed.
- `pytest test_role_eval.py test_roleinfo.py test_followon_eval.py`: 74 passed.
- The `Verdict:` grep returns 1 file.
- Iteration-1 `capacity`, `primary`, `curve` and `reliability` exit 0 with unchanged numbers.

## Integrity
- No weights are tracked.
- Protected paths have no diff since a80122f.
- `training.py` and `role_eval.py` changes preserve behaviour.
- Lint passes.
- The only uncommitted file is the owner's 2026-11-04 note.

## Stage 13 (not covered by earlier reviews)
- The selected weights equal steps 9500 and 6500, and that selection reproduces from dev `validation.csv`.
- A CPU re-score matches `scores.csv` within 1e-6 on 8 cells.
- Two-arm mode leaves the default stage-9 path unchanged.

## Results note
- Every number matches the report output or the cited review files.
- The verdict line follows D-08 and the stage-11 wording.
- The caveat about training-run variance is honest.

## Findings and triage
1. **P3. Corruption-CI overlap understated (l.103).** The CIs overlap at every n except wrong vs omitted at n = 8. → Fixed in the note.
2. **P3. Encoder interpretation bullet untagged and overstated (l.128).** "Ties" sat beside a printed "harmful". → Fixed: tagged [INFERENCE], now reads "at best null (D-08 harmful, marginal)", and the cause is marked plausible.
