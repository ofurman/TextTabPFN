# 04 review (omp/reviewer, high) — commit b97088c

Verdict: correct. `report primary` and `report curve` honestly deliver DoD items 4 and 5 under D-08. No P0, P1 or P2 findings. The `harmful` verdict reproduces and is stable across bootstrap seeds 0–49.

## Checks
- **Report output.** `uv run python -m texttabpfn.report primary` exits 0. Story − schema is −0.0008 [−0.0019, +0.0003] at n = 8, +0.0001 [−0.0008, +0.0010] at n = 16, and +0.0012 [+0.0005, +0.0019] at n = 32; the verdict is `harmful`. `report curve` exits 0 with all 8 n.
- **CPU re-score.** Scenes (4, 2000000) and (5, 2000001) were re-scored on CPU at n = 8 and 32 for all 3 arms. All 12 values equal `scores.csv` to 6 decimals, so the MPS run left no artifact.
- **Selected weights.** `*-selected.pt` is bit-equal to the step-3250 checkpoint for both arms, and the arms' weights differ from each other. θ is rebuilt from the TabPFN checkpoint.
- **Texts.**
  - Schema arm → `omitted` text; story arm → `story` text (`evaluation.py:56,116`).
  - The story/schema embedding cosine is 0.82–0.85, so the two arms do see different inputs.
  - The encoder cache is keyed by text, and the embedding is moved to the adapter's device and dtype in `set_text`.
- **No leaks between arms.**
  - The `conditioned()` and `thinking_rows()` context managers restore state; afterwards `add_thinking_rows is orig` and `_text_bD is None`.
  - Each arm has its own adapter object.
- **Identical rows.**
  - One episode tensor per (scene, n) is shared by all arms.
  - The query rows (stream 50) are disjoint from the support pool (stream 51).
- **Stratification and nesting.** Both classes are present at every n, and the n = 8 support is a subset of the n = 32 support for all 5 seeds.
- **Selection.** Checkpoints were selected on dev families 0–3 (`DEV_SEED_RANGE`) only. Evaluation used families 4–5 from `EVAL_SEED_RANGE`, 100 scenes each.
- **Bootstrap.** Per-scene paired differences, resampled over scenes 1000 times, percentile CI. A normal-approximation CI agrees at n = 32: [+0.00051, +0.00183].
- **Verdict code** matches the D-08 wording.
- **Tests.** `uv run pytest -q tests_exp/test_evaluation.py` → 15 passed.

## Findings
1. **P3: verdict precedence (`report.py` `d08_verdict`).** "Harmful" is checked before "supported", and D-08 does not say which wins if both conditions hold. This has no effect here, because no upper bound is below 0.
2. **P3: the reports do not cross-check `config.json` checkpoints and `arm_variant` against stage 3.** Only `evaluation.run` checks them, and only on resume. The committed results are consistent.

## Caveats
- A timing probe scored 2 evaluation scenes before the main run. The design did not change afterwards.
- The effects are about 1e-3 nats; the 6-decimal CSV is precise enough for that.
