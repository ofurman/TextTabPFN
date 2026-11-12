# Stage 11 implement: B-09 bge follow-on held-out evaluation

Status: done. The done-when check passes. The run went from 2026-11-12T06:40:30Z to 06:53:26Z, detached with nohup and resumable.
Launch/resume command (a restart keeps every complete scene and the rescore sample):
`cd /Users/ofurman/pwr/texttabpfn && nohup uv run python -u -m texttabpfn.followon_eval >> results/11-followon-eval/eval.log 2>&1 < /dev/null &`

## Changed paths
- `src/texttabpfn/followon_eval.py` (new): scores stage 10's selected bge `story` and `control` adapters (both step 3000) on `evaluation.scene_episodes(evaluation.Settings(), f, s)` over `eval_scene_specs(200)`, at all 8 n. Story reads stage 4's `story` text and control reads its `schema` (`omitted`) text. It first re-scores native on the first 8 scenes into `rescore.csv` and stops unless every row matches `results/04-primary` within 1e-6. It reuses `evaluation` and `role_eval` helpers without changing them.
- `src/texttabpfn/report.py`: new `followon-primary` command (`--results --adapters --primary --roleinfo --stage9 --role10k --role10k-adapters`) and `_followon_scores`. It exits 1 on a missing or non-finite row, a design mismatch, or rescore drift above 1e-6. It only reads CSV/JSON. Existing commands are unchanged.
- `tests_exp/test_followon_eval.py` (new, 7 tests): row set, arm texts, pass on complete synthetic results, and fail on a missing row (n = 32, 1024), a NaN row, or rescore drift.
- New results: `results/11-followon-eval/{config.json,scores.csv,rescore.csv,summary.json}` (3200 + 64 rows). `eval.log` is gitignored.
- Not mine: `docs/research/web/2026-11-04-text-tabular-attention.md`.

## Checks
- `uv run python -m texttabpfn.report followon-primary` -> exit 0, verdict "harmful".
- `uv run pytest -q tests_exp` -> 207 passed.
- 19 earlier report commands -> exit 0 each:
  - `capacity --arm shared|per-column` and `stories`;
  - `training`, `role-training`, `followon-training` and `role-training --results results/13-role-training-10k`, each with and without `--validation`;
  - `primary`, `curve`, `reliability`, `roleinfo`, `role-primary`, `role10k-primary`, `role-curve` and `role-reliability`.
- Native rescore sample: max |rescored − stage 4| = 0.0 over 64 rows.

## followon-primary: bge story − bge control, 95% scene-bootstrap CI (1000 resamples)
| n | control | story | story − control | family 4 | family 5 |
|---|---|---|---|---|---|
| 8 | 0.6248 | 0.6256 | +0.0008 [+0.0000, +0.0017] | +0.0006 [−0.0006, +0.0018] | +0.0011 [−0.0000, +0.0022] |
| 16 | 0.5759 | 0.5763 | +0.0004 [−0.0001, +0.0010] | +0.0005 [−0.0004, +0.0013] | +0.0004 [−0.0004, +0.0012] |
| 32 | 0.5265 | 0.5266 | +0.0001 [−0.0004, +0.0005] | +0.0001 [−0.0005, +0.0007] | −0.0000 [−0.0006, +0.0005] |

D-08 verdict: **harmful**. This is a marginal call. At n = 8 the CI lower bound is +4.2e-6, just above 0, so `d08_verdict` returns harmful. The effect is +0.0008 nats, and neither family's CI excludes 0 on its own. In plain terms: bge story text gives no held-out gain over schema text, and may be very slightly worse at n = 8. This matches the iteration-1 MiniLM pair (−0.0008 / +0.0001 / +0.0012, also "harmful"). For n from 64 to 1024, story − control is −0.0001 to −0.0003. Only n = 128 has a CI that excludes 0 (upper bound −0.0000).

## Context comparison (same rows, mean query log loss)
| n | bge ctrl | bge story | native | preproc | MiniLM schema | MiniLM story | role 4k story | role 10k story |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.6248 | 0.6256 | 0.6292 | 0.6128 | 0.6256 | 0.6248 | 0.6212 | 0.6185 |
| 16 | 0.5759 | 0.5763 | 0.5879 | 0.5675 | 0.5762 | 0.5763 | 0.5721 | 0.5693 |
| 32 | 0.5265 | 0.5266 | 0.5353 | 0.5197 | 0.5267 | 0.5278 | 0.5229 | 0.5214 |
| 128 | 0.4446 | 0.4443 | 0.4443 | 0.4442 | 0.4440 | 0.4443 | 0.4431 | 0.4426 |
| 1024 | 0.3814 | 0.3813 | 0.3806 | 0.3938 | 0.3808 | 0.3812 | 0.3814 | 0.3810 |

Paired differences, with CIs:
- bge story − native: −0.0036 / −0.0115 / −0.0087 at n = 8 / 16 / 32. That gain is shared with bge control.
- bge story − MiniLM story: +0.0008 [−0.0007, +0.0022] / +0.0000 / −0.0013 [−0.0020, −0.0006].
- bge story − role 10k story: +0.0072 / +0.0071 / +0.0052, with every CI above 0.
- So the bge arms land where the MiniLM arms are. The role-token arms, and preprocessing at small n, stay ahead. All n are printed by the command.

## Wall time and NOT MEASURED
- About 25 min total: about 8 min build, 12.9 min run (16 s rescore, then 200 scenes at 3 to 5 s each on MPS), then checks.
- NOT MEASURED: corruptions for the bge arm (B-15, out of scope).

## What in this brief was wrong
- Nothing material. The predicted "null" is, by the D-08 rule, a marginal "harmful": the n = 8 lower bound is +4e-6. I report that as computed and did not tune anything.
- `results/07-roleinfo` holds only `native` and `roles` (support+query), so "preprocessing" here means stage 7's `roles` arm.
