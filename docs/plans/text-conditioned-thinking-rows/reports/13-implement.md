# Stage 13 implement: 10k-step role-token held-out evaluation

Status: done. Both done-when checks pass. The run went from 2026-11-12T03:04:30Z to 03:17:13Z, detached with nohup and resumable.
Launch/resume command (a restart keeps every complete scene):
`cd /Users/ofurman/pwr/texttabpfn && nohup uv run python -u -m texttabpfn.role_eval --out results/13-role-eval-10k --adapters results/13-role-training-10k --arms-only >> results/13-role-eval-10k/eval.log 2>&1 &`

## Changed paths
- `src/texttabpfn/role_eval.py`: adds `--adapters DIR` and `--arms-only`, through `run(..., adapters_dir, full)`, `required_rows(full)` and `_config_core(adapters_dir, full)`. `full=False` skips the rescore sample, preproc-s and the corruptions (B-16), and scores only `role_story` and `role_control`. Defaults are unchanged, so the stage-9 config, rows and reports are identical.
- `src/texttabpfn/report.py`: `_role_eval_scores(results_dir, adapters_dir, full)` and the new `role10k-primary` command (`--results`, `--adapters`, `--stage9`).
- `tests_exp/test_role_eval.py`: +4 tests. They cover the two-arm rows, and that role10k-primary passes on complete results and fails on a missing or NaN row.
- New results: `results/13-role-eval-10k/{config.json,scores.csv,summary.json}` (3200 rows); `eval.log` is gitignored.
- Not mine: `docs/research/web/...attention.md` and `briefs/10-follow-on-arm-training.md`.

## Checks
- `uv run python -m texttabpfn.report role-training --results results/13-role-training-10k` -> exit 0, story 80000 / control 80000 episodes. With no `--results`, the output is byte-identical to the output before my edits (`diff` gives no differences).
- `uv run python -m texttabpfn.report role10k-primary` -> exit 0, verdict "supported".
- `uv run pytest -q tests_exp` -> 182 passed.
- `capacity`, `training`, `primary`, `curve`, `reliability`, `roleinfo`, `role-training`, `role-primary`, `role-curve`, `role-reliability` -> exit 0 each.

## role10k-primary: 10k story (step 9500) − 10k control (step 6500), 95% scene-bootstrap CI
| n | control | story | story − control | family 4 | family 5 |
|---|---|---|---|---|---|
| 8 | 0.6283 | 0.6185 | −0.0098 [−0.0121, −0.0075] | −0.0108 [−0.0146, −0.0069] | −0.0089 [−0.0117, −0.0064] |
| 16 | 0.5773 | 0.5693 | −0.0080 [−0.0097, −0.0064] | −0.0083 [−0.0108, −0.0060] | −0.0077 [−0.0099, −0.0056] |
| 32 | 0.5271 | 0.5214 | −0.0057 [−0.0071, −0.0044] | −0.0055 [−0.0076, −0.0037] | −0.0059 [−0.0078, −0.0041] |

D-08 verdict: **supported**. The CI upper bound is below 0 at all three n, and in both families. The gap is about 2.3× the stage-9 gap at n = 8 (−0.0042) and 1.4× at n = 32 (−0.0041). Story − control stays below 0 at every n: −0.0035 at 64, −0.0024 at 128, −0.0019 at 256, −0.0012 at 512 and −0.0006 at 1024, each with a CI below 0. At 4000 steps, the CI included 0 at n = 512 and 1024.

## 10k vs 4000-step arms (same rows, paired, 95% CI)
| n | story 10k − story 4k | control 10k − control 4k |
|---|---|---|
| 8 | −0.0027 [−0.0042, −0.0012] | +0.0029 [+0.0013, +0.0044] |
| 16 | −0.0029 [−0.0037, −0.0019] | +0.0009 [−0.0002, +0.0021] |
| 32 | −0.0015 [−0.0024, −0.0006] | +0.0001 [−0.0007, +0.0009] |

- The longer schedule improves story at every n; the CI excludes 0 everywhere except n = 128.
- At n = 8, and at 128 to 1024, control gets slightly worse.
- So the wider gap comes from both sides: story gains and control does not. The control's selected step (6500) is earlier than the story's (9500).

## Wall time and NOT MEASURED
- About 15 min total: about 3 min build, 12.7 min run (200 scenes, 3 to 5 s each, alone on MPS), then checks.
- NOT MEASURED: references (native, preproc) are not re-scored or compared here. role10k-primary does not print 10k story − native; stage-9 numbers apply to native.

## What in this brief was wrong
- Done-when 1 already passed at HEAD: `role-training` has had `--results` since stage 8, and it exits 1 on a missing arm or on unequal counts. No change was needed; the claim "Today the command fails" was wrong.
- The stop-by time (04:40Z) gave more than enough room. The run without corruptions took 12.7 min, not about 27.
- `keep_complete_scenes` creates an empty `rescore.csv` even when nothing is re-scored. I deleted it from the results directory; the code does not need it when `full=False`.
