# 07 implement: value of story roles to native TabPFN

Implementer: omp/task, effort high. Nothing is committed. Run 2026-11-11T11:38Z–11:49Z, complete: 200 scenes, 3200 rows.

## Changed paths
- `src/texttabpfn/roleinfo.py` (new):
  - `role_preprocess(roles, x_support, x_query)`, which is reusable by stage 9, and `preprocess_episode(roles, (xs, ys, xq, yq))`; labels pass through untouched;
  - the resumable stage 7 run. It writes `results/07-roleinfo/`.
- `src/texttabpfn/report.py`: new `roleinfo` subcommand. It reads `results/07-roleinfo/` and `results/04-primary/` and never recomputes.
- `tests_exp/test_roleinfo.py` (new, CPU only, 30 tests):
  - the preprocessing takes no labels, and flipping labels changes no output;
  - irrelevant columns are removed (the output width is mb + groups; scrambling them changes nothing), and mb columns are kept exactly;
  - each group collapses to one column (changing a member changes only that column), and a sign flip of a member leaves it unchanged;
  - PC1 recovers a hidden quantity measured with opposite signs (corr > 0.95, where the plain mean gives |corr| < 0.3);
  - each support seed is fitted independently;
  - the report passes on complete results and exits 1 on a missing row, a NaN, native drift from stage 4, or another method;
  - resume keeps only complete scenes.
- `results/07-roleinfo/`: `config.json`, `scores.csv` (132 KB) and `summary.json` can be committed. `eval.log` is gitignored.

## Collapse method (fixed before any result)
- Drop `irrelevant`. Keep `mb` unchanged.
- Each `measure` group becomes the first principal component of its columns, each standardized. The sign is chosen so the group's first column (in column order) loads ≥ 0. The component is placed at that first column's position.
- The collapse is fitted per support seed on support + query feature values (all unlabeled). Role facts come from `Scene.describe()["roles"]`.
- Native is re-scored in the same run, on MPS and the same rows. It matches stage 4's native arm exactly: max |diff| = 0 over 1600 cells. The report fails above 1e-6.

## Checks
- `uv run python -m texttabpfn.report roleinfo` -> exit 0 (table below).
- Negative probe, on a /tmp copy (deleted afterwards): deleting row (4, 2000000, 8, roles) -> exit 1, "1 of 3200 ... missing".
- `uv run pytest -q tests_exp` -> 129 passed (99 existing + 30 new).
- Iteration-1 reports `capacity`, `stories`, `training`, `training --validation`, `primary`, `curve`, `reliability` -> all exit 0.
- Real resume: I killed the run after 26 scenes and restarted it. It logged "26 of 200 scenes already scored", and the CSV ended with 3200 rows and no duplicate keys.

## Role information (preprocessed − native query log loss, 200 scenes, 95% scene bootstrap, 1000 resamples)
| n | native | preproc | preproc − native [95% CI] | family 4 | family 5 |
|---|---|---|---|---|---|
| 8 | 0.6292 | 0.6128 | −0.0165 [−0.0200, −0.0130] | −0.0179 [−0.0235, −0.0126] | −0.0151 [−0.0200, −0.0108] |
| 16 | 0.5879 | 0.5675 | −0.0204 [−0.0233, −0.0171] | −0.0223 [−0.0269, −0.0177] | −0.0184 [−0.0221, −0.0145] |
| 32 | 0.5353 | 0.5197 | −0.0156 [−0.0183, −0.0129] | −0.0167 [−0.0207, −0.0123] | −0.0146 [−0.0180, −0.0110] |
| 64 | 0.4845 | 0.4773 | −0.0072 [−0.0096, −0.0044] | −0.0074 | −0.0069 |
| 128 | 0.4443 | 0.4442 | −0.0001 [−0.0028, +0.0029] | −0.0016 | +0.0013 |
| 256 | 0.4144 | 0.4202 | +0.0058 [+0.0029, +0.0091] | +0.0049 | +0.0067 |
| 512 | 0.3940 | 0.4044 | +0.0104 [+0.0075, +0.0138] | +0.0098 | +0.0110 |
| 1024 | 0.3806 | 0.3938 | +0.0132 [+0.0103, +0.0167] | +0.0123 | +0.0140 |

- D-13: the CI excludes 0 at all of n = 8, 16 and 32 (it lies entirely below 0), so the stop rule does not fire.
- The gain is 0.016–0.020 nats at n ≤ 32. That is about 2–4× the iteration-1 story adapter's gain over native (0.004–0.012).
- [INFERENCE] From n ≥ 256 the preprocessing hurts. With enough rows, TabPFN learns the group structure itself, and collapsing the group discards per-member signal.

## Wall time, NOT MEASURED, infra
- Scoring took about 11 min including the kill and resume, about 3 s per scene: 16 forward passes of 5 seeds each.
- Nothing is NOT MEASURED. No check attempt failed, so there are no validation files. There were no infra failures.

## What in the brief was wrong
- Nothing material. The run took about 11 min, not the ~20 min of a stage-4-sized run, because it scores 2 arms instead of 3.
