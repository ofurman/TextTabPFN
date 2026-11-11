# 04 implement: primary evaluation over support sizes

Implementer: omp/task, effort high. Nothing committed. The run went from 2026-11-11T02:39Z to 02:58Z and finished complete: 200 scenes, 4800 rows.

## Changed paths
- `src/texttabpfn/scenes.py`: `eval_scene_specs(num)`. Families 4 and 5 alternate; seeds come from `EVAL_SEED_RANGE` (2,000,000-2,000,199).
- `src/texttabpfn/evaluation.py` (new): the D-08 evaluation. It can resume and writes to `results/04-primary/`.
- `src/texttabpfn/report.py`: new `primary` and `curve` subcommands, plus `d08_verdict`.
- `tests_exp/test_evaluation.py` (new, CPU only, 15 tests): held-out balanced scenes; supports are nested and stratified, share the query rows, are disjoint from them and are deterministic; seed streams do not depend on the seed count; verdict logic; reports pass on complete synthetic results and exit 1 on a missing or NaN row; resume keeps only complete scenes.
- `results/04-primary/`: `config.json`, `scores.csv` (145 KB, columns family/seed/n/arm/logloss) and `summary.json` can be committed. `eval.log` is gitignored.

## Design (fixed before any result)
- Per scene: 512 query rows and a separate 4096-row support pool. Each has its own RNG stream (50 and 51), so support and query rows are separate draws.
- 5 support seeds. Each seed is a nested stratified sequence over n = 8...1024 (`capacity.nested_supports`, both classes present) with its own stream `(family, seed, 52, k)`. The rows inside each support are permuted.
- Arms: native (raw path), schema (`omitted` text) and story (schema + story), using the stage 3 selected weights (step 3250 for both arms). `config.json` records these checkpoints. All arms score identical tensors.
- One forward pass per (scene, n, arm), with the 5 seeds in the batch dimension, under `torch.no_grad()` on MPS (no OOM at n = 1024). The per-(scene, n, arm) loss is the mean over 5 x 512 seed-query pairs.
- Resume: each scene's 24 rows are appended in one write. A restart drops partial scenes and refuses to continue if the settings or checkpoints changed.
- Reports re-validate everything: the settings must equal D-08, the scene list must equal `eval_scene_specs(200)`, and every (scene, n, arm) must be present and finite. They read the CSV and never recompute.

## Checks
- `uv run python -m texttabpfn.report primary` -> exit 0. Table below; verdict `harmful`.
- `uv run python -m texttabpfn.report curve` -> exit 0; all 8 n values.
- Negative probes on a /tmp copy of the results:
  - one row deleted -> `primary` exit 1 ("1 of 1800 ... missing") and `curve` exit 1 ("1 of 4800");
  - all n = 1024 rows deleted -> `primary` exit 0 (it does not need n = 1024) and `curve` exit 1 ("600 of 4800").
- `uv run pytest -q tests_exp` -> 89 passed.
- Resume tested for real: I killed the process after about 23 scenes and restarted it. The log showed "22 of 200 scenes already scored", and the run finished with 4800 rows and no duplicates (4801 CSV lines).
- Determinism: the row for (4, 2000000, 8, schema) is 0.619924 in both the timing probe and the main run.

## Primary (story − schema-only query log loss, 200 scenes, 95% scene bootstrap, 1000 resamples)
| n | native | schema | story | story − schema [95% CI] |
|---|---|---|---|---|
| 8 | 0.6292 | 0.6256 | 0.6248 | −0.0008 [−0.0019, +0.0003] |
| 16 | 0.5879 | 0.5762 | 0.5763 | +0.0001 [−0.0008, +0.0010] |
| 32 | 0.5353 | 0.5267 | 0.5278 | +0.0012 [+0.0005, +0.0019] |

D-08 verdict: **harmful** (the CI lower bound is above 0 at n = 32).
Per family, story − schema: family 4 gives −0.0010 / −0.0005 / +0.0009 [−0.0000, +0.0018] at n = 8/16/32; family 5 gives −0.0006 / +0.0007 / +0.0015 [+0.0005, +0.0026].

## Curve (mean differences; CIs in `report curve`)
| n | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| story − schema | −0.0008 | +0.0001 | +0.0012* | +0.0005 | +0.0003 | +0.0005* | +0.0005* | +0.0004* |
| story − native | −0.0044* | −0.0116* | −0.0075* | −0.0033* | −0.0000 | +0.0006* | +0.0007* | +0.0006* |
| schema − native | −0.0036* | −0.0117* | −0.0086* | −0.0038* | −0.0003 | +0.0001 | +0.0003 | +0.0002 |

\* = the 95% CI excludes 0.

## Wall time, NOT MEASURED, infra
- Evaluation: 17.0 min after the resume, about 19 min in total at about 5-6 s per scene. Timing probe: 2 scenes in a scratch /tmp dir, 21 s (the dir is deleted). No cuts: 200 scenes, 5 seeds, all 8 n.
- Nothing NOT MEASURED. No failed check attempts, so there are no validation files. No infra failures, and the API connection did not drop.

## What in the brief was wrong
- The budget worry did not apply. Batched no-grad inference on MPS took about 0.25 s per forward pass (5 seeds) even at n = 1024, so the full design ran in about 20 min instead of hours.
- Caveat: the timing probe scored 2 of the 200 evaluation scenes before the main run. I made no design change after it; the same values appear again in the main run.
