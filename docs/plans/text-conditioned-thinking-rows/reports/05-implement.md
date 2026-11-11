# 05 implement: context reliability evaluation

Implementer: omp/task, effort high. Nothing is committed. The run went from 2026-11-11T03:03Z to 03:10Z and finished complete: 200 scenes, 2400 rows.

## Changed paths
- `src/texttabpfn/reliability.py` (new): the resumable stage 5 run. It writes `results/05-reliability/`.
- `src/texttabpfn/evaluation.py`: the resume logic is now the generic `keep_complete_scenes(path, fields, need)`, and `read_scores` takes a `label=` column. `_logloss` was renamed `query_logloss`. Stage 4 behaviour is unchanged.
- `src/texttabpfn/report.py`: new `reliability` subcommand. `_diff_cell` takes `digits=`; the reliability table uses 5 decimals.
- `tests_exp/test_reliability.py` (new, CPU only, 10 tests):
  - texts are reproducible for all 200 scenes, each corruption differs from the story, and the correct story equals stage 4's text;
  - corruptions use separate rng streams;
  - the report passes on complete synthetic results and exits 1 on a missing or NaN 05 row, a missing 04 row, or a 04/05 story mismatch;
  - the report ignores 04 rows above n = 32;
  - resume keeps only complete scenes.
- `results/05-reliability/`: `config.json`, `scores.csv` (73 KB; family/seed/n/text/logloss) and `summary.json` can be committed. `eval.log` is gitignored.

## Design (fixed before any result)
- Rows: `evaluation.scene_episodes(evaluation.Settings(), ...)` with the full D-08 sizes, scoring only n = 8, 16, 32. The rows are therefore identical to stage 4's.
- Model: the story adapter at stage 3 step 3250 (recorded in `config.json`), scored on 4 texts: story, omitted, irrelevant and wrong (`stories.scene_text`).
  - Each corruption has its own per-scene rng stream, `(family, seed, 60, variant index)`.
  - The run refuses to score a scene if a corruption equals the story or if the correct story differs from stage 4's text.
- The correct story was rescored on the same rows. It matches the stage 4 story arm exactly (max |diff| = 0 over 600 cells). The report fails if the difference is above 1e-4.
- The report reads only the CSV/JSON files under `results/05-reliability/` and `results/04-primary/`. It re-validates settings, scenes and texts, and checks that every row is present and finite.

## Checks
- `uv run python -m texttabpfn.report reliability` -> exit 0; the table is below.
- Negative probes on a /tmp copy (deleted afterwards): row (4, 2000000, 8, wrong) deleted -> exit 1 ("1 of 2400 ... missing"); (5, 2000199, 32, irrelevant) set to nan -> exit 1.
- `uv run pytest -q tests_exp` -> 99 passed (89 existing + 10 new).
- Real resume: I killed the run after 38 scenes (457 CSV lines) and restarted it. The log showed "38 of 200 scenes already scored", and the run ended with 2400 rows and no duplicates.

## Reliability (story adapter; corrupted − correct story query log loss, 200 scenes, 95% scene bootstrap, 1000 resamples)
Training exposed the story arm to 15% omitted and 10% one-wrong-fact episodes (D-09). Irrelevant stories and fully permuted roles were never trained on.

| n | story | omitted − story | irrelevant − story | wrong − story |
|---|---|---|---|---|
| 8 | 0.6248 | +0.00151 [+0.00098, +0.00204] | +0.00061 [−0.00010, +0.00134] | +0.00008 [−0.00002, +0.00020] |
| 16 | 0.5763 | +0.00068 [+0.00037, +0.00100] | +0.00025 [−0.00003, +0.00052] | +0.00002 [−0.00004, +0.00007] |
| 32 | 0.5278 | +0.00066 [+0.00046, +0.00085] | +0.00008 [−0.00004, +0.00019] | −0.00001 [−0.00004, +0.00001] |

Secondary context (not a check): stage 4 arms on the same rows minus the correct story.

| n | schema-only arm − story | native − story |
|---|---|---|
| 8 | +0.00082 [−0.00034, +0.00193] | +0.00441 [+0.00176, +0.00718] |
| 16 | −0.00012 [−0.00105, +0.00078] | +0.01156 [+0.00932, +0.01414] |
| 32 | −0.00117 [−0.00185, −0.00053] | +0.00747 [+0.00604, +0.00902] |

Reading:
- Only omitting the story significantly hurts the story adapter, and only by 0.0007–0.0015 nats. An irrelevant story costs at most 0.0006 (CI includes 0); permuted roles change nothing measurable (|mean| ≤ 0.0001).
- [INFERENCE] The adapter reacts to whether a story is present, not to what it says. Every corruption effect is far below the story-vs-native gap (0.004–0.012).

## Wall time, NOT MEASURED, infra
- Scoring took about 6.5 min, including the kill and resume: about 1.5 s per scene, 12 forward passes of 5 seeds each. Timing probe: 2 scenes in /tmp, 14 s, deleted afterwards.
- Nothing is NOT MEASURED. No check attempt failed, so there are no validation files. There were no infra failures.

## What in the brief was wrong
- "Reuse `scene_episodes` restricted to n = 8, 16, 32" would not guarantee identical rows. `scene_episodes` extends the support pool according to `stratified_counts(max(sizes))`, and a larger pool changes the nested-support permutations. I passed the full D-08 settings and scored only n ≤ 32.
- The time estimate (15–25 min) was high: the run took about 6.5 min.
