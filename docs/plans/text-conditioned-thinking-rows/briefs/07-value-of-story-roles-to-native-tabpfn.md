# 07 Value of story roles to native TabPFN (implementer omp/task, high; box 2 h, stop by 2026-11-11T06:20Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (iteration 2 goal, DoD, constraints), `decisions.md` (D-07, D-08, D-12, D-13) and the stage card `stages/07-value-of-story-roles-to-native-tabpfn.md`.

Goal: measure how much the story's role facts are worth to frozen native TabPFN-2.5 when used as label-free preprocessing. This deciding number triggers the D-13 stop rule.

## Done when

1. `uv run python -m texttabpfn.report roleinfo` prints, at n = 8, 16 and 32:
   - mean query log loss for native and for preprocessed;
   - the paired per-scene difference preprocessed − native, with a 95% scene-bootstrap CI (1000 resamples, same helper as stage 4).

   Also print the same difference for every n in 8…1024 as a secondary table, plus a per-family (4 vs 5) breakdown. Exit non-zero if any (scene, n) is missing or non-finite. Today: exit 2, invalid choice 'roleinfo'.

The report reads committed small CSV/JSON under `results/07-roleinfo/` and never recomputes.

## What to build (outcomes; design is yours)

- The preprocessing per D-12 uses only the scene's role facts (`Scene.describe()["roles"]`, which is the exact content of the template story) and feature values, never labels:
  - drop the features with role `irrelevant`;
  - collapse each measurement group (role `measure`, same `group`) to one feature with a label-free method;
  - keep the Markov-blanket features (`mb`) unchanged.
- Measurement weights carry random signs (`scenes.py` `Scene._make_node`), so a plain mean can cancel. Pick a sign-robust label-free method: for example, the first principal component of the group's standardized features, or sign-aligning to the group's first member via correlation.
  - You may fit it on support features only, or on support plus query features. Both are unlabeled inputs available at prediction time.
  - State your choice and keep it fixed before looking at results.
- Score native TabPFN-2.5 (raw path, `model.binary_logits`) on the preprocessed rows. Use exactly the stage 4 rows: `evaluation.scene_episodes(evaluation.Settings(), family, seed)` over `scenes.eval_scene_specs(200)`, all 8 n, all 5 support seeds, 512 query rows.
  - Native losses on the unprocessed rows already exist in `results/04-primary/scores.csv` (arm `native`).
  - Either read them (and assert that a re-scored sample matches to 1e-6) or re-score them.
- Make the run resumable with incremental results. Write a draft report when it starts.
- Stage 9 will reuse this arm on the same rows, so make the preprocessing a reusable function.
- Add CPU-only tests in `tests_exp/` for the preprocessing:
  - no label access;
  - the irrelevant features are removed;
  - the measurement group collapses to one column.

## Facts

- Iteration-1 reference at n = 8 / 16 / 32 (`report curve`): native 0.6292 / 0.5879 / 0.5353; story adapter 0.6248 / 0.5763 / 0.5278.
- A stage-4-sized inference run took about 20 min on MPS. The 5 support seeds are batched in one forward, at about 0.25 s per pass even at n = 1024 (reports/04-implement.md).
- `_remove_constant_features` drops constant columns. The collapsed features are continuous, so this is not a concern.
- Row construction must use the full stage-4 `Settings()`, because the support pool size depends on the largest n (reports/05-implement.md).
- D-13 stop rule: if the CI includes 0 at all of n = 8, 16, 32, the orchestrator stops the run. Report the numbers plainly. Do not tune the preprocessing after seeing results.

## Not in this stage

- Any training. Extracting roles from free-form text (B-14).
- Iteration-1 code paths and results stay unchanged; their report commands must still exit 0.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/07-implement.md`, at most 60 lines:
- changed paths;
- each check as `command` -> result;
- the chosen collapse method;
- the roleinfo table;
- wall time;
- anything NOT MEASURED (reason);
- what in this brief was wrong.

Failed check attempts go to `validation/07-<check>-<k>.txt` (≤4 KB).
