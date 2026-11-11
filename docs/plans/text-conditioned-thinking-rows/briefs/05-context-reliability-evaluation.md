# 05 Context reliability evaluation (implementer omp/task, high; box 2 h, stop by STOPBY)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (constraints), `decisions.md` (D-08, D-09, D-10) and the stage card `stages/05-context-reliability-evaluation.md`.

Goal: produce the H4 number, i.e. how much omitted, irrelevant and wrong stories change the story arm's loss relative to the correct story.

## Done when

1. `uv run python -m texttabpfn.report reliability` prints, for each corruption (omitted, irrelevant, wrong), the mean per-scene log-loss change (corrupted − correct story) of the story adapter with a 95% scene-bootstrap CI (1000 resamples) at n = 8, 16, 32; exits non-zero on missing or non-finite rows (today: no `reliability` subcommand).

Report reads committed small CSV/JSON under `results/05-reliability/`; never recomputes.

## What to build (outcomes; design is yours)

- Same 200 held-out scenes, support seeds, nested supports and 512 query rows as stage 4 (reuse its row construction so rows are identical; the correct-story losses may be read from `results/04-primary/` if computed on identical rows, otherwise recompute them).
- Corruptions via `texttabpfn.stories.scene_text(scene, variant, rng)` (D-09): omitted (schema only), irrelevant (another same-family scene's story), wrong (feature roles permuted). Fix the corruption rng per scene so results are reproducible.
- Story adapter only (selected stage-3 checkpoint). Also print, as context, schema-only − correct story and native − correct story on the same rows (secondary, not a check).
- Training exposed the story arm to 15% omitted and 10% one-wrong-fact episodes (D-09); state this next to the table.
- Resumable, incremental results; draft report when the run starts.

## Facts

- STAGE4FACTS
- The MPS device is shared with stage 4; stage 4 has finished before this starts.

## Not in this stage

- Retraining for robustness. Corrupting the schema-only arm (it has no story).
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/05-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; the reliability table; wall time; anything NOT MEASURED (reason); infra failures; what in this brief was wrong. Failed check attempts: `validation/05-<check>-<k>.txt` (≤4 KB). No pasted logs.
