# 13 Ten-thousand-step role-token evaluation (implementer omp/task, high; box: the evaluation part, 1.5 h, stop by STOPBY)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md`, `decisions.md` (D-08, D-11, D-14, D-15), the stage card `stages/13-ten-thousand-step-role-token-training.md`, and `reports/08-implement.md` and `reports/09-implement.md`.

Goal: report the held-out story-vs-control result for the 10k-step role-token arms. The orchestrator applies D-14 to this verdict (D-15).

## Done when

1. `uv run python -m texttabpfn.report role-training --results results/13-role-training-10k` shows that both 10k-step arms completed the same episode budget. Exit non-zero if an arm is missing or the counts differ. With no `--results` option, the output must be unchanged from today's stage-8 output. Today the command fails: there is no `--results` option.
2. `uv run python -m texttabpfn.report role10k-primary` prints, at n = 8, 16 and 32, the paired per-scene difference 10k story − 10k control with a 95% scene-bootstrap CI and the D-08 verdict from `report.d08_verdict`. Exit non-zero on missing or non-finite rows. Secondary, not checks:
   - the per-family table;
   - every n from 8 to 1024;
   - 10k story − 4000-step story and 10k control − 4000-step control on the same rows.

   Today the command fails with exit 2, invalid choice.

Reports read committed small CSV/JSON under `results/13-role-eval-10k/` (and `results/13-role-training-10k/`) and never recompute.

## What to build (outcomes; design is yours)

- Score the 10k-step selected weights (`role_training.load_adapter(model, arm, out_dir=Path("results/13-role-training-10k"))`) on exactly the stage-4 rows. Use `evaluation.scene_episodes(evaluation.Settings(), f, s)` over `eval_scene_specs(200)`, all 8 n.
- Reuse `texttabpfn.role_eval` (stage 9). Parameterize it rather than copying it, keeping stage-9 behaviour and results unchanged; `role-primary`, `role-curve` and `role-reliability` must still exit 0.
- No corruption runs (B-16).
- Make the run detached and resumable (`nohup ... &`). Write the draft report, with the launch and resume command, before waiting. The API has dropped twice in this run.

## Facts

- STAGE13FACTS
- Stage 9 (4000 steps): story − control −0.0042 / −0.0042 / −0.0041 at n = 8 / 16 / 32, verdict supported (`report role-primary`).
- The stage-9 evaluation took about 27 min for 2 arms × 8 n plus corruptions, while the 10k training ran on the same MPS device. This evaluation runs alone.

## Not in this stage

- Retraining, checkpoint reselection, any held-out-based choice, corruption runs (B-16), stage 10.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/13-implement.md`, at most 60 lines:
- changed paths;
- each check as `command` -> result;
- the 10k primary table and its verdict;
- the 10k vs 4000-step comparison;
- wall time;
- anything NOT MEASURED;
- what in this brief was wrong.
