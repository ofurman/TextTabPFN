# 09 Role-token held-out evaluation (implementer omp/task, high; box 2 h, stop by 2026-11-11T21:15Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (iteration 2), `decisions.md` (D-08, D-09, D-11, D-12, D-14) and the stage card `stages/09-role-token-held-out-evaluation.md`.

Goal: report the H1, H3 and H4 numbers for role tokens on held-out families: role-token story vs matched control, across support sizes and under corrupted roles.

## Done when

1. `uv run python -m texttabpfn.report role-primary` prints, at n = 8, 16, 32, the paired per-scene difference role-token story − role-token control with a 95% scene-bootstrap CI (1000 resamples), plus the D-08 verdict string from `report.d08_verdict`. It also prints a per-family (4 vs 5) secondary table. Exit non-zero on missing or non-finite rows. Today: exit 2, invalid choice.
2. `uv run python -m texttabpfn.report role-curve` prints the same paired difference at every n in 8…1024, plus each role-token arm minus native, minus story-guided preprocessing, and minus the iteration-1 story adapter. Exit non-zero on missing rows. Today: exit 2.
3. `uv run python -m texttabpfn.report role-reliability` prints, for each corruption (omitted, irrelevant, wrong), the mean change of the role-token story arm (corrupted − correct roles) with a 95% CI at n = 8, 16, 32. Exit non-zero on missing rows. Today: exit 2.

Reports read committed small CSV/JSON under `results/09-role-eval/` and never recompute.

## What to build (outcomes; design is yours)

- Rows: the same 200 scenes, support seeds, nested supports and 512 query rows as stage 4, built with `evaluation.scene_episodes(evaluation.Settings(), family, seed)` over `scenes.eval_scene_specs(200)`. Score all 8 n.
- Arms, all on identical rows:
  - role-token story and role-token control, using the stage-8 selected checkpoints;
  - for reference, existing scores may be read rather than rescored if a re-scored sample matches to 1e-6:
    - native, and the iteration-1 story and schema adapters, from `results/04-primary/scores.csv`;
    - story-guided preprocessing, from `results/07-roleinfo/scores.csv`, with the support+query fit;
  - add one secondary arm: story-guided preprocessing with the collapse fitted on support features only (reviewer P3 in reports/07-review.md). This separates the role facts from unlabeled query information.
- `roleinfo.preprocess_episode` must receive the full 512-row query batch before any chunking. Its output depends on the whole query batch (reports/07-review.md).
- Corruptions (D-09) of the role-token story arm, n = 8, 16, 32 only, each with its own fixed per-scene RNG stream:
  - omitted: all slots neutral;
  - irrelevant: the roles of a same-family donor (reuse `stories.donor_scene`), truncated or padded to this scene's p;
  - wrong: roles permuted among this scene's features (`stories.wrong_roles`).
  - Refuse to score a corruption identical to the correct roles.
- Make the run resumable, with incremental results and a draft report at the start. Inference only: stage 4 took about 20 min for 3 arms × 8 n.

## Facts

- Stage 8 (commits eaee451, 26aacb7):
  - `texttabpfn.role_training.load_adapter(model, arm)`, with arm `story` or `control`, loads the selected weights: story at step 4000, control at step 3750.
  - `texttabpfn.role_tokens`:
    - `variant_sentences(scene, variant, rng)` builds the sentences, with variants `story`, `omitted`, `wrong` and `one_wrong_fact`;
    - `slot_tokens(encoder, sentences)` returns a (15, 384) tensor;
    - `RoleTokenThinkingRows.set_text` takes (S, D) or (B, S, D).
  - `adapter.episode_logits` and `conditioned` work unchanged with this adapter.
  - The role-token module has no `irrelevant` variant. Build it from `stories.donor_scene`, then truncate or pad to this scene's p.
  - Dev validation (64 scenes): story 0.5403 / 0.4163 / 0.3440 at n = 8 / 32 / 128; control 0.5509 / 0.4193 / 0.3442; native 0.5634 / 0.4279 / 0.3479.
- Iteration-1 baseline on these rows (`report primary`): story − schema = −0.0008 / +0.0001 / +0.0012 at n = 8 / 16 / 32, verdict "harmful".
- Stage 7 (`report roleinfo`): preprocessing − native = −0.0165 / −0.0204 / −0.0156 at n = 8 / 16 / 32.
- `report.d08_verdict` checks "harmful" before "supported". Reuse it unchanged.
- The orchestrator applies D-14 to the `role-primary` verdict after this stage.

## Not in this stage

- Retraining or checkpoint reselection on held-out families. The follow-on arm (stage 10).
- Iteration-1 code paths and results stay unchanged, and their report commands must still exit 0.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/09-implement.md`, at most 60 lines:
- changed paths;
- each check as `command` -> result;
- the role-primary table and its verdict;
- the curve summary;
- the reliability table;
- wall time;
- anything NOT MEASURED;
- what in this brief was wrong.

Failed attempts go to `validation/09-<check>-<k>.txt`.
