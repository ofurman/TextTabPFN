# 08 Role-token adapter and matched control training (implementer omp/task, high; box 6 h, stop by 2026-11-11T10:45Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (iteration 2), `decisions.md` (D-01, D-04, D-09, D-11) and the stage card `stages/08-role-token-adapter-and-matched-control.md`.

Goal: train the per-feature role-token adapter (D-11) and its matched schema control on the training families, with the same episodes and budget, and save their selected weights locally for stage 9.

## Done when

1. `uv run pytest tests_exp/test_role_tokens.py` passes (today: exit 4, file not found). It must show:
   - at initialization, the role-token adapter reproduces native raw TabPFN-2.5 logits within 1e-5 on CPU and on MPS;
   - changing query labels leaves its logits unchanged;
   - with nonzero weights, logits depend on the role assignment, so the test cannot pass trivially.
2. `uv run python -m texttabpfn.report role-training` shows both role-token arms completed the same number of episodes. It must exit non-zero if an arm is missing or the counts differ (today: exit 2, invalid choice). Also provide `role-training --validation`, the per-checkpoint dev validation table, as in iteration 1.

Reports read committed small CSV/JSON under `results/08-role-training/` and never recompute. Weights (`*.pt`) stay local and gitignored.

## What to build (outcomes; design is yours within D-11)

- Per-feature tokens:
  - Feature x_j gets the frozen MiniLM embedding (`texttabpfn.encoder`) of a sentence stating only x_j's role from the story (`Scene.describe()["roles"]`): influences y, is influenced by y, measures hidden quantity k, or is unrelated to y.
  - The sentence must not contain the feature name, so all tokens of one role are identical across features. The slot carries the position.
  - Measurement-group identity must be distinguishable between groups within a scene, e.g. "hidden quantity A" vs "B".
  - Tokens sit in slot j (column order) of at most 15 slots; empty slots are zero.
  - The control arm's slot j is the embedding of one neutral sentence ("a numeric feature"), present only for existing features. The two arms then differ only in role content.
- Conditioning:
  - θᵢ + Wᵢ·s, shared over all columns as in iteration 1 (rows identical across columns), W zero-initialized, so parity holds exactly at initialization.
  - s is a learned reduction of the slot tokens. A shared per-slot projection 384→d followed by concatenation is one option.
  - Keep the total trainable parameters near iteration 1's 4.7M and state the count.
- Training, reusing `texttabpfn.training` where possible:
  - the same episode stream (`make_episode`: scenes, n, rows, order), the D-09 mix and dev-only checkpoint selection;
  - the same budget (4000 steps × 8 episodes), Adam with peak lr 1e-3, warm-up and cosine.
  - The mix applied to roles: omitted = all slots neutral; one_wrong_fact = exactly one feature's role changed (`stories.one_wrong_fact_roles`).
  - Generate the mix with a separate RNG stream, so the rows stay identical to iteration 1's episodes.
  - The learning rate may differ from 1e-3 if dev-only pilots justify it; record the pilots. Both arms must use the same lr.
- Validation: the same dev scenes and sizes as iteration 1, with each arm reading its training-condition tokens. Record native as a reference.
- Run as a resumable background process. Write a draft report when training starts, with settings, the parameter count and an ETA.

## Facts

- Stage 7 (commit 241746f): native TabPFN-2.5 scored on story-role-preprocessed rows has these preprocessed − native differences on held-out families:
  - n = 8: −0.0165 [−0.0200, −0.0130];
  - n = 16: −0.0204;
  - n = 32: −0.0156;
  - n = 1024: +0.0132.
  - In the preprocessing, irrelevant features are dropped and measurement groups become their first principal component.
  - Role knowledge is therefore worth about 0.016–0.020 nats at small n, compared with 0.004–0.012 for the iteration-1 adapter over native. A role-token adapter has room to capture this.
- Iteration 1 (reports/03-implement.md):
  - 4000 × 8 episodes, two lockstep arms, 3.65 h on MPS under heavy unrelated CPU load.
  - Peak 7.4 GB at n = 1024 with `force_recompute_layer=False`.
  - Batching episodes gave no speed-up, so gradients were accumulated over single-episode passes.
- Iteration-1 adapter: `texttabpfn.adapter.TextConditionedThinkingRows`, with W of shape 64×192×384 and no bias, θ a frozen buffer, `conditioned()` and `episode_logits()`. A new module may subclass or mirror it, but must not change iteration-1 behaviour. `tests_exp` (99 tests) and all iteration-1 report commands must still pass.
- Column grouping: feature j sits in TabPFN column j // 3 at position j % 3 (`features_per_group = 3`). There are at most 15 features.
- Iteration-1 diagnostic: pooled MiniLM story embeddings had a same-p cosine of 0.989 (journal 2026-11-10T23:30Z). That limitation motivates this stage.

## Not in this stage

- Per-column rows (stage 10, if D-14 selects B-01). A new encoder (B-09). Held-out evaluation (stage 9). Any use of evaluation families 4–5 or `EVAL_SEED_RANGE`.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/08-implement.md`, at most 60 lines:
- changed paths;
- each check as `command` -> result;
- the token design and parameter count;
- budget, lr, wall time;
- the selected checkpoints and their dev losses next to native;
- anything NOT MEASURED;
- what in this brief was wrong.

Failed attempts go to `validation/08-<check>-<k>.txt`.
