# 03 Episodic training of story and schema-only arms (implementer omp/task, high; box 6 h, stop by 2026-11-11T05:30Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (constraints), `decisions.md` (D-01, D-04, D-06, D-07, D-09, D-10) and the stage card `stages/03-episodic-training-of-story-and-schema-on.md`.

Goal: the story adapter and the schema-only adapter are trained on training families with an identical budget, and their adapter weights are saved locally for stages 4–5.

## Done when

1. `uv run python -m texttabpfn.report training` shows both arms completed the same number of episodes and exits non-zero if either arm is missing or the counts differ (today: no `training` subcommand).
2. `uv run python -m texttabpfn.report training --validation` shows each arm's dev-scene validation log loss at every saved checkpoint, and exits non-zero if a checkpoint row is missing (today: no `training` subcommand).

Reports read committed small CSV/JSON under `results/03-training/`; they never recompute. Adapter weights (`*.pt`) stay local and gitignored (license: no distribution of weights).

## What to build (outcomes; design is yours)

- Episodic training of the stage 2 adapter, one run per arm, frozen backbone and frozen encoder:
  - an episode = one fresh scene from training families 0–3 with a seed from `TRAIN_SEED_RANGE`; support size n drawn log-uniformly from 8…1024 (stratified so both classes appear); 512 query rows disjoint from support; loss = mean query cross-entropy through the raw TabPFN-2.5 path.
  - story arm input text: schema + story, except 15% of episodes schema only (omitted) and 10% with one wrong fact (D-09).
  - schema-only arm: schema only, on the same episode sequence (same scenes, n, rows, order, seeds) and the same optimizer, learning rate, steps and batch size.
- Validation: fixed dev scenes (from `DEV_SEED_RANGE`, families 0–3), fixed support/query rows at n = 8, 32, 128 at least; record each arm's validation log loss at each checkpoint. The arm being validated reads its own training-condition text (correct story for the story arm, schema for the schema-only arm). Native TabPFN on the same rows is a useful reference line.
- Choose the episode budget so that both arms finish inside the box on MPS. Run the arms sequentially or concurrently, but the MPS device is the only accelerator. State the budget, learning rate and wall time in the report. Select the final checkpoint per arm by its dev validation loss (never held-out families) and record which checkpoint was selected.

## Facts

- Stage 2 API (commit 8cbb249):
  - `texttabpfn.adapter.TextConditionedThinkingRows.from_model(model)` gives the adapter. Its W is 64×192×384, zero-initialized, with no bias. θ is a frozen buffer and is not saved.
  - `adapter.conditioned(model, adapter, embedding)` is the context manager that installs the adapter.
  - `adapter.episode_logits(model, adapter, embedding, (xs, ys, xq, yq))` returns query logits (M, B, 2); yq is never fed forward.
  - `texttabpfn.stories.scene_text(scene, variant, rng=None)` takes variant ∈ `stories.VARIANTS` = ('story', 'omitted', 'irrelevant', 'wrong', 'one_wrong_fact'). 'irrelevant', 'wrong' and 'one_wrong_fact' need an rng. 'omitted' is the schema text and is the schema-only arm's input.
  - `texttabpfn.encoder` is the frozen MiniLM encoder with a per-text cache. Its e(t) is L2-normalized, so ‖e‖ = 1. W's learning rate must move the rows meaningfully against θ entries of order 1.
  - Tests in `tests_exp/` (parity, label isolation, stories) must still pass after this stage.
- Stage 1 (commits bdeb972, c9f99c5):
  - Per-scene optimized shared rows gained 0.162 / 0.124 / 0.075 nats at n = 8 / 16 / 32 over native on 50 dev scenes, using Adam lr 0.03 on the rows directly. That is the ceiling a perfect text-to-rows map could reach on dev scenes.
  - `texttabpfn.capacity` has stratified and nested support helpers (`stratified_index`, `nested_supports`).
  - One step of 8 episodes × (≤32 + 128 rows) took about 0.4–0.6 s on MPS.
- The interaction mechanism (families 4–5) was changed in c9f99c5 to bounded tanh products. Families 0–3 were unchanged. Training uses families 0–3 only.
- An episode needs about 0.1–1 GB with `force_recompute_layer=True` (2026-11-04 design note, implementation notes). Large n (1024) dominates time; batching several episodes of one scene in the batch dimension is allowed, but each forward sees one scene's text (one e(t) per scene).
- Training protocol background: `docs/research/web/2026-10-21-first-context-tabpfn-experiment.md#training-unit-and-required-data`.
- Resilience:
  - In stage 1 the implementer agent died on an API outage while its background run continued.
  - Run training as a background process that is resumable from its last checkpoint, and write results incrementally to `results/03-training/`.
  - Write a draft report as soon as the run starts (settings, budget, ETA), and update it at the end.
- The schema-only arm's text depends only on p, the number of features. Its W therefore learns a p-conditioned retuning of the rows. This is intended: it matches the story arm's capacity and separates the gain from story information from the gain from meta-tuning the rows.

## Not in this stage

- Any tuning on held-out families 4–5 or on `EVAL_SEED_RANGE`. LoRA or per-layer conditioning (B-06). Encoder ablation (B-09). Evaluation (stages 4, 5).
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/03-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; budget, learning rate, wall time, selected checkpoints and their validation losses; anything NOT MEASURED (reason); infra failures; what in this brief was wrong. Failed check attempts: `validation/03-<check>-<k>.txt` (≤4 KB). No pasted logs.
