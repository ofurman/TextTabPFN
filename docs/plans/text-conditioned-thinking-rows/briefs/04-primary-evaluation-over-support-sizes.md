# 04 Primary evaluation over support sizes (implementer omp/task, high; box 3 h, stop by 2026-11-11T05:50Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (constraints), `decisions.md` (D-06, D-08, D-10) and the stage card `stages/04-primary-evaluation-over-support-sizes.md`.

Goal: produce the headline H1/H3 numbers on held-out scene families: the paired story-minus-schema-only log-loss difference per support size.

## Done when

1. `uv run python -m texttabpfn.report primary` prints mean paired difference (story − schema-only) query log loss with a 95% scene-bootstrap CI (1000 resamples) at n = 8, 16, 32, and the D-08 verdict string it implies; exits non-zero if any (scene, n, arm) is missing or non-finite (today: no `primary` subcommand).
2. `uv run python -m texttabpfn.report curve` prints the same paired difference (and story − native, schema-only − native) for every n in {8, 16, 32, 64, 128, 256, 512, 1024}; exits non-zero on missing rows (today: no `curve` subcommand).

Reports read committed small CSV/JSON under `results/04-primary/`; never recompute.

## What to build (outcomes; design is yours)

- Evaluation per D-08, fixed before looking at any result:
  - 200 scenes from held-out families 4–5 (100 each), seeds from `EVAL_SEED_RANGE`.
  - Per scene: 512 fixed query rows; 5 support seeds, each a nested stratified sequence over n ∈ {8, …, 1024} (both classes present); all arms score the same rows.
  - Arms: native raw TabPFN-2.5, schema-only adapter (schema text), story adapter (schema + story text), using the checkpoints selected in stage 3 by dev validation loss.
  - Metric: per-scene mean query log loss (averaged over support seeds); paired difference per scene; scene bootstrap.
- Verdict string per D-08: "supported" if CI upper < 0 at ≥ 2 of n = 8, 16, 32; "harmful" if CI lower > 0 at any of them; otherwise "not supported". Print it; do not interpret beyond it.
- Also record per-family (4 vs 5) breakdown as a secondary table (not a check).
- Write results incrementally and make the run resumable (the API connection has dropped before); write a draft report when the run starts.

## Facts

- Stage 3 (commit bd92e7c):
  - `texttabpfn.training.load_adapter(model, arm)` with arm 'story' or 'schema' loads the selected weights (`results/03-training/weights/<arm>-selected.pt`, local, gitignored). Both arms were selected at step 3250.
  - Dev validation mean log loss: native 0.4464, story 0.4387, schema-only 0.4376.
  - Texts come from `stories.scene_text(scene, 'story')` (schema + story) and `scene_text(scene, 'omitted')` (schema only), embedded with `texttabpfn.encoder`.
  - Query logits come from `adapter.episode_logits(model, adapter, embedding, episode)`; native uses `model.binary_logits`.
- Memory and speed:
  - On MPS a training episode at n = 1024 needs about 7.4 GB with gradients. Under `torch.no_grad()` a batch of the 5 support seeds should fit, but check this.
  - Stage 3 ran at about 3.3 s per 8 forward+backward episodes while the machine had heavy unrelated CPU load.
  - Budget: 200 scenes × 8 n × 3 arms must fit in the box. If it does not, cut support seeds before scenes and state the cut.
- `texttabpfn.capacity` has nested stratified support helpers. `scenes.EVAL_SEED_RANGE` starts at 2,000,000. Families 4–5 use the bounded interaction mechanism from c9f99c5.
- The MPS device is shared with stage 5; run nothing else on it meanwhile.
- Raw TabPFN path only (D-05); no wrapper, no ensembling.

## Not in this stage

- Corrupted stories (stage 5). H2 label savings (B-03). Paraphrased stories (B-04). Wrapper ensembles (B-02). Any retraining or checkpoint reselection using held-out families.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/04-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; the primary table and verdict string; the curve table; wall time; anything NOT MEASURED (reason); infra failures; what in this brief was wrong. Failed check attempts: `validation/04-<check>-<k>.txt` (≤4 KB). No pasted logs.
