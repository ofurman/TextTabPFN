# Stage 03: Episodic training of story and schema-only arms

Goal: the story adapter and the schema-only adapter are trained on the training families with an identical budget.
Box: 6h · Review: no · Depends: 2

## Done when

- Both arms completed the same episode budget — check: `uv run python -m texttabpfn.report training` — today: fails, no module `texttabpfn` <!-- lint: one-condition -->
- The dev-scene validation log loss of each arm is recorded per checkpoint — check: `uv run python -m texttabpfn.report training --validation` — today: fails, no module `texttabpfn`

## Facts

- Each episode is one fresh scene from a training family. n is drawn log-uniformly from 8 … 1024, with 512 query rows. The loss is query BCE through the frozen backbone ([2026-10-21 protocol](../../../research/web/2026-10-21-first-context-tabpfn-experiment.md#training-unit-and-required-data)).
- The story arm reads schema plus story. 15% of its episodes omit the story, and 10% carry one wrong fact (D-09).
- The schema-only arm reads the schema only, on the same episodes and seeds.
- With checkpointing, an episode needs about 0.1–1 GB ([2026-11-04 note](../../../research/web/2026-11-04-text-tabular-attention.md#implementation-notes-for-design-b)).
- The run uses the local MPS device only (D-05).

## Not in this stage

- Any tuning on held-out families. LoRA or per-layer conditioning (B-06). An encoder ablation (B-09).
