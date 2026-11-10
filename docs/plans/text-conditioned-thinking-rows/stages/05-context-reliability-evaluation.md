# Stage 05: Context reliability evaluation

Goal: the H4 number, i.e. how much omitted, irrelevant and wrong stories change the story arm's loss.
Box: 2h · Review: no · Depends: 3

## Done when

- The change of each corruption type relative to the correct story at n = 8, 16, 32 has a 95% scene-bootstrap CI — check: `uv run python -m texttabpfn.report reliability` — today: fails, no module `texttabpfn` <!-- lint: one-condition -->

## Facts

- Same evaluation scenes, support draws and query rows as stage 4 (D-08).
- The three corruptions (D-09):
  - omitted: the schema only;
  - irrelevant: the story of another scene from the same family;
  - wrong: the feature roles permuted.
- Training saw one-fact-wrong stories in 10% of episodes. Results are reported against that rate ([2026-11-04 note, stress-test claim 3](../../../research/web/2026-11-04-text-tabular-attention.md#claims-that-do-not-hold)).
- The MPS device is shared with stage 4, so the stages run one after the other.

## Not in this stage

- Retraining for robustness to wrong stories. The schema-only arm under corruption, which has no story to corrupt.
