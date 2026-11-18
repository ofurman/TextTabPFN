# Stage 10: Follow-on arm training

Goal: train the follow-on arm that D-14 selects, B-09 or B-01, and its matched control, with the same episodes and budget.
Box: 6h · Review: no · Depends: 9, 13

## Done when

- At initialization the follow-on adapter reproduces native raw TabPFN-2.5 logits — check: `uv run pytest tests_exp/test_followon.py` — today: fails, exit 4, file not found
- Both follow-on arms completed the same episode budget — check: `uv run python -m texttabpfn.report followon-training` — today: fails, exit 2, invalid choice <!-- lint: one-condition -->

## Facts

- Branch B-09, if stage 9 is "supported" (D-14):
  - the iteration-1 architecture θᵢ + Wᵢ·e(t), with e(t) from frozen `BAAI/bge-base-en-v1.5` (768-d, 512 tokens; its card recommends CLS pooling);
  - story = schema + story text; control = schema text; the download is allowed (index.md).
- Branch B-01, otherwise (D-14):
  - per-column rows θᵢ + Wᵢ·s_c, where s_c holds the stage-8 role tokens of the features in TabPFN column c; the target column gets a fixed conditioning input;
  - control = the stage-8 neutral tokens;
  - stage 1 per-column capacity was 0.200 / 0.154 / 0.100, versus shared 0.162 / 0.124 / 0.075, at n = 8 / 16 / 32 (`report capacity`).
- Trainer, D-09 mix, budget and dev-only selection are as in stage 8. The parameter budget follows D-11.

## Not in this stage

- The branch D-14 does not select (backlog). Held-out evaluation (stage 11). Per-layer conditioning (B-06).
