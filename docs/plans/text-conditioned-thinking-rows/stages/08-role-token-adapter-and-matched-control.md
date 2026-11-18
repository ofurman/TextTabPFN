# Stage 08: Role-token adapter and matched control training

Goal: train the per-feature role-token adapter and its matched schema control on the training families, with the same episodes and budget.
Box: 6h · Review: no · Depends: 7

## Done when

- At initialization the role-token adapter reproduces native raw TabPFN-2.5 logits — check: `uv run pytest tests_exp/test_role_tokens.py` — today: fails, exit 4, file not found
- Both role-token arms completed the same episode budget — check: `uv run python -m texttabpfn.report role-training` — today: fails, exit 2, invalid choice <!-- lint: one-condition -->

## Facts

- The design is fixed by D-11: per-feature tokens in column order through the shared thinking-row channel, with a matched control.
- Per-feature roles come from `Scene.describe()["roles"]`, which lists roles, measurement groups and parent/child relations in column order. The corruption helpers are in `texttabpfn.stories`: `one_wrong_fact_roles`, `wrong_roles` and `donor_scene`.
- Reuse the iteration-1 trainer `texttabpfn.training`:
  - `make_episode` and `DevSet`, with checkpoints selected on dev scenes only;
  - the D-09 mix, where "omitted" means every slot gets the neutral schema token.
- Iteration-1 budget: 4000 steps × 8 episodes per arm, Adam lr 1e-3 with cosine decay. It took 3.65 h for two lockstep arms on MPS, with a peak of 7.4 GB at n = 1024 (reports/03-implement.md).
- Column grouping: feature j sits in TabPFN column j // 3 at position j % 3 (`features_per_group = 3`, tabpfn 9.2.0 `_pad_and_reshape_feature_groups`). There are at most 15 features.
- Iteration-1 code paths and results stay unchanged (index.md, Protected).

## Not in this stage

- Per-column rows (stage 10, if D-14 selects B-01). A new encoder (B-09). Held-out evaluation (stage 9).
