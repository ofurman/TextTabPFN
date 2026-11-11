# Stage 07: Value of story roles to native TabPFN

Goal: measure how much the story's role facts are worth to frozen native TabPFN-2.5 when applied as label-free preprocessing. This deciding number triggers the D-13 stop rule.
Box: 2h · Review: yes · Depends: none

## Done when

- The preprocessing-minus-native log-loss difference has a 95% scene-bootstrap CI at n = 8, 16, 32 on the 200 held-out scenes — check: `uv run python -m texttabpfn.report roleinfo` — today: fails, exit 2, invalid choice 'roleinfo'

## Facts

- The preprocessing follows D-12. It uses only the scene's role facts and unlabeled feature values.
- The story states no signs (D-07), and measurement weights have random signs (`scenes.py` `Scene._make_node`). A plain mean of a measurement group can therefore cancel.
- Rows must equal stage 4's: `evaluation.scene_episodes(evaluation.Settings(), family, seed)` over `scenes.eval_scene_specs(200)`. Native losses for these rows are in `results/04-primary/scores.csv` (arm `native`). Full settings matter because the pool size depends on the largest n (reports/05-implement.md).
- Iteration-1 reference at n = 8 / 16 / 32: native 0.6292 / 0.5879 / 0.5353; story adapter 0.6248 / 0.5763 / 0.5278 (`report curve`).
- Inference only. A stage-4-sized run took about 20 min on MPS (reports/04-implement.md).
- The preprocessing arm is reused as a baseline in stage 9.

## Not in this stage

- Any training. Extracting roles from free-form text (B-14). Iteration-1 results, which stay unchanged.
