# Stage 09: Role-token held-out evaluation

Goal: report the H1, H3 and H4 numbers for role tokens on held-out families. This result is the D-14 input.
Box: 2h · Review: yes · Depends: 8

## Done when

- The role-token-story minus matched-control difference has a 95% CI at n = 8, 16, 32 with its D-08 verdict — check: `uv run python -m texttabpfn.report role-primary` — today: fails, exit 2, invalid choice
- The same difference is reported at every n in 8 … 1024 — check: `uv run python -m texttabpfn.report role-curve` — today: fails, exit 2, invalid choice
- Each role corruption's change relative to the correct roles has a 95% CI at n = 8, 16, 32 — check: `uv run python -m texttabpfn.report role-reliability` — today: fails, exit 2, invalid choice

## Facts

- Use stage 4's 200 scenes, support seeds and query rows (`evaluation.scene_episodes` with full `evaluation.Settings()`).
- All arms score identical rows: native; story-guided preprocessing (stage 7, D-12); the iteration-1 story and schema adapters (`training.load_adapter`) as references; role-token story; role-token control.
- Verdict and bootstrap follow D-08; `report.d08_verdict` checks "harmful" first.
- Corruptions (D-09) on the roles: omitted = all slots neutral; irrelevant = a same-family donor's roles; wrong = roles permuted among the features.
- Iteration-1 baseline: story − schema-only was +0.0012 [+0.0005, +0.0019] at n = 32 (`report primary`).
- After this stage the orchestrator applies D-14 and records the choice in `journal.md`.

## Not in this stage

- Any retraining or checkpoint reselection on held-out families. The follow-on arm (stage 10).
