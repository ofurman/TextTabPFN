# Stage 11: Follow-on held-out evaluation

Goal: report the follow-on arm's held-out result against its matched control, using the iteration-1 protocol.
Box: 2h · Review: yes · Depends: 10

## Done when

- The follow-on story-minus-control difference has a 95% CI at n = 8, 16, 32 with its D-08 verdict — check: `uv run python -m texttabpfn.report followon-primary` — today: fails, exit 2, invalid choice

## Facts

- Use the same 200 held-out scenes, support seeds and query rows as stages 4 and 9 (`evaluation.scene_episodes` with full `evaluation.Settings()`).
- Show the same-row losses of the native, preprocessing (D-12) and stage-9 role-token arms as context.
- Verdict and bootstrap follow D-08.
- Score n = 8 … 1024 so the results note can show the curve. The DoD check covers n = 8, 16, 32 only.

## Not in this stage

- Corruption runs for the follow-on arm (backlog). Retraining or selecting on held-out families.
