# Stage 13: Ten-thousand-step role-token training and evaluation

Goal: check whether the role-token result in stage 9 is limited by training length, by retraining both role-token arms for 10,000 steps and scoring them on the held-out rows.
Box: 9h · Review: no · Depends: 8

## Done when

- Both 10k-step arms completed the same episode budget — check: `uv run python -m texttabpfn.report role-training --results results/13-role-training-10k` — today: fails, no `--results` option <!-- lint: one-condition -->
- The 10k-step story-minus-control difference has a 95% CI at n = 8, 16, 32 with its D-08 verdict — check: `uv run python -m texttabpfn.report role10k-primary` — today: fails, exit 2, invalid choice

## Facts

- Owner request (2026-11-11): train for 10k steps, because the 4000-step story arm was selected at its final step while still improving (reports/08-implement.md).
- Launched by the orchestrator at 19:17Z, detached: `uv run python -m texttabpfn.role_training --steps 10000 --val-every 500 --out results/13-role-training-10k`. The command resumes from the last checkpoint.
- Everything matches stage 8 except the step count and validation interval: same episode stream, D-09 mix, peak lr 1e-3, and cosine schedule, here stretched over 10,000 steps.
- It ran concurrently with stage 9's inference, which shared the MPS device.
- Expected time: about 2.65 s per step, so about 7.4 h of training plus about 20 min of validation.
- D-15 decides which run feeds D-14.

## Not in this stage

- Changing lr, architecture or data (backlog). Corruption runs at 10k steps (B-16).
