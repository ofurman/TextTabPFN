# 11 Follow-on held-out evaluation (implementer omp/task, high; box 2 h, stop by 2026-11-12T08:50Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md`, `decisions.md` (D-08, D-14, D-15, D-16) and the stage card `stages/11-follow-on-held-out-evaluation.md`.

Goal: report the held-out result of the B-09 bge-base story arm against its schema control, using the iteration-1 protocol.

## Done when

1. `uv run python -m texttabpfn.report followon-primary` exits 0 (today: exit 2, invalid choice). It prints:
   - at n = 8, 16 and 32, the paired per-scene difference bge story − bge control, with a 95% scene-bootstrap CI (1000 resamples);
   - the D-08 verdict from `report.d08_verdict`.

   It exits non-zero on missing or non-finite rows.

   Secondary output, not part of the check, all on the same rows:
   - the per-family table;
   - every n from 8 to 1024;
   - context arms: native, story-guided preprocessing (support+query), the iteration-1 MiniLM story and schema adapters, and the 4000-step and 10k role-token story arms.

The report reads committed small CSV/JSON under `results/11-followon-eval/` and never recomputes.

## What to build (outcomes; design is yours)

- Score the stage-10 selected bge story and control weights on exactly the stage-4 rows: `evaluation.scene_episodes(evaluation.Settings(), f, s)` over `eval_scene_specs(200)`, all 8 n.
- Context arms:
  - Read them from the existing CSVs: `results/04-primary` (native, iteration-1), `results/07-roleinfo` (preprocessing), `results/09-role-eval` (4000-step role tokens) and `results/13-role-eval-10k` (10k role tokens).
  - Assert that a re-scored sample of the native arm matches to 1e-6.
- Reuse and parameterize `texttabpfn.evaluation` / `texttabpfn.role_eval`; do not change earlier behaviour or results.
- Run detached with `nohup ... &`, resumable. Write the draft report with the launch/resume command before waiting. The API connection has dropped twice in this run.

## Facts

- Stage 10 (commit b583d2c), code in `src/texttabpfn/followon.py`:
  - Frozen bge-base-en-v1.5 (CLS, L2-normalized). Adapter θ + W·P·e(t) with 5,013,504 parameters per arm. Results are in `results/10-followon-training/`; weights are local.
  - Both arms were selected at step 3000. Dev mean: story 0.4377, control 0.4377. Dev n = 8: 0.5511 vs 0.5507.
  - bge same-p story cosine is 0.9936 (MiniLM 0.9886).
  - A held-out null is the expected result. Report it plainly; do not tune anything.
- Iteration-1 MiniLM pair on these rows (`report primary`): −0.0008 / +0.0001 / +0.0012 at n = 8 / 16 / 32, verdict harmful.
- 10k role tokens (`report role10k-primary`): −0.0098 / −0.0080 / −0.0057, verdict supported.
- Two-arm scoring without corruptions took 12.7 min for stage 13.

## Not in this stage

- Corruption runs for the bge arm (B-15). Retraining or reselecting on held-out data. Any earlier stage's results.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/11-implement.md`, at most 60 lines. Include:
- changed paths;
- each check as `command` -> result;
- the primary table and verdict;
- the context comparison;
- wall time;
- anything NOT MEASURED;
- what in this brief was wrong.
