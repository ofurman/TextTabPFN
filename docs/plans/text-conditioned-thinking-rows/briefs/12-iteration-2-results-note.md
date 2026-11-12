# 12 Iteration-2 results note (implementer omp/task, high; box 1 h, stop by 2026-11-12T09:00Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (iteration 2), `decisions.md` (D-07, D-08, D-11 through D-16), `journal.md`, `backlog.md`, the stage card `stages/12-iteration-2-results-note.md`, and every `reports/07-*` through `reports/13-*`.

Goal: one dated research note that records the iteration-2 numbers (stages 7, 9, 13, 11) and states each verdict.

## Done when

1. `grep -l '^Verdict:' docs/research/web/*-per-feature-role-tokens-results.md` lists exactly one file. Today: no such file.
2. `./scripts/lint_related_works.sh` passes (AGENTS.md, Required validation; it checks repo-local Markdown links and anchors).

## What to write

- File: `docs/research/web/2026-11-12-per-feature-role-tokens-results.md`. Copy the frontmatter and section layout of `docs/research/web/2026-11-11-text-conditioned-thinking-rows-results.md`. That note is protected; do not edit it, but you may link to it.
- `Verdict:` lines:
  - Write exactly one line starting with `Verdict:`.
  - On that line, name the three iteration-2 verdicts as printed by the reports: `role-primary` (4000-step role tokens), `role10k-primary` (10k role tokens) and `followon-primary` (bge pooled story).
  - Give each verdict a one-clause magnitude.
- Numbers come only from the report commands. Quote them; do not recompute. The commands: `roleinfo`, `role-training`, `role-training --results results/13-role-training-10k`, `role-primary`, `role-curve`, `role-reliability`, `role10k-primary`, `followon-training`, `followon-primary`. Iteration-1 reference numbers come from `primary`. Use markdown tables for comparisons.
- Content. Keep reported results apart from interpretation:
  - **Setup.** D-11 role tokens (an exact parse of the template stories, not free-text understanding), the D-12 preprocessing baseline, the D-13 stop rule, D-15 (owner-requested 10k retrain; the 10k verdict decided D-14), and D-16 (the bge arm).
  - **Stage 7.** The value of roles to native TabPFN through preprocessing. Part of the gain uses unlabeled query features; the support-only fit keeps most of it (reports/07-review.md, reports/09-implement.md).
  - **Stage 9 and stage 13.**
    - Show the story − control results at 4000 and 10k steps side by side.
    - Split the 10k widening into the story arm's gain and the control's loss (reports/13-implement.md). The 10k control sits near native at n = 8.
    - Compare against preprocessing and native.
  - **Reliability (stage 9).**
    - "Irrelevant/wrong cost more than omitted" holds for point estimates only; the CIs overlap at n = 16 (reports/09-review.md F3).
    - Corruptions change only part of the slots: irrelevant 0.79 on average, wrong 0.67.
  - **Stage 11.** The bge story − control result. The dev-set tie (0.4377 = 0.4377) and the same-p cosine diagnostic (bge 0.9936 vs MiniLM 0.9886; reports/10-implement.md) are a plausible mechanism, not a proven cause.
  - **Interpretation.**
    - Exposing per-feature structure, rather than pooling it, is what made the shared thinking-row channel use the story.
    - A stronger pooled encoder did not.
    - Story-guided preprocessing still beats the learned adapter at small n.
  - **Limits.** Template stories only (roles parsed exactly); held-out mechanism families; raw path, no ensembling; H2 not measured; no corruption runs at 10k or for bge.
  - **Next steps.** Point to backlog items by content: B-14 (extract roles from free text with an LLM), B-01 (per-column rows), B-16, B-15, B-17, B-02, B-03.
- Links: relative links to `../../plans/text-conditioned-thinking-rows/index.md` and to the iteration-1 results note are fine. Every local link and anchor must resolve; the linter checks this.

## Facts

- Stage 11 (`report followon-primary`):
  - bge story − control: +0.0008 [+0.0000, +0.0017] / +0.0004 / +0.0001 at n = 8 / 16 / 32. D-08 verdict "harmful".
  - The verdict is marginal (reports/11-review.md F1). The n = 8 lower bound is +4.2e-6 at seed 0 and is above 0 in only 69/100 bootstrap seeds (95% CI 60–78%). A paired t-test gives p = 0.051.
  - Write the bge verdict as: "harmful (D-08, bootstrap seed 0) — marginal: holds in 69/100 seeds, paired t p = 0.051, +0.0008 nats".
- Training-run variance (reports/11-review.md F2):
  - Each arm was trained once. About 86% of the bge n = 8 gap is the difference between the two trained adapters; the same story adapter fed story vs schema text differs by only +1.2e-4. The scene bootstrap does not cover training-run variance, and the same caveat applies to the role-token verdicts.
  - Within-adapter evidence that avoids run variance comes from the stage-9 "omitted" corruption. That is the same 4000-step story adapter fed neutral tokens: omitted − story = +0.0037 / +0.0046 / +0.0035 at n = 8 / 16 / 32, all CIs above 0 (`report role-reliability`). This matches the 4000-step story − control gap (−0.0042), so the 4000-step role-token result holds within one adapter.
  - No such within-adapter check exists for the 10k pair (B-16). The extra 10k widening comes about half from the control getting worse, which may be run variance. Say so.
  - Name B-19 (multiple training seeds) as the fix.
- The verdict rule checks "harmful" before "supported" (`report.d08_verdict`). Mention it only if it matters for a verdict here.
- The working-tree edit to `docs/research/web/2026-11-04-text-tabular-attention.md` is the owner's and uncommitted. Do not modify it. If you link to it, use the file name only, with no anchor.

## Not in this stage

- README changes, related-work notes (B-08), new experiments or recomputation, edits to the iteration-1 note.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/12-implement.md`, at most 60 lines. Include changed paths, each check as `command` -> result, and what in this brief was wrong.
