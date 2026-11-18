# 06 Results note and verdict (implementer omp/task, high; box 1 h, stop by 2026-11-11T04:30Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md`, `decisions.md` (D-01…D-10), `journal.md`, `backlog.md`, the stage card `stages/06-results-note-and-verdict.md`, and every report in `reports/`.

Goal: one dated research note records the numbers from stages 1, 3, 4 and 5 and states the verdict on the hypothesis under the D-08 rule.

## Done when

1. `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` lists exactly one file (today: no such file).
2. `./scripts/lint_related_works.sh` passes (AGENTS.md, Required validation; it checks repo-local Markdown links and anchors).

## What to write

- File: `docs/research/web/2026-11-11-text-conditioned-thinking-rows-results.md`. Frontmatter and section layout follow the existing snapshots (e.g. `docs/research/web/2026-10-21-first-context-tabpfn-experiment.md`: `date`, `researcher: "Oleksii Furman"`, `topic`, `tags`, `sources`, `status`, `last_updated`, `last_updated_note`; H1 `# Research: …`; sections such as Research Question, Summary, Detailed Findings, Confidence Notes, Open Questions).
- Exactly one line starting `Verdict:` that states the verdict string printed by `uv run python -m texttabpfn.report primary` (D-08) verbatim, followed on the same line by a one-clause magnitude qualifier.
- Numbers must come from the report commands (`report capacity`, `report capacity --arm per-column`, `report training`, `report training --validation`, `report primary`, `report curve`, `report reliability`); quote them, do not recompute. Use markdown tables for the numeric comparisons.
- Content, distinguishing reported results from interpretation:
  - setup in brief: channel θᵢ + Wᵢ·e(t), frozen TabPFN-2.5 (tabpfn 9.2.0) and frozen all-MiniLM-L6-v2, generator families, template stories (D-06, D-07), training budget, evaluation protocol (D-08), corruptions (D-09);
  - stage 1 channel capacity (per-scene optimized rows, shared vs per-column) — the ceiling;
  - stage 3 dev validation (story vs schema-only vs native);
  - stage 4 primary and curve (H1, H3), incl. the per-family secondary table;
  - stage 5 reliability (H4);
  - interpretation: the gap between the stage-1 ceiling and the learned gain; the encoder-similarity diagnostic in `journal.md` (stage 03 diagnostic: same-p story cosine 0.989); the effect magnitudes relative to the D-08 rule; what the result does and does not show (H2 not measured; held-out families only; template stories only; raw path, no ensembling);
  - next steps pointing to backlog items (B-01, B-09, B-12, etc.) by content, not by plan IDs alone.
- Links: relative links to `../../plans/text-conditioned-thinking-rows/index.md` and to the 2026-11-04 design note are fine; every local link and anchor must resolve (the linter checks).

## Facts

- Stage 5 (commit 6b10c09):
  - The story adapter does respond to whether a story is present. Leaving it out costs +0.0015 [+0.0010, +0.0020] at n = 8.
  - It does not measurably respond to what the story says. The CIs for the irrelevant and wrong (permuted-roles) corruptions include 0 at n = 8, 16 and 32.
  - Rows are identical to stage 4.
  - The implementer's reading, which you should label as inference: the adapter uses whether a story is present, not its content.
- Stage 3 encoder diagnostic (`journal.md`, 2026-11-10T23:30Z), read-only and run by the orchestrator: 40 dev stories embedded with MiniLM.
  - Mean cosine between stories is 0.964, and 0.989 among stories with the same p.
  - After removing the mean, the residual norm is 0.185.
  - This is a plausible mechanism for the null content effect. It is not a proven cause; say so.
- The working-tree edit to `docs/research/web/2026-11-04-text-tabular-attention.md` is uncommitted and belongs to the owner. Do not modify that file. Link to it by file name only, with no `#anchor`, so the link stays valid at every commit.
- Stage 4 review (reports/04-review.md):
  - The D-08 rule says nothing about precedence. `d08_verdict` checks "harmful" before "supported". In this run no upper bound is below 0, so the order does not change the verdict. Say this in one sentence.
  - The effects are about 1e-3 nats. Story − schema at n = 32 is +0.0012 [+0.0005, +0.0019]. For scale, both adapters beat native by about 0.012 at n = 16. Give the magnitude next to the verdict so readers do not take "harmful" to mean a large effect.

## Not in this stage

- README changes. Related-work notes (B-08). Any new experiment or recomputation.
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/06-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; anything NOT MEASURED (reason); what in this brief was wrong.
