# Stage 12: Iteration-2 results note

Goal: one dated research note records the numbers from stages 7, 9 and 11, with their two verdicts.
Box: 1h · Review: no · Depends: 9, 11

## Done when

- The note states the iteration-2 verdicts — check: `grep -l '^Verdict:' docs/research/web/*-per-feature-role-tokens-results.md` — today: fails, no such file

## Facts

- The verdict rule (D-08) and the follow-on choice (D-14) come from decisions.md. Numbers are quoted from the `report` subcommands, not recomputed.
- The layout follows `docs/research/web/2026-11-11-text-conditioned-thinking-rows-results.md`. That note is protected and is not edited.
- Per AGENTS.md, `./scripts/lint_related_works.sh` must pass after the note is added; it checks repo-local links. It passes today (2026-11-11).
- Label the role tokens as an exact parse of template stories (D-11), not as free-form text understanding.

## Not in this stage

- README or related-work notes (B-08). New experiments.
