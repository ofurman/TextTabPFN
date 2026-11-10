# Stage 06: Results note and verdict

Goal: one dated research note records the numbers from stages 1, 4 and 5, and gives the verdict on the hypothesis.
Box: 1h · Review: no · Depends: 4, 5

## Done when

- The note states the verdict under the D-08 rule — check: `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` — today: fails, no such file

## Facts

- Verdict rule and stop rule: D-08. If the stop rule fired after stage 1, the note reports that result as the verdict.
- The note follows the frontmatter and section layout of the existing snapshots in `docs/research/web/`.
- After the note is added, `./scripts/lint_related_works.sh` must pass, because it checks repository-local links (AGENTS.md, Required validation). Today it passes (run 2026-11-10).

## Not in this stage

- README changes. Related-work notes for TabPFN-2.5 or TuneTables (B-08).
