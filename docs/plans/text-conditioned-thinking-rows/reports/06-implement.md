# 06 Results note and verdict — implement report

## Changed paths
- Added `docs/research/web/2026-11-11-text-conditioned-thinking-rows-results.md` (uncommitted).

## Content
- Frontmatter and sections follow the 2026-10-21 snapshot: Research Question, Summary, Detailed Findings (setup, stages 1/3/4/5, interpretation, scope), Confidence Notes, Open Questions.
- There is one `Verdict:` line: `Verdict: harmful — but only by about 1e-3 nats (+0.0012 [+0.0005, +0.0019] at n = 32) ...`. "harmful" is copied exactly from `report primary`.
- The note includes the precedence sentence: harmful is checked before supported, and here that order does not change the verdict.
- All numbers are copied from the seven report commands, run on 2026-11-11, and shown in markdown tables. The family names come from `src/texttabpfn/scenes.py` FAMILIES.
- The stage-5 reading is labeled [INFERENCE]. The encoder diagnostic (0.964 / 0.989 / 0.185) is called a plausible mechanism, not a proven cause.
- The note says H2 was not measured. Next steps cite B-12, B-09, B-01, B-06, B-02, B-04, B-05 and B-03, each described by its content.
- Links: the plan `index.md` and the design note `2026-11-04-text-tabular-attention.md`, by file name with no anchor. The design note was not modified.

## Checks
- `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` -> exactly one file listed, the new note.
- `./scripts/lint_related_works.sh` -> "Related-work lint passed", exit 0.
- Report commands (`capacity`, `capacity --arm per-column`, `training`, `training --validation`, `primary`, `curve`, `reliability`) -> all printed complete tables.

## NOT MEASURED
- None in this stage. The note reports H2 as not measured; the run never measured it.

## Brief errors
- None found.
