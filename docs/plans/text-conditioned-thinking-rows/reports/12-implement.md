# Stage 12 implement: iteration-2 results note

Status: done. Both done-when checks pass. Not committed.

## Changed paths
- `docs/research/web/2026-11-12-per-feature-role-tokens-results.md` (new). Frontmatter and sections copied from the iteration-1 note: Research Question, Summary, Detailed Findings (setup, stage 7, stages 9/13, reliability, stage 11, training-run variance, interpretation, scope), Confidence Notes, Open Questions.

## Content
- One `Verdict:` line naming `role-primary` supported (−0.0041/−0.0042 at n = 8–32), `role10k-primary` supported (−0.0098 at n = 8, −0.0057 at 32), and `followon-primary` with the brief's wording ("harmful (D-08, bootstrap seed 0) — marginal: holds in 69/100 seeds, paired t p = 0.051, +0.0008 nats").
- Numbers are copied from the report commands run 2026-11-12, all exit 0: `roleinfo`, `role-training`, `role-training --results results/13-role-training-10k`, `role-primary`, `role-curve`, `role-reliability`, `role10k-primary`, `followon-training`, `followon-primary`, `primary`. Review-sourced figures: slot-change rates 0.79/0.67 (09-review), 69/100, p = 0.051, 86%, +1.2e-4 (11-review), MiniLM cosine 0.9886 (10-implement). The Confidence Notes name these sources.
- The 10k widening split is quoted from `role10k-primary`'s 10k-minus-4k table: at n = 8, story −0.0027 and control +0.0029. The 10k control is 0.6283 against native 0.6292.
- The note includes both run-variance caveats: the 4000-step result holds within one adapter (omitted corruption), and there is no such check at 10k (B-16). B-19 is named as the fix.
- Interpretation is kept separate from results and marked [INFERENCE] where needed. The precedence rule is mentioned and shown not to decide either verdict.
- Links: the iteration-1 note and `../../plans/.../index.md`. There are no anchors, and the owner's 2026-11-04 note is not linked.

## Checks
- `grep -l '^Verdict:' docs/research/web/*-per-feature-role-tokens-results.md` -> 1 file (the new note).
- `./scripts/lint_related_works.sh` -> "Related-work lint passed", exit 0.

## What in this brief was wrong
- Nothing material. One note: `role-curve` shows the 4000-step story − control CI includes 0 at n = 512 (upper bound +0.0000). So the note says "CI below 0 at n = 8…256" for 4000 steps and "8…1024" for 10k.
- `report roleinfo`'s D-13 line reads "includes 0 ...: no", so the stop rule did not trigger. The brief did not give this value; the note quotes it.
