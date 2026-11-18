# Execution protocol: text-conditioned-thinking-rows

Rendered by `init_run.sh` on 2026-11-10. This file is the only protocol for this run; `PROTOCOL.previous.md`, if present, is history. `index.md` says what to deliver (goal, Definition of Done, constraints); this file says how. Settings: reviews `flagged`, tripwire 3 h without a product commit, harness `omp`.

## Roles

| Role | Tier | Assigned | How it runs |
|---|---|---|---|
| Orchestrator | coordinator | `omp/claude-opus-5-5` | this session: judges, dispatches, commits, keeps the record |
| Implementer | strong | `omp/task` | effort `high`, native agent definition |
| Reviewer | strong | `omp/reviewer` | effort `high`, native agent definition, read-only |
| Helper | light | `omp/sonic` | effort `medium`, native agent definition |

The hierarchy is flat: no subagent spawns another, and none runs on the coordinator tier. Subagents never commit or edit `state.json`, `journal.md`, `index.md`, `stages/`, or `backlog.md`. Each stage in `state.json` records the implementer and reviewer that actually ran, as `agent/model, effort`.

## Stage loop

1. **Pick** the next `PENDING` stage whose dependencies are `DONE`. Independent stages may run in parallel, one implementer each on disjoint files, unless they share a named exclusive resource. Set `IN_PROGRESS`, `startedAt`, `implementer`.
2. **Brief.** Write `briefs/NN-<slug>.md`, at most 150 lines, from the stage card: goal, done-when checks with exact commands, facts with sources, "Not in this stage", box and stop-by time, report path.
3. **Build.** The implementer builds, runs the done-when checks, and writes `reports/NN-implement.md`, at most 60 lines.
4. **Accept.** Read the diff and the report. Re-run the done-when checks, and any DoD check the stage touches, once and read-only. Do not re-derive results, rehash files, or write counter-proof harnesses; a suspected defect becomes a fix or diagnosis assignment.
5. **Commit** by explicit path as parts go green. Record the hashes in the stage's `commit`.
6. **Review.** Stages marked `review: yes` get one post-review; the whole run gets one final review. A review is one `goal-reviewer` pass in single-reviewer mode, saved to `reports/NN-review.md` (the final one to `reports/final-review.md`). Triage each in one line: fix, backlog, or dismiss (a finding you cannot reproduce is dismissed). One fix round, re-run the checks, ship. No re-review. A spent review never comes back.
7. **Close.** `stage_close_check.sh <run dir> NN` must pass; if it fails, prune and re-run it. Then set `DONE` with a one-line `note`.
8. **Re-plan.** A 3-line journal entry: keep, cut, add. You may rewrite the next `PENDING` cards; the Definition of Done changes only with the owner.

Behaviour already correct at HEAD needs a passing test, not a manufactured red.

## When a check fails

- **Product failure:** at most 2 fix assignments after the build report. Then defer the item to `backlog.md` with its failing command and result, and move on. A DoD item that stays red is reported as open with a `STATUS:` line; work on independent stages continues.
- **Wrong check:** one a correct implementation would fail, or one that tests a mechanism instead of an outcome. If it is not a DoD item, amend it with one journal line (old, new, why). A wrong DoD check goes to the owner.
- **Infra failure:** tool crash, timeout, EPERM, auth, network or DNS, a provider or content flag, a helper timeout. Retry once with a fresh box, then reroute: another harness, inline, or a narrower scope. It is never an outcome and never makes a stage `BLOCKED`.
- A value that could not be measured is reported as `NOT MEASURED (reason)`. That is a value label, not a status, and it blocks only a DoD item.

## Tripwires

A stage past 2x its box, no product commit for 3 h, or a check that looks wrong: cut the stage to its done-when minimum, defer the rest to `backlog.md`, write `STATUS: <what tripped>; <what was cut>; <what runs next>` in the journal, send a harness notification if one exists, and keep going on independent work.

Stop and wait for the owner only when a DoD item cannot be reached, or when one of these is needed beyond the allowed actions in `index.md`:

1. deleting or overwriting data;
2. pushing to a shared branch, or force-pushing anywhere;
3. deploying, releasing, or tagging;
4. sending anything to a person or an external service;
5. spending money.

Any other reversible action proceeds with a journal line. Owner questions never stall independent stages.

## Evidence and caps

- Evidence is the command and its result on one journal line: `uv run pytest -q tests/x` -> 14 passed.
- One report of at most 60 lines per assignment in `reports/`. One file of at most 4 KB per failed check attempt in `validation/`. Nothing for passing runs: no raw dumps, per-attempt directories, seals, or hashes.
- Journal entries are at most 5 lines. Product outputs live outside the run artifacts.
- `stage_close_check.sh` enforces: run artifacts at most 2 MB, at most 8 files per stage, the line and byte caps above.

## Statuses and integrity

- Stage status: `PENDING` | `IN_PROGRESS` | `DONE` | `DEFERRED` (descoped to backlog) | `BLOCKED` (a DoD item is unreachable, or a hard stop needs the owner).
- Never pad a count or hardcode a PASS. Never weaken or delete a test to get green. Never `--no-verify`, never `git add .`.
- No backward-compatibility shims unless `index.md` asks for them. No credentials under the run directory.
- The run ends when every stage is `DONE`, `DEFERRED`, or `BLOCKED`, after the final review the settings call for, with `reports/final.md` (at most 60 lines) and one journal entry.
