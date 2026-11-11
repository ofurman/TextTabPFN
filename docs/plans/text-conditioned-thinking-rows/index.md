# Plan: Text-conditioned thinking rows for TabPFN-2.5

Date: 2026-11-11 (iteration 2; iteration 1 began 2026-11-10) · Branch: thinking-rows (local) · Source: owner message 2026-11-11, "encode the story so each feature's role is visible to the adapter, for example per-feature role tokens (B-12). After that, try a stronger encoder (B-09) or a trained per-column arm (B-01)"
Status lives in state.json. Executed by goal-execution-loop.

## Goal

Iteration 1 answered the original question; see [reports/final.md](reports/final.md) and the [results note](../../research/web/2026-11-11-text-conditioned-thinking-rows-results.md). The verdict was "harmful", at about 1e-3 nats. The story adapter reacted to whether a story was present, not to what it said. The pooled MiniLM story embeddings were nearly identical across scenes.

Iteration 2 tests whether per-feature role tokens let the frozen TabPFN-2.5 use the story:
- each feature's role from the story becomes its own token, in column order;
- the tokens feed the same shared thinking-row channel;
- the comparison is against a matched schema-only control.

It then runs one follow-on arm, chosen by that result under D-14: a stronger pooled encoder (B-09) or per-column thinking rows (B-01). A null or negative answer counts as a complete result.

## Definition of Done

1. The log-loss gain of story-guided preprocessing over native TabPFN-2.5 is reported at n = 8, 16, 32 on held-out scenes — check: `uv run python -m texttabpfn.report roleinfo` — today: fails, exit 2, invalid choice 'roleinfo'
2. The role-token adapter at initialization reproduces native raw TabPFN-2.5 logits — check: `uv run pytest tests_exp/test_role_tokens.py` — today: fails, exit 4, file not found
3. The paired role-token-story minus matched-control log-loss difference has a 95% scene-bootstrap CI at n = 8, 16, 32 with its D-08 verdict — check: `uv run python -m texttabpfn.report role-primary` — today: fails, exit 2, invalid choice
4. The same paired difference is reported for every n in 8 … 1024 — check: `uv run python -m texttabpfn.report role-curve` — today: fails, exit 2, invalid choice
5. The log-loss change of each corrupted role assignment relative to the correct one is reported at n = 8, 16, 32 — check: `uv run python -m texttabpfn.report role-reliability` — today: fails, exit 2, invalid choice
6. The follow-on arm's paired story-minus-control difference has a 95% scene-bootstrap CI at n = 8, 16, 32 with its D-08 verdict — check: `uv run python -m texttabpfn.report followon-primary` — today: fails, exit 2, invalid choice
7. A results note states the iteration-2 verdicts — check: `grep -l '^Verdict:' docs/research/web/*-per-feature-role-tokens-results.md` — today: fails, no such file

Iteration 1's DoD (items 1–7 of 2026-11-10) passed. The evidence is in [reports/final.md](reports/final.md).

## Constraints

- Allowed:
  - Add code under `src/texttabpfn/`, tests under `tests_exp/`, and results under `results/`. Commit small CSV/JSON files; weights stay gitignored.
  - Download `BAAI/bge-base-en-v1.5` into the local Hugging Face cache if D-14 selects B-09.
  - Commit on the local branch `thinking-rows`.
  - Add one results note under `docs/research/web/`.
- Protected:
  - `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md` and the existing research notes, including the iteration-1 results note.
  - Iteration-1 results under `results/01-capacity` … `results/05-reliability`, and the code paths that produce them. Their report commands must still exit 0.
  - `main`, which is changed only by an owner merge.
- Hard stops:
  - Pushing to any remote.
  - Remote or paid compute.
  - Distributing any weights or outputs (license).
  - Changing a DoD item.
  - Continuing past stage 7 when the D-13 stop rule fires.

## Stages

| id | title | box (h) | review | depends |
|----|-------|---------|--------|---------|
| 1 | [Channel capacity of the thinking rows](stages/01-channel-capacity-of-the-thinking-rows.md) (iteration 1) | 6 | yes | - |
| 2 | [Stories and parity-checked adapter](stages/02-stories-and-parity-checked-adapter.md) (iteration 1) | 4 | no | 1 |
| 3 | [Episodic training of story and schema-only arms](stages/03-episodic-training-of-story-and-schema-on.md) (iteration 1) | 6 | no | 2 |
| 4 | [Primary evaluation over support sizes](stages/04-primary-evaluation-over-support-sizes.md) (iteration 1) | 3 | yes | 3 |
| 5 | [Context reliability evaluation](stages/05-context-reliability-evaluation.md) (iteration 1) | 2 | no | 3 |
| 6 | [Results note and verdict](stages/06-results-note-and-verdict.md) (iteration 1) | 1 | no | 4,5 |
| 7 | [Value of story roles to native TabPFN](stages/07-value-of-story-roles-to-native-tabpfn.md) | 2 | yes | - |
| 8 | [Role-token adapter and matched control training](stages/08-role-token-adapter-and-matched-control.md) | 6 | no | 7 |
| 9 | [Role-token held-out evaluation](stages/09-role-token-held-out-evaluation.md) | 2 | yes | 8 |
| 10 | [Follow-on arm training](stages/10-follow-on-arm-training.md) | 6 | no | 9 |
| 11 | [Follow-on held-out evaluation](stages/11-follow-on-held-out-evaluation.md) | 2 | yes | 10 |
| 12 | [Iteration-2 results note](stages/12-iteration-2-results-note.md) | 1 | no | 9,11 |

All stages use the one local MPS device. They run one after another.

## Iteration

Iteration 2 delivers the deciding number first. Stage 7 measures how much perfect knowledge of the story's roles is worth to native TabPFN when used through simple label-free preprocessing. Stages 8–9 test whether the shared channel can carry those roles. Stages 10–11 run the follow-on arm that D-14 selects, and stage 12 reports.
Deferred on purpose: extracting per-feature facts from free-form stories with an LLM (B-14), the follow-on arm D-14 does not select, and everything else in backlog.md.
