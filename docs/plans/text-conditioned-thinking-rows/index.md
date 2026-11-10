# Plan: Text-conditioned thinking rows for TabPFN-2.5

Date: 2026-11-10 · Branch: thinking-rows (local, from main) · Source: owner message 2026-11-10, "test the idea of text-conditioned thinking rows θᵢ ← θᵢ + Wᵢ·e(t); drop K1, K3"
Status lives in state.json. Executed by goal-execution-loop.

## Goal

We get a measured answer to one hypothesis. Conditioning TabPFN-2.5's 64 thinking rows on one shared task story, θᵢ ← θᵢ + Wᵢ·e(t), is the change being tested. TabPFN-2.5 and the text encoder stay frozen, and Wᵢ starts at zero and is trained across synthetic scenes. The hypothesis is that this lowers held-out log loss at small support sizes compared with the same adapter trained on the schema alone. The answer is reported with confidence intervals across support sizes and under unreliable stories. A negative or null answer counts as a complete result. Design background: [2026-11-04 design note](../../research/web/2026-11-04-text-tabular-attention.md).

## Definition of Done

1. The log-loss gain of per-scene-optimized shared thinking rows over native TabPFN-2.5 is reported at n = 8, 16, 32 — check: `uv run python -m texttabpfn.report capacity` — today: fails, no module `texttabpfn`
2. The adapter with W = 0 reproduces native raw TabPFN-2.5 logits within 1e-5 — check: `uv run pytest tests_exp/test_parity.py` — today: fails, no pytest or `tests_exp/`
3. Changing query labels leaves adapter logits unchanged — check: `uv run pytest tests_exp/test_label_isolation.py` — today: fails, no pytest or `tests_exp/`
4. The paired story-minus-schema-only log-loss difference has a 95% scene-bootstrap CI at n = 8, 16, 32 on held-out families — check: `uv run python -m texttabpfn.report primary` — today: fails, no module `texttabpfn`
5. The same paired difference is reported for every n in 8 … 1024 — check: `uv run python -m texttabpfn.report curve` — today: fails, no module `texttabpfn`
6. The log-loss change of each corrupted story type relative to the correct story is reported at n = 8, 16, 32 — check: `uv run python -m texttabpfn.report reliability` — today: fails, no module `texttabpfn`
7. A results note states the verdict under the D-08 rule — check: `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` — today: fails, no such file

## Constraints

- Allowed:
  - Create `pyproject.toml`, `uv.lock`, `src/texttabpfn/`, `tests_exp/` and `results/` in this repo. Commit small CSV/JSON results; gitignore checkpoints and caches.
  - Download `tabpfn-v2.5-classifier-v2.5_default.ckpt` (D-05) and all-MiniLM-L6-v2 into the local Hugging Face cache.
  - Commit on the local branch `thinking-rows`.
  - Add one results note under `docs/research/web/`.
- Protected: `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, and the existing research notes; `main` is changed only by a merge the owner makes.
- Hard stops:
  - Pushing to any remote.
  - Remote or paid compute.
  - Distributing TabPFN-2.5 weights, adapter weights or outputs (license).
  - Changing a DoD item.
  - Continuing past stage 1 when the D-08 stop rule fires.

## Stages

| id | title | box (h) | review | depends |
|----|-------|---------|--------|---------|
| 1 | [Channel capacity of the thinking rows](stages/01-channel-capacity-of-the-thinking-rows.md) | 6 | yes | - |
| 2 | [Stories and parity-checked adapter](stages/02-stories-and-parity-checked-adapter.md) | 4 | no | 1 |
| 3 | [Episodic training of story and schema-only arms](stages/03-episodic-training-of-story-and-schema-on.md) | 6 | no | 2 |
| 4 | [Primary evaluation over support sizes](stages/04-primary-evaluation-over-support-sizes.md) | 3 | yes | 3 |
| 5 | [Context reliability evaluation](stages/05-context-reliability-evaluation.md) | 2 | no | 3 |
| 6 | [Results note and verdict](stages/06-results-note-and-verdict.md) | 1 | no | 4,5 |

Stages 4 and 5 share the local MPS device. They run one after the other.

## Iteration

Iteration 1 delivers the deciding numbers first. Stage 1 measures the channel's upper limit with no text involved. Stages 2–6 then train the story and schema-only arms and report the H1, H3 and H4 results.
Deferred on purpose: per-column or cross-attention conditioning, the ensembled wrapper path, H2 label savings, LLM-paraphrased stories and CausalDS. Each waits for this iteration's answer (see backlog.md).
