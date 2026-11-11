# Final report: text-conditioned thinking rows (iteration 1)

The run executed every stage. The hypothesis is not supported. Under D-08 the verdict is **harmful**, but the effect is about 1e-3 nats. Results note: `docs/research/web/2026-11-11-text-conditioned-thinking-rows-results.md`.

## Definition of Done
1. **Capacity.** `uv run python -m texttabpfn.report capacity` → exit 0. Shared-row gain at n = 8 / 16 / 32: 0.162 / 0.124 / 0.075, and every CI excludes 0. Per-column gain: 0.200 / 0.154 / 0.100. Commits bdeb972 and c9f99c5.
2. **Parity.** `uv run pytest tests_exp/test_parity.py` → 15 passed. With W = 0 the difference from native logits is exactly 0 on both CPU and MPS. Commit 8cbb249.
3. **Label isolation.** `uv run pytest tests_exp/test_label_isolation.py` → 6 passed. Commit 8cbb249.
4. **Primary.** `uv run python -m texttabpfn.report primary` → exit 0. Story − schema-only at n = 8 / 16 / 32: −0.0008 / +0.0001 / +0.0012, with n = 32 at [+0.0005, +0.0019]. Verdict: harmful. Commit b97088c.
5. **Curve.** `uv run python -m texttabpfn.report curve` → exit 0 for n = 8…1024. Both adapters beat native by up to 0.012 at n ≤ 64. Above n ≥ 256 the story arm is about 0.0006 worse than native. Commit b97088c.
6. **Reliability.** `uv run python -m texttabpfn.report reliability` → exit 0. Omitting the story costs +0.0015 at n = 8. The irrelevant and wrong stories are indistinguishable from the correct story (CI includes 0). Commit 6b10c09.
7. **Verdict line.** `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` → 1 file. `./scripts/lint_related_works.sh` passes. Commit d6cdcef, plus the P3 fix from the final review.

## Main reading
- The channel has a large ceiling: 0.16 nats at n = 8 with per-scene optimization.
- The trained adapters reach about 0.004 nats on held-out families (about 0.014 on dev), and the schema-only arm reaches the same.
- The story adapter reacts to whether a story is present, not to what it says. This is an inference, labelled as such in the note.
- One plausible cause: MiniLM mean-pooled stories are nearly identical (same-p cosine 0.989). This is a diagnostic, not an ablation.

## Reviews
- **Stage 1.** P1: family 5 was nearly unlearnable because of heavy-tailed squared terms. Fixed in c9f99c5; families 0–3 stayed bit-identical. P3: report strictness, fixed.
- **Stage 4.** Two P3 findings. One is noted in the results note; the other is B-13.
- **Final review.** One P3 (provenance of the encoder diagnostic), fixed in the results note.

## Deferred and blocked
- Nothing is blocked or deferred.
- The backlog holds B-01…B-13. The most relevant follow-ups are B-12 (structured story encoding), B-09 (stronger encoder) and B-01 (per-column arm).

## Infra and models
- The stage 1 implementer died on an API outage after its run completed. The orchestrator wrote `01-implement.md` from the committed results; no rerun was needed.
- Orchestrator: omp/claude-opus-5-5. Implementers: omp/task, high. Reviewers: omp/reviewer, high. The helper was not used.
- Model weights (TabPFN checkpoint, adapter `*.pt`) stay local and gitignored. Nothing was pushed.
