# Final report, iteration 2: per-feature role tokens and the follow-on arm

Iteration 2 executed every stage (7–13). Exposing per-feature story roles as tokens made the frozen TabPFN-2.5 thinking rows use the story. A stronger pooled encoder did not. Results note: `docs/research/web/2026-11-12-per-feature-role-tokens-results.md`.

## Definition of Done (iteration 2)
1. **Story-guided preprocessing vs native.** `report roleinfo` exits 0. Preprocessed − native is −0.0165 / −0.0204 / −0.0156 at n = 8 / 16 / 32. The D-13 stop rule did not fire. Commit 241746f.
2. **Parity of the role-token adapter.** `pytest tests_exp/test_role_tokens.py` gives 16 passed: native logits are reproduced at init on CPU and MPS. Commit eaee451.
3. **Role-token primary result.** `report role-primary` exits 0. Story − control is −0.0042 / −0.0042 / −0.0041, verdict supported. Commit 5b8cd0d.
4. **Role-token curve.** `report role-curve` exits 0 for n = 8…1024. The CI is below 0 from n = 8 to 256 and includes 0 at 512 (upper bound +0.0000) and 1024. Commit 5b8cd0d.
5. **Role-token reliability.** `report role-reliability` exits 0. Omitted, irrelevant and wrong roles change the loss by +0.0035 to +0.0081, with every CI above 0. Commit 5b8cd0d.
6. **Follow-on primary result.** `report followon-primary` exits 0. bge story − control is +0.0008 [+0.0000, +0.0017] at n = 8, +0.0004 at 16 and +0.0001 at 32. The verdict is "harmful" but marginal: it holds in 69 of 100 bootstrap seeds, with paired t p = 0.051. Commit 05de41b.
7. **Verdict line.** The `grep '^Verdict:'` check finds one file, and the lint passes. Commit 3f81e5e, plus the wording fixes from the final review.

## Owner-requested stage 13 (D-15)
- 10k-step role tokens: `report role10k-primary` exits 0. Story − control is −0.0098 / −0.0080 / −0.0057, verdict supported. Commits 81c0182 and a3815c8.
- At n = 8, half of the wider gap comes from the story arm improving (−0.0027) and half from the control getting worse (+0.0029). The 10k control's loss is close to native's.
- Under D-14 and D-15 this verdict selected B-09 (the bge encoder) for stage 10.

## Main reading
- Knowing the roles is worth 0.016–0.020 nats to native TabPFN at small n.
- The learned role-token adapter recovers about half of that at 4000 steps, and more at 10k.
- Story-guided preprocessing still leads at n ≤ 32 but hurts at large n (+0.013 at n = 1024).
- **Pooled stories:** pooled MiniLM (iteration 1) and pooled bge (stage 11) both stay at or near null. The bge stories are even more similar to each other (same-p cosine 0.9936).
- **Caveat:** each arm was trained once, and the scene bootstrap does not include run-to-run variance. For the 4000-step role-token result, the same-adapter "omitted" corruption confirms the effect (+0.0037 / +0.0046 / +0.0035). No such check exists for the 10k pair (B-16, B-19).

## Reviews
- **Stage 7:** correct. Two P3 notes, both carried into later briefs and the results note.
- **Stage 9:** correct. Three P3 findings: the D-15 timing in the journal and the stage-9 report sentence were corrected; the corruption-ranking caveat went into the stage-12 brief and the results note.
- **Stage 11:** two findings. P2 (the knife-edge verdict) and P3 (run variance) are both worded in the results note; B-18 and B-19 were added.
- **Final review:** two P3 wording fixes, both made in the results note.

## Deferred and blocked
- Nothing is blocked. B-01 (per-column rows), the arm not selected by D-14, remains in the backlog.
- Other open items: B-14 (role extraction with an LLM), B-15 and B-16 (missing corruption runs), B-17 (bge at 10k), B-18 (bootstrap-seed stability), B-19 (training seeds).

## Infra and models
- The stage 8 implementer died on a DNS failure before launching training. The orchestrator launched the run detached and wrote `08-implement.md`.
- Later stages launched their runs with nohup before waiting.
- Journal timestamps 04:10Z–04:45Z and 19:55Z–20:10Z were estimates; correction entries were added.
- Models: orchestrator omp/claude-opus-5-5, implementers omp/task (high), reviewers omp/reviewer (high).
- Weights stay local and are not in git. Nothing was pushed.
