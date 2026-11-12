# 11 review (omp/reviewer, high), commits b583d2c and 05de41b

Verdict: no correctness bugs. `report followon-primary` delivers DoD 6 honestly, and the D-08 label "harmful" stays as-is, but it is a knife-edge call.

## Findings and triage
1. **P2. "Harmful" holds for only 69 of 100 bootstrap seeds.**
   - Only n = 8 can trigger it: the n = 16 and n = 32 lower bounds are below 0 for all 100 seeds.
   - At n = 8 the share of seeds with a lower bound above 0 has a normal-approximation 95% CI of 0.60–0.78. With seed 0 the lower bound is +4.2e-6; with 10,000 resamples it is +1.65e-5.
   - A paired t-test gives p = 0.051; the Wilcoxon test gives p = 0.158.
   - The story arm beats the control on 48% of scenes.
   - Mirror image: at n = 128 the upper bound is −1.7e-5, just below 0.
   - → Fix in the note (stage 12 brief): "harmful (D-08, seed 0), marginal: 69/100 seeds, t p = 0.051, +0.0008 nats". A stability line in the report output goes to backlog B-18.
2. **P3. Most of the gap comes from the two separately trained weight sets.** CPU re-score at n = 8:
   - story adapter fed schema text − control: +7.25e-4 [−1.2e-4, +1.6e-3];
   - story adapter, story text − schema text: +1.21e-4 [+4.4e-5, +1.95e-4].

   About 86% of the gap is run-to-run training variance, which the scene bootstrap does not cover. → Fix in the note: report this split. The same caveat applies to the role-token story − control verdicts (orchestrator addition, see the stage 12 brief).

## Checks that passed
- **Encoder.** bge modules are [Transformer, Pooling(cls), Normalize]. A manual CLS embedding plus normalize equals `encode` exactly. The cache is keyed by text. Maximum length is 159–160 tokens.
- **Texts per arm.** Training uses story with the D-09 mix vs schema; dev uses story vs omitted; eval uses story vs schema.
- **Matched arms.**
  - Step-0 weights are bit-identical across arms.
  - Same episodes, lr schedule and budget; 32,000 episodes each.
  - Selection is dev-only: both arms at step 3000, and the selected file equals the step-3000 file.
- **No leakage, same rows.** No held-out data is used. Rows are identical to stage 4 (native rescore drift 0). The CPU re-score matches `scores.csv` within 6.3e-7.
- **W = 0.5 test is legitimate.** The trained adapters shift the rows 15–25× more than W = 0.5 does, and the 1e-3 threshold is unchanged.
- **Tests.** Non-MPS tests in `test_followon*` pass (19).
