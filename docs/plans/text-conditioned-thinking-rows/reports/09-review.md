# 09 review (omp/reviewer, high): commits eaee451, 26aacb7, 5b8cd0d

Verdict: correct. DoD items 3–5 are delivered. Nothing could fake the "supported" story-vs-control gain. No P0–P2 findings. Confidence 0.85, CPU only.

## Checks
- **Reports.** All three reproduce.
  - `role-primary`: −0.0042 / −0.0042 / −0.0041 at n = 8 / 16 / 32, every CI below 0, verdict supported. Each family's upper bound is below 0 at all three n.
  - `role-curve`: reference drift is 0.
  - `role-reliability`: every corruption CI is above 0.
- **Sentences** (`role_tokens.py:50-75`). There are four fixed role sentences with no feature name, base rate, mechanism, sign or coefficient. They carry the same facts as the iteration-1 story. Both arms see p identically: p filled slots, the rest zero.
- **Control.**
  - Same `make_role_episode` arrays and step loop (`role_training.py:209-221`).
  - One lr for both arms; 32,000 episodes each.
  - Step-0 weights are equal: P is the same and W = 0.
  - Same P seed, slot count and slot width.
- **Selection.**
  - Only dev `validation.csv` is used (families 0–3, seeds 1,000,000–1,000,063).
  - The selected files equal steps 4000 (story) and 3750 (control).
  - The control's 3750 vs 4000 gap is 0.00024. Using step 4000 would make the control slightly worse, so the different steps do not favour the story arm.
- **No held-out data.**
  - `make_episode` draws only training families and the training seed range.
  - `donor_scene` is never used in training.
  - The selected weights were written before the evaluation started.
- **Rows.**
  - All arms and corruptions score the same `episode` tensor.
  - `scores.csv` has 6600 rows, as expected.
  - Stage-4 and stage-7 references match exactly.
- **Bootstrap and verdict.** These follow D-08 (`report.py:38-46, 284-291`).
- **Irrelevant padding.**
  - 78 of 200 scenes are padded, averaging 28% of their slots.
  - Share of slots changed: omitted 1.00, irrelevant 0.79 (min 0.30), wrong 0.67 (min 0.20).
  - So the reliability figures measure partial corruption. The padding does not affect story vs control.

## Findings and triage
1. **P3: the D-15 timing claim is inaccurate.** At 19:20Z, about 80 of the 200 stage-9 scene results already existed in `scores.csv`. No one read them; the change was the owner's request. → Fixed: journal correction line.
2. **P3: `09-implement.md` said role story is worse than native "from 256 up".** At 256 the CI includes 0; it is worse only at 512 and 1024. → Fixed in the report.
3. **P3: "irrelevant and wrong cost more than omitted" holds only for point estimates.** The CIs overlap at n = 16. → Carried to the stage 12 brief: phrase it carefully and state the slot-change rates.
