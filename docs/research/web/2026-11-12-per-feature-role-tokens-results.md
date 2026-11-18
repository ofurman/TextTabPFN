---
date: "2026-11-12T09:00:00+01:00"
researcher: "Oleksii Furman"
topic: "Results of per-feature role tokens and a bge-base pooled story for frozen TabPFN-2.5 thinking rows"
tags: ["tabpfn", "thinking-rows", "task-context", "text-conditioning", "role-tokens", "experiment-results"]
sources: ["repository-experiment", "texttabpfn-report"]
status: complete
last_updated: "2026-11-12"
last_updated_note: "Iteration 2 results for stages 7, 9, 13 and 11 of the thinking-rows plan"
---

# Research: Per-feature role tokens for thinking rows — results

## Research Question

Iteration 1 found that one pooled MiniLM story embedding, fed to TabPFN-2.5's 64 thinking rows, did not beat a schema-only control ([iteration-1 results](2026-11-11-text-conditioned-thinking-rows-results.md)). Iteration 2 asks: do the same shared thinking rows use the story when its per-feature roles are exposed as tokens, and does a stronger pooled encoder (bge-base-en-v1.5) recover a benefit from the plain story text? Plan: [text-conditioned thinking rows, iteration 2](../../plans/text-conditioned-thinking-rows/index.md).

## Summary

Verdict: role-primary supported — role-token story beats its matched control by 0.0041–0.0042 nats at n = 8–32 (4000 steps); role10k-primary supported — the gap widens to 0.0098 nats at n = 8 and 0.0057 at n = 32 (10k steps); followon-primary harmful (D-08, bootstrap seed 0) — marginal: holds in 69/100 seeds, paired t p = 0.051, +0.0008 nats.

Reported results:

- Role knowledge is worth 0.016–0.020 nats to native TabPFN-2.5 at n = 8–32 when applied as story-guided preprocessing (stage 7); this passes the D-13 stop rule.
- Per-feature role tokens through the shared thinking rows beat the matched control with CI below 0 at every n = 8…256 (4000 steps) and at every n = 8…1024 (10k steps); at n = 8, 16, 32 this holds in both held-out families.
- Corrupting the roles of the 4000-step story adapter costs +0.0035 to +0.0081 nats, all CIs above 0.
- A bge-base pooled story arm does not beat its schema control; the n = 8 difference is +0.0008 nats with a lower bound at 0.
- Story-guided preprocessing on native TabPFN still beats the best learned role-token adapter at n ≤ 32.

## Detailed Findings

### Setup

- Role tokens (D-11): each observed feature x_j gets one token in slot j, the frozen MiniLM embedding of a sentence stating only x_j's role (influences y, is influenced by y, measures hidden quantity k, or unrelated). A shared learned reduction feeds θᵢ + Wᵢ·s, rows identical across columns, Wᵢ zero-initialized. Control: same architecture with the neutral schema sentence in every slot. Roles come from `Scene.describe()`, an exact parse of the template stories; this is not free-text understanding.
- Preprocessing baseline (D-12): drop features the story calls unrelated, keep Markov-blanket features, collapse each measurement group to its first principal component. Uses only story facts and unlabeled feature values; no training.
- Stop rule (D-13): stages 8–12 would wait for the owner if the preprocessing − native CI included 0 at all of n = 8, 16, 32.
- D-15: the owner asked for a 10k-step retrain of both role-token arms (stage 13); its verdict, not the 4000-step one, decided D-14 (which selected the bge arm, B-09).
- D-16 (bge arm): θᵢ + Wᵢ·P·e(t), e(t) the frozen bge-base-en-v1.5 CLS embedding (768-d), P (384×768) shared and learned; 5,013,504 trainable parameters per arm. Story arm input schema + story, control schema only.
- Evaluation (D-08), unchanged from iteration 1: 200 held-out scenes (families 4–5), 5 nested support seeds, 512 query rows, paired per-scene difference, 95% scene-bootstrap CI (1000 resamples). Raw module path, no ensembling.

Training (`report role-training`, `report role-training --results results/13-role-training-10k`, `report followon-training`); all Adam lr 0.001, warm-up 100, cosine to 0.1×, 8 episodes per step.

| run | steps | train wall | story selected step / dev val loss | control selected step / dev val loss |
|---|---|---|---|---|
| role tokens (stage 8) | 4000 | 2.92 h | 4000 / 0.4335 | 3750 / 0.4381 |
| role tokens (stage 13) | 10000 | 7.50 h | 9500 / 0.4334 | 6500 / 0.4364 |
| bge pooled (stage 10) | 4000 | 3.01 h | 3000 / 0.4377 | 3000 / 0.4377 |

### Stage 7: value of roles to native TabPFN through preprocessing

`report roleinfo` (preproc fits PCA on support + query feature values) and the support-only secondary of `report role-curve` (preproc-s):

| n | native | preproc | preproc − native [95% CI] | preproc-s − native [95% CI] | preproc-s − preproc [95% CI] |
|---|---|---|---|---|---|
| 8 | 0.6292 | 0.6128 | −0.0165 [−0.0200, −0.0130] | −0.0153 [−0.0190, −0.0115] | +0.0012 [−0.0001, +0.0026] |
| 16 | 0.5879 | 0.5675 | −0.0204 [−0.0233, −0.0171] | −0.0197 [−0.0227, −0.0163] | +0.0007 [−0.0004, +0.0017] |
| 32 | 0.5353 | 0.5197 | −0.0156 [−0.0183, −0.0129] | −0.0151 [−0.0177, −0.0125] | +0.0006 [−0.0001, +0.0011] |

`roleinfo` prints "D-13: 95% CI includes 0 at all of n = 8, 16, 32: no", so the run continued. Part of the gain uses unlabeled query features (the arm is "roles + unlabeled query features"); the support-only fit keeps most of it. Above n = 128 preprocessing hurts (+0.0058 at 256, +0.0132 at 1024).

### Stage 9 and stage 13: role-token story − control

`report role-primary` (4000 steps) and `report role10k-primary` (10k steps; story step 9500, control step 6500):

| n | 4k control | 4k story | 4k story − control [95% CI] | 10k control | 10k story | 10k story − control [95% CI] |
|---|---|---|---|---|---|---|
| 8 | 0.6254 | 0.6212 | −0.0042 [−0.0064, −0.0020] | 0.6283 | 0.6185 | −0.0098 [−0.0121, −0.0075] |
| 16 | 0.5764 | 0.5721 | −0.0042 [−0.0057, −0.0027] | 0.5773 | 0.5693 | −0.0080 [−0.0097, −0.0064] |
| 32 | 0.5270 | 0.5229 | −0.0041 [−0.0052, −0.0030] | 0.5271 | 0.5214 | −0.0057 [−0.0071, −0.0044] |

Both print `D-08 verdict: supported`; per-family CIs are below 0 in both families at all three n. The "harmful before supported" order in `report.d08_verdict` does not matter here: no lower bound is above 0.

Where the 10k widening comes from (`report role10k-primary`, same rows, paired):

| n | story 10k − story 4k [95% CI] | control 10k − control 4k [95% CI] |
|---|---|---|
| 8 | −0.0027 [−0.0042, −0.0012] | +0.0029 [+0.0013, +0.0044] |
| 16 | −0.0029 [−0.0037, −0.0019] | +0.0009 [−0.0002, +0.0021] |
| 32 | −0.0015 [−0.0024, −0.0006] | +0.0001 [−0.0007, +0.0009] |

At n = 8 about half the widening is the story arm's gain and half the control's loss; the 10k control (0.6283) sits near native (0.6292). The control's selected step (6500) is earlier than the story's (9500).

Against references (mean query log loss, `report role-curve`, `report followon-primary`):

| n | native | preproc | it1 MiniLM story | 4k role story | 10k role story | 4k role story − preproc [95% CI] |
|---|---|---|---|---|---|---|
| 8 | 0.6292 | 0.6128 | 0.6248 | 0.6212 | 0.6185 | +0.0084 [+0.0042, +0.0123] |
| 16 | 0.5879 | 0.5675 | 0.5763 | 0.5721 | 0.5693 | +0.0046 [+0.0012, +0.0080] |
| 32 | 0.5353 | 0.5197 | 0.5278 | 0.5229 | 0.5214 | +0.0033 [+0.0003, +0.0061] |

4k role story − native: −0.0080 / −0.0157 / −0.0124 at n = 8 / 16 / 32; 4k role story − iteration-1 story: −0.0036 / −0.0042 / −0.0049, all CIs below 0. Iteration-1 reference (`report primary`): story − schema −0.0008 / +0.0001 / +0.0012, verdict harmful.

### Reliability (stage 9, 4000-step story adapter)

`report role-reliability`, corrupted − correct roles:

| n | correct | omitted [95% CI] | irrelevant [95% CI] | wrong [95% CI] |
|---|---|---|---|---|
| 8 | 0.6212 | +0.00366 [+0.00199, +0.00541] | +0.00784 [+0.00465, +0.01118] | +0.00807 [+0.00574, +0.01038] |
| 16 | 0.5721 | +0.00460 [+0.00341, +0.00587] | +0.00595 [+0.00412, +0.00784] | +0.00582 [+0.00399, +0.00781] |
| 32 | 0.5229 | +0.00348 [+0.00271, +0.00440] | +0.00475 [+0.00371, +0.00585] | +0.00462 [+0.00339, +0.00593] |

- "Irrelevant and wrong roles cost more than omitted ones" holds for point estimates only. The CIs overlap at every n, except wrong vs omitted at n = 8.
- Corruptions change only part of the slots (stage 9 review): omitted 1.00, irrelevant 0.79 on average (min 0.30), wrong 0.67 (min 0.20). The figures measure partial corruption.
- Omitted − story (+0.0037 / +0.0046 / +0.0035) is the same adapter fed neutral tokens, so it avoids training-run variance; it matches the 4000-step story − control gap (−0.0042). The 4000-step role-token result therefore holds within one adapter.

### Stage 11: bge pooled story − control

`report followon-primary` (both arms step 3000):

| n | bge control | bge story | bge story − control [95% CI] | bge story − native [95% CI] | bge story − 10k role story [95% CI] |
|---|---|---|---|---|---|
| 8 | 0.6248 | 0.6256 | +0.0008 [+0.0000, +0.0017] | −0.0036 [−0.0065, −0.0009] | +0.0072 [+0.0048, +0.0095] |
| 16 | 0.5759 | 0.5763 | +0.0004 [−0.0001, +0.0010] | −0.0115 [−0.0143, −0.0090] | +0.0071 [+0.0055, +0.0085] |
| 32 | 0.5265 | 0.5266 | +0.0001 [−0.0004, +0.0005] | −0.0087 [−0.0105, −0.0072] | +0.0052 [+0.0040, +0.0063] |

- Printed verdict: `D-08 verdict: harmful`. It rests on an n = 8 lower bound of +4.2e-6 at bootstrap seed 0, which is above 0 in 69/100 seeds (95% CI 60–78%); a paired t-test gives p = 0.051 (stage 11 review F1). Here the "harmful before supported" order does not decide anything, since no upper bound is below 0.
- On dev the two arms tied: 0.4377 = 0.4377.
- Encoder diagnostic (`report followon-training`, stage 10 report): over 40 dev stories, bge same-p cosine 0.9936 (mean 0.9758, residual norm 0.1516) vs MiniLM 0.9886 (0.9641, 0.1849). bge stories are more alike than MiniLM's. This is a plausible mechanism for the null, not a proven cause.

### Training-run variance

Each arm was trained once. In the bge pair about 86% of the n = 8 gap is the difference between the two trained adapters; the same story adapter fed story vs schema text differs by only +1.2e-4 (stage 11 review F2). The scene bootstrap does not cover training-run variance, and the same caveat applies to the role-token verdicts. For the 4000-step pair the omitted corruption above is within-adapter evidence. No such check exists for the 10k pair (B-16), and about half of its extra widening at n = 8 comes from the control getting worse, which may be run variance. Multiple training seeds (B-19) would fix this.

### Interpretation

- [INFERENCE] Exposing per-feature structure, rather than pooling it, is what made the shared thinking-row channel use the story: the same channel that ignored pooled stories in iteration 1 responds to role tokens, and corrupting them hurts.
- [INFERENCE] A stronger pooled encoder did not help. The bge arm is at best null against its control (D-08 "harmful", marginal), and bge's pooled story embeddings are even more alike than MiniLM's. That similarity is a plausible mechanism, not a proven cause.
- Story-guided preprocessing still beats the learned adapter at small n: 0.6128 vs 0.6185 for the 10k role story at n = 8. At large n it hurts much more than the adapter: preprocessing − native is +0.0058 at n = 256 and +0.0132 at n = 1024, while 4000-step role story − native is +0.0006 to +0.0008 at n = 512–1024.

### What this does and does not show

- Shows: on held-out mechanism families with template stories, exact per-feature roles exposed as tokens improve a frozen TabPFN-2.5 through shared thinking rows over a matched control.
- Does not show: understanding of free text — roles are parsed exactly from templates. Only held-out families 4–5; only the raw module path, no ensembling. H2 (label savings) was not measured. No corruption runs at 10k or for the bge arm. One training run per arm.

## Confidence Notes

- All numbers were quoted on 2026-11-12 from `uv run python -m texttabpfn.report` with `roleinfo`, `role-training`, `role-training --results results/13-role-training-10k`, `role-primary`, `role-curve`, `role-reliability`, `role10k-primary`, `followon-training`, `followon-primary` and `primary`, except: slot-change rates (stage 9 review), the bootstrap-seed share, t-test and +1.2e-4 / 86% figures (stage 11 review), and the MiniLM cosine figures (stage 10 implement report).
- CIs are 95% scene-bootstrap (1000 resamples) over paired per-scene differences; they do not include training-run variance.

## Open Questions

- Can a local LLM extract per-feature roles from free-form or paraphrased stories and feed the role-token interface (B-14)?
- Do per-column rows that bind each column's role tokens directly do better than shared rows (B-01)?
- Do the 10k role-token arms respond to role corruption like the 4000-step arm (B-16), and does the bge arm (B-15)?
- Does a 10k-step bge arm move off the verdict boundary (B-17)?
- Do the results hold inside the ensembled sklearn wrapper (B-02), and how many labels does each arm save (B-03)?
- Do the verdicts survive 2–3 training seeds per arm (B-19)?
