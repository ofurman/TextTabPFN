---
date: "2026-11-11T04:30:00+01:00"
researcher: "Oleksii Furman"
topic: "Results of text-conditioned thinking rows for frozen TabPFN-2.5"
tags: ["tabpfn", "thinking-rows", "task-context", "text-conditioning", "experiment-results"]
sources: ["repository-experiment", "texttabpfn-report"]
status: complete
last_updated: "2026-11-11"
last_updated_note: "Iteration 1 results for stages 1, 3, 4 and 5 of the thinking-rows plan"
---

# Research: Text-conditioned thinking rows — results

## Research Question

Does conditioning TabPFN-2.5's 64 thinking rows on one shared task story, θᵢ ← θᵢ + Wᵢ·e(t), lower held-out query log loss at small support sizes compared with the same adapter trained on the schema alone? Plan: [text-conditioned thinking rows](../../plans/text-conditioned-thinking-rows/index.md). Design background: [2026-11-04 design note](2026-11-04-text-tabular-attention.md).

## Summary

Verdict: harmful — but only by about 1e-3 nats (story − schema-only +0.0012 [+0.0005, +0.0019] at n = 32), an order of magnitude smaller than the ≈0.012 gain both adapters show over native TabPFN-2.5 at n = 16.

Reported results:

- The shared-row channel has large capacity: per-scene optimized shared rows improve on native by 0.1616 nats at n = 8 (per-column cells: 0.2004).
- Trained adapters capture a small part of it: both the story and the schema-only arm beat native by 0.0036–0.0117 at n = 8–32, and the two arms are indistinguishable or the story arm is slightly worse.
- The story adapter responds to whether a story is present (omitting it costs +0.0015 at n = 8) but not measurably to its content (irrelevant and role-permuted stories have CIs including 0 at n = 8, 16, 32).

Under the D-08 rule, "harmful" is checked before "supported"; the rule itself states no precedence, but no upper bound is below 0 in this run, so the order does not change the verdict.

## Detailed Findings

### Setup

- Channel: each of the 64 thinking rows gets θᵢ + Wᵢ·e(t), with Wᵢ (64×192×384 ≈ 4.7M parameters, no bias) zero-initialized. TabPFN-2.5 (`tabpfn==9.2.0`, raw module path, no ensembling) and all-MiniLM-L6-v2 (L2-normalized mean pool, 384-d) are frozen (D-01, D-04, D-05).
- Scenes: own random-DAG SCM generator, 6–15 continuous features, binary target; roles are Markov-blanket, irrelevant, and grouped noisy measurements of a hidden cause (D-06). Families: 0 linear/gaussian, 1 linear/heavy, 2 tanh/gaussian, 3 tanh/heavy (train and dev); 4 interaction/gaussian, 5 interaction/heavy (held out).
- Text: schema lists neutral names x1…xp and defines y; the story adds templated role facts only (D-07). Training mix: 15% of story-arm episodes omit the story, 10% contain one wrong fact (D-09).
- Training: 4000 steps × 8 episodes per arm, Adam lr 0.001 (warm-up 100, cosine to 0.1×), 3.65 h train wall time; checkpoint selected on dev validation.
- Evaluation (D-08): 200 held-out-family scenes, 5 nested support seeds, 512 query rows, paired per-scene difference with 1000 scene-bootstrap resamples. Corruptions (D-09): omitted, irrelevant (another same-family scene's story), wrong (permuted roles).

### Stage 1: channel capacity (ceiling)

Gain = native − per-scene optimized query log loss, 50 scenes, 95% CI (`report capacity`, `report capacity --arm per-column`).

| n | native | shared opt. | shared gain [95% CI] | per-column opt. | per-column gain [95% CI] |
|---|---|---|---|---|---|
| 8 | 0.5517 | 0.3902 | 0.1616 [0.1455, 0.1759] | 0.3513 | 0.2004 [0.1842, 0.2160] |
| 16 | 0.4829 | 0.3593 | 0.1235 [0.1075, 0.1397] | 0.3292 | 0.1537 [0.1366, 0.1696] |
| 32 | 0.4185 | 0.3436 | 0.0749 [0.0641, 0.0859] | 0.3185 | 0.1001 [0.0877, 0.1126] |

These use labels in the optimization and are an upper bound for the channel, not a method.

### Stage 3: dev validation

64 dev scenes (families 0–3), 3 support seeds, 512 query rows (`report training --validation`). Both arms selected step 3250.

| arm | n = 8 | n = 32 | n = 128 | mean |
|---|---|---|---|---|
| native | 0.5634 | 0.4279 | 0.3479 | 0.4464 |
| story (step 3250) | 0.5497 | 0.4205 | 0.3459 | 0.4387 |
| schema-only (step 3250) | 0.5493 | 0.4187 | 0.3447 | 0.4376 |

### Stage 4: primary (H1) and curve (H3)

Mean query log loss and paired differences, 200 held-out scenes (`report primary`, `report curve`).

| n | native | schema | story | story − schema [95% CI] | story − native [95% CI] | schema − native [95% CI] |
|---|---|---|---|---|---|---|
| 8 | 0.6292 | 0.6256 | 0.6248 | −0.0008 [−0.0019, +0.0003] | −0.0044 [−0.0072, −0.0018] | −0.0036 [−0.0067, −0.0007] |
| 16 | 0.5879 | 0.5762 | 0.5763 | +0.0001 [−0.0008, +0.0010] | −0.0116 [−0.0141, −0.0093] | −0.0117 [−0.0145, −0.0091] |
| 32 | 0.5353 | 0.5267 | 0.5278 | +0.0012 [+0.0005, +0.0019] | −0.0075 [−0.0090, −0.0060] | −0.0086 [−0.0104, −0.0071] |
| 64 | 0.4845 | 0.4806 | 0.4811 | +0.0005 [−0.0000, +0.0010] | −0.0033 [−0.0043, −0.0025] | −0.0038 [−0.0049, −0.0027] |
| 128 | 0.4443 | 0.4440 | 0.4443 | +0.0003 [−0.0001, +0.0006] | −0.0000 [−0.0007, +0.0006] | −0.0003 [−0.0010, +0.0004] |
| 256 | 0.4144 | 0.4146 | 0.4151 | +0.0005 [+0.0002, +0.0008] | +0.0006 [+0.0002, +0.0011] | +0.0001 [−0.0003, +0.0006] |
| 512 | 0.3940 | 0.3942 | 0.3947 | +0.0005 [+0.0002, +0.0007] | +0.0007 [+0.0004, +0.0011] | +0.0003 [−0.0001, +0.0006] |
| 1024 | 0.3806 | 0.3808 | 0.3812 | +0.0004 [+0.0002, +0.0006] | +0.0006 [+0.0003, +0.0009] | +0.0002 [−0.0000, +0.0005] |

`report primary` prints `D-08 verdict: harmful` (lower bound above 0 at n = 32).

Secondary, story − schema-only by held-out family [95% CI]:

| n | family 4 (100 scenes) | family 5 (100 scenes) |
|---|---|---|
| 8 | −0.0010 [−0.0028, +0.0007] | −0.0006 [−0.0019, +0.0009] |
| 16 | −0.0005 [−0.0016, +0.0007] | +0.0007 [−0.0007, +0.0022] |
| 32 | +0.0009 [−0.0000, +0.0018] | +0.0015 [+0.0005, +0.0026] |

### Stage 5: reliability (H4)

Corrupted − correct story query log loss for the story adapter, same rows as stage 4 (max |diff| to stage 4 story arm 0.00e+00 over 600 cells) (`report reliability`).

| n | story | omitted − story [95% CI] | irrelevant − story [95% CI] | wrong − story [95% CI] |
|---|---|---|---|---|
| 8 | 0.6248 | +0.00151 [+0.00098, +0.00204] | +0.00061 [−0.00010, +0.00134] | +0.00008 [−0.00002, +0.00020] |
| 16 | 0.5763 | +0.00068 [+0.00037, +0.00100] | +0.00025 [−0.00003, +0.00052] | +0.00002 [−0.00004, +0.00007] |
| 32 | 0.5278 | +0.00066 [+0.00046, +0.00085] | +0.00008 [−0.00004, +0.00019] | −0.00001 [−0.00004, +0.00001] |

Irrelevant stories and fully permuted roles were never seen in training.

### Interpretation

- Ceiling vs learned gain: at n = 8 the shared channel can deliver 0.1616 nats per scene, while the trained adapters deliver about 0.004 on held-out families (about 0.014 on dev). Almost none of the ceiling is reached, and what is reached does not depend on the story: the schema-only arm gets the same gain.
- [INFERENCE] The adapter uses whether a story is present, not what it says: omitting the story has a measurable cost, but swapping in another scene's story or permuting roles does not.
- A plausible mechanism, not a proven cause: on 40 dev stories, MiniLM embeddings have mean pairwise cosine 0.964 and 0.989 among stories with the same p; after removing the mean, the residual norm is 0.185. Role information is a small residual of e(t), so a linear Wᵢ·e(t) may see near-identical inputs for different role assignments.
- Magnitude: the "harmful" verdict rests on +0.0012 nats at n = 32, driven mostly by family 5; it is real under D-08 but practically small next to the adapters' shared ≈0.012 gain over native at n = 16.

### What this does and does not show

- Shows: on held-out mechanism families, with templated stories, a shared input-level shift of the thinking rows does not turn story content into lower log loss than the schema alone.
- Does not show: H2 (label savings) was not measured. Only held-out families 4–5 with template stories were tested; no paraphrases or real-world text. Only the raw module path, without the ensembled sklearn wrapper. Only one encoder (MiniLM).

## Confidence Notes

- All numbers except the encoder-similarity figures were quoted on 2026-11-11 from `uv run python -m texttabpfn.report` with the `capacity`, `capacity --arm per-column`, `training`, `training --validation`, `primary`, `curve` and `reliability` subcommands.
- The encoder-similarity figures (0.964, 0.989, 0.185) come from a one-off diagnostic run during training. That diagnostic is recorded in the run [journal](../../plans/text-conditioned-thinking-rows/journal.md), entry 2026-11-10T23:30Z, and no committed command reproduces it.
- CIs are 95% scene-bootstrap (1000 resamples) over paired per-scene differences; effects of 1e-3 nats are resolvable but small.
- The encoder-similarity mechanism is a diagnostic correlation, not an ablation.

## Open Questions

- Would a structured story encoding (per-feature role tokens or index-aware text encoding) make roles distinguishable to the adapter, given same-p cosine 0.989 (backlog B-12)?
- Does a stronger encoder (bge-base-en-v1.5, 768-d) help (B-09)?
- Per-column capacity exceeds shared capacity by about 0.04 nats at n = 8; is a trained per-column / cross-attention arm worth it (B-01)?
- Does deeper conditioning (every layer or LoRA) help if the input-only channel saturates (B-06)?
- How do results change inside the ensembled wrapper (B-02), with paraphrased or CausalDS stories (B-04, B-05), and when measuring label savings (B-03)?
