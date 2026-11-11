# Decisions

## D-01: Shared text-conditioned thinking rows; K1 and K3 dropped
2026-11-10, owner: test θᵢ ← θᵢ + Wᵢ·e(t), with one zero-initialized Wᵢ per thinking row. The shift is shared across all columns, and TabPFN-2.5 and the text encoder stay frozen. The owner chose this channel because the 64 rows are already dataset-level context that every column and test row reads, it keeps the KV cache and it adds no tokens. Not taken: per-group cross-attention (design B of the 2026-11-04 note) and the two-stage graph interface (design S). The oracle-edit (K1) and LLM-mask (K3) tests are dropped. Planning noted the risk: the shift is identical across columns, so per-feature facts must bind through the data cells' column embeddings. Stage 1 measures this.

## D-02: Capacity test includes a per-column comparison
2026-11-10, owner: stage 1 also optimizes per-column thinking cells (64×C×E) per scene. This is a diagnostic of how much a shared shift loses, not a trained arm. Labels used in this optimization never enter a reported method.

## D-03: Own synthetic scene generator
2026-11-10, owner: scenes come from our own generator, split by scene family. Not taken: the public CausalDS release. It is not in the repo and has about 100 scenes, too few to train about 4.7M adapter parameters (B-05).

## D-04: Frozen all-MiniLM-L6-v2 encoder
2026-11-10, owner: e(t) is the mean-pooled all-MiniLM-L6-v2 embedding (384-d, 256-token window). Wᵢ then totals 64×192×384 ≈ 4.7M parameters. Not taken: bge-base-en-v1.5, which is 768-d with a 512-token window (B-09).

## D-05: Local compute, pinned model and license
2026-11-10, owner: everything runs on this M4 Pro (48 GB, MPS). The TABPFN-2.5 Non-Commercial License is accepted, so the run may download `tabpfn-v2.5-classifier-v2.5_default.ckpt` from Prior-Labs/tabpfn_2_5. Planning pins `tabpfn==9.2.0` and uses the raw module path only; the ensembled wrapper is B-02.

## D-06: Scene generator contract
2026-11-10, planning. Each scene is a random-DAG SCM with 6–15 continuous observed features and a binary target from a thresholded latent outcome. Roles:
- Markov-blanket features;
- irrelevant features;
- groups of two or three noisy measurements of one hidden cause of the target.

Six mechanism families (mechanism class × noise regime): four for training and dev scenes, two held out for evaluation, so evaluation covers unseen data-generating processes. Features are continuous only, so `_remove_constant_features` never shifts the column order that stories refer to.

## D-07: Text contract
2026-11-10, planning. The schema lists neutral names x1…xp in column order and defines the binary outcome y. It gives no role hints. The story is templated facts only: which features directly relate to y, which are unrelated, and which groups measure one hidden quantity. No coefficients, signs, equations or noise levels. Not taken in iteration 1: LLM paraphrases, which need a local LLM and add a transfer question (B-04).

## D-08: Evaluation, verdict and stop rule
2026-11-10, planning, fixed before any method run.
- Evaluation design:
  - 200 held-out-family scenes, n ∈ {8 … 1024}, 5 nested support seeds, 512 fixed query rows.
  - Support draws are stratified so both classes appear.
  - Metric: the paired per-scene log-loss difference (story minus schema-only), averaged by scene, with 1000 scene-bootstrap resamples.
- Verdict:
  - "supported" if the CI upper bound is below 0 at two or more of n = 8, 16, 32;
  - "harmful" if the CI lower bound is above 0 at any of them;
  - otherwise "not supported".
- Stop rule: after stage 1, if the shared-row capacity CI includes 0 at all of n = 8, 16, 32, stages 2–5 wait for the owner. The note in stage 6 then reports the channel result.

## D-09: Story mix in training and corruptions in evaluation
2026-11-10, planning.
- Training: 15% of story-arm episodes omit the story, and 10% contain one wrong fact. This exercises the fallback and sets a fixed, reportable prior on trust.
- Evaluation corruptions: omitted (schema only), irrelevant (another same-family scene's story) and wrong (feature roles permuted).

## D-10: Code layout
2026-11-10, planning.
- Experiment code lives in `src/texttabpfn/` under a uv project pinned to Python 3.12 for torch and tabpfn wheels.
- Its tests live in `tests_exp/`, so the related-works suite (`unittest discover -s tests`) stays torch-free.
- Results go to `results/<stage>/`.
- `python -m texttabpfn.report <name>` prints a table and exits non-zero when the table is incomplete.

## D-11: Per-feature role tokens through the shared thinking-row channel
2026-11-11, planning, following the owner's direction to do B-12 first.
- Each observed feature x_j gets one token: the frozen MiniLM embedding of a sentence that states only x_j's role from the story (influences y, is influenced by y, measures hidden quantity k, or is unrelated to y).
- Tokens sit in slot j, in column order; there are at most 15 slots.
- A shared learned reduction feeds θᵢ + Wᵢ·s. The rows stay identical across columns, Wᵢ starts at zero, and the parameter count stays near iteration 1's ≈4.7M.
- Matched control: the same architecture, with the neutral schema sentence for x_j in every slot.
- Roles come from `Scene.describe()`, which is an exact parse of the template stories. Free-form stories would need an extraction step (B-14).

Not taken: one-hot role vectors. With templates they carry the same information, but text keeps the encoder in the loop for B-09 and for real stories. Iteration 1's pooled story embeddings were near-identical (same-p cosine 0.989).

## D-12: Story-guided preprocessing baseline
2026-11-11, planning. A context-aware baseline that needs no training:
- drop the features the story calls unrelated;
- collapse each measurement group to one feature with a label-free method over support and query feature values.

A plain mean is not used because the story gives no signs and measurement weights have random signs. The baseline uses only story facts and unlabeled inputs, so it is a valid p_c (the feature engineering from context of the CAAFE line). Applying it to native TabPFN-2.5 gives the first deciding number.

## D-13: Stop rule after stage 7
2026-11-11, planning. If the 95% CI of preprocessing-minus-native log loss includes 0 at all of n = 8, 16 and 32, stages 8–12 wait for the owner. In that case role knowledge is worth nothing measurable to TabPFN on these scenes, even when applied directly. The run then reports this result instead of spending about 10 h training adapters that are unlikely to beat it.

## D-14: Follow-on arm selection
2026-11-11, planning. The owner asked for "B-09 or B-01" after B-12. The choice depends on the stage-9 D-08 verdict for role-token story versus matched control:
- **"supported"** → B-09. The shared channel can use roles when they are exposed, so the test is whether a stronger pooled encoder (bge-base-en-v1.5) recovers a benefit from the plain story text without hand structure.
- **"not supported" or "harmful"** → B-01. Even exact per-feature roles fail through rows that are identical across columns, so the next test is per-column rows that bind each column's role tokens directly.

The orchestrator records the choice in `journal.md`. The branch not taken stays in the backlog.

## D-15: 10k-step role-token retrain decides D-14
2026-11-11, owner: retrain both role-token arms for 10,000 steps (stage 13). Planning fixed the rest at 19:20Z, before any stage 9 held-out result existed.
- D-14 is applied to the 10k-step `role10k-primary` verdict instead of the 4000-step `role-primary` verdict.
- Both are reported.
- Stage 10 waits for stage 13.

Reason: the 4000-step story arm was still improving at its last checkpoint, so the better-trained pair is the fairer test of whether the shared channel uses roles. Not taken: continuing the 4000-step run with a new schedule, which would not give a clean budget-matched pair.
