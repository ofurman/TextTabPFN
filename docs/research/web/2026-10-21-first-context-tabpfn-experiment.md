---
date: "2026-10-21T12:32:11+02:00"
researcher: "Oleksii Furman"
topic: "First CausalDS experiment and architecture for shared task-context conditioning of TabPFN"
tags: ["tabpfn", "causalds", "task-context", "architecture", "sample-efficiency"]
sources: ["primary-papers", "author-code", "repository-inspection"]
status: complete
last_updated: "2026-10-21"
last_updated_note: "Added TabICL cross-attention architecture assessment"
---

# Research: First context-conditioned TabPFN experiment

## Research Question

> Check repo and think what first approach we should test. Data is data from CausalDS where we have dataset and task-context. Think about the architecture how to merge task context to improve tabpfn predictions

## Summary

Test a **feature-aligned residual adapter**: a frozen language encoder represents the story together with each feature group's meaning and the prediction target; a small trained MLP modulates TabPFN's feature embeddings. Freeze TabPFN initially and train the adapter across scenes, using held-out rows as the episodic training objective. This is a proposed experiment, not an established improvement. If only the public CausalDS release is available, first run a fixed-budget story-guided feature-engineering pilot while preparing fresh scenes for adapter training.

Repository inspection found a research specification, literature collection and validation tools, but no experiment implementation, dependency configuration or scene data. The working tree was clean before this document. This recommendation follows the [canonical problem](../../../AGENTS.md#research-problem); maintained paper records remain in the [catalogue](../../related_works/README.md).

## Detailed Findings

### Data readiness and information contract

CausalDS stories communicate graph relationships, including latent variables, while numerical mechanisms are sampled separately. Its public release has 100 main scenes/tasks; ablation views can reuse the same formal problem. Native tasks include causal questions and prediction, rather than a ready-made binary-classification benchmark. Public and grading artifacts are distributed together, requiring an explicit input whitelist. Calibration can disclose additional conceptual labels. These facts constrain adaptation to our question. [CausalDS v2](https://arxiv.org/html/2607.08093v2), [release data card](https://github.com/andleb/causalds/blob/main/data/README.md).

Start with observed binary outcomes and ordinary IID prediction. Verify actual target types and prevalence before selecting the initial cohort; binary-scene availability was not measured here. Do not silently threshold continuous targets using full-dataset or query statistics. If binary generation is needed, define its mechanism in the generator before sampling.

Give every method the same schema, feature meanings, target/class meanings and support rows. Only the story differs. Keep SCM equations, parameters, graph diagnostics and query outcomes out of model inputs. Exclude extra calibration labels initially. Keep descendants when available at prediction time: causal adjustment and observational prediction have different objectives. Explicitly whitelist inputs rather than recursively serializing scene directories.

### Recommended first learned architecture

Use one pinned, accessible **TabPFN 2.5** checkpoint for the initial implementation and all matched baselines. The existing text-adapter work establishes frozen text encoder–adapter–frozen TabPFN feasibility for row text, not for one shared story. Its classification results do not establish uniform superiority. [Text-adapter paper](https://arxiv.org/html/2606.04876), [maintained note](../../related_works/2026-text-encoders-for-tabpfn.md).

Proposed information flow:

```text
story + public schema + target/class meanings + current group description
                      │
              frozen text encoder
                      │ z_g (cached per scene and feature group)
               small shared MLP
                      │ gamma_g, beta_g
                      ▼
numeric table → native TabPFN embedding h_ig → residual modulation
                                              │
                                 frozen TabPFN blocks/head
                                              │
                                  query class probabilities
```

For scene s, feature group g and row i:

\[
z_{sg}=E(c_s,\Sigma_s,\text{target meanings},\text{ordered group }g),
\qquad (\gamma_{sg},\beta_{sg})=A_\theta(z_{sg}),
\]

\[
\widetilde h_{sig}=h_{sig}+\gamma_{sg}\odot h_{sig}+\beta_{sg}.
\]

The same modulation is shared across rows, but it interacts with each row's numerical representation. Feature/target grounding lets the adapter distinguish, for example, related measurements from unrelated variables; a pooled story alone does not explicitly bind statements to columns. Encode the permitted text once per group and cache it. Check tokenizer length explicitly so the story and target are not silently truncated. A compact sentence encoder is a reasonable initial implementation choice, with an untested ability to represent long relational stories.

Use a two-layer MLP with hidden width 128 as an initial configuration, not a tuned optimum. Its last linear layer emits twice the backbone embedding dimension and is initialized to zero. This recovers the native model at initialization without adding attention tokens. Do not also initialize a separate multiplicative gate to zero. Leave target tokens and synthetic thinking rows unchanged.

The current v2.5 implementation groups original features into tokens and embeds the target separately. Insert modulation after native embeddings/feature-position information are prepared and before thinking rows and transformer blocks. The source exposes this as an internal integration point, not a stable sklearn API. Pin the package, source commit and checkpoint before implementation. [Official v2.5 source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2_5.py).

Maintain an explicit map from transformed columns to native feature groups, including order, dropped columns and padding. Start with one deterministic, simple preprocessing configuration and no ensemble permutations; use precisely that configuration in the control. Anonymous group indices are insufficient semantic grounding. A later experiment can expand preprocessing while transporting metadata consistently.

Train only the adapter. Freezing backbone parameters still requires differentiating through its forward pass; wrapping it in `no_grad` would prevent learning. Disable detached inference caches and inference-only/in-place memory optimizations in the training path. Verify finite nonzero adapter gradients and absence of backbone gradients before running training. Match native predictions at zero initialization within numerical tolerance. Check row permutation behavior, feature/metadata permutation consistency and query-label isolation.

**Why this interface first:** it is small, preserves token counts and attention masks, and provides an exact unconditioned initialization. Appending the same story embedding as numeric columns can lose it through constant-column removal/normalization. Adding story tokens is a valid later ablation, but changes attention normalization even when their embeddings are zero. Cross-attention and graph-derived attention biases are subsequent options if this simple interface is insufficient. The partial-graph causal PFN is a structural-conditioning precedent, not evidence that frozen TabPFN can exploit stories for classification. [Partial-graph paper](https://arxiv.org/abs/2602.14972), [maintained note](../../related_works/2026-use-what-you-know.md).

### Training unit and required data

One episode is a scene, a support subset and a disjoint query subset. Minimize query binary cross-entropy, averaged by scene, through the frozen backbone into the adapter. Training-scene query labels are legitimate supervision; final evaluation query labels remain inaccessible until scoring. Train across support sizes, with missing-story episodes to exercise the fallback. Learn across scenes, not separately from the handful of labels in each evaluation scene.

The public 100-scene release is a pilot resource, not sufficient evidence for broad generalization of a learned bridge. Generate additional independent scenes, with distinct development and final evaluation groups. Fresh generation is supported but entails an LLM story-generation pipeline and cost. [Generation guide](https://github.com/andleb/causalds/blob/main/docs/generation.md). Determine the necessary training-scene count using a development learning curve rather than claiming a known sample requirement.

Keep every target, story paraphrase, measurement variant and support-size episode from the same underlying scene/SCM in one split. Also hold out mechanism/motif families where feasible; held-out seeds alone are a weaker test. Report meta-training scenes/labels and generation cost separately from per-test-scene label savings.

### First experiment and controls

Use nested support sizes n = 8, 16, 32, 64, 128, 256, 512, 1024 where scene sizes permit, with fixed disjoint query rows and several support-sampling seeds. Predefine treatment of support subsets with a single observed class; do not resample until outcomes look favorable.

| Arm | Purpose |
|---|---|
| Native frozen TabPFN | Reference performance and implementation parity |
| Schema-only trained adapter | Matched capacity/training control for the story contribution |
| Schema-plus-story trained adapter | Proposed method |
| Same story model with story omitted | Within-model dependence on context |
| Same model with irrelevant or deliberately incorrect story | Reliability and harmful-context tails |

The schema-only adapter receives all public semantics and trains with the same scene/episode budgets. Its learned numerical predictor can improve without any story, so beating vanilla TabPFN alone does not establish context benefit. Make irrelevant stories schema-compatible and hold target/class meanings fixed. Distinguish truthful counterintuitive stories from erroneous stories. Do not assume a missing-story fallback guarantees robustness to plausible false context.

Primary metric: mean paired query log-loss difference, aggregated over scenes with scene-level bootstrap uncertainty. Also report Brier score, AUROC where both classes occur, and the distribution of per-scene harm. Plot improvement against support size. Choose any sample-efficiency loss threshold on development scenes and report unreached thresholds honestly. Avoid treating rows, support sizes or paraphrases as independent scenes.

If only the public release is available now, the first executable pilot should use a fixed prompt to propose a small number of deterministic story-guided transformations, followed by unchanged TabPFN. Compare schema-only and schema-plus-story prompts with identical budgets and no query-label selection; validate any selected transformations using support labels only. This follows the existing feature-engineering interface rather than claiming a new method. [CAAFE paper](https://arxiv.org/abs/2305.03403), [maintained note](../../related_works/2023-caafe.md). Positive results justify investment but do not prove the latent adapter will work; negative results are not decisive evidence that the stories are useless.

## Sources Consulted

Primary papers and author documentation/code are linked at their supporting claims. This is an architecture recommendation and targeted verification, not a benchmark reproduction. No additional paper records or PDFs were introduced.

## Key Insights

The scientific comparison is the same trained architecture with and without the story. The engineering choice is a zero-initialized, feature-grounded modulation inside TabPFN. The data bottleneck is independent scenes for learning the interface, not the number of rows within one scene. A benefit would concern additional communicated structural information, not improved inference from identical inputs.

## Confidence Notes

No CausalDS scenes were loaded, generator executed, checkpoint downloaded or adapter trained. Source interfaces were inspected remotely; implementation details must be verified against the pinned version. Frozen-backbone modulation may fail to exploit useful structure even when the text encoder represents it. Graph information alone does not specify numerical predictive mechanisms, so a genuine zero or negative result is plausible.

## Open Questions

- How many independent training scenes and observed binary targets are actually available to this project beyond the public release?
- Does the simple text encoder retain the relevant relationships, and can frozen TabPFN use its modulation?
- Does conditioning improve held-out-scene log loss beyond the equally trained schema-only adapter, especially at low support sizes?

The first is a data inventory needed before implementation. The others are empirical questions for the proposed experiment.

## Clarifications Log

No additional user preferences were required for this recommendation.

## Follow-up Research [2026-10-21 12:38]

> Check tabicl and think how we want to add task semantic information with language model embeddings and cross attention, is this a good idea?

### Recommendation

**Yes: test one feature-grounded residual cross-attention block before TabICL row compression.** This is a stronger first architecture hypothesis for the requested cross-attention experiment than broadcasting a pooled task vector. Retain the earlier FiLM proposal as a cheaper comparison. The benefit of either interface remains untested.

TabICL uses column-distribution encoding, within-row feature interaction/CLS compression, and dataset-level ICL. V2 retains that structure and adds feature grouping and early observed-support-label injection. Target label embeddings are distinct from natural-language target meanings. [Original architecture](https://arxiv.org/html/2502.05564v1#S3), [v2 architecture](https://arxiv.org/html/2602.11139v1#S3), [maintained v2 note](../../related_works/2026-tabicl-v2.md).

The code was inspected at official commit `c91f00df184a5097e584cb376f7e699e9f3444c4`. It has a concrete seam between `col_embedder` and `row_interactor`; learned CLS slots are initialized by the latter. Modify data/group tokens only. V2 groups noncontiguous columns, so obtain the ordered membership from the actual transformation rather than assuming one feature per token. [Core forward](https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/src/tabicl/_model/tabicl.py#L347), [CLS pooling](https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/src/tabicl/_model/interaction.py#L138), [grouping code](https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/src/tabicl/_model/embedding.py#L208).

### Architecture and equations

```text
story + public target meaning → frozen LM → contextual text memory T
                                                        │ keys/values
feature/group meanings → frozen LM → semantic vectors S  │
                                         │              │
table + support labels → TabICL column encoder → H       │
                                         │              │
                              queries from [H, S] ───────┘
                                         │
                               residual cross-attention
                                         │
                         original row interaction + CLS pooling
                                         │
                           original dataset ICL + prediction head
```

For each row i and feature group g, let h_ig be the native distribution-aware representation. Let s_g encode the ordered group descriptions, units/category meanings and target definition. Preserve the story as L contextual token states or a sequence of sentence/chunk embeddings T, rather than one averaged vector:

\[
q_{ig}=W_q\operatorname{LN}(h_{ig})+W_s s_g,
\qquad K=TW_k,\quad V=TW_v,
\]

\[
a_{ig}=\operatorname{MultiHeadAttention}(q_{ig},K,V),
\qquad \widetilde h_{ig}=h_{ig}+W_o a_{ig}.
\]

Use token-level contextual hidden states from a frozen language encoder, not raw word-embedding lookups. Schema vectors can be pooled because they identify groups; story memory must retain multiple elements for selective attention. Frozen LM encodings do not intrinsically align with tabular hidden states: learn W_q, W_s, W_k, W_v and W_o across scenes.

Start with one block, attention width 128 and four heads as provisional settings. Zero-initialize W_o and its bias; keep other projections ordinarily initialized. Do not also multiply by a zero-initialized gate. This begins with native predictions while allowing the interface to learn. Freeze original LM and TabICL parameters initially. Use contextual token memory within the encoder's limit and track truncation explicitly. Longer-story chunking is a later ablation, because compressing each sentence can itself lose relations.

Pre-pooling fusion lets row compression depend on the story while feature identity is still explicit. For example, a statement that two columns are noisy measures of one quantity can influence their representations before aggregation. Post-pooling attention is cheaper and worth comparing, but cannot assume that all feature-specific distinctions survived compression. A latent attention map is not a recovered causal graph, and the block does not guarantee correct causal interpretation.

With one text key/value, the softmax has one entry and equals one for every query. The result is a shared vector before further gating, so it cannot retrieve different story statements for different features. With multiple keys, row values and feature semantics can jointly determine which text information is retrieved. This is the principal reason cross-attention could improve on pooled conditioning.

### Training and implementation constraints

Train episodically on independent scenes, using support rows for ICL and disjoint training queries for the loss. At evaluation, hold scene families out and vary support size. No query outcomes enter text encoding or the adapter. Story embeddings are shared and cached per scene; adaptation is learned across training scenes rather than fitted anew from eight evaluation labels.

The normal sklearn prediction route is insufficient for training this block. In the inspected code, evaluation paths use inference managers with `torch.no_grad`; downstream row/ICL modules must follow their differentiable training forwards. Freeze weights with `requires_grad_(False)` while retaining gradients with respect to the adapter output. Check train/eval and dropout behavior explicitly; simply calling `.eval()` to freeze TabICL would sever the needed path in this implementation. Normalization still occurs downstream; a vector-valued residual can alter representation direction, unlike a uniform scalar offset that layer normalization can remove. [Pinned implementation](https://github.com/soda-inria/tabicl/tree/c91f00df184a5097e584cb376f7e699e9f3444c4/src/tabicl).

Keep native support/query attention masks. Preserve metadata alignment through preprocessing, grouping and permutations. Start with numeric inference caches disabled; downstream caches must be invalidated when the adapter or context changes, while upstream numeric caches might later be reusable. Text-memory caching is safe when the text encoder is frozen. Return fresh adapter tensors because native downstream code mutates CLS slots and adds labels in place; do not mutate reused cached representations. Verify zero-initialization parity, nonzero adapter gradients, correct feature/group alignment, and query-label isolation before training.

Attention-score cost is proportional to rows × groups × text length, in addition to projection costs and the backbone. Reusing text K/V saves encoding work but not those attention scores. If expensive, a later variant can retrieve text once per semantic feature group and use value-dependent FiLM modulation per row; this trades row-specific retrieval for lower cost.

### Experiment and interpretation

Prioritize native TabICL, a trained schema-only cross-attention control, and the proposed schema-plus-story model. Compare pooled-context FiLM and post-pooling cross-attention next. Give controls identical schema/target semantics, training scenes and episode budgets. Evaluate correct, omitted, irrelevant and plausibly false stories with the same support/query rows. Train with some story omission to exercise fallback; that does not guarantee protection against false context.

Use paired scene-level log loss and support-size curves, following the earlier protocol. The strongest result would be lower log loss on unseen scenes than both the trained schema-only interface and pooled FiLM, without unacceptable incorrect-story degradation. If cross-attention fails, distinguish inadequate scene diversity, weak relational text representations, and a frozen backbone that cannot use the injected signal before concluding that stories contain no useful information.

The public-release scene-count limitation still applies. No checkpoint was loaded or model trained in this follow-up. The recommendation is supported by architecture/source inspection, not measured gains. No new paper records or PDFs were added; the maintained wiki remains the source of individual paper records.

## Follow-up Research [2026-10-21: TabPFN v1]

> What if we want to use tabpfnv1 instead of tabicl

TabPFN v1 is viable for a small-data conditioning experiment, but its tokens represent entire rows. Its encoder mixes numerical columns before the row transformer; support-label embeddings are added only to support rows. There is no native feature-token stage at which to insert the earlier feature-level attention. The original evaluation focused on up to 1,000 support rows, 100 numerical features and 10 classes. [Original paper](https://arxiv.org/html/2207.01848), [official v1 transformer source](https://github.com/PriorLabs/TabPFN/blob/tabpfn_v1/tabpfn/transformer.py), [encoders](https://github.com/PriorLabs/TabPFN/blob/tabpfn_v1/tabpfn/encoders.py).

For the first experiment, preserve the checkpoint and add one residual cross-attention adapter to native row tokens at the transformer input, after any native input normalization. Let H contain row embeddings with support labels added exactly as in the original model, and T contain contextual LM tokens for the story, ordered schema and target/class meanings:

\[
\widetilde H=H+W_o\operatorname{Attention}(\operatorname{LN}(H)W_q,TW_k,TW_v).
\]

Freeze the LM and v1 backbone; train only the adapter projections across scenes. Zero-initialize W_o and its bias. Keep original row masks, sequence length and query-label isolation. Text remains external memory rather than synthetic labeled examples appended to the row sequence. Use multiple text tokens, not a single pooled key/value. Verify autograd and initialization parity through a custom raw-model wrapper rather than assuming the prediction API supports adapter training.

Feature grounding is weaker here than in the TabICL proposal. Put explicit slot-to-feature mappings in the public schema memory and preserve them through all preprocessing. Start with a single deterministic feature order and preprocessing configuration; apply any column permutation consistently to values and metadata. Even then, row cross-attention must learn to bind already-mixed numeric coordinates to language semantics. A negative result may reflect this bottleneck.

If that interface fails, add a feature-wise side branch: encode each numerical value with its feature semantics, retrieve story information per feature, pool the results per row, and add a zero-initialized projection to the native row token. Preserve the original numerical encoder. This adds explicit binding but is a larger architectural change requiring more training evidence.

Compare native v1, a trained schema-only row adapter, and a story-conditioned row adapter; include absent/incorrect-story tests and the same held-out-scene protocol. Use support sizes within v1's intended small-data regime. This is a reasonable prototype of task conditioning, with limited evidence about feature-level structural reasoning or performance in larger-data regimes. No v1 checkpoint or gradient experiment was run here.

## Follow-up Research [2026-10-21: Stronger v1 architecture]

> think of better architecture with tabpfnv1

### Revised recommendation: condition the row encoder through feature-specific weights

Promote feature grounding from a later extension to the main architecture. Keep v1's numerical encoder and add a **story-conditioned residual row encoder**. Cross-attention operates on semantic feature queries before numeric row compression, then its outputs generate column-specific corrections to the native encoder. This supersedes row-token cross-attention as the recommended first architecture; keep the simpler row adapter as an ablation.

The official v1 source first calls its numerical encoder, adds labels to support tokens, then runs the row transformer. It supplies linear encoders among its encoder implementations. Check the actual checkpoint encoder before using its weights algebraically; do not assume every supported encoder is affine. [V1 forward](https://github.com/PriorLabs/TabPFN/blob/tabpfn_v1/tabpfn/transformer.py), [encoder implementations](https://github.com/PriorLabs/TabPFN/blob/tabpfn_v1/tabpfn/encoders.py).

Let T be a frozen contextual encoding of the permitted story, schema and target meaning, and e_j a frozen encoding of feature j and target meaning. Cross-attention gives a story-informed state for each feature:

\[
a_j=\operatorname{Attention}(W_qe_j,TW_k,TW_v),
\qquad \Delta w_j=W_o\operatorname{MLP}([e_j;a_j]).
\]

For preprocessed values x_ij, use:

\[
h_i=E_0(x_i)+\frac{1}{\sqrt m}\sum_{j=1}^{m}x_{ij}\Delta w_j.
\]

The 1/sqrt(m) factor is an initial stabilization choice for the residual only, not a claimed variance guarantee for correlated features. Preserve native normalization, padding and scaling. Compute the correction in the same numeric coordinate system used by E_0, and inject before native label addition; retain the rest of the forward exactly. Do not normalize corrected scalar columns again, which could erase simple rescaling effects.

When E_0 is affine, this is precisely a scene-conditioned correction to each encoder column. Even if E_0 is nonlinear, the separate residual branch remains implementable but loses that exact interpretation. Every language-conditioned vector multiplies a known feature value; the network does not have to rediscover column identity from a mixed row token.

Important distinction: this is not just a scalar importance gate per column. Delta w_j is a vector in the backbone's row-embedding space, allowing text to change feature directions and combine related measurements. Numerical sign, relevance and strength must still be learned from the permitted story and training scenes. A causal relationship does not justify assigning an uncommunicated numerical coefficient.

Compute e_j, T, a_j and Delta w_j once per dataset/preprocessing configuration; reuse them for support and query rows. Cross-attention score cost is O(mL), rather than O(nmL) for per-cell retrieval. Numeric aggregation still costs O(nmd), comparable in structure to a linear row encoder. Feature permutation must transport the value–metadata association and the native encoder input mapping consistently; the residual branch alone does not make the frozen model permutation-invariant.

Start with an attention bottleneck of 64 dimensions, two heads and one small MLP, as provisional settings. Freeze LM and TabPFN; train only adapter weights across scenes using query loss. Zero-initialize W_o and its bias while initializing other components normally. This gives native predictions at initialization. Missing story retains public schema/target memory and is included during training; false-story robustness must be measured separately. Keep representations and class mappings consistent across any preprocessing ensembles, initially disabling ensembles to simplify verification.

### What this can and cannot express

This interface explicitly binds story information to columns while preserving the pretrained path and avoiding a separately trained numerical feature transformer. It can represent story-conditioned linear combinations, including different weightings of repeated noisy measurements. It does not directly provide arbitrary nonlinear or pairwise transformations. The frozen transformer may or may not turn its row representations into the desired predictions.

If linear conditioning works, the next extension is a small shared numeric basis phi(x) = [x, x squared] with separate story-conditioned residual directions for each basis component. This gives nonlinear row encoding but must be evaluated independently. Do not start with a full new feature transformer or simultaneous changes throughout the backbone: those make failure attribution and scene-data requirements harder.

The earlier row-token attention is more data-dependent in its retrieval because its queries contain row values, whereas this design retrieves text once from feature semantics. That is an intentional tradeoff, not universal superiority. The revised design prioritizes explicit binding, small training burden and cheap scene-level caching. A later value-dependent retrieval variant is warranted if the story's relevance varies strongly with feature values.

### Decisive initial comparison

Compare native v1, equally trained schema-only encoder correction, story-conditioned encoder correction and the earlier row-token cross-attention. Use the same encoder/episode budgets for both semantic arms. The hypothesis is lower held-out-scene log loss from explicitly binding context to columns. Also inspect wrong-story degradation and feature/metadata permutation checks; attention heatmaps alone do not validate causal understanding.

These equations are our proposed architecture, not a reproduced method or an established improvement. No checkpoint was loaded, trained or evaluated. The main unresolved risk remains whether a small learned encoder correction can usefully communicate structure to a frozen v1 backbone with the available diversity of training scenes.
