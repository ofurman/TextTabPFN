---
date: "2026-11-04T12:00:00+01:00"
researcher: "Oleksii Furman"
topic: "Text-tabular attention: injecting a shared task story into TabPFN-style alternating attention"
tags: ["tabpfn", "tabpfn-2.5", "contexttab", "task-context", "architecture", "cross-attention"]
sources: ["primary-papers", "author-code"]
status: complete
last_updated: "2026-11-04"
last_updated_note: "Added stress test, kill tests, simple LLM-mask baseline, frozen-backbone options and verified implementation details"
---

# Research: Text-tabular attention for TabPFN

## Research Question

> Check TabPFN v1, TabPFN v2, TabPFN-2.5 and ConTextTab (arXiv 2506.10707) and think whether we can create something like text-tabular attention. Our goal is to combine the text of the task and the tabular data to improve the results.

Follow-up questions in the same session: how the thinking rows work, what to do without training the whole TabPFN-2.5 model, and a stress test of the proposal.

## Summary

A text-tabular interface is technically easy to build into TabPFN-2.5. The most natural channel is its 64 learned **thinking rows**: condition them on the story through cross-attention, per feature group. The data cells then read the story through the native row attention. A zero-initialized residual gives the native predictions at initialization. The cost does not grow with the number of rows. The backbone can stay frozen; only about 0.22M adapter parameters are trained across scenes.

The stress test below weakens the case for this design:

- The main risk is **too little information in the stories**, not the attention block. With randomly sampled mechanisms, a story communicates structure (which features are relevant or redundant), not signs or effect sizes.
- Input-only conditioning of a small frozen model may be **too narrow a channel**. Test rows read the thinking rows through one K/V head only.
- A training-free **LLM relevance mask** in front of unchanged TabPFN-2.5 may capture most of the benefit. The learned adapter must beat that baseline, not only native TabPFN.

Run three cheap kill tests (K1–K3) before training any adapter. Treat the thinking rows as the first channel to *measure*, not a committed architecture. The research contribution is the empirical answer to H1–H4; text-to-soft-prompt conditioning is a standard mechanism.

## What the four papers give us

| Model | Token unit | Attention | Slot for text | Lesson for us |
|---|---|---|---|---|
| [TabPFN v1](../../related_works/2023-tabpfn.md) | One token per row | Row attention only | Row-level cross-attention or prepended pseudo-rows | No per-feature tokens, so the story cannot bind directly to columns. Use v1 only as a small prototype. |
| [TabPFN v2](../../related_works/2025-tabpfn-v2.md) | One token per (row, group of 2 features); the target is its own column | Feature attention, then row attention, then MLP; test rows attend only to train rows | A random "feature positional embedding" is added to every cell of a column, so columns with the same statistics are different | This random embedding is the slot ConTextTab fills with column-name semantics. |
| [TabPFN-2.5](https://arxiv.org/abs/2511.08667) | Same as v2, groups of 3 features | Same as v2, 24 layers (classification) | **64 learned thinking rows** prepended on the context side | The thinking rows are a ready-made, parity-preserving entry point for text. |
| [ConTextTab](../../related_works/2025-contexttab.md) | Same as v2 | Same as v2, with weight sharing | A frozen MiniLM encodes headers and text cells; a linear projection maps them and they are *summed* with cell embeddings | Semantics work only after pretraining on 2.18 M real tables (T4). Column names give about 1 % accuracy. LLM-enriched descriptions were not significant (57 % win rate, p = 0.4). It has no task/story input. |

## Verified source facts (TabPFN-2.5)

From the [official v2.5 source](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2_5.py) on `main`, inspected 2026-11-04. Pin a commit before implementation.

- **Config defaults:** `emsize` 192, 24 layers, 3 heads (head size 64), MLP hidden size 384, `features_per_group` 3, 64 thinking rows. No dropout in the architecture file. The backbone has about 10.7M parameters. These are defaults; check the loaded checkpoint's values.
- **Thinking rows** (`AddThinkingRows`): one learned 64 × E parameter. The same vector is copied to every column, including the target column, because the number of features varies. They get no column positional embedding and no label. At input, the cells of a thinking row are identical across columns.
- **Position:** the thinking rows are put before the training rows, and `single_eval_pos` moves past them. They are context: training and test rows attend to them, and they attend to all training rows. After layer 1 they depend on the data. They are never decoded.
- **Multi-query attention for test rows:** in per-column row attention, training rows use all 3 K/V heads, but **test rows attend through the first K/V head only** (`k_BcNHD[:, :, :1]`). The KV cache stores only that head.
- **Column embeddings:** random vectors (fixed seed, E/4 wide) passed through a learned linear layer, added to every cell of a feature group.
- **Grouping:** constant columns are removed **before** grouping (`_remove_constant_features`). Remaining columns are grouped consecutively in threes, with zero padding at the end.
- **Gradients:** `save_peak_memory_factor` must be `None`; otherwise `chunked_evaluate_maybe_inplace` asserts that gradients are off. `force_recompute_layer=True` wraps each block in `torch.utils.checkpoint`. The sklearn wrapper predicts without gradients, so training must call the raw module.

## Design space

All designs keep the language model frozen. Use T for the contextual token states of the permitted text (story + schema + target meaning), and s_g for a pooled encoding of the description of feature group g.

### 0. LLM relevance mask — training-free baseline

An LLM reads the story and schema and returns, per feature: relevant, redundant (with which features), or irrelevant. Drop irrelevant features, average redundant measurements, then run unchanged TabPFN-2.5 with its normal ensembling. A schema-only prompt with the same budget is the control. This is close to [CAAFE](../../related_works/2023-caafe.md), restricted to structural edits. It needs no training, and it is the number every learned design must beat.

### A. Semantic column embeddings (ConTextTab-style)

Replace or add to the random column embedding: e_g = e_g^rand + W s_g, with W initialized to zero. The story goes into s_g ("story + description of group g"). It is cheap and directly comparable with ConTextTab, but gives one pooled vector per column. The frozen backbone learned these embeddings as *identifiers*, not meanings. This is close to the FiLM adapter of the [2026-10-21 proposal](2026-10-21-first-context-tabpfn-experiment.md#recommended-first-learned-architecture).

### B. Story-conditioned thinking rows — first learned channel to measure

For thinking row t and column g (feature groups and the target column):

$$
u_{tg} = \theta_t + W_s s_g,\qquad
\tilde\theta_{tg} = \theta_t + W_o\,\mathrm{MHA}\big(\mathrm{LN}(u_{tg}),\,T W_k,\,T W_v\big),
$$

with θ_t the pretrained thinking-row vector and W_o (and its bias) initialized to zero. Replace `row_token_values_TE` expanded over columns with the tensor θ̃ of shape T × C × E. The rest of the forward stays the same.

```text
story + schema + target meaning ──► frozen LM ──► text tokens T ──┐ keys/values
feature-group descriptions ──► frozen LM ──► s_g ─┐                │
                                                  ▼                ▼
                  pretrained thinking rows θ_t ─► cross-attention ─► θ̃_tg  (64 × C cells)
                                                                     │ prepend (context side)
table + support labels ─► native cell embedding ─────────────────────┤
                                                                     ▼
            24 × [feature attention → row attention (reads θ̃) → MLP]   (frozen)
                                                                     ▼
                                                     target-column head → p(y | x, D, c)
```

What it does well:

- Uses the backbone's own attention: feature attention inside thinking rows mixes story information across columns, and row attention gives every data cell access to it in every layer.
- s_g gives each column of the thinking rows its own query, so different groups can retrieve different statements.
- Cross-attention cost is 64 · C · L per scene, independent of n, and cacheable for support and query rows.
- W_o = 0 gives the native model at initialization, which makes optimization start safely.

Its limits are in the [stress test](#stress-test).

### B-deep. Per-layer conditioning of the thinking rows

If input-only conditioning is too weak (kill test K2), condition the thinking rows in every layer, prefix-tuning style ([prefix tuning](https://arxiv.org/abs/2101.00190), [P-Tuning v2](https://arxiv.org/abs/2110.07602)). Two ways:

- Add a zero-initialized gated text cross-attention sublayer for the thinking rows only (Flamingo-style, [arXiv 2204.14198](https://arxiv.org/abs/2204.14198)). The thinking rows then see the text and the data in every layer, like a Q-Former / Perceiver bottleneck ([BLIP-2](https://arxiv.org/abs/2301.12597), [Perceiver](https://arxiv.org/abs/2103.03206)).
- Add story-generated residuals to the thinking rows' K/V projections in each block.

The cost stays 64 · C · L per layer, independent of n.

### C. Full text cross-attention for every cell

Add a text cross-attention sublayer for all cells (i, g) in some blocks. Retrieval then depends on row values. The cost is n · C · L per layer, with many more trained parameters and a much larger scene requirement. Do this only after a cheaper channel shows a signal.

### D. Text-derived feature-attention bias

Add a bias b_{gh} = MLP(s_g, s_h, pooled story) to the feature-attention logits, with the last layer initialized to zero. This is the soft version of the graph masks in [Use What You Know](../../related_works/2026-use-what-you-know.md). It directly expresses "X1 and X2 measure the same quantity" or "X5 is unrelated", subject to the group-of-three limit below. It combines well with B.

### E. Late-fusion head — no gradients through TabPFN

Run frozen TabPFN once per episode without gradients and cache its test-row logits ℓ and embeddings h. Train p(y | x, D, c) = softmax(ℓ + g_φ(h, story, schema)) with g_φ initialized to zero. It is very cheap, but the story cannot change how TabPFN uses the support rows. Use it as a quick signal check.

### Not recommended

- **Story as a constant numeric column.** Constant columns are removed before grouping.
- **Story vector added to every cell.** Layer normalization and the pretrained statistics can absorb a uniform shift. It is the FiLM case without column binding.

## Stress test

### Claims that do not hold

1. **Parity at initialization is an optimization convenience, not evidence.** The schema-only control also starts at parity. After training, the adapter can make the no-story case worse. The parity is with the *raw* model, which is weaker than the sklearn wrapper with ensembling and preprocessing. A gain smaller than the ensembling gain has no practical value.
2. **A decreasing story effect (H3) is not a property of the design.** Attention weight follows learned key similarity, not row counts. The thinking rows were pretrained as attention sinks, so they can take most of the attention for any n. H3 must be measured.
3. **Bayesian behaviour comes from the training data, not the architecture.** Any interface trained on jointly sampled (story, data) episodes approximates p(y | x, D, c). The trust in the story is set by the wrong-statement rate of our generator. Fix that rate in advance and report H4 results against it. At small n the model cannot detect a wrong story from the data, so a wrong story will reduce performance there; that is the correct Bayesian behaviour, not a failure of the method.

### Structural risks

1. **Little information in the stories.** With randomly sampled mechanisms, a story communicates structure, not signs or effect sizes. For observational prediction, the useful structure is mostly Markov-blanket membership and redundancy; edge direction barely matters. TabPFN already does implicit feature selection. If scenes have few irrelevant features, even a perfect interface gains little.
2. **Narrow channel in a small frozen model.** Input-only conditioning is soft prompt tuning. Input-only prompts match fine-tuning only for very large language models ([Lester et al.](https://arxiv.org/abs/2104.08691)); small models need per-layer prompts (P-Tuning v2). TabPFN-2.5 has about 11M parameters. Test rows also read the thinking rows through one 64-dim K/V head per layer and column; information can also pass through training rows, which use all 3 heads. On the positive side, [TuneTables](https://arxiv.org/abs/2402.11137) shows that learned context tokens can carry much dataset information into a frozen TabPFN. That was per-dataset optimization with labels, not amortized generation from text.
3. **Groups of three features weaken column binding.** One token holds three features, so s_g mixes three descriptions. A statement like "ignore X2, keep X1 and X3" has no token of its own. Binding holds at group level only. One-feature groups through custom padding are possible but outside the pretraining distribution; they need an ablation.
4. **Gains from prior mismatch.** We do not have Prior Labs' prior generator, so our synthetic scenes differ from TabPFN's pretraining data. An adapter can lower loss by adapting to our scene distribution without using the story. The equally trained schema-only arm controls this and is required. Audit what the schema reveals: a feature description such as "second noisy reading of temperature" already contains part of the story.

Smaller risks:

- all-MiniLM-L6-v2 has a short context window, so long stories are truncated.
- Sentence encoders bind roles weakly ("A causes B" vs "B causes A"), though direction matters less for prediction.
- Adapters trained on template stories may not transfer to LLM-written stories.
- About 100 public scenes, with fewer binary targets, give low statistical power.
- Attention maps are not explanations.

## Kill tests before adapter training

| Test | What it measures | Decision rule |
|---|---|---|
| **K1. Headroom:** native TabPFN-2.5 on all features vs only the true Markov-blanket features, n = 8 … 1024 | The gain available from perfect structural knowledge | Little gap at small n → stop, or change the scenes |
| **K2. Channel capacity:** optimize the 64 × C thinking cells *per scene* with privileged labels on a large set of extra rows; evaluate on different held-out rows. Repeat for column embeddings (design A) and per-layer thinking-row K/V (B-deep). | The upper limit of any story-to-channel mapping, for each channel | Input thinking rows far below the K1 gap, but B-deep close → build B-deep instead of B |
| **K3. Simple baseline:** design 0 (LLM relevance mask) vs its schema-only control | What a training-free story pipeline achieves | The learned adapter must beat this number |

K1 and K3 need no training. K2 needs a few gradient runs per scene; its privileged labels are a diagnostic only and must never enter a reported method. All three need the gated TabPFN-2.5 checkpoint (the user accepts its license) and scenes with binary targets.

## Training data

TabPFN learned p(y | x, D) under a synthetic SCM prior. We want p(y | x, D, c) = ∫ p(y | x, φ) p(φ | D, c) dφ. Training on pairs (c, D) sampled jointly gives a PFN with a prior over (φ, c). This needs many cheap episodes, not 100 CausalDS scenes:

1. Sample an SCM and a binary target from a TabPFN-like prior (or the CausalDS generator).
2. **Verbalize** a controlled subset of graph facts with templates: parents of the target, descendants, noisy measurements of a latent, irrelevant variables, optionally signs or monotonicity. Do not include coefficients or equations.
3. Control the information budget: the number of revealed facts, and a fixed fraction of omitted, irrelevant and *wrong* statements (for H4).
4. Paraphrase a subset with an LLM, so the model does not learn only the template syntax. Keep LLM-written CausalDS-style stories for held-out evaluation.

Use episodic training with support rows as context and the loss on disjoint query rows. Mix support sizes and include episodes without a story. Keep the controls of the [2026-10-21 protocol](2026-10-21-first-context-tabpfn-experiment.md#first-experiment-and-controls): a schema-only arm with the same architecture and budget; omitted, irrelevant and wrong stories; paired scene-level log-loss curves over n.

## Implementation notes for design B

**Size.** With E = 192, a 384-dim MiniLM and a 128-wide attention: W_s 73,728 + W_q 24,576 + W_k 49,152 + W_v 49,152 + W_o 24,576 ≈ **0.22M parameters**, about 2 % of the backbone.

**Why freeze.** Fine-tuning the whole ~11M-parameter backbone costs little extra memory, because the backward pass to the thinking rows already keeps the activations of all 24 layers. The reasons to freeze are data efficiency, keeping the pretrained prior, clean attribution, and exact parity. LoRA on some blocks is a later option if B-deep saturates.

**Memory (rough, ±2×).** Tokens per episode are (64 + support + query) × (groups + 1); one layer keeps about 16 × tokens × E values for backward, in bf16.

| Episode | Tokens | Without checkpointing | With checkpointing |
|---|---|---|---|
| 512 support + 512 query, 20 features | ~8.7k | ~1.3 GB | ~0.1 GB |
| 2048 support + 512 query, 100 features | ~92k | ~13 GB | ~1 GB |

**Pitfalls.**

1. Set `save_peak_memory_factor=None`; use `force_recompute_layer=True` for checkpointing.
2. Compute s_g **after** the constant-column mask. With n = 8, a binary feature is often constant, and all later groups shift.
3. The thinking-row tensor is shared across the batch. Start with one scene per forward (B = 1).
4. The raw module skips the wrapper's ensemble preprocessing (quantile transforms, SVD features, shuffles). The native baseline and the adapter arm must use the same raw path; report wrapper results separately.

**Sketch (not run).**

```python
class StoryThinkingRows(nn.Module):
    """Drop-in replacement for AddThinkingRows; W_o = 0 gives the native rows."""

    def __init__(self, base, d_text=384, d_attn=128, n_heads=4):
        super().__init__()
        self.base = base  # pretrained AddThinkingRows, frozen
        self.num_thinking_rows = base.num_thinking_rows
        E = base.row_token_values_TE.shape[1]
        self.h = n_heads
        self.w_s = nn.Linear(d_text, E)
        self.w_q = nn.Linear(E, d_attn, bias=False)
        self.w_k = nn.Linear(d_text, d_attn, bias=False)
        self.w_v = nn.Linear(d_text, d_attn, bias=False)
        self.w_o = nn.Linear(d_attn, E)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)
        self.ctx = None  # (text_LD, cols_CD), set per scene; C = groups + target

    def forward(self, x_BRiCE, single_eval_pos):
        text_LD, cols_CD = self.ctx
        theta_TE = self.base.row_token_values_TE
        T, E = theta_TE.shape
        C = x_BRiCE.shape[2]
        assert cols_CD.shape[0] == C, "column semantics misaligned with groups"

        u_TCE = theta_TE[:, None] + self.w_s(cols_CD)[None]
        q = self.w_q(F.layer_norm(u_TCE, (E,))).reshape(T * C, self.h, -1).transpose(0, 1)
        k = self.w_k(text_LD).reshape(-1, self.h, q.shape[-1]).transpose(0, 1)
        v = self.w_v(text_LD).reshape(-1, self.h, q.shape[-1]).transpose(0, 1)
        a = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(T, C, -1)

        rows_TCE = theta_TE[:, None] + self.w_o(a)
        rows_BTCE = rows_TCE[None].expand(x_BRiCE.shape[0], -1, -1, -1).to(x_BRiCE.dtype)
        return torch.cat([rows_BTCE, x_BRiCE], dim=1), single_eval_pos + T
```

Install it with `model.add_thinking_rows = StoryThinkingRows(model.add_thinking_rows)`. Call `model.requires_grad_(False)`, then enable gradients for the adapter only.

**Checks before training.**

1. Parity: with W_o = 0, outputs equal the unpatched model's, apart from the dtype cast.
2. Gradients: finite, nonzero adapter gradients (W_o first); no backbone parameter has `.grad`.
3. Alignment: a test with a constant column confirms that s_g matches the right groups.
4. Label isolation: changing query labels changes only the loss, never the logits.

## Recommended order

1. K1 headroom and K3 simple baseline (no training).
2. K2 channel capacity for input thinking rows, column embeddings and per-layer thinking-row K/V.
3. Text-conditioned synthetic prior generator (templates, fixed wrong-statement rate, LLM paraphrases).
4. The learned channel that K2 favours (B or B-deep), with its schema-only twin, design A and design E as controls. Compare against the K3 baseline.
5. D, LoRA, or design C only after a clear signal.

## Model choice

Use **TabPFN-2.5** because of the thinking rows. With TabPFN v2 (no thinking rows), you can add new learned rows, but you lose exact parity because the softmax normalization changes. The v2.5 license is the [TABPFN-2.5 License v1.0](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE); check that it permits our research use and derived adapters. TabPFN v1 has only row tokens. The same pseudo-row idea works there, but without column binding (see [the v1 follow-up](2026-10-21-first-context-tabpfn-experiment.md#follow-up-research-2026-10-21-tabpfn-v1)).

## Confidence Notes

These designs are our proposals; no model was built, trained or evaluated. Facts about TabPFN-2.5 come from the [technical report](https://arxiv.org/abs/2511.08667) (v2) and the current `main` source; the checkpoint config was not loaded. Parameter and memory numbers are estimates from the default config. The report's architecture section gives the motivation for thinking rows but no ablation was found there. I did not search systematically for text-conditioned register or thinking tokens in tabular PFNs, so no novelty claim is made. TabPFN-2.5 and TuneTables are not yet records in the [related-works catalogue](../../related_works/README.md).

## Open Questions

- Does oracle structural knowledge help TabPFN-2.5 at small n on our scenes (K1)?
- Is input-layer conditioning of the thinking rows enough, or is per-layer conditioning needed (K2)?
- How much of the K1 gap does a training-free LLM relevance mask recover (K3)?
- Does a small frozen sentence encoder keep relational statements, or do we need the hidden states of a decoder LLM?
- Do models trained on template stories transfer to LLM-written stories?
