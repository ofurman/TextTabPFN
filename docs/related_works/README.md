# Related works wiki

A catalogue of papers on text and tabular foundation models. Each paper has one linked Markdown note covering its problem and positioning, architecture, input, output, and relevance to this project. Notes also state text-source scope and link to related papers. This is an initial literature briefing, not a systematic review or a reproduction of reported results.

## Maintaining the wiki

Follow the [mandatory rules in AGENTS.md](../../AGENTS.md#related-works-maintenance-mandatory). Keep the catalogue, individual notes, and [PDF manifest](manifest.json) synchronized. Downloaded papers live in [raw/](raw/); notes link directly to them. Update an existing note for a new paper revision.

Run from the repository root:

```sh
./scripts/lint_related_works.sh
uv run --no-project python -m unittest discover -s tests -v
```

The linter requires Python 3.10+ and Poppler's `pdfinfo` on PATH. It checks note metadata and sections, catalogue consistency, tags, wiki links, repository Markdown links and anchors, and PDF provenance/checksums/page counts. It reports failures and exits nonzero; it does not rewrite content or fetch sources. Scientific accuracy and tag interpretation still require human review.

## Text-source tags

Tag definitions and assignment rules are maintained in [AGENTS.md](../../AGENTS.md#related-works-maintenance-mandatory):

- `row-level-text` — independently observed text associated with an example.
- `column-label-semantics` — meaningful feature names, categorical strings, or target names.
- `task-domain-context` — task instructions, table/domain descriptions, or causal stories.
- `no-text` — no semantic text input; used alone.

Multiple text tags can apply. Variant-specific scope and classification rationale appear in the notes. Serializing a table into sentences does not by itself make it row-level text.

## Architecture block tags

The catalogue keeps architecture blocks separate from text-source tags. `tabular-transformer` denotes a backbone operating on table features/rows; `text-transformer-encoder`, `text-transformer-decoder`, and `text-transformer-encoder-decoder` distinguish language-model architectures processing text tokens. Attention and pooling tags prefixed `tabular-` describe the table backbone, not the language model.

Other tags identify modality projectors, gated MLPs, cross-attention poolers, graph attention, target tokens, retrieval heads, and modality encoders. `class-token-logit-filter` identifies classification by selecting existing language-model vocabulary logits, distinct from interacting target tokens. `static-text-embeddings` identifies non-transformer language features. `pca` is a preprocessing component, `tree-predictor` a downstream model, and `benchmark-generator` identifies a work without a proposed predictive model. Full definitions are in the [canonical rules](../../AGENTS.md#architecture-block-tags).

Each note's **Block roles** paragraph identifies the text, tabular, and interface components and explains variant scope. Architecture tags describe the main studied pipeline; comparison baselines are not included, and undisclosed text internals are not guessed.

`feature-code-generator` identifies code-based feature engineering rather than latent fusion. `tabular-structural-attention-bias`, `graph-convolution`, and `adaptive-layer-norm` identify structured-knowledge conditioning; they do not imply a language-model block.

## Catalogue

| Paper | Year / venue | Text-source tags | Architecture blocks | Brief |
|---|---|---|---|---|
| [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md) | 2026 | `row-level-text`, `column-label-semantics` | `text-transformer-encoder`, `modality-projector`, `tabular-transformer`, `tabular-feature-attention`, `tabular-row-attention` | Direct overlap: frozen text encoder and TabPFN connected by a trained adapter. |
| [MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](2026-multimodalpfn.md) | 2026 | `row-level-text` | `text-transformer-encoder`, `vision-encoder`, `modality-projector`, `gated-mlp`, `cross-attention-pooler`, `tabular-transformer`, `tabular-feature-attention`, `tabular-row-attention` | Direct multimodal PFN architecture for combining text/image representations with tabular features. |
| [TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings](2026-tabpfn-beyond-tabular-data.md) | 2026 | `row-level-text` | `text-transformer-encoder`, `vision-encoder`, `audio-encoder`, `pca`, `tabular-transformer`, `tabular-feature-attention`, `tabular-row-attention` | Frozen multimodal embeddings with a TabPFN prediction head; relevant accuracy and calibration baselines. |
| [TabPFN-3: Technical Report](2026-tabpfn-3.md) | 2026 | `row-level-text` | `tabular-transformer`, `tabular-inducing-attention`, `tabular-feature-attention`, `tabular-row-pooling`, `tabular-row-attention`, `retrieval-head` | Current TabPFN generation; includes tabular-text results and a TabPFN-3-Plus variant. |
| [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md) | 2025 | `row-level-text`, `column-label-semantics` | `text-transformer-encoder`, `tabular-transformer`, `tabular-feature-attention`, `target-tokens` | Text-tabular foundation model with task-aware text representations; competitor and benchmark source. |
| [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md) | 2025 | `row-level-text`, `column-label-semantics` | `text-transformer-encoder`, `modality-projector`, `tabular-transformer`, `tabular-feature-attention`, `tabular-row-attention` | Semantic representations integrated with tabular in-context learning. |
| [CARTE: Pretraining and Transfer for Tabular Learning](2024-carte.md) | 2024 | `column-label-semantics` | `static-text-embeddings`, `graph-attention` | Embeds string values and column names in graph representations; transfer across different schemas. |
| [TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling](2022-tabtext.md) | 2022 preprint; revised 2025 | `column-label-semantics`, `task-domain-context` | `text-transformer-encoder`, `tree-predictor` | Early language representation approach for augmenting structured health data. |
| [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md) | 2023; preprint 2022 | `column-label-semantics`, `task-domain-context` | `text-transformer-encoder-decoder` | Low-label baseline using natural-language serialization and language-model knowledge. |
| [SALSA: Single-pass Autoregressive LLM Structured Classification](2025-salsa.md) | 2025 preprint | `row-level-text`, `column-label-semantics`, `task-domain-context` | `text-transformer-decoder`, `class-token-logit-filter` | Decoder-only LLM classifier using structured prompts, class-token logits, and LoRA; a text-only baseline for fusion. |
| [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](2023-tabpfn.md) | ICLR 2023; preprint 2022 | `no-text` | `tabular-transformer`, `tabular-row-attention` | Original prior-data fitted network and small-data tabular classification foundation. |
| [Accurate predictions on small data with a tabular foundation model](2025-tabpfn-v2.md) | Nature 2025 | `no-text` | `tabular-transformer`, `tabular-row-attention`, `tabular-feature-attention` | TabPFNv2 classification and regression baseline for small tabular datasets. |
| [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](2025-tabicl.md) | ICML 2025 | `no-text` | `tabular-transformer`, `tabular-inducing-attention`, `tabular-feature-attention`, `tabular-row-pooling`, `tabular-row-attention` | Alternative tabular foundation model; compare text benefits across architectures. |
| [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](2026-tabicl-v2.md) | ICML 2026 | `no-text` | `tabular-transformer`, `tabular-inducing-attention`, `tabular-feature-attention`, `tabular-row-pooling`, `tabular-row-attention` | Updated open TabICL classification/regression baseline. |
| [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](2026-causalds.md) | 2026 | `task-domain-context` | `benchmark-generator` | User-requested paper: tables plus natural-language causal stories. Adjacent evaluation work, rather than a TabPFN extension. |
| [Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering](2023-caafe.md) | 2023 | `column-label-semantics`, `task-domain-context` | `feature-code-generator`, `tabular-transformer`, `tabular-row-attention`, `tree-predictor` | Dataset descriptions drive LLM-generated feature code selected with TabPFN validation; task-context baseline. |
| [Use What You Know: Causal Foundation Models with Partial Graphs](2026-use-what-you-know.md) | ICML 2026 | `no-text` | `tabular-transformer`, `tabular-feature-attention`, `tabular-row-attention`, `tabular-structural-attention-bias`, `graph-convolution`, `adaptive-layer-norm` | Partial ancestral graphs condition a causal PFN through feature-attention biases and GCN/AdaLN; structural-prior precedent. |

## Suggested reading order

- **Task-level and structural conditioning:** [CAAFE](2023-caafe.md), [Use What You Know](2026-use-what-you-know.md).
- **Closest text–TabPFN approaches:** [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md), [MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](2026-multimodalpfn.md).
- **Semantic and text-tabular competitors:** [TabPFN-3: Technical Report](2026-tabpfn-3.md), [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md), [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md).
- **Embedding and few-label baselines:** [TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings](2026-tabpfn-beyond-tabular-data.md), [TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling](2022-tabtext.md), [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md).
- **Tabular foundations:** [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](2023-tabpfn.md), [Accurate predictions on small data with a tabular foundation model](2025-tabpfn-v2.md), [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](2025-tabicl.md), [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](2026-tabicl-v2.md).
- **Semantic transfer and context-sensitive evaluation:** [CARTE: Pretraining and Transfer for Tabular Learning](2024-carte.md), [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](2026-causalds.md).

## Problem and project positioning

Combining text with TabPFN already has direct precedents. Our proposed focus is: **Can text-conditioned representations improve tabular in-context prediction under a small label budget, while preserving performance when text is unhelpful?** This is a research question, not an established novelty claim.

Distinguish three sources of text:

- **Row-level text:** reviews, notes, or descriptions that supply additional observations.
- **Column and label semantics:** names such as “annual income” or “default” that explain the meaning of values and targets.
- **Task or domain context:** descriptions and assumptions that may supply information absent from the table.

Also distinguish fixed-weight in-context learning, which conditions on labeled rows, from downstream fine-tuning, which updates weights. Their label and computation requirements differ.

## Implications for our first study

The following are proposed experiments, not established gaps or findings:

| Question | Closest references |
|---|---|
| Does a learned token adapter beat embedding concatenation/PCA? | Text Adapter; TabPFN beyond Tabular Data |
| Does task-aware text encoding help with very few labels? | TabSTAR; TabICLv2's target-aware compression |
| Can text help without fine-tuning the tabular backbone? | Text Adapter versus MultiModalPFN |
| Do gains come from extra observations or semantic metadata? | ConTextTab; TabText; CARTE |
| Can the model ignore irrelevant or misleading text? | Robustness question to test explicitly |

A controlled small-label study should separate these effects with shared splits and model-selection budgets. Any new architecture should address a demonstrated failure mode. The collection alone does not establish a novel gap.

## Provenance

PDFs were downloaded from arXiv or the publisher. [manifest.json](manifest.json) records note/PDF paths, primary sources, download URLs, identified arXiv versions, retrieval dates, page counts, sizes, and SHA-256 hashes. Years in filenames denote the initial preprint or publication year; PDFs can contain later revisions. Papers remain subject to their original licenses.

See the dated [research synthesis](../research/web/2026-10-15-text-tabular-foundation-models.md) for the original collection findings and open questions. Individual paper notes are the maintained records.
