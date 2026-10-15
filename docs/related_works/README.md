# Related works

Updated: 2026-10-15. This is the canonical collection index and briefing for the papers in this folder. It covers each work's problem and positioning, architecture, input, output, and relevance to this project. This is an initial literature briefing, not a systematic review or a reproduction of reported results.

## Maintenance

All additions and revisions must follow the [mandatory related-work rules in AGENTS.md](../../AGENTS.md#related-works-maintenance-mandatory). Those rules are the single source of maintenance requirements for this repository. Update this README and, when PDFs change, [manifest.json](manifest.json) in the same change. Keep all work records here; do not create a separate brief or a second rule set.

Years in filenames denote the initial preprint or publication year; PDFs can contain later revisions.

## Text-source tags

Tags identify the text actually used by the described method or benchmark, not possible future extensions. Multiple tags can apply. The [canonical tagging rules](../../AGENTS.md#related-works-maintenance-mandatory) define the classification requirements.

- `row-level-text`: independently observed text belonging to an example, such as a review or note.
- `column-label-semantics`: meaningful feature names, categorical strings, or target/class names.
- `task-domain-context`: explicit task descriptions, table/domain descriptions, or causal stories.
- `no-text`: a foundation work with no semantic text input.

Serializing structured values into sentences does not by itself qualify as `row-level-text`. Tags describe input use, not whether text adds information or improves results. Variant-specific scope is stated in the work record.

## Paper collection

| Paper | Year / venue | Local PDF | Primary source | Tags | Relevance |
|---|---|---|---|---|---|
| Towards Pretraining Text Encoders for TabPFN | 2026 | [PDF](2026_text_encoders_for_tabpfn.pdf) | [Source](https://arxiv.org/abs/2606.04876) | `row-level-text`, `column-label-semantics` | Direct overlap: frozen text encoder and TabPFN connected by a trained adapter. |
| MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning | 2026 | [PDF](2026_multimodalpfn.pdf) | [Source](https://arxiv.org/abs/2602.20223) | `row-level-text` | Direct multimodal PFN architecture for combining text/image representations with tabular features. |
| TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings | 2026 | [PDF](2026_tabpfn_beyond_tabular_data.pdf) | [Source](https://arxiv.org/abs/2607.11007) | `row-level-text` | Frozen multimodal embeddings with a TabPFN prediction head; relevant accuracy and calibration baselines. |
| TabPFN-3: Technical Report | 2026 | [PDF](2026_tabpfn_3.pdf) | [Source](https://arxiv.org/abs/2605.13986) | `row-level-text` | Current TabPFN generation; includes tabular-text results and a TabPFN-3-Plus variant. |
| TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields | 2025 | [PDF](2025_tabstar.pdf) | [Source](https://arxiv.org/abs/2505.18125) | `row-level-text`, `column-label-semantics` | Text-tabular foundation model with task-aware text representations; competitor and benchmark source. |
| ConTextTab: A Semantics-Aware Tabular In-Context Learner | 2025 | [PDF](2025_contexttab.pdf) | [Source](https://arxiv.org/abs/2506.10707) | `row-level-text`, `column-label-semantics` | Semantic representations integrated with tabular in-context learning. |
| CARTE: Pretraining and Transfer for Tabular Learning | 2024 | [PDF](2024_carte.pdf) | [Source](https://arxiv.org/abs/2402.16785) | `column-label-semantics` | Embeds string values and column names in graph representations; transfer across different schemas. |
| TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling | 2022 preprint; revised 2025 | [PDF](2022_tabtext.pdf) | [Source](https://arxiv.org/abs/2206.10381) | `column-label-semantics`, `task-domain-context` | Early language representation approach for augmenting structured health data. |
| TabLLM: Few-shot Classification of Tabular Data with Large Language Models | 2023; preprint 2022 | [PDF](2023_tabllm.pdf) | [Source](https://arxiv.org/abs/2210.10723) | `column-label-semantics`, `task-domain-context` | Low-label baseline using natural-language serialization and language-model knowledge. |
| TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second | ICLR 2023; preprint 2022 | [PDF](2023_tabpfn.pdf) | [Source](https://arxiv.org/abs/2207.01848) | `no-text` | Original prior-data fitted network and small-data tabular classification foundation. |
| Accurate predictions on small data with a tabular foundation model | Nature 2025 | [PDF](2025_tabpfn_v2.pdf) | [Source](https://www.nature.com/articles/s41586-024-08328-6) | `no-text` | TabPFNv2 classification and regression baseline for small tabular datasets. |
| TabICL: A Tabular Foundation Model for In-Context Learning on Large Data | ICML 2025 | [PDF](2025_tabicl.pdf) | [Source](https://arxiv.org/abs/2502.05564) | `no-text` | Alternative tabular foundation model; compare text benefits across architectures. |
| TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model | ICML 2026 | [PDF](2026_tabicl_v2.pdf) | [Source](https://arxiv.org/abs/2602.11139) | `no-text` | Updated open TabICL classification/regression baseline. |
| CausalDS: Benchmarking Causal Reasoning in Data-Science Agents | 2026 | [PDF](2026_causalds.pdf) | [Source](https://arxiv.org/abs/2607.08093) | `task-domain-context` | User-requested paper: tables plus natural-language causal stories. Adjacent evaluation work, rather than a TabPFN extension. |

## Suggested reading order

1. **Towards Pretraining Text Encoders for TabPFN** and **MultiModalPFN**: establish the closest existing approaches.
2. **TabPFN-3**, **TabSTAR**, and **ConTextTab**: establish current semantic and text-tabular competitors.
3. **TabPFN beyond Tabular Data**, **TabText**, and **TabLLM**: choose embedding and low-label baselines.
4. **TabPFN / TabPFNv2** and **TabICL / TabICLv2**: understand the underlying tabular models.
5. **CARTE** for semantic transfer; **CausalDS** for context-sensitive evaluation inspiration.

## Problem and project positioning

Combining text with TabPFN already has direct precedents. Our proposed focus is: **Can text-conditioned representations improve tabular in-context prediction under a small label budget, while preserving performance when text is unhelpful?** This is a research question, not an established novelty claim.

Distinguish three sources of text:

- **Row-level text:** reviews, notes, or descriptions that supply additional observations.
- **Column and label semantics:** names such as “annual income” or “default” that explain the meaning of values and targets.
- **Task or domain context:** descriptions and assumptions that may supply information absent from the table.

Also distinguish fixed-weight in-context learning, which conditions on labeled rows, from downstream fine-tuning, which updates weights. Their label and computation requirements differ.

## Tabular foundations

### 1. Original TabPFN

[Primary source](https://arxiv.org/abs/2207.01848) · [PDF](2023_tabpfn.pdf)

- **Tags:** `no-text`. Numerical foundation; no semantic text input.
- **Problem and positioning:** Learn a reusable prediction algorithm for small tabular datasets rather than train a new model from scratch for each dataset.
- **Architecture:** A transformer embeds entire rows and attends over labeled training examples. Offline pretraining on synthetic tasks teaches approximate Bayesian prediction under the synthetic dataset prior; ordinary downstream inference keeps weights fixed.
- **Input:** Labeled numerical training rows and unlabeled query rows. The original work targets roughly 1,000 training examples, 100 numerical features, and 10 classes.
- **Output:** Class probabilities for query rows.
- **For us:** Establishes the small-data, labeled-context mechanism to preserve. It does not natively understand column names or free text.

### 2. TabPFNv2 — Accurate predictions on small data with a tabular foundation model

[Primary source](https://www.nature.com/articles/s41586-024-08328-6) · [PDF](2025_tabpfn_v2.pdf)

- **Tags:** `no-text`. Categorical IDs and observed labels do not constitute semantic text input.
- **Problem and positioning:** Extend TabPFN to stronger real-world classification and regression with categorical features and missing values.
- **Architecture:** Cell/feature representations with alternating attention across features within a row and across examples. Pretrained on a richer synthetic prior, then used for fixed-weight in-context prediction.
- **Input:** Labeled mixed-type tables and query rows.
- **Output:** Class probabilities or a predictive distribution for a numerical target, from which means and quantiles can be derived.
- **For us:** Its internal feature-token space is the insertion point for several text extensions. Synthetic causal pretraining does not turn ordinary supervised predictions into causal-effect estimates.

### 3. TabICL

[Primary source](https://arxiv.org/abs/2502.05564) · [PDF](2025_tabicl.pdf)

- **Tags:** `no-text`. Distribution-aware embeddings infer structure from values, not natural-language metadata.
- **Problem and positioning:** Scale tabular in-context learning without repeatedly processing the full cell grid.
- **Architecture:** Column-distribution-aware embedding, within-row attention that compresses features into a fixed-size row representation, then a dataset-level transformer over labeled and query row representations. Pretrained on synthetic tasks.
- **Input:** Encoded numerical/categorical tables, training labels, and query rows.
- **Output:** Class probabilities; the original paper focuses on classification.
- **For us:** An alternative backbone for testing whether text benefits generalize. Inserting text before row compression is our proposed extension, not this paper's contribution.

### 4. TabICLv2

[Primary source](https://arxiv.org/abs/2602.11139) · [PDF](2026_tabicl_v2.pdf)

- **Tags:** `no-text`. Early target injection uses observed labels, not natural-language label meanings.
- **Problem and positioning:** Improve TabICL's accuracy, scalability, and support for classification and regression.
- **Architecture:** Retains column embedding, row compression, and dataset-level ICL. Adds feature grouping, early training-target injection, query-aware attention scaling for longer contexts, and broader synthetic pretraining.
- **Input:** Labeled tables and query rows; observed training labels also inform representations before compression.
- **Output:** Classification probabilities or regression predictions.
- **For us:** Suggests that compression can be conditioned on the prediction task. Its target-aware embeddings use observed labels, which differs from encoding natural-language target descriptions.

### 5. TabPFN-3

[Primary source](https://arxiv.org/abs/2605.13986) · [PDF](2026_tabpfn_3.pdf)

- **Tags:** `row-level-text`. Applies to the text-capable Plus variant; the base model is a tabular foundation.
- **Problem and positioning:** Scale strong tabular prediction to substantially larger datasets.
- **Architecture:** A TabICL-style pipeline: column-distribution embedding, feature aggregation into row embeddings, then dataset-level ICL. Classification uses an attention-based retrieval decoder over labeled training representations. The report separately introduces text-capable TabPFN-3-Plus; its text internals are less fully disclosed than the base architecture.
- **Input:** Tables and training labels. The Plus variant additionally accepts text columns directly.
- **Output:** Classification probabilities or regression predictions.
- **For us:** A relevant modern baseline and existing text-tabular competitor. Distinguish the base model from Plus when reporting comparisons.

## Direct text and multimodal TabPFN approaches

### 6. Towards Pretraining Text Encoders for TabPFN

[Primary source](https://arxiv.org/abs/2606.04876) · [PDF](2026_text_encoders_for_tabpfn.pdf)

- **Tags:** `row-level-text`, `column-label-semantics`. Encodes free-text cells with their column names.
- **Problem and positioning:** Avoid the information bottleneck from reducing large language embeddings with PCA before treating them as scalar TabPFN features.
- **Architecture:** Text cell plus column name → frozen sentence encoder → learned adapter → a short sequence of TabPFN feature tokens. Structured features retain the standard path. The sentence encoder and TabPFN remain frozen while the adapter is pretrained across text-containing datasets.
- **Input:** Labeled mixed text/tabular rows and query rows.
- **Output:** Class probabilities or regression predictions.
- **For us:** The closest precedent for lightweight text support. Reported regression results are strong, but classification does not uniformly beat alternatives. A new adapter needs a specific advantage beyond the broad integration idea.

### 7. MultiModalPFN

[Primary source](https://arxiv.org/abs/2602.20223) · [PDF](2026_multimodalpfn.pdf)

- **Tags:** `row-level-text`. Text fields are observations associated with individual examples.
- **Problem and positioning:** Combine structured features with text and images when labeled multimodal data is limited.
- **Architecture:** Frozen modality encoders feed a multi-head gated MLP that expands representations into candidate tokens. A cross-attention pooler reduces these to a controlled number of tokens, which join tabular tokens inside TabPFN. Downstream training updates the projector, TabPFN backbone, and prediction head while modality encoders remain frozen.
- **Input:** Labeled rows containing structured features and text/images, plus query rows.
- **Output:** Class probabilities in the reported experiments, including tasks formed by discretizing numerical targets.
- **For us:** Direct architectural competition. Token-count imbalance can let text/image tokens dominate attention and suppress structured signals; fusion should control this balance.

### 8. TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings

[Primary source](https://arxiv.org/abs/2607.11007) · [PDF](2026_tabpfn_beyond_tabular_data.pdf)

- **Tags:** `row-level-text`. Text examples are encoded into embeddings; this does not imply joint text/table fusion.
- **Problem and positioning:** Test whether TabPFN is an accurate, well-calibrated prediction head for pretrained representations outside conventional tables.
- **Architecture:** Text/image/audio → frozen encoder → standardization and PCA → TabPFN. The main frozen-encoder pipeline requires no gradient updates; separate experiments examine adapted encoders.
- **Input:** Labeled encoder embeddings and query embeddings.
- **Output:** Class probabilities, evaluated for accuracy and calibration.
- **For us:** A simple embedding baseline rather than a dedicated text-plus-table fusion model. The reported accuracy advantage concentrates at moderate-to-higher shot counts and smaller feature dimensions; very few labels do not automatically favor TabPFN on embeddings.

## Semantic and text-tabular alternatives

### 9. ConTextTab

[Primary source](https://arxiv.org/abs/2506.10707) · [PDF](2025_contexttab.pdf)

- **Tags:** `row-level-text`, `column-label-semantics`. Uses text cells, categorical string meanings, and column headers.
- **Problem and positioning:** Preserve meaning lost when categorical strings become arbitrary IDs and column names are ignored.
- **Architecture:** Modality-specific number, date, and text encodings. A frozen sentence encoder represents text cells and column headers; learned projections map them into the model space. A TabPFN-like backbone alternates cross-feature and cross-row attention and is pretrained on real tables.
- **Input:** Mixed-type tables with meaningful headers, labeled context rows, and query rows.
- **Output:** Classification or regression predictions.
- **For us:** A strong precedent for combining semantic metadata with table-native ICL, beyond adding free-text embeddings.

### 10. TabSTAR

[Primary source](https://arxiv.org/abs/2505.18125) · [PDF](2025_tabstar.pdf)

- **Tags:** `row-level-text`, `column-label-semantics`. Uses text fields, feature names, and candidate target names; target names are label semantics.
- **Problem and positioning:** Make text representations useful for the particular prediction target rather than rely on generic frozen embeddings.
- **Architecture:** Encodes each column-name/value pair; numerical features also retain precise numerical representations. Feature-level fusion precedes cross-feature attention. Target tokens representing candidate classes interact with features and use a shared prediction head. Cross-dataset pretraining is followed by downstream fine-tuning, including adaptation of the text encoder.
- **Input:** Rows, feature names and values, and target descriptions or candidate class names. Candidate classes are provided for every example; the true query label remains hidden.
- **Output:** Class probabilities or a regression value.
- **For us:** A strong task-aware competitor. It transfers through pretrained weights and downstream adaptation rather than TabPFN-style labeled-context inference.

### 11. CARTE

[Primary source](https://arxiv.org/abs/2402.16785) · [PDF](2024_carte.pdf)

- **Tags:** `column-label-semantics`. Focuses on column names and categorical/entity strings rather than document-like text.
- **Problem and positioning:** Transfer knowledge between tables with different columns and schemas.
- **Architecture:** Converts each row into a star-shaped graph: values become leaf nodes, column names describe edges, and a central node aggregates the row. Language embeddings initialize semantic information; graph attention contextualizes it. Knowledge-graph pretraining is followed by downstream fine-tuning.
- **Input:** Mixed-type rows with column names, plus labels for downstream adaptation.
- **Output:** Classification or regression predictions through a learned row representation.
- **For us:** Demonstrates semantic transfer across unmatched schemas. More directly relevant to names and categorical strings than to rich document-length text.

### 12. TabText

[Primary source](https://arxiv.org/abs/2206.10381) · [PDF](2022_tabtext.pdf)

- **Tags:** `column-label-semantics`, `task-domain-context`. Uses variable names and table descriptions. Its generated row paragraphs are serialized table values, not independently observed free text.
- **Problem and positioning:** Build reusable representations of heterogeneous clinical tables with less manual preprocessing.
- **Architecture:** Verbalizes structured patient information into a paragraph, embeds it with a clinical language model, and feeds the representation to a conventional predictor. Embeddings can augment original structured features. The saved revision includes self-supervised language-model adaptation before embedding extraction.
- **Input:** Clinical tables and metadata; downstream predictors additionally require training labels.
- **Output:** Reusable embeddings, followed by downstream predictions such as clinical classification outcomes.
- **For us:** A practical feature-augmentation baseline. Text is largely constructed from the table, so gains may reflect representation and prior knowledge rather than new observations.

### 13. TabLLM

[Primary source](https://arxiv.org/abs/2210.10723) · [PDF](2023_tabllm.pdf)

- **Tags:** `column-label-semantics`, `task-domain-context`. Uses feature names and an explicit classification-task description. Serialized rows are not independently observed free text.
- **Problem and positioning:** Use language-model knowledge for tabular classification when very few labels are available.
- **Architecture:** Serializes each row into natural language and supplies a task description to an instruction-trained language model. Answer tokens map to classes. Studies zero-shot prediction and parameter-efficient few-shot fine-tuning.
- **Input:** A verbalized row and classification question; labeled examples are used for few-shot adaptation.
- **Output:** Class-label scores or predictions.
- **For us:** Tests a different route: let the language model predict directly. Useful for checking whether semantic knowledge alone explains benefits attributed to text–TabPFN integration.

## Context-sensitive evaluation

### 14. CausalDS: Benchmarking Causal Reasoning in Data-Science Agents

[Primary source](https://arxiv.org/abs/2607.08093) · [PDF](2026_causalds.pdf)

- **Tags:** `task-domain-context`. The domain story describes the data-generating setting; it is not a per-row text feature.
- **Problem and positioning:** Evaluate agents that must combine domain descriptions, observational data, causal reasoning, uncertainty, and computation.
- **Architecture:** A benchmark generator rather than a predictive model. Samples structural causal models, generates observational tables and corresponding natural-language stories, and derives tasks with known ground truth and deterministic scoring.
- **Input:** An observational table, domain story, and task question supplied to an evaluated agent.
- **Output:** Task-dependent estimates, predictions, causal answers, uncertainty statements, or abstention when the target is not identifiable.
- **For us:** Inspiration for context that supplies assumptions rather than another feature column. Adapting it would move the project toward causal reasoning and require objectives beyond ordinary supervised prediction.

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

PDFs were downloaded from arXiv or the publisher. All 14 files passed PDF parsing, and first-page titles were checked. [manifest.json](manifest.json) records source URLs, identified arXiv versions, sizes, page counts, and SHA-256 hashes. Unidentified version fields are null. Papers remain subject to their original licenses.

See the [research synthesis](../research/web/2026-10-15-text-tabular-foundation-models.md) for implications and open research questions.
