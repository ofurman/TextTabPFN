---
id: "2026-tabicl-v2"
title: "TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model"
year_or_venue: "ICML 2026"
tags: ["no-text"]
architecture_tags: ["tabular-transformer", "tabular-inducing-attention", "tabular-feature-attention", "tabular-row-pooling", "tabular-row-attention"]
brief: "Updated open TabICL classification/regression baseline."
source_url: "https://arxiv.org/abs/2602.11139"
pdf: "raw/2026_tabicl_v2.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2602.11139) · [PDF](raw/2026_tabicl_v2.pdf)

## Text-source scope

Early target injection uses observed labels, not natural-language label meanings.

## Problem and positioning

Improve TabICL's accuracy, scalability, and support for classification and regression.

## Architecture

**Block roles:** Tabular: distribution embedding, feature interaction, CLS row compression, and dataset-level ICL are tabular transformer stages. Text: no text transformer is present; early label injection is not language encoding.

Retains column embedding, row compression, and dataset-level ICL. Adds feature grouping, early training-target injection, query-aware attention scaling for longer contexts, and broader synthetic pretraining.

## Input

Labeled tables and query rows; observed training labels also inform representations before compression.

## Output

Classification probabilities or regression predictions.

## For us

Suggests that compression can be conditioned on the prediction task. Its target-aware embeddings use observed labels, which differs from encoding natural-language target descriptions.

## Related works

- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](2025-tabicl.md)
- [TabPFN-3: Technical Report](2026-tabpfn-3.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
