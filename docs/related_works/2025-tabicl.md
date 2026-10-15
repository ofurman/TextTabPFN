---
id: "2025-tabicl"
title: "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data"
year_or_venue: "ICML 2025"
tags: ["no-text"]
architecture_tags: ["tabular-transformer", "tabular-inducing-attention", "tabular-feature-attention", "tabular-row-pooling", "tabular-row-attention"]
brief: "Alternative tabular foundation model; compare text benefits across architectures."
source_url: "https://arxiv.org/abs/2502.05564"
pdf: "raw/2025_tabicl.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabICL: A Tabular Foundation Model for In-Context Learning on Large Data

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2502.05564) · [PDF](raw/2025_tabicl.pdf)

## Text-source scope

Distribution-aware embeddings infer structure from values, not natural-language metadata.

## Problem and positioning

Scale tabular in-context learning without repeatedly processing the full cell grid.

## Architecture

**Block roles:** Tabular: column set-transformer blocks use inducing points; within-row feature attention and CLS pooling compress rows before dataset-level attention. Text: no text transformer is present.

Column-distribution-aware embedding, within-row attention that compresses features into a fixed-size row representation, then a dataset-level transformer over labeled and query row representations. Pretrained on synthetic tasks.

## Input

Encoded numerical/categorical tables, training labels, and query rows.

## Output

Class probabilities; the original paper focuses on classification.

## For us

An alternative backbone for testing whether text benefits generalize. Inserting text before row compression is our proposed extension, not this paper's contribution.

## Related works

- [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](2026-tabicl-v2.md)
- [TabPFN-3: Technical Report](2026-tabpfn-3.md)
