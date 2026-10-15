---
id: "2025-contexttab"
title: "ConTextTab: A Semantics-Aware Tabular In-Context Learner"
year_or_venue: "2025"
tags: ["row-level-text", "column-label-semantics"]
architecture_tags: ["text-transformer-encoder", "modality-projector", "tabular-transformer", "tabular-feature-attention", "tabular-row-attention"]
brief: "Semantic representations integrated with tabular in-context learning."
source_url: "https://arxiv.org/abs/2506.10707"
pdf: "raw/2025_contexttab.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# ConTextTab: A Semantics-Aware Tabular In-Context Learner

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2506.10707) · [PDF](raw/2025_contexttab.pdf)

## Text-source scope

Uses text cells, categorical string meanings, and column headers.

## Problem and positioning

Preserve meaning lost when categorical strings become arbitrary IDs and column names are ignored.

## Architecture

**Block roles:** Text: frozen MiniLM encodes cells and headers. Interface: learned linear projections map language embeddings into cell space. Tabular: a separate table-native backbone alternates feature and row attention.

Modality-specific number, date, and text encodings. A frozen sentence encoder represents text cells and column headers; learned projections map them into the model space. A TabPFN-like backbone alternates cross-feature and cross-row attention and is pretrained on real tables.

## Input

Mixed-type tables with meaningful headers, labeled context rows, and query rows.

## Output

Classification or regression predictions.

## For us

A strong precedent for combining semantic metadata with table-native ICL, beyond adding free-text embeddings.

## Related works

- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
- [CARTE: Pretraining and Transfer for Tabular Learning](2024-carte.md)
- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
