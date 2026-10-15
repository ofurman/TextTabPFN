---
id: "2026-text-encoders-for-tabpfn"
title: "Towards Pretraining Text Encoders for TabPFN"
year_or_venue: "2026"
tags: ["row-level-text", "column-label-semantics"]
architecture_tags: ["text-transformer-encoder", "modality-projector", "tabular-transformer", "tabular-feature-attention", "tabular-row-attention"]
brief: "Direct overlap: frozen text encoder and TabPFN connected by a trained adapter."
source_url: "https://arxiv.org/abs/2606.04876"
pdf: "raw/2026_text_encoders_for_tabpfn.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# Towards Pretraining Text Encoders for TabPFN

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2606.04876) · [PDF](raw/2026_text_encoders_for_tabpfn.pdf)

## Text-source scope

Encodes free-text cells with their column names.

## Problem and positioning

Avoid the information bottleneck from reducing large language embeddings with PCA before treating them as scalar TabPFN features.

## Architecture

**Block roles:** Text: a frozen MiniLM sentence transformer encodes each text cell. Interface: a learned linear/MLP adapter projects embeddings into feature tokens. Tabular: frozen TabPFN 2.5 performs feature and row attention on the combined tokens.

Text cell plus column name → frozen sentence encoder → learned adapter → a short sequence of TabPFN feature tokens. Structured features retain the standard path. The sentence encoder and TabPFN remain frozen while the adapter is pretrained across text-containing datasets.

## Input

Labeled mixed text/tabular rows and query rows.

## Output

Class probabilities or regression predictions.

## For us

The closest precedent for lightweight text support. Reported regression results are strong, but classification does not uniformly beat alternatives. A new adapter needs a specific advantage beyond the broad integration idea.

## Related works

- [MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](2026-multimodalpfn.md)
- [TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings](2026-tabpfn-beyond-tabular-data.md)
- [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md)
