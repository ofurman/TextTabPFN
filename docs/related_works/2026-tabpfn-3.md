---
id: "2026-tabpfn-3"
title: "TabPFN-3: Technical Report"
year_or_venue: "2026"
tags: ["row-level-text"]
architecture_tags: ["tabular-transformer", "tabular-inducing-attention", "tabular-feature-attention", "tabular-row-pooling", "tabular-row-attention", "retrieval-head"]
brief: "Current TabPFN generation; includes tabular-text results and a TabPFN-3-Plus variant."
source_url: "https://arxiv.org/abs/2605.13986"
pdf: "raw/2026_tabpfn_3.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabPFN-3: Technical Report

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2605.13986) · [PDF](raw/2026_tabpfn_3.pdf)

## Text-source scope

Applies to the text-capable Plus variant; the base model is a tabular foundation.

## Problem and positioning

Scale strong tabular prediction to substantially larger datasets.

## Architecture

**Block roles:** Tabular: the disclosed base model uses column inducing attention, CLS feature aggregation, row-level ICL, and an attention retrieval head. Text: Plus accepts text, but its text internals are undisclosed; no text-transformer tag is inferred.

A TabICL-style pipeline: column-distribution embedding, feature aggregation into row embeddings, then dataset-level ICL. Classification uses an attention-based retrieval decoder over labeled training representations. The report separately introduces text-capable TabPFN-3-Plus; its text internals are less fully disclosed than the base architecture.

## Input

Tables and training labels. The Plus variant additionally accepts text columns directly.

## Output

Classification probabilities or regression predictions.

## For us

A relevant modern baseline and existing text-tabular competitor. Distinguish the base model from Plus when reporting comparisons.

## Related works

- [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](2026-tabicl-v2.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
