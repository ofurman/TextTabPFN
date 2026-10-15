---
id: "2025-tabpfn-v2"
title: "Accurate predictions on small data with a tabular foundation model"
year_or_venue: "Nature 2025"
tags: ["no-text"]
architecture_tags: ["tabular-transformer", "tabular-row-attention", "tabular-feature-attention"]
brief: "TabPFNv2 classification and regression baseline for small tabular datasets."
source_url: "https://www.nature.com/articles/s41586-024-08328-6"
pdf: "raw/2025_tabpfn_v2.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# Accurate predictions on small data with a tabular foundation model

[Catalogue](README.md) · [Primary source](https://www.nature.com/articles/s41586-024-08328-6) · [PDF](raw/2025_tabpfn_v2.pdf)

## Text-source scope

Categorical IDs and observed labels do not constitute semantic text input.

## Problem and positioning

Extend TabPFN to stronger real-world classification and regression with categorical features and missing values.

## Architecture

**Block roles:** Tabular: the backbone alternates attention across examples and within-row features. Text: no text transformer is present.

Cell/feature representations with alternating attention across features within a row and across examples. Pretrained on a richer synthetic prior, then used for fixed-weight in-context prediction.

## Input

Labeled mixed-type tables and query rows.

## Output

Class probabilities or a predictive distribution for a numerical target, from which means and quantiles can be derived.

## For us

Its internal feature-token space is the insertion point for several text extensions. Synthetic causal pretraining does not turn ordinary supervised predictions into causal-effect estimates.

## Related works

- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](2023-tabpfn.md)
- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
- [MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning](2026-multimodalpfn.md)
