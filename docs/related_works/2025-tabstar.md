---
id: "2025-tabstar"
title: "TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields"
year_or_venue: "2025"
tags: ["row-level-text", "column-label-semantics"]
architecture_tags: ["text-transformer-encoder", "tabular-transformer", "tabular-feature-attention", "target-tokens"]
brief: "Text-tabular foundation model with task-aware text representations; competitor and benchmark source."
source_url: "https://arxiv.org/abs/2505.18125"
pdf: "raw/2025_tabstar.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2505.18125) · [PDF](raw/2025_tabstar.pdf)

## Text-source scope

Uses text fields, feature names, and candidate target names; target names are label semantics.

## Problem and positioning

Make text representations useful for the particular prediction target rather than rely on generic frozen embeddings.

## Architecture

**Block roles:** Text: an e5 transformer encodes individual feature verbalizations and target names, with upper layers adapted. Tabular: separate feature-fusion and cross-feature transformer blocks combine semantic and numerical representations. Target tokens support prediction; there is no TabPFN-style row-context transformer.

Encodes each column-name/value pair; numerical features also retain precise numerical representations. Feature-level fusion precedes cross-feature attention. Target tokens representing candidate classes interact with features and use a shared prediction head. Cross-dataset pretraining is followed by downstream fine-tuning, including adaptation of the text encoder.

## Input

Rows, feature names and values, and target descriptions or candidate class names. Candidate classes are provided for every example; the true query label remains hidden.

## Output

Class probabilities or a regression value.

## For us

A strong task-aware competitor. It transfers through pretrained weights and downstream adaptation rather than TabPFN-style labeled-context inference.

## Related works

- [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md)
- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md)
- [TabPFN-3: Technical Report](2026-tabpfn-3.md)
