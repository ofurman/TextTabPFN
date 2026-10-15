---
id: "2022-tabtext"
title: "TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling"
year_or_venue: "2022 preprint; revised 2025"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["text-transformer-encoder", "tree-predictor"]
brief: "Early language representation approach for augmenting structured health data."
source_url: "https://arxiv.org/abs/2206.10381"
pdf: "raw/2022_tabtext.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2206.10381) · [PDF](raw/2022_tabtext.pdf)

## Text-source scope

Uses variable names and table descriptions. Its generated row paragraphs are serialized table values, not independently observed free text.

## Problem and positioning

Build reusable representations of heterogeneous clinical tables with less manual preprocessing.

## Architecture

**Block roles:** Text: the selected Clinical-Longformer encodes serialized patient descriptions after self-supervised adaptation. Tabular: gradient-boosted trees consume the resulting embeddings, optionally alongside structured features; there is no separate tabular transformer.

Verbalizes structured patient information into a paragraph, embeds it with a clinical language model, and feeds the representation to a conventional predictor. Embeddings can augment original structured features. The saved revision includes self-supervised language-model adaptation before embedding extraction.

## Input

Clinical tables and metadata; downstream predictors additionally require training labels.

## Output

Reusable embeddings, followed by downstream predictions such as clinical classification outcomes.

## For us

A practical feature-augmentation baseline. Text is largely constructed from the table, so gains may reflect representation and prior knowledge rather than new observations.

## Related works

- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md)
- [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md)
