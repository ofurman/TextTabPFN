---
id: "2023-caafe"
title: "Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering"
year_or_venue: "2023"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["feature-code-generator", "tabular-transformer", "tabular-row-attention", "tree-predictor"]
brief: "Dataset descriptions drive LLM-generated feature code selected with TabPFN validation; task-context baseline."
source_url: "https://arxiv.org/abs/2305.03403"
pdf: "raw/2023_caafe.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2305.03403) · [PDF](raw/2023_caafe.pdf)

## Text-source scope

Uses dataset descriptions (`task-domain-context`) and feature names (`column-label-semantics`). Training-derived summaries and sample rows also enter prompts. Serialized examples are not independently observed row-level text.

## Problem and positioning

Bring semantic domain knowledge into tabular prediction through automated feature engineering. CAAFE already connects dataset-level language with TabPFN, using executable transformations rather than a neural bridge.

## Architecture

**Block roles:** Text: fixed GPT-3.5/GPT-4 language models generate Python feature code; the paper does not disclose GPT-4's internal architecture, so tags identify its code-generation role rather than infer a transformer subtype. Tabular: original TabPFN uses row attention to evaluate transformed data. Downstream experiments also use random forests and logistic regression. Interface: code execution, validation, and feedback; no learned text-to-tabular projector.

Iteratively propose transformations, evaluate them with TabPFN, retain improvements, and return errors or performance feedback to the LLM. LLM and pretrained TabPFN weights remain fixed; feature selection is dataset-specific and uses labels through validation. Tree/linear predictors require ordinary downstream fitting.

## Input

Labeled training/validation data, a user-written dataset description, feature names, and train-derived descriptive information. Feature search must not inspect final test labels.

## Output

Executable transformation code, feature-utility explanations, transformed tables, and downstream classification predictions.

## For us

An essential task-context baseline: our proposed bridge must improve on language-guided feature engineering, not merely demonstrate that descriptions help TabPFN. Compare identical support budgets and include feature-search cost. A shared latent bridge would differ by conditioning internal representations without per-dataset code search; any advantage remains untested.

## Related works

- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](2023-tabpfn.md)
- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md)
- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
- [Use What You Know: Causal Foundation Models with Partial Graphs](2026-use-what-you-know.md)
