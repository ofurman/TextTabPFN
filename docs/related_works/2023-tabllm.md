---
id: "2023-tabllm"
title: "TabLLM: Few-shot Classification of Tabular Data with Large Language Models"
year_or_venue: "2023; preprint 2022"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["text-transformer-encoder-decoder"]
brief: "Low-label baseline using natural-language serialization and language-model knowledge."
source_url: "https://arxiv.org/abs/2210.10723"
pdf: "raw/2023_tabllm.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabLLM: Few-shot Classification of Tabular Data with Large Language Models

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2210.10723) · [PDF](raw/2023_tabllm.pdf)

## Text-source scope

Uses feature names and an explicit classification-task description. Serialized rows are not independently observed free text.

## Problem and positioning

Use language-model knowledge for tabular classification when very few labels are available.

## Architecture

**Block roles:** Text: the T0 encoder-decoder language transformer reads serialized rows and scores label answers, with T-Few adaptation in few-shot experiments. Tabular: there is no dedicated tabular transformer; the table is represented as language.

Serializes each row into natural language and supplies a task description to an instruction-trained language model. Answer tokens map to classes. Studies zero-shot prediction and parameter-efficient few-shot fine-tuning.

## Input

A verbalized row and classification question; labeled examples are used for few-shot adaptation.

## Output

Class-label scores or predictions.

## For us

Tests a different route: let the language model predict directly. Useful for checking whether semantic knowledge alone explains benefits attributed to text–TabPFN integration.

## Related works

- [SALSA: Single-pass Autoregressive LLM Structured Classification](2025-salsa.md)
- [TabText: Language-Based Representations of Tabular Health Data for Predictive Modelling](2022-tabtext.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
- [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](2026-causalds.md)
