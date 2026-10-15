---
id: "2026-causalds"
title: "CausalDS: Benchmarking Causal Reasoning in Data-Science Agents"
year_or_venue: "2026"
tags: ["task-domain-context"]
architecture_tags: ["benchmark-generator"]
brief: "User-requested paper: tables plus natural-language causal stories. Adjacent evaluation work, rather than a TabPFN extension."
source_url: "https://arxiv.org/abs/2607.08093"
pdf: "raw/2026_causalds.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# CausalDS: Benchmarking Causal Reasoning in Data-Science Agents

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2607.08093) · [PDF](raw/2026_causalds.pdf)

## Text-source scope

The domain story describes the data-generating setting; it is not a per-row text feature.

## Problem and positioning

Evaluate agents that must combine domain descriptions, observational data, causal reasoning, uncertainty, and computation.

## Architecture

**Block roles:** This work proposes an SCM-based benchmark generator and scorer, not a predictive architecture. Evaluated agents may contain language transformers, but those are not tagged as blocks introduced by the benchmark.

A benchmark generator rather than a predictive model. Samples structural causal models, generates observational tables and corresponding natural-language stories, and derives tasks with known ground truth and deterministic scoring.

## Input

An observational table, domain story, and task question supplied to an evaluated agent.

## Output

Task-dependent estimates, predictions, causal answers, uncertainty statements, or abstention when the target is not identifiable.

## For us

Inspiration for context that supplies assumptions rather than another feature column. Adapting it would move the project toward causal reasoning and require objectives beyond ordinary supervised prediction.

## Related works

- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md)
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](2023-tabpfn.md)
