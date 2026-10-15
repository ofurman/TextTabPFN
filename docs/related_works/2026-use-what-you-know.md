---
id: "2026-use-what-you-know"
title: "Use What You Know: Causal Foundation Models with Partial Graphs"
year_or_venue: "ICML 2026"
tags: ["no-text"]
architecture_tags: ["tabular-transformer", "tabular-feature-attention", "tabular-row-attention", "tabular-structural-attention-bias", "graph-convolution", "adaptive-layer-norm"]
brief: "Partial ancestral graphs condition a causal PFN through feature-attention biases and GCN/AdaLN; structural-prior precedent."
source_url: "https://arxiv.org/abs/2602.14972"
pdf: "raw/2026_use_what_you_know.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# Use What You Know: Causal Foundation Models with Partial Graphs

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2602.14972) · [PDF](raw/2026_use_what_you_know.pdf)

## Text-source scope

`no-text` means no natural-language input: domain knowledge is supplied as a structured partial ancestral matrix, distinguishing known ancestry, known non-ancestry, and unknown relations. This is task-level structural context, but it does not qualify for the repository's text-source tags.

## Problem and positioning

Let one general-purpose causal foundation model exploit complete, partial, or absent structural knowledge without retraining a specialist for each graph. The objective is interventional outcome prediction, not ordinary supervised prediction.

## Architecture

**Block roles:** Text: no language transformer. Tabular: a causal PFN alternates feature and sample attention. Interface: ancestral relations bias feature-attention logits; a GCN provides graph-conditioned node states through adaptive layer normalization. These are structured-knowledge conditioning modules, not an LLM-to-TabPFN adapter.

Compares soft learned biases, hard attention masks, GCN modulation, and combinations. An outcome head parameterizes a discretized predictive distribution. The PFN and conditioning modules are trained on synthetic observational/interventional tasks; this is not an adapter attached to an existing frozen pretrained TabPFN. Downstream prediction conditions on data and available graph information.

## Input

Observational data, a query specifying treatment intervention and feature values, variable roles, and optional partial ancestral information. Unknown relations remain distinct from known non-ancestry.

## Output

A predictive distribution over the outcome under the intervention. Conditional or average treatment effects can be derived from contrasts; these are distinct from observational class probabilities.

## For us

Strong precedent for soft structural conditioning and generalization across knowledge availability. Translating a dataset story into these conditioning signals is our proposed extension. For the first CausalDS row-prediction experiment, adapt the interface idea while retaining the observational objective; do not claim this paper validates a frozen language–TabPFN bridge. Its misspecification analysis also motivates testing unknown or withheld context separately from confidently incorrect descriptions.

## Related works

- [CausalDS: Benchmarking Causal Reasoning in Data-Science Agents](2026-causalds.md)
- [Accurate predictions on small data with a tabular foundation model](2025-tabpfn-v2.md)
- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
- [Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering](2023-caafe.md)
