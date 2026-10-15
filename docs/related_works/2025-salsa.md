---
id: "2025-salsa"
title: "SALSA: Single-pass Autoregressive LLM Structured Classification"
year_or_venue: "2025 preprint"
tags: ["row-level-text", "column-label-semantics", "task-domain-context"]
architecture_tags: ["text-transformer-decoder", "class-token-logit-filter"]
brief: "Decoder-only LLM classifier using structured prompts, class-token logits, and LoRA; a text-only baseline for fusion."
source_url: "https://arxiv.org/abs/2510.22691"
pdf: "raw/2025_salsa.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# SALSA: Single-pass Autoregressive LLM Structured Classification

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2510.22691) · [PDF](raw/2025_salsa.pdf)

## Text-source scope

Classifies observed text, including reviews and sentence pairs (`row-level-text`). Prompts describe the task (`task-domain-context`) and map meaningful class names to single-token answers (`column-label-semantics`, for label meanings rather than table headers). This is text classification; “structured” refers to the prompt and response format, not a mixed tabular input architecture.

## Problem and positioning

Adapt instruction-tuned generative LLMs for classification without multi-token answer generation or a randomly initialized classification head. The paper studies zero/few-shot prompting and supervised adaptation on NLP benchmarks; it does not evaluate text-plus-TabPFN fusion.

## Architecture

**Block roles:** Text: an instruction-tuned decoder-only Llama 3.3 70B transformer processes the prompt. Output interface: filter its existing vocabulary logits to mapped class tokens and apply softmax. Tabular: no dedicated tabular transformer or modality projector is present.

Task instructions, class mappings, and a fixed answer template specify the answer position. One forward pass supplies its logits. Supervised experiments train LoRA parameters with classification cross-entropy while retaining the pretrained backbone; zero/few-shot variants use fixed weights. Class-logit filtering is distinct from TabSTAR's feature-interacting target tokens.

## Input

Text or a text pair, task instructions, class-to-token mappings, and an answer template. Few-shot inference adds ten balanced demonstrations; supervised adaptation uses labeled examples.

## Output

Softmax probabilities over selected class tokens and a predicted label. An appendix extends scoring to regression by taking an expectation over discretized values. Access to model logits is required.

## For us

Use as a text-only comparator and potential text branch for prediction-level fusion with TabPFN. Comparing text-only, table-only, and fused predictions could separate language knowledge from integration benefits; this is our proposed experiment. SALSA's single inference pass should not be confused with training-free adaptation. Its 70B backbone, fine-tuning cost, label-token sensitivity, and possible pretraining contamination matter when comparing small-label performance and compute budgets. The paper establishes no gain for mixed text/tabular prediction.

## Related works

- [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](2023-tabllm.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
- [TabPFN beyond Tabular Data: Calibration and Accuracy on Multimodal Embeddings](2026-tabpfn-beyond-tabular-data.md)
