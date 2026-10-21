---
id: "2024-autoelicit"
title: "AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling"
year_or_venue: "ICML 2025; preprint 2024"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["text-transformer-decoder"]
brief: "Task descriptions elicit parameter priors for Bayesian linear models; a direct sample-efficiency baseline outside tabular foundation models."
source_url: "https://arxiv.org/abs/2411.17284v5"
pdf: "raw/2024_autoelicit.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# AutoElicit: Using Large Language Models for Expert Prior Elicitation in Predictive Modelling

[Catalogue](README.md) · [Primary source, v5](https://arxiv.org/abs/2411.17284v5) · [Local PDF](raw/2024_autoelicit.pdf)

## Text-source scope

Feature names and shared task descriptions support both tags. Additional expert knowledge can refine the description. Numeric rows serialized for the separate ICL comparison are not independently observed row-level text. The synthetic experiment can disclose exact coefficients in its prompt; that variant exceeds our permitted story information.

## Problem and positioning

AutoElicit transfers language-model knowledge into Bayesian linear/logistic regression rather than asking an LLM to predict every query. Its reported low-label improvements are evidence for an alternative integration mechanism, not proof that stories improve a PFN.

## Architecture

A pretrained LLM produces Gaussian parameter priors for 100 rephrasings of the task. Their mixture conditions a separate regression model, updated from labeled data with No-U-Turn sampling. Main experiments use GPT-3.5-turbo, GPT-4-turbo and quantized DeepSeek-R1-32B. Language-model weights remain fixed; Bayesian downstream fitting updates regression parameters rather than fine-tuning the LLM.

**Block roles:** Decoder language-transformer blocks generate prior parameters; there are no tabular-transformer blocks. The interface is explicit numerical distributions, with no learned modality projector. Exact proprietary model internals are undisclosed; the decoder tag also covers the disclosed open decoder-model variants.

## Input

Elicitation receives feature meanings, task/model descriptions and optional expert statements. Downstream inference receives numeric support features/labels and query features. Query labels are unnecessary. ICL comparisons additionally place examples into the language prompt.

## Output

Parameter and posterior predictive distributions; classification probabilities or regression predictions. Evaluation varies labeled support sizes and reports accuracy or mean squared error.

## For us

Proposed baseline: elicit priors once per scene and fit Bayesian logistic regression over the same support sets. Compare schema-only and schema-plus-story prompts; the paper's uninformative-prior comparison alone does not isolate this difference. Keep hidden SCM equations and coefficients private, unlike its fully specified synthetic variant. Evaluate label efficiency, calibration and wrong-story damage with the same queries. Its public clinical benchmarks and restricted linear model family leave our unseen-scene, nonlinear PFN question unresolved.

## Related works

[TabLLM](2023-tabllm.md) supplies a direct language-prediction comparison; [CAAFE](2023-caafe.md) uses descriptions through executable feature engineering. [Using large language models to suggest informative prior distributions in Bayesian regression analysis](2025-llm-informative-priors.md) provides complementary evidence about overconfident elicited priors.
