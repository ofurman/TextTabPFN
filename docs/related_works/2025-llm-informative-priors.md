---
id: "2025-llm-informative-priors"
title: "Using large language models to suggest informative prior distributions in Bayesian regression analysis"
year_or_venue: "Scientific Reports 2025"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["text-transformer-decoder"]
brief: "Evaluates LLM-suggested regression priors, finding overconfidence and no statistically significant predictive improvement on two real datasets."
source_url: "https://www.nature.com/articles/s41598-025-18425-9"
pdf: "raw/2025_llm_informative_priors.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# Using large language models to suggest informative prior distributions in Bayesian regression analysis

[Catalogue](README.md) · [Primary source](https://www.nature.com/articles/s41598-025-18425-9) · [Local PDF](raw/2025_llm_informative_priors.pdf)

## Text-source scope

The prompts describe the prediction problem, named variables and regression specification: task/domain context plus column/label semantics. There is no independently observed row-level text. Shared problem descriptions elicit prior knowledge, rather than attach text to individual patients or concrete mixtures.

## Problem and positioning

This study tests whether an LLM can supply useful numerical regression priors and whether these agree with observed data. Its reported lack of statistically significant predictive gains complements positive prior-elicitation results and supports treating context benefit as a hypothesis.

## Architecture

ChatGPT-4o-mini, Gemini 2.5 Pro and Claude Opus suggest moderately and weakly informative Gaussian coefficient priors with explanations and confidence scores. Bayesian regression combines these with data; predictive experiments use R-INLA approximation. The work prompts existing pretrained models without reported downstream LLM fine-tuning. Regression inference estimates parameters separately.

**Block roles:** The decoder tag describes autoregressive language-model elicitation, including the GPT model; exact proprietary transformer configurations are undisclosed. No tabular-transformer blocks or learned cross-modal adapter are used. The text-to-table-model interface consists of explicit prior hyperparameters, not token embeddings; Gemini/Claude internals are not inferred from their names.

## Input

Elicitation receives the problem, variable definitions and Bayesian model specification, with requests to justify and reassess suggested priors. Regression consumes observed numeric features and outcomes from heart-disease and concrete-strength datasets. Prior generation and posterior inference are distinct steps.

## Output

Coefficient priors, posterior estimates and predictions. The study examines prior–MLE disagreement and five-fold predictive performance, reporting frequent overconfidence and no statistically significant improvement on either task.

## For us

Proposed tests: vary support size and score both predictive loss and calibration; include weak and strong context conditioning. Accurate directional knowledge can coexist with inaccurate magnitudes or excessive confidence, so a sign-correct story is insufficient evidence of a useful prior. This paper does not test freshly generated hidden SCMs, contradictory stories or text-conditioned PFNs. Its real-dataset results neither establish nor refute our low-label benefit hypothesis. Any adaptation should keep query outcomes private during prompt selection and report unsuccessful contexts as well as successful ones.

## Related works

[AutoElicit](2024-autoelicit.md) is the direct prior-elicitation and sample-efficiency comparator. [TabPFN](2023-tabpfn.md) amortizes a synthetic prior without natural-language input; [Use What You Know](2026-use-what-you-know.md) conditions a PFN on structural knowledge rather than elicited regression coefficients.
