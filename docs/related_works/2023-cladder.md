---
id: "2023-cladder"
title: "CLadder: Assessing Causal Reasoning in Language Models"
year_or_venue: "NeurIPS 2023"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["benchmark-generator"]
brief: "Formal causal questions verbalized as stories; commonsense and nonsense variants motivate controls for semantic priors."
source_url: "https://arxiv.org/abs/2312.04350"
pdf: "raw/2023_cladder.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# CLadder: Assessing Causal Reasoning in Language Models

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2312.04350) · [PDF, v3](raw/2023_cladder.pdf) · [Authors' code and variant descriptions](https://github.com/causalNLP/cladder)

## Text-source scope

`column-label-semantics` applies to meaningful variable names in ordinary stories; `task-domain-context` applies to the supplied causal narrative, probability statements, and question. Nonsense variants replace meaningful variables with arbitrary names. These are generated problems, not independently observed row-level text.

## Problem and positioning

CLadder separates formal causal reasoning from recall of familiar causal associations. Symbolic models and queries determine answers before template-based verbalization. Its three-rung coverage includes associational questions, interventions, and counterfactuals. This is adjacent evaluation methodology rather than a study of observational classification from finite labeled tables.

## Architecture

A causal inference engine computes answers from small binary-variable models; templates turn model information and queries into stories. The work also evaluates CausalCoT, a prompting procedure that decomposes reasoning, using existing pretrained LLMs. It introduces no trained tabular backbone or story adapter. Benchmark generation needs no language-model pretraining or downstream fine-tuning.

**Block roles:** `benchmark-generator` describes the maintained record's generator and evaluation contribution. Text-transformer blocks belong to evaluated LLMs, including the CausalCoT prompting evaluation, rather than to the benchmark generator; tabular-transformer blocks and a learned text–tabular interface are absent. This tag does not describe the architectures of evaluated agents.

## Input

Evaluated models receive natural-language causal assumptions, numerical conditional or interventional probabilities, and a question. Crucially, the paper's section 3.1 explicitly defines these “data” as probability values rather than sampled rows. In-context exemplars demonstrate solved questions; they are not labeled support rows from one predictive dataset.

## Output

Models answer yes/no questions; generated answers and explanations support correctness and reasoning analysis. The generator also records graph, story, and query metadata.

## For us

Borrow the separation between formal ground truth and its verbalization. Commonsense, anti-commonsense, and nonsense variants motivate testing whether our model follows communicated information or familiar variable associations. The authors' code also distinguishes counterintuitive mechanism strengths from replacing meaningful variable names. These are distinct controls: a truthful but counterintuitive story is not an incorrect story. Our proposed extension is to preserve the same sampled rows, schema, and hidden mechanisms while varying story correctness and wording, then measure predictive loss across support sizes. CLadder itself does not establish sample-efficiency gains in that setting, and its explicit probabilities would reveal information our protocol keeps private.

## Related works

[CausalDS](2026-causalds.md) is closer to observational tables paired with domain stories. [TabLLM](2023-tabllm.md) studies low-label tabular prediction but uses a different evaluation design.
