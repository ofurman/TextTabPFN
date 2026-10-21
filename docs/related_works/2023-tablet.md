---
id: "2023-tablet"
title: "TABLET: Learning From Instructions For Tabular Data"
year_or_venue: "2023 preprint"
tags: ["column-label-semantics", "task-domain-context"]
architecture_tags: ["benchmark-generator"]
brief: "Instruction benchmark with shared task knowledge, few-shot comparisons, and flipped-rule faithfulness tests."
source_url: "https://arxiv.org/abs/2304.13188"
pdf: "raw/2023_tablet.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TABLET: Learning From Instructions For Tabular Data

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2304.13188) · [PDF](raw/2023_tablet.pdf)

## Text-source scope

`column-label-semantics` covers meaningful feature names, categorical values, and class names. `task-domain-context` covers the task title and shared natural-language instructions about feature–label relationships. Structured symptom records serialized into text do not establish independently observed row-level prose. Instructions differ by source and generation method; they are additional task information, rather than row-specific features.

## Problem and positioning

TABLET tests learning tabular classification from instructions with zero or few labels. Its 20 tasks combine ten UCI datasets and ten differential-diagnosis tasks derived from DDXPlus. The main experiments report that instructions can help while showing failures of instruction faithfulness. This is a particularly close precedent for evaluating shared context, though it is a benchmark of prompted language models rather than a text-conditioned tabular foundation model.

## Architecture

The benchmark supplies datasets, prompt templates, instruction annotations, executable instruction logic, and modified instructions. Medical instructions come from consumer/professional references. Generated instructions (Section 3.2.2) start from shallow rule-set or nearest-centroid classifiers fitted on the full training data; templates are paraphrased with GPT-3 and reviewed for faithfulness. Thus the instruction channel can transmit information learned from substantially more labels than the few examples shown to the evaluated model.

**Block roles:** TABLET proposes no text-transformer or tabular-transformer backbone or learned fusion interface; `benchmark-generator` describes the benchmark pipeline. Evaluated predictors include pretrained/instruction-tuned encoder–decoder T5 models (Flan-T5 and Tk-Instruct), decoder-only GPT-J, and ChatGPT with proprietary internals. They consume serialized tables and instructions through text tokens, not table-native transformer blocks. Downstream evaluation keeps their weights fixed; it does not train an adapter or fine-tune a tabular backbone.

## Input

Prompts contain a task title, instructions, permitted class names, optional labeled demonstrations, and a serialized query row. Removing the instructions for the same LLM yields the LIFT-style comparator while retaining the title, feature values/names, and labels. The studied few-shot curves use zero through four examples (Section 5.2 and Appendix A.3). Generated rules/prototypes use full training labels during benchmark construction; naturally occurring medical instructions have a separate provenance.

## Output

The evaluated LLM emits a class prediction, extracted from generated text and scored using macro-F1. Additional evaluations compare predictions with executable instruction logic and use flipped instructions to test whether the model follows changed rules. Aggregate benefits do not imply reliable adherence on individual cases. The abstract and main-text summaries differ in some percentage descriptions, so this note does not adopt a single headline improvement.

## For us

Use the matched with/without-instructions protocol and faithfulness probes as direct experimental precedents. Retain schema and target definition in both arms, then vary only the shared story and support size. Our proposed fresh SCM scenes differ from reused domain datasets and from instructions distilled using the full labeled training set. Compare correct, irrelevant, and misleading context; do not treat flipped-rule compliance as automatically desirable when the prediction target remains unchanged. The paper already studies few-label instruction benefits, so a novelty claim must depend on the controlled hidden-mechanism setting and integration method.

## Related works

[TabLLM](2023-tabllm.md) supplies a complementary low-label language-model classifier. [CAAFE](2023-caafe.md) turns task descriptions into feature code. [TabLens](2026-tablens.md) extends context–demonstration diagnostics to larger shot budgets.
