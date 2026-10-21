---
date: "2026-10-15T16:35:55+02:00"
researcher: "Oleksii Furman"
topic: "Additional related work on shared dataset descriptions, label efficiency, and controlled context evaluation"
tags: ["tabular-foundation-models", "task-context", "sample-efficiency", "prior-elicitation", "synthetic-evaluation"]
sources: ["primary-papers", "conference-proceedings", "author-code"]
status: "complete"
last_updated: "2026-10-15"
---

# Research: Shared context for tabular prediction

**Date**: 2026-10-15T16:35:55+02:00
**Researcher**: Oleksii Furman

## Research Question

> Research more related works based on our research problem.

The [canonical problem](../../../AGENTS.md#research-problem) concerns one shared description per dataset, observational binary prediction, fresh hidden SCMs, and paired support-size curves. The story effect must be isolated from schema/target semantics. Maintained paper records and reading guidance are in the [catalogue](../../related_works/README.md); this is a dated synthesis rather than a second paper catalogue.

## Summary

Six additional works were reviewed and added with verified PDFs. TABLET and TabLens are close precedents for shared instructions and context–label tradeoffs; AutoElicit supplies an explicit prior-elicitation alternative to neural fusion. A separate regression-prior study supplies counterevidence against assuming benefit. CLadder informs semantic controls, while ACE demonstrates numerical prior conditioning without language. Together they sharpen the proposed contribution toward controlled information access, fresh noisy scenes, proper predictive evaluation, and generalization of a shared integration method; they do not establish its novelty or success.

## Detailed Findings

### Shared instructions and support-size curves

TABLET compares instruction-conditioned classification against instruction removal and tests modified-rule faithfulness. Some instructions are distilled from models fitted to full training data, so their knowledge budget differs from the demonstrated few-shot rows. Its findings motivate matched ablations, but do not establish gains from qualitative hidden-SCM stories. [Paper](https://arxiv.org/abs/2304.13188), [maintained note](../../related_works/2023-tablet.md).

TabLens v3 directly crosses schema/task context with demonstrations up to 64 shots. Its explicit synthetic rules reveal the deterministic labeling function. It also flips below-chance AUC using evaluation outcomes, so reported ranking performance is not directly comparable with ordinary deployed probabilities. The note records version-specific numerical and generator-description inconsistencies. Interpret individual configurations rather than treating a combined rich/minimal prompt contrast as an isolated story effect. [Paper v3](https://arxiv.org/abs/2605.24417v3), [maintained note](../../related_works/2026-tablens.md).

### Explicit priors as an alternative interface

AutoElicit turns task descriptions into parameter priors for Bayesian linear/logistic models and reports label-efficiency gains. Its synthetic coefficient-disclosing prompt exceeds our information contract; its real-data experiments remain useful adjacent evidence. An informative-versus-uninformative prior comparison differs from schema-only versus schema-plus-story elicitation. [Paper v5](https://arxiv.org/abs/2411.17284v5), [maintained note](../../related_works/2024-autoelicit.md).

The Scientific Reports study reports overconfident elicited regression priors and no statistically significant predictive improvement on either of its two real datasets. Directionally plausible knowledge need not produce useful uncertainty or coefficient magnitudes. This motivates calibration and conditioning-strength controls; it does not itself test incorrect-story robustness. [Publisher paper](https://www.nature.com/articles/s41598-025-18425-9), [maintained note](../../related_works/2025-llm-informative-priors.md).

### Ground-truth and structural-conditioning controls

CLadder verbalizes formal causal questions and varies common-sense associations. Its numerical inputs are probabilities, not finite sampled support tables. Borrow its distinction between familiar, counterintuitive and arbitrary semantics, without using its outcomes as evidence of tabular label efficiency. [Paper](https://arxiv.org/abs/2312.04350), [authors' code](https://github.com/causalNLP/cladder), [maintained note](../../related_works/2023-cladder.md).

ACE accepts runtime latent values or numerical prior histograms in an amortized transformer. It assumes predefined latent variables and a factorized prior interface; it does not process natural language. This is a conditioning precedent and potential diagnostic design, rather than a ready-made story-to-TabPFN solution. [AISTATS paper](https://proceedings.mlr.press/v258/chang25a.html), [maintained note](../../related_works/2024-ace.md).

## Sources Consulted

Primary papers, conference records, and relevant author code are linked with their claims above. All six maintained notes link downloaded PDFs; the [manifest](../../related_works/manifest.json) records versions, retrieval provenance, page counts and checksums.

## Key Insights

These are proposals for our study, not reproduced experimental findings:

1. Use schema-only, correct-story, irrelevant-story and incorrect-story arms with identical support/query rows. Separate truthful counterintuitive scenes from deliberate story errors.
2. Compare language-only prediction, CAAFE, elicited-prior logistic regression, vanilla TabPFN and a shared text-conditioned tabular interface. Match permitted information and label budgets; report differing fitting/search costs.
3. Evaluate log loss, Brier score, ordinary AUROC, calibration and paired label-efficiency curves. Set any performance threshold from development data before inspecting final test curves.
4. Hold out scene/mechanism families and retain unsuccessful contexts. Hidden coefficients and query labels remain private. A privileged structural diagnostic must be reported separately.

## Conflicting Information

Positive prior-elicitation findings and the regression study's negative results differ in datasets, inference procedures and prompt/model choices; they are not a matched replication contradiction. They support testing when conditioning helps rather than averaging them into a universal claim. TabLens also contains internal version-specific inconsistencies documented in its note; unresolved generator identities should be checked in code before reproduction.

## Confidence Notes

The paper inputs and major interfaces were reviewed against primary sources and parsed PDFs. No model was trained, benchmark reproduced, generator executed or empirical benefit measured here. This is a targeted expansion of the collection, not an exhaustive or systematic novelty search. Proprietary model internals remain undisclosed where noted. Neither reviewed prior benchmark fully matches our proposed controlled hidden noisy SCM protocol with a shared language-conditioned tabular foundation model.

## Open Questions

- Can a shared language-to-tabular interface improve proper predictive loss on unseen scenes beyond explicit-prior and feature-code baselines?
- How much do label savings and harmful-context tails change across support sizes when schema semantics remain fixed?
- Which component fails first: extracting the information a story communicates, or using that information in prediction?

These are empirical research questions for later experiments, not missing preferences that block this literature update.

## Clarifications Log

No additional user clarification was needed; scope follows the repository's canonical research problem.
