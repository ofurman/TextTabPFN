# Text + TabPFN

## Research problem

Can a tabular foundation model use a natural-language description of the data-generating setting to improve prediction from a limited labeled dataset? How does the benefit change as the number of labeled examples increases?

Use freshly generated scenes to create matched pairs of task/domain context and labeled tabular datasets. The context is one shared description per dataset, rather than a text feature attached to each row. The goal is to test whether context improves predictive performance and sample efficiency on previously unseen data-generating processes, across few-data and many-data regimes. Improvement is a hypothesis to evaluate, not an assumption used to select scenes or results.

## Formal setting

A scene $s$ provides

$$
\mathcal T_s = (c_s, \Sigma_s, D_s, Q_s),
$$

where:

- $c_s$ is the natural-language domain story.
- $\Sigma_s$ is the public schema, feature meanings, and target definition.
- $D_s = \{(x_i, y_i)\}_{i=1}^{n}$ is the labeled support dataset.
- $Q_s$ is a held-out query dataset whose labels remain private during prediction.

Compare a predictor without the story,

$$
p_0(y_* \mid x_*, D_s, \Sigma_s),
$$

with a context-aware predictor,

$$
p_c(y_* \mid x_*, D_s, \Sigma_s, c_s).
$$

Both receive the same support examples, query features, schema, and target definition. The experimental difference is access to the domain story. Vary $n$ within each scene to measure the effect of context across labeled-data regimes.

The hidden structural causal model supplies the data-generating ground truth. Its equations, parameters, and query outcomes are not model inputs. Sampled outcomes may contain noise. The initial objective is binary observational prediction; regression is a possible extension, while causal-effect estimation is a separate research problem.

## Why context could help

With few observations, several predictive relationships may explain the table. The story can supply useful preferences about relevant variables, dependencies, and measurement relationships before the labeled data is sufficient to infer them reliably. For example, it may explain that several columns are noisy measurements of the same underlying quantity.

A causal graph alone does not specify numerical functions, coefficients, or noise. Randomly sampled mechanisms may also conflict with a language model's everyday expectations. The study must therefore distinguish information actually communicated by the story from assumptions supplied by the model. A benefit from context can reflect additional structural information, rather than a better model of identical inputs.

## Hypotheses

- **H1 — Predictive benefit:** Correct task/domain context reduces predictive loss at small labeled support sizes.
- **H2 — Sample efficiency:** Correct context reduces the number of labels needed to reach a fixed level of predictive performance.
- **H3 — Dependence on data size:** The advantage generally decreases as observations resolve uncertainty, although finite-model limitations can preserve a benefit in larger-data regimes.
- **H4 — Context reliability:** Irrelevant or incorrect context can hurt prediction; a useful integration method should limit that damage.

## Related work

See the [related-work wiki](docs/related_works/README.md) for a catalogue with text-source and architecture block tags, short briefs, and one detailed note per paper. Notes distinguish text-transformer and tabular-transformer components and cover problem and positioning, architecture, input, output, and relevance to this project, with links to PDFs in `docs/related_works/raw/`.

All related-work additions and revisions must follow the [mandatory maintenance rules in AGENTS.md](AGENTS.md#related-works-maintenance-mandatory). Keep the catalogue, notes, and PDF manifest synchronized and run `./scripts/lint_related_works.sh` before completing a change.

Existing work already combines text representations with TabPFN and other tabular foundation models. Our initial focus is to test when text helps in smaller labeled-data regimes, rather than assume the broad combination is novel.

This repository starts as a research idea and literature collection; experiments are not implemented yet.
