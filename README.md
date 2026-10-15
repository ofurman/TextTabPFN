# Text + TabPFN

Investigate whether text can improve tabular foundation models, especially when labeled data is scarce.

The idea is to combine structured features with useful text—such as row descriptions, free-text fields, or column descriptions—so that pretrained language representations support TabPFN's predictions. Start with frozen text embeddings and simple fusion, then explore richer integration if the baselines justify it.

## Initial experiments

- Compare tabular-only TabPFN, text-only predictors, and text + tabular variants on the same splits.
- Test embedding concatenation, dimension reduction, and prediction-level fusion; include TabICL as a tabular foundation-model baseline.
- Measure performance across small labeled training sets, with ablations for each text source, runtime, and robustness to missing or irrelevant text.
- Keep preprocessing and model selection within training folds, and check for label leakage in text.

**Research question:** When, and through which integration method, does text provide useful information beyond tabular features in low-data prediction?

## Related work

See the [related-work collection and briefing](docs/related_works/README.md) for papers, primary sources, text-source tags, and each work's problem and positioning, architecture, input, output, and relevance to this project.

All related-work additions and revisions must follow the [mandatory maintenance rules in AGENTS.md](AGENTS.md#related-works-maintenance-mandatory). The collection README is the single maintained index and briefing; update it and any affected PDF metadata in the same change.

Existing work already combines text representations with TabPFN and other tabular foundation models. Our initial focus is to test when text helps in smaller labeled-data regimes, rather than assume the broad combination is novel.

This repository starts as a research idea and literature collection; experiments are not implemented yet.
