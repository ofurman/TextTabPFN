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

See the [related-work wiki](docs/related_works/README.md) for a catalogue with text-source and architecture block tags, short briefs, and one detailed note per paper. Notes distinguish text-transformer and tabular-transformer components and cover problem and positioning, architecture, input, output, and relevance to this project, with links to PDFs in `docs/related_works/raw/`.

All related-work additions and revisions must follow the [mandatory maintenance rules in AGENTS.md](AGENTS.md#related-works-maintenance-mandatory). Keep the catalogue, notes, and PDF manifest synchronized and run `./scripts/lint_related_works.sh` before completing a change.

Existing work already combines text representations with TabPFN and other tabular foundation models. Our initial focus is to test when text helps in smaller labeled-data regimes, rather than assume the broad combination is novel.

This repository starts as a research idea and literature collection; experiments are not implemented yet.
