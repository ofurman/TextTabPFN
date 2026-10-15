---
id: "2023-tabpfn"
title: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
year_or_venue: "ICLR 2023; preprint 2022"
tags: ["no-text"]
architecture_tags: ["tabular-transformer", "tabular-row-attention"]
brief: "Original prior-data fitted network and small-data tabular classification foundation."
source_url: "https://arxiv.org/abs/2207.01848"
pdf: "raw/2023_tabpfn.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2207.01848) · [PDF](raw/2023_tabpfn.pdf)

## Text-source scope

Numerical foundation; no semantic text input.

## Problem and positioning

Learn a reusable prediction algorithm for small tabular datasets rather than train a new model from scratch for each dataset.

## Architecture

**Block roles:** Tabular: the PFN transformer attends over row embeddings and labeled examples. Text: no text transformer is present.

A transformer embeds entire rows and attends over labeled training examples. Offline pretraining on synthetic tasks teaches approximate Bayesian prediction under the synthetic dataset prior; ordinary downstream inference keeps weights fixed.

## Input

Labeled numerical training rows and unlabeled query rows. The original work targets roughly 1,000 training examples, 100 numerical features, and 10 classes.

## Output

Class probabilities for query rows.

## For us

Establishes the small-data, labeled-context mechanism to preserve. It does not natively understand column names or free text.

## Related works

- [Accurate predictions on small data with a tabular foundation model](2025-tabpfn-v2.md)
- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](2025-tabicl.md)
