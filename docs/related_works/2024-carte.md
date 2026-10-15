---
id: "2024-carte"
title: "CARTE: Pretraining and Transfer for Tabular Learning"
year_or_venue: "2024"
tags: ["column-label-semantics"]
architecture_tags: ["static-text-embeddings", "graph-attention"]
brief: "Embeds string values and column names in graph representations; transfer across different schemas."
source_url: "https://arxiv.org/abs/2402.16785"
pdf: "raw/2024_carte.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# CARTE: Pretraining and Transfer for Tabular Learning

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2402.16785) · [PDF](raw/2024_carte.pdf)

## Text-source scope

Focuses on column names and categorical/entity strings rather than document-like text.

## Problem and positioning

Transfer knowledge between tables with different columns and schemas.

## Architecture

**Block roles:** Text: FastText initializes static string and column-name embeddings; it is not a text transformer. Tabular: graph-attention layers process row graphlets, not a tabular transformer backbone.

Converts each row into a star-shaped graph: values become leaf nodes, column names describe edges, and a central node aggregates the row. Static FastText embeddings initialize semantic information; graph attention contextualizes it. Knowledge-graph pretraining is followed by downstream fine-tuning.

## Input

Mixed-type rows with column names, plus labels for downstream adaptation.

## Output

Classification or regression predictions through a learned row representation.

## For us

Demonstrates semantic transfer across unmatched schemas. More directly relevant to names and categorical strings than to rich document-length text.

## Related works

- [ConTextTab: A Semantics-Aware Tabular In-Context Learner](2025-contexttab.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](2025-tabstar.md)
