---
id: "2026-multimodalpfn"
title: "MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning"
year_or_venue: "2026"
tags: ["row-level-text"]
architecture_tags: ["text-transformer-encoder", "vision-encoder", "modality-projector", "gated-mlp", "cross-attention-pooler", "tabular-transformer", "tabular-feature-attention", "tabular-row-attention"]
brief: "Direct multimodal PFN architecture for combining text/image representations with tabular features."
source_url: "https://arxiv.org/abs/2602.20223"
pdf: "raw/2026_multimodalpfn.pdf"
pdf_status: "available"
updated: "2026-10-15"
---

# MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning

[Catalogue](README.md) · [Primary source](https://arxiv.org/abs/2602.20223) · [PDF](raw/2026_multimodalpfn.pdf)

## Text-source scope

Text fields are observations associated with individual examples.

## Problem and positioning

Combine structured features with text and images when labeled multimodal data is limited.

## Architecture

**Block roles:** Text: a frozen ELECTRA transformer encodes text; a frozen DINOv2 encoder handles images. Interface: gated MLP projections and a cross-attention pooler create modality tokens. Tabular: the adapted TabPFN v2 backbone applies feature and row attention.

Frozen modality encoders feed a multi-head gated MLP that expands representations into candidate tokens. A cross-attention pooler reduces these to a controlled number of tokens, which join tabular tokens inside TabPFN. Downstream training updates the projector, TabPFN backbone, and prediction head while modality encoders remain frozen.

## Input

Labeled rows containing structured features and text/images, plus query rows.

## Output

Class probabilities in the reported experiments, including tasks formed by discretizing numerical targets.

## For us

Direct architectural competition. Token-count imbalance can let text/image tokens dominate attention and suppress structured signals; fusion should control this balance.

## Related works

- [Towards Pretraining Text Encoders for TabPFN](2026-text-encoders-for-tabpfn.md)
- [Accurate predictions on small data with a tabular foundation model](2025-tabpfn-v2.md)
