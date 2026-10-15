# Repository instructions

## Team conventions

- We use Vertica as our analytical database — prefer Vertica-specific SQL patterns.
- Use `uv` for Python tooling, not pip/pipx, unless there is a specific reason.
- OKE (Oracle Kubernetes Engine) kubeconfig scripts: use `get_oke_kubeconfig_v3`; v1 is legacy.

## Related works maintenance (mandatory)

This section is the single source of related-work maintenance requirements. Other repository documents MUST link here rather than introduce competing rules. The linter implements the structural checks; scientific accuracy and tag interpretation require source review.

### Wiki layout and synchronization

- `docs/related_works/README.md` is the catalogue: one row per paper with a note link, year/venue, text-source tags, architecture block tags, and a short brief. Keep project positioning and reading guidance here; do not duplicate full paper records in the catalogue or a separate brief file.
- Each paper MUST have exactly one Markdown note directly in `docs/related_works/`, with a stable lowercase hyphenated filename such as `2026-text-encoders-for-tabpfn.md`.
- Downloaded PDFs MUST live directly in `docs/related_works/raw/`. No notes, partial downloads, or other files belong in `raw/`.
- `docs/related_works/manifest.json` records downloaded PDF provenance and maps each PDF to its note. Reference-only notes do not receive PDF manifest entries.
- New works, including reference-only links, MUST receive a catalogue row and note in the same change. Update the note, catalogue, PDF, and manifest together when affected. No orphan notes/PDFs, duplicate works, or stale metadata are allowed.
- A new PDF revision updates the existing note and catalogue row rather than creating a second paper record. Remove a work's note, row, PDF, and manifest entry together and fix incoming links. Supporting documentation does not need paper records.
- Dated research snapshots may summarize findings, but the individual notes are the maintained paper records.

### Required note format

Notes MUST begin with `---` frontmatter using the exact keys below, one `key: JSON-value` per line. JSON strings, arrays, and null keep this a simple YAML-compatible format readable without extra Python dependencies. Do not use multiline YAML, unquoted strings, comments, or extra keys.

```yaml
---
id: "2026-paper-slug"
title: "Exact paper title"
year_or_venue: "2026"
tags: ["row-level-text", "column-label-semantics"]
architecture_tags: ["text-transformer-encoder", "modality-projector", "tabular-transformer"]
brief: "A concise description of the contribution and its relevance."
source_url: "https://arxiv.org/abs/paper-id"
pdf: "raw/2026_paper_slug.pdf"
pdf_status: "available"
updated: "2026-10-15"
---
```

The example illustrates syntax; replace it with verified metadata. `id` MUST equal the filename stem. `title` MUST be the paper's actual title, and the note MUST have exactly one H1 matching it. `year_or_venue` and `brief` MUST be nonempty. `updated` MUST be an ISO `YYYY-MM-DD` date. The catalogue's title, year/venue, text-source tags, architecture tags (including order), and brief MUST exactly match the note metadata. Use the exact table columns `Paper`, `Year / venue`, `Text-source tags`, `Architecture blocks`, and `Brief`.

Each note MUST link back to `README.md` and to its HTTP(S) primary source. If downloaded, `pdf_status` MUST be `available`, `pdf` MUST be a `raw/*.pdf` path, and the body MUST link directly to that PDF. Otherwise, use `pdf: null`, set `pdf_status` to `unavailable` or `reference-only`, and add a nonempty `## PDF availability` section explaining why.

Each note MUST contain exactly one nonempty section with each of these exact H2 headings:

1. **Text-source scope:** explain the assigned tags, variant-specific scope, and any uncertainty.
2. **Problem and positioning:** the problem addressed and relation to other approaches.
3. **Architecture:** the main components and information flow; identify pretraining, frozen components, and downstream fine-tuning. Include exactly one nonempty `**Block roles:**` paragraph distinguishing text-transformer blocks, tabular-transformer blocks, and their interface; explicitly state when either is absent or undisclosed. For benchmarks, describe the generator and evaluation setup instead of attributing evaluated agents' architectures to the benchmark.
4. **Input:** the actual data, metadata, labels, and context supplied to the method.
5. **Output:** predictions, representations, distributions, or benchmark responses produced.
6. **For us:** concrete relevance, overlap, baselines, limitations, or proposed extensions for text plus tabular foundation models.
7. **Related works:** link to at least one other paper note when the collection contains multiple papers. Use existing note filenames, not self-links or external URLs in this section.

Use standard relative inline Markdown links, including section anchors when useful. Keep records concise and grounded in primary sources. Distinguish reported findings from our interpretations, proposals, and novelty claims. Explicitly explain undisclosed or unverified details; do not guess or leave empty sections or bare placeholders such as TODO/TBD/N/A.

### Text-source tags

Apply all supported tags from this vocabulary in note frontmatter and the catalogue:

- `row-level-text`: independently observed text associated with an example, such as a review, clinical note, or document. Serializing structured row values into sentences does not alone qualify.
- `column-label-semantics`: natural-language feature names, categorical/entity strings, or target/class names used for their meaning. Arbitrary category IDs and observed numeric class labels do not qualify.
- `task-domain-context`: explicit task instructions, table/domain descriptions, or causal stories supplied as input. Domain-specific pretraining alone does not qualify; target names alone belong under `column-label-semantics`.
- `no-text`: none of the three text-source categories applies. This tag MUST appear alone.

Tags MUST be nonempty, unique, and drawn only from this vocabulary. Classify actual input use, not hypothetical extensions. Explain variant-specific scope and uncertainty in the note; do not infer undisclosed text support. Keep tags synchronized when revising a note.

### Architecture block tags

`architecture_tags` MUST be a nonempty, unique array drawn from this vocabulary, separate from text-source `tags`. Apply tags to salient components in the main studied pipeline, not every comparison baseline or incidental transformer feedforward layer. Explain scope and roles in the Architecture section.

- `tabular-transformer`: a transformer backbone operating on structured table features or row representations, including table-native ICL. Serializing a table into an LLM does not qualify.
- `text-transformer-encoder`: an encoder-style language transformer processing text tokens into representations, such as MiniLM, e5, ELECTRA, or Clinical-Longformer.
- `text-transformer-encoder-decoder`: an encoder-decoder language transformer such as T0, used for text-based prediction.
- `static-text-embeddings`: non-transformer semantic embeddings, such as FastText; do not label these as a text transformer.
- `vision-encoder`: a component encoding images; explain its specific architecture in the note.
- `audio-encoder`: a component encoding audio; explain its specific architecture in the note.
- `tabular-row-attention`: attention across examples/rows in a tabular backbone.
- `tabular-feature-attention`: attention across features within a row in a tabular backbone.
- `tabular-inducing-attention`: inducing-point/set attention summarizing column distributions in a tabular backbone.
- `tabular-row-pooling`: learned CLS-style aggregation of a row's feature tokens into a compact row representation.
- `modality-projector`: a learned projection/adapter aligning text or other modality representations with tabular tokens.
- `gated-mlp`: an explicit gated MLP component used in modality representation or fusion.
- `cross-attention-pooler`: learned queries pool modality tokens by cross-attention; distinct from tabular row pooling.
- `graph-attention`: graph-attention/message-passing blocks contextualizing structured row graphlets.
- `target-tokens`: explicit candidate-target/name tokens interacting with features; observed training-label embeddings alone do not qualify.
- `retrieval-head`: an attention-based prediction head retrieving/aggregating labeled context examples.
- `pca`: PCA preprocessing at the embedding/model interface; this is a pipeline component, not a transformer block.
- `tree-predictor`: a downstream tree-based predictor consuming representations, not a synthetic pretraining prior.
- `benchmark-generator`: a benchmark/data-generation work without a proposed predictive model; MUST appear alone.

Tabular attention/pooling tags MUST be accompanied by `tabular-transformer`. Do not use a generic transformer tag that obscures the distinction between text and tabular blocks. Both families may apply to the same paper; a tabular model consuming text embeddings is not thereby a text transformer. Use only disclosed components, state variant-specific scope, and do not infer a text transformer merely because a model accepts strings. Changes to this vocabulary MUST update these rules, the linter, tests, and affected notes/catalogue together.

### PDF manifest

Every downloaded PDF MUST have exactly one manifest entry, with exactly these fields:

- `file`: path relative to the collection, such as `raw/2026_paper_slug.pdf`.
- `note`: the paper note filename, such as `2026-paper-slug.md`.
- `title`, `year_or_venue`, `source_url`: exactly matching note metadata.
- `pdf_url`: HTTP(S) URL used to download the PDF; pin the arXiv version when identified.
- `arxiv_version`: identified version such as `v2`, or null when unidentified/not applicable.
- `retrieved_on`: ISO `YYYY-MM-DD` retrieval date.
- `pages`, `bytes`: positive integer page count and byte size.
- `sha256`: lowercase SHA-256 checksum of the stored PDF bytes.

Verify PDF parsing and title identity against the source before adding or replacing a PDF. Downloaded bytes MUST remain unchanged during reorganization; moving a file updates paths without changing retrieval dates or checksums. Manifest entries MUST match the note, local bytes, page counts, and provenance. Do not fabricate download metadata.

### Required validation

Before completing any wiki change, run from the repository root:

```sh
./scripts/lint_related_works.sh
```

This uses `uv`, Python 3.10+, and Poppler's `pdfinfo` on PATH, with no third-party Python packages. The Python entry point is `uv run --no-project python scripts/lint_related_works.py`; `--root PATH` selects a repository for validation. The check is offline and read-only. Missing `pdfinfo` is a failure, not a reason to skip PDF validation.

The linter MUST pass: required metadata/sections and block-role explanations, unique note/catalogue/manifest coverage, matching titles/years/briefs and both tag families, valid tag combinations and tabular-block dependencies, wiki cross-links, repository local inline Markdown links and anchors, raw PDF placement, checksums, byte sizes, and PDF parsing/page counts. Linting does not verify remote availability, scientific claims, tag semantics, or whether PDF contents match the intended title; review those against the source.

When modifying the linter or its schema, also run its regression suite:

```sh
uv run --no-project python -m unittest discover -s tests -v
```

Update these canonical rules, linter checks, tests, and affected wiki records together when changing the format. No wiki change is complete until required validation passes.
