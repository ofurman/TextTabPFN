# Repository instructions

## Team conventions

- We use Vertica as our analytical database — prefer Vertica-specific SQL patterns.
- Use `uv` for Python tooling, not pip/pipx, unless there is a specific reason.
- OKE (Oracle Kubernetes Engine) kubeconfig scripts: use `get_oke_kubeconfig_v3`; v1 is legacy.

## Related works maintenance (mandatory)

This section is the single source of related-work maintenance requirements for the repository. Other documents must link here rather than copy or introduce competing rules.

`docs/related_works/README.md` is the single maintained collection index and briefing. Do not create a separate brief. `docs/related_works/manifest.json` records downloaded PDF provenance. Dated research notes may summarize findings, but do not replace the collection records.

Every related work added to the collection, including a PDF, source link, or reference-only entry, MUST receive both an index row and exactly one structured record in the collection README in the same change. Every record MUST include all five fields, using these exact labels:

1. **Problem and positioning:** the problem addressed and how the work relates to other approaches.
2. **Architecture:** the main components and information flow; identify pretraining, frozen components, and downstream fine-tuning when relevant. For benchmarks, describe the generator and evaluation setup instead.
3. **Input:** the actual data, metadata, labels, and context supplied to the method.
4. **Output:** the predictions, representations, distributions, or benchmark responses produced.
5. **For us:** the concrete relevance to combining text with tabular foundation models, including overlap, useful baselines, limitations, or proposed extensions.

Every index row and structured record MUST also contain matching text-source tags. Use a `Tags` column in the index and a `**Tags:**` field in the record. Apply all supported tags from this vocabulary:

- `row-level-text`: independently observed text associated with an example, such as a review, clinical note, or document. Serializing structured row values into sentences does not alone qualify.
- `column-label-semantics`: natural-language feature names, categorical/entity strings, or target/class names used for their meaning. Arbitrary category IDs and observed numeric class labels do not qualify.
- `task-domain-context`: explicit task instructions, table/domain descriptions, or causal stories supplied as input. Domain-specific pretraining alone does not qualify; target names alone belong under `column-label-semantics`.
- `no-text`: none of the three text-source categories applies to the described method. This tag MUST appear alone.

Classify actual input use in the source, not proposed extensions. State variant-specific scope and any uncertainty in the record; do not infer undisclosed text support. Tags MUST remain synchronized when a record is revised. The five explanatory fields remain mandatory in addition to the tags.

Each record MUST link to a primary source and its local PDF when downloaded. If a PDF is unavailable or a work is reference-only, explicitly state that status; do not use a broken or invented PDF link. The index and record MUST identify the same work and source.

Records MUST remain concise and grounded in the source. Distinguish reported findings from our interpretations, proposed extensions, and novelty claims. Explicitly mark missing, undisclosed, or unverified details instead of guessing. A required field may state that a detail is undisclosed; it must not be omitted.

Every downloaded PDF MUST have exactly one manifest entry with its filename, title, year/venue, source URL, PDF URL, retrieval date, arXiv version when identifiable (otherwise null), page count, byte size, and SHA-256 checksum. Verify PDF parsing and title identity before adding it. The file, index row, structured record, and manifest entry MUST stay synchronized in the same change; orphan PDFs and stale metadata are not allowed.

For revisions, update the existing work record and index row rather than duplicating the work; replace the PDF and refresh its manifest entry when applicable. For removals, remove the affected index row, record, PDF, and manifest entry together and fix incoming links. Supporting documentation changes do not require a paper record.

Before completing a collection change, verify that every indexed work has exactly one record with all five explanatory fields and matching valid tags, every local PDF is indexed and covered by the manifest, all local links resolve (including section anchors), and changed PDFs match their manifest metadata and checksums. No collection change is complete until these checks pass.
