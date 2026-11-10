# Stage 01: Channel capacity of the thinking rows

Goal: measure the most any text-to-thinking-row mapping could gain. Optimize the rows per scene, before any text is involved.
Box: 6h · Review: yes · Depends: none

## Done when

- The shared-row gain over native at n = 8, 16, 32 has a 95% scene-bootstrap CI on 50 dev scenes — check: `uv run python -m texttabpfn.report capacity` — today: fails, no module `texttabpfn`
- The per-column-row gain at n = 8, 16, 32 has the same CI — check: `uv run python -m texttabpfn.report capacity --arm per-column` — today: fails, no module `texttabpfn`

## Facts

- `AddThinkingRows.row_token_values_TE` is a single 64×E parameter, expanded over all feature groups. No column embedding is added to it (source: tabpfn 9.2.0 `architectures/tabpfn_v2_5.py`).
- Data cells carry fixed per-index column embeddings (`_add_column_embeddings`, `load_column_embeddings`). This is the only handle a shared shift has for per-group facts (same file).
- Test rows read context through K/V head 0 (`k_BcNHD[:, :, :1]`). The KV cache stores only that head (same file).
- Gradients need `save_peak_memory_factor=None` (assert in `shared/chunked_evaluate.py`). `force_recompute_layer=True` checkpoints each block.
- The checkpoint is not gated (42,935,499 bytes), and the license is accepted (D-05). An MPS backend exists (`shared/torch_mps_backend.py`).
- Protocol: per dev scene, optimize θ with Adam on support/query draws from a 20k-row pool. Score on fixed disjoint evaluation rows with fixed support subsets. The labels are a diagnostic only (D-02).
- Generator and text-free scene contract: D-06. Stop rule after this stage: D-08.

## Not in this stage

- Stories, the text encoder, Wᵢ (stage 2). Per-layer conditioning (B-06).
