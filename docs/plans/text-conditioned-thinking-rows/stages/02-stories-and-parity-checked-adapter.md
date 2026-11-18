# Stage 02: Stories and parity-checked adapter

Goal: every scene has schema and story text, and the adapter θᵢ + Wᵢ·e(t) starts exactly at native TabPFN-2.5.
Box: 4h · Review: no · Depends: 1

## Done when

- The adapter with W = 0 reproduces native raw logits within 1e-5 — check: `uv run pytest tests_exp/test_parity.py` — today: fails, no pytest or `tests_exp/`
- Changing query labels leaves adapter logits unchanged — check: `uv run pytest tests_exp/test_label_isolation.py` — today: fails, no pytest or `tests_exp/`
- Every scene's full input text fits within 256 MiniLM tokens — check: `uv run python -m texttabpfn.report stories` — today: fails, no module `texttabpfn`

## Facts

- Wᵢ is 192×384 per thinking row, zero-initialized; there are about 4.7M parameters in total. e(t) is the frozen, mean-pooled all-MiniLM-L6-v2 embedding (D-01, D-04).
- Install by replacing `model.add_thinking_rows`. θ stays frozen, and e(t) is cached per scene.
- Thinking rows are context rows, so the KV cache built in `TabPFNV2p5.forward` (`kv_out`) already holds the conditioned rows (tabpfn 9.2.0 `architectures/tabpfn_v2_5.py`).
- `_remove_constant_features` drops constant columns before grouping, which would shift the story's column references. The generator therefore emits continuous features only (D-06).
- Text contract for schema and story: D-07.

## Not in this stage

- Training (stage 3). A per-column conditioning arm (B-01). A shared or low-rank W (B-10).
