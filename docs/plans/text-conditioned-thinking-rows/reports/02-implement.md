# 02 implement: stories and parity-checked adapter

Implementer: omp/task, effort high. Nothing committed.

## Changed paths
- `pyproject.toml`, `uv.lock`: `uv add sentence-transformers` (6.2.0; pulls transformers 5.22.2, tokenizers 0.23.3). pytest was already a dev dependency.
- `src/texttabpfn/stories.py` (new): `schema_text(p)`, `story_text(roles)`, `full_text`, `scene_text(scene, variant, rng)` for `VARIANTS = story, omitted, irrelevant, wrong, one_wrong_fact`; builders `donor_scene`, `wrong_roles`, `one_wrong_fact_roles`; `story_sample_specs()` (fixed check sample).
- `src/texttabpfn/encoder.py` (new): `TextEncoder` (frozen all-MiniLM-L6-v2, per-text CPU cache, rejects > 256 tokens instead of truncating), `token_lengths`.
- `src/texttabpfn/adapter.py` (new): `TextConditionedThinkingRows`, `conditioned(model, adapter, e)` context manager, `episode_logits(model, adapter, e, (xs, ys, xq, yq))`.
- `src/texttabpfn/report.py`: new `stories` subcommand; `main` now returns the subcommand's status. `capacity` unchanged (re-run: same table, exit 0).
- `tests_exp/conftest.py`, `test_parity.py`, `test_label_isolation.py`, `test_stories.py` (new).
- MiniLM downloaded into the HF cache (`models--sentence-transformers--all-MiniLM-L6-v2`).

## Checks
- `uv run pytest tests_exp/test_parity.py` -> 15 passed (6 scenes, one per family, x {CPU, MPS}; plus non-zero-W and shape tests)
- `uv run pytest tests_exp/test_label_isolation.py` -> 6 passed (3 tests x {CPU, MPS})
- `uv run python -m texttabpfn.report stories` -> exit 0: 1250 scenes (125 per cell: families 0-3 x train and dev ranges, families 4-5 x eval range), max p 15, max schema+story 159 tokens, max over all corrupted texts 160 (limit 256, counts include [CLS]/[SEP])
- Extra: `uv run pytest tests_exp/test_stories.py` -> 49 passed (text contract, schema depends only on p, corruption builders)
- `uv run python -m texttabpfn.report capacity` -> unchanged table, exit 0

## Results
- Max MiniLM token length: 159 (schema+story); 160 over any variant. The report gates on the max over all variants, so corruptions also fit the window.
- Adapter: 64 x 192 x 384 = 4,718,592 trainable parameters (W only, no bias). θ is a non-persistent buffer (frozen, never in `state_dict`).
- Parity: W = 0 gives max |adapter - native| = 0 exactly on both CPU and MPS (measured over the 6 test scenes, n = 8 and 32, batch 3, 128 queries; native rerun on MPS also differed by 0). The test asserts <= 1e-5 on both devices; MPS is the experiments' device.

## Design notes for stages 3 and 5
- Encoder input: `schema + "\n" + story`; schema-only arm and `omitted` corruption = `schema_text(p)` (identical strings, so one cached embedding).
- e(t) is the model's standard sentence embedding: mean pooling then the pipeline's own L2 normalisation (all-MiniLM-L6-v2 ships a Normalize module).
- Story: one sentence per role: "... directly influence(s) y", "... is/are directly influenced by y" (MB children), "... are noisy measurements of one/a second hidden quantity that influences y", "... is/are unrelated to y". Names sorted; groups ordered by first column. Every feature is named exactly once.
- `wrong`: uniform permutation of the roles, resampled until the story text changes. `one_wrong_fact`: one feature moved to a different coarse class (mb / irrelevant / measure; mb gets a random parent/child relation, measure joins an existing group). `irrelevant`: donor of the same family and seed range, different seed and different story; its own feature names (may name columns beyond this scene's p).
- `set_text` accepts (D,) for one text or (B, D) for one text per batch member; rows are expanded over columns, so the output is identical across columns (tested).
- Label isolation: `episode_logits` takes the full episode; query labels never enter. The test also asserts support labels change logits (> 1e-2) and each query row is scored independently of the other query rows.

## NOT MEASURED / infra
- Nothing NOT MEASURED. No infra failures; no failed check attempts (no validation files).

## What in the brief was wrong
- Nothing material. The brief says "Parity must hold for both W = 0 and on the device the experiments use": read as "W = 0 parity on MPS"; parity was checked on both MPS and CPU.
- Thinking-row counts (64 x 192) were taken from the checkpoint, as the brief states.
