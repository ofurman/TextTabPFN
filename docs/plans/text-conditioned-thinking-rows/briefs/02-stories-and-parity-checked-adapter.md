# 02 Stories and parity-checked adapter (implementer omp/task, high; box 4 h, stop by 2026-11-11T02:25Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (constraints), `decisions.md` (D-01, D-04, D-06, D-07, D-09, D-10) and the stage card `stages/02-stories-and-parity-checked-adapter.md`.

Goal: every scene has schema text and story text, and the adapter θᵢ + Wᵢ·e(t) exists and starts exactly at native TabPFN-2.5.

## Done when

1. `uv run pytest tests_exp/test_parity.py` passes: the adapter with W = 0 reproduces native raw TabPFN-2.5 logits within 1e-5 (today: fails, no `tests_exp/`).
2. `uv run pytest tests_exp/test_label_isolation.py` passes: changing query labels leaves adapter logits unchanged (today: fails, no `tests_exp/`).
3. `uv run python -m texttabpfn.report stories` prints the max schema+story MiniLM token length over a fixed sample of scenes from every family (train, dev and held-out seed ranges, at least 1000 scenes) and exits 0 only if it is ≤ 256; non-zero otherwise (today: `report` has no `stories` subcommand).

Tests must test real behaviour on the real checkpoint (skip cleanly with a reason only if the checkpoint is absent from the HF cache; it is present). Parity must hold for both W = 0 and on the device the experiments use; check CPU too if MPS non-determinism exceeds 1e-5, and state which device the test uses. For label isolation, make the test meaningful: also assert that changing *support* labels does change the logits, so the test cannot pass trivially.

## What to build (outcomes; design is yours)

- Text, per D-07:
  - schema: neutral names x1…xp in column order plus the definition of the binary outcome y; no role hints; depends only on p.
  - story: templated structural facts from `Scene.describe()["roles"]`: which features relate directly to y (Markov blanket; parent/child may be stated as "influences y" / "is influenced by y"), which are unrelated, which groups measure one hidden quantity. No coefficients, signs, equations, noise levels, mechanism names or base rates.
  - corruption builders used later (D-09): omitted (schema only), irrelevant (another same-family scene's story, with its own feature names), wrong (feature roles permuted among this scene's features), and one-wrong-fact (exactly one feature's role changed) for training. Build them now so stages 3 and 5 only call them.
- Encoder: frozen `sentence-transformers/all-MiniLM-L6-v2` (D-04), mean-pooled 384-d e(t); download into the HF cache (allowed by index.md). Cache embeddings per text; the encoder never needs gradients. Pick the input text = schema + "\n" + story (schema alone for the schema-only arm and the omitted corruption).
- Adapter module replacing `model.add_thinking_rows` (see `texttabpfn.model.thinking_rows` context manager and `SharedThinkingRows`): θ frozen at the checkpoint value, per-row Wᵢ ∈ R^{192×384} (+ bias optional), zero-initialized, ≈4.7M params; output identical across columns (D-01). The same class serves both arms; only the text differs.
- Extend `texttabpfn.report` with `stories`; keep `capacity` unchanged.

## Facts

- Stage 1 code: `src/texttabpfn/{scenes,model,capacity,report}.py`. `scenes.Scene(family, seed)`, `.sample(n, rng)`, `.describe()`; columns are randomly permuted so position does not reveal role; seed ranges TRAIN/DEV/EVAL are disjoint; families 0–3 train/dev, 4–5 held out (interaction mechanism).
- Stage 1 result (`report capacity`): per-scene optimized shared rows gain 0.162 / 0.124 / 0.075 nats at n = 8 / 16 / 32 over native (0.552 / 0.483 / 0.419); per-column 0.200 / 0.154 / 0.100. The channel is real; stop rule did not fire. Stage 1 used lr 0.03 Adam on the 64×192 rows; `capacity.py` has stratified/nested support helpers worth reusing.
- Thinking rows span C = feature groups + 1 target column and carry no column embedding (tabpfn 9.2.0 `architectures/tabpfn_v2_5.py`, `AddThinkingRows.forward`). Thinking rows are context rows, so the KV cache built in `TabPFNV2p5.forward` (`kv_out`) already holds the conditioned rows.
- `_remove_constant_features` drops constant columns before grouping; generator features are continuous, so story column references stay aligned.
- pytest must be a dev dependency of the uv project (add it if stage 1 did not).

## Not in this stage

- Training (stage 3). A per-column conditioning arm (B-01). A shared or low-rank W (B-10). LLM paraphrases (B-04).
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`). Leave the uncommitted edit to `docs/research/web/2026-11-04-text-tabular-attention.md`. No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/02-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; max token length; adapter parameter count; anything NOT MEASURED (reason); infra failures; what in this brief was wrong. Failed check attempts: `validation/02-<check>-<k>.txt` (≤4 KB). No pasted logs.
