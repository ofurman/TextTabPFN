# 01 Channel capacity of the thinking rows (implementer omp/task, high; box 6 h, stop by 2026-11-10T23:20Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md` (constraints), `decisions.md` (D-01…D-10), and the stage card `stages/01-channel-capacity-of-the-thinking-rows.md`.

Goal: measure the upper limit of what any text→thinking-row mapping could gain, by optimizing TabPFN-2.5's thinking rows per scene with label access, before any text exists. This decides the D-08 stop rule.

## Done when

1. `uv run python -m texttabpfn.report capacity` prints, for the shared-row arm, gain = native − optimized query log loss at n = 8, 16, 32 with a 95% scene-bootstrap CI (1000 resamples) over 50 dev scenes, and exits 0 (today: fails, no module `texttabpfn`).
2. `uv run python -m texttabpfn.report capacity --arm per-column` prints the same table for the per-column arm and exits 0 (today: fails, no module `texttabpfn`).

Both commands read committed small results under `results/01-capacity/` (CSV/JSON) and must exit non-zero if any (arm, n) row is missing (D-10). They must not recompute the experiment.

## What to build (outcomes, design is yours)

- uv project at repo root: `pyproject.toml` (Python 3.12 pinned via `requires-python` and `.python-version`), `uv.lock`, package `src/texttabpfn/`, `tabpfn==9.2.0` pinned, `pytest` as a dev dependency (later stages run `uv run pytest tests_exp/...`). Gitignore checkpoints, caches and any large artifacts.
- Scene generator per D-06: random-DAG SCM, 6–15 continuous observed features, binary y from a thresholded latent; roles = Markov-blanket features, irrelevant features, groups of 2–3 noisy measurements of one hidden cause of y. Six mechanism families (mechanism class × noise regime), families 0–3 train/dev, 4–5 held out. Deterministic from (family, seed). The generator must also expose each scene's role assignment (stage 2 turns it into story text; no text in this stage). Keep the base rate of y away from extremes so n = 8 stratified draws are feasible.
- Raw TabPFN-2.5 path: load `tabpfn-v2.5-classifier-v2.5_default.ckpt` (Prior-Labs/tabpfn_2_5 on HF, not gated, 42,935,499 bytes, license accepted D-05) into the `TabPFNV2p5` module and call its `forward` directly, no sklearn wrapper, no ensembling. Binary logits = first two of the output classes.
- Native baseline and two optimized arms per dev scene:
  - shared: replace `row_token_values_TE` (64×192) with a per-scene trainable copy, initialized at the checkpoint value;
  - per-column: a per-scene trainable 64×C×192 tensor (C = feature groups + 1 target column), initialized at the expanded checkpoint value. Diagnostic only (D-02).
  - Optimize with Adam on support/query draws sampled from a large per-scene training pool (~20k rows); evaluate on fixed disjoint evaluation rows (512 query rows) with fixed, stratified support subsets at n = 8, 16, 32 (several support seeds per scene, averaged per scene). Backbone frozen.
- 50 dev scenes drawn from families 0–3 (not the seeds later used for training episodes; document the seed ranges in code).

## Facts (verified in the tabpfn 9.2.0 wheel; unzipped copy at /tmp/tpfn/tabpfn/architectures/tabpfn_v2_5.py)

- `TabPFNV2p5Config`: emsize 192, nlayers 24, nhead 3, features_per_group 3, num_thinking_rows 64.
- `AddThinkingRows.forward` expands `row_token_values_TE` identically over all C columns (feature groups + target column) and is applied after `_add_column_embeddings` (forward ~line 832), so thinking rows carry no column embedding. Data cells get fixed per-group-index column embeddings (`_add_column_embeddings`, `load_column_embeddings`).
- Test rows attend only to K/V head 0 (`k_BcNHD[:, :, :1]` in `AlongColumnAttention.forward`); the KV cache stores that head.
- Gradients require `PerformanceOptions.save_peak_memory_factor=None` (`shared/chunked_evaluate.py` asserts otherwise). `force_recompute_layer=True` wraps blocks in `torch.utils.checkpoint` to save memory.
- Input layout: `x` is (rows, batch, features) with train rows first, `y` is the train labels; feature preprocessing (normalization, NaN indicators, padding to groups of 3) happens inside `forward`. `_remove_constant_features` runs before grouping, so constant columns are dropped (continuous features avoid this).
- An MPS backend exists (`shared/torch_mps_backend.py`). Hardware: Apple M4 Pro, 48 GB, no CUDA. MPS is the only accelerator; fall back to CPU only for ops MPS lacks.
- Rough compute: one forward+backward on ~550 rows × ≤6 columns should be well under a second; budget the optimization steps so the full stage runs inside the box. Reducing steps or support seeds is fine if stated in the report.

## Not in this stage

- Stories, text encoder, Wᵢ adapter (stage 2). Training across scenes (stage 3). Per-layer conditioning (B-06). The sklearn wrapper or ensembles (B-02).
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, existing research notes, or the run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`). The working-tree edit to `docs/research/web/2026-11-04-text-tabular-attention.md` is not yours; leave it.
- No pushing, no remote compute, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/01-implement.md`, at most 60 lines: changed paths; each check as `command` -> result; the capacity numbers (both arms, n = 8/16/32, mean and CI); optimization settings actually used; anything NOT MEASURED (reason); infra failures and retries; what in this brief was wrong. Failed check attempts: `validation/01-<check>-<k>.txt` (≤4 KB each). No pasted logs.
