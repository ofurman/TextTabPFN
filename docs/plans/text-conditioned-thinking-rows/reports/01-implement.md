# 01 implement report

Written by the orchestrator. The implementer (omp/task, high) finished the experiment, then died on an API outage ("Couldn't connect to api.anthropic.com", 10 retries) before writing this report. Contents come from the committed files and the checks, re-run once.

## Changed paths
- `pyproject.toml`, `.python-version` (3.12), `uv.lock`. The project pins `tabpfn==9.2.0` (torch 2.15.0) and has a dev group with `pytest`.
- `.gitignore`: adds `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, tool caches, and `results/**/*.{npy,npz,log}`.
- `src/texttabpfn/scenes.py`: D-06 generator.
  - Roles are `mb`, with a parent/child relation, `irrelevant`, and `measure`, with a group.
  - Columns are randomly permuted.
  - The base rate is in [0.3, 0.7].
  - Families: linear, tanh, interaction × gaussian, t3. Families 4–5 are the interaction mechanism, held out.
  - Seed ranges are disjoint: TRAIN [0, 1e6), DEV [1e6, 1.1e6), EVAL [2e6, 2.1e6).
- `src/texttabpfn/model.py`:
  - checkpoint loader, with `GRAD_PERFORMANCE` setting save_peak_memory_factor to None;
  - `SharedThinkingRows` and `PerColumnThinkingRows`;
  - the `thinking_rows` swap context manager;
  - `binary_logits`.
- `src/texttabpfn/capacity.py`: the per-scene optimization runner, resumable, one CSV per arm.
- `src/texttabpfn/report.py`: the `capacity [--arm]` table, with a 1000-resample scene bootstrap (seed 0). It exits non-zero on a missing scene or n.
- `results/01-capacity/{config.json,shared.csv,per-column.csv}`. The logs are gitignored.

## Settings used (results/01-capacity/config.json)
- Data per scene:
  - 50 dev scenes (families 0–3, seeds 1000000–1000049) on MPS;
  - a 20k-row training pool;
  - 2048 validation rows;
  - 512 evaluation query rows, plus 256 evaluation support rows from which nested stratified n = 8, 16, 32 subsets are drawn, with 5 support seeds.
  - Each set is an independent RNG stream, so the evaluation rows never appear in training or validation.
- Optimization:
  - Adam, lr 0.03, 400 steps, 8 episodes per step, 128 training query rows.
  - Validation every 25 steps. The best step is kept, and step 0 (the checkpoint value) is a candidate.
  - Each scene gets one set of rows, used for every n.
- Wall time was about 155–240 s per scene per arm. The two arms ran concurrently, about 3.3 h in total.

## Checks
- `uv run python -m texttabpfn.report capacity` -> exit 0. Shared gain:
  - n = 8: 0.1616 [0.1455, 0.1759]
  - n = 16: 0.1235 [0.1075, 0.1397]
  - n = 32: 0.0749 [0.0641, 0.0859]
- `uv run python -m texttabpfn.report capacity --arm per-column` -> exit 0. Per-column gain:
  - n = 8: 0.2004 [0.1842, 0.2160]
  - n = 16: 0.1537 [0.1366, 0.1696]
  - n = 32: 0.1001 [0.0877, 0.1126]
- Native log loss is 0.5517, 0.4829 and 0.4185 at n = 8, 16 and 32.

## Notes
- The D-08 stop rule does not fire: the shared-row CI excludes 0 at all three n.
- The per-column arm gains about 1.25–1.35× the shared gain. That is more, but not much more, so B-01 stays in the backlog.
- In the logs, many scenes have best_step = 400, the last step. The capacity may therefore be underestimated, which is conservative for the stop rule.
- NOT MEASURED: capacity on held-out families 4–5, because the card scopes this stage to dev scenes.
- Brief errors: none found.
