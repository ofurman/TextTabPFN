# 08 implement report

Written by the orchestrator. The implementer (omp/task, high) built the code, then died on a DNS failure (getaddrinfo ENOTFOUND) after its learning-rate pilots and before launching the full run. The orchestrator committed the code (eaee451) and launched the run detached.

## Changed paths
- `src/texttabpfn/role_tokens.py`: role sentences, slot tokens, `RoleTokenThinkingRows`.
  - Each feature gets one MiniLM token from a sentence stating only its role. The sentence never names the feature, and measurement groups are lettered A, B, C.
  - Tokens fill up to 15 slots in column order.
  - The shift is θ + W·concat_j(P t_j), with P: 384→25 shared across slots and seeded, and W = 64×192×375 starting at zero.
  - Trainable parameters: 4,617,600.
- `src/texttabpfn/role_training.py`: the lockstep trainer for the story and control arms.
  - It reuses stage 3's `make_episode` rows and D-09 mix. The one_wrong_fact roles are drawn from stream 50.
  - Validation uses stage 3's dev set.
  - `load_adapter(model, arm)` loads an arm's selected weights.
- `src/texttabpfn/training.py`:
  - `select_checkpoints` takes `arms`;
  - `parse_settings` is factored out of `main`.
  - Iteration-1 behaviour is unchanged; its reports still exit 0.
- `src/texttabpfn/report.py`: the `role-training [--validation]` report.
- `tests_exp/test_role_tokens.py`: 16 tests.
  - Parity within 1e-5 on CPU and MPS.
  - Query-label isolation.
  - Logits depend on the roles when W ≠ 0.
- `results/08-role-training/`: `config.json`, `progress.csv`, `validation.csv`, `summary.json`. Weights and the log are gitignored.

## Pilots (dev only, /tmp/rt_pilots; 200 steps, 16 scenes)
- Rates tried: 3e-4, 1e-3 and 3e-3. All stayed within ±0.009 of native; 3e-3 was unstable for the control.
- The full run used 1e-3, the iteration-1 setting, for both arms.

## Run
- 4000 steps × 8 episodes = 32,000 episodes per arm, Adam with peak lr 1e-3, 100 warm-up steps, cosine decay to 0.1×.
- Training took 2.92 h on MPS, finishing at 2026-11-11T19:05Z.

## Checks
- `uv run pytest tests_exp/test_role_tokens.py` -> 16 passed.
- `uv run python -m texttabpfn.report role-training` -> exit 0. Each arm ran 32,000 episodes.
- `uv run python -m texttabpfn.report role-training --validation` -> exit 0. 17 checkpoints.
- `uv run pytest -q tests_exp` -> 145 passed. All iteration-1 reports and `roleinfo` exit 0.

## Dev validation (64 scenes)

| arm | n = 8 | n = 32 | n = 128 | mean | selected step |
|---|---|---|---|---|---|
| native | 0.5634 | 0.4279 | 0.3479 | 0.4464 | – |
| story | 0.5403 | 0.4163 | 0.3440 | 0.4335 | 4000 |
| control | 0.5509 | 0.4193 | 0.3442 | 0.4381 | 3750 |

## Notes
- The story arm leads the control by 0.011 at n = 8. In iteration 1 the two arms were tied on these dev scenes.
- The story arm was selected at the last step and was still improving, so it is likely undertrained. This is conservative for the stage 9 verdict.
- Brief errors: none found.
