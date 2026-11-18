# Stage 10 implement: B-09 follow-on arm training (bge-base-en-v1.5)

Status: done. Both done-when checks pass. Training ran 2026-11-12T03:24:43Z to 06:36:06Z, detached with nohup and resumable.
Launch/resume command (a restart resumes from the last 250-step checkpoint and refuses to resume with different settings):
`cd /Users/ofurman/pwr/texttabpfn && nohup uv run python -u -m texttabpfn.followon >> results/10-followon-training/train.log 2>&1 < /dev/null &`
The draft ETA was about 06:45Z (2.7 s/step over the first 25 steps); the run finished at 06:36Z.

## Changed paths
- `src/texttabpfn/followon.py` (new): `BgeEncoder` (frozen bge, CLS pooling, L2-normalised, 768-d; rejects texts over 512 tokens; per-text cache), `FollowOnThinkingRows`, `FollowOnDevSet`, the token-length and cosine diagnostics, and the lockstep trainer `run`, which mirrors `role_training.run`.
- `src/texttabpfn/report.py`: new `followon-training [--validation] [--results]`. It prints the encoder facts, then reuses `print_training` / `print_training_validation`.
- `tests_exp/test_followon.py` (new, 18 tests).
- New results: `results/10-followon-training/{config.json,progress.csv,validation.csv,summary.json}`. `train.log` and `weights/` (817 MB) are gitignored.
- Unchanged: the MiniLM encoder, `adapter.py`, `training.py`, `role_training.py` and every earlier report command. Not mine, left untouched: `docs/research/web/...attention.md` and `briefs/11-...md`.

## Design and parameter count
- Adapter: θᵢ + Wᵢ·P·e(t). P is 384×768 with no bias, from the `nn.Linear` default init with seed 0, the same in both arms. W is 64×192×384 and zero-initialised, so W = 0 reproduces the native rows exactly. Rows are identical across columns.
- Trainable parameters per arm: 64·192·384 + 384·768 = **5,013,504**.
- Episodes come from `training.make_episode`, unchanged, so rows and story-arm texts are exactly stage 3's, including the D-09 mix (15/10/75 per block of 20). Story arm reads `ep.texts["story"]`; control reads `ep.texts["schema"]` (`omitted`). Dev scenes, sizes and dev-only selection are the same; step 0 is a candidate.
- bge runs on MPS during training (one batched encode of 16 texts per step). MPS and CPU embeddings differ by at most 3.3e-7.

## Encoder facts (in config.json; printed by the report)
- bge tokens over the 1250-scene sample of `report stories` (every variant): max schema+story 159, max of any text 160. The limit is 512.
- Pairwise cosine over 40 dev stories (schema+story): bge mean 0.9758, same-p 0.9936 (98 pairs), residual norm 0.1516. The same function with MiniLM gives 0.9641 / 0.9886 / 0.1849, reproducing the iteration-1 journal figures. bge stories are *more* alike than MiniLM's.

## Checks
- `uv run pytest tests_exp/test_followon.py` -> 18 passed: parity within 1e-5 on CPU and MPS (4 scenes × n = 8, 32 × one text or one per batch member), query-label isolation, text dependence.
- `uv run python -m texttabpfn.report followon-training` -> exit 0: story 32000 / control 32000 episodes, last logged step 4000 for both.
- `... followon-training --validation` -> exit 0: 17 checkpoints, all rows present.
- `uv run pytest -q tests_exp` -> 200 passed.
- 19 earlier report commands -> exit 0 each:
  - `capacity` (both arms) and `stories`;
  - `training`, `role-training` and the stage-13 `role-training --results` form, each with and without `--validation`;
  - `primary`, `curve`, `reliability`, `roleinfo`, `role-primary`, `role10k-primary`, `role-curve` and `role-reliability`.

## Budget, lr, wall time
- `training.Settings()` defaults, i.e. iteration 1's: 4000 steps × 8 episodes, Adam lr 1e-3, 100 warm-up steps, cosine decay to 0.1×, validation every 250 steps. No lr pilot was run.
- Train wall 10848 s (3.01 h, about 2.7 s/step). End-to-end 3 h 11 min, including 17 validations of about 56 s each.

## Selected checkpoints (dev log loss, 64 dev scenes; mean over n = 8 / 32 / 128)
| arm | step | n=8 | n=32 | n=128 | mean |
|---|---|---|---|---|---|
| native | — | 0.5634 | 0.4279 | 0.3479 | 0.4464 |
| bge story | 3000 | 0.5511 | 0.4181 | 0.3441 | 0.4377 |
| bge control | 3000 | 0.5507 | 0.4183 | 0.3441 | 0.4377 |

- Story − control on dev: +0.0004 / −0.0002 / 0.0000. Over the last 2000 steps the arms stay within about 0.003 of each other.
- Earlier selections for reference: iteration-1 MiniLM story 0.4387 / schema 0.4376; stage-8 role tokens story 0.4335 / control 0.4381.
- [INFERENCE] On dev, bge story text gives no gain over schema text, so a held-out benefit is unlikely. Stage 11 measures it.

## NOT MEASURED
- Held-out evaluation (stage 11, out of scope).

## What in this brief was wrong
- Nothing material.
- At random W = 0.05 (the role-token tests' scale), bge story and corrupted texts move logits by only about 4e-4, under the 1e-3 bar. The text-dependence test therefore uses W = 0.5. This follows from bge's 0.994 same-p cosine; it is not a weaker test.
- `sentence-transformers` 6.2 has no `Pooling.get_pooling_mode_str`, so the CLS check reads `get_config_dict()["pooling_mode"]`.
