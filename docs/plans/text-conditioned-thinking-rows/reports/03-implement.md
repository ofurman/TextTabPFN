# 03 implement: episodic training of story and schema-only arms

Implementer: omp/task, effort high. Nothing committed. Training ran 2026-11-10T22:42Z to 2026-11-11T02:33Z and finished complete.

## Changed paths
- `src/texttabpfn/training.py` (new): lockstep episodic trainer for both arms, dev validation, checkpoints and selection; `load_adapter(model, arm)` loads an arm's selected weights for stages 4-5.
- `src/texttabpfn/report.py`: new `training` subcommand; `--validation` prints the per-checkpoint table.
- `tests_exp/test_training.py` (new, CPU only, 4 tests): episodes are deterministic and well formed (both classes in the support, schema text = `omitted`), the D-09 mix is exact, n covers the log range, schedule and checkpoint steps.
- `results/03-training/`: `config.json`, `progress.csv`, `validation.csv`, `summary.json` (21 KB total, can be committed). `weights/*.pt` (37 files, about 760 MB: 17 checkpoints per arm, `<arm>-selected.pt`, resume `state.pt`) and `train.log` are covered by the existing `.gitignore` (`*.pt`, `results/**/*.log`).

## Design
- One process trains both arms. Each episode is generated once from `(seed, index)` and fed to both arms, so they get the same scenes, n, rows, order and seeds. Each arm has its own Adam (betas 0.9/0.999, no weight decay) and both follow the same schedule.
- Episode: family uniform over 0-3, seed from `TRAIN_SEED_RANGE`, n = round(exp(U(ln 8, ln 1024))), stratified support (`capacity.stratified_index`), 512 query rows drawn separately (disjoint). Loss is the mean query cross-entropy on the raw path (`adapter.episode_logits`).
- Story arm text mix, exact per block of 20 episodes: 15 story, 3 `omitted`, 2 `one_wrong_fact`. The schema-only arm always reads `omitted` (the schema).
- Step = 8 episodes with gradient accumulation, one scene and one e(t) per forward. Batching episodes in the batch dimension gave no speed-up (4 episodes took 3.5-4x as long) and ran out of memory at n = 1024.
- Validation: 64 dev scenes (`dev_scene_specs(64)`, 16 per family). n = 8, 32, 128, with nested stratified supports from 256 fixed rows, 3 support seeds and 512 fixed query rows. Runs every 250 steps, 17 checkpoints including step 0. Step 0 is W = 0, which equals native. The native reference is recorded once.
- Selection: per arm, the checkpoint with the lowest mean dev loss over the three n values. Step 0 was a candidate. Held-out families and `EVAL_SEED_RANGE` were never touched.
- Resumable: `uv run python -m texttabpfn.training` restarts from `weights/state.pt` and drops CSV rows written after it. It refuses to resume if the settings differ. No resume was needed.

## Budget, lr, wall time
- 4000 steps x 8 episodes = 32,000 episodes per arm (64,000 forward/backward passes in total).
- Learning rate: Adam peak 1e-3, linear warm-up over 100 steps, then cosine decay to 1e-4.
- Chosen from dev-only pilots on 16 dev scenes, both arms:
  - constant 1e-2 for 80 steps: worse than native at n = 8 and n = 128;
  - constant 3e-3: better at n = 32, worse at n = 8;
  - 1e-3 for 30 steps: small gains at all n.
  The 1e-3 80-step pilot never launched (launcher glitch); the 30-step pilot stood in for it.
- Wall time: 3.65 h of training (`train_wall_s` 13,142) plus validation, 3 h 51 min end to end on MPS, about 3.3 s per step. Unrelated CPU jobs held the load average at 60-96 throughout.

## Results (dev scenes, mean query log loss; full table: `report training --validation`)
| | n=8 | n=32 | n=128 | mean |
|---|---|---|---|---|
| native (step 0) | 0.5634 | 0.4279 | 0.3479 | 0.4464 |
| story, selected step 3250 | 0.5497 | 0.4205 | 0.3459 | 0.4387 |
| schema-only, selected step 3250 | 0.5493 | 0.4187 | 0.3447 | 0.4376 |
- Both arms gain about 0.014 nats over native at n = 8, against a stage-1 per-scene ceiling of 0.162.
- On these dev scenes the schema-only arm matches or beats the story arm at every n and at most checkpoints. This is a dev observation only; stage 4 decides on held-out families.
- Training loss is nearly identical between arms (`progress.csv`). Selected ‖W‖_F is 101.2 (story) and 100.9 (schema).

## Checks
- `uv run python -m texttabpfn.report training` -> exit 0; story 32,000 and schema 32,000 episodes, both logged through step 4000.
- `uv run python -m texttabpfn.report training --validation` -> exit 0; 105 of 105 rows (native plus 17 checkpoints x 2 arms x 3 n); selected step 3250 for both arms, which matches `summary.json`.
- Negative probes on a temporary copy of the results, each -> exit 1:
  - schema episodes reduced by 8 -> "episode counts differ";
  - schema arm removed -> "arm(s) ['schema'] missing";
  - one validation row dropped -> "1 of 105 checkpoint rows missing".
- `uv run pytest -q tests_exp` -> 74 passed: parity, label isolation, stories, and the new training tests.
- `uv run python -m texttabpfn.report capacity` -> exit 0 (unchanged).

## NOT MEASURED / infra
- Nothing NOT MEASURED.
- No failed check attempts, so there are no validation files.
- Infra: one sweep process (lr 1e-3) never started; it was not retried.
- No crash, no resume, no MPS OOM in the main run.

## What in the brief was wrong
- "An episode needs about 0.1-1 GB with `force_recompute_layer=True`": the stage 2 raw path uses `force_recompute_layer=False`. Measured peak was about 1.4 GB at n = 8 and 7.4 GB at n = 1024 per episode, which fits easily, so recompute was not needed.
- The stage card says "query BCE"; the loss used is 2-class cross-entropy on the binary logits, which is the same objective.
