# 10 Follow-on arm training (implementer omp/task, high; box 6 h, stop by 2026-11-12T09:40Z)

Repo: /Users/ofurman/pwr/texttabpfn, branch `thinking-rows`. Read `docs/plans/text-conditioned-thinking-rows/index.md`, `decisions.md` (D-01, D-04, D-09, D-11, D-14, D-15) and the stage card `stages/10-follow-on-arm-training.md`.

Goal: train the follow-on arm selected by D-14 (B-09, stronger pooled encoder) and its matched control on the training families, with the same episodes and budget.

## Done when

1. `uv run pytest tests_exp/test_followon.py` passes. Today: exit 4, file not found. The tests must show three things:
   - At initialization the follow-on adapter reproduces native raw TabPFN-2.5 logits within 1e-5, on CPU and on MPS.
   - Query labels do not change its logits.
   - With nonzero weights, logits depend on the conditioning input.
2. `uv run python -m texttabpfn.report followon-training` shows both follow-on arms completed the same number of episodes, and exits non-zero if an arm is missing or the counts differ. Also provide `--validation`. Today: exit 2, invalid choice.

Reports read committed small CSV/JSON under `results/10-followon-training/` and never recompute. Weights stay local and gitignored.

## Branch

Branch B-09, fixed by D-16:
- Encoder: frozen `BAAI/bge-base-en-v1.5`, CLS pooling, L2-normalized, 768-d, 512-token limit.
  - Download it into the HF cache; this is allowed by index.md.
  - Cache embeddings per text, as `texttabpfn.encoder` does for MiniLM.
  - Leave the MiniLM encoder and every earlier stage unchanged.
  - Check that the longest schema+story text fits in 512 bge tokens, and record the maximum.
- Adapter: θᵢ + Wᵢ·P·e(t).
  - P (384×768) is shared and learned, initialized from a fixed seed.
  - Wᵢ (64×192×384) is zero-initialized, so parity at initialization is exact.
  - The rows are identical across columns, as in iteration 1. State the parameter count, which should be about 5.0M.
- Arms:
  - story = schema + story text (`stories.scene_text(scene, variant, rng)`, with the same D-09 mix as stage 3);
  - control = schema text (`omitted`).
  - Both start from identical weights.
- Optional diagnostic for the report: the iteration-1 same-p story cosine was 0.989 with MiniLM. Report the same statistic for bge on 40 dev stories.

## Shared requirements

- Reuse `texttabpfn.training` and `texttabpfn.role_training` patterns:
  - the same episode stream (`make_episode`), the D-09 mix and dev-only checkpoint selection;
  - lockstep arms with identical episodes, optimizer and lr.
- Training budget: 4000 steps × 8 episodes, Adam peak lr 1e-3, 100 warm-up steps, cosine to 0.1×, validation every 250 steps. This matches iteration 1 exactly (D-16). Run detached with `nohup ... &`, resumable, and write the draft report (with the launch/resume command) before waiting. The API connection has dropped twice in this run.
- Keep every earlier report command and `tests_exp` passing. Do not change the behaviour of earlier stages.

## Facts

- MPS is the only accelerator. An episode needs about 7.4 GB at n = 1024. The 10k-step role-token run took 2.7 s/step running alone.
- Stage 9 (4000 steps): role-token story − control = −0.0042 / −0.0042 / −0.0041 at n = 8 / 16 / 32, verdict supported.
- Stage 13 (10k-step role tokens): story − control −0.0098 / −0.0080 / −0.0057, verdict supported. This verdict selected B-09 under D-14/D-15.
- Iteration-1 baseline for this arm (MiniLM, same architecture, 4000 steps; `report primary`): story − schema = −0.0008 / +0.0001 / +0.0012 at n = 8 / 16 / 32, verdict "harmful".
- Iteration-1 adapter code: `texttabpfn.adapter.TextConditionedThinkingRows` and `texttabpfn.training`. Mirror or parameterize them; do not change their behaviour.

## Not in this stage

- The D-14 branch not selected (backlog). Held-out evaluation (stage 11). Per-layer conditioning (B-06).
- Do not touch `docs/related_works/`, `scripts/`, `tests/`, `AGENTS.md`, `README.md`, research notes, or run files (`state.json`, `journal.md`, `index.md`, `stages/`, `backlog.md`, `briefs/`, `decisions.md`). No pushing, no committing.

## Report

`docs/plans/text-conditioned-thinking-rows/reports/10-implement.md`, at most 60 lines. Include:
- changed paths;
- each check as `command` -> result;
- the design and parameter count;
- budget, lr and wall time;
- the selected checkpoints and their dev losses, with native alongside;
- anything NOT MEASURED;
- what in this brief was wrong.
