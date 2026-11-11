# Journal

Append-only. Entries of at most 5 lines. Evidence is `command` -> result on one line.

## 2026-11-10T17:18Z run started
- Reviews `flagged`, tripwire 3 h, harness omp. Implementer `omp/task` high, reviewer `omp/reviewer` high, helper `omp/sonic` medium.
- Owner's reply: {"model": "default", "reviews": "flagged"}. Defaults taken: omp task/reviewer/sonic agents on omp default model, tripwire 3 h, branch thinking-rows from 0412dca.
- DoD checks 1–7 unreachable only because code is unbuilt (planner dry-runs); a correct build can pass each. Base commit 0412dca.

## 2026-11-10T22:05Z 01 Channel capacity: build accepted
- Infra: implementer died on API outage after the full 50-scene run; orchestrator wrote reports/01-implement.md from committed results (no rerun needed).
- `uv run python -m texttabpfn.report capacity` -> exit 0; shared gain 0.162/0.124/0.075 at n=8/16/32, CIs exclude 0, stop rule not fired.
- `uv run python -m texttabpfn.report capacity --arm per-column` -> exit 0; 0.200/0.154/0.100. Commit bdeb972.
- Review: P1 (family 5 near-unlearnable, heavy-tail squares) fix; P3 (report accepts NaN/mismatch) fix. One fix round dispatched; families 0-3 must stay bit-identical.

## 2026-11-10T22:20Z 01 Channel capacity: DONE
- Fix1 (c9f99c5): interaction terms now w·2.54·tanh(xi)·tanh(xj); families 0-3 bit-identical to bdeb972 (200/200 scene hashes equal); family 5 native n=1024 logloss 0.533 -> 0.371 (family 1: 0.328).
- Re-run: both `report capacity` commands -> exit 0, numbers unchanged. Report now rejects NaN and (family, seed) mismatches.

## 2026-11-10T22:20Z 01 re-plan
- Keep 02-06 as carded; stop rule did not fire. Cut nothing.
- Add B-11: interaction measurement nodes with one parent can get a sign-dropping tanh(h)^2 term (12/73 low-MI measure columns in family 4).

## 2026-11-10T23:25Z 02 Stories and adapter: DONE
- `uv run pytest tests_exp/test_parity.py` -> 15 passed; `uv run pytest tests_exp/test_label_isolation.py` -> 6 passed (re-run once).
- `uv run python -m texttabpfn.report stories` -> exit 0, max schema+story 159 tokens over 1250 scenes. Commit 8cbb249.
- Adapter: W 64x192x384 = 4,718,592 params, no bias; e(t) is L2-normalized MiniLM mean pool.

## 2026-11-10T23:25Z 02 re-plan
- Keep 03 as carded; brief fills stage-2 API (adapter.conditioned, episode_logits, stories.scene_text variants). Cut nothing. Add nothing.

## 2026-11-10T23:30Z 03 diagnostic (orchestrator, read-only)
- MiniLM e(t) on 40 dev stories: story-story cos mean 0.964; same-p 0.989; residual norm after mean removal 0.185. Role info is a small residual of e(t).
- Step 500 dev validation: story ≈ schema-only (n=8 0.5566 vs 0.5549; native 0.5634). Risk: encoder bottleneck, not channel (stage 1 ceiling 0.16). Run continues as carded.

## 2026-11-11T02:45Z 03 Episodic training: DONE
- `uv run python -m texttabpfn.report training` -> exit 0, 32000 episodes each arm; `--validation` -> exit 0, 17 checkpoints, both select step 3250.
- Dev val mean: native 0.4464, story 0.4387, schema-only 0.4376; n=8 gain ~0.014 vs stage-1 ceiling 0.162. Commit bd92e7c.
- Brief corrections: force_recompute_layer was False (7.4 GB at n=1024); gradient accumulation over 8 single-episode passes.

## 2026-11-11T02:45Z 03 re-plan
- Keep 04 and 05 as carded; run 04 then 05 (shared MPS). Cut nothing. B-12 stands as the main follow-up if 04 is null.
