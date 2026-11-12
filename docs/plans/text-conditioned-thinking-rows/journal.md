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

## 2026-11-11T03:10Z 04 Primary evaluation: build accepted
- `uv run python -m texttabpfn.report primary` -> exit 0; story-schema -0.0008/+0.0001/+0.0012 at n=8/16/32; CI at 32 [+0.0005,+0.0019]; D-08 verdict "harmful".
- `uv run python -m texttabpfn.report curve` -> exit 0; both adapters beat native at n<=64 (n=16: -0.012); story slightly worse than native at n>=256 (+0.0006).
- Commit b97088c. Review dispatched; stage 5 starts in parallel (reviewer runs CPU-only, read-only).

## 2026-11-11T03:20Z 04 Primary evaluation: DONE
- Review: verdict correct, CPU re-score of 12 values equals scores.csv; no P0-P2.
- F1 (P3, verdict precedence harmful-before-supported) -> note in stage 6 results; no effect here. F2 (P3, report lacks checkpoint cross-check) -> backlog B-13.
- No fix round needed.

## 2026-11-11T03:20Z 04 re-plan
- Keep 05 (running) and 06. Stage 6 brief adds the precedence note and the ~1e-3 magnitude framing. Add B-13.

## 2026-11-11T03:25Z 05 Context reliability: DONE
- `uv run python -m texttabpfn.report reliability` -> exit 0; omitted-story +0.0015 [+0.0010,+0.0020] at n=8; irrelevant, wrong CIs include 0 at all n.
- Rows identical to stage 4 (max |diff| 0 over 600 cells). Commit 6b10c09.
- Reading: the adapter responds to story presence, not content. Brief error: rows must be built with full stage-4 settings (pool size depends on max n).

## 2026-11-11T03:25Z 05 re-plan
- Keep 06 as carded; brief gets stage-5 numbers. Cut nothing. Add nothing.

## 2026-11-11T03:40Z 06 Results note: DONE
- `grep -l '^Verdict:' docs/research/web/*-text-conditioned-thinking-rows-results.md` -> 1 file; `./scripts/lint_related_works.sh` -> passed (re-run once).
- Note quotes report commands only; inference and mechanism labelled. Commit d6cdcef. Final review next.

## 2026-11-11T03:50Z run complete
- Final review: correct, no P0-P2; P3 (encoder-diagnostic provenance) fixed by orchestrator in the results note.
- `./scripts/lint_related_works.sh` -> passed after the fix. All 6 stages DONE; see reports/final.md.

## 2026-11-11T04:10Z iteration 2 started
- Owner: "Do this as a followup: ... per-feature role tokens (backlog B-12). After that, try a stronger encoder (B-09) or a trained per-column arm (B-01)."
- Planned stages 7-12, DoD 1-7 (iteration 2), D-11..D-14; lint_plan OK; run continues under existing PROTOCOL.md, same roles and reviews setting.
- Added scope: stage 7 story-guided preprocessing as deciding number (D-12, D-13 stop rule).

## 2026-11-11T04:45Z 07 Value of story roles: build accepted
- `uv run python -m texttabpfn.report roleinfo` -> exit 0; preproc-native -0.0165/-0.0204/-0.0156 at n=8/16/32, all CIs < 0; D-13 does not fire.
- Native rescored = stage 4 native (max diff 0, 1600 cells). Preproc hurts at n>=256 (+0.013 at 1024). Commit 241746f.
- Review dispatched (CPU-only) in parallel with stage 8 (MPS); stage 8 does not consume stage 7 code.

## 2026-11-11T11:52Z clock correction
- Iteration-2 journal timestamps 04:10Z-04:45Z were estimates; real clock (date -u) at stage 8 start is 11:51Z. Stage 8 stop-by corrected to 17:50Z; all later times use date -u.

## 2026-11-11T11:58Z 07 Value of story roles: DONE
- Review: correct, no P0-P2. P3-1 (part of gain from transductive PCA on query features; support-only keeps ~80% at n=8) -> stage 9 and 12 briefs label the arm. P3-2 (pass full query batch) -> stage 9 brief.
- No fix round.

## 2026-11-11T11:58Z 07 re-plan
- Keep 08 (running), 09-12 as carded. Stage 9 adds a support-only-fit preprocessing variant as a secondary arm. Cut nothing.

## 2026-11-11T16:05Z 08 infra failure and relaunch
- Implementer died on DNS failure (getaddrinfo ENOTFOUND) after 1h; code built, full run never launched, no report.
- Kept: `uv run pytest -q tests_exp` -> 145 passed; all iteration-1 reports + roleinfo exit 0. Code committed eaee451.
- Pilots (/tmp/rt_pilots, 200 steps, 16 dev scenes): lr 3e-4/1e-3/3e-3 all within ±0.009 of native; 3e-3 unstable for control. Full run uses lr 1e-3 (iteration-1 setting).
- Orchestrator launched `nohup uv run python -m texttabpfn.role_training` directly (detached) so an API drop cannot stop it.

## 2026-11-11T19:12Z 08 Role-token training: DONE
- `uv run pytest tests_exp/test_role_tokens.py` -> 16 passed; `report role-training` -> exit 0, 32000 episodes each; `--validation` -> exit 0.
- Dev val n=8: story 0.5403, control 0.5509, native 0.5634 (iteration 1: arms tied). Story selected at step 4000 (still improving). Commit 26aacb7.

## 2026-11-11T19:12Z 08 re-plan
- Keep 09 as carded. Cut nothing. Add nothing.

## 2026-11-11T19:20Z 13 added (owner request)
- Owner: "Try also to train for 10k steps". Stage 13 added; D-15 fixes D-14 input to the 10k-step verdict, decided before any stage 9 held-out result.
- Launched `uv run python -m texttabpfn.role_training --steps 10000 --val-every 500 --out results/13-role-training-10k` detached at 19:17Z; shares MPS with stage 9 inference.
- Stage 10 now depends on 9 and 13. ETA for stage 13 training ~03:00Z.

## 2026-11-11T19:55Z 09 Role-token evaluation: build accepted
- `report role-primary` -> exit 0; story-control -0.0042/-0.0042/-0.0041 at n=8/16/32, all CIs < 0; D-08 "supported" (4000-step); both families.
- `report role-curve` -> exit 0; `report role-reliability` -> exit 0, all corruptions +0.0035..+0.0081 (CI > 0). Commit 5b8cd0d.
- Preprocessing still beats role story at n<=32 (+0.008 at n=8). D-14 waits for stage 13 per D-15. Review dispatched (CPU only).

## 2026-11-11T20:10Z 09 Role-token evaluation: DONE
- Review: correct, no P0-P2; selection-step difference favours the control, not the story arm.
- Correction to 19:20Z entry (F1): D-15 was decided while stage 9 was running (~80/200 scenes scored), without reading partial scores; it was an owner request.
- F2 fixed in reports/09-implement.md (worse than native only at n>=512). F3 -> stage 12 brief (corruption ranking is point-estimate only; give slot-change rates).

## 2026-11-11T20:10Z 09 re-plan
- Keep 13 (training, ETA ~03:00Z), then D-14 on 10k verdict, 10, 11, 12. Cut nothing. Add nothing.

## 2026-11-11T19:43Z timestamp correction
- Entries stamped 19:55Z and 20:10Z were written earlier (~19:40Z); stamps were estimates. From here, stamps come from date -u.
- 10k run: step 275 at 1358 s (≈4.9 s/step while sharing MPS with stage 9); now alone, expected ≈2.65 s/step, ETA ≈03:15Z.

## 2026-11-12T03:03Z 13 10k training done
- 80000 episodes/arm, 7.50 h; selected story step 9500 (dev 0.4334), control 6500 (dev 0.4364); 4000-step pair was 0.4335 vs 0.4381.
- Longer training mainly improved the control on dev. Commit 81c0182. Held-out evaluation dispatched.

## 2026-11-12T03:18Z 13 10k role tokens: DONE
- `report role-training --results results/13-role-training-10k` -> exit 0, 80000 each; `report role10k-primary` -> exit 0; -0.0098/-0.0080/-0.0057, D-08 "supported". Commit a3815c8.
- Gap widened vs 4000 steps from both sides: story -0.0027 at n=8, control +0.0029 at n=8 (10k control 0.6283 ≈ native 0.6292). Report this split, not just the gap.
- Card check 1 already passed before stage 13 (--results existed); noted, not amended.

## 2026-11-12T03:18Z D-14 applied
- Input per D-15: role10k-primary verdict "supported" -> B-09 (bge-base-en-v1.5 pooled stories vs schema control). B-01 stays in backlog. Budget and size: D-16.
