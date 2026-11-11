# Stage 09 implement: role-token held-out evaluation

Status: done. All three done-when checks pass. Run 2026-11-11T19:09:38Z to 19:36:56Z, detached with nohup and resumable.
Launch/resume command (a restart keeps every complete scene): `cd /Users/ofurman/pwr/texttabpfn && nohup uv run python -u -m texttabpfn.role_eval >> results/09-role-eval/eval.log 2>&1 &`

## Changed paths
- `src/texttabpfn/role_eval.py` (new): the scoring run. It writes `results/09-role-eval/{config.json,rescore.csv,scores.csv,summary.json}`. `eval.log` is gitignored.
- `src/texttabpfn/roleinfo.py`: adds `role_preprocess_support_fit` and `preprocess_episode(..., fit=)`. The default stays the stage-7 support+query fit, and `role_preprocess` keeps its signature because `test_roleinfo` pins it.
- `src/texttabpfn/report.py`: adds the `role-primary`, `role-curve` and `role-reliability` commands.
- `tests_exp/test_role_eval.py` (new, 28 tests): corruption builders, support-only fit, report pass/fail on missing, NaN and drift.
- Not mine (left untouched): `docs/research/web/...attention.md` (modified) and `results/13-role-training-10k/` (untracked).

## Design
- Rows: `evaluation.scene_episodes(evaluation.Settings(), f, s)` over `eval_scene_specs(200)`, all 8 n.
- Scored arms: role story (stage 8, step 4000), role control (step 3750) and preproc-s (support-only fit). Preprocessing gets the whole 512-row batch; `query_logloss` chunks support seeds, not query rows.
- Corruptions use RNG stream (family, seed, 70, k). Irrelevant takes the donor's role sentences, truncated to p, or padded with the neutral sentence. 78 of 200 scenes were padded; iteration 1's donor story likewise said nothing about missing features. A donor equal to the correct roles is redrawn. Any corruption equal to the correct roles raises an error before scoring.
- References (native, iteration-1 story and schema, preproc) are read from stages 4 and 7. The first 8 scenes were re-scored at every n: max |diff| = 0 over 256 cells, and the report fails above 1e-6.

## Checks
- `uv run python -m texttabpfn.report role-primary` -> exit 0, verdict "supported"
- `uv run python -m texttabpfn.report role-curve` -> exit 0, reference drift 0.00e+00
- `uv run python -m texttabpfn.report role-reliability` -> exit 0
- `uv run pytest -q tests_exp` -> 178 passed
- Iteration-1 reports `primary`, `curve`, `reliability`, `roleinfo`, `role-training` -> exit 0 each

## role-primary (role-token story − control, 95% scene-bootstrap CI)
| n | control | story | story − control | family 4 | family 5 |
|---|---|---|---|---|---|
| 8 | 0.6254 | 0.6212 | −0.0042 [−0.0064, −0.0020] | −0.0040 [−0.0075, −0.0006] | −0.0043 [−0.0068, −0.0019] |
| 16 | 0.5764 | 0.5721 | −0.0042 [−0.0057, −0.0027] | −0.0033 [−0.0056, −0.0012] | −0.0051 [−0.0073, −0.0029] |
| 32 | 0.5270 | 0.5229 | −0.0041 [−0.0052, −0.0030] | −0.0037 [−0.0054, −0.0020] | −0.0045 [−0.0059, −0.0031] |

D-08 verdict: **supported** (upper bound < 0 at all three n). Iteration 1 at these n was "harmful".

## role-curve summary (paired differences, 95% CI)
- Story − control: −0.0023 at n=64, −0.0013 at 128, −0.0006 at 256. At 512 and 1024 the CI includes 0 (−0.0003 and +0.0002).
- Role story − native: −0.0080 / −0.0157 / −0.0124 at n = 8 / 16 / 32, −0.0061 at 64. From 256 up it is slightly worse than native (+0.0006 and +0.0008 at 512 and 1024, CIs above 0).
- Role story − preproc (support+query): +0.0084 / +0.0046 / +0.0033 at n = 8 / 16 / 32, CIs above 0. Preprocessing still wins at small n; the role tokens recover about half of preproc's gain over native at n = 8. From n = 256 on, role story beats preproc, which degrades at large n.
- Role story − iteration-1 story adapter: −0.0036 / −0.0042 / −0.0049 at n = 8 / 16 / 32, all CIs below 0.
- Role control − native: −0.0038 / −0.0115 / −0.0083 at n = 8 / 16 / 32. Part of the role-token arms' gain is generic adaptation, which control also gets.
- Reviewer P3: preproc-s − preproc = +0.0012 [−0.0001, +0.0026] / +0.0007 / +0.0006 at n = 8 / 16 / 32. The support-only fit keeps almost all of preproc's gain (preproc-s − native −0.0153 / −0.0197 / −0.0151), so that gain comes from the role facts, not from unlabeled query rows.

## role-reliability (corrupted − correct roles, role-token story adapter)
| n | correct | omitted | irrelevant | wrong |
|---|---|---|---|---|
| 8 | 0.6212 | +0.00366 [+0.00199, +0.00541] | +0.00784 [+0.00465, +0.01118] | +0.00807 [+0.00574, +0.01038] |
| 16 | 0.5721 | +0.00460 [+0.00341, +0.00587] | +0.00595 [+0.00412, +0.00784] | +0.00582 [+0.00399, +0.00781] |
| 32 | 0.5229 | +0.00348 [+0.00271, +0.00440] | +0.00475 [+0.00371, +0.00585] | +0.00462 [+0.00339, +0.00593] |

Every corruption hurts, so the adapter reads the roles. Omitted costs about the story − control gap. Irrelevant and wrong roles cost more than omitted, but stay below +0.009.

## Wall time and NOT MEASURED
- About 35 min in total: about 3 min build, 27.3 min run (rescore sample about 1 min, then 200 scenes at 6–10 s each), then checks. NOT MEASURED: nothing.

## What in the brief was wrong
- Nothing material.
- The brief left open how to pad irrelevant roles when the donor has fewer features. I padded with the neutral sentence.
- `test_roleinfo` pins `role_preprocess`'s signature, so the support-only fit is a separate function and not a keyword argument.
- Stage 7 already passed the whole query batch to preprocessing, so the "before any chunking" requirement held at HEAD.
