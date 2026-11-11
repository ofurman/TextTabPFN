# 07 review (omp/reviewer, high) — commit 241746f

Verdict: correct (confidence 0.9). `report roleinfo` honestly delivers DoD item 1 under D-12 and D-13. No P0–P2 findings.

## Checks
- `uv run python -m texttabpfn.report roleinfo` → exit 0, matching the implement report. `uv run pytest -q tests_exp/test_roleinfo.py` → 30 passed.
- A CPU re-score of the roles arm (40 scenes, n = 8 and 32) matches `scores.csv` within 6.7e-7.
- No labels are used.
  - `role_preprocess` reads only `x_support` and `x_query`; `preprocess_episode` passes `ys` and `yq` through.
  - Flipping the labels leaves the output identical.
- Rows are identical to stage 4.
  - Both arms use the same `scene_episodes(Settings(), ...)` arrays.
  - The report asserts the settings, the scene list, and native drift ≤ 1e-6.
- The paired per-scene difference uses the stage-4 bootstrap (1000 resamples, seed 0). The D-13 test `lo <= 0 <= hi` is correct.
- The role mapping is correct.
  - `describe()["roles"][j]` is column j.
  - Group members share one hidden parent.
  - Every measure column has |corr(member, hidden)| ≥ 0.2; every irrelevant column has |corr with y| ≤ 0.024.
  - In the output layout, each principal-component column sits at the group's first member and irrelevant columns are dropped.
- The report exits 1 when results are missing or non-finite, or when the method, arms, settings or scenes don't match, or when native drifts.

## P3 notes (nothing to fix)
1. **Part of the gain comes from the unlabeled query rows.** The PCA is fitted on support plus query rows, which D-12 allows. A 40-scene probe compared that with a support-only fit (preprocessed − native):

   | n | support + query fit | support-only fit |
   |---|---|---|
   | 8 | −0.0153 | −0.0125 |
   | 32 | −0.0132 | −0.0127 |

   The D-13 conclusion is unchanged. The write-up should call this arm "roles + unlabeled query features".
2. **Stage 9 must pass the full 512-row query batch to `preprocess_episode` before chunking.** The output depends on the whole query batch: sub-batching changed the features by up to 0.415. `query_logloss` chunks by support seed only, so stage 7 is unaffected.
