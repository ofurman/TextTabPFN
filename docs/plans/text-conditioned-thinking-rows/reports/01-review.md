# 01 review (omp/reviewer, high) — commit bdeb972

The stage-1 deliverable is correct. One generator fault in held-out family 5 would bias stages 2–5. Confidence 0.85.

## P1: family 5 (interaction, heavy-tailed noise) carries almost no feature signal
Location: `src/texttabpfn/scenes.py:178-181, 201-205, 226-235`.

Cause:
- The interaction mechanism adds products and squares of parents.
- With Student-t(3) noise, those squares have infinite variance.
- Std calibration over 4096 rows is driven by outliers, which shrinks the typical signal below the noise level, and this compounds down the DAG.

Evidence:
- Columns with mutual information with y below 0.005 nats, 40 scenes per family:
  - family 1: 8/159 Markov-blanket (mb) columns and 0/155 measurement columns;
  - family 4: 11/162 mb and 42/145 measurement;
  - family 5: 64/168 mb and 119/157 measurement.
- Median std of the linear signal reaching the latent outcome z: 0.999 in family 1, 0.762 in family 4, 0.41 in family 5.
- Native log loss at n = 1024 on evaluation seeds 2000000–2000009, against the prior entropy:
  - family 1: 0.322 vs 0.655;
  - family 3: 0.205 vs 0.676;
  - family 4: 0.402 vs 0.664;
  - family 5: 0.527 vs 0.649.

Impact:
- The stories would claim relevance that the data do not support.
- The story-minus-schema-only difference would be diluted toward 0.
- The reliability corruptions would be confounded.
- Stage-1 numbers come from families 0–3 and are unaffected.

Suggested fix: bound the interaction terms, for example with products of tanh(x), or use a robust scale. Then re-check how learnable family 5 is.

## P3: `report capacity` accepts non-finite or mismatched rows
Location: `src/texttabpfn/report.py:39-79`.
- It checks only that every (scene, n) key is present.
- A NaN or inf gain still exits 0.
- It never checks each row's (family, seed) against `config.json`.
- The current CSVs are clean.

## Checks that passed
- Both report commands reproduce the implement report exactly. Each CSV has 150 rows, with no duplicates and no config mismatches.
- Both thinking-row modules reproduce native logits exactly at init (CPU, scenes with 9, 9 and 15 features, max diff 0.0). They expand over C columns including the target column, add T to `single_eval_pos`, and set the `num_thinking_rows` attribute that `_decode` reads.
- No leakage:
  - the pool, validation, evaluation query and evaluation support rows each come from a distinct RNG stream;
  - support and query rows are disjoint within each episode;
  - rows are selected on validation data only.
- The bootstrap is a 1000-resample percentile CI over 50 per-scene gains. Each per-scene gain is the mean over 5 support seeds that share the same query rows.
- The report exits 1 on a missing config, a missing CSV, a wrong `num_scenes`, or a missing (scene, n).
- Markov-blanket structure is correct:
  - co-parents of children are drawn from the mb parents only;
  - measurement nodes hang off hidden roots, so they stay outside the blanket.
- Irrelevant columns: max |corr| with y is 0.025 over 706 columns, consistent with noise. Base rates are within 0.03 of nominal.
- Stage-2 interface: `describe()["roles"]` follows the permuted column order of X, and all features are continuous.
- Already noted in the implement report: in many scenes the best step was the last one (400), so capacity is underestimated, which is conservative.
