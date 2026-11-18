# Stage 04: Primary evaluation over support sizes

Goal: the headline H1 and H3 numbers on held-out scene families.
Box: 3h · Review: yes · Depends: 3

## Done when

- The story-minus-schema-only log-loss difference has a 95% scene-bootstrap CI at n = 8, 16, 32 — check: `uv run python -m texttabpfn.report primary` — today: fails, no module `texttabpfn`
- The same paired difference is reported for every n in 8 … 1024 — check: `uv run python -m texttabpfn.report curve` — today: fails, no module `texttabpfn`

## Facts

- 200 evaluation scenes from the held-out families. n ∈ {8, 16, 32, 64, 128, 256, 512, 1024}, with 5 nested support seeds and 512 fixed query rows per scene (D-08).
- Native, schema-only and story arms all score the same rows. Native is reported alongside.
- The metric is the per-scene mean query log loss. The paired difference is averaged by scene, with 1000 scene-bootstrap resamples (D-08).
- Support draws are stratified so that both classes appear; this rule is fixed before the run (D-08).
- Raw TabPFN path only (D-05). The MPS device is shared with stage 5, so the stages run one after the other.

## Not in this stage

- H2 label savings (B-03). Paraphrased stories (B-04). Wrapper ensembles (B-02).
