# Final review (omp/reviewer, high), diff 0412dca..HEAD excluding docs/plans

Verdict: correct. The run meets the goal and DoD 1–7. There are no P0–P2 findings and one P3. Confidence 0.85.

## DoD commands
- `report capacity` exits 0. Shared gain: 0.1616 [0.1455, 0.1759] at n = 8, 0.1235 at n = 16, 0.0749 at n = 32.
- `pytest tests_exp/test_parity.py tests_exp/test_label_isolation.py` gives 21 passed.
- `report primary` exits 0. Story − schema-only: −0.0008 at n = 8, +0.0001 at n = 16, +0.0012 [+0.0005, +0.0019] at n = 32. Printed verdict: `harmful`.
- `report curve` exits 0 and reports all 8 support sizes.
- `report reliability` exits 0. Max |story − stage-4 story| is 0 over 600 cells.
- The `Verdict:` grep finds the note.

## Integrity
- No weights are tracked (`git ls-files` shows no .pt, .ckpt or .safetensors files).
- Protected paths have no diff.
- Under `docs/research`, the only change is the added results note.

## Stage 3 (training.py)
- **Identical episodes.** Each episode is built once from (seed, index) and both arms receive it. Text sampling happens after all arrays are drawn.
- **Same budget and optimizer.** Both arms use the same Adam settings, lr schedule and steps, with 8-episode accumulation; each ran 32,000 episodes.
- **Training-text mix (D-09).** It is exactly 15% omitted and 10% one-wrong-fact (`_MIX`, assigned by a permutation per block of 20). The one-wrong-fact story always changes the text.
- **Checkpoint selection.** Dev scenes only (families 0–3, dev seed range). Step 0 is a candidate.
- **No systematic handicap for the story arm.** The cache is keyed on the text and the encoder is frozen. e(t) is L2-normalized and the lr is shared. The small residual between stories (same-p cosine 0.989) is a limit of the design, not a defect; it is covered by B-12.

## Stage 5 (reliability.py)
- **Corruptions are correct.**
  - Irrelevant: the donor is another same-family scene with a different story.
  - Wrong: roles are permuted until the text changes.
  - Each corruption has its own RNG stream.
- **Rows match stage 4.** They come from the full stage-4 Settings, which matters because the pool size depends on the largest n, and are checked equal.

## Stage 6 note
- All table numbers match the report output.
- The verdict follows D-08, and the precedence point is explained.
- The inference is labelled as such.

## Finding
- **P3.** Note l.113 claimed every number came from the report commands. The encoder-similarity figures on l.103 (0.964, 0.989, 0.185) come only from the orchestrator's journal diagnostic. Fix: state their provenance.
