"""Stage 11: the bge follow-on held-out evaluation and its ``followon-primary`` report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from test_role_eval import SCENES, SIZES, _write_10k

from texttabpfn import evaluation, followon_eval, report


def test_rows_are_both_arms_at_every_n() -> None:
    rows = followon_eval.required_rows()
    assert {arm for _, arm in rows} == {followon_eval.STORY, followon_eval.CONTROL}
    assert {n for n, _ in rows} == set(SIZES)


def test_arms_read_their_training_texts() -> None:
    # Story reads stage 4's schema+story text, control the schema-only text.
    se = evaluation.scene_episodes(evaluation.Settings(), *SCENES[0])
    assert {arm: se.texts[t] for arm, t in followon_eval.TEXT.items()} == {
        followon_eval.STORY: se.texts["story"],
        followon_eval.CONTROL: se.texts["schema"],
    }
    assert se.texts["story"].startswith(se.texts["schema"])


def _write_followon(
    out: Path, drop: tuple | None = None, nan: tuple | None = None, rescore_shift: float = 0.0
) -> list[str]:
    """Synthetic stage 11 results next to synthetic stage 4, 7, 9 and 13 results."""
    args = _write_10k(out)
    paths = dict(zip(args[1::2], args[2::2], strict=True))
    stage9 = Path(paths["--stage9"])
    primary, roleinfo_dir = out / "04", out / "07"
    adapters, results = out / "10-train", out / "11-eval"
    adapters.mkdir()
    results.mkdir()
    selected = {"story": {"step": 3000}, "control": {"step": 3000}}
    (adapters / "summary.json").write_text(json.dumps({"selected": selected}))
    (results / followon_eval.CONFIG_JSON).write_text(json.dumps(followon_eval._config_core(adapters)))
    native = evaluation.read_scores(primary)
    with (results / followon_eval.RESCORE_CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(evaluation.SCORE_FIELDS)
        for f, s in SCENES[: followon_eval.RESCORE_SCENES]:
            for n in SIZES:
                w.writerow([f, s, n, evaluation.NATIVE, f"{native[(f, s, n, evaluation.NATIVE)] + rescore_shift:.6f}"])
    rng = np.random.default_rng(2)
    with (results / followon_eval.SCORES_CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(evaluation.SCORE_FIELDS)
        for f, s in SCENES:
            for n, arm in sorted(followon_eval.required_rows()):
                if (f, s, n, arm) != drop:
                    v = "nan" if (f, s, n, arm) == nan else f"{rng.uniform(0.2, 0.7):.6f}"
                    w.writerow([f, s, n, arm, v])
    return [
        "followon-primary",
        "--results", str(results),
        "--adapters", str(adapters),
        "--primary", str(primary),
        "--roleinfo", str(roleinfo_dir),
        "--stage9", str(stage9),
        "--role10k", paths["--results"],
        "--role10k-adapters", paths["--adapters"],
    ]  # fmt: skip


def test_followon_primary_passes_on_complete_results(tmp_path: Path, capsys) -> None:
    assert report.main(_write_followon(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "D-08 verdict: " in out and "role 10k story" in out and "MiniLM story" in out
    assert all(f"\n{n:>5} " in out for n in SIZES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": (4, 2_000_000, 32, followon_eval.CONTROL)},
        {"nan": (5, 2_000_001, 8, followon_eval.STORY)},
        {"drop": (4, 2_000_002, 1024, followon_eval.STORY)},
        {"rescore_shift": 1e-5},
    ],
)
def test_followon_primary_fails_on_missing_non_finite_or_drifted_rows(tmp_path: Path, kwargs) -> None:
    assert report.main(_write_followon(tmp_path, **kwargs)) == 1
