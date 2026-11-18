"""Stage 5 reliability contract: reproducible corruptions, stage 4 rows, strict report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from texttabpfn import evaluation, reliability, report
from texttabpfn.scenes import Scene, eval_scene_specs
from texttabpfn.stories import scene_text

SCENES = eval_scene_specs(reliability.Settings().num_scenes)


def test_texts_are_reproducible_and_every_corruption_changes_the_story() -> None:
    for family, seed in SCENES:
        texts = reliability.scene_texts(family, seed)
        assert texts == reliability.scene_texts(family, seed)
        scene = Scene(family, seed)
        assert texts["story"] == scene_text(scene, "story")
        assert texts["omitted"] == scene_text(scene, "omitted")
        assert len(set(texts.values())) == len(reliability.TEXTS)
    family, seed = SCENES[0]
    assert evaluation.scene_episodes(evaluation.Settings(), family, seed).texts["story"] == (
        reliability.scene_texts(family, seed)["story"]
    )


def test_corruptions_draw_from_independent_streams() -> None:
    family, seed = SCENES[1]
    rng = reliability.corruption_rng(family, seed, "wrong")
    assert reliability.scene_texts(family, seed)["wrong"] == scene_text(Scene(family, seed), "wrong", rng)


def _write_results(
    out: Path, drop: tuple | None = None, nan: tuple | None = None, primary_drop: tuple | None = None,
    story_shift: float = 0.0,
) -> tuple[Path, Path]:
    rel_dir, pri_dir = out / "05", out / "04"
    rel_dir.mkdir()
    pri_dir.mkdir()
    rows_settings = evaluation.Settings()
    (rel_dir / reliability.CONFIG_JSON).write_text(json.dumps({
        "settings": {"num_scenes": 200, "sizes": list(reliability.SIZES)},
        "row_settings": evaluation._settings_json(rows_settings),
        "texts": list(reliability.TEXTS),
        "checkpoint": {"step": 3250, "val_logloss_mean": 0.43},
        "training_story_mix_per_20": {"omitted": 3, "one_wrong_fact": 2, "story": 15},
        "scenes": [list(s) for s in SCENES],
    }))
    (pri_dir / evaluation.CONFIG_JSON).write_text(
        json.dumps({"settings": evaluation._settings_json(rows_settings), "scenes": [list(s) for s in SCENES]})
    )
    rng = np.random.default_rng(0)
    with (rel_dir / reliability.SCORES_CSV).open("w", newline="") as rel, (
        pri_dir / evaluation.SCORES_CSV
    ).open("w", newline="") as pri:
        rel_w, pri_w = csv.writer(rel), csv.writer(pri)
        rel_w.writerow(reliability.SCORE_FIELDS)
        pri_w.writerow(evaluation.SCORE_FIELDS)
        for f, s in SCENES:
            for n in rows_settings.sizes:
                values = {k: f"{rng.uniform(0.2, 0.7):.6f}" for k in (*reliability.TEXTS, *evaluation.ARMS)}
                for arm in evaluation.ARMS:
                    value = float(values[arm]) + (story_shift if arm == "story" else 0.0)
                    if (f, s, n, arm) != primary_drop:
                        pri_w.writerow([f, s, n, arm, f"{value:.6f}"])
                if n not in reliability.SIZES:
                    continue
                for text in reliability.TEXTS:
                    key = (f, s, n, text)
                    if key != drop:
                        rel_w.writerow([f, s, n, text, "nan" if key == nan else values[text]])
    return rel_dir, pri_dir


def _main(rel_dir: Path, pri_dir: Path) -> int:
    return report.main(["reliability", "--results", str(rel_dir), "--primary", str(pri_dir)])


def test_report_passes_on_complete_results(tmp_path: Path, capsys) -> None:
    assert _main(*_write_results(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "15% of story-arm episodes omitted" in out and "10% had one wrong fact" in out
    for c in reliability.CORRUPTIONS:
        assert f"{c} - story [95% CI]" in out
    assert all(f"\n{n:>5} " in out for n in reliability.SIZES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": (4, 2_000_000, 8, "wrong")},
        {"drop": (5, 2_000_199, 32, "story")},
        {"nan": (4, 2_000_100, 16, "irrelevant")},
        {"primary_drop": (5, 2_000_001, 32, "native")},
        {"story_shift": 0.01},
    ],
)
def test_report_fails_on_missing_non_finite_or_mismatched_rows(tmp_path: Path, kwargs) -> None:
    assert _main(*_write_results(tmp_path, **kwargs)) == 1


def test_report_ignores_stage4_rows_beyond_n32(tmp_path: Path) -> None:
    assert _main(*_write_results(tmp_path, primary_drop=(4, 2_000_000, 64, "story"))) == 0


def test_resume_keeps_only_complete_scenes(tmp_path: Path) -> None:
    rel_dir, _ = _write_results(tmp_path, drop=(5, 2_000_001, 16, "omitted"))
    done = reliability._complete_scenes(reliability.Settings(), rel_dir)
    assert len(done) == 199 and (5, 2_000_001) not in done
    with (rel_dir / reliability.SCORES_CSV).open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 199 * len(reliability.SIZES) * len(reliability.TEXTS)
