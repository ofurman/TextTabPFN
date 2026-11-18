"""Stage 4 evaluation contract: held-out scenes, shared nested supports, strict reports."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from texttabpfn import report
from texttabpfn.evaluation import (
    ARMS,
    CONFIG_JSON,
    SCORE_FIELDS,
    SCORES_CSV,
    Settings,
    _complete_scenes,
    _settings_json,
    scene_episodes,
)
from texttabpfn.scenes import EVAL_SEED_RANGE, HELDOUT_FAMILIES, Scene, eval_scene_specs
from texttabpfn.stories import scene_text

SETTINGS = Settings()


def test_eval_scenes_are_held_out_and_balanced() -> None:
    specs = eval_scene_specs(SETTINGS.num_scenes)
    assert len(set(specs)) == len({s for _, s in specs}) == 200
    assert all(f in HELDOUT_FAMILIES and EVAL_SEED_RANGE[0] <= s < EVAL_SEED_RANGE[1] for f, s in specs)
    assert sum(f == 4 for f, _ in specs) == sum(f == 5 for f, _ in specs) == 100


def _rows(x: np.ndarray) -> set[bytes]:
    return {r.tobytes() for r in x}


@pytest.mark.parametrize("family, seed", eval_scene_specs(4))
def test_supports_are_nested_stratified_and_share_queries(family: int, seed: int) -> None:
    a, b = scene_episodes(SETTINGS, family, seed), scene_episodes(SETTINGS, family, seed)
    k = SETTINGS.support_seeds
    for n in SETTINGS.sizes:
        for x, y in zip(a.arrays[n], b.arrays[n], strict=True):
            np.testing.assert_array_equal(x, y)
        xs, ys, xq, yq = a.arrays[n]
        assert xs.shape[:2] == (n, k) and ys.shape == (n, k)
        assert xq.shape[:2] == (SETTINGS.query_rows, k) and yq.shape == (SETTINGS.query_rows, k)
        for j in range(k):
            assert set(np.unique(ys[:, j])) == {0, 1}
            np.testing.assert_array_equal(xq[:, j], xq[:, 0])
            assert not _rows(xs[:, j]) & _rows(xq[:, 0]), "support and query rows are disjoint"
    for small, large in zip(SETTINGS.sizes[:-1], SETTINGS.sizes[1:], strict=True):
        for j in range(k):
            assert _rows(a.arrays[small][0][:, j]) <= _rows(a.arrays[large][0][:, j])
    scene = Scene(family, seed)
    assert a.texts == {"schema": scene_text(scene, "omitted"), "story": scene_text(scene, "story")}


def test_support_seeds_do_not_depend_on_the_seed_count() -> None:
    family, seed = eval_scene_specs(1)[0]
    full = scene_episodes(SETTINGS, family, seed)
    fewer = scene_episodes(dataclasses.replace(SETTINGS, support_seeds=2), family, seed)
    for n in SETTINGS.sizes:
        np.testing.assert_array_equal(fewer.arrays[n][0], full.arrays[n][0][:, :2])


def test_d08_verdict() -> None:
    neg, pos, zero = (-0.02, -0.01), (0.01, 0.02), (-0.01, 0.01)
    assert report.d08_verdict({8: neg, 16: neg, 32: zero}) == "supported"
    assert report.d08_verdict({8: neg, 16: zero, 32: zero}) == "not supported"
    assert report.d08_verdict({8: neg, 16: neg, 32: pos}) == "harmful"
    assert report.d08_verdict({8: zero, 16: zero, 32: zero}) == "not supported"


def _write_results(out: Path, drop: tuple | None = None, nan: tuple | None = None) -> None:
    specs = eval_scene_specs(SETTINGS.num_scenes)
    (out / CONFIG_JSON).write_text(
        json.dumps({"settings": _settings_json(SETTINGS), "scenes": [list(s) for s in specs]})
    )
    rng = np.random.default_rng(0)
    with (out / SCORES_CSV).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SCORE_FIELDS)
        for f, s in specs:
            for n in SETTINGS.sizes:
                for arm in ARMS:
                    key = (f, s, n, arm)
                    if key == drop:
                        continue
                    value = "nan" if key == nan else f"{rng.uniform(0.2, 0.7):.6f}"
                    writer.writerow([f, s, n, arm, value])


@pytest.mark.parametrize("name", ["primary", "curve"])
def test_reports_pass_on_complete_results(tmp_path: Path, name: str, capsys) -> None:
    _write_results(tmp_path)
    assert report.main([name, "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    if name == "primary":
        assert "D-08 verdict: " in out
    else:
        assert all(f"\n{n:>5} " in out for n in SETTINGS.sizes)


@pytest.mark.parametrize(
    "name, drop, nan",
    [
        ("primary", (4, 2_000_000, 8, "story"), None),
        ("primary", None, (5, 2_000_199, 32, "schema")),
        ("curve", (5, 2_000_001, 1024, "native"), None),
        ("curve", None, (4, 2_000_198, 512, "story")),
    ],
)
def test_reports_fail_on_missing_or_non_finite_rows(tmp_path: Path, name, drop, nan) -> None:
    _write_results(tmp_path, drop=drop, nan=nan)
    assert report.main([name, "--results", str(tmp_path)]) == 1


def test_primary_ignores_large_n_but_curve_does_not(tmp_path: Path) -> None:
    _write_results(tmp_path, drop=(4, 2_000_000, 256, "story"))
    assert report.main(["primary", "--results", str(tmp_path)]) == 0
    assert report.main(["curve", "--results", str(tmp_path)]) == 1


def test_resume_keeps_only_complete_scenes(tmp_path: Path) -> None:
    _write_results(tmp_path, drop=(5, 2_000_001, 64, "schema"))
    done = _complete_scenes(SETTINGS, tmp_path)
    assert len(done) == 199 and (5, 2_000_001) not in done
    with (tmp_path / SCORES_CSV).open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 199 * len(SETTINGS.sizes) * len(ARMS)
    assert all((int(r["family"]), int(r["seed"])) != (5, 2_000_001) for r in rows)
