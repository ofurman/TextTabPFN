"""Stage 7 contract: label-free story-role preprocessing (D-12) and a strict report."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from texttabpfn import evaluation, report, roleinfo
from texttabpfn.scenes import Scene, eval_scene_specs

SCENES = eval_scene_specs(evaluation.Settings().num_scenes)
SIZES = evaluation.Settings().sizes


def _scene_rows(family: int, seed: int, n: int = 16, k: int = 3, m: int = 64):
    """Unlabeled support (n, k, p) and query (m, k, p) rows of one scene, plus its roles."""
    scene = Scene(family, seed)
    rng = np.random.default_rng([family, seed, 999])
    xs = np.stack([scene.sample(n, rng)[0] for _ in range(k)], axis=1)
    xq = np.repeat(scene.sample(m, rng)[0][:, None], k, axis=1)
    return scene.describe()["roles"], xs, xq


def _output_sources(roles: list[dict]) -> list[int]:
    """Input column behind each output column: mb columns and each group's first member."""
    seen, sources = set(), []
    for j, r in enumerate(roles):
        if r["role"] == "mb":
            sources.append(j)
        elif r["role"] == "measure" and r["group"] not in seen:
            seen.add(r["group"])
            sources.append(j)
    return sources


def test_preprocessing_never_reads_labels() -> None:
    assert list(inspect.signature(roleinfo.role_preprocess).parameters) == ["roles", "x_support", "x_query"]
    family, seed = SCENES[0]
    arrays = evaluation.scene_episodes(evaluation.Settings(), family, seed).arrays[16]
    roles = Scene(family, seed).describe()["roles"]
    xs, ys, xq, yq = arrays
    out = roleinfo.preprocess_episode(roles, arrays)
    flipped = roleinfo.preprocess_episode(roles, (xs, 1 - ys, xq, 1 - yq))
    assert out[1] is ys and out[3] is yq
    np.testing.assert_array_equal(out[0], flipped[0])
    np.testing.assert_array_equal(out[2], flipped[2])


@pytest.mark.parametrize("scene", SCENES[:10])
def test_irrelevant_columns_are_removed_and_mb_columns_kept(scene) -> None:
    roles, xs, xq = _scene_rows(*scene)
    out_s, out_q = roleinfo.role_preprocess(roles, xs, xq)
    num_mb = sum(r["role"] == "mb" for r in roles)
    num_groups = len({r["group"] for r in roles if r["role"] == "measure"})
    assert out_s.shape == (*xs.shape[:2], num_mb + num_groups) and out_q.shape[-1] == out_s.shape[-1]
    assert out_s.dtype == xs.dtype and np.isfinite(out_s).all() and np.isfinite(out_q).all()
    sources = _output_sources(roles)
    for c, j in enumerate(sources):
        if roles[j]["role"] == "mb":
            np.testing.assert_array_equal(out_s[..., c], xs[..., j])
            np.testing.assert_array_equal(out_q[..., c], xq[..., j])
    irrelevant = [j for j, r in enumerate(roles) if r["role"] == "irrelevant"]
    assert irrelevant
    rng = np.random.default_rng(1)
    xs2, xq2 = xs.copy(), xq.copy()
    xs2[..., irrelevant] = rng.normal(size=xs2[..., irrelevant].shape) * 5
    xq2[..., irrelevant] = rng.normal(size=xq2[..., irrelevant].shape) * 5
    out2_s, out2_q = roleinfo.role_preprocess(roles, xs2, xq2)
    np.testing.assert_array_equal(out_s, out2_s)
    np.testing.assert_array_equal(out_q, out2_q)


@pytest.mark.parametrize("scene", SCENES[:10])
def test_each_measurement_group_collapses_to_one_sign_robust_column(scene) -> None:
    roles, xs, xq = _scene_rows(*scene)
    out_s, out_q = roleinfo.role_preprocess(roles, xs, xq)
    sources = _output_sources(roles)
    groups = sorted({r["group"] for r in roles if r["role"] == "measure"})
    for g in groups:
        members = [j for j, r in enumerate(roles) if r["role"] == "measure" and r["group"] == g]
        col = sources.index(members[0])
        # Changing one member changes exactly the group's column.
        xs2 = xs.copy()
        xs2[..., members[-1]] += np.random.default_rng(2).normal(size=xs2.shape[:2])
        changed = np.abs(roleinfo.role_preprocess(roles, xs2, xq)[0] - out_s).max(axis=(0, 1)) > 1e-6
        assert changed.tolist() == [c == col for c in range(out_s.shape[-1])]
        # Flipping a non-first member's sign leaves the column unchanged.
        flip_s, flip_q = xs.copy(), xq.copy()
        flip_s[..., members[1]] *= -1
        flip_q[..., members[1]] *= -1
        res_s, res_q = roleinfo.role_preprocess(roles, flip_s, flip_q)
        np.testing.assert_allclose(res_s, out_s, atol=1e-5)
        np.testing.assert_allclose(res_q, out_q, atol=1e-5)


def test_collapse_recovers_a_hidden_quantity_measured_with_opposite_signs() -> None:
    rng = np.random.default_rng(3)
    rows, k = 400, 2
    h = rng.normal(size=(rows, k))
    x = np.stack(
        [h + 0.3 * rng.normal(size=(rows, k)), -h + 0.3 * rng.normal(size=(rows, k)), rng.normal(size=(rows, k))],
        axis=-1,
    ).astype(np.float32)
    roles = [{"role": "measure", "group": 0}, {"role": "measure", "group": 0}, {"role": "mb", "group": None}]
    out_s, out_q = roleinfo.role_preprocess(roles, x[:100], x[100:])
    collapsed = np.concatenate([out_s, out_q])[..., 0]
    for e in range(k):
        assert np.corrcoef(collapsed[:, e], h[:, e])[0, 1] > 0.95
        # A plain mean would cancel: its correlation with h is near zero.
        assert abs(np.corrcoef(x[:, e, :2].mean(-1), h[:, e])[0, 1]) < 0.3


def test_support_seeds_are_fitted_independently() -> None:
    roles, xs, xq = _scene_rows(*SCENES[3])
    out_s, out_q = roleinfo.role_preprocess(roles, xs, xq)
    for e in range(xs.shape[1]):
        one_s, one_q = roleinfo.role_preprocess(roles, xs[:, e : e + 1], xq[:, e : e + 1])
        np.testing.assert_allclose(one_s[:, 0], out_s[:, e], atol=1e-6)
        np.testing.assert_allclose(one_q[:, 0], out_q[:, e], atol=1e-6)


def _write_results(
    out: Path, drop: tuple | None = None, nan: tuple | None = None, native_shift: float = 0.0,
    method: str = roleinfo.METHOD,
) -> tuple[Path, Path]:
    rol_dir, pri_dir = out / "07", out / "04"
    rol_dir.mkdir()
    pri_dir.mkdir()
    settings = evaluation._settings_json(evaluation.Settings())
    (rol_dir / roleinfo.CONFIG_JSON).write_text(
        json.dumps(roleinfo._config_core() | {"method": method})
    )
    (pri_dir / evaluation.CONFIG_JSON).write_text(
        json.dumps({"settings": settings, "scenes": [list(s) for s in SCENES]})
    )
    rng = np.random.default_rng(0)
    with (rol_dir / roleinfo.SCORES_CSV).open("w", newline="") as rol, (
        pri_dir / evaluation.SCORES_CSV
    ).open("w", newline="") as pri:
        rol_w, pri_w = csv.writer(rol), csv.writer(pri)
        rol_w.writerow(roleinfo.SCORE_FIELDS)
        pri_w.writerow(evaluation.SCORE_FIELDS)
        for f, s in SCENES:
            for n in SIZES:
                values = {arm: f"{rng.uniform(0.2, 0.7):.6f}" for arm in (*evaluation.ARMS, roleinfo.ROLES)}
                for arm in evaluation.ARMS:
                    shift = native_shift if arm == evaluation.NATIVE else 0.0
                    pri_w.writerow([f, s, n, arm, f"{float(values[arm]) + shift:.6f}"])
                for arm in roleinfo.ARMS:
                    key = (f, s, n, arm)
                    if key != drop:
                        rol_w.writerow([f, s, n, arm, "nan" if key == nan else values[arm]])
    return rol_dir, pri_dir


def _main(rol_dir: Path, pri_dir: Path) -> int:
    return report.main(["roleinfo", "--results", str(rol_dir), "--primary", str(pri_dir)])


def test_report_passes_on_complete_results(tmp_path: Path, capsys) -> None:
    assert _main(*_write_results(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "preproc - native [95% CI]" in out and "D-13: 95% CI includes 0" in out
    assert "family 4 (100 scenes)" in out and "family 5 (100 scenes)" in out
    assert all(f"\n{n:>5} " in out for n in SIZES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": (4, 2_000_000, 8, "roles")},
        {"drop": (5, 2_000_199, 1024, "native")},
        {"nan": (4, 2_000_100, 64, "roles")},
        {"native_shift": 1e-5},
        {"method": "plain mean"},
    ],
)
def test_report_fails_on_missing_non_finite_or_mismatched_rows(tmp_path: Path, kwargs) -> None:
    assert _main(*_write_results(tmp_path, **kwargs)) == 1


def test_resume_keeps_only_complete_scenes(tmp_path: Path) -> None:
    rol_dir, _ = _write_results(tmp_path, drop=(5, 2_000_001, 16, "roles"))
    done = roleinfo._complete_scenes(rol_dir)
    assert len(done) == 199 and (5, 2_000_001) not in done
    with (rol_dir / roleinfo.SCORES_CSV).open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 199 * len(SIZES) * len(roleinfo.ARMS)
