"""Stage 9 contract: role corruptions, support-only preprocessing and strict reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from texttabpfn import evaluation, report, role_eval, roleinfo
from texttabpfn.role_tokens import NEUTRAL_SENTENCE, role_sentences, variant_sentences
from texttabpfn.scenes import Scene, eval_scene_specs
from texttabpfn.stories import donor_scene, scene_roles

SCENES = eval_scene_specs(evaluation.Settings().num_scenes)
SIZES = evaluation.Settings().sizes


@pytest.mark.parametrize("scene", SCENES[:20])
def test_corruptions_differ_from_correct_roles_and_fit_p(scene) -> None:
    sentences = role_eval.scene_sentences(*scene)
    assert sentences == role_eval.scene_sentences(*scene)  # a pure function of the scene
    p = Scene(*scene).num_features
    correct = sentences[role_eval.STORY]
    assert correct == variant_sentences(Scene(*scene), "story")
    assert sentences[role_eval.CONTROL] == [NEUTRAL_SENTENCE] * p
    assert sentences[role_eval.corrupted_label("omitted")] == [NEUTRAL_SENTENCE] * p
    for c in role_eval.CORRUPTIONS:
        corrupted = sentences[role_eval.corrupted_label(c)]
        assert len(corrupted) == p and corrupted != correct
    assert sorted(sentences[role_eval.corrupted_label("wrong")]) == sorted(correct)


def test_irrelevant_roles_come_from_a_same_family_donor() -> None:
    padded = 0
    for family, seed in SCENES[:40]:
        scene = Scene(family, seed)
        correct = variant_sentences(scene, "story")
        rng = role_eval.corruption_rng(family, seed, "irrelevant")
        while True:  # the first donor whose padded/truncated roles differ from the scene's
            donor = donor_scene(scene, rng)
            assert donor.family == family and donor.seed != seed
            expected = role_sentences(scene_roles(donor))[: scene.num_features]
            pad = scene.num_features - len(expected)
            if expected + [NEUTRAL_SENTENCE] * pad != correct:
                break
        got = role_eval.scene_sentences(family, seed)[role_eval.corrupted_label("irrelevant")]
        assert got == expected + [NEUTRAL_SENTENCE] * pad
        padded += pad > 0
    assert padded > 0


def test_support_fit_ignores_query_rows_and_labels() -> None:
    family, seed = SCENES[1]
    arrays = evaluation.scene_episodes(evaluation.Settings(), family, seed).arrays[16]
    roles = Scene(family, seed).describe()["roles"]
    xs, ys, xq, yq = arrays
    out = roleinfo.preprocess_episode(roles, arrays, fit=roleinfo.FIT_SUPPORT)
    assert out[1] is ys and out[3] is yq
    flipped = roleinfo.preprocess_episode(roles, (xs, 1 - ys, xq, 1 - yq), fit=roleinfo.FIT_SUPPORT)
    np.testing.assert_array_equal(out[0], flipped[0])
    np.testing.assert_array_equal(out[2], flipped[2])
    # Support output does not depend on the query rows; each query row is mapped alone.
    half = roleinfo.role_preprocess_support_fit(roles, xs, xq[:7])
    np.testing.assert_allclose(half[0], out[0], atol=1e-6)
    np.testing.assert_allclose(half[1], out[2][:7], atol=1e-6)
    # The default fit is stage 7's support + query fit, which does depend on the queries.
    default = roleinfo.preprocess_episode(roles, arrays)
    sq = roleinfo.role_preprocess(roles, xs, xq)
    np.testing.assert_array_equal(default[0], sq[0])
    if any(r["role"] == "measure" for r in roles):
        assert not np.allclose(default[0], out[0])
    with pytest.raises(ValueError):
        roleinfo.preprocess_episode(roles, arrays, fit="query")


def _write_results(
    out: Path, drop: tuple | None = None, nan: tuple | None = None, rescore_shift: float = 0.0
) -> list[str]:
    """Synthetic complete stage 4, 7 and 9 results; returns the report path arguments."""
    dirs = {name: out / name for name in ("04", "07", "09")}
    for d in dirs.values():
        d.mkdir()
    settings = evaluation._settings_json(evaluation.Settings())
    (dirs["04"] / evaluation.CONFIG_JSON).write_text(
        json.dumps({"settings": settings, "scenes": [list(s) for s in SCENES]})
    )
    (dirs["07"] / roleinfo.CONFIG_JSON).write_text(json.dumps(roleinfo._config_core()))
    (dirs["09"] / role_eval.CONFIG_JSON).write_text(json.dumps(role_eval._config_core()))
    rng = np.random.default_rng(0)
    need = sorted(role_eval.required_rows())
    sample = set(SCENES[: role_eval.RESCORE_SCENES])
    files = {
        "04": dirs["04"] / evaluation.SCORES_CSV,
        "07": dirs["07"] / roleinfo.SCORES_CSV,
        "09": dirs["09"] / role_eval.SCORES_CSV,
        "re": dirs["09"] / role_eval.RESCORE_CSV,
    }
    handles = {k: p.open("w", newline="") for k, p in files.items()}
    writers = {k: csv.writer(h) for k, h in handles.items()}
    for w in writers.values():
        w.writerow(evaluation.SCORE_FIELDS)
    for f, s in SCENES:
        for n in SIZES:
            refs = {arm: rng.uniform(0.2, 0.7) for arm in role_eval.REFERENCE_ARMS}
            for arm in evaluation.ARMS:
                writers["04"].writerow([f, s, n, arm, f"{refs[arm]:.6f}"])
            for arm, v in ((roleinfo.NATIVE, refs[evaluation.NATIVE]), (roleinfo.ROLES, refs[role_eval.PREPROC])):
                writers["07"].writerow([f, s, n, arm, f"{v:.6f}"])
            if (f, s) in sample:
                for arm, v in refs.items():
                    writers["re"].writerow([f, s, n, arm, f"{v + rescore_shift:.6f}"])
        for n, label in need:
            key = (f, s, n, label)
            if key != drop:
                writers["09"].writerow([f, s, n, label, "nan" if key == nan else f"{rng.uniform(0.2, 0.7):.6f}"])
    for h in handles.values():
        h.close()
    return ["--results", str(dirs["09"])], ["--primary", str(dirs["04"]), "--roleinfo", str(dirs["07"])]


def _run(name: str, paths) -> int:
    results, refs = paths
    return report.main([name, *results, *(refs if name == "role-curve" else [])])


def test_reports_pass_on_complete_results(tmp_path: Path, capsys) -> None:
    paths = _write_results(tmp_path)
    for name in ("role-primary", "role-curve", "role-reliability"):
        assert _run(name, paths) == 0
    out = capsys.readouterr().out
    assert "D-08 verdict: " in out
    assert all(f"\n{n:>5} " in out for n in SIZES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop": (4, 2_000_000, 1024, role_eval.CONTROL)},
        {"drop": (5, 2_000_001, 16, role_eval.corrupted_label("wrong"))},
        {"nan": (4, 2_000_002, 8, role_eval.STORY)},
    ],
)
@pytest.mark.parametrize("name", ["role-primary", "role-curve", "role-reliability"])
def test_reports_fail_on_missing_or_non_finite_rows(tmp_path: Path, kwargs, name) -> None:
    assert _run(name, _write_results(tmp_path, **kwargs)) == 1


def test_curve_fails_when_rescored_references_differ(tmp_path: Path) -> None:
    assert _run("role-curve", _write_results(tmp_path, rescore_shift=1e-5)) == 1


def _write_10k(out: Path, drop: tuple | None = None, nan: tuple | None = None) -> list[str]:
    """Synthetic stage 13 two-arm results next to complete stage 9 results."""
    stage9, _ = _write_results(out)
    adapters, results = out / "13-train", out / "13-eval"
    adapters.mkdir()
    results.mkdir()
    selected = {"story": {"step": 9500}, "control": {"step": 6500}}
    (adapters / "summary.json").write_text(json.dumps({"selected": selected}))
    (results / role_eval.CONFIG_JSON).write_text(json.dumps(role_eval._config_core(adapters, full=False)))
    rng = np.random.default_rng(1)
    with (results / role_eval.SCORES_CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(evaluation.SCORE_FIELDS)
        for f, s in SCENES:
            for n, label in sorted(role_eval.required_rows(full=False)):
                if (f, s, n, label) != drop:
                    v = "nan" if (f, s, n, label) == nan else f"{rng.uniform(0.2, 0.7):.6f}"
                    w.writerow([f, s, n, label, v])
    return ["role10k-primary", "--results", str(results), "--adapters", str(adapters), "--stage9", stage9[1]]


def test_two_arm_run_needs_no_corruptions_or_preproc() -> None:
    rows = role_eval.required_rows(full=False)
    assert {label for _, label in rows} == {role_eval.STORY, role_eval.CONTROL}
    assert {n for n, _ in rows} == set(SIZES)


def test_role10k_primary_passes_on_complete_results(tmp_path: Path, capsys) -> None:
    assert report.main(_write_10k(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "D-08 verdict: " in out and "story 10k - story 4k" in out
    assert all(f"\n{n:>5} " in out for n in SIZES)


@pytest.mark.parametrize(
    "kwargs", [{"drop": (4, 2_000_000, 32, role_eval.CONTROL)}, {"nan": (5, 2_000_001, 8, role_eval.STORY)}]
)
def test_role10k_primary_fails_on_missing_or_non_finite_rows(tmp_path: Path, kwargs) -> None:
    assert report.main(_write_10k(tmp_path, **kwargs)) == 1
