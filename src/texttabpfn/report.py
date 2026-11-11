"""Print result tables from committed results (decision D-10).

``python -m texttabpfn.report <name>`` reads small CSV/JSON files under
``results/`` and never recomputes an experiment. It exits non-zero when the
table is incomplete. ``stories`` is the exception: it is a check on the text
builders, computed from the scene generator and the MiniLM tokenizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from texttabpfn import evaluation, reliability, roleinfo, training
from texttabpfn.capacity import ARMS, CONFIG_JSON, RESULTS_DIR, SUPPORT_SIZES, Settings, arm_csv
from texttabpfn.scenes import HELDOUT_FAMILIES, eval_scene_specs

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
# Per-row loss columns: the three table columns first, then the validation losses.
_LOGLOSS_FIELDS = ("native_logloss", "optimized_logloss", "gain", "val_native", "val_best")
# Stage 2 story check: minimum sample size and the per-scene RNG stream for
# the corrupted texts (scene RNG streams 0-2 are the generator's, 10-15 stage 1's).
MIN_STORY_SCENES = 1000
_STORY_CHECK_STREAM = 20


class IncompleteResults(Exception):
    """A required (arm, n) row or scene is missing, mismatched or non-finite."""


def scene_bootstrap_ci(
    values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """95% percentile CI of the mean over scenes, resampling scenes."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def capacity_table(results_dir: Path, arm: str) -> list[dict[str, float]]:
    config_path = results_dir / CONFIG_JSON
    csv_path = arm_csv(results_dir, arm)
    if not config_path.exists() or not csv_path.exists():
        raise IncompleteResults(f"missing {config_path} or {csv_path}")
    config = json.loads(config_path.read_text())
    expected = config["settings"]["num_scenes"]
    if expected != Settings().num_scenes:
        raise IncompleteResults(
            f"results cover {expected} scenes; the stage requires {Settings().num_scenes}"
        )
    scenes = [tuple(spec) for spec in config["scenes"]]
    gains: dict[int, dict[int, tuple[float, float, float]]] = {n: {} for n in SUPPORT_SIZES}
    with csv_path.open() as fh:
        for line, row in enumerate(csv.DictReader(fh), start=2):
            if row["arm"] != arm or int(row["n"]) not in gains:
                continue
            scene = int(row["scene"])
            spec = (int(row["family"]), int(row["seed"]))
            if scene >= len(scenes) or spec != scenes[scene]:
                expected_spec = scenes[scene] if scene < len(scenes) else None
                raise IncompleteResults(
                    f"{csv_path}:{line}: scene {scene} is (family, seed) {spec}; "
                    f"config.json has {expected_spec}"
                )
            values = tuple(float(row[k]) for k in _LOGLOSS_FIELDS)
            if not all(math.isfinite(v) for v in values):
                raise IncompleteResults(
                    f"{csv_path}:{line}: non-finite value in "
                    f"{dict(zip(_LOGLOSS_FIELDS, values, strict=True))}"
                )
            gains[int(row["n"])][scene] = values[:3]
    table = []
    for n in SUPPORT_SIZES:
        missing = sorted(set(range(expected)) - set(gains[n]))
        if missing:
            raise IncompleteResults(
                f"arm {arm}, n={n}: {len(missing)} of {expected} scenes missing "
                f"(first: {missing[:5]})"
            )
        vals = np.array([gains[n][i] for i in range(expected)])
        lo, hi = scene_bootstrap_ci(vals[:, 2])
        table.append({
            "n": n,
            "scenes": expected,
            "native": float(vals[:, 0].mean()),
            "optimized": float(vals[:, 1].mean()),
            "gain": float(vals[:, 2].mean()),
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return table


def print_capacity(args: argparse.Namespace) -> None:
    table = capacity_table(args.results, args.arm)
    print(
        f"Stage 1 capacity, arm={args.arm}: gain = native - optimized query log loss "
        f"(mean over scenes; 95% scene-bootstrap CI, {BOOTSTRAP_RESAMPLES} resamples)"
    )
    print(f"{'n':>4} {'scenes':>6} {'native':>8} {'optimized':>9} {'gain':>8}  95% CI")
    for r in table:
        print(
            f"{r['n']:>4} {r['scenes']:>6} {r['native']:>8.4f} {r['optimized']:>9.4f} "
            f"{r['gain']:>8.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        )


def print_stories(args: argparse.Namespace) -> int:
    """Max MiniLM token length of schema+story (and corrupted) texts.

    Exits non-zero unless the fixed sample has at least ``MIN_STORY_SCENES``
    scenes covering every family and every text fits ``MAX_TOKENS``.
    """
    from texttabpfn.encoder import MAX_TOKENS, token_lengths
    from texttabpfn.scenes import FAMILIES, Scene
    from texttabpfn.stories import VARIANTS, scene_text, story_sample_specs

    specs = story_sample_specs()
    corrupt = [v for v in VARIANTS if v != "story"]
    cells: dict[tuple[str, int], dict[str, list[int]]] = {}
    for split, family, seed in specs:
        scene = Scene(family, seed)
        rng = np.random.default_rng([family, seed, _STORY_CHECK_STREAM])
        texts = [scene_text(scene, v, rng) for v in VARIANTS]
        lengths = token_lengths(texts)
        cell = cells.setdefault((split, family), {"p": [], "story": [], "corrupt": []})
        cell["p"].append(scene.num_features)
        cell["story"].append(lengths[0])
        cell["corrupt"].append(max(lengths[1:]))

    print(
        f"Stage 2 stories: MiniLM tokens incl. [CLS]/[SEP] (limit {MAX_TOKENS}); "
        f"story = schema + story, corrupted = max over {', '.join(corrupt)}"
    )
    print(f"{'split':>5} {'family':>6} {'scenes':>6} {'max p':>5} {'max story':>9} {'max corrupted':>13}")
    for (split, family), c in cells.items():
        print(
            f"{split:>5} {family:>6} {len(c['p']):>6} {max(c['p']):>5} "
            f"{max(c['story']):>9} {max(c['corrupt']):>13}"
        )
    max_story = max(max(c["story"]) for c in cells.values())
    max_all = max(max_story, *(max(c["corrupt"]) for c in cells.values()))
    families = {family for _, family in cells}
    print(f"scenes: {len(specs)}  max schema+story tokens: {max_story}  max any text: {max_all}")
    if len(specs) < MIN_STORY_SCENES or families != set(FAMILIES):
        print(f"incomplete sample: need >= {MIN_STORY_SCENES} scenes from every family", file=sys.stderr)
        return 1
    if max_all > MAX_TOKENS:
        print(f"FAIL: a text exceeds {MAX_TOKENS} tokens", file=sys.stderr)
        return 1
    return 0


def _training_config(results_dir: Path) -> tuple[dict, training.Settings]:
    config_path = results_dir / training.CONFIG_JSON
    if not config_path.exists():
        raise IncompleteResults(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    return config, training.settings_from_json(config["settings"])


def print_training(args: argparse.Namespace) -> int:
    """Episodes each arm completed; non-zero unless both arms finished the same full budget."""
    if args.validation:
        return print_training_validation(args)
    config, settings = _training_config(args.results)
    summary_path = args.results / training.SUMMARY_JSON
    if not summary_path.exists():
        raise IncompleteResults(f"missing {summary_path}: training has not finished")
    summary = json.loads(summary_path.read_text())
    missing = [arm for arm in training.ARMS if arm not in summary["episodes"]]
    if missing:
        raise IncompleteResults(f"arm(s) {missing} missing from {summary_path}")
    # The last progress row per arm must agree with the summary counts.
    last: dict[str, tuple[int, int]] = {}
    with (args.results / training.PROGRESS_CSV).open() as fh:
        for row in csv.DictReader(fh):
            last[row["arm"]] = (int(row["step"]), int(row["episodes"]))
    print(
        f"Stage 3 training: {settings.steps} steps x {settings.episodes_per_step} episodes, "
        f"Adam lr {settings.lr:g} (warm-up {settings.warmup_steps}, cosine to "
        f"{settings.final_lr_ratio:g}x), train wall {summary['train_wall_s'] / 3600:.2f} h"
    )
    print(f"{'arm':>6} {'episodes':>9} {'last logged step':>16} {'selected step':>13} {'dev val loss':>12}")
    for arm in training.ARMS:
        sel = summary["selected"][arm]
        step, _ = last.get(arm, (-1, -1))
        print(
            f"{arm:>6} {summary['episodes'][arm]:>9} {step:>16} {sel['step']:>13} "
            f"{sel['val_logloss_mean']:>12.4f}"
        )
    counts = {summary["episodes"][arm] for arm in training.ARMS}
    if len(counts) != 1:
        print(f"FAIL: episode counts differ: {summary['episodes']}", file=sys.stderr)
        return 1
    if counts != {settings.episodes}:
        print(f"FAIL: arms completed {counts.pop()} episodes; budget is {settings.episodes}", file=sys.stderr)
        return 1
    for arm in training.ARMS:
        if last.get(arm) != (settings.steps, settings.episodes):
            print(f"FAIL: progress.csv for {arm} ends at {last.get(arm)}", file=sys.stderr)
            return 1
    return 0


def print_training_validation(args: argparse.Namespace) -> int:
    """Dev validation log loss per arm at every checkpoint; non-zero if a row is missing."""
    _, settings = _training_config(args.results)
    val_path = args.results / training.VALIDATION_CSV
    if not val_path.exists():
        raise IncompleteResults(f"missing {val_path}")
    val = training.read_validation(val_path.parent)
    sizes = settings.val_sizes
    steps = training.checkpoint_steps(settings)
    required = [(0, training.NATIVE, n) for n in sizes]
    required += [(s, arm, n) for s in steps for arm in training.ARMS for n in sizes]
    missing = [key for key in required if key not in val or not math.isfinite(val[key])]
    if missing:
        raise IncompleteResults(
            f"{len(missing)} of {len(required)} checkpoint rows missing (first: {missing[:3]})"
        )
    selected = training.select_checkpoints(settings, args.results)
    summary_path = args.results / training.SUMMARY_JSON
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text())["selected"]
        for arm, (step, _) in selected.items():
            if recorded[arm]["step"] != step:
                print(f"FAIL: summary.json selects step {recorded[arm]['step']} for {arm}", file=sys.stderr)
                return 1
    print(
        f"Stage 3 dev validation log loss ({settings.val_scenes} dev scenes, families 0-3, "
        f"{settings.val_support_seeds} support seeds, {settings.val_query_rows} query rows); "
        "* = selected checkpoint (lowest mean)"
    )
    cols = [f"n={n}" for n in sizes] + ["mean"]
    head = "  ".join(f"{arm}:" + " ".join(f"{c:>7}" for c in cols) for arm in training.ARMS)
    print(f"{'step':>6}  {head}")

    def cells(step: int, arm: str) -> str:
        v = [val[(step, arm, n)] for n in sizes]
        return " ".join(f"{x:>7.4f}" for x in [*v, float(np.mean(v))])

    native = [val[(0, training.NATIVE, n)] for n in sizes]
    native_cells = " ".join(f"{x:>7.4f}" for x in [*native, float(np.mean(native))])
    print(f"{'native':>6}  " + "  ".join(f"{' ' * len(arm)} {native_cells}" for arm in training.ARMS))
    for step in steps:
        parts = []
        for arm in training.ARMS:
            mark = "*" if selected[arm][0] == step else " "
            parts.append(f"{' ' * (len(arm) - 1)}{mark} {cells(step, arm)}")
        print(f"{step:>6}  " + "  ".join(parts))
    for arm, (step, v) in selected.items():
        print(f"selected {arm}: step {step}, mean dev log loss {v:.4f}")
    return 0


# --- Stage 4: primary evaluation ------------------------------------------------

PRIMARY_SIZES = (8, 16, 32)
# (label, minuend arm, subtrahend arm) for the paired per-scene differences.
_PAIRS = (
    ("story-schema", "story", "schema"),
    ("story-native", "story", evaluation.NATIVE),
    ("schema-native", "schema", evaluation.NATIVE),
)


def d08_verdict(cis: dict[int, tuple[float, float]]) -> str:
    """D-08 verdict from the story-minus-schema CIs at n = 8, 16, 32."""
    cis = {n: cis[n] for n in PRIMARY_SIZES}
    if any(lo > 0 for lo, _ in cis.values()):
        return "harmful"
    if sum(hi < 0 for _, hi in cis.values()) >= 2:
        return "supported"
    return "not supported"


def _evaluation_scores(
    results_dir: Path, sizes: tuple[int, ...]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int, int, str], float]]:
    """Scenes and scores; raises unless every (scene, n, arm) at ``sizes`` is finite."""
    config_path = results_dir / evaluation.CONFIG_JSON
    if not config_path.exists():
        raise IncompleteResults(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    settings = evaluation.settings_from_json(config["settings"])
    if settings != evaluation.Settings():
        raise IncompleteResults(f"settings {settings} differ from the D-08 design {evaluation.Settings()}")
    scenes = [tuple(s) for s in config["scenes"]]
    if scenes != eval_scene_specs(settings.num_scenes):
        raise IncompleteResults(f"{config_path} lists other scenes than eval_scene_specs")
    if not (results_dir / evaluation.SCORES_CSV).exists():
        raise IncompleteResults(f"missing {results_dir / evaluation.SCORES_CSV}")
    scores = evaluation.read_scores(results_dir)
    missing = [
        (f, s, n, arm)
        for f, s in scenes
        for n in sizes
        for arm in evaluation.ARMS
        if not math.isfinite(scores.get((f, s, n, arm), math.nan))
    ]
    if missing:
        required = len(scenes) * len(sizes) * len(evaluation.ARMS)
        raise IncompleteResults(
            f"{len(missing)} of {required} (scene, n, arm) rows missing or non-finite (first: {missing[:3]})"
        )
    return scenes, scores


def _per_scene(scores: dict, scenes: list[tuple[int, int]], n: int, arm: str) -> np.ndarray:
    return np.array([scores[(f, s, n, arm)] for f, s in scenes])


def _diff_cell(values: np.ndarray, digits: int = 4) -> str:
    lo, hi = scene_bootstrap_ci(values)
    return f"{values.mean():+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def print_primary(args: argparse.Namespace) -> int:
    """Story minus schema-only query log loss at n = 8, 16, 32 and the D-08 verdict."""
    scenes, scores = _evaluation_scores(args.results, PRIMARY_SIZES)
    print(
        f"Stage 4 primary (D-08): {len(scenes)} held-out scenes (families 4-5), "
        f"{evaluation.Settings().support_seeds} nested support seeds, "
        f"{evaluation.Settings().query_rows} query rows; paired per-scene difference "
        f"story - schema-only query log loss, 95% scene-bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples)"
    )
    print(f"{'n':>5} {'native':>8} {'schema':>8} {'story':>8}  {'story - schema [95% CI]':>28}")
    cis = {}
    for n in PRIMARY_SIZES:
        means = [_per_scene(scores, scenes, n, arm).mean() for arm in evaluation.ARMS]
        diff = _per_scene(scores, scenes, n, "story") - _per_scene(scores, scenes, n, "schema")
        cis[n] = scene_bootstrap_ci(diff)
        print(f"{n:>5} " + " ".join(f"{m:>8.4f}" for m in means) + f"  {_diff_cell(diff):>28}")
    print(f"D-08 verdict: {d08_verdict(cis)}")
    print("\nSecondary: story - schema-only by held-out family [95% CI]")
    families = {fam: [sc for sc in scenes if sc[0] == fam] for fam in HELDOUT_FAMILIES}
    print(f"{'n':>5} " + " ".join(f"{f'family {fam} ({len(sub)} scenes)':>28}" for fam, sub in families.items()))
    for n in PRIMARY_SIZES:
        cells = []
        for sub in families.values():
            diff = _per_scene(scores, sub, n, "story") - _per_scene(scores, sub, n, "schema")
            cells.append(f"{_diff_cell(diff):>28}")
        print(f"{n:>5} " + " ".join(cells))
    return 0


def print_curve(args: argparse.Namespace) -> int:
    """Paired differences between the three arms at every support size."""
    sizes = evaluation.Settings().sizes
    scenes, scores = _evaluation_scores(args.results, sizes)
    print(
        f"Stage 4 curve: {len(scenes)} held-out scenes; mean query log loss per arm and paired "
        f"per-scene differences, 95% scene-bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples)"
    )
    head = " ".join(f"{label + ' [95% CI]':>28}" for label, _, _ in _PAIRS)
    print(f"{'n':>5} {'native':>8} {'schema':>8} {'story':>8}  {head}")
    for n in sizes:
        means = [_per_scene(scores, scenes, n, arm).mean() for arm in evaluation.ARMS]
        cells = [
            f"{_diff_cell(_per_scene(scores, scenes, n, a) - _per_scene(scores, scenes, n, b)):>28}"
            for _, a, b in _PAIRS
        ]
        print(f"{n:>5} " + " ".join(f"{m:>8.4f}" for m in means) + "  " + " ".join(cells))
    return 0


# --- Stage 5: context reliability ----------------------------------------------

# Largest |story rescored in stage 5 - story arm of stage 4| allowed on identical rows.
_STORY_RESCORE_TOL = 1e-4


def _reliability_scores(
    results_dir: Path,
) -> tuple[dict, list[tuple[int, int]], dict[tuple[int, int, int, str], float]]:
    """Config, scenes and scores; raises unless every (scene, n, text) is finite."""
    config_path = results_dir / reliability.CONFIG_JSON
    if not config_path.exists():
        raise IncompleteResults(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    saved = config["settings"]
    settings = reliability.Settings(num_scenes=saved["num_scenes"], sizes=tuple(saved["sizes"]))
    if settings != reliability.Settings():
        raise IncompleteResults(f"settings {settings} differ from the stage 5 design {reliability.Settings()}")
    if evaluation.settings_from_json(config["row_settings"]) != evaluation.Settings():
        raise IncompleteResults(f"{config_path} rows were not built with the D-08 settings")
    if tuple(config["texts"]) != reliability.TEXTS:
        raise IncompleteResults(f"{config_path} texts {config['texts']} differ from {reliability.TEXTS}")
    scenes = [tuple(s) for s in config["scenes"]]
    if scenes != eval_scene_specs(settings.num_scenes):
        raise IncompleteResults(f"{config_path} lists other scenes than eval_scene_specs")
    if not (results_dir / reliability.SCORES_CSV).exists():
        raise IncompleteResults(f"missing {results_dir / reliability.SCORES_CSV}")
    scores = reliability.read_scores(results_dir)
    missing = [
        (f, s, n, text)
        for f, s in scenes
        for n in settings.sizes
        for text in reliability.TEXTS
        if not math.isfinite(scores.get((f, s, n, text), math.nan))
    ]
    if missing:
        required = len(scenes) * len(settings.sizes) * len(reliability.TEXTS)
        raise IncompleteResults(
            f"{len(missing)} of {required} (scene, n, text) rows missing or non-finite (first: {missing[:3]})"
        )
    return config, scenes, scores


def print_reliability(args: argparse.Namespace) -> int:
    """Corrupted minus correct-story log loss of the story adapter at n = 8, 16, 32."""
    config, scenes, scores = _reliability_scores(args.results)
    primary_scenes, primary = _evaluation_scores(args.primary, reliability.SIZES)
    if primary_scenes != scenes:
        raise IncompleteResults(f"{args.primary} scores other scenes than {args.results}")
    correct = reliability.CORRECT
    mix = config["training_story_mix_per_20"]
    share = {k: v / sum(mix.values()) for k, v in mix.items()}
    print(
        f"Stage 5 reliability (D-09, H4): story adapter (stage 3 step {config['checkpoint']['step']}), "
        f"{len(scenes)} held-out scenes (families 4-5), stage 4 rows ({evaluation.Settings().support_seeds} "
        f"nested support seeds, {evaluation.Settings().query_rows} query rows); paired per-scene change "
        f"corrupted - correct story query log loss, 95% scene-bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples)"
    )
    print(
        f"Training exposure (D-09): {share.get('omitted', 0):.0%} of story-arm episodes omitted the story "
        f"and {share.get('one_wrong_fact', 0):.0%} had one wrong fact; irrelevant stories and fully "
        "permuted roles were never trained on."
    )
    head = " ".join(f"{c + ' - story [95% CI]':>32}" for c in reliability.CORRUPTIONS)
    print(f"{'n':>5} {'story':>8}  {head}")
    for n in reliability.SIZES:
        base = _per_scene(scores, scenes, n, correct)
        cells = [
            f"{_diff_cell(_per_scene(scores, scenes, n, c) - base, digits=5):>32}"
            for c in reliability.CORRUPTIONS
        ]
        print(f"{n:>5} {base.mean():>8.4f}  " + " ".join(cells))

    print("\nSecondary (context, not a check): stage 4 arms on the same rows minus the correct story [95% CI]")
    print(f"{'n':>5} {'schema-only arm - story':>32} {'native - story':>32}")
    drift = 0.0
    for n in reliability.SIZES:
        base = _per_scene(scores, scenes, n, correct)
        drift = max(drift, float(np.abs(base - _per_scene(primary, scenes, n, "story")).max()))
        cells = [
            f"{_diff_cell(_per_scene(primary, scenes, n, arm) - base, digits=5):>32}"
            for arm in ("schema", evaluation.NATIVE)
        ]
        print(f"{n:>5} " + " ".join(cells))
    print(f"max |story rescored here - stage 4 story arm| over {len(scenes) * len(reliability.SIZES)} cells: {drift:.2e}")
    if drift > _STORY_RESCORE_TOL:
        print(f"FAIL: rows differ from stage 4 (tolerance {_STORY_RESCORE_TOL:g})", file=sys.stderr)
        return 1
    return 0


# --- Stage 7: value of story roles to native TabPFN ----------------------------

# Largest |native rescored in stage 7 - native arm of stage 4| allowed on identical rows.
_NATIVE_RESCORE_TOL = 1e-6


def _roleinfo_scores(
    results_dir: Path,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int, int, str], float]]:
    """Scenes and scores; raises unless the design matches and every (scene, n, arm) is finite."""
    config_path = results_dir / roleinfo.CONFIG_JSON
    if not config_path.exists():
        raise IncompleteResults(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    if evaluation.settings_from_json(config["row_settings"]) != evaluation.Settings():
        raise IncompleteResults(f"{config_path} rows were not built with the D-08 settings")
    if tuple(config["arms"]) != roleinfo.ARMS or config["method"] != roleinfo.METHOD:
        raise IncompleteResults(f"{config_path} holds other arms or another preprocessing method")
    scenes = [tuple(s) for s in config["scenes"]]
    if scenes != eval_scene_specs(evaluation.Settings().num_scenes):
        raise IncompleteResults(f"{config_path} lists other scenes than eval_scene_specs")
    if not (results_dir / roleinfo.SCORES_CSV).exists():
        raise IncompleteResults(f"missing {results_dir / roleinfo.SCORES_CSV}")
    scores = roleinfo.read_scores(results_dir)
    sizes = evaluation.Settings().sizes
    missing = [
        (f, s, n, arm)
        for f, s in scenes
        for n in sizes
        for arm in roleinfo.ARMS
        if not math.isfinite(scores.get((f, s, n, arm), math.nan))
    ]
    if missing:
        required = len(scenes) * len(sizes) * len(roleinfo.ARMS)
        raise IncompleteResults(
            f"{len(missing)} of {required} (scene, n, arm) rows missing or non-finite (first: {missing[:3]})"
        )
    return scenes, scores


def print_roleinfo(args: argparse.Namespace) -> int:
    """Story-role-preprocessed minus raw-row native TabPFN log loss at every n (D-12, D-13)."""
    scenes, scores = _roleinfo_scores(args.results)
    sizes = evaluation.Settings().sizes
    primary_scenes, primary = _evaluation_scores(args.primary, sizes)
    if primary_scenes != scenes:
        raise IncompleteResults(f"{args.primary} scores other scenes than {args.results}")
    native, roles = roleinfo.NATIVE, roleinfo.ROLES

    def diff(sub: list[tuple[int, int]], n: int) -> np.ndarray:
        return _per_scene(scores, sub, n, roles) - _per_scene(scores, sub, n, native)

    print(
        f"Stage 7 role information (D-12, D-13): native TabPFN-2.5 on raw rows vs rows preprocessed "
        f"with the story's role facts; {len(scenes)} held-out scenes (families 4-5), stage 4 rows "
        f"({evaluation.Settings().support_seeds} nested support seeds, {evaluation.Settings().query_rows} "
        f"query rows); paired per-scene difference preprocessed - native query log loss, "
        f"95% scene-bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples)"
    )
    print(f"Preprocessing: {roleinfo.METHOD}")
    head = f"{'n':>5} {'native':>8} {'preproc':>8}  {'preproc - native [95% CI]':>28}"
    print(head)
    cis = {}
    for n in PRIMARY_SIZES:
        d = diff(scenes, n)
        cis[n] = scene_bootstrap_ci(d)
        means = [_per_scene(scores, scenes, n, arm).mean() for arm in (native, roles)]
        print(f"{n:>5} " + " ".join(f"{m:>8.4f}" for m in means) + f"  {_diff_cell(d):>28}")
    zero_in_all = all(lo <= 0 <= hi for lo, hi in cis.values())
    print(f"D-13: 95% CI includes 0 at all of n = 8, 16, 32: {'yes' if zero_in_all else 'no'}")

    print("\nSecondary: every support size")
    print(head)
    for n in sizes:
        means = [_per_scene(scores, scenes, n, arm).mean() for arm in (native, roles)]
        print(f"{n:>5} " + " ".join(f"{m:>8.4f}" for m in means) + f"  {_diff_cell(diff(scenes, n)):>28}")

    print("\nSecondary: preprocessed - native by held-out family [95% CI]")
    families = {fam: [sc for sc in scenes if sc[0] == fam] for fam in HELDOUT_FAMILIES}
    print(f"{'n':>5} " + " ".join(f"{f'family {fam} ({len(sub)} scenes)':>28}" for fam, sub in families.items()))
    for n in sizes:
        print(f"{n:>5} " + " ".join(f"{_diff_cell(diff(sub, n)):>28}" for sub in families.values()))

    drift = max(
        float(np.abs(_per_scene(scores, scenes, n, native) - _per_scene(primary, scenes, n, evaluation.NATIVE)).max())
        for n in sizes
    )
    print(f"\nmax |native rescored here - stage 4 native arm| over {len(scenes) * len(sizes)} cells: {drift:.2e}")
    if drift > _NATIVE_RESCORE_TOL:
        print(f"FAIL: rows differ from stage 4 (tolerance {_NATIVE_RESCORE_TOL:g})", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m texttabpfn.report")
    sub = parser.add_subparsers(dest="name", required=True)
    cap = sub.add_parser("capacity", help="stage 1: thinking-row channel capacity")
    cap.add_argument("--arm", choices=ARMS, default="shared")
    cap.add_argument("--results", type=Path, default=RESULTS_DIR)
    cap.set_defaults(func=print_capacity)
    sto = sub.add_parser("stories", help="stage 2: schema+story token lengths")
    sto.set_defaults(func=print_stories)
    tra = sub.add_parser("training", help="stage 3: episodes per arm and dev validation")
    tra.add_argument("--validation", action="store_true", help="dev log loss per checkpoint")
    tra.add_argument("--results", type=Path, default=training.RESULTS_DIR)
    tra.set_defaults(func=print_training)
    pri = sub.add_parser("primary", help="stage 4: story - schema-only at n = 8, 16, 32, D-08 verdict")
    pri.add_argument("--results", type=Path, default=evaluation.RESULTS_DIR)
    pri.set_defaults(func=print_primary)
    cur = sub.add_parser("curve", help="stage 4: paired arm differences at every n")
    cur.add_argument("--results", type=Path, default=evaluation.RESULTS_DIR)
    cur.set_defaults(func=print_curve)
    rel = sub.add_parser("reliability", help="stage 5: corrupted - correct story at n = 8, 16, 32")
    rel.add_argument("--results", type=Path, default=reliability.RESULTS_DIR)
    rel.add_argument("--primary", type=Path, default=evaluation.RESULTS_DIR)
    rel.set_defaults(func=print_reliability)
    rol = sub.add_parser("roleinfo", help="stage 7: role-preprocessed - native TabPFN at every n")
    rol.add_argument("--results", type=Path, default=roleinfo.RESULTS_DIR)
    rol.add_argument("--primary", type=Path, default=evaluation.RESULTS_DIR)
    rol.set_defaults(func=print_roleinfo)
    args = parser.parse_args(argv)
    try:
        status = args.func(args)
    except IncompleteResults as exc:
        print(f"incomplete results: {exc}", file=sys.stderr)
        return 1
    return status or 0


if __name__ == "__main__":
    sys.exit(main())
