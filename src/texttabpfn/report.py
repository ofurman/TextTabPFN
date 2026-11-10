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

from texttabpfn.capacity import ARMS, CONFIG_JSON, RESULTS_DIR, SUPPORT_SIZES, Settings, arm_csv

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m texttabpfn.report")
    sub = parser.add_subparsers(dest="name", required=True)
    cap = sub.add_parser("capacity", help="stage 1: thinking-row channel capacity")
    cap.add_argument("--arm", choices=ARMS, default="shared")
    cap.add_argument("--results", type=Path, default=RESULTS_DIR)
    cap.set_defaults(func=print_capacity)
    sto = sub.add_parser("stories", help="stage 2: schema+story token lengths")
    sto.set_defaults(func=print_stories)
    args = parser.parse_args(argv)
    try:
        status = args.func(args)
    except IncompleteResults as exc:
        print(f"incomplete results: {exc}", file=sys.stderr)
        return 1
    return status or 0


if __name__ == "__main__":
    sys.exit(main())
