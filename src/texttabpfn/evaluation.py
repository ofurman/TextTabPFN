"""Stage 4: primary evaluation over support sizes on held-out scene families (D-08).

Design, fixed before any result was seen:

- Scenes: ``eval_scene_specs(200)``, families 4 and 5 alternating (100 each),
  seeds from ``EVAL_SEED_RANGE``.
- Per scene: 512 fixed query rows and a separate support pool, both drawn from
  the scene (so query and support rows are disjoint draws). Each of 5 support
  seeds is a nested stratified sequence over n = 8 ... 1024 (both classes in
  every support, class counts from the scene's base rate).
- Arms: native raw TabPFN-2.5 (W = 0), the schema-only adapter reading the
  schema (``omitted``) and the story adapter reading schema + story, both with
  the weights selected in stage 3 by dev validation loss. All arms score the
  same support and query rows.
- Metric: per (scene, n, arm), the mean query log loss over the support seeds.
  Every seed scores the same 512 query rows, so this equals the mean over all
  seed x query pairs.

Run: ``uv run python -m texttabpfn.evaluation`` (resumable: a restart keeps
every scene with all its rows and recomputes the rest). Writes ``config.json``,
``scores.csv`` and, when done, ``summary.json`` to ``results/04-primary/``.
Report: ``python -m texttabpfn.report primary`` and ``... curve``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from texttabpfn import training
from texttabpfn.adapter import episode_logits
from texttabpfn.capacity import nested_supports, stratified_counts
from texttabpfn.encoder import TextEncoder
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.scenes import Scene, eval_scene_specs
from texttabpfn.stories import scene_text

RESULTS_DIR = Path("results/04-primary")
CONFIG_JSON = "config.json"
SCORES_CSV = "scores.csv"
SUMMARY_JSON = "summary.json"
SCORE_FIELDS = ("family", "seed", "n", "arm", "logloss")

NATIVE = training.NATIVE
ARMS = (NATIVE, "schema", "story")
# The text each adapter arm reads (its training condition; D-08).
ARM_VARIANT = {"schema": "omitted", "story": "story"}

# Per-scene RNG streams, seeded by (family, seed, stream[, k]). Generator streams
# are 0-2, stage 1's 10-15, the stage 2 story check's 20, stage 3's 30 and 40-42.
_QUERY_STREAM, _POOL_STREAM, _SUPPORT_SEED_STREAM = 50, 51, 52


@dataclasses.dataclass(frozen=True)
class Settings:
    num_scenes: int = 200
    sizes: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
    support_seeds: int = 5
    query_rows: int = 512
    pool_rows: int = 4096


@dataclasses.dataclass
class SceneEpisodes:
    family: int
    seed: int
    texts: dict[str, str]  # adapter arm -> encoder input
    arrays: dict[int, tuple[np.ndarray, ...]]  # n -> xs (n, K, p), ys (n, K), xq (M, K, p), yq (M, K)


def scene_episodes(settings: Settings, family: int, seed: int) -> SceneEpisodes:
    """Fixed query rows and nested stratified supports of one scene, a pure function
    of ``(settings, family, seed)``. Support seed ``k`` has its own RNG stream, so
    its supports do not depend on how many seeds are drawn."""
    scene = Scene(family, seed)
    q_x, q_y = scene.sample(settings.query_rows, np.random.default_rng([family, seed, _QUERY_STREAM]))
    pool_rng = np.random.default_rng([family, seed, _POOL_STREAM])
    p_x, p_y = scene.sample(settings.pool_rows, pool_rng)
    neg, pos = stratified_counts(max(settings.sizes), scene.base_rate)
    while (p_y == 0).sum() < neg or (p_y == 1).sum() < pos:
        more_x, more_y = scene.sample(settings.pool_rows, pool_rng)
        p_x, p_y = np.concatenate([p_x, more_x]), np.concatenate([p_y, more_y])
    supports = [
        nested_supports(
            p_y,
            settings.sizes,
            scene.base_rate,
            np.random.default_rng([family, seed, _SUPPORT_SEED_STREAM, k]),
        )
        for k in range(settings.support_seeds)
    ]
    k = settings.support_seeds
    arrays = {}
    for n in settings.sizes:
        # Row order within a support: permuted with the seed's own stream.
        order = [
            np.random.default_rng([family, seed, _SUPPORT_SEED_STREAM, j, n]).permutation(n)
            for j in range(k)
        ]
        idx = [sp[n][o] for sp, o in zip(supports, order, strict=True)]
        arrays[n] = (
            np.stack([p_x[i] for i in idx], axis=1),
            np.stack([p_y[i] for i in idx], axis=1),
            np.repeat(q_x[:, None], k, axis=1),
            np.repeat(q_y[:, None], k, axis=1),
        )
    texts = {arm: scene_text(scene, variant) for arm, variant in ARM_VARIANT.items()}
    return SceneEpisodes(family, seed, texts, arrays)


def _settings_json(settings: Settings) -> dict:
    d = dataclasses.asdict(settings)
    d["sizes"] = list(settings.sizes)
    return d


def settings_from_json(d: dict) -> Settings:
    return Settings(**{**d, "sizes": tuple(d["sizes"])})


def read_scores(out_dir: Path) -> dict[tuple[int, int, int, str], float]:
    """``(family, seed, n, arm) -> logloss`` from ``scores.csv``."""
    with (out_dir / SCORES_CSV).open() as fh:
        return {
            (int(r["family"]), int(r["seed"]), int(r["n"]), r["arm"]): float(r["logloss"])
            for r in csv.DictReader(fh)
        }


def _complete_scenes(settings: Settings, out_dir: Path) -> set[tuple[int, int]]:
    """Keep the rows of scenes with every (n, arm) row; drop the rest from the CSV."""
    path = out_dir / SCORES_CSV
    if not path.exists():
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(SCORE_FIELDS)
        return set()
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    per_scene: dict[tuple[int, int], set[tuple[int, str]]] = {}
    for r in rows:
        per_scene.setdefault((int(r["family"]), int(r["seed"])), set()).add((int(r["n"]), r["arm"]))
    need = {(n, arm) for n in settings.sizes for arm in ARMS}
    done = {scene for scene, keys in per_scene.items() if keys == need}
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(r for r in rows if (int(r["family"]), int(r["seed"])) in done)
    return done


@torch.no_grad()
def _logloss(logits_fn, episode: tuple[torch.Tensor, ...], chunk: int) -> float:
    """Mean query cross-entropy over all support seeds, ``chunk`` seeds per forward."""
    xs, ys, xq, yq = episode
    total, count = 0.0, 0
    for b in range(0, xs.shape[1], chunk):
        part = (xs[:, b : b + chunk], ys[:, b : b + chunk], xq[:, b : b + chunk], yq[:, b : b + chunk])
        logits = logits_fn(part)
        total += float(F.cross_entropy(logits.reshape(-1, 2), part[3].reshape(-1), reduction="sum"))
        count += part[3].numel()
    return total / count


def run(settings: Settings, out_dir: Path, chunk: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    device = default_device()
    stage3 = json.loads((training.RESULTS_DIR / training.SUMMARY_JSON).read_text())["selected"]
    specs = eval_scene_specs(settings.num_scenes)
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["settings"] != _settings_json(settings):
            raise SystemExit(f"{config_path} holds different settings; refusing to resume")
        if saved["checkpoints"] != stage3:
            raise SystemExit(f"{config_path} was run with other stage 3 checkpoints; refusing to resume")
    else:
        config = {
            "settings": _settings_json(settings),
            "arms": list(ARMS),
            "arm_variant": ARM_VARIANT,
            "checkpoints": stage3,
            "scenes": [list(s) for s in specs],
            "device": str(device),
            "torch": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    done = _complete_scenes(settings, out_dir)
    todo = [s for s in specs if s not in done]
    print(f"{len(done)} of {len(specs)} scenes already scored; {len(todo)} to go", flush=True)
    if todo:
        model = load_tabpfn(device)
        encoder = TextEncoder("cpu")
        adapters = {arm: training.load_adapter(model, arm) for arm in ARM_VARIANT}

        def logits_fn(arm: str, embedding: torch.Tensor | None):
            if arm == NATIVE:
                return lambda ep: binary_logits(model, *ep[:3])
            return lambda ep: episode_logits(model, adapters[arm], embedding, ep)

        t_start = time.time()
        for i, (family, seed) in enumerate(todo, 1):
            t0 = time.time()
            se = scene_episodes(settings, family, seed)
            embeddings = {arm: encoder.encode_one(text) for arm, text in se.texts.items()}
            rows = []
            for n in settings.sizes:
                episode = tuple(torch.as_tensor(a, device=device) for a in se.arrays[n])
                for arm in ARMS:
                    loss = _logloss(logits_fn(arm, embeddings.get(arm)), episode, chunk)
                    rows.append([family, seed, n, arm, f"{loss:.6f}"])
            # One append per scene: a scene's rows land together or not at all.
            with (out_dir / SCORES_CSV).open("a", newline="") as fh:
                csv.writer(fh).writerows(rows)
            elapsed = time.time() - t_start
            print(
                f"scene {len(done) + i}/{len(specs)} (family {family}, seed {seed}) "
                f"{time.time() - t0:.1f}s, run {elapsed / 60:.1f} min, "
                f"eta {elapsed / i * (len(todo) - i) / 60:.0f} min",
                flush=True,
            )

    scores = read_scores(out_dir)
    missing = [
        (f, s, n, a)
        for f, s in specs
        for n in settings.sizes
        for a in ARMS
        if not math.isfinite(scores.get((f, s, n, a), math.nan))
    ]
    if missing:
        raise SystemExit(f"{len(missing)} score rows missing or non-finite (first: {missing[:3]})")
    summary = {
        "scenes": len(specs),
        "rows": len(scores),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {summary}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chunk", type=int, default=5, help="support seeds per forward pass")
    defaults = Settings()
    for field in dataclasses.fields(Settings):
        default = getattr(defaults, field.name)
        flag = f"--{field.name.replace('_', '-')}"
        if isinstance(default, tuple):
            parser.add_argument(flag, type=int, nargs="+", default=list(default))
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = parser.parse_args(argv)
    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)}
    values["sizes"] = tuple(values["sizes"])
    run(Settings(**values), args.out, args.chunk)


if __name__ == "__main__":
    main()
