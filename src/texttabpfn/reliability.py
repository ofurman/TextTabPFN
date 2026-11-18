"""Stage 5: context reliability of the story adapter (D-09, H4).

Design, fixed before any result was seen:

- Rows: the stage 4 rows exactly. ``evaluation.scene_episodes`` is called with
  the D-08 settings (n = 8 ... 1024, so the support pool is the stage 4 pool)
  and only n = 8, 16, 32 are scored: same 200 held-out scenes, 5 nested
  support seeds and 512 query rows.
- Model: the story adapter only, with the weights selected in stage 3.
- Texts (``stories.scene_text``): the correct story and three corruptions,
  ``omitted`` (schema only), ``irrelevant`` (another same-family scene's story)
  and ``wrong`` (feature roles permuted). Each corruption draws from its own
  per-scene RNG stream ``(family, seed, 60, variant index)``, so the texts are
  reproducible and do not depend on each other. The correct story is rescored
  here on the same rows (a consistency check against stage 4's story arm).
- Metric: per (scene, n, text), the mean query log loss over the support seeds.

Run: ``uv run python -m texttabpfn.reliability`` (resumable: a restart keeps
every scene with all its rows and recomputes the rest). Writes ``config.json``,
``scores.csv`` and, when done, ``summary.json`` to ``results/05-reliability/``.
Report: ``python -m texttabpfn.report reliability``.
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

from texttabpfn import evaluation, training
from texttabpfn.adapter import episode_logits
from texttabpfn.encoder import TextEncoder
from texttabpfn.model import default_device, load_tabpfn
from texttabpfn.scenes import Scene, eval_scene_specs
from texttabpfn.stories import scene_text

RESULTS_DIR = Path("results/05-reliability")
CONFIG_JSON = "config.json"
SCORES_CSV = evaluation.SCORES_CSV
SUMMARY_JSON = "summary.json"
SCORE_FIELDS = ("family", "seed", "n", "text", "logloss")

ARM = "story"
SIZES = (8, 16, 32)
CORRECT = "story"
CORRUPTIONS = ("omitted", "irrelevant", "wrong")
TEXTS = (CORRECT, *CORRUPTIONS)

# Per-scene RNG stream of the corrupted texts; see evaluation.py for streams in use.
_CORRUPTION_STREAM = 60


@dataclasses.dataclass(frozen=True)
class Settings:
    num_scenes: int = 200
    sizes: tuple[int, ...] = SIZES


def corruption_rng(family: int, seed: int, variant: str) -> np.random.Generator:
    return np.random.default_rng([family, seed, _CORRUPTION_STREAM, TEXTS.index(variant)])


def scene_texts(family: int, seed: int) -> dict[str, str]:
    """Correct and corrupted encoder inputs of one scene, a pure function of the scene."""
    scene = Scene(family, seed)
    return {v: scene_text(scene, v, corruption_rng(family, seed, v)) for v in TEXTS}


def _settings_json(settings: Settings) -> dict:
    return {"num_scenes": settings.num_scenes, "sizes": list(settings.sizes)}


def _complete_scenes(settings: Settings, out_dir: Path) -> set[tuple[int, int]]:
    need = {(n, text) for n in settings.sizes for text in TEXTS}
    return evaluation.keep_complete_scenes(out_dir / SCORES_CSV, SCORE_FIELDS, need)


def read_scores(out_dir: Path) -> dict[tuple[int, int, int, str], float]:
    """``(family, seed, n, text) -> logloss`` from ``scores.csv``."""
    return evaluation.read_scores(out_dir, label="text")


def run(settings: Settings, out_dir: Path, chunk: int) -> None:
    if not set(settings.sizes) <= set(evaluation.Settings().sizes):
        raise SystemExit(f"sizes {settings.sizes} are not stage 4 support sizes")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    device = default_device()
    rows_settings = evaluation.Settings()
    stage3 = json.loads((training.RESULTS_DIR / training.SUMMARY_JSON).read_text())["selected"][ARM]
    train_config = json.loads((training.RESULTS_DIR / training.CONFIG_JSON).read_text())
    specs = eval_scene_specs(settings.num_scenes)
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["settings"] != _settings_json(settings):
            raise SystemExit(f"{config_path} holds different settings; refusing to resume")
        if saved["checkpoint"] != stage3:
            raise SystemExit(f"{config_path} was run with another stage 3 checkpoint; refusing to resume")
    else:
        config = {
            "settings": _settings_json(settings),
            "row_settings": dataclasses.asdict(rows_settings) | {"sizes": list(rows_settings.sizes)},
            "arm": ARM,
            "checkpoint": stage3,
            "texts": list(TEXTS),
            "corruption_stream": _CORRUPTION_STREAM,
            "training_story_mix_per_20": train_config["story_mix_per_20"],
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
        adapter = training.load_adapter(model, ARM)
        t_start = time.time()
        for i, (family, seed) in enumerate(todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(rows_settings, family, seed)
            texts = scene_texts(family, seed)
            if texts[CORRECT] != se.texts[ARM]:
                raise SystemExit(f"scene {(family, seed)}: correct story differs from stage 4's")
            changed = [v for v in CORRUPTIONS if texts[v] == texts[CORRECT]]
            if changed:
                raise SystemExit(f"scene {(family, seed)}: corruption(s) {changed} equal the story")
            embeddings = {v: encoder.encode_one(t) for v, t in texts.items()}
            rows = []
            for n in settings.sizes:
                episode = tuple(torch.as_tensor(a, device=device) for a in se.arrays[n])
                for v in TEXTS:
                    loss = evaluation.query_logloss(
                        lambda ep, e=embeddings[v]: episode_logits(model, adapter, e, ep), episode, chunk
                    )
                    rows.append([family, seed, n, v, f"{loss:.6f}"])
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
        (f, s, n, v)
        for f, s in specs
        for n in settings.sizes
        for v in TEXTS
        if not math.isfinite(scores.get((f, s, n, v), math.nan))
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
    parser.add_argument("--num-scenes", type=int, default=Settings().num_scenes)
    args = parser.parse_args(argv)
    run(Settings(num_scenes=args.num_scenes), args.out, args.chunk)


if __name__ == "__main__":
    main()
