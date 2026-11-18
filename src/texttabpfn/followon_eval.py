"""Stage 11: the B-09 bge follow-on arms on held-out families (D-08, D-14, D-16).

Scores stage 10's selected bge story and control adapters on exactly stage 4's rows
(``evaluation.scene_episodes(evaluation.Settings(), f, s)`` over ``eval_scene_specs(200)``,
every n). Each arm reads its training-condition text: story reads the schema+story text,
control the schema-only (``omitted``) text, as ``evaluation.ARM_VARIANT``.

Before the adapter scores, native TabPFN is re-scored on the first ``RESCORE_SCENES``
scenes into ``rescore.csv`` and must match ``results/04-primary`` to ``RESCORE_TOL``.
The run appends one scene at a time, so a restart keeps every complete scene.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from texttabpfn import evaluation, followon
from texttabpfn.adapter import episode_logits
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.role_eval import _append, _progress, read_scores
from texttabpfn.scenes import eval_scene_specs

RESULTS_DIR = Path("results/11-followon-eval")
CONFIG_JSON = evaluation.CONFIG_JSON
SCORES_CSV = evaluation.SCORES_CSV
RESCORE_CSV = "rescore.csv"
SUMMARY_JSON = evaluation.SUMMARY_JSON
SCORE_FIELDS = evaluation.SCORE_FIELDS

STORY, CONTROL = "bge_story", "bge_control"
ARMS = (STORY, CONTROL)
# The stage 10 adapter behind each arm, and the stage 4 text variant it reads.
ADAPTER = {STORY: "story", CONTROL: "control"}
TEXT = {STORY: "story", CONTROL: "schema"}  # keys of evaluation.SceneEpisodes.texts
RESCORE_SCENES = 8
RESCORE_TOL = 1e-6


def required_rows() -> set[tuple[int, str]]:
    return {(n, arm) for n in evaluation.Settings().sizes for arm in ARMS}


def _config_core(adapters_dir: Path) -> dict:
    rows = evaluation.Settings()
    stage10 = json.loads((adapters_dir / followon.SUMMARY_JSON).read_text())["selected"]
    return {
        "row_settings": evaluation._settings_json(rows),
        "arms": list(ARMS),
        "text_variant": {arm: evaluation.ARM_VARIANT[TEXT[arm]] for arm in ARMS},
        "encoder": followon.ENCODER_NAME,
        "adapters_dir": str(adapters_dir),
        "adapters": {arm: {"stage10_arm": a, "step": stage10[a]["step"]} for arm, a in ADAPTER.items()},
        "rescore_scenes": RESCORE_SCENES,
        "scenes": [list(s) for s in eval_scene_specs(rows.num_scenes)],
    }


def rescore_mismatches(out_dir: Path, primary_dir: Path = evaluation.RESULTS_DIR) -> list[tuple]:
    """Re-scored native rows differing from stage 4 by more than ``RESCORE_TOL``."""
    primary = evaluation.read_scores(primary_dir)
    return [
        (key, v, primary.get(key))
        for key, v in read_scores(out_dir, RESCORE_CSV).items()
        if not abs(v - primary.get(key, math.nan)) <= RESCORE_TOL
    ]


def run(out_dir: Path, chunk: int, adapters_dir: Path = followon.RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    device = default_device()
    settings = evaluation.Settings()
    core = _config_core(adapters_dir)
    specs = eval_scene_specs(settings.num_scenes)
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if {k: saved.get(k) for k in core} != core:
            raise SystemExit(f"{config_path} holds another design; refusing to resume")
    else:
        config = core | {
            "device": str(device),
            "torch": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    sample = specs[:RESCORE_SCENES]
    need_ref = {(n, evaluation.NATIVE) for n in settings.sizes}
    ref_done = evaluation.keep_complete_scenes(out_dir / RESCORE_CSV, SCORE_FIELDS, need_ref)
    ref_todo = [s for s in sample if s not in ref_done]
    done = evaluation.keep_complete_scenes(out_dir / SCORES_CSV, SCORE_FIELDS, required_rows())
    todo = [s for s in specs if s not in done]
    print(
        f"rescore sample: {len(ref_done)} of {len(sample)} done; scores: {len(done)} of "
        f"{len(specs)} scenes done, {len(todo)} to go",
        flush=True,
    )
    if ref_todo or todo:
        model = load_tabpfn(device)

    if ref_todo:
        t_start = time.time()
        for i, (family, seed) in enumerate(ref_todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(settings, family, seed)
            rows = []
            for n in settings.sizes:
                episode = tuple(torch.as_tensor(a, device=device) for a in se.arrays[n])
                loss = evaluation.query_logloss(lambda ep: binary_logits(model, *ep[:3]), episode, chunk)
                rows.append([family, seed, n, evaluation.NATIVE, f"{loss:.6f}"])
            _append(out_dir / RESCORE_CSV, rows)
            _progress(t_start, i, len(ref_done), len(sample), len(ref_todo), "rescore scene", t0)
    bad = rescore_mismatches(out_dir)
    if bad:
        raise SystemExit(f"{len(bad)} re-scored native rows differ from stage 4 (first: {bad[:3]})")
    print(f"rescore sample matches stage 4 within {RESCORE_TOL}", flush=True)

    if todo:
        encoder = followon.BgeEncoder("cpu")
        adapters = {arm: followon.load_adapter(model, a, adapters_dir).to(device) for arm, a in ADAPTER.items()}
        t_start = time.time()
        for i, (family, seed) in enumerate(todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(settings, family, seed)
            embeddings = {arm: encoder.encode_one(se.texts[TEXT[arm]]) for arm in ARMS}
            rows = []
            for n in settings.sizes:
                episode = tuple(torch.as_tensor(a, device=device) for a in se.arrays[n])
                for arm in ARMS:
                    loss = evaluation.query_logloss(
                        lambda ep, a=adapters[arm], e=embeddings[arm]: episode_logits(model, a, e, ep),
                        episode,
                        chunk,
                    )
                    rows.append([family, seed, n, arm, f"{loss:.6f}"])
            _append(out_dir / SCORES_CSV, rows)
            _progress(t_start, i, len(done), len(specs), len(todo), f"scene ({family}, {seed})", t0)

    scores = read_scores(out_dir)
    missing = [
        (f, s, n, arm)
        for f, s in specs
        for n, arm in sorted(required_rows())
        if not math.isfinite(scores.get((f, s, n, arm), math.nan))
    ]
    if missing:
        raise SystemExit(f"{len(missing)} score rows missing or non-finite (first: {missing[:3]})")
    summary = {
        "scenes": len(specs),
        "rows": len(scores),
        "rescore_rows": len(read_scores(out_dir, RESCORE_CSV)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {summary}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--adapters", type=Path, default=followon.RESULTS_DIR)
    parser.add_argument("--chunk", type=int, default=5, help="support seeds per forward")
    args = parser.parse_args(argv)
    run(args.out, args.chunk, args.adapters)


if __name__ == "__main__":
    main()
