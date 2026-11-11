"""Stage 9: role-token adapters on held-out families (D-08, D-09, D-11, D-12).

Design, fixed before any result was seen:

- Rows: the stage 4 rows exactly. ``evaluation.scene_episodes`` is called with
  the D-08 settings over ``eval_scene_specs(200)``: every n = 8 ... 1024, 5
  nested support seeds, 512 query rows.
- Arms scored here, all on identical rows:
  - ``role_story``: the stage 8 story role-token adapter reading the correct
    per-feature role sentences;
  - ``role_control``: the stage 8 matched control reading the neutral sentence
    in every slot;
  - ``preproc_support``: story-guided preprocessing (D-12) with each
    measurement-group collapse fitted on the support rows only, scored by
    native TabPFN-2.5. It separates the role facts from the unlabeled query
    information that stage 7's support + query fit also uses.
  Both adapters use the checkpoints selected in stage 8 by dev validation loss.
- Corruptions (D-09) of the ``role_story`` arm at n = 8, 16, 32, each from its
  own per-scene RNG stream ``(family, seed, 70, corruption index)``:
  - ``omitted``: every slot neutral;
  - ``irrelevant``: the role sentences of a same-family donor scene
    (``stories.donor_scene``), truncated to this scene's p, or padded with the
    neutral sentence when the donor has fewer features (iteration 1's
    irrelevant story likewise stated nothing about features the donor lacks).
    A donor whose sentences equal the correct ones is redrawn from the stream;
  - ``wrong``: roles permuted among this scene's features (``stories.wrong_roles``).
  A corruption equal to the correct sentences is refused (the run stops).
- References are read, not rescored: native and the iteration-1 story and
  schema adapters from ``results/04-primary``, story-guided preprocessing
  (support + query fit) from ``results/07-roleinfo``. The first
  ``RESCORE_SCENES`` scenes are re-scored for all four at every n into
  ``rescore.csv``; the report fails unless they match to 1e-6.
- ``roleinfo.preprocess_episode`` gets the whole 512-row query batch before any
  chunking (``query_logloss`` chunks support seeds, not query rows).
- Metric: per (scene, n, arm), the mean query log loss over the support seeds.

Run: ``uv run python -m texttabpfn.role_eval`` (resumable: a restart keeps
every scene with all its rows and recomputes the rest). Writes ``config.json``,
``rescore.csv``, ``scores.csv`` and, when done, ``summary.json`` to
``results/09-role-eval/``. Reports: ``python -m texttabpfn.report role-primary``,
``... role-curve`` and ``... role-reliability``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from texttabpfn import evaluation, role_training, roleinfo, training
from texttabpfn.adapter import episode_logits
from texttabpfn.encoder import TextEncoder
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.role_tokens import NEUTRAL_SENTENCE, role_sentences, slot_tokens, variant_sentences
from texttabpfn.scenes import Scene, eval_scene_specs
from texttabpfn.stories import donor_scene, scene_roles

RESULTS_DIR = Path("results/09-role-eval")
CONFIG_JSON = "config.json"
SCORES_CSV = evaluation.SCORES_CSV
RESCORE_CSV = "rescore.csv"
SUMMARY_JSON = "summary.json"
SCORE_FIELDS = evaluation.SCORE_FIELDS

NATIVE = evaluation.NATIVE
STORY, CONTROL, PREPROC_SUPPORT = "role_story", "role_control", "preproc_support"
ARMS = (STORY, CONTROL, PREPROC_SUPPORT)
# The stage 8 adapter behind each role-token arm.
ADAPTER = {STORY: "story", CONTROL: "control"}
CORRUPTIONS = ("omitted", "irrelevant", "wrong")
RELIABILITY_SIZES = (8, 16, 32)

# Reference arms re-scored on a sample: stage 4's arms and stage 7's preprocessing.
PREPROC = "preproc"
REFERENCE_ARMS = (*evaluation.ARMS, PREPROC)
RESCORE_SCENES = 8

# Per-scene RNG stream of the corrupted roles; see evaluation.py for streams in use
# (stage 5's corrupted texts use 60).
_CORRUPTION_STREAM = 70
_MAX_DONOR_DRAWS = 100


def corrupted_label(corruption: str) -> str:
    return f"{STORY}_{corruption}"


def corruption_rng(family: int, seed: int, corruption: str) -> np.random.Generator:
    return np.random.default_rng([family, seed, _CORRUPTION_STREAM, CORRUPTIONS.index(corruption)])


def _irrelevant_sentences(scene: Scene, correct: list[str], rng: np.random.Generator) -> list[str]:
    p = scene.num_features
    for _ in range(_MAX_DONOR_DRAWS):
        donor = role_sentences(scene_roles(donor_scene(scene, rng)))[:p]
        sentences = donor + [NEUTRAL_SENTENCE] * (p - len(donor))
        if sentences != correct:
            return sentences
    raise RuntimeError(f"scene {(scene.family, scene.seed)}: no donor with different roles")


def scene_sentences(family: int, seed: int) -> dict[str, list[str]]:
    """Per-feature sentences for every label scored with slot tokens, a pure
    function of the scene: the two arms and the role-story corruptions."""
    scene = Scene(family, seed)
    correct = variant_sentences(scene, "story")
    out = {STORY: correct, CONTROL: variant_sentences(scene, "omitted")}
    for c in CORRUPTIONS:
        rng = corruption_rng(family, seed, c)
        if c == "irrelevant":
            sentences = _irrelevant_sentences(scene, correct, rng)
        else:
            sentences = variant_sentences(scene, c, rng)
        if sentences == correct:
            raise ValueError(f"scene {(family, seed)}: corruption {c!r} equals the correct roles")
        out[corrupted_label(c)] = sentences
    return out


def required_rows() -> set[tuple[int, str]]:
    """Every ``(n, label)`` row one scene needs in ``scores.csv``."""
    need = {(n, arm) for n in evaluation.Settings().sizes for arm in ARMS}
    return need | {(n, corrupted_label(c)) for n in RELIABILITY_SIZES for c in CORRUPTIONS}


def _config_core() -> dict:
    rows = evaluation.Settings()
    stage8 = json.loads((role_training.RESULTS_DIR / training.SUMMARY_JSON).read_text())["selected"]
    return {
        "row_settings": evaluation._settings_json(rows),
        "arms": list(ARMS),
        "adapters": {arm: {"stage8_arm": a, "step": stage8[a]["step"]} for arm, a in ADAPTER.items()},
        "corruptions": [corrupted_label(c) for c in CORRUPTIONS],
        "reliability_sizes": list(RELIABILITY_SIZES),
        "corruption_stream": _CORRUPTION_STREAM,
        "preproc_support_method": roleinfo.METHOD.replace("support + query", "support"),
        "reference_arms": list(REFERENCE_ARMS),
        "rescore_scenes": RESCORE_SCENES,
        "scenes": [list(s) for s in eval_scene_specs(rows.num_scenes)],
    }


def read_scores(out_dir: Path, name: str = SCORES_CSV) -> dict[tuple[int, int, int, str], float]:
    """``(family, seed, n, arm) -> logloss`` from ``scores.csv`` or ``rescore.csv``."""
    with (out_dir / name).open() as fh:
        return {
            (int(r["family"]), int(r["seed"]), int(r["n"]), r["arm"]): float(r["logloss"])
            for r in csv.DictReader(fh)
        }


def _append(path: Path, rows: list[list]) -> None:
    # One append per scene: a scene's rows land together or not at all.
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _progress(t_start: float, i: int, done: int, total: int, todo: int, label: str, t0: float) -> None:
    elapsed = time.time() - t_start
    print(
        f"{label} {done + i}/{total} {time.time() - t0:.1f}s, run {elapsed / 60:.1f} min, "
        f"eta {elapsed / i * (todo - i) / 60:.0f} min",
        flush=True,
    )


def run(out_dir: Path, chunk: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    device = default_device()
    rows_settings = evaluation.Settings()
    core = _config_core()
    specs = eval_scene_specs(rows_settings.num_scenes)
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

    sentences = {spec: scene_sentences(*spec) for spec in specs}  # refuses equal corruptions up front
    sample = specs[:RESCORE_SCENES]
    need_ref = {(n, arm) for n in rows_settings.sizes for arm in REFERENCE_ARMS}
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
        encoder = TextEncoder("cpu")

        def native_logits(ep):
            return binary_logits(model, *ep[:3])

    if ref_todo:
        it1 = {arm: training.load_adapter(model, arm) for arm in evaluation.ARM_VARIANT}
        t_start = time.time()
        for i, (family, seed) in enumerate(ref_todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(rows_settings, family, seed)
            roles = scene_roles(Scene(family, seed))
            embeddings = {arm: encoder.encode_one(text) for arm, text in se.texts.items()}
            rows = []
            for n in rows_settings.sizes:
                inputs = {arm: se.arrays[n] for arm in evaluation.ARMS}
                inputs[PREPROC] = roleinfo.preprocess_episode(roles, se.arrays[n])
                for arm in REFERENCE_ARMS:
                    episode = tuple(torch.as_tensor(a, device=device) for a in inputs[arm])
                    if arm in it1:
                        fn = lambda ep, a=arm: episode_logits(model, it1[a], embeddings[a], ep)  # noqa: E731
                    else:
                        fn = native_logits
                    loss = evaluation.query_logloss(fn, episode, chunk)
                    rows.append([family, seed, n, arm, f"{loss:.6f}"])
            _append(out_dir / RESCORE_CSV, rows)
            _progress(t_start, i, len(ref_done), len(sample), len(ref_todo), "rescore scene", t0)
        del it1

    if todo:
        adapters = {arm: role_training.load_adapter(model, a) for arm, a in ADAPTER.items()}
        t_start = time.time()
        for i, (family, seed) in enumerate(todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(rows_settings, family, seed)
            roles = scene_roles(Scene(family, seed))
            tokens = {label: slot_tokens(encoder, s) for label, s in sentences[(family, seed)].items()}
            rows = []
            for n in rows_settings.sizes:
                episode = tuple(torch.as_tensor(a, device=device) for a in se.arrays[n])
                labels = [(STORY, STORY), (CONTROL, CONTROL)]
                if n in RELIABILITY_SIZES:
                    labels += [(corrupted_label(c), STORY) for c in CORRUPTIONS]
                for label, arm in labels:
                    loss = evaluation.query_logloss(
                        lambda ep, a=adapters[arm], t=tokens[label]: episode_logits(model, a, t, ep),
                        episode,
                        chunk,
                    )
                    rows.append([family, seed, n, label, f"{loss:.6f}"])
                # The whole query batch enters the preprocessing before any chunking.
                pre = roleinfo.preprocess_episode(roles, se.arrays[n], fit=roleinfo.FIT_SUPPORT)
                pre_episode = tuple(torch.as_tensor(a, device=device) for a in pre)
                loss = evaluation.query_logloss(native_logits, pre_episode, chunk)
                rows.append([family, seed, n, PREPROC_SUPPORT, f"{loss:.6f}"])
            _append(out_dir / SCORES_CSV, rows)
            _progress(t_start, i, len(done), len(specs), len(todo), f"scene ({family}, {seed})", t0)

    scores = read_scores(out_dir)
    missing = [
        (f, s, n, label)
        for f, s in specs
        for n, label in sorted(required_rows())
        if not math.isfinite(scores.get((f, s, n, label), math.nan))
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
    parser.add_argument("--chunk", type=int, default=5, help="support seeds per forward pass")
    args = parser.parse_args(argv)
    run(args.out, args.chunk)


if __name__ == "__main__":
    main()
