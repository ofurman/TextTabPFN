"""Stage 3: episodic training of the story arm and the schema-only arm.

Both arms are a ``TextConditionedThinkingRows`` adapter (W zero-initialised)
around the frozen TabPFN-2.5 backbone and the frozen MiniLM encoder. They are
trained in lockstep in one process: every episode is generated once and fed to
both arms, so the arms see the same scenes, support sizes, rows, order and seeds,
and share the optimizer settings, learning-rate schedule, steps and batch size.

- Episode ``i``: a fresh scene from training families 0-3 with a seed from
  ``TRAIN_SEED_RANGE``; support size n log-uniform in 8 ... 1024, stratified so
  both classes appear; 512 query rows drawn separately (disjoint from the
  support); loss = mean query cross-entropy through the raw TabPFN-2.5 path.
- Story arm text: schema + story, except exactly 15% of episodes schema only
  (``omitted``) and 10% with one wrong fact (D-09), assigned per block of 20.
- Schema-only arm text: the schema (``omitted`` variant).
- Validation: fixed dev scenes (``DEV_SEED_RANGE``, families 0-3) with fixed
  nested stratified supports at ``val_sizes`` and 512 fixed query rows; each arm
  reads its training-condition text (story arm: correct story). The native
  model on the same rows is recorded once as a reference.

The final checkpoint per arm is the one with the lowest dev validation loss
(mean over ``val_sizes``); step 0 (W = 0, the native model) is a candidate.

Run: ``uv run python -m texttabpfn.training`` (resumable from the last
checkpoint). Writes ``config.json``, ``progress.csv``, ``validation.csv`` and,
when done, ``summary.json`` to ``results/03-training/``; adapter weights go to
``results/03-training/weights/`` (``*.pt``, gitignored, never distributed).
Report: ``python -m texttabpfn.report training [--validation]``.
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

from texttabpfn.adapter import TextConditionedThinkingRows, episode_logits
from texttabpfn.capacity import nested_supports, stratified_counts, stratified_index
from texttabpfn.encoder import TextEncoder
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.scenes import TRAIN_FAMILIES, TRAIN_SEED_RANGE, Scene, dev_scene_specs
from texttabpfn.stories import scene_text

RESULTS_DIR = Path("results/03-training")
CONFIG_JSON = "config.json"
PROGRESS_CSV = "progress.csv"
VALIDATION_CSV = "validation.csv"
SUMMARY_JSON = "summary.json"
WEIGHTS_DIR = "weights"
STATE_FILE = "state.pt"

ARMS = ("story", "schema")
NATIVE = "native"
# The text each arm reads for a validation scene (its training condition).
VALIDATION_VARIANT = {"story": "story", "schema": "omitted"}
PROGRESS_FIELDS = ("step", "arm", "episodes", "lr", "train_logloss", "elapsed_s")
VALIDATION_FIELDS = ("step", "arm", "n", "scenes", "logloss")

# RNG stream ids: episodes are seeded by (_EPISODE_STREAM, settings.seed, index);
# dev scenes by (family, seed, stream). Generator streams are 0-2, stage 1's 10-15,
# the stage 2 story check's 20.
_EPISODE_STREAM = 30
_VAL_QUERY_STREAM, _VAL_SUPPORT_STREAM, _VAL_SEED_STREAM = 40, 41, 42
# Story-arm text mix (D-09), assigned exactly per block of 20 episodes.
_MIX_BLOCK = 20
_MIX = (("omitted", 3), ("one_wrong_fact", 2), ("story", 15))


@dataclasses.dataclass(frozen=True)
class Settings:
    steps: int = 4000
    episodes_per_step: int = 8
    lr: float = 1e-3
    warmup_steps: int = 100
    final_lr_ratio: float = 0.1
    n_min: int = 8
    n_max: int = 1024
    query_rows: int = 512
    seed: int = 0
    log_every: int = 25
    val_every: int = 250
    val_scenes: int = 64
    val_sizes: tuple[int, ...] = (8, 32, 128)
    val_support_seeds: int = 3
    val_support_rows: int = 256
    val_query_rows: int = 512

    @property
    def episodes(self) -> int:
        return self.steps * self.episodes_per_step


def checkpoint_steps(settings: Settings) -> list[int]:
    """Steps with a validation row and saved weights (step 0 = W = 0)."""
    steps = list(range(0, settings.steps + 1, settings.val_every))
    if steps[-1] != settings.steps:
        steps.append(settings.steps)
    return steps


def learning_rate(settings: Settings, step: int) -> float:
    """Linear warm-up, then cosine decay to ``final_lr_ratio * lr`` (step >= 1)."""
    if step <= settings.warmup_steps:
        return settings.lr * step / settings.warmup_steps
    t = (step - settings.warmup_steps) / max(1, settings.steps - settings.warmup_steps)
    ratio = settings.final_lr_ratio + (1 - settings.final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return settings.lr * ratio


# --- Episodes -------------------------------------------------------------------


@dataclasses.dataclass
class Episode:
    index: int
    family: int
    seed: int
    n: int
    variant: str  # story-arm text variant
    texts: dict[str, str]  # arm -> encoder input
    arrays: tuple[np.ndarray, ...]  # xs (n, 1, p), ys (n, 1), xq (M, 1, p), yq (M, 1)


def story_variant(settings: Settings, index: int) -> str:
    """Story-arm variant of episode ``index``: an exact 15/10/75 mix per block of 20."""
    block = np.repeat([v for v, _ in _MIX], [k for _, k in _MIX])
    rng = np.random.default_rng([_EPISODE_STREAM, settings.seed, index // _MIX_BLOCK, 1])
    return str(block[rng.permutation(_MIX_BLOCK)[index % _MIX_BLOCK]])


def make_episode(settings: Settings, index: int) -> Episode:
    """Episode ``index``: a pure function of ``(settings.seed, index)``, shared by both arms."""
    rng = np.random.default_rng([_EPISODE_STREAM, settings.seed, index])
    family = int(rng.choice(TRAIN_FAMILIES))
    seed = int(rng.integers(*TRAIN_SEED_RANGE))
    scene = Scene(family, seed)
    log_n = rng.uniform(math.log(settings.n_min), math.log(settings.n_max))
    n = int(np.clip(round(math.exp(log_n)), settings.n_min, settings.n_max))
    neg, pos = stratified_counts(n, scene.base_rate)
    pool_x, pool_y = scene.sample(4 * n + 64, rng)
    while (pool_y == 0).sum() < neg or (pool_y == 1).sum() < pos:
        more_x, more_y = scene.sample(4 * n + 64, rng)
        pool_x, pool_y = np.concatenate([pool_x, more_x]), np.concatenate([pool_y, more_y])
    idx = stratified_index(pool_y, n, scene.base_rate, rng)
    q_x, q_y = scene.sample(settings.query_rows, rng)
    variant = story_variant(settings, index)
    texts = {
        "story": scene_text(scene, variant, rng),
        "schema": scene_text(scene, "omitted"),
    }
    arrays = (pool_x[idx][:, None], pool_y[idx][:, None], q_x[:, None], q_y[:, None])
    return Episode(index, family, seed, n, variant, texts, arrays)


# --- Validation -----------------------------------------------------------------


class DevSet:
    """Fixed dev scenes: per scene and n, a batch over support seeds sharing 512 query rows."""

    def __init__(self, settings: Settings, encoder: TextEncoder, device: torch.device) -> None:
        self.specs = dev_scene_specs(settings.val_scenes)
        self.sizes = settings.val_sizes
        self.episodes: list[dict[int, tuple[torch.Tensor, ...]]] = []
        self.embeddings: dict[str, list[torch.Tensor]] = {arm: [] for arm in ARMS}
        k = settings.val_support_seeds
        for family, seed in self.specs:
            scene = Scene(family, seed)
            q_x, q_y = scene.sample(
                settings.val_query_rows, np.random.default_rng([family, seed, _VAL_QUERY_STREAM])
            )
            s_x, s_y = scene.sample(
                settings.val_support_rows, np.random.default_rng([family, seed, _VAL_SUPPORT_STREAM])
            )
            seed_rng = np.random.default_rng([family, seed, _VAL_SEED_STREAM])
            supports = [nested_supports(s_y, self.sizes, scene.base_rate, seed_rng) for _ in range(k)]
            per_n = {}
            for n in self.sizes:
                arrays = (
                    np.stack([s_x[sp[n]] for sp in supports], axis=1),
                    np.stack([s_y[sp[n]] for sp in supports], axis=1),
                    np.repeat(q_x[:, None], k, axis=1),
                    np.repeat(q_y[:, None], k, axis=1),
                )
                per_n[n] = tuple(torch.as_tensor(a, device=device) for a in arrays)
            self.episodes.append(per_n)
            for arm in ARMS:
                text = scene_text(scene, VALIDATION_VARIANT[arm])
                self.embeddings[arm].append(encoder.encode_one(text))

    @torch.no_grad()
    def logloss(self, model, adapter: TextConditionedThinkingRows, arm: str) -> dict[int, float]:
        """Mean over dev scenes of the per-scene query log loss (averaged over support seeds)."""
        per_n: dict[int, list[float]] = {n: [] for n in self.sizes}
        for episodes, e in zip(self.episodes, self.embeddings[arm], strict=True):
            for n, ep in episodes.items():
                logits = episode_logits(model, adapter, e, ep)
                per_n[n].append(float(F.cross_entropy(logits.reshape(-1, 2), ep[3].reshape(-1))))
        return {n: float(np.mean(v)) for n, v in per_n.items()}


# --- Files ----------------------------------------------------------------------


def weights_path(out_dir: Path, arm: str, step: int) -> Path:
    return out_dir / WEIGHTS_DIR / f"{arm}-step{step:05d}.pt"


def selected_weights_path(out_dir: Path, arm: str) -> Path:
    return out_dir / WEIGHTS_DIR / f"{arm}-selected.pt"


def load_adapter(model, arm: str, out_dir: Path = RESULTS_DIR) -> TextConditionedThinkingRows:
    """The arm's selected adapter (by dev validation loss), on the model's device."""
    adapter = TextConditionedThinkingRows.from_model(model)
    saved = torch.load(selected_weights_path(out_dir, arm), map_location="cpu")
    adapter.load_state_dict(saved["state_dict"])
    return adapter


def _save_weights(path: Path, adapter: TextConditionedThinkingRows, arm: str, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    torch.save({"arm": arm, "step": step, "state_dict": state}, path)


def _keep_rows(path: Path, fields: tuple[str, ...], max_step: int) -> None:
    """Drop rows written after the checkpoint a resumed run restarts from."""
    if not path.exists():
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(fields)
        return
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= max_step]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append(path: Path, rows: list[list]) -> None:
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _settings_json(settings: Settings) -> dict:
    d = dataclasses.asdict(settings)
    d["val_sizes"] = list(settings.val_sizes)
    return d


def settings_from_json(d: dict) -> Settings:
    return Settings(**{**d, "val_sizes": tuple(d["val_sizes"])})


def read_validation(out_dir: Path) -> dict[tuple[int, str, int], float]:
    """``(step, arm, n) -> logloss`` from ``validation.csv``."""
    with (out_dir / VALIDATION_CSV).open() as fh:
        return {
            (int(r["step"]), r["arm"], int(r["n"])): float(r["logloss"]) for r in csv.DictReader(fh)
        }


def select_checkpoints(
    settings: Settings, out_dir: Path, arms: tuple[str, ...] = ARMS
) -> dict[str, tuple[int, float]]:
    """Per arm: the checkpoint step with the lowest mean dev loss over ``val_sizes``."""
    val = read_validation(out_dir)
    out = {}
    for arm in arms:
        means = {
            step: float(np.mean([val[(step, arm, n)] for n in settings.val_sizes]))
            for step in checkpoint_steps(settings)
        }
        best = min(means, key=lambda s: (means[s], s))
        out[arm] = (best, means[best])
    return out


# --- Training -------------------------------------------------------------------


def run(settings: Settings, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_JSON
    state_path = out_dir / WEIGHTS_DIR / STATE_FILE
    device = default_device()
    if config_path.exists():
        saved = json.loads(config_path.read_text())["settings"]
        if saved != _settings_json(settings):
            raise SystemExit(f"{config_path} holds different settings; refusing to resume")
    else:
        config = {
            "settings": _settings_json(settings),
            "arms": list(ARMS),
            "validation_variant": VALIDATION_VARIANT,
            "story_mix_per_20": {v: k for v, k in _MIX},
            "optimizer": "Adam(betas=(0.9, 0.999), eps=1e-8, weight_decay=0), one per arm",
            "dev_scenes": [list(s) for s in dev_scene_specs(settings.val_scenes)],
            "device": str(device),
            "torch": torch.__version__,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    model = load_tabpfn(device)
    encoder = TextEncoder("cpu")
    adapters = {arm: TextConditionedThinkingRows.from_model(model) for arm in ARMS}
    opts = {arm: torch.optim.Adam(adapters[arm].parameters(), lr=settings.lr) for arm in ARMS}
    episodes_done = {arm: 0 for arm in ARMS}
    start_step, elapsed = 0, 0.0
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        for arm in ARMS:
            adapters[arm].load_state_dict(state["adapters"][arm])
            opts[arm].load_state_dict(state["optimizers"][arm])
        episodes_done = state["episodes"]
        start_step, elapsed = state["step"], state["elapsed_s"]
        print(f"resuming from step {start_step}", flush=True)
    _keep_rows(out_dir / PROGRESS_CSV, PROGRESS_FIELDS, start_step)
    _keep_rows(out_dir / VALIDATION_CSV, VALIDATION_FIELDS, start_step)

    dev = DevSet(settings, encoder, device)
    t_dev = time.time()

    def checkpoint(step: int) -> None:
        rows = []
        if step == 0:
            with torch.no_grad():
                native = {n: [] for n in dev.sizes}
                for episodes in dev.episodes:
                    for n, ep in episodes.items():
                        logits = binary_logits(model, *ep[:3])
                        native[n].append(float(F.cross_entropy(logits.reshape(-1, 2), ep[3].reshape(-1))))
            rows += [[0, NATIVE, n, len(dev.specs), f"{np.mean(v):.6f}"] for n, v in native.items()]
        for arm in ARMS:
            losses = dev.logloss(model, adapters[arm], arm)
            rows += [[step, arm, n, len(dev.specs), f"{v:.6f}"] for n, v in losses.items()]
            _save_weights(weights_path(out_dir, arm, step), adapters[arm], arm, step)
        _append(out_dir / VALIDATION_CSV, rows)
        torch.save(
            {
                "step": step,
                "elapsed_s": elapsed,
                "episodes": episodes_done,
                "adapters": {a: adapters[a].state_dict() for a in ARMS},
                "optimizers": {a: opts[a].state_dict() for a in ARMS},
            },
            state_path,
        )
        print(
            f"checkpoint step {step}: "
            + " ".join(f"{r[1]}@{r[2]}={r[4]}" for r in rows),
            flush=True,
        )

    if start_step == 0:
        checkpoint(0)
        print(f"dev set + step-0 validation: {time.time() - t_dev:.0f}s", flush=True)

    ckpts = set(checkpoint_steps(settings))
    window = {arm: [] for arm in ARMS}
    for step in range(start_step + 1, settings.steps + 1):
        t0 = time.time()
        lr = learning_rate(settings, step)
        for arm in ARMS:
            for group in opts[arm].param_groups:
                group["lr"] = lr
            opts[arm].zero_grad(set_to_none=True)
        step_loss = {arm: 0.0 for arm in ARMS}
        for j in range(settings.episodes_per_step):
            ep = make_episode(settings, (step - 1) * settings.episodes_per_step + j)
            tensors = tuple(torch.as_tensor(a, device=device) for a in ep.arrays)
            embeddings = dict(zip(ARMS, encoder.encode([ep.texts[a] for a in ARMS]), strict=True))
            for arm in ARMS:
                logits = episode_logits(model, adapters[arm], embeddings[arm], tensors)
                loss = F.cross_entropy(logits.reshape(-1, 2), tensors[3].reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss, arm {arm}, episode {ep.index}")
                (loss / settings.episodes_per_step).backward()
                step_loss[arm] += loss.item() / settings.episodes_per_step
                episodes_done[arm] += 1
        for arm in ARMS:
            opts[arm].step()
            window[arm].append(step_loss[arm])
        elapsed += time.time() - t0
        if step % settings.log_every == 0 or step == settings.steps:
            _append(
                out_dir / PROGRESS_CSV,
                [
                    [step, arm, episodes_done[arm], f"{lr:.6g}", f"{np.mean(window[arm]):.6f}", f"{elapsed:.1f}"]
                    for arm in ARMS
                ],
            )
            print(
                f"step {step} lr {lr:.2e} "
                + " ".join(f"{a}={np.mean(window[a]):.4f}" for a in ARMS)
                + f" elapsed {elapsed:.0f}s",
                flush=True,
            )
            window = {arm: [] for arm in ARMS}
        if step in ckpts:
            checkpoint(step)

    selected = select_checkpoints(settings, out_dir)
    for arm, (step, _) in selected.items():
        saved = torch.load(weights_path(out_dir, arm, step), map_location="cpu")
        torch.save(saved, selected_weights_path(out_dir, arm))
    summary = {
        "steps": settings.steps,
        "episodes": episodes_done,
        "train_wall_s": round(elapsed, 1),
        "selected": {
            arm: {"step": step, "val_logloss_mean": round(v, 6)} for arm, (step, v) in selected.items()
        },
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: {summary}", flush=True)


def parse_settings(
    argv: list[str] | None, description: str, default_out: Path
) -> tuple[Settings, Path]:
    """``Settings`` (one flag per field, e.g. ``--lr``) and ``--out`` from the command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, default=default_out)
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
    values["val_sizes"] = tuple(values["val_sizes"])
    return Settings(**values), args.out


def main(argv: list[str] | None = None) -> None:
    run(*parse_settings(argv, __doc__.splitlines()[0], RESULTS_DIR))


if __name__ == "__main__":
    main()
