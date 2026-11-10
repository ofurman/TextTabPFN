"""Stage 1: channel capacity of TabPFN-2.5's thinking rows.

For each dev scene, the thinking rows are optimised with label access (a
diagnostic, D-02), with the backbone frozen:

- ``shared``: a per-scene trainable copy of ``row_token_values_TE`` (64 x 192);
- ``per-column``: a per-scene trainable 64 x C x 192 tensor (C = feature groups
  + target column).

One set of rows per scene serves every support size, as a text-to-rows mapping
would. Training draws support/query episodes from a 20k-row pool; the rows
with the lowest loss on a separate validation set (checkpoint value included)
are kept. Scoring uses disjoint evaluation rows: 512 fixed query rows and fixed
stratified, nested support subsets at n = 8, 16, 32 for several support seeds,
averaged per scene.

Run: ``uv run python -m texttabpfn.capacity [--arm shared|per-column]``
(resumable; appends per scene to ``results/01-capacity/<arm>.csv``). Report:
``python -m texttabpfn.report capacity [--arm per-column]``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from texttabpfn.model import (
    PerColumnThinkingRows,
    SharedThinkingRows,
    binary_logits,
    checkpoint_rows,
    default_device,
    load_tabpfn,
    num_model_columns,
    thinking_rows,
)
from texttabpfn.scenes import Scene, dev_scene_specs

RESULTS_DIR = Path("results/01-capacity")
CONFIG_JSON = "config.json"
ARMS = ("shared", "per-column")
SUPPORT_SIZES = (8, 16, 32)
CSV_FIELDS = (
    "scene",
    "family",
    "seed",
    "num_features",
    "arm",
    "n",
    "native_logloss",
    "optimized_logloss",
    "gain",
    "best_step",
    "val_native",
    "val_best",
)

# Per-scene RNG stream ids (each stream is seeded by (family, seed, stream)).
_POOL_STREAM, _VAL_STREAM, _QUERY_STREAM, _SUPPORT_STREAM = 10, 11, 12, 13
_SUPPORT_SEED_STREAM, _TRAIN_STREAM = 14, 15


@dataclasses.dataclass(frozen=True)
class Settings:
    num_scenes: int = 50
    pool_rows: int = 20_000
    val_rows: int = 2_048
    eval_query_rows: int = 512
    eval_support_rows: int = 256
    support_seeds: int = 5
    steps: int = 400
    batch_episodes: int = 8
    train_query_rows: int = 128
    lr: float = 3e-2
    val_every: int = 25
    val_episodes_per_n: int = 4
    val_query_rows: int = 256


def stratified_counts(n: int, base_rate: float) -> tuple[int, int]:
    """(negatives, positives) for a stratified draw of size n; both classes >= 1."""
    pos = int(np.clip(round(n * base_rate), 1, n - 1))
    return n - pos, pos


def stratified_index(
    y: np.ndarray, n: int, base_rate: float, rng: np.random.Generator
) -> np.ndarray:
    neg, pos = stratified_counts(n, base_rate)
    neg_idx = rng.choice(np.flatnonzero(y == 0), size=neg, replace=False)
    pos_idx = rng.choice(np.flatnonzero(y == 1), size=pos, replace=False)
    return rng.permutation(np.concatenate([neg_idx, pos_idx]))


def nested_supports(
    y: np.ndarray, sizes: tuple[int, ...], base_rate: float, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Nested stratified supports: the size-n set contains every smaller set."""
    neg_order = rng.permutation(np.flatnonzero(y == 0))
    pos_order = rng.permutation(np.flatnonzero(y == 1))
    out = {}
    for n in sizes:
        neg, pos = stratified_counts(n, base_rate)
        out[n] = np.concatenate([neg_order[:neg], pos_order[:pos]])
    return out


class SceneData:
    """Pool, validation and evaluation rows of one scene, on device."""

    def __init__(self, scene: Scene, settings: Settings, device: torch.device) -> None:
        f, s = scene.family, scene.seed
        self.scene = scene
        self.base_rate = scene.base_rate
        pool_x, pool_y = scene.sample(settings.pool_rows, np.random.default_rng([f, s, _POOL_STREAM]))
        val_x, val_y = scene.sample(settings.val_rows, np.random.default_rng([f, s, _VAL_STREAM]))
        q_x, q_y = scene.sample(settings.eval_query_rows, np.random.default_rng([f, s, _QUERY_STREAM]))
        sup_x, sup_y = scene.sample(
            settings.eval_support_rows, np.random.default_rng([f, s, _SUPPORT_STREAM])
        )
        self.pool_y_np = pool_y
        self.val_y_np = val_y
        self.pool_x = torch.tensor(pool_x, device=device)
        self.pool_y = torch.tensor(pool_y, device=device)
        self.val_x = torch.tensor(val_x, device=device)
        self.val_y = torch.tensor(val_y, device=device)

        # Fixed evaluation episodes: per n, a batch over support seeds that all
        # share the same 512 query rows.
        seed_rng = np.random.default_rng([f, s, _SUPPORT_SEED_STREAM])
        supports = [
            nested_supports(sup_y, SUPPORT_SIZES, self.base_rate, seed_rng)
            for _ in range(settings.support_seeds)
        ]
        k = settings.support_seeds
        self.eval_episodes: dict[int, tuple[torch.Tensor, ...]] = {}
        for n in SUPPORT_SIZES:
            xs = np.stack([sup_x[sp[n]] for sp in supports], axis=1)
            ys = np.stack([sup_y[sp[n]] for sp in supports], axis=1)
            xq = np.repeat(q_x[:, None], k, axis=1)
            yq = np.repeat(q_y[:, None], k, axis=1)
            self.eval_episodes[n] = _to_device((xs, ys, xq, yq), device)

        # Fixed validation episodes drawn from the validation rows.
        val_rng = np.random.default_rng([f, s, _VAL_STREAM, 1])
        self.val_episodes = {
            n: _to_device(
                _draw_episodes(
                    val_x, val_y, n, settings.val_episodes_per_n,
                    settings.val_query_rows, self.base_rate, val_rng,
                ),
                device,
            )
            for n in SUPPORT_SIZES
        }

    def train_episodes(
        self, n: int, batch: int, query_rows: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, ...]:
        idx_s, idx_q = _episode_indices(
            self.pool_y_np, n, batch, query_rows, self.base_rate, rng
        )
        idx_s = torch.as_tensor(idx_s, device=self.pool_x.device)
        idx_q = torch.as_tensor(idx_q, device=self.pool_x.device)
        return (
            self.pool_x[idx_s], self.pool_y[idx_s],
            self.pool_x[idx_q], self.pool_y[idx_q],
        )


def _episode_indices(
    y: np.ndarray, n: int, batch: int, query_rows: int, base_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Support (n, batch) and disjoint query (query_rows, batch) row indices."""
    sup, qry = [], []
    for _ in range(batch):
        s_idx = stratified_index(y, n, base_rate, rng)
        mask = np.ones(len(y), dtype=bool)
        mask[s_idx] = False
        q_idx = rng.choice(np.flatnonzero(mask), size=query_rows, replace=False)
        sup.append(s_idx)
        qry.append(q_idx)
    return np.stack(sup, axis=1), np.stack(qry, axis=1)


def _draw_episodes(
    x: np.ndarray, y: np.ndarray, n: int, batch: int, query_rows: int,
    base_rate: float, rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    s_idx, q_idx = _episode_indices(y, n, batch, query_rows, base_rate, rng)
    return x[s_idx], y[s_idx], x[q_idx], y[q_idx]


def _to_device(arrays: tuple[np.ndarray, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(a, device=device) for a in arrays)


def episode_logloss(model: nn.Module, episode: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Mean query cross-entropy over all query rows and batch members."""
    xs, ys, xq, yq = episode
    logits = binary_logits(model, xs, ys, xq)
    return F.cross_entropy(logits.reshape(-1, 2), yq.reshape(-1))


@torch.no_grad()
def eval_per_n(model: nn.Module, episodes: dict[int, tuple[torch.Tensor, ...]]) -> dict[int, float]:
    """Log loss per support size. Every support seed scores the same query rows,
    so the batch mean equals the per-scene average over support seeds."""
    return {n: float(episode_logloss(model, ep)) for n, ep in episodes.items()}


def optimize_rows(
    model: nn.Module, rows: nn.Module, data: SceneData, settings: Settings
) -> tuple[int, float, float]:
    """Adam on pool episodes; keep the rows with the lowest validation loss.

    Returns (best step, validation loss at init, best validation loss). Step 0 is
    the checkpoint value, so the kept rows are never worse on validation.
    """
    f, s = data.scene.family, data.scene.seed
    rng = np.random.default_rng([f, s, _TRAIN_STREAM])
    opt = torch.optim.Adam(rows.parameters(), lr=settings.lr)

    def val_loss() -> float:
        return float(np.mean(list(eval_per_n(model, data.val_episodes).values())))

    best_state = {k: v.detach().clone() for k, v in rows.state_dict().items()}
    val_init = best_val = val_loss()
    best_step = 0
    for step in range(1, settings.steps + 1):
        n = SUPPORT_SIZES[(step - 1) % len(SUPPORT_SIZES)]
        episode = data.train_episodes(n, settings.batch_episodes, settings.train_query_rows, rng)
        loss = episode_logloss(model, episode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % settings.val_every == 0 or step == settings.steps:
            v = val_loss()
            if v < best_val:
                best_val, best_step = v, step
                best_state = {k: t.detach().clone() for k, t in rows.state_dict().items()}
    rows.load_state_dict(best_state)
    return best_step, val_init, best_val


def arm_csv(out_dir: Path, arm: str) -> Path:
    """Per-arm results file, so the two arms can run as separate processes."""
    return out_dir / f"{arm}.csv"


def run(settings: Settings, out_dir: Path, arm: str) -> None:
    """Optimise one arm on every dev scene; resumes from the arm's CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = arm_csv(out_dir, arm)
    done: set[int] = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            done = {int(row["scene"]) for row in csv.DictReader(fh)}
    else:
        with csv_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(CSV_FIELDS)

    device = default_device()
    model = load_tabpfn(device)
    base_rows = checkpoint_rows(model)
    config = {
        "settings": dataclasses.asdict(settings),
        "arms": list(ARMS),
        "support_sizes": list(SUPPORT_SIZES),
        "scenes": [list(spec) for spec in dev_scene_specs(settings.num_scenes)],
        "device": str(device),
        "torch": torch.__version__,
    }
    (out_dir / CONFIG_JSON).write_text(json.dumps(config, indent=2) + "\n")

    for scene_idx, (family, seed) in enumerate(dev_scene_specs(settings.num_scenes)):
        if scene_idx in done:
            continue
        t0 = time.time()
        scene = Scene(family, seed)
        data = SceneData(scene, settings, device)
        native = eval_per_n(model, data.eval_episodes)
        torch.manual_seed(0)
        if arm == "shared":
            rows: nn.Module = SharedThinkingRows(base_rows)
        else:
            rows = PerColumnThinkingRows(base_rows, num_model_columns(scene.num_features))
        rows = rows.to(device)
        with thinking_rows(model, rows):
            best_step, val_init, val_best = optimize_rows(model, rows, data, settings)
            optimized = eval_per_n(model, data.eval_episodes)
        with csv_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            for n in SUPPORT_SIZES:
                writer.writerow([
                    scene_idx, family, seed, scene.num_features, arm, n,
                    f"{native[n]:.6f}", f"{optimized[n]:.6f}",
                    f"{native[n] - optimized[n]:.6f}", best_step,
                    f"{val_init:.6f}", f"{val_best:.6f}",
                ])
        print(
            f"scene {scene_idx} fam {family} arm {arm} best_step {best_step} "
            + " ".join(f"n{n}:{native[n]:.4f}->{optimized[n]:.4f}" for n in SUPPORT_SIZES)
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--arm", choices=ARMS, action="append",
        help="arm to run (repeatable; default: both, one after the other)",
    )
    defaults = Settings()
    for field in dataclasses.fields(Settings):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=type(getattr(defaults, field.name)),
            default=getattr(defaults, field.name),
        )
    args = parser.parse_args(argv)
    settings = Settings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)})
    for arm in args.arm or ARMS:
        run(settings, args.out, arm)


if __name__ == "__main__":
    main()
