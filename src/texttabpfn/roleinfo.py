"""Stage 7: value of the story's role facts to native TabPFN-2.5 (D-12, D-13).

Design, fixed before any result was seen:

- Rows: the stage 4 rows exactly. ``evaluation.scene_episodes`` is called with
  the D-08 settings over ``eval_scene_specs(200)``: every n = 8 ... 1024, 5
  nested support seeds, 512 query rows.
- Preprocessing (``role_preprocess``, D-12) reads only the scene's role facts
  (``Scene.describe()["roles"]``, the content of the template story) and
  unlabeled feature values:
  - columns with role ``irrelevant`` are dropped;
  - ``mb`` columns are kept unchanged;
  - each measurement group (role ``measure``, same ``group``) becomes one column:
    the first principal component of the group's columns, each standardized
    with its own mean and std. Its sign is fixed so the group's first column (in
    column order) has a non-negative loading, so random measurement signs
    cannot cancel. The column takes the place of the group's first member.
  - The collapse is fitted per support seed on that seed's support rows plus
    the query rows (all unlabeled at prediction time).
- Arms: ``native`` (raw rows) and ``roles`` (preprocessed rows), both scored by
  raw TabPFN-2.5 (``model.binary_logits``) on the same device. Native is
  re-scored here; the report checks it against stage 4's native arm.
- Metric: per (scene, n, arm), the mean query log loss over the support seeds.

Run: ``uv run python -m texttabpfn.roleinfo`` (resumable: a restart keeps every
scene with all its rows and recomputes the rest). Writes ``config.json``,
``scores.csv`` and, when done, ``summary.json`` to ``results/07-roleinfo/``.
Report: ``python -m texttabpfn.report roleinfo``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from texttabpfn import evaluation
from texttabpfn.model import binary_logits, default_device, load_tabpfn
from texttabpfn.scenes import Scene, eval_scene_specs

RESULTS_DIR = Path("results/07-roleinfo")
CONFIG_JSON = "config.json"
SCORES_CSV = evaluation.SCORES_CSV
SUMMARY_JSON = "summary.json"
SCORE_FIELDS = evaluation.SCORE_FIELDS

NATIVE = evaluation.NATIVE
ROLES = "roles"
ARMS = (NATIVE, ROLES)
METHOD = (
    "drop irrelevant; keep mb; each measure group -> first principal component of its "
    "standardized columns, sign so the group's first column loads >= 0, fitted per support "
    "seed on support + query feature values"
)

# Rows the measurement-group collapse is fitted on: stage 7's (D-12) design uses
# support + query feature values; "support" fits on support rows only and applies
# the fitted transform to the query rows (stage 9 secondary arm).
FIT_SUPPORT_QUERY = "support+query"
FIT_SUPPORT = "support"
FITS = (FIT_SUPPORT_QUERY, FIT_SUPPORT)


def _first_component(x: np.ndarray, fit_rows: int | None = None) -> np.ndarray:
    """First principal component scores of ``x`` (R, K, g), fitted per episode k.

    The mean, std and loadings come from the first ``fit_rows`` rows (all rows
    when ``None``); the scores cover every row.
    """
    f = x if fit_rows is None else x[:fit_rows]
    std = f.std(axis=0)
    z = (x - f.mean(axis=0)) / np.where(std > 0, std, 1.0)
    zf = z if fit_rows is None else z[:fit_rows]
    cov = np.einsum("rki,rkj->kij", zf, zf) / len(zf)
    vec = np.linalg.eigh(cov)[1][..., -1]  # (K, g): eigenvector of the largest eigenvalue
    vec = vec * np.where(vec[:, :1] < 0, -1.0, 1.0)
    return np.einsum("rkg,kg->rk", z, vec)


def _preprocess(
    roles: Sequence[Mapping[str, object]],
    x_support: np.ndarray,
    x_query: np.ndarray,
    fit_rows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_support.shape[-1] != len(roles) or x_query.shape[-1] != len(roles):
        raise ValueError(f"{len(roles)} roles for {x_support.shape[-1]} / {x_query.shape[-1]} columns")
    rows = np.concatenate([x_support, x_query]).astype(np.float64)
    columns, groups_done = [], set()
    for j, role in enumerate(roles):
        kind = role["role"]
        if kind == "mb":
            columns.append(rows[..., j])
        elif kind == "measure":
            group = role["group"]
            if group in groups_done:
                continue
            groups_done.add(group)
            members = [i for i, r in enumerate(roles) if r["role"] == "measure" and r["group"] == group]
            columns.append(_first_component(rows[..., members], fit_rows))
        elif kind != "irrelevant":
            raise ValueError(f"unknown role {kind!r}")
    out = np.stack(columns, axis=-1).astype(x_support.dtype)
    return out[: len(x_support)], out[len(x_support) :]


def role_preprocess(
    roles: Sequence[Mapping[str, object]], x_support: np.ndarray, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Story-guided preprocessing (D-12) of support (N, K, p) and query (M, K, p) rows.

    ``roles`` is ``Scene.describe()["roles"]``, one entry per column. The K
    episodes (support seeds) are independent: each group collapse is fitted on
    that episode's support plus query rows, so the query output depends on the
    whole query batch. Returns (N, K, p') and (M, K, p').
    """
    return _preprocess(roles, x_support, x_query, None)


def role_preprocess_support_fit(
    roles: Sequence[Mapping[str, object]], x_support: np.ndarray, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``role_preprocess`` with each group collapse fitted on the episode's support
    rows only and applied unchanged to the query rows."""
    return _preprocess(roles, x_support, x_query, len(x_support))


def preprocess_episode(
    roles: Sequence[Mapping[str, object]], arrays: tuple[np.ndarray, ...], fit: str = FIT_SUPPORT_QUERY
) -> tuple[np.ndarray, ...]:
    """Preprocessing of a whole ``scene_episodes`` tuple (xs, ys, xq, yq); labels pass
    through. Pass the full query batch: the support+query fit depends on all of it."""
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}; expected one of {FITS}")
    xs, ys, xq, yq = arrays
    fn = role_preprocess if fit == FIT_SUPPORT_QUERY else role_preprocess_support_fit
    xs, xq = fn(roles, xs, xq)
    return xs, ys, xq, yq


def _config_core() -> dict:
    rows = evaluation.Settings()
    return {
        "row_settings": evaluation._settings_json(rows),
        "arms": list(ARMS),
        "method": METHOD,
        "scenes": [list(s) for s in eval_scene_specs(rows.num_scenes)],
    }


def _complete_scenes(out_dir: Path) -> set[tuple[int, int]]:
    need = {(n, arm) for n in evaluation.Settings().sizes for arm in ARMS}
    return evaluation.keep_complete_scenes(out_dir / SCORES_CSV, SCORE_FIELDS, need)


def read_scores(out_dir: Path) -> dict[tuple[int, int, int, str], float]:
    """``(family, seed, n, arm) -> logloss`` from ``scores.csv``."""
    return evaluation.read_scores(out_dir)


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

    done = _complete_scenes(out_dir)
    todo = [s for s in specs if s not in done]
    print(f"{len(done)} of {len(specs)} scenes already scored; {len(todo)} to go", flush=True)
    if todo:
        model = load_tabpfn(device)

        def native_logits(ep):
            return binary_logits(model, *ep[:3])

        t_start = time.time()
        for i, (family, seed) in enumerate(todo, 1):
            t0 = time.time()
            se = evaluation.scene_episodes(rows_settings, family, seed)
            roles = Scene(family, seed).describe()["roles"]
            rows = []
            for n in rows_settings.sizes:
                inputs = {NATIVE: se.arrays[n], ROLES: preprocess_episode(roles, se.arrays[n])}
                for arm in ARMS:
                    episode = tuple(torch.as_tensor(a, device=device) for a in inputs[arm])
                    loss = evaluation.query_logloss(native_logits, episode, chunk)
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
        for n in rows_settings.sizes
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
    args = parser.parse_args(argv)
    run(args.out, args.chunk)


if __name__ == "__main__":
    main()
