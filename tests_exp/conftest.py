"""Shared fixtures: the real TabPFN-2.5 checkpoint, the real MiniLM encoder, episodes.

Tests skip (with a reason) only when the checkpoint or the encoder is absent
from the local Hugging Face cache.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from texttabpfn.capacity import stratified_index
from texttabpfn.encoder import ENCODER_NAME, TextEncoder
from texttabpfn.model import CHECKPOINT_FILE, CHECKPOINT_REPO, load_tabpfn
from texttabpfn.scenes import Scene

# The experiments run on MPS (``model.default_device``); CPU is checked as well.
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _cached(repo: str, filename: str) -> bool:
    return isinstance(try_to_load_from_cache(repo, filename), str)


requires_assets = pytest.mark.skipif(
    not (_cached(CHECKPOINT_REPO, CHECKPOINT_FILE) and _cached(ENCODER_NAME, "config.json")),
    reason="TabPFN-2.5 checkpoint or all-MiniLM-L6-v2 not in the local HF cache",
)


@functools.cache
def model_on(device: str):
    return load_tabpfn(device)


@pytest.fixture(scope="session")
def encoder() -> TextEncoder:
    return TextEncoder("cpu")


def make_episode(
    scene: Scene, n: int, batch: int, query_rows: int, device: str, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """Stratified supports (n, batch) and shared query rows (query_rows, batch)."""
    rng = np.random.default_rng([scene.family, scene.seed, 100, seed])
    pool_x, pool_y = scene.sample(1024, rng)
    q_x, q_y = scene.sample(query_rows, rng)
    idx = np.stack(
        [stratified_index(pool_y, n, scene.base_rate, rng) for _ in range(batch)], axis=1
    )
    arrays = (
        pool_x[idx],
        pool_y[idx],
        np.repeat(q_x[:, None], batch, axis=1),
        np.repeat(q_y[:, None], batch, axis=1),
    )
    return tuple(torch.as_tensor(a, device=device) for a in arrays)
