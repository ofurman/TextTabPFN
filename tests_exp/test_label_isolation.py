"""Changing query labels leaves adapter logits unchanged.

The adapter runs with a non-zero W and a real story embedding, through
``episode_logits`` (the episode path later stages use), on the real checkpoint
on every device in ``DEVICES`` (MPS, the experiments' device, and CPU).
So that the test cannot pass trivially, the same call must react to support
labels, and each query row's logits must not depend on the other query rows
(no transductive path through which query labels could enter).
"""

from __future__ import annotations

import pytest
import torch
from conftest import DEVICES, make_episode, model_on, requires_assets

from texttabpfn.adapter import TextConditionedThinkingRows, episode_logits
from texttabpfn.scenes import EVAL_SEED_RANGE, Scene
from texttabpfn.stories import scene_text

pytestmark = requires_assets

SCENE = (4, EVAL_SEED_RANGE[0] + 11)
N, BATCH, QUERY_ROWS = 16, 3, 128
SAME = 1e-6  # identical inputs on one device; allows for kernel nondeterminism
CHANGED = 1e-2


@pytest.fixture(scope="module", params=DEVICES)
def setup(request, encoder):
    device = request.param
    model = model_on(device)
    adapter = TextConditionedThinkingRows.from_model(model)
    with torch.no_grad():
        adapter.weight_TED.copy_(
            torch.randn(adapter.weight_TED.shape, generator=torch.Generator().manual_seed(1)) * 0.05
        )
    scene = Scene(*SCENE)
    embedding = encoder.encode_one(scene_text(scene, "story"))
    episode = make_episode(scene, N, BATCH, QUERY_ROWS, device)

    def logits(ep):
        with torch.no_grad():
            return episode_logits(model, adapter, embedding, ep)

    return logits, episode


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_query_labels_do_not_change_logits(setup) -> None:
    logits, (xs, ys, xq, yq) = setup
    base = logits((xs, ys, xq, yq))
    gen = torch.Generator().manual_seed(0)
    flipped = 1 - yq
    shuffled = yq[torch.randperm(len(yq), generator=gen).to(yq.device)]
    random = torch.randint(0, 2, yq.shape, generator=gen).to(yq.device)
    for other in (flipped, shuffled, random):
        assert not torch.equal(other, yq)
        assert _max_diff(logits((xs, ys, xq, other)), base) <= SAME


def test_support_labels_do_change_logits(setup) -> None:
    logits, (xs, ys, xq, yq) = setup
    base = logits((xs, ys, xq, yq))
    assert _max_diff(logits((xs, 1 - ys, xq, yq)), base) > CHANGED
    rolled = torch.roll(ys, shifts=1, dims=0)
    assert not torch.equal(rolled, ys)
    assert _max_diff(logits((xs, rolled, xq, yq)), base) > CHANGED


def test_query_rows_are_scored_independently(setup) -> None:
    logits, (xs, ys, xq, yq) = setup
    base = logits((xs, ys, xq, yq))
    half = QUERY_ROWS // 2
    xq_other = xq.clone()
    xq_other[half:] = xq[half:].flip(0) * -1.0
    moved = logits((xs, ys, xq_other, yq))
    assert _max_diff(moved[:half], base[:half]) <= 1e-5
    assert _max_diff(moved[half:], base[half:]) > CHANGED
