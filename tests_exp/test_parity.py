"""The adapter with W = 0 reproduces native raw TabPFN-2.5 logits within 1e-5.

Runs on the real checkpoint and real MiniLM story embeddings, on every device in
``DEVICES``: MPS (the experiments' device) and CPU. Native and adapted logits are
compared on the same device. Non-triviality: with W != 0 the same call path
changes the logits and depends on the text, so the adapter is in the forward
pass and parity is not met by bypassing it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import DEVICES, make_episode, model_on, requires_assets

from texttabpfn.adapter import TextConditionedThinkingRows, episode_logits
from texttabpfn.model import binary_logits, checkpoint_rows
from texttabpfn.scenes import DEV_SEED_RANGE, EVAL_SEED_RANGE, Scene
from texttabpfn.stories import scene_text

pytestmark = requires_assets

TOLERANCE = 1e-5
# One scene per family: training families from dev seeds, held-out from eval seeds.
SCENES = [(f, DEV_SEED_RANGE[0] + 7 + f) for f in range(4)] + [
    (f, EVAL_SEED_RANGE[0] + 7 + f) for f in (4, 5)
]
SUPPORT_SIZES = (8, 32)
BATCH, QUERY_ROWS = 3, 128


def _story_embedding(encoder, scene: Scene) -> torch.Tensor:
    return encoder.encode_one(scene_text(scene, "story"))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family,seed", SCENES)
def test_zero_w_matches_native(device: str, family: int, seed: int, encoder) -> None:
    model = model_on(device)
    native_module = model.add_thinking_rows
    adapter = TextConditionedThinkingRows.from_model(model)
    scene = Scene(family, seed)
    rng = np.random.default_rng(seed)
    one_text = _story_embedding(encoder, scene)
    per_member = encoder.encode([scene_text(scene, v, rng) for v in ("story", "omitted", "wrong")])
    with torch.no_grad():
        for n in SUPPORT_SIZES:
            episode = make_episode(scene, n, BATCH, QUERY_ROWS, device)
            native = binary_logits(model, *episode[:3])
            assert torch.isfinite(native).all()
            for embedding in (one_text, per_member):
                adapted = episode_logits(model, adapter, embedding, episode)
                diff = (adapted - native).abs().max().item()
                assert diff <= TOLERANCE, f"n={n} {device}: max |diff| {diff:.2e}"
    assert model.add_thinking_rows is native_module


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_w_changes_logits_and_depends_on_text(device: str, encoder) -> None:
    model = model_on(device)
    scene = Scene(*SCENES[0])
    adapter = TextConditionedThinkingRows.from_model(model)
    with torch.no_grad():
        adapter.weight_TED.copy_(
            torch.randn(adapter.weight_TED.shape, generator=torch.Generator().manual_seed(0)) * 0.05
        )
    episode = make_episode(scene, 16, BATCH, QUERY_ROWS, device)
    story = _story_embedding(encoder, scene)
    schema = encoder.encode_one(scene_text(scene, "omitted"))
    with torch.no_grad():
        native = binary_logits(model, *episode[:3])
        with_story = episode_logits(model, adapter, story, episode)
        with_schema = episode_logits(model, adapter, schema, episode)
    assert (with_story - native).abs().max().item() > 1e-3
    assert (with_story - with_schema).abs().max().item() > 1e-3


def test_adapter_shape_parameters_and_column_sharing() -> None:
    model = model_on("cpu")
    theta = checkpoint_rows(model)
    adapter = TextConditionedThinkingRows.from_model(model)
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    assert [tuple(p.shape) for p in trainable] == [(64, 192, 384)]
    assert sum(p.numel() for p in trainable) == 64 * 192 * 384 == 4_718_592
    assert not adapter.weight_TED.any()
    assert torch.equal(adapter.theta_TE, theta)
    assert "theta_TE" not in adapter.state_dict()  # θ is frozen, never saved

    with torch.no_grad():
        adapter.weight_TED.normal_()
    adapter.set_text(torch.randn(2, 384))
    x = torch.randn(2, 5, 4, 192)
    out, pos = adapter(x, single_eval_pos=3)
    assert out.shape == (2, 69, 4, 192) and pos == 67
    rows = out[:, :64]
    assert torch.equal(rows, rows[:, :, :1].expand_as(rows))  # identical across columns
    assert not torch.equal(rows[0], rows[1])  # one text per batch member
    assert torch.equal(out[:, 64:], x)
