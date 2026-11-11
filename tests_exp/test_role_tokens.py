"""Role-token adapter (D-11): parity at initialisation, label isolation, role dependence.

- At W = 0 the role-token adapter reproduces native raw TabPFN-2.5 logits within
  1e-5 on every device in ``DEVICES`` (MPS, the experiments' device, and CPU),
  with real MiniLM role tokens, neutral tokens and per-member token batches.
- Changing query labels leaves its logits unchanged (non-zero W); the same call
  reacts to support labels, so the check is not vacuous.
- With non-zero weights the logits depend on the role assignment, so parity is
  not met by bypassing the adapter.
- Token contract: sentences never name a feature, one role = one token, groups
  are distinguishable, slots follow column order, empty slots are zero, and the
  control differs from the story arm only in role content.
- Stage 8 episodes: rows identical to stage 3's, the D-09 mix applied to roles.

Only training families and dev seeds are used; held-out families never enter.
"""

from __future__ import annotations

import collections
import re

import numpy as np
import pytest
import torch
from conftest import DEVICES, make_episode, model_on, requires_assets

from texttabpfn import training
from texttabpfn.adapter import episode_logits
from texttabpfn.model import binary_logits, checkpoint_rows
from texttabpfn.role_tokens import (
    MAX_SLOTS,
    NEUTRAL_SENTENCE,
    RoleTokenThinkingRows,
    role_sentences,
    slot_tokens,
    variant_sentences,
)
from texttabpfn.role_training import _ROLE_STREAM, make_role_episode
from texttabpfn.scenes import DEV_SEED_RANGE, Scene
from texttabpfn.stories import one_wrong_fact_roles, scene_roles

TOLERANCE = 1e-5
SAME = 1e-6  # identical inputs on one device; allows for kernel nondeterminism
CHANGED = 1e-3
# Training families on dev seeds; at least one with two measurement groups.
SCENES = [(f, DEV_SEED_RANGE[0] + 7 + f) for f in range(4)]
SUPPORT_SIZES = (8, 32)
BATCH, QUERY_ROWS = 3, 128
SETTINGS = training.Settings()


def _scene_with_groups(k: int) -> Scene:
    for seed in range(DEV_SEED_RANGE[0], DEV_SEED_RANGE[1]):
        scene = Scene(seed % 4, seed)
        if len({r["group"] for r in scene_roles(scene) if r["role"] == "measure"}) >= k:
            return scene
    raise AssertionError(f"no dev scene with {k} measurement groups")


def _random_weights(adapter: RoleTokenThinkingRows, seed: int) -> None:
    with torch.no_grad():
        w = torch.randn(adapter.weight_TES.shape, generator=torch.Generator().manual_seed(seed))
        adapter.weight_TES.copy_(w * 0.05)


# --- Token contract (CPU, no TabPFN) ----------------------------------------------


def test_sentences_state_only_the_role() -> None:
    for seed in range(DEV_SEED_RANGE[0], DEV_SEED_RANGE[0] + 40):
        roles = scene_roles(Scene(seed % 4, seed))
        sentences = role_sentences(roles)
        assert len(sentences) == len(roles)
        for s in sentences:
            assert not re.search(r"\bx\d+\b", s), s  # no feature name
        by_role = collections.defaultdict(set)
        for role, s in zip(roles, sentences, strict=True):
            by_role[(role["role"], role["group"], role["relation"])].add(s)
        assert all(len(v) == 1 for v in by_role.values())  # one role, one sentence
        groups = {k[1]: next(iter(v)) for k, v in by_role.items() if k[0] == "measure"}
        assert len(set(groups.values())) == len(groups)  # groups are distinguishable


def test_tokens_slots_and_control(encoder) -> None:
    scene = _scene_with_groups(2)
    p = scene.num_features
    story = slot_tokens(encoder, variant_sentences(scene, "story"))
    control = slot_tokens(encoder, variant_sentences(scene, "omitted"))
    assert story.shape == control.shape == (MAX_SLOTS, 384)
    assert not story[p:].any() and not control[p:].any()  # empty slots are zero
    neutral = encoder.encode_one(NEUTRAL_SENTENCE)
    assert torch.equal(control[:p], neutral.expand(p, -1))
    roles = scene_roles(scene)
    first_member: dict[int, int] = {}
    for j, r in enumerate(roles):
        if r["role"] == "measure":
            first_member.setdefault(r["group"], j)
    a, b = list(first_member.values())[:2]  # one member each of two different groups
    assert torch.cosine_similarity(story[a], story[b], dim=0) < 0.999
    for i in range(p):
        for j in range(p):
            if roles[i] == roles[j]:
                assert torch.equal(story[i], story[j])  # position is carried by the slot only


def test_role_episode_rows_and_mix() -> None:
    counts = collections.Counter()
    for index in range(60):
        rep = make_role_episode(SETTINGS, index)
        ep = training.make_episode(SETTINGS, index)
        for x, y in zip(rep.episode.arrays, ep.arrays, strict=True):
            np.testing.assert_array_equal(x, y)  # rows identical to stage 3
        assert rep.episode.variant == training.story_variant(SETTINGS, index)
        counts[rep.episode.variant] += 1
        scene = Scene(ep.family, ep.seed)
        assert rep.sentences["control"] == [NEUTRAL_SENTENCE] * scene.num_features
        if rep.episode.variant == "omitted":
            assert rep.sentences["story"] == rep.sentences["control"]
        elif rep.episode.variant == "story":
            assert rep.sentences["story"] == role_sentences(scene_roles(scene))
        else:
            rng = np.random.default_rng([_ROLE_STREAM, SETTINGS.seed, index])
            wrong = one_wrong_fact_roles(scene_roles(scene), rng)
            changed = [a != b for a, b in zip(wrong, scene_roles(scene), strict=True)]
            assert sum(changed) == 1
            assert rep.sentences["story"] == role_sentences(wrong)
            assert rep.sentences["story"] != role_sentences(scene_roles(scene))
    assert counts == {"story": 45, "omitted": 9, "one_wrong_fact": 6}


@requires_assets
def test_shape_parameters_and_column_sharing() -> None:
    model = model_on("cpu")
    adapter = RoleTokenThinkingRows.from_model(model)
    trainable = {name: tuple(p.shape) for name, p in adapter.named_parameters() if p.requires_grad}
    assert trainable == {"projection_PD": (25, 384), "weight_TES": (64, 192, 375)}
    assert sum(p.numel() for p in adapter.parameters()) == 4_617_600
    assert not adapter.weight_TES.any()
    assert torch.equal(adapter.theta_TE, checkpoint_rows(model))
    assert "theta_TE" not in adapter.state_dict()
    assert torch.equal(adapter.projection_PD, RoleTokenThinkingRows.from_model(model).projection_PD)

    _random_weights(adapter, 0)
    adapter.set_text(torch.randn(2, MAX_SLOTS, 384))
    x = torch.randn(2, 5, 4, 192)
    out, pos = adapter(x, single_eval_pos=3)
    assert out.shape == (2, 69, 4, 192) and pos == 67
    rows = out[:, :64]
    assert torch.equal(rows, rows[:, :, :1].expand_as(rows))  # identical across columns
    assert not torch.equal(rows[0], rows[1])  # one assignment per batch member
    assert torch.equal(out[:, 64:], x)


# --- Real checkpoint, CPU and MPS ---------------------------------------------------


@requires_assets
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family,seed", SCENES)
def test_init_matches_native(device: str, family: int, seed: int, encoder) -> None:
    model = model_on(device)
    native_module = model.add_thinking_rows
    adapter = RoleTokenThinkingRows.from_model(model)
    scene = Scene(family, seed)
    rng = np.random.default_rng(seed)
    story = slot_tokens(encoder, variant_sentences(scene, "story"))
    per_member = torch.stack(
        [slot_tokens(encoder, variant_sentences(scene, v, rng)) for v in ("story", "omitted", "wrong")]
    )
    with torch.no_grad():
        for n in SUPPORT_SIZES:
            episode = make_episode(scene, n, BATCH, QUERY_ROWS, device)
            native = binary_logits(model, *episode[:3])
            assert torch.isfinite(native).all()
            for tokens in (story, per_member):
                adapted = episode_logits(model, adapter, tokens, episode)
                diff = (adapted - native).abs().max().item()
                assert diff <= TOLERANCE, f"n={n} {device}: max |diff| {diff:.2e}"
    assert model.add_thinking_rows is native_module


@requires_assets
@pytest.mark.parametrize("device", DEVICES)
def test_query_labels_do_not_change_logits(device: str, encoder) -> None:
    model = model_on(device)
    adapter = RoleTokenThinkingRows.from_model(model)
    _random_weights(adapter, 1)
    scene = _scene_with_groups(2)
    tokens = slot_tokens(encoder, variant_sentences(scene, "story"))
    xs, ys, xq, yq = make_episode(scene, 16, BATCH, QUERY_ROWS, device)

    def logits(*ep):
        with torch.no_grad():
            return episode_logits(model, adapter, tokens, ep)

    base = logits(xs, ys, xq, yq)
    gen = torch.Generator().manual_seed(0)
    shuffled = yq[torch.randperm(len(yq), generator=gen).to(yq.device)]
    random = torch.randint(0, 2, yq.shape, generator=gen).to(yq.device)
    for other in (1 - yq, shuffled, random):
        assert not torch.equal(other, yq)
        assert (logits(xs, ys, xq, other) - base).abs().max().item() <= SAME
    assert (logits(xs, 1 - ys, xq, yq) - base).abs().max().item() > CHANGED


@requires_assets
@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_weights_depend_on_role_assignment(device: str, encoder) -> None:
    model = model_on(device)
    adapter = RoleTokenThinkingRows.from_model(model)
    _random_weights(adapter, 2)
    scene = _scene_with_groups(2)
    rng = np.random.default_rng(0)
    episode = make_episode(scene, 16, BATCH, QUERY_ROWS, device)
    with torch.no_grad():
        native = binary_logits(model, *episode[:3])
        out = {
            v: episode_logits(model, adapter, slot_tokens(encoder, variant_sentences(scene, v, rng)), episode)
            for v in ("story", "omitted", "wrong", "one_wrong_fact")
        }
    assert (out["story"] - native).abs().max().item() > CHANGED
    for v in ("omitted", "wrong", "one_wrong_fact"):
        assert (out["story"] - out[v]).abs().max().item() > CHANGED, v
