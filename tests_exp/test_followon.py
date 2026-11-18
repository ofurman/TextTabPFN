"""Follow-on adapter (B-09, D-16): parity at initialisation, label isolation, text dependence.

- At W = 0 the bge follow-on adapter reproduces native raw TabPFN-2.5 logits
  within 1e-5 on every device in ``DEVICES`` (MPS, the experiments' device, and
  CPU), with real bge embeddings of schema+story, schema-only and corrupted
  texts, one text or one per batch member.
- Changing query labels leaves its logits unchanged (non-zero W); the same call
  reacts to support labels, so the check is not vacuous.
- With non-zero weights the logits depend on the conditioning text, so parity
  is not met by bypassing the adapter.
- Encoder contract: CLS pooling, L2-normalised, 768-d, cached per text, texts
  over 512 tokens rejected. Episodes: stage 3's rows and texts, D-09 mix.
- ``report followon-training`` passes on complete results and fails when an arm
  is missing or the episode counts differ.

Only training families and dev seeds are used; held-out families never enter.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import DEVICES, make_episode, model_on, requires_assets
from huggingface_hub import try_to_load_from_cache

from texttabpfn import followon, report, training
from texttabpfn.adapter import episode_logits
from texttabpfn.followon import ARMS, BgeEncoder, FollowOnThinkingRows
from texttabpfn.model import binary_logits, checkpoint_rows
from texttabpfn.scenes import DEV_SEED_RANGE, Scene
from texttabpfn.stories import scene_text

TOLERANCE = 1e-5
SAME = 1e-6  # identical inputs on one device; allows for kernel nondeterminism
CHANGED = 1e-3
SCENES = [(f, DEV_SEED_RANGE[0] + 7 + f) for f in range(4)]  # training families, dev seeds
SUPPORT_SIZES = (8, 32)
BATCH, QUERY_ROWS = 3, 128
SETTINGS = training.Settings()

requires_bge = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(followon.ENCODER_NAME, "config.json"), str),
    reason="bge-base-en-v1.5 not in the local HF cache",
)


@pytest.fixture(scope="module")
def bge() -> BgeEncoder:
    return BgeEncoder("cpu")


def _random_weights(adapter: FollowOnThinkingRows, seed: int, scale: float = 0.05) -> None:
    with torch.no_grad():
        w = torch.randn(adapter.weight_TED.shape, generator=torch.Generator().manual_seed(seed))
        adapter.weight_TED.copy_(w * scale)


def _texts(scene: Scene, rng: np.random.Generator) -> dict[str, str]:
    return {v: scene_text(scene, v, rng) for v in ("story", "omitted", "wrong", "one_wrong_fact")}


# --- Encoder and episodes ---------------------------------------------------------


@requires_bge
def test_encoder_contract(bge: BgeEncoder) -> None:
    scene = Scene(*SCENES[0])
    texts = [scene_text(scene, "story"), scene_text(scene, "omitted")]
    emb = bge.encode(texts)
    assert emb.shape == (2, followon.EMBED_DIM) and emb.dtype == torch.float32
    torch.testing.assert_close(emb.norm(dim=1), torch.ones(2), atol=1e-5, rtol=0)
    assert torch.equal(bge.encode_one(texts[0]), emb[0])  # cached per text
    assert not torch.equal(emb[0], emb[1])
    with pytest.raises(ValueError, match="exceeds 512"):
        bge.encode(["word " * 600])


def test_token_lengths_fit_the_window() -> None:
    lengths = followon.token_lengths([scene_text(Scene(f, s), "story") for f, s in SCENES])
    assert 0 < max(lengths) <= followon.MAX_TOKENS


def test_episode_texts_and_mix() -> None:
    counts = collections.Counter()
    for index in range(60):
        ep = training.make_episode(SETTINGS, index)
        texts = followon.episode_texts(ep)
        scene = Scene(ep.family, ep.seed)
        assert texts["control"] == scene_text(scene, "omitted")
        assert texts["story"] == ep.texts["story"]  # stage 3's story-arm text, D-09 mix
        if ep.variant == "omitted":
            assert texts["story"] == texts["control"]
        elif ep.variant == "story":
            assert texts["story"] == scene_text(scene, "story")
        else:
            assert texts["story"] not in (scene_text(scene, "story"), texts["control"])
        counts[ep.variant] += 1
    assert counts == {"story": 45, "omitted": 9, "one_wrong_fact": 6}


# --- Adapter ----------------------------------------------------------------------


@requires_assets
def test_shape_parameters_and_column_sharing() -> None:
    model = model_on("cpu")
    adapter = FollowOnThinkingRows.from_model(model)
    trainable = {name: tuple(p.shape) for name, p in adapter.named_parameters() if p.requires_grad}
    assert trainable == {"weight_TED": (64, 192, 384), "projection_PD": (384, 768)}
    assert followon.trainable_parameters(adapter) == 5_013_504
    assert not adapter.weight_TED.any()
    assert torch.equal(adapter.theta_TE, checkpoint_rows(model))
    assert "theta_TE" not in adapter.state_dict()
    assert torch.equal(adapter.projection_PD, FollowOnThinkingRows.from_model(model).projection_PD)

    _random_weights(adapter, 0)
    adapter.set_text(torch.randn(2, 768))
    x = torch.randn(2, 5, 4, 192)
    out, pos = adapter(x, single_eval_pos=3)
    assert out.shape == (2, 69, 4, 192) and pos == 67
    rows = out[:, :64]
    assert torch.equal(rows, rows[:, :, :1].expand_as(rows))  # identical across columns
    assert not torch.equal(rows[0], rows[1])  # one text per batch member
    assert torch.equal(out[:, 64:], x)
    with pytest.raises(ValueError):
        adapter.set_text(torch.randn(384))  # a MiniLM-sized embedding is rejected


@requires_assets
@requires_bge
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family,seed", SCENES)
def test_init_matches_native(device: str, family: int, seed: int, bge: BgeEncoder) -> None:
    model = model_on(device)
    native_module = model.add_thinking_rows
    adapter = FollowOnThinkingRows.from_model(model)
    scene = Scene(family, seed)
    texts = _texts(scene, np.random.default_rng(seed))
    story = bge.encode_one(texts["story"])
    per_member = bge.encode([texts[v] for v in ("story", "omitted", "wrong")])
    with torch.no_grad():
        for n in SUPPORT_SIZES:
            episode = make_episode(scene, n, BATCH, QUERY_ROWS, device)
            native = binary_logits(model, *episode[:3])
            assert torch.isfinite(native).all()
            for emb in (story, per_member):
                adapted = episode_logits(model, adapter, emb, episode)
                diff = (adapted - native).abs().max().item()
                assert diff <= TOLERANCE, f"n={n} {device}: max |diff| {diff:.2e}"
    assert model.add_thinking_rows is native_module


@requires_assets
@requires_bge
@pytest.mark.parametrize("device", DEVICES)
def test_query_labels_do_not_change_logits(device: str, bge: BgeEncoder) -> None:
    model = model_on(device)
    adapter = FollowOnThinkingRows.from_model(model)
    _random_weights(adapter, 1)
    scene = Scene(*SCENES[1])
    emb = bge.encode_one(scene_text(scene, "story"))
    xs, ys, xq, yq = make_episode(scene, 16, BATCH, QUERY_ROWS, device)

    def logits(*ep):
        with torch.no_grad():
            return episode_logits(model, adapter, emb, ep)

    base = logits(xs, ys, xq, yq)
    gen = torch.Generator().manual_seed(0)
    shuffled = yq[torch.randperm(len(yq), generator=gen).to(yq.device)]
    random = torch.randint(0, 2, yq.shape, generator=gen).to(yq.device)
    for other in (1 - yq, shuffled, random):
        assert not torch.equal(other, yq)
        assert (logits(xs, ys, xq, other) - base).abs().max().item() <= SAME
    assert (logits(xs, 1 - ys, xq, yq) - base).abs().max().item() > CHANGED


@requires_assets
@requires_bge
@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_weights_depend_on_text(device: str, bge: BgeEncoder) -> None:
    model = model_on(device)
    adapter = FollowOnThinkingRows.from_model(model)
    # bge schema+story embeddings are close (dev same-p cosine about 0.99), so text
    # differences need larger weights than 0.05 to move logits by CHANGED.
    _random_weights(adapter, 2, scale=0.5)
    scene = Scene(*SCENES[2])
    texts = _texts(scene, np.random.default_rng(0))
    episode = make_episode(scene, 16, BATCH, QUERY_ROWS, device)
    with torch.no_grad():
        native = binary_logits(model, *episode[:3])
        out = {v: episode_logits(model, adapter, bge.encode_one(t), episode) for v, t in texts.items()}
    assert (out["story"] - native).abs().max().item() > CHANGED
    for v in ("omitted", "wrong", "one_wrong_fact"):
        assert (out["story"] - out[v]).abs().max().item() > CHANGED, v


# --- Report -----------------------------------------------------------------------

_SMALL = training.Settings(steps=500, val_every=250, log_every=250)


def _write_results(out: Path, episodes: dict[str, int] | None = None, arms=ARMS) -> Path:
    """Synthetic complete follow-on training results."""
    out.mkdir(parents=True, exist_ok=True)
    budget = _SMALL.episodes
    episodes = episodes or {arm: budget for arm in arms}
    config = {
        "settings": training._settings_json(_SMALL),
        "arms": list(arms),
        "trainable_parameters": 5_013_504,
        "encoder": {
            "name": followon.ENCODER_NAME,
            "sample_tokens": {"scenes": 1250, "max_story": 159, "max_any": 160},
            "dev_story_cosines": {"scenes": 40, "cos_mean": 0.97, "cos_same_p": 0.99, "residual_norm": 0.15},
        },
    }
    (out / training.CONFIG_JSON).write_text(json.dumps(config))
    with (out / training.PROGRESS_CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(training.PROGRESS_FIELDS)
        for step in (250, 500):
            w.writerows([[step, arm, episodes[arm] * step // 500, 1e-3, 0.5, 1.0] for arm in arms])
    with (out / training.VALIDATION_CSV).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(training.VALIDATION_FIELDS)
        w.writerows([[0, training.NATIVE, n, 64, 0.5] for n in _SMALL.val_sizes])
        for step in training.checkpoint_steps(_SMALL):
            w.writerows([[step, arm, n, 64, 0.5 - step * 1e-5] for arm in arms for n in _SMALL.val_sizes])
    summary = {
        "steps": _SMALL.steps,
        "episodes": episodes,
        "train_wall_s": 10.0,
        "selected": {arm: {"step": 500, "val_logloss_mean": 0.495} for arm in arms},
    }
    (out / training.SUMMARY_JSON).write_text(json.dumps(summary))
    return out


def _run(results: Path, *extra: str) -> int:
    return report.main(["followon-training", "--results", str(results), *extra])


def test_report_passes_on_complete_results(tmp_path: Path, capsys) -> None:
    results = _write_results(tmp_path)
    assert _run(results) == 0
    assert _run(results, "--validation") == 0
    out = capsys.readouterr().out
    assert "5,013,504" in out and "control" in out


def test_report_fails_on_missing_arm_or_unequal_counts(tmp_path: Path) -> None:
    assert _run(_write_results(tmp_path / "one", arms=("story",))) == 1
    budget = _SMALL.episodes
    assert _run(_write_results(tmp_path / "diff", {"story": budget, "control": budget - 8})) == 1
    assert _run(tmp_path / "absent") == 1
    assert _run(tmp_path / "absent", "--validation") == 1
