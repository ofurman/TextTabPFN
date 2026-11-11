"""Stage 3 episode contract: shared, deterministic episodes and the D-09 story mix."""

from __future__ import annotations

import collections
import math

import numpy as np

from texttabpfn.scenes import TRAIN_FAMILIES, TRAIN_SEED_RANGE, Scene
from texttabpfn.stories import scene_text
from texttabpfn.training import (
    Settings,
    checkpoint_steps,
    learning_rate,
    make_episode,
    story_variant,
)

SETTINGS = Settings()


def test_episode_is_deterministic_and_well_formed() -> None:
    for index in range(40):
        a, b = make_episode(SETTINGS, index), make_episode(SETTINGS, index)
        assert (a.family, a.seed, a.n, a.variant, a.texts) == (b.family, b.seed, b.n, b.variant, b.texts)
        for x, y in zip(a.arrays, b.arrays, strict=True):
            np.testing.assert_array_equal(x, y)
        xs, ys, xq, yq = a.arrays
        assert a.family in TRAIN_FAMILIES
        assert TRAIN_SEED_RANGE[0] <= a.seed < TRAIN_SEED_RANGE[1]
        assert SETTINGS.n_min <= a.n <= SETTINGS.n_max
        assert xs.shape[:2] == (a.n, 1) and ys.shape == (a.n, 1)
        assert xq.shape[:2] == (SETTINGS.query_rows, 1) and yq.shape == (SETTINGS.query_rows, 1)
        assert set(np.unique(ys)) == {0, 1}, "stratified support holds both classes"
        scene = Scene(a.family, a.seed)
        assert a.texts["schema"] == scene_text(scene, "omitted")
        if a.variant in ("story", "omitted"):
            assert a.texts["story"] == scene_text(scene, a.variant)


def test_story_mix_is_exact_per_block() -> None:
    for block in range(10):
        counts = collections.Counter(story_variant(SETTINGS, block * 20 + j) for j in range(20))
        assert counts == {"story": 15, "omitted": 3, "one_wrong_fact": 2}


def test_support_sizes_cover_the_log_range() -> None:
    ns = np.array([make_episode(SETTINGS, i).n for i in range(200)])
    logs = np.log(ns)
    assert ns.min() < 16 and ns.max() > 512
    mid = 0.5 * (math.log(SETTINGS.n_min) + math.log(SETTINGS.n_max))
    assert 0.35 < (logs < mid).mean() < 0.65


def test_schedule_and_checkpoints() -> None:
    s = SETTINGS
    assert learning_rate(s, s.warmup_steps) == s.lr
    assert math.isclose(learning_rate(s, s.steps), s.lr * s.final_lr_ratio)
    steps = checkpoint_steps(s)
    assert steps[0] == 0 and steps[-1] == s.steps
    assert all(b - a == s.val_every for a, b in zip(steps[:-2], steps[1:-1], strict=True))
