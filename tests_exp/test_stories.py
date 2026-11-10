"""Schema/story text contract (D-07) and the corruption builders (D-09)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from texttabpfn.scenes import DEV_SEED_RANGE, EVAL_SEED_RANGE, Scene
from texttabpfn.stories import (
    donor_scene,
    one_wrong_fact_roles,
    scene_roles,
    scene_text,
    schema_text,
    story_text,
    wrong_roles,
)

SCENES = [Scene(f, DEV_SEED_RANGE[0] + 30 + i) for i, f in enumerate((0, 1, 2, 3) * 5)] + [
    Scene(f, EVAL_SEED_RANGE[0] + 30 + i) for i, f in enumerate((4, 5) * 10)
]
FORBIDDEN = re.compile(
    r"\d+\.\d|coefficient|weight|sign|positive|negative|noise|linear|tanh|interaction"
    r"|gaussian|heavy|rate|percent|threshold",
    re.IGNORECASE,
)


def _names(text: str) -> set[str]:
    return set(re.findall(r"\bx\d+\b", text))


@pytest.mark.parametrize("scene", SCENES, ids=lambda s: f"{s.family}-{s.seed}")
def test_schema_and_story_contract(scene: Scene) -> None:
    p = scene.num_features
    assert schema_text(p) == scene_text(scene, "omitted")
    story = story_text(scene_roles(scene))
    assert scene_text(scene, "story") == schema_text(p) + "\n" + story
    # Every feature is named exactly once in the story, and nothing else is.
    mentions = re.findall(r"\bx\d+\b", story)
    assert sorted(mentions) == sorted(f"x{i + 1}" for i in range(p))
    assert not FORBIDDEN.search(story), story
    assert _names(schema_text(p)) == {f"x{i + 1}" for i in range(p)}


def test_schema_depends_only_on_p() -> None:
    by_p: dict[int, set[str]] = {}
    for scene in SCENES:
        by_p.setdefault(scene.num_features, set()).add(scene_text(scene, "omitted"))
    assert all(len(texts) == 1 for texts in by_p.values())


@pytest.mark.parametrize("scene", SCENES[:8], ids=lambda s: f"{s.family}-{s.seed}")
def test_corruptions(scene: Scene) -> None:
    roles = scene_roles(scene)
    story = story_text(roles)
    rng = np.random.default_rng(0)

    wrong = wrong_roles(roles, rng)
    assert sorted(map(str, wrong)) == sorted(map(str, roles))  # a permutation
    assert story_text(wrong) != story

    for _ in range(20):
        one = one_wrong_fact_roles(roles, rng)
        changed = [i for i, (a, b) in enumerate(zip(roles, one, strict=True)) if a != b]
        assert len(changed) == 1
        assert one[changed[0]]["role"] != roles[changed[0]]["role"]

    donor = donor_scene(scene, rng)
    assert donor.family == scene.family and donor.seed != scene.seed
    assert story_text(scene_roles(donor)) != story
    irrelevant = scene_text(scene, "irrelevant", np.random.default_rng(1))
    assert irrelevant.startswith(schema_text(scene.num_features) + "\n")
    assert irrelevant != scene_text(scene, "story")
