"""Schema and story text per scene, plus the corrupted stories (D-07, D-09).

- Schema: the neutral column names ``x1 ... xp`` in column order and the
  definition of the binary outcome ``y``. No role hints; depends only on ``p``.
- Story: templated structural facts from ``Scene.describe()["roles"]``: which
  features influence ``y`` or are influenced by it (the Markov blanket), which
  groups are noisy measurements of one hidden quantity, which features are
  unrelated. No coefficients, signs, equations, noise levels, mechanism names
  or base rates.

The encoder input is ``schema + "\\n" + story``; the schema-only arm and the
``omitted`` corruption use the schema alone.

Corruptions (built here so later stages only call them):

- ``omitted``: schema only;
- ``irrelevant``: another same-family scene's story, with its own feature names;
- ``wrong``: the scene's feature roles permuted among its features;
- ``one_wrong_fact``: exactly one feature's role changed (training, D-09).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from texttabpfn.scenes import (
    DEV_SEED_RANGE,
    EVAL_SEED_RANGE,
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    TRAIN_SEED_RANGE,
    Scene,
)

# Encoder inputs per variant. "story" is the correct story; the rest are D-09
# corruptions. The schema-only arm's text equals the "omitted" variant.
VARIANTS = ("story", "omitted", "irrelevant", "wrong", "one_wrong_fact")

_SEED_RANGES = (TRAIN_SEED_RANGE, DEV_SEED_RANGE, EVAL_SEED_RANGE)
_ORDINALS = ("one hidden quantity", "a second hidden quantity", "a third hidden quantity")
_COARSE_ROLES = ("mb", "irrelevant", "measure")


def feature_names(num_features: int) -> list[str]:
    return [f"x{i + 1}" for i in range(num_features)]


def _join(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def schema_text(num_features: int) -> str:
    """Column names in order and the outcome definition; depends only on ``p``."""
    names = _join(feature_names(num_features))
    return (
        f"Columns: {names} are {num_features} numeric features, listed in column "
        "order. Each row is one record. The target y is a binary outcome with "
        "value 0 or 1."
    )


def story_text(roles: Sequence[dict]) -> str:
    """Structural facts for roles in column order (``describe()["roles"]`` format)."""
    names = feature_names(len(roles))
    parents, children, irrelevant = [], [], []
    groups: dict[int, list[str]] = {}
    for name, role in zip(names, roles, strict=True):
        kind = role["role"]
        if kind == "mb":
            (parents if role["relation"] == "parent" else children).append(name)
        elif kind == "measure":
            groups.setdefault(role["group"], []).append(name)
        elif kind == "irrelevant":
            irrelevant.append(name)
        else:
            raise ValueError(f"unknown role {kind!r}")

    def verb(items: list[str], one: str, many: str) -> str:
        return one if len(items) == 1 else many

    facts = []
    if parents:
        facts.append(f"{_join(parents)} {verb(parents, 'directly influences', 'directly influence')} y.")
    if children:
        facts.append(f"{_join(children)} {verb(children, 'is', 'are')} directly influenced by y.")
    # Groups in order of their first column, so group ids carry no information.
    for k, members in enumerate(sorted(groups.values(), key=lambda m: names.index(m[0]))):
        what = verb(members, "is a noisy measurement", "are noisy measurements")
        facts.append(f"{_join(members)} {what} of {_ORDINALS[k]} that influences y.")
    if irrelevant:
        facts.append(f"{_join(irrelevant)} {verb(irrelevant, 'is', 'are')} unrelated to y.")
    return "Known structure: " + " ".join(facts)


def full_text(num_features: int, story: str) -> str:
    return f"{schema_text(num_features)}\n{story}"


def scene_roles(scene: Scene) -> list[dict]:
    return scene.describe()["roles"]


# --- Corruptions ----------------------------------------------------------------


def _seed_range(seed: int) -> tuple[int, int]:
    for lo, hi in _SEED_RANGES:
        if lo <= seed < hi:
            return lo, hi
    raise ValueError(f"seed {seed} is outside every scene seed range")


def donor_scene(scene: Scene, rng: np.random.Generator) -> Scene:
    """Another scene of the same family and seed range with a different story."""
    lo, hi = _seed_range(scene.seed)
    own = story_text(scene_roles(scene))
    while True:
        seed = int(rng.integers(lo, hi))
        if seed == scene.seed:
            continue
        donor = Scene(scene.family, seed)
        if story_text(scene_roles(donor)) != own:
            return donor


def wrong_roles(roles: Sequence[dict], rng: np.random.Generator) -> list[dict]:
    """Roles permuted among the scene's features; resampled until the story changes."""
    own = story_text(roles)
    while True:
        permuted = [dict(roles[i]) for i in rng.permutation(len(roles))]
        if story_text(permuted) != own:
            return permuted


def one_wrong_fact_roles(roles: Sequence[dict], rng: np.random.Generator) -> list[dict]:
    """Exactly one feature moved to a different role class (mb/irrelevant/measure)."""
    out = [dict(r) for r in roles]
    i = int(rng.integers(len(out)))
    new_kind = str(rng.choice([k for k in _COARSE_ROLES if k != out[i]["role"]]))
    if new_kind == "mb":
        out[i] = {"role": "mb", "group": None, "relation": str(rng.choice(["parent", "child"]))}
    elif new_kind == "measure":
        groups = sorted({r["group"] for r in roles if r["role"] == "measure"}) or [0]
        out[i] = {"role": "measure", "group": int(rng.choice(groups)), "relation": None}
    else:
        out[i] = {"role": "irrelevant", "group": None, "relation": None}
    return out


def scene_text(scene: Scene, variant: str, rng: np.random.Generator | None = None) -> str:
    """Encoder input for ``scene`` under ``variant`` (see ``VARIANTS``).

    ``irrelevant``, ``wrong`` and ``one_wrong_fact`` draw from ``rng``.
    """
    p = scene.num_features
    roles = scene_roles(scene)
    if variant == "story":
        return full_text(p, story_text(roles))
    if variant == "omitted":
        return schema_text(p)
    if rng is None:
        raise ValueError(f"variant {variant!r} needs an rng")
    if variant == "irrelevant":
        return full_text(p, story_text(scene_roles(donor_scene(scene, rng))))
    if variant == "wrong":
        return full_text(p, story_text(wrong_roles(roles, rng)))
    if variant == "one_wrong_fact":
        return full_text(p, story_text(one_wrong_fact_roles(roles, rng)))
    raise ValueError(f"unknown text variant {variant!r}")


# --- Fixed scene sample for the token-length check ------------------------------

SAMPLE_PER_CELL = 125
SAMPLE_SEED = 0


def story_sample_specs(per_cell: int = SAMPLE_PER_CELL) -> list[tuple[str, int, int]]:
    """``(split, family, seed)``: a fixed random sample per family and seed range.

    Training families 0-3 from the train and dev ranges, held-out families 4-5
    from the evaluation range.
    """
    cells = [("train", f, TRAIN_SEED_RANGE) for f in TRAIN_FAMILIES]
    cells += [("dev", f, DEV_SEED_RANGE) for f in TRAIN_FAMILIES]
    cells += [("eval", f, EVAL_SEED_RANGE) for f in HELDOUT_FAMILIES]
    rng = np.random.default_rng(SAMPLE_SEED)
    specs = []
    for split, family, (lo, hi) in cells:
        seeds = lo + rng.choice(hi - lo, size=per_cell, replace=False)
        specs += [(split, family, int(s)) for s in np.sort(seeds)]
    return specs
