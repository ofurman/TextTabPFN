"""Per-feature role tokens through the shared thinking-row channel (decision D-11).

Feature ``x_j`` gets one token: the frozen MiniLM embedding of a sentence that
states only ``x_j``'s role from the story (``Scene.describe()["roles"]``). The
sentence never names the feature, so every feature with the same role gets the
same token; the slot it sits in (column order, at most ``MAX_SLOTS``) carries
the position. Measurement groups are lettered A, B, C in order of their first
column, so group ids carry no information but groups within a scene differ.
Empty slots are zero.

Matched control: slot ``j`` holds the token of one neutral sentence for every
existing feature, so the two arms differ only in role content. The "omitted"
role corruption is the control's tokens.

``RoleTokenThinkingRows`` computes ``θᵢ + Wᵢ·s`` with ``s = concat_j(P t_j)``:
one projection ``P: 384 -> SLOT_DIM`` (no bias, so empty slots stay zero)
shared by all slots, then concatenation over the 15 slots. ``W`` (64 x 192 x
15·SLOT_DIM) starts at zero, so at initialisation the module computes exactly
the native rows; ``P`` starts from a fixed seeded random draw so both arms start
from the same weights. The rows are identical across columns, like iteration 1.
Trainable parameters: 64·192·375 + 384·25 = 4,617,600 (iteration 1: 4,718,592).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tabpfn.architectures import tabpfn_v2_5
from torch import nn

from texttabpfn.adapter import TextConditionedThinkingRows
from texttabpfn.encoder import EMBED_DIM, TextEncoder
from texttabpfn.model import checkpoint_rows
from texttabpfn.scenes import Scene
from texttabpfn.stories import one_wrong_fact_roles, scene_roles, wrong_roles

MAX_SLOTS = 15
SLOT_DIM = 25
PROJECTION_SEED = 0

NEUTRAL_SENTENCE = "This is a numeric feature."
_GROUP_LETTERS = "ABC"
# Role variants with token builders here. "story" = correct roles, "omitted" =
# all slots neutral (the control's input), the rest are D-09 corruptions.
ROLE_VARIANTS = ("story", "omitted", "wrong", "one_wrong_fact")


def role_sentences(roles: Sequence[dict]) -> list[str]:
    """One sentence per feature (column order) stating only that feature's role."""
    first_column: dict[int, int] = {}
    for j, role in enumerate(roles):
        if role["role"] == "measure":
            first_column.setdefault(role["group"], j)
    letters = {g: _GROUP_LETTERS[k] for k, g in enumerate(sorted(first_column, key=first_column.get))}
    out = []
    for role in roles:
        kind = role["role"]
        if kind == "mb":
            if role["relation"] == "parent":
                out.append("This feature directly influences y.")
            else:
                out.append("This feature is directly influenced by y.")
        elif kind == "measure":
            out.append(
                f"This feature is a noisy measurement of hidden quantity {letters[role['group']]}, "
                "which influences y."
            )
        elif kind == "irrelevant":
            out.append("This feature is unrelated to y.")
        else:
            raise ValueError(f"unknown role {kind!r}")
    return out


def neutral_sentences(num_features: int) -> list[str]:
    return [NEUTRAL_SENTENCE] * num_features


def variant_sentences(
    scene: Scene, variant: str, rng: np.random.Generator | None = None
) -> list[str]:
    """Per-feature sentences for ``scene`` under a role variant (see ``ROLE_VARIANTS``).

    ``wrong`` and ``one_wrong_fact`` draw from ``rng``.
    """
    roles = scene_roles(scene)
    if variant == "story":
        return role_sentences(roles)
    if variant == "omitted":
        return neutral_sentences(len(roles))
    if rng is None:
        raise ValueError(f"variant {variant!r} needs an rng")
    if variant == "wrong":
        return role_sentences(wrong_roles(roles, rng))
    if variant == "one_wrong_fact":
        return role_sentences(one_wrong_fact_roles(roles, rng))
    raise ValueError(f"unknown role variant {variant!r}")


def slot_tokens(encoder: TextEncoder, sentences: Sequence[str]) -> torch.Tensor:
    """Tokens ``(MAX_SLOTS, 384)``: sentence j's embedding in slot j, empty slots zero."""
    if not 0 < len(sentences) <= MAX_SLOTS:
        raise ValueError(f"{len(sentences)} features; expected 1 ... {MAX_SLOTS}")
    tokens = torch.zeros(MAX_SLOTS, EMBED_DIM)
    tokens[: len(sentences)] = encoder.encode(list(sentences))
    return tokens


class RoleTokenThinkingRows(TextConditionedThinkingRows):
    """Thinking rows ``θ + W·concat_j(P t_j)``, shared over all columns.

    ``set_text`` takes slot tokens ``(S, D)`` for one assignment or ``(B, S, D)``
    for one per batch member, so ``adapter.conditioned`` and
    ``adapter.episode_logits`` work unchanged.
    """

    def __init__(
        self,
        rows_TE: torch.Tensor,
        token_dim: int = EMBED_DIM,
        slot_dim: int = SLOT_DIM,
        max_slots: int = MAX_SLOTS,
    ) -> None:
        nn.Module.__init__(self)
        num_rows, emsize = rows_TE.shape
        self.num_thinking_rows = num_rows
        self.text_dim = token_dim
        self.max_slots = max_slots
        self.register_buffer("theta_TE", rows_TE.detach().clone(), persistent=False)
        projection = torch.empty(slot_dim, token_dim, dtype=rows_TE.dtype)
        nn.init.kaiming_uniform_(
            projection, a=5**0.5, generator=torch.Generator().manual_seed(PROJECTION_SEED)
        )  # nn.Linear's default init, from a fixed seed
        self.projection_PD = nn.Parameter(projection.to(rows_TE.device))
        self.weight_TES = nn.Parameter(
            torch.zeros(num_rows, emsize, max_slots * slot_dim, dtype=rows_TE.dtype, device=rows_TE.device)
        )
        self._text_bD: torch.Tensor | None = None  # (b, S, D) slot tokens

    @classmethod
    def from_model(cls, model: tabpfn_v2_5.TabPFNV2p5) -> RoleTokenThinkingRows:
        """A zero-initialised (W = 0) role-token adapter around the checkpoint's rows."""
        return cls(checkpoint_rows(model))

    def set_text(self, embedding: torch.Tensor) -> None:
        t = embedding.detach().to(device=self.theta_TE.device, dtype=self.theta_TE.dtype)
        if t.ndim == 2:
            t = t[None]
        if t.ndim != 3 or t.shape[1:] != (self.max_slots, self.text_dim):
            raise ValueError(f"expected (S, D) or (B, S, D) slot tokens with S={self.max_slots}, D={self.text_dim}")
        self._text_bD = t

    def summary(self) -> torch.Tensor:
        """``s = concat_j(P t_j)``, shape ``(b, S·d)``."""
        if self._text_bD is None:
            raise RuntimeError("no slot tokens set; call set_text first")
        return torch.einsum("pd,bsd->bsp", self.projection_PD, self._text_bD).flatten(1)

    def rows(self) -> torch.Tensor:
        """Conditioned rows ``(b, T, E)``; ``b`` is 1 or the batch size."""
        shift_bTE = torch.einsum("tes,bs->bte", self.weight_TES, self.summary())
        return self.theta_TE[None] + shift_bTE
