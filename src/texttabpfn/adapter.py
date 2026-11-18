"""Text-conditioned thinking rows: θᵢ + Wᵢ·e(t) (decision D-01).

``TextConditionedThinkingRows`` replaces ``model.add_thinking_rows``. θ is the
checkpoint's 64 x 192 thinking rows, frozen (a non-persistent buffer). Each row
has its own zero-initialised ``Wᵢ ∈ R^{192 x 384}`` (no bias), 64 x 192 x 384 =
4,718,592 trainable parameters. The shifted rows are identical across columns,
like the native module, and at ``W = 0`` the module computes exactly the native
rows. The story arm and the schema-only arm use the same class; only the text
behind ``e(t)`` differs.

Thinking rows are context rows, so a KV cache built by ``TabPFNV2p5.forward``
with the adapter installed already holds the conditioned rows.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from tabpfn.architectures import tabpfn_v2_5
from torch import nn

from texttabpfn.encoder import EMBED_DIM
from texttabpfn.model import binary_logits, checkpoint_rows, thinking_rows


class TextConditionedThinkingRows(nn.Module):
    """Thinking rows ``θ + W·e(t)``, shared over all columns.

    Set the text embedding with ``set_text`` (shape ``(D,)`` for one text, or
    ``(B, D)`` for one text per batch member) or use ``conditioned``.
    """

    def __init__(self, rows_TE: torch.Tensor, text_dim: int = EMBED_DIM) -> None:
        super().__init__()
        num_rows, emsize = rows_TE.shape
        self.num_thinking_rows = num_rows
        self.text_dim = text_dim
        self.register_buffer("theta_TE", rows_TE.detach().clone(), persistent=False)
        self.weight_TED = nn.Parameter(
            torch.zeros(num_rows, emsize, text_dim, dtype=rows_TE.dtype, device=rows_TE.device)
        )
        self._text_bD: torch.Tensor | None = None

    @classmethod
    def from_model(cls, model: tabpfn_v2_5.TabPFNV2p5) -> TextConditionedThinkingRows:
        """A zero-initialised adapter around the checkpoint's thinking rows."""
        return cls(checkpoint_rows(model))

    def set_text(self, embedding: torch.Tensor) -> None:
        e = embedding.detach().to(device=self.theta_TE.device, dtype=self.theta_TE.dtype)
        if e.ndim == 1:
            e = e[None]
        if e.ndim != 2 or e.shape[1] != self.text_dim:
            raise ValueError(f"expected a (D,) or (B, D) embedding with D={self.text_dim}")
        self._text_bD = e

    def clear_text(self) -> None:
        self._text_bD = None

    def rows(self) -> torch.Tensor:
        """Conditioned rows ``(b, T, E)``; ``b`` is 1 or the batch size."""
        if self._text_bD is None:
            raise RuntimeError("no text embedding set; call set_text first")
        shift_bTE = torch.einsum("ted,bd->bte", self.weight_TED, self._text_bD)
        return self.theta_TE[None] + shift_bTE

    def forward(
        self, x_BRiCE: torch.Tensor, single_eval_pos: int
    ) -> tuple[torch.Tensor, int]:
        batch_size, _, num_columns, _ = x_BRiCE.shape
        rows_bTE = self.rows()
        if rows_bTE.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{rows_bTE.shape[0]} text embeddings for a batch of {batch_size}"
            )
        rows_BTCE = rows_bTE[:, :, None].expand(batch_size, -1, num_columns, -1)
        x_BRCE = torch.cat([rows_BTCE, x_BRiCE], dim=1)
        return x_BRCE, single_eval_pos + self.num_thinking_rows


@contextlib.contextmanager
def conditioned(
    model: tabpfn_v2_5.TabPFNV2p5,
    adapter: TextConditionedThinkingRows,
    embedding: torch.Tensor,
) -> Iterator[None]:
    """Install ``adapter`` conditioned on ``embedding``; restore native rows after."""
    adapter.set_text(embedding)
    try:
        with thinking_rows(model, adapter):
            yield
    finally:
        adapter.clear_text()


def episode_logits(
    model: tabpfn_v2_5.TabPFNV2p5,
    adapter: TextConditionedThinkingRows,
    embedding: torch.Tensor,
    episode: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Query logits ``(M, B, 2)`` for an episode ``(xs, ys, xq, yq)``.

    Query labels ``yq`` belong to the loss only; they never enter the forward pass.
    """
    xs, ys, xq, _ = episode
    with conditioned(model, adapter, embedding):
        return binary_logits(model, xs, ys, xq)
