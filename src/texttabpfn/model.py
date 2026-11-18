"""Raw TabPFN-2.5 path: checkpoint loading, thinking-row modules, binary logits.

No sklearn wrapper and no ensembling: the ``TabPFNV2p5`` module's ``forward`` is
called directly on (rows, batch, features) tensors.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from huggingface_hub import hf_hub_download
from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.interface import PerformanceOptions
from torch import nn

CHECKPOINT_REPO = "Prior-Labs/tabpfn_2_5"
CHECKPOINT_FILE = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
CHECKPOINT_BYTES = 42_935_499

# Gradients through the backbone need unchunked evaluation
# (``shared/chunked_evaluate.py`` asserts ``save_peak_memory_factor is None``).
GRAD_PERFORMANCE = PerformanceOptions(
    save_peak_memory_factor=None,
    force_recompute_layer=False,
    use_chunkwise_inference=False,
)


def default_device() -> torch.device:
    """MPS when available (the only accelerator on the target box), else CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_tabpfn(device: torch.device | str = "cpu") -> tabpfn_v2_5.TabPFNV2p5:
    """Load the frozen TabPFN-2.5 classifier from the local Hugging Face cache."""
    path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config, _ = tabpfn_v2_5.parse_config(checkpoint["config"])
    model = tabpfn_v2_5.get_architecture(config)
    state = {k: v for k, v in checkpoint["state_dict"].items() if "criterion." not in k}
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


class SharedThinkingRows(nn.Module):
    """Trainable copy of ``row_token_values_TE`` (T x E), shared over all columns.

    Same computation as ``tabpfn_v2_5.AddThinkingRows``; at initialisation it is
    numerically identical to the checkpoint module.
    """

    def __init__(self, rows_TE: torch.Tensor) -> None:
        super().__init__()
        self.num_thinking_rows = rows_TE.shape[0]
        self.rows_TE = nn.Parameter(rows_TE.detach().clone())

    def forward(
        self, x_BRiCE: torch.Tensor, single_eval_pos: int
    ) -> tuple[torch.Tensor, int]:
        batch_size, _, num_columns, _ = x_BRiCE.shape
        rows_BTCE = self.rows_TE[None, :, None].expand(batch_size, -1, num_columns, -1)
        x_BRCE = torch.cat([rows_BTCE, x_BRiCE], dim=1)
        return x_BRCE, single_eval_pos + self.num_thinking_rows


class PerColumnThinkingRows(nn.Module):
    """Trainable T x C x E thinking cells, one per column (D-02 diagnostic).

    ``C`` is the number of feature groups plus the target column. Initialised at
    the checkpoint rows expanded over columns, so it starts at the native model.
    """

    def __init__(self, rows_TE: torch.Tensor, num_columns: int) -> None:
        super().__init__()
        self.num_thinking_rows = rows_TE.shape[0]
        init_TCE = rows_TE.detach()[:, None].expand(-1, num_columns, -1).clone()
        self.rows_TCE = nn.Parameter(init_TCE)

    def forward(
        self, x_BRiCE: torch.Tensor, single_eval_pos: int
    ) -> tuple[torch.Tensor, int]:
        batch_size, _, num_columns, _ = x_BRiCE.shape
        if num_columns != self.rows_TCE.shape[1]:
            raise ValueError(
                f"per-column rows built for {self.rows_TCE.shape[1]} columns, "
                f"input has {num_columns}"
            )
        rows_BTCE = self.rows_TCE[None].expand(batch_size, -1, -1, -1)
        x_BRCE = torch.cat([rows_BTCE, x_BRiCE], dim=1)
        return x_BRCE, single_eval_pos + self.num_thinking_rows


def checkpoint_rows(model: tabpfn_v2_5.TabPFNV2p5) -> torch.Tensor:
    """The checkpoint's 64 x 192 thinking-row values."""
    return model.add_thinking_rows.row_token_values_TE


def num_model_columns(num_features: int, features_per_group: int = 3) -> int:
    """Columns the backbone sees: feature groups plus the target column."""
    return -(-num_features // features_per_group) + 1


@contextlib.contextmanager
def thinking_rows(model: tabpfn_v2_5.TabPFNV2p5, module: nn.Module) -> Iterator[None]:
    """Temporarily replace the model's thinking-row module."""
    original = model.add_thinking_rows
    model.add_thinking_rows = module
    try:
        yield
    finally:
        model.add_thinking_rows = original


def binary_logits(
    model: tabpfn_v2_5.TabPFNV2p5,
    x_support_NBF: torch.Tensor,
    y_support_NB: torch.Tensor,
    x_query_MBF: torch.Tensor,
) -> torch.Tensor:
    """Query logits for classes {0, 1}, shape (M, B, 2).

    Inputs follow the module layout: rows first, then batch, then features. The
    first two of the model's output classes are the binary logits.
    """
    x = torch.cat([x_support_NBF, x_query_MBF], dim=0)
    out_MBK = model(
        x,
        y_support_NB.to(x.dtype),
        only_return_standard_out=True,
        performance_options=GRAD_PERFORMANCE,
    )
    return out_MBK[..., :2]
