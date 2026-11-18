"""Frozen all-MiniLM-L6-v2 text encoder with a per-text embedding cache (D-04).

``e(t)`` is the model's standard sentence embedding: mean-pooled token states,
L2-normalised by the model's own pipeline, 384-d. Inputs longer than the
256-token window are rejected instead of silently truncated.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
MAX_TOKENS = 256


@functools.cache
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(ENCODER_NAME)


def token_lengths(texts: Sequence[str]) -> list[int]:
    """MiniLM token counts including [CLS] and [SEP], without truncation."""
    ids = tokenizer()(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    return [len(x) for x in ids]


class TextEncoder:
    """Frozen encoder; embeddings are cached per text on the CPU as float32."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self._model = SentenceTransformer(ENCODER_NAME, device=str(device))
        self._model.eval()
        self._model.requires_grad_(False)
        if self._model.max_seq_length != MAX_TOKENS:
            raise ValueError(f"expected a {MAX_TOKENS}-token window")
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeddings ``(len(texts), 384)`` for texts within the token window."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            too_long = [n for n in token_lengths(missing) if n > MAX_TOKENS]
            if too_long:
                raise ValueError(f"text of {max(too_long)} tokens exceeds {MAX_TOKENS}")
            with torch.no_grad():
                emb = self._model.encode(missing, convert_to_tensor=True)
            for text, e in zip(missing, emb.float().cpu(), strict=True):
                self._cache[text] = e.clone()
        return torch.stack([self._cache[t] for t in texts])

    def encode_one(self, text: str) -> torch.Tensor:
        """Embedding ``(384,)`` of one text."""
        return self.encode([text])[0]
