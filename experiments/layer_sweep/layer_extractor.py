"""Multi-layer frozen-feature extraction for the sMRI MAE encoder.

The standard evaluation model (evaluation.models.smri_mae.SmriMaeBackbone)
pools only the final, post-norm encoder output. This wraps the same encoder
with forward hooks on every transformer block, so a layer-wise linear-probe
sweep costs one encoder pass per image instead of one pass per candidate
layer.

Pooling logic is copied verbatim from SmriMaeBackbone.forward to guarantee
layer == depth-1 differs from the standard model in exactly one respect: it
is the last block's output *before* the final encoder.norm(), since norm is
only meaningful applied once, at the true final layer.
"""

import torch
import torch.nn as nn
from torch import Tensor

import smri_mae.model_mae as models_mae
from smri_mae.modules import unpack_tokens


class SmriMaeMultiLayerBackbone(nn.Module):
    """Wraps a MaskedEncoder; forward() returns pooled features per block.

    Returns dict[int, Tensor[B, D]], keyed by block index in [0, depth-1].
    """

    def __init__(
        self,
        encoder: models_mae.MaskedEncoder,
        global_pool: str = "patch",
        pad_to_multiple: int | None = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.global_pool = global_pool
        self.pad_to_multiple = pad_to_multiple
        self._captured: dict[int, Tensor] = {}
        for index, block in enumerate(encoder.blocks):
            block.register_forward_hook(self._make_hook(index))

    def _make_hook(self, index: int):
        def hook(module, inputs, output):
            self._captured[index] = output

        return hook

    def _pool(self, cls: Tensor | None, reg: Tensor | None, patch: Tensor, token_mask: Tensor) -> Tensor:
        if self.global_pool == "cls":
            return cls[:, 0, :]
        if self.global_pool == "reg":
            return reg.mean(dim=1)
        # "patch" -- same as SmriMaeBackbone.forward
        token_mask = token_mask.to(device=patch.device, dtype=torch.bool)
        denom = token_mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=patch.dtype)
        return (patch * token_mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, batch: dict[str, Tensor]) -> dict[int, Tensor]:
        images = batch["image"]
        mask = batch["mask"]
        self._captured.clear()

        # patch-level token_mask (no cls/reg prefix) -- matches SmriMaeBackbone's
        # pooling input exactly.
        _, _, _, _, _, token_mask = self.encoder(
            images,
            mask=mask,
            pad_to_multiple=self.pad_to_multiple,
        )

        # cat_token_mask is a pure function of (token_mask, batch_size), so
        # calling it again here reproduces -- exactly -- the prefix-inclusive
        # mask the encoder used internally to pack tokens before the block
        # loop. This lets us unpack each hooked block's raw output without
        # duplicating any of the encoder's masking/patchify internals.
        full_mask = self.encoder.cat_token_mask(token_mask, token_mask.shape[0])

        out = {}
        for layer, packed in self._captured.items():
            unpacked = unpack_tokens(packed, full_mask)
            cls, reg, patch = self.encoder.chunk_tokens(unpacked)
            out[layer] = self._pool(cls, reg, patch, token_mask)
        return out
