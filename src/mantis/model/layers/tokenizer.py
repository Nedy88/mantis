"""Tokenizer modules for converting patches into transformer ready tokens."""

import torch
from torch import Tensor, nn


class PatchEmbed(nn.Module):
    """Shallow tokenizer."""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        *,
        normalize: bool,
        in_channels: int = 3,
    ) -> None:
        """Init."""
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size)
        self.normalize = normalize
        if normalize:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: A batch of patches (B, C, patch_size, patch_size).

        Returns:
            A batch of embeddings (B, D).

        """
        torch._assert(
            x.ndim == 4,  # noqa: PLR2004
            f"Input tensor must have 4 dimensions, got {x.ndim}",
        )
        torch._assert(
            x.size(-1) == x.size(-2),
            f"PatchEmbed expect input to be square, but got {x.shape}",
        )
        x = self.proj(x).squeeze(-2, -1) # (B, D)
        if self.normalize:
            x = self.norm(x)
        return x
