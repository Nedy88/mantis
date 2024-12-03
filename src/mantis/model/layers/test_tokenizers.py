"""Test the tokenizers."""

import torch

from mantis.model.layers.tokenizer import PatchEmbed


def test_patch_embed_output_shape() -> None:
    """Test the output shape of the PatchEmbed module."""
    patch_size = 16
    batch_size = 4
    embed_dim = 128
    M = 8  # noqa: N806
    module = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, normalize=False)
    patches = torch.randn(batch_size, M, 3, patch_size, patch_size)
    out = module(patches)
    assert out.shape == (batch_size, M, embed_dim)


def test_patch_embed_normalization() -> None:
    """Test the shape and normalization of the PatchEmbed module."""
    patch_size = 10
    batch_size = 16
    embed_dim = 64
    M = 10  # noqa: N806
    patches = torch.randn(batch_size, M, 3, patch_size, patch_size)
    module = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, normalize=True)
    tokens = module(patches)
    assert tokens.shape == (batch_size, M, embed_dim)
    tokens_norm = (tokens**2).mean(dim=-1)
    assert (tokens_norm - torch.ones(batch_size, M)).abs().max() < 1e-3  # noqa: PLR2004
