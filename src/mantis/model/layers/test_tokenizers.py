"""Test the tokenizers."""

import torch

from mantis.model.layers.tokenizer import PatchEmbed


def test_patch_embed_output_shape() -> None:
    """Test the output shape of the PatchEmbed module."""
    patch_size = 16
    batch_size = 4
    embed_dim = 128
    module = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, normalize=False)
    patches = torch.randn(batch_size, 3, patch_size, patch_size)
    out = module(patches)
    assert out.shape == (batch_size, embed_dim)
    module = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, normalize=True)
    out = module(patches)
    assert out.shape == (batch_size, embed_dim)
    assert ((out**2).mean(dim=1) - torch.ones(batch_size)).abs().max() < 1e-3  # noqa: PLR2004
