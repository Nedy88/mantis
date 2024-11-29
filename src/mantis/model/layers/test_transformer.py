"""Test the DETR-style transformer decoder."""

import torch

from mantis.configs.transformer import TransformerConfig
from mantis.model.transformer import Transformer


def test_transformer_shape(transformer_config: TransformerConfig) -> None:
    """Test the output shape of the transformer."""
    model = Transformer(transformer_config)
    batch_size = 4
    N = 10  # noqa: N806
    M = 6  # noqa: N806
    state = torch.randn(batch_size, N, transformer_config.attention.embed_dim)
    patches = torch.randn(batch_size, M, transformer_config.attention.embed_dim)
    patches_pos_embed = torch.randn(batch_size, M, transformer_config.attention.embed_dim)
    state_query = torch.randn(batch_size, N, transformer_config.attention.embed_dim)
    out = model(state, patches, patches_pos_embed, state_query)
    assert out.shape == (batch_size, N, transformer_config.attention.embed_dim)
