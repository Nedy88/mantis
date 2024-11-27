"""Test the transformer layer."""

import torch

from mantis.model.transformer import Activation, AttentionConfig, TransformerLayer


def test_transformer_layer_shape(attention_config: AttentionConfig) -> None:
    """Test the shape of the output of the transformer layer."""
    layer = TransformerLayer(
        attention_config,
        mlp_ratio=4.0,
        activation=Activation.gelu,
        drop_path=0.1,
        pre_norm=True,
    )
    batch_size = 4
    N = 10  # noqa: N806
    M = 6  # noqa: N806
    state = torch.randn(batch_size, N, attention_config.embed_dim)
    patches = torch.randn(batch_size, M, attention_config.embed_dim)
    patches_pos_embed = torch.randn(batch_size, M, attention_config.embed_dim)
    state_query = torch.randn(batch_size, N, attention_config.embed_dim)
    out = layer(state, patches, patches_pos_embed, state_query)
    assert out.shape == (batch_size, N, attention_config.embed_dim)
    out = layer(state, patches, patches_pos_embed, None)
    assert out.shape == (batch_size, N, attention_config.embed_dim)
