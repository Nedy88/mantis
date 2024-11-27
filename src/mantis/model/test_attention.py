"""Test the Attention module."""

import torch
from torch import nn

from mantis.model.transformer import Attention, AttentionConfig


def test_attention_shape(attention_config: AttentionConfig) -> None:
    """Test the output shape of the Attention module."""
    model = Attention(attention_config)
    embed_dim = attention_config.embed_dim
    batch_size = 4
    N = 10  # noqa: N806
    M = 6  # noqa: N806
    query_inp = torch.randn(batch_size, N, embed_dim)
    key_inp = torch.randn(batch_size, M, embed_dim)
    value_inp = torch.randn(batch_size, M, embed_dim)
    out = model(query_inp, key_inp, value_inp)
    assert out.shape == (batch_size, N, embed_dim)


def test_attention_heads(attention_config: AttentionConfig) -> None:
    """Confirm we are doing multi-head attention correctly."""
    embed_dim = attention_config.embed_dim
    num_heads = attention_config.num_heads
    head_dim = embed_dim // num_heads
    mha = Attention(attention_config)
    head_config = AttentionConfig(
        embed_dim=head_dim,
        num_heads=1,
        qkv_bias=attention_config.qkv_bias,
        proj_bias=attention_config.proj_bias,
        attn_drop=attention_config.attn_drop,
        proj_drop=attention_config.proj_drop,
    )
    sha = Attention(head_config)
    sha.query = nn.Identity()  # type: ignore
    sha.key = nn.Identity()  # type: ignore
    sha.value = nn.Identity()  # type: ignore
    sha.proj = nn.Identity()  # type: ignore
    mha.query = nn.Identity()  # type: ignore
    mha.key = nn.Identity()  # type: ignore
    mha.value = nn.Identity()  # type: ignore
    mha.proj = nn.Identity()  # type: ignore
    batch_size, N, M = 4, 10, 6  # noqa: N806
    query_inp = torch.randn(batch_size, N, embed_dim)
    key_inp = torch.randn(batch_size, M, embed_dim)
    value_inp = torch.randn(batch_size, M, embed_dim)
    out = mha(query_inp, key_inp, value_inp)
    assert out.shape == (batch_size, N, embed_dim)

    for i in range(num_heads):
        outi = sha(
            query_inp[:, :, i * head_dim : (i + 1) * head_dim],
            key_inp[:, :, i * head_dim : (i + 1) * head_dim],
            value_inp[:, :, i * head_dim : (i + 1) * head_dim],
        )
        assert outi.shape == (batch_size, N, head_dim)
        assert outi.allclose(out[:, :, i * head_dim : (i + 1) * head_dim])
