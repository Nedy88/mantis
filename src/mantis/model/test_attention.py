"""Test the Attention module."""

import torch
from torch import nn

from mantis.model.transformer import Attention, AttentionConfig


def test_attention_shape() -> None:
    """Test the output shape of the Attention module."""
    config = AttentionConfig(
        embed_dim=512,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0,
        proj_drop=0,
    )
    model = Attention(config)
    batch_size = 4
    N = 10  # noqa: N806
    M = 6  # noqa: N806
    query_inp = torch.randn(batch_size, N, config.embed_dim)
    key_inp = torch.randn(batch_size, M, config.embed_dim)
    value_inp = torch.randn(batch_size, M, config.embed_dim)
    out = model(query_inp, key_inp, value_inp)
    assert out.shape == (batch_size, N, config.embed_dim)


def test_attention_heads() -> None:
    """Confirm we are doing multi-head attention correctly."""
    config = AttentionConfig(
        embed_dim=512,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0,
        proj_drop=0,
    )
    mha = Attention(config)
    head_config = AttentionConfig(
        embed_dim=64,
        num_heads=1,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0,
        proj_drop=0,
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
    query_inp = torch.randn(batch_size, N, config.embed_dim)
    key_inp = torch.randn(batch_size, M, config.embed_dim)
    value_inp = torch.randn(batch_size, M, config.embed_dim)
    out = mha(query_inp, key_inp, value_inp)
    assert out.shape == (batch_size, N, config.embed_dim)
    out1 = sha(query_inp[:, :, :64], key_inp[:, :, :64], value_inp[:, :, :64])
    assert out1.shape == (batch_size, N, 64)
    assert out1.allclose(out[:, :, :64])
