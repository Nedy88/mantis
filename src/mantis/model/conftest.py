"""Fixtures for testing."""

import numpy as np
import pytest

from mantis.configs.transformer import Activation, TransformerConfig
from mantis.model.transformer import AttentionConfig

rng = np.random.default_rng(seed=42)


@pytest.fixture(params=list(rng.uniform(0.1, 0.9, 20)))
def drop_path_prob(request: pytest.FixtureRequest) -> float:
    """Drop probability for DropPath module testing."""
    return request.param


@pytest.fixture
def attention_config() -> AttentionConfig:
    """Dummy config for attention layer."""  # noqa: D401
    return AttentionConfig(
        embed_dim=512,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0,
        proj_drop=0,
    )


@pytest.fixture
def transformer_config(attention_config: AttentionConfig) -> TransformerConfig:
    """Get dummy config for transformer layer."""
    return TransformerConfig(
        depth=4,
        attention=attention_config,
        mlp_ratio=4.0,
        activation=Activation.gelu,
        drop_path_rate=0.1,
        pre_norm=True,
    )
