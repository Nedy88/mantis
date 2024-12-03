"""Fixtures for testing."""

from pathlib import Path
import numpy as np
import pytest
from torch.nn.modules import transformer

from mantis.configs.config import Config
from mantis.configs.data import DatasetConfig, DatasetName
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
        n_states=10,
    )


@pytest.fixture
def caltech_dataset_config() -> DatasetConfig:
    """Caltech 256 dataset config."""
    return DatasetConfig(
        name=DatasetName.caltech256,
        hf_dataset="ilee0022/caltech256",
        root=Path("/scratch/nedyalko_prisadnikov/hf-ds"),
        img_width=400,
        img_height=300,
        num_classes=257,
    )


@pytest.fixture
def config(transformer_config: TransformerConfig, caltech_dataset_config: DatasetConfig) -> Config:
    """Get dummy config."""
    return Config(
        transformer=transformer_config,
        dataset=caltech_dataset_config,
        patch_size=16,
    )
