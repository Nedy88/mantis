"""Configs for the transformers."""
from enum import Enum

import pydantic
from torch import nn


class AttentionConfig(pydantic.BaseModel):
    """Config for the MHA layer."""

    embed_dim: int
    num_heads: int
    qkv_bias: bool
    proj_bias: bool
    attn_drop: float
    proj_drop: float


class Activation(str, Enum):
    """Configurable activations."""

    relu = "relu"
    gelu = "gelu"


def get_activation_module(activation: Activation) -> nn.Module:
    """Get the nn.Module for the activation."""
    match activation:
        case Activation.relu:
            return nn.ReLU()
        case Activation.gelu:
            return nn.GELU()
        case _:
            msg = f"Activation {activation} is not implemented."
            raise NotImplementedError(msg)
