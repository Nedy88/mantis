"""Stochastic Depth (DropPath) module."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

x = nnx.BatchNorm
x = nnx.Conv
x = nnx.MultiHeadAttention
x = nnx.Linear


class TrainableModule(nnx.Module):
    """Trainable NNX module with train and eval methods."""

    def __init__(self) -> None:
        """Initialize the TrainableModule."""
        self.is_training = True

    def train(self, **attributes: Any) -> None:  # noqa: ANN401
        """Set training mode."""
        return super().train(is_training=True, **attributes)

    def eval(self, **attributes: Any) -> None:  # noqa: ANN401
        """Set evaluation mode."""
        return super().train(is_training=False, **attributes)


class DropPath(TrainableModule):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, batch_size: int, drop_prob: float, *, rngs: nnx.Rngs) -> None:
        """Initialize the DropPath module."""
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.rngs = rngs

    @nnx.split_rngs(splits=self.batch_size)
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply DropPath to the input tensor."""
        if not self.is_training:
            return x
        coin = jax.random.bernoulli(self.rngs(), self.drop_prob)
        if coin:
            return jnp.zeros_like(x)
        return x
