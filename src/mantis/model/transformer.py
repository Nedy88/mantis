"""The modules necessary for the transformer model."""

import pydantic
import torch
from torch import Tensor, nn

EPS = 1e-6


class AttentionConfig(pydantic.BaseModel):
    """Config for the MHA layer."""

    embed_dim: int
    num_heads: int
    qkv_bias: bool
    proj_bias: bool
    attn_drop: float
    proj_drop: float


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config: AttentionConfig) -> None:
        """Initialize the Attention module."""
        super().__init__()
        torch._assert(
            config.embed_dim % config.num_heads == 0,
            "embed_dim must be divisible by num_heads",
        )
        self.num_heads = config.num_heads
        head_dim = config.embed_dim // self.num_heads
        self.scale = head_dim**-0.5
        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.proj_bias)
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.proj_drop = nn.Dropout(config.proj_drop)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Forward pass of the Attention module."""
        # query is (B, N, D)
        # key and value are (B, M, D)
        B, N, D = query.shape  # noqa: N806
        M = key.shape[1]  # noqa: N806
        torch._assert(
            key.shape == value.shape,
            f"The shapes for key ({key.shape}) and value ({value.shape}) must match",
        )
        q = self.query(query).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) * self.scale
        k = self.key(key).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.value(value).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        # q is (B, num_heads, N, head_dim)
        # k, v are (B, num_heads, M, head_dim)
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k)  # (B, num_heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhnm,bhmd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        out = self.proj(out)
        return self.proj_drop(out)  # (B, N, D)


class DropPath(nn.Module):
    """Drop path (Stochastic Depth) per sample applied in the main path of the residual blocks."""

    def __init__(self, drop_prob: float) -> None:
        """Initialize the DropPath module."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Drop Path module."""
        if self.drop_prob < EPS or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0:
            random_tensor.div_(keep_prob)
        return x * random_tensor
