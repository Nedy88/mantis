"""The modules necessary for the transformer model."""

import numpy as np
import torch
from torch import Tensor, nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from mantis.configs.transformer import (
    Activation,
    AttentionConfig,
    TransformerConfig,
    get_activation_module,
)
from mantis.model.layers.drop_path import DropPath
from mantis.model.layers.layer_scale import LayerScale


class Transformer(nn.Module):
    """DETR-style transformer decoder."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the transformer."""
        super().__init__()
        # stochastic depth decay rule
        dpr = np.linspace(0, config.drop_path_rate, config.depth)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    attention_config=config.attention,
                    mlp_ratio=config.mlp_ratio,
                    activation=config.activation,
                    drop_path=dpr[i],
                    pre_norm=config.pre_norm,
                )
                for i in range(config.depth)
            ],
        )
        self.norm = nn.LayerNorm(config.attention.embed_dim)

    def forward(
        self,
        state: Tensor,
        patches: Tensor,
        patches_pos_embed: Tensor,
        state_query: Tensor,
    ) -> Tensor:
        """Forward pass of the transformer."""
        out = state
        for layer in self.layers:
            out = layer(out, patches, patches_pos_embed, state_query)
        return self.norm(out)


class DummyTransformerLayer(nn.Module):
    def __init__(self, attention_config: AttentionConfig, mlp_ratio: float, activation: Activation):
        super().__init__()
        # self.cross_attn = DummyAttention(attention_config)

    def forward(
        self, state: Tensor, patches: Tensor, patches_pos_embed: Tensor, query: Tensor
    ) -> Tensor:
        print(f"Rank[{xr.global_ordinal()}]: {state.shape=}, {state.dtype=}, {state.device=}")
        print(f"Rank[{xr.global_ordinal()}]: {query.shape=}, {query.dtype=}, {query.device=}")
        p = patches.sum(dim=1, keepdim=True).expand(-1, state.shape[1], -1)
        print(f"Rank[{xr.global_ordinal()}]: {p.shape=}, {p.dtype=}, {p.device=}")
        pos = patches_pos_embed.sum(dim=1, keepdim=True).expand(-1, state.shape[1], -1)
        print(f"Rank[{xr.global_ordinal()}]: {pos.shape=}, {pos.dtype=}, {pos.device=}")
        return (
            state
            + query
            + p
            + pos
        )
        # state = state + query
        # new_state = self.cross_attn(
        #     query=state,
        #     key=patches + patches_pos_embed,
        #     value=patches,
        # )
        # return state + new_state


class TransformerLayer(nn.Module):
    """A single transformer decoder layer inspired by DETR."""

    def __init__(
        self,
        attention_config: AttentionConfig,
        mlp_ratio: float,
        activation: Activation,
        drop_path: float,
        *,
        pre_norm: bool,
    ) -> None:
        """Init."""
        super().__init__()
        self.self_attn = Attention(attention_config)
        self.cross_attn = Attention(attention_config)
        ffn_mlp_dim = int(attention_config.embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(attention_config.embed_dim, ffn_mlp_dim),
            get_activation_module(activation),
            nn.Linear(ffn_mlp_dim, attention_config.embed_dim),
        )
        self.norm1 = nn.LayerNorm(attention_config.embed_dim)
        self.norm2 = nn.LayerNorm(attention_config.embed_dim)
        self.norm3 = nn.LayerNorm(attention_config.embed_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.pre_norm = pre_norm
        if self.pre_norm:
            self.ls1 = LayerScale(dim=attention_config.embed_dim, init_val=1e-5, inplace=True)
            self.ls2 = LayerScale(dim=attention_config.embed_dim, init_val=1e-5, inplace=True)
            self.ls3 = LayerScale(dim=attention_config.embed_dim, init_val=1e-5, inplace=True)

    def forward(
        self,
        state: Tensor,
        patches: Tensor,
        patches_pos_embed: Tensor,
        state_query: Tensor,
    ) -> Tensor:
        """Forward pass of the transformer layer."""
        if self.pre_norm:
            return self.forward_pre(state, patches, patches_pos_embed, state_query)
        return self.forward_post(state, patches, patches_pos_embed, state_query)

    def forward_pre(
        self,
        state: Tensor,
        patches: Tensor,
        patches_pos_embed: Tensor,
        state_query: Tensor,
    ) -> Tensor:
        """Forward pass of the transformer layer with pre-norm."""
        # state is (B, N, D)
        # patches is (B, M, D)
        # patches_pos_embed is (B, M, D)
        # state_query is (B, N, D) if present
        state = state + state_query

        # First apply cross-attention to the patches
        cross_attn_out = self.ls1(
            self.cross_attn(
                query=self.norm1(state),
                key=patches + patches_pos_embed,
                value=patches,
            ),
        )
        state = state + self.drop_path1(cross_attn_out)

        # Then self-attention between the state vectors
        norm_state = self.norm2(state)
        self_attn_out = self.ls2(
            self.self_attn(
                query=norm_state,
                key=norm_state,
                value=norm_state,
            ),
        )
        state = state + self.drop_path2(self_attn_out)

        # Finally, apply the MLP
        ffn_out = self.ls3(self.mlp(self.norm3(state)))
        return state + self.drop_path3(ffn_out)

    def forward_post(
        self,
        state: Tensor,
        patches: Tensor,
        patches_pos_embed: Tensor,
        state_query: Tensor,
    ) -> Tensor:
        """Forward pass of the transformer layer with post-norm."""
        # state is (B, N, D)
        # patches is (B, M, D)
        # patches_pos_embed is (B, M, D)
        # state_query is (B, N, D) if present
        state = state + state_query

        # First apply cross-attention to the patches
        cross_attn_out = self.cross_attn(
            query=state,
            key=patches + patches_pos_embed,
            value=patches,
        )
        state = state + self.norm1(self.drop_path1(cross_attn_out))

        # Then self-attention between the state vectors
        self_attn_out = self.self_attn(
            query=state,
            key=state,
            value=state,
        )
        state = state + self.norm2(self.drop_path2(self_attn_out))

        # Finally, apply the MLP
        ffn_out = self.mlp(state)
        return state + self.norm3(self.drop_path3(ffn_out))


class DummyAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return query


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
