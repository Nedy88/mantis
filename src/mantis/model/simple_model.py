"""Simple model without RL and an agent."""

from typing import Any

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch import Tensor, nn

from mantis.configs.config import Config
from mantis.configs.transformer import get_activation_module
from mantis.model.layers.simple_head import SimpleHead
from mantis.model.layers.tokenizer import PatchEmbed
from mantis.model.transformer import DummyTransformerLayer, Transformer, TransformerLayer


class DummyModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embed_dim = config.transformer.attention.embed_dim
        self.n_states = config.transformer.n_states
        self.tokenizer = PatchEmbed(
            patch_size=config.patch_size, embed_dim=self.embed_dim, normalize=True
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 512),
            get_activation_module(config.transformer.activation),
            nn.Linear(512, self.embed_dim),
        )
        self.prompt_embed = nn.Embedding(self.n_states, self.embed_dim)
        assert config.dataset.num_classes is not None
        self.task_head = SimpleHead(
            n_states=1,
            embed_dim=self.embed_dim,
            activation=config.transformer.activation,
            output_dim=config.dataset.num_classes,
        )
        # self.transformer = DummyTransformerLayer(
        #     attention_config=config.transformer.attention,
        #     mlp_ratio=config.transformer.mlp_ratio,
        #     activation=config.transformer.activation,
        #     # drop_path=0.,
        #     # pre_norm=config.transformer.pre_norm,
        # )

    def forward(self, patches: Tensor, locs: Tensor) -> Tensor:
        batch_size = patches.shape[0]
        state = torch.zeros(batch_size, self.n_states, self.embed_dim).to(patches.device)
        patches = self.tokenizer(patches)  # (B, M, D)
        pos_embeds = self.pos_embed(locs)  # (B, M, D)
        patches = (patches + pos_embeds).sum(dim=1, keepdim=True)  # (B, 1, D)
        state_query = self.prompt_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # new_state = self.transformer(state, patches, pos_embeds, state_query)
        new_state = (state + state_query).sum(dim=1, keepdim=True)  # (B, 1, D)
        return self.task_head(new_state + patches)  # (B, num_classes)


class SimpleModel(nn.Module):
    """Simple model with single iteration and no RL."""

    def __init__(self, config: Config) -> None:
        """Init."""
        super().__init__()
        self.transformer = Transformer(config.transformer)
        self.n_states = config.transformer.n_states
        self.embed_dim = config.transformer.attention.embed_dim
        assert config.dataset.num_classes is not None
        self.task_head = SimpleHead(
            n_states=self.n_states,
            embed_dim=self.embed_dim,
            activation=config.transformer.activation,
            output_dim=config.dataset.num_classes,
        )
        self.tokenizer = PatchEmbed(
            patch_size=config.patch_size,
            embed_dim=self.embed_dim,
            normalize=True,
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 512),
            get_activation_module(config.transformer.activation),
            nn.Linear(512, self.embed_dim),
        )
        self.prompt_embed = nn.Embedding(self.n_states, self.embed_dim)

    def forward(
        self,
        patches: Tensor,
        patch_locs: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            patches: A batch of M patches (B, M, 3, patch_size, patch_size).
            state: Input state for the transformer (B, N, D).
            patch_locs: The locations of the patches (B, M, 3).

        Returns:
            Batch of logits (B, num_classes).

        """
        xm.master_print("model: inside forward")
        batch_size = patches.shape[0]
        xm.master_print(f"{batch_size =}")
        state = torch.zeros(batch_size, self.n_states, self.embed_dim).to(patches.device)
        xm.master_print(f"model: {state.shape=}, {state.dtype=}, {state.device=}")
        patches = self.tokenizer(patches)  # (B, M, D)
        xm.master_print(f"model: {patches.shape=}, {patches.dtype=}, {patches.device=}")
        patch_pos_embeds = self.pos_embed(patch_locs)  # (B, M, D)
        xm.master_print(
            f"model: {patch_pos_embeds.shape=}, {patch_pos_embeds.dtype=}, {patch_pos_embeds.device=}"
        )
        state_query = self.prompt_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        xm.master_print(f"model: {state_query.shape=}, {state_query.dtype=}, {state_query.device=}")
        state = self.transformer(state, patches, patch_pos_embeds, state_query)
        xm.master_print(f"model: new state {state.shape=}, {state.dtype=}, {state.device=}")
        logits = self.task_head(state)
        xm.master_print(f"model: new state {logits.shape=}, {logits.dtype=}, {logits.device=}")
        return logits

    def get_param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Get param groupd with different learning rates."""
        param_groups: dict[str, dict[str, Any]] = {
            "wd": {
                "weight_decay": weight_decay,
                "name": "wd",
                "params": [],
            },
            "no_wd": {
                "weight_decay": 0.0,
                "name": "no_wd",
                "params": [],
            },
        }

        no_weight_decay = ["prompt_embed.weight"]

        for name, param in self.named_parameters():
            if param.ndim == 1 or name in no_weight_decay:
                param_groups["no_wd"]["params"].append(param)
            else:
                param_groups["wd"]["params"].append(param)

        return list(param_groups.values())
