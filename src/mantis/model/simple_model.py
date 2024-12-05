"""Simple model without RL and an agent."""

import torch
from torch import Tensor, nn

from mantis.configs.config import Config
from mantis.configs.transformer import get_activation_module
from mantis.model.layers.simple_head import SimpleHead
from mantis.model.layers.tokenizer import PatchEmbed
from mantis.model.transformer import Transformer


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
        batch_size = patches.shape[0]
        state = torch.zeros(batch_size, self.n_states, self.embed_dim).to(patches.device)
        patches = self.tokenizer(patches)  # (B, M, D)
        patch_pos_embeds = self.pos_embed(patch_locs)  # (B, M, D)
        state_query = self.prompt_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        state = self.transformer(state, patches, patch_pos_embeds, state_query)
        return self.task_head(state)
