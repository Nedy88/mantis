"""Simple model without RL and an agent."""

import torch
from torch import Tensor, nn

from mantis.configs.config import Config
from mantis.model.layers.simple_head import SimpleHead
from mantis.model.transformer import Transformer


class SimpleModel(nn.Module):
    """Simple model without RL and an agent."""

    def __init__(self, config: Config) -> None:
        """Init."""
        super().__init__()
        self.transformer = Transformer(config.transformer)
        assert config.dataset.num_classes is not None
        self.task_head = SimpleHead(
            n_states=config.transformer.n_states,
            embed_dim=config.transformer.attention.embed_dim,
            activation=config.transformer.activation,
            output_dim=config.dataset.num_classes,
        )

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        patch_locs: Tensor,
        *,
        include_task: bool,
    ) -> Tensor:
        """Forward pass."""
        # images is (B, 3, H, W)
        # state is (B, N, D)
        # patch_locs is (B, M, 3)
        # TODO: Finish the forward pass
        return torch.zeros_like(state)
