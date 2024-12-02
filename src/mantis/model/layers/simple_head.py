"""Simple classification head."""

from torch import Tensor, nn

from mantis.configs.transformer import Activation, get_activation_module


class SimpleHead(nn.Module):
    """Simple classification head."""

    def __init__(
        self,
        n_states: int,
        embed_dim: int,
        activation: Activation,
        output_dim: int,
    ) -> None:
        """Init."""
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_states * embed_dim, embed_dim),
            get_activation_module(activation),
            nn.Linear(embed_dim, output_dim),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the module."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass."""
        return self.head(state.reshape(state.shape[0], -1))  # (B, output_dim)
