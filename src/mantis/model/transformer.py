"""The modules necessary for the transformer model."""

from torch import Tensor, nn

EPS = 1e-6

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
