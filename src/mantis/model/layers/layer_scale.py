"""Layer scale module."""

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Layer scale inspired by DINOv2.

    See: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/layer_scale.py
    """

    def __init__(self, dim: int, init_val: float, *, inplace: bool) -> None:
        """Init."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_val * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the LayerScale module."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
