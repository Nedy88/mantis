"""Test the simple model."""

import torch

from mantis.configs.config import Config
from mantis.model.simple_model import SimpleModel


def test_simple_model_output_shape(config: Config) -> None:
    """Test the output shape of the simple model."""
    model = SimpleModel(config)
    batch_size = 4
    M = 8  # noqa: N806
    input_patches = torch.randn(batch_size, M, 3, config.patch_size, config.patch_size)
    patch_locs = torch.rand(batch_size, M, 3)
    # Convert from 0 to 1 to -1 to 1 for the x and y coordinates
    patch_locs[:, :, :2] = patch_locs[:, :, :2] * 2 - 1
    # Make max zoom level 4
    patch_locs[:, :, 2] = patch_locs[:, :, 2] * 4
    output = model(input_patches, patch_locs)
    assert output.shape == (batch_size, config.dataset.num_classes)
