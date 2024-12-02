"""Test fluid patch extraction."""

import torch

from mantis.model.patch_extractor import extract_patches


@torch.no_grad()
def test_extract_patches_output_shape() -> None:
    """Test the output shape of the extract_patches function."""
    B, H, W = 4, 300, 400  # noqa: N806
    M, P = 10, 16  # noqa: N806
    images = torch.randn(B, 3, H, W)
    locs = torch.zeros(B, M, 3)
    locs[:, :, :2] = torch.rand(B, M, 2) # Between 0 and 1
    locs[:, :, 2] = torch.rand(B, M) * 4 # Between 0 and 3
    patches = extract_patches(images, locs, P)
    assert patches.shape == (B, M, 3, P, P)
