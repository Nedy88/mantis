"""Test fluid patch extraction."""

import torch

from mantis.model.patch_extractor import extract_equal_zoom_patches


@torch.no_grad()
def test_equal_zoom_patches_output_shape() -> None:
    """Test the output shape of the extract_equal_zoom_patches function."""
    B, H, W = 4, 300, 400  # noqa: N806
    P = 16  # noqa: N806
    images = torch.randn(B, 3, H, W)
    locs = torch.rand(B, 2) * 2 - 1
    assert (locs >= -1).all()
    assert (locs <= 1).all()
    patches = extract_equal_zoom_patches(images, locs, 4.0, P)
    assert patches.shape == (B, 3, P, P)
