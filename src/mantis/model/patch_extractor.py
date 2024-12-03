"""Extracting patches from images given locations."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@torch.no_grad()
def extract_equal_zoom_patches(
    images: Tensor,
    locs: Tensor,
    z: float,
    patch_size: int,
) -> Tensor:
    """Extract equal zoom patches from a batch of images.

    Extract single patch per image at the same zoom level across the batch.

    Args:
        images: Batch of images of shape (B, 3, H, W).
        locs: Batch of locations of shape (B, 2) containing (x, y).
        z: The zoom level for all patches.
        patch_size: The size of each output patch is (3, patch_size, patch_size).

    Returns:
        Patches of shape (B, 3, patch_size, patch_size).

    """
    B, _, H, W = images.shape  # noqa: N806
    aspect_ratio = W / H
    y_coords = torch.linspace(-1, 1, patch_size)
    x_coords = torch.linspace(-1, 1, patch_size)
    if aspect_ratio > 1:
        # Image in landscape orientation.
        x_coords = x_coords * aspect_ratio
    else:
        # Image in portrait orientation.
        y_coords = y_coords / aspect_ratio
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)  # (P, P, 2)
    # Apply zoom level
    sampling_grid /= 2**z
    sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Change the centers according to the locs
    sampling_grid = sampling_grid + locs[:, None, None, :]

    return F.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        align_corners=False,
    )
